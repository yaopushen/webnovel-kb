# WebNovel Knowledge Base - 项目维护指南

> 版本：v1.8 | 更新日期：2026-05-17

## 目录
1. [架构概览](#架构概览)
2. [服务部署](#服务部署)
3. [OAuth 2.0 PKCE 认证](#oauth-20-pkce-认证)
4. [日常运维](#日常运维)
5. [数据备份与恢复](#数据备份与恢复)
6. [故障排查](#故障排查)
7. [MCP 工具参考](#mcp-工具参考)

---

## 架构概览

### 模块化架构（v1.7）
```
webnovel_kb/
├── server.py          # MCP 服务入口（_OAuthFastMCP 子类化）
├── oauth_auth.py      # OAuth 2.0 PKCE 授权服务器（v1.7 新增）
├── __main__.py        # 入口点
├── config.py          # 环境变量配置
├── data_models.py     # 数据模型
├── prompts.py         # LLM 提示词
├── core/
│   ├── knowledge_base.py  # 核心协调层
│   ├── chunker.py         # 文本分块
│   ├── indexer.py         # 索引管理（ChromaDB）
│   └── state.py           # 状态持久化
├── search/
│   ├── semantic.py    # 语义搜索（query_embeddings）
│   ├── bm25_search.py # BM25 关键词搜索
│   ├── hybrid.py      # 混合搜索（RRF 融合）
│   ├── rerank.py      # LLM Rerank 精排
│   └── unified.py     # 统一搜索入口
├── extraction/
│   ├── entities.py         # 实体提取
│   ├── plot_patterns.py    # 情节模式提取
│   ├── scene_patterns.py   # 场景模式提取
│   └── writing_templates.py # 写作模板提取
├── analysis/
│   ├── humor.py       # 幽默场景分析
│   └── style.py       # 写作风格分析
├── api/
│   ├── clients.py     # API 客户端（Embedding/Rerank/Chat）
│   └── mcp_tools.py   # MCP 工具注册（11个工具）
└── utils/
    ├── dedupe.py      # 去重逻辑
    └── format.py      # 输出格式化
```

### 数据流向
```
MCP Client (Trae IDE)
       │ streamable-http (port 8765)
       ▼
  _OAuthFastMCP (server.py)
       │
       ├── /mcp           → MCP 工具（无需认证）
       ├── /authorize     → OAuth 授权端点
       ├── /token         → OAuth 令牌端点
       └── /.well-known/  → OAuth 发现端点
       │
       ▼
  KnowledgeBase (core/knowledge_base.py)
       │
  ┌────┼────────────────┐
  ▼    ▼                ▼
Search  Extraction    Analysis
  │    │                │
  ▼    ▼                ▼
ChromaDB + JSON 文件存储
       │
       ▼
  LLM API（Embedding / Rerank / Chat）
```

### 网络拓扑
```
内网（局域网）                     公网（CDN）
Trae IDE ──→ localhost:8765        外部客户端 ──→ your-domain.com
             无需认证                              OAuth 2.0 PKCE 认证
```

---

## 服务部署

### 环境变量

所有配置通过环境变量管理：

| 变量名 | 说明 |
|--------|------|
| `MCP_TRANSPORT` | 传输方式：`streamable-http`（推荐）或 `stdio` |
| `MCP_HOST` | 监听地址，公网部署建议 `0.0.0.0` |
| `MCP_PORT` | 监听端口，默认 `8765` |
| `MCP_OAUTH_ISSUER_URL` | OAuth 签发者 URL（如 `https://your-domain.com`），留空则不启用 OAuth |
| `OAUTH_JWT_SECRET` | JWT 签名密钥，必须与客户端配置一致 |
| `OAUTH_TOKEN_EXPIRY` | Token 有效期（秒），默认 `86400`（24小时） |
| `MCP_API_KEY` | 静态 API Key 认证（可选，设置后内网也需认证） |
| `WEBNOVEL_KB_DATA` | 数据目录绝对路径 |
| `LLM_API_KEY` | LLM 服务 API 密钥 |
| `LLM_BASE_URL` | Embedding API 地址 |
| `LLM_CHAT_BASE_URL` | Chat API 地址 |
| `LLM_EMBEDDING_MODEL` | Embedding 模型名称 |
| `LLM_RERANK_MODEL` | Rerank 模型名称 |
| `LLM_CHAT_MODEL` | Chat 模型名称 |
| `LLM_EMBEDDING_DIMENSIONS` | Embedding 向量维度 |

> **重要**：`MCP_TRANSPORT` 必须设为 `streamable-http`，不再支持 `sse`（SSE 存在多 session 冲突问题）。

### systemd 服务管理

```bash
# 查看状态
sudo systemctl status webnovel-mcp

# 启动
sudo systemctl start webnovel-mcp

# 停止
sudo systemctl stop webnovel-mcp

# 重启
sudo systemctl restart webnovel-mcp

# 查看日志
journalctl -u webnovel-mcp -f           # 实时
journalctl -u webnovel-mcp --since '1h'  # 最近1小时
```

### 手动启动（调试用）

```bash
cd /path/to/webnovel-kb
source venv/bin/activate
MCP_TRANSPORT=streamable-http \
MCP_OAUTH_ISSUER_URL=https://your-domain.com \
OAUTH_JWT_SECRET=your-oauth-secret-key \
MCP_HOST=0.0.0.0 MCP_PORT=8765 \
python3 -m webnovel_kb
```

### 更新部署

```bash
# 1. 拉取最新代码
cd /path/to/webnovel-kb
git pull origin main

# 2. 清理缓存 + 重启服务
find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null
sudo systemctl restart webnovel-mcp

# 3. 检查状态
sudo systemctl status webnovel-mcp --no-pager

# 4. 验证 OAuth 端点
curl -s http://127.0.0.1:8765/.well-known/oauth-authorization-server | python3 -m json.tool
```

---

## OAuth 2.0 PKCE 认证

### 概述

v1.7 新增内建 OAuth 2.0 PKCE 授权服务器，保护公网端点。

- **内网访问**（`localhost:8765`）：无需认证，直接使用 MCP 工具
- **公网访问**（`your-domain.com`）：需通过 OAuth PKCE 流程获取 access token

### OAuth 端点

| 端点 | URL | 说明 |
|------|-----|------|
| Discovery | `https://your-domain.com/.well-known/oauth-authorization-server` | OAuth 元数据 |
| Authorize | `https://your-domain.com/authorize` | 授权码获取 |
| Token | `https://your-domain.com/token` | 令牌交换 |

### Trae 自定义连接器配置

在 Trae IDE 的 MCP 自定义连接器 OAuth 表单中填入：

| 字段 | 值 |
|------|-----|
| Client ID | `mcp-client` |
| Client Secret | 留空（PKCE 模式不需要） |
| Authorization Endpoint | `https://your-domain.com/authorize` |
| Token Endpoint | `https://your-domain.com/token` |
| Scopes | `mcp:read mcp:write` |
| Token Authentication Method | `none` |

### 技术实现

- **授权服务器**：`oauth_auth.py`，内建 PKCE S256 challenge 验证
- **Token 签发**：JWT (HS256)，密钥为 `OAUTH_JWT_SECRET` 环境变量
- **路由注入**：通过子类化 `_OAuthFastMCP(FastMCP)`，重写 `run_streamable_http_async()` 方法，在 SDK 生成的 Starlette app 上注入 OAuth 路由
- **Token 验证**：`_SimpleTokenVerifier` 实现 MCP SDK 的 `TokenVerifier` 协议
- **授权码存储**：内存 dict，有效期 60 秒
- **Token 有效期**：默认 86400 秒（24 小时），可通过 `OAUTH_TOKEN_EXPIRY` 配置

### 认证流程

```
1. 客户端生成 code_verifier + code_challenge(S256)
2. GET /authorize?client_id=mcp-client&redirect_uri=...&code_challenge=...&state=...
3. 服务器返回 307 重定向到 redirect_uri?code=xxx&state=xxx
4. 客户端 POST /token {code, code_verifier, redirect_uri}
5. 服务器验证 PKCE，签发 JWT access_token
6. 后续请求携带 Authorization: Bearer <access_token>
```

---

## 日常运维

### 查看知识库统计

通过 MCP 工具调用：
```
mcp_webnovel-kb_stats()
```

### 手动执行维护操作（通过命令行）

**导入新小说：**
```bash
cd /path/to/webnovel-kb
source venv/bin/activate
python3 -c "
from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
kb = WebNovelKnowledgeBase()
result = kb.ingest_novel(
    file_path='/path/to/novel.txt',
    title='小说名',
    author='作者名',
    genre='类型'  # 可选: 修仙/科幻/悬疑/奇幻/赛博朋克/克苏鲁/高武
)
print(result)
"
```

**提取结构化知识（实体、情节模式）：**
```bash
cd /path/to/webnovel-kb
source venv/bin/activate
python3 -c "
from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
kb = WebNovelKnowledgeBase()
result = kb.extract('小说名', extract_type='all', max_chunks=200, cross_chunk=True)
print(result)
"
```

**异步提取（大书推荐）：**
```bash
cd /path/to/webnovel-kb
source venv/bin/activate
python3 -c "
import time
from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
kb = WebNovelKnowledgeBase()
task = kb.start_async_extraction('小说名', extract_type='all', max_chunks=500)
print(f'Task ID: {task[\"task_id\"]}')

# 轮询进度
while True:
    status = kb.get_task_status(task['task_id'])
    print(f'Progress: {status.get(\"progress\", 0)}%')
    if status['status'] in ('completed', 'error'):
        print(status)
        break
    time.sleep(30)
"
```

### 日志位置
- systemd 日志：`journalctl -u webnovel-mcp`
- 服务输出：`/tmp/mcp_server.log`

### 资源监控
```bash
# 进程状态
ps aux | grep 'webnovel_kb' | grep -v grep

# 内存占用
top -p $(pgrep -f 'webnovel_kb')

# 磁盘占用
du -sh /path/to/webnovel-data/
```

---

## 数据备份与恢复

### 备份
```bash
# 完整备份（JSON 数据 + 知识图谱）
DATA_DIR="/path/to/data"
BACKUP_DIR="$DATA_DIR/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR
cp $DATA_DIR/*.json $BACKUP_DIR/

# 备份 embedding 缓存（可选，较大）
cp $DATA_DIR/embeddings_cache.pkl $BACKUP_DIR/
```

### 恢复
```bash
sudo systemctl stop webnovel-mcp
cp /path/to/data/backups/YYYYMMDD_HHMMSS/*.json /path/to/data/
sudo systemctl start webnovel-mcp
```

### 数据文件清单
| 文件 | 说明 |
|------|------|
| `chroma_db/` | ChromaDB 向量数据库 |
| `embeddings_cache.pkl` | Embedding 缓存（numpy.float32） |
| `entities.json` | 实体数据 |
| `plot_patterns.json` | 情节模式 |
| `writing_templates.json` | 写作模板 |
| `style_profiles.json` | 风格档案 |
| `knowledge_graph.json` | 知识图谱 |
| `relationships.json` | 关系数据 |
| `state.json` | 服务状态 |
| `internet_memes.json` | 网络热梗 |

---

## 故障排查

### 1. MCP 连接失败
```
症状：客户端显示 "list tools failed"
原因：服务未运行或端口被占用
排查：
  ps aux | grep webnovel_kb
  ss -tlnp 'sport = :8765'
解决：
  sudo systemctl restart webnovel-mcp
```

### 2. 端口被占用（服务无法启动）
```
症状：systemd 日志显示 "address already in use"
原因：僵尸进程或旧服务自动重启
排查：
  ss -tlnp 'sport = :8765'
  systemctl list-units --all | grep -i 'webnovel'
解决：
  sudo fuser -k 8765/tcp
  sleep 3
  sudo systemctl restart webnovel-mcp
```

### 3. 搜索返回空结果
```
症状：search/smart_search 返回空
原因：ChromaDB 数据损坏或维度不匹配
排查：
  du -sh /path/to/data/chroma_db/
  mcp_webnovel-kb_stats()
解决：
  # 如果 total_chunks 为 0，需要重新导入小说
  # 如果 total_chunks 正常但搜索为空，检查 embedding 维度配置
```

### 4. Embedding 维度不匹配
```
症状：语义搜索报错或返回空结果
原因：LLM_EMBEDDING_DIMENSIONS 配置与存储数据维度不一致
排查：
  python3 -c "
  import chromadb
  client = chromadb.PersistentClient(path='/path/to/data/chroma_db')
  col = client.get_collection('webnovel_chunks')
  sample = col.get(limit=1, include=['embeddings'])
  print(f'Stored dims: {len(sample[\"embeddings\"][0])}')
  "
解决：
  确保 systemd 服务中 LLM_EMBEDDING_DIMENSIONS 与存储维度一致
```

### 5. OAuth 端点返回 401
```
症状：公网访问 /authorize 或 /token 返回 401
原因：BearerAuthMiddleware 拦截了 OAuth 路由
排查：
  curl -s http://127.0.0.1:8765/.well-known/oauth-authorization-server
解决：
  # 确认 server.py 中 BearerAuthMiddleware 跳过了 OAUTH_PATHS
  # 确认 MCP_API_KEY 未设置时不会强制认证
```

### 6. 内存不足
```
症状：服务崩溃或 OOM
原因：ChromaDB + embedding 缓存占用较大
排查：
  free -h
  ps aux --sort=-%mem | head -5
解决：
  sudo systemctl restart webnovel-mcp
```

### 7. API 连接问题
```
症状：embedding 或 rerank 调用失败
原因：API 服务不可用或密钥过期
排查：
  curl -s -H "Authorization: Bearer $LLM_API_KEY" $LLM_BASE_URL/models
解决：
  检查 systemd 服务中的 LLM_API_KEY 是否有效
```

---

## MCP 工具参考（v1.8，共 11 个）

### 浏览类
| 工具 | 说明 | 关键参数 |
|------|------|----------|
| `stats` | 全局/单本统计（无参数=全局，传书名=单本） | novel_title（可选） |
| `list_novels` | 列出所有已导入小说 | — |
| `list_chapters` | 列出章节列表 | novel_title |
| `read_chapter` | 读取章节完整正文 | novel_title, chapter |
| `get_chapter_edges` | 提取章前章末段落（学习开头/结尾写法） | novel_title, chapter, paragraphs |

### 搜索类
| 工具 | 说明 | 关键参数 |
|------|------|----------|
| `search` | 全文统一检索（语义+BM25混合/纯语义/纯关键词/精排） | query, mode, n_results, novel_filter, genre_filter, alpha, use_rerank |
| `smart_search` | 智能搜索（LLM 函数调用模式，多轮迭代深度分析） | query, n_results, novel_filter, genre_filter |
| `search_knowledge` | 搜索情节模式/写法模板 | query, knowledge_type, type_filter, source_novel |
| `search_entities` | 语义搜索角色/功法/地点/组织等实体 | query, entity_type, source_novel |
| `get_entity_relations` | 查询实体关系网 | entity_name, source_novel |

### 分析类
| 工具 | 说明 | 关键参数 |
|------|------|----------|
| `style_analysis` | 风格分析/对比（单书名=分析，逗号分隔=对比） | novel_titles |

### 工具详细参数

#### `stats` — 知识库统计
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| novel_title | str | "" | 书名（模糊匹配），留空=全局统计 |

#### `list_novels` — 列出小说
无参数。返回所有小说的 title, author, genre, word_count, chunk_count。

#### `list_chapters` — 列出章节
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| novel_title | str | 必填 | 书名（模糊匹配） |

#### `read_chapter` — 读取章节
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| novel_title | str | 必填 | 书名（模糊匹配） |
| chapter | int | 1 | 章节号（1-based） |

#### `get_chapter_edges` — 章前章末段落
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| novel_title | str | 必填 | 书名（模糊匹配） |
| chapter | int | 1 | 章节号（1-based） |
| paragraphs | int | 2 | 章前/章末各提取几段 |

#### `search` — 全文检索
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| query | str | 必填 | 搜索关键词或自然语言描述 |
| mode | str | "hybrid" | hybrid/semantic/bm25/rerank |
| n_results | int | 10 | 返回结果数量 |
| novel_filter | str | "" | 限定书名（模糊匹配） |
| genre_filter | str | "" | 限定类型（修仙/科幻/悬疑/奇幻/赛博朋克/克苏鲁/高武） |
| chapter_filter | str | "" | 限定章节（按标题匹配） |
| alpha | float | 0.6 | hybrid 语义权重（0~1，越大越偏语义） |
| use_rerank | bool | False | 是否启用 Cross-encoder 精排 |
| output_format | str | "compact" | compact/clean/raw |
| max_content_length | int | 0 | 每条结果字数上限（0=不限） |
| dedupe | bool | True | 是否去重 |

#### `smart_search` — 智能搜索
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| query | str | 必填 | 自然语言描述（可模糊、口语化） |
| n_results | int | 5 | 每次工具搜索返回数 |
| novel_filter | str | "" | 限定书名（模糊匹配） |
| genre_filter | str | "" | 限定类型 |
| output_format | str | "compact" | compact/clean/raw |

> **注意**：smart_search 调用 Chat 模型进行函数调用模式搜索，响应时间取决于模型和搜索复杂度。模型自主调用 3 个内部工具（search_text/search_patterns/search_entities），支持多轮迭代和并行工具调用。LLM 调用失败时自动降级为 hybrid 搜索。

#### `search_knowledge` — 搜索结构化知识
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| query | str | "" | 搜索描述，留空不过滤 |
| knowledge_type | str | "plot_patterns" | plot_patterns/writing_templates |
| n_results | int | 10 | 返回条数 |
| use_semantic | bool | True | True=语义搜索，False=关键词过滤 |
| type_filter | str | "" | 情节模式类型（悬念链/跨距伏笔/反转铺垫/情感爆发点/世界观展开/力量体系引入/角色弧光/高潮设计/节奏控制/对比映衬/身份揭示） |
| source_novel | str | "" | 限定书名（模糊匹配） |
| output_format | str | "compact" | compact/clean/raw |
| max_content_length | int | 0 | 每条字数上限（0=不限） |
| dedupe | bool | True | 是否去重 |

> **前提**：需先通过命令行执行 `kb.extract(novel_title, extract_type="all")` 提取过知识，否则结果可能为空。

#### `search_entities` — 搜索实体
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| query | str | 必填 | 实体描述（如"反派角色"、"修炼功法"） |
| n_results | int | 10 | 返回条数 |
| entity_type | str | "" | 角色/功法/组织/地点/物品/概念/事件/种族 |
| source_novel | str | "" | 限定书名（模糊匹配） |
| output_format | str | "compact" | compact/clean/raw |
| max_content_length | int | 0 | 每条字数上限（0=不限） |
| dedupe | bool | True | 是否去重 |

> **前提**：需先通过命令行执行 `kb.extract(novel_title, extract_type="entities")` 提取过实体。

#### `get_entity_relations` — 实体关系
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| entity_name | str | 必填 | 实体名称（需精确，可通过 search_entities 获取） |
| source_novel | str | "" | 限定书名（模糊匹配） |

#### `style_analysis` — 风格分析
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| novel_titles | str | 必填 | 书名或逗号分隔的书名列表（模糊匹配） |

> **注意**：首次分析较慢（约 30-120 秒），结果会缓存。

---

## 项目历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2025-04-30 | 初始版本，单体架构（server.py 2200+ 行） |
| v1.5 | 2026-05-01 | 模块化重构，修复语义搜索维度问题，清理旧文件 |
| v1.6 | 2026-05-11 | 工具精简（15→10），新增 smart_search，移除管理工具，合并 stats/style_analysis |
| v1.7 | 2026-05-15 | SSE→Streamable-HTTP，新增 OAuth 2.0 PKCE 认证，公网端点保护 |
| v1.8 | 2026-05-17 | 工具参考全面更新（10→11，新增 get_chapter_edges，详细参数表），smart_search 描述修正 |

### v1.7 变更详记

**传输层变更：**
- SSE → Streamable-HTTP（解决多 session 冲突导致返回 null 的问题）
- 每次请求独立 HTTP POST，不再维护持久连接

**新增 OAuth 2.0 PKCE 认证：**
- `oauth_auth.py` — 内建授权服务器（authorize + token + well-known 端点）
- `_OAuthFastMCP(FastMCP)` — 子类化 SDK，注入 OAuth 路由到 Starlette app
- 内网无需认证，公网通过 PKCE 流程保护
- JWT (HS256) 签发 access token

**修复章节解析：**
- `chunker.py` — 正则添加 `|\d{3,5}` 匹配裸三位数字章节号
- `knowledge_base.py` — list_chapters/read_chapter 多模式修复

**systemd 服务变更：**
- 新服务名：`webnovel-mcp`
- 入口脚本：`-m webnovel_kb`
- 新增环境变量：`MCP_OAUTH_ISSUER_URL`、`OAUTH_JWT_SECRET`

**部署清单（每次更新代码后执行）：**
```bash
# 1. 拉取最新代码
git pull origin main

# 2. 清理缓存 + 重启
find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null
sudo systemctl restart webnovel-mcp

# 3. 检查状态
sudo systemctl status webnovel-mcp --no-pager

# 4. 验证 OAuth 端点
curl -s http://127.0.0.1:8765/.well-known/oauth-authorization-server | python3 -m json.tool
```

**systemd 服务配置：**
- 服务名: `webnovel-mcp`
- 服务文件: `/etc/systemd/system/webnovel-mcp.service`
- 数据目录: `/path/to/webnovel-data/`
- 项目目录: `/path/to/webnovel-kb/`
- Chat 模型: 示例模型 (1M 上下文)
- Embedding 模型: 示例模型 (4096 维)
- Rerank 模型: 示例模型
- OAuth Client ID: `mcp-client`
- 公网端点: `https://your-domain.com`（CDN 反代）