# WebNovel Knowledge Base - 项目维护指南

> 版本：v1.12.1 | 更新日期：2026-06-10
>
> 完整版本变更记录详见 [CHANGELOG.md](file:///f:/MCP/CHANGELOG.md)
>
> 服务侧连接示例：ssh your_username\@YOUR_SERVER_IP "sudo systemctl stop webnovel-mcp"
>
> 测试测示例：直接调用get status工具

## 目录

1. [架构概览](#架构概览)
2. [服务部署](#服务部署)
3. [OAuth 2.0 PKCE 认证](#oauth-20-pkce-认证)
4. [日常运维](#日常运维)
5. [数据备份与恢复](#数据备份与恢复)
6. [故障排查](#故障排查)
7. [MCP 工具参考](#mcp-工具参考)
8. [已知问题与限制](#已知问题与限制)

***

## 架构概览

### 模块化架构（v1.12 — 职责拆分）

```
webnovel_kb/
├── server.py              # MCP 服务入口（_OAuthFastMCP + 双协议支持）
├── oauth_auth.py          # OAuth 2.0 PKCE 授权服务器
├── __main__.py            # 入口点
├── config.py              # 环境变量配置（LLM_* 标准化 + 知乎 API）
├── data_models.py         # 数据模型
├── prompts.py             # LLM 提示词
├── search_engines.py      # TantivyBM25 + HybridSearchEngine
├── core/
│   ├── knowledge_base.py  # 门面类（~711 行，从 1866 行精简）
│   ├── novel_reader.py    # 章节读取（从 knowledge_base 提取）
│   ├── outline_manager.py # 章纲管理（从 knowledge_base 提取）
│   ├── task_manager.py    # 异步任务管理（从 knowledge_base 提取）
│   ├── chunker.py         # 文本分块
│   ├── indexer.py         # 索引管理（仅 Tantivy+ChromaDB）
│   └── state.py           # 状态持久化（pickle）
├── search/
│   ├── smart.py           # SmartSearchEngine（从 knowledge_base 提取）
│   ├── external.py        # ExternalSearch 知乎/Tavily（从 knowledge_base 提取）
│   ├── semantic.py        # 语义搜索（ChromaDB HNSW）
│   ├── bm25_search.py     # BM25 关键词搜索（Tantivy）
│   ├── hybrid.py          # 混合搜索（RRF 融合）
│   ├── rerank.py          # LLM Rerank 精排
│   └── unified.py         # 统一搜索入口
├── extraction/
│   ├── entities.py         # 实体提取
│   ├── plot_patterns.py    # 情节模式提取
│   ├── scene_patterns.py   # 场景模式提取
│   ├── outlines.py         # 章纲自动提取（v1.11）
│   └── writing_templates.py # 写作模板提取
├── analysis/
│   ├── humor.py       # 幽默场景分析
│   └── style.py       # 写作风格分析
├── api/
│   ├── clients.py     # API 客户端（RemoteEmbedding/RemoteRerank/RemoteChat）
│   ├── mcp_tools.py   # MCP 工具门面（~72 行，从 518 行精简）
│   └── tools/         # 工具分组目录（从 mcp_tools 提取）
│       ├── __init__.py    # register_all_tools() 入口
│       ├── browse.py      # stats / read_chapter / get_chapter_edges
│       ├── search.py      # search / smart_search
│       ├── outline.py     # save_outline / get_outline / extract_outline
│       ├── analysis.py    # style_analysis
│       ├── knowledge.py   # add_knowledge
│       ├── creation.py    # generate_sample
│       └── task.py        # manage_task
├── utils/
│   ├── novel_resolver.py  # 统一小说模糊匹配（从 knowledge_base 提取）
│   ├── chinese_numbers.py # 中文数字转换（从 knowledge_base 提取）
│   ├── dedupe.py          # 去重逻辑
│   ├── format.py          # 输出格式化
│   ├── exceptions.py      # 自定义异常基类
│   ├── logging_config.py  # 日志配置（轮转+分级）
│   └── query_cache.py     # 查询缓存
└── scripts/
    └── build_optimized_indexes.py  # Tantivy 索引构建脚本
```

### 数据流向

```
MCP Client (Trae IDE / Claude / Grok)
       │
       ├── SSE:  GET /sse + POST /messages    (老客户端)
       └── HTTP: POST /mcp                    (新客户端)
       │
       ▼
  _OAuthFastMCP + _build_app (server.py)
       │  Dual-protocol mode: 共享同一个 FastMCP 实例
       ├── /mcp           → Streamable HTTP MCP 工具
       ├── /sse           → SSE 事件流
       ├── /messages      → SSE 消息通道
       ├── /authorize     → OAuth 授权端点
       ├── /token         → OAuth 令牌端点
       └── /.well-known/  → OAuth 发现端点
       │
       ▼
  KnowledgeBase (core/knowledge_base.py — 门面类)
       │
  ┌────┼────────────────┐
  ▼    ▼                ▼
Search  Extraction    Analysis
  │    │                │
  ▼    ▼                ▼
HybridSearchEngine  →  ChromaDB (HNSW向量) + TantivyBM25 (关键词)
       │
       ▼
  RemoteEmbedding (4096维 API 嵌入，965MB pkl 缓存)
```

### smart\_search 内部工具架构（v1.9.2）

```
                        ┌─────────────────────────────────┐
                        │       smart_search (MiMo)        │
                        │   函数调用模式，最多 200 轮迭代     │
                        └────────────┬────────────────────┘
                                     │
          ┌──────────────┬───────────┼───────────┬──────────────┐
          ▼              ▼           ▼           ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
   │search_text │ │search_     │ │search_     │ │web_search  │ │zhihu_search│
   │知识库全文   │ │patterns    │ │entities    │ │知乎全网搜索 │ │知乎站内搜索│
   │hybrid/     │ │情节模式    │ │角色/功法等  │ │行业趋势    │ │写作技巧    │
   │semantic/   │ │悬念/伏笔等 │ │实体信息    │ │读者偏好    │ │作者经验    │
   │bm25        │ │            │ │            │ │平台分析    │ │套路拆解    │
   └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘
        内部知识库（3个工具）                            外部知识（2个工具）
```

### 搜索架构

```
                        ┌─────────────────────┐
                        │   HybridSearchEngine │
                        │   (RRF 融合 Rank)    │
                        └─────────┬───────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 ▼                                 ▼
     ┌───────────────────┐            ┌───────────────────┐
     │  ChromaDB (HNSW)  │            │  TantivyBM25      │
     │  向量语义搜索      │            │  关键词 BM25 搜索   │
     │  磁盘: 1.7GB      │            │  磁盘: 639MB       │
     └───────────────────┘            └───────────────────┘
```

### 网络拓扑

```
内网（局域网）                     公网（Cloudflare CDN）
Trae IDE ──→ YOUR_SERVER_IP:8765   外部客户端 ──→ mcp.your_username-1.tech
  SSE: /sse, /messages               双协议自动协商
  HTTP: /mcp                         OAuth 2.0 PKCE 认证
             无需认证
```

***

## 服务部署

### 环境变量

所有配置通过环境变量管理：

| 变量名                        | 说明                                                         |
| -------------------------- | ---------------------------------------------------------- |
| `MCP_TRANSPORT`            | 传输方式：`http`（双协议，推荐）/ `streamable-http` / `sse` / `stdio` |
| `MCP_HOST`                 | 监听地址，公网部署建议 `0.0.0.0`                                      |
| `MCP_PORT`                 | 监听端口，默认 `8765`                                             |
| `MCP_OAUTH_ISSUER_URL`     | OAuth 签发者 URL（如 `https://mcp.your_username-1.tech`），留空则不启用 OAuth |
| `OAUTH_JWT_SECRET`         | JWT 签名密钥，必须与客户端配置一致                                        |
| `OAUTH_TOKEN_EXPIRY`       | Token 有效期（秒），默认 `86400`（24小时）                              |
| `MCP_API_KEY`              | 静态 API Key 认证（可选，设置后内网也需认证）                                |
| `WEBNOVEL_KB_DATA`         | 数据目录绝对路径                                                   |
| `LLM_API_KEY`              | API 密钥                                                     |
| `LLM_BASE_URL`             | Embedding/Rerank API 地址                                    |
| `LLM_CHAT_BASE_URL`        | Chat API 地址                                                |
| `LLM_CHAT_API_KEY`         | Chat API 密钥（可与 LLM\_API\_KEY 不同）                           |
| `LLM_EMBEDDING_MODEL`      | Embedding 模型名称                                             |
| `LLM_RERANK_MODEL`         | Rerank 模型名称                                                |
| `LLM_CHAT_MODEL`           | Chat 模型名称                                                  |
| `LLM_EMBEDDING_DIMENSIONS` | Embedding 向量维度，默认 `4096`                                   |
| `ZHIHU_ACCESS_SECRET`      | 知乎开发者 API 密钥（smart\_search 外部搜索用）                          |

> **重要**：
>
> - `MCP_TRANSPORT=http` 启动双协议模式，同时支持 SSE (`/sse`, `/messages`) 和 Streamable HTTP (`/mcp`)。
> - `MCP_TRANSPORT=streamable-http` 或 `sse` 可单独启用对应协议。
> - `LLM_EMBEDDING_DIMENSIONS` 必须与已存储数据维度一致（4096），否则语义搜索失败。
> - `ZHIHU_ACCESS_SECRET` 未配置时，smart\_search 的 web\_search 和 zhihu\_search 工具将返回错误提示，不影响内部搜索。

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
cd /home/your_username/webnovel-kb
source venv/bin/activate
WEBNOVEL_KB_DATA=/home/your_username/webnovel-data \
ZHIHU_ACCESS_SECRET=your-secret \
MCP_HOST=0.0.0.0 MCP_PORT=8000 \
python -m webnovel_kb
```

### 更新部署

```bash
# 1. 上传代码（从 Windows 开发机）
scp f:\MCP\webnovel_kb\config.py your_username@YOUR_SERVER_IP:/home/your_username/webnovel-kb/webnovel_kb/config.py
scp f:\MCP\webnovel_kb\core\knowledge_base.py your_username@YOUR_SERVER_IP:/home/your_username/webnovel-kb/webnovel_kb/core/knowledge_base.py
# ... 或其他修改的文件

# 2. 清理缓存 + 杀僵尸进程 + 重启（必须先杀端口！）
ssh your_username@YOUR_SERVER_IP "find /home/your_username/webnovel-kb -name '__pycache__' -exec rm -rf {} + 2>/dev/null; sudo fuser -k 8765/tcp 2>/dev/null; sleep 3; sudo systemctl restart webnovel-mcp"

# 3. 检查状态
ssh your_username@YOUR_SERVER_IP "sudo systemctl status webnovel-mcp --no-pager"

# 4. 验证 OAuth 端点
ssh your_username@YOUR_SERVER_IP "curl -s http://127.0.0.1:8765/.well-known/oauth-authorization-server | python3 -m json.tool"
```

***

## OAuth 2.0 PKCE 认证

### 概述

内建 OAuth 2.0 PKCE 授权服务器，保护公网端点 `https://mcp.your_username-1.tech`。

- **内网访问**（`YOUR_SERVER_IP:8765`）：无需认证，直接使用 MCP 工具
- **公网访问**（`mcp.your_username-1.tech`）：需通过 OAuth PKCE 流程获取 access token

### OAuth 端点

| 端点        | URL                                                                 | 说明        |
| --------- | ------------------------------------------------------------------- | --------- |
| Discovery | `https://mcp.your_username-1.tech/.well-known/oauth-authorization-server` | OAuth 元数据 |
| Authorize | `https://mcp.your_username-1.tech/authorize`                              | 授权码获取     |
| Token     | `https://mcp.your_username-1.tech/token`                                  | 令牌交换      |

### 公网自定义连接器配置

在公网如grok的 MCP 自定义连接器 OAuth 表单中填入：

| 字段                          | 值                                      |
| --------------------------- | -------------------------------------- |
| Client ID                   | `mcp-client`                           |
| Client Secret               | 留空（PKCE 模式不需要）                         |
| Authorization Endpoint      | `https://mcp.your_username-1.tech/authorize` |
| Token Endpoint              | `https://mcp.your_username-1.tech/token`     |
| Scopes                      | `mcp:read mcp:write`                   |
| Token Authentication Method | `none`                                 |

### 技术实现

- **授权服务器**：`oauth_auth.py`，内建 PKCE S256 challenge 验证
- **Token 签发**：JWT (HS256)，密钥为 `OAUTH_JWT_SECRET` 环境变量
- **路由注入**：通过 `_build_app()` 函数构建 Starlette app，支持三种模式：`http`（双协议组合 SSE+SH）、`sse`（仅SSE）、`streamable-http`（仅SH）。双协议模式下 SSE 和 SH 共享同一个 FastMCP 实例，通过自定义 lifespan 初始化 StreamableHTTP session manager
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

***

## 日常运维

### 查看知识库统计

通过 MCP 工具调用：

```
mcp_webnovel-kb_stats()
```

### 手动执行维护操作（通过 SSH + Python）

以下操作已从 MCP 工具中移除，改为维护者通过服务器命令行执行：

**导入新小说：**

```bash
ssh your_username@YOUR_SERVER_IP
cd /home/your_username/webnovel-kb
source venv/bin/activate
python3 -c "
from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
kb = WebNovelKnowledgeBase()
result = kb.ingest_novel(
    file_path='/home/your_username/novels/小说名.txt',
    title='小说名',
    author='作者名',
    genre='修仙'  # 可选: 修仙/科幻/悬疑/奇幻/赛博朋克/克苏鲁/高武
)
print(result)
"
```

**提取结构化知识（实体、情节模式）：**

```bash
ssh your_username@YOUR_SERVER_IP
cd /home/your_username/webnovel-kb
source venv/bin/activate
python3 -c "
from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
kb = WebNovelKnowledgeBase()
result = kb.extract('小说名', extract_type='all', max_chunks=200, cross_chunk=True)
print(result)
"
```

**异步提取（大书推荐，避免 SSH 超时断开）：**

```bash
ssh your_username@YOUR_SERVER_IP
cd /home/your_username/webnovel-kb
source venv/bin/activate
python3 -c "
import time
from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
kb = WebNovelKnowledgeBase()
task = kb.start_async_extraction('小说名', extract_type='all', max_chunks=500)
print(f'Task ID: {task[\"task_id\"]}')

while True:
    status = kb.get_task_status(task['task_id'])
    print(f'Progress: {status.get(\"progress\", 0)}%')
    if status['status'] in ('completed', 'error'):
        print(status)
        break
    time.sleep(30)
"
```

**章纲自动提取与封存（v1.11.0 新增）：**

服务端 LLM 串行读取章节→生成叙事摘要→存入知识库→封存状态。一次一章，避免竞态写入。

```bash
ssh your_username@YOUR_SERVER_IP
cd /home/your_username/webnovel-kb
source venv/bin/activate
python3 -c "
import time
from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
kb = WebNovelKnowledgeBase()

# 启动批量提取（串行，一章一章来）
task = kb.start_outline_extraction('小说名', start_chapter=1, end_chapter=100)
print(f'Task ID: {task[\"task_id\"]}')  # 记录这个，断开 SSH 后可用 MCP 工具追踪

while True:
    status = kb.get_task_status(task['task_id'])
    print(f'Progress: {status.get(\"progress\", 0)}% ({status.get(\"total\", 0)} chapters)')
    if status['status'] in ('completed', 'error'):
        print(status['result'] if status['status'] == 'completed' else status['error'])
        break
    time.sleep(10)  # 每10秒查一次，比30秒更合理
"
```

也可以通过 MCP 工具调用（无需 SSH）：

```
# 1. 先看看小说有多少章
stats("小说名")  → 记下 chapter_count

# 2. 单章提取
extract_outline("小说名", chapter=1)
→ {novel, chapter, outline: "叙事摘要...", saved: true}

# 3. 批量提取（返回 task_id，异步后台运行）
extract_outline("小说名", chapter=1, end_chapter=100)
→ {task_id: "abc123", status: "started", total: 100}

# 4. 轮询进度（可以断开 MCP 重连后继续查）
get_task_status("abc123")
→ {status: "running", progress: 45}

# 5. 完成后自动封存，返回汇总
get_task_status("abc123")
→ {
    status: "completed",
    progress: 100,
    result: {
        "novel": "小说名",
        "total_chapters": 100,
        "success_count": 98,
        "error_count": 2,
        "elapsed_seconds": 845.3,
        "chapters_extracted": [1,2,3,...,100],
        "errors": [{"chapter": 50, "error": "LLM返回空内容"}, ...]
    }
  }
```

> **封存说明**：批量提取完成后 `_save_state()` 自动触发，章纲数据同时写入 `chapter_outlines.json` 和 ChromaDB `chapter_outlines` collection。失败的章节会在 errors 列表中标注，可以单独重试 `extract_outline("小说名", chapter=50)`。

**重建 Tantivy 索引（通常无需手动执行，首次启动自动构建）：**

```bash
ssh your_username@YOUR_SERVER_IP
cd /home/your_username/webnovel-kb
source venv/bin/activate
WEBNOVEL_KB_DATA=/home/your_username/webnovel-data python3 -c "
from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
kb = WebNovelKnowledgeBase()
kb.index_manager.build_all_indexes(kb.novels)
print(f'Tantivy built: {kb.index_manager.tantivy_index.doc_count} documents')
"
```

### 日志位置

- systemd 日志：`journalctl -u webnovel-mcp`
- 应用日志：`$WEBNOVEL_KB_DATA/logs/webnovel-kb.log`（自动轮转，默认 10MB×5 份）

### 资源监控

```bash
# 进程状态
ps aux | grep 'webnovel_kb' | grep -v grep

# 内存占用
top -p $(pgrep -f 'webnovel_kb')

# 磁盘占用
du -sh /home/your_username/webnovel-data/
du -sh /home/your_username/webnovel-data/chroma_db/ /home/your_username/webnovel-data/tantivy_index/
```

***

## 数据备份与恢复

### 备份

```bash
DATA_DIR="/home/your_username/webnovel-data"
BACKUP_DIR="/home/your_username/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/webnovel-data-backup.tar.gz \
  -C $DATA_DIR \
  chroma_db/ tantivy_index/ state/ embeddings_cache.pkl

# 保留最近 5 份备份
ls -t /home/your_username/backups/webnovel-data-backup-*.tar.gz | tail -n +6 | xargs rm -f
```

### 恢复

```bash
sudo systemctl stop webnovel-mcp
DATA_DIR="/home/your_username/webnovel-data"
BACKUP_FILE="/home/your_username/backups/webnovel-data-backup-YYYYMMDD_HHMMSS.tar.gz"
tar -xzf $BACKUP_FILE -C $DATA_DIR
sudo systemctl start webnovel-mcp
```

### 数据文件清单

| 文件/目录                         | 大小       | 说明                        |
| ----------------------------- | -------- | ------------------------- |
| `chroma_db/`                  | 1.7 GB   | ChromaDB 向量数据库            |
| `tantivy_index/`              | 160 MB   | Tantivy BM25 全文索引         |
| `embeddings_cache.pkl`        | 965 MB   | Embedding 缓存（MD5→4096维向量） |
| `chapter_outlines.json`       | \~50 KB  | 章纲数据（仅通过 MCP 工具写入）        |
| `state/novels.pkl`            | \~100 KB | 小说元数据                     |
| `state/entities.pkl`          | \~200 KB | 实体数据                      |
| `state/plot_patterns.pkl`     | \~50 KB  | 情节模式                      |
| `state/relationships.pkl`     | \~20 KB  | 关系数据                      |
| `state/graph.pkl`             | \~50 KB  | 知识图谱                      |
| `state/style_profiles.pkl`    | \~20 KB  | 风格档案                      |
| `state/writing_templates.pkl` | \~20 KB  | 写作模板                      |
| `logs/`                       | 自动轮转     | 应用日志                      |

***

## 故障排查

### 1. MCP 连接失败

```
症状：客户端显示 "list tools failed"
原因：服务未运行或端口被占用
排查：
  ps aux | grep webnovel_kb | grep -v grep
  sudo ss -tlnp 'sport = :8765'
解决：
  sudo fuser -k 8765/tcp
  sleep 3
  sudo systemctl restart webnovel-mcp
```

### 2. 端口被占用（服务无法启动）

```
症状：journalctl 显示 "address already in use"
原因：僵尸进程或旧 systemd 服务自动重启
排查：
  sudo ss -tlnp 'sport = :8765'
解决：
  sudo fuser -k 8765/tcp
  sleep 3
  sudo systemctl restart webnovel-mcp
```

### 3. 搜索返回空结果

```
症状：search/smart_search 返回空
原因：ChromaDB 数据损坏、维度不匹配或 Tantivy 索引未加载
排查：
  du -sh /home/your_username/webnovel-data/chroma_db/
  du -sh /home/your_username/webnovel-data/tantivy_index/
  mcp_webnovel-kb_stats()   # 检查 tantivy_ready 字段
解决：
  # 如果 total_chunks 为 0，需重新导入小说
  # 如果 tantivy_ready = False，执行索引重建（见日常运维）
  # 如果 total_chunks 正常但搜索为空，检查 embedding 维度配置
```

### 4. Embedding 维度不匹配

```
症状：语义搜索报错或返回空结果
原因：LLM_EMBEDDING_DIMENSIONS 配置与存储数据维度不一致
排查：
  cd /home/your_username/webnovel-kb && source venv/bin/activate
  python3 -c "
  import chromadb
  client = chromadb.PersistentClient(path='/home/your_username/webnovel-data/chroma_db')
  col = client.get_collection('webnovel_chunks')
  sample = col.get(limit=1, include=['embeddings'])
  print(f'Stored dims: {len(sample[\"embeddings\"][0])}')  # 应为 4096
  "
解决：
  确保 systemd 服务中 LLM_EMBEDDING_DIMENSIONS=4096 与存储维度一致
```

### 5. OAuth 端点返回 401

```
症状：公网访问 /authorize 或 /token 返回 401
原因：BearerAuthMiddleware 拦截了 OAuth 路由
排查：
  curl -s http://127.0.0.1:8765/.well-known/oauth-authorization-server
解决：
  确认 server.py 中 BearerAuthMiddleware 跳过了 OAUTH_PATHS
  确认 MCP_API_KEY 未设置时不会强制认证
```

### 6. 内存不足

```
症状：服务崩溃或 OOM
原因：ChromaDB + embedding 缓存占用
当前基线：17本小说，运行时内存约 1.5-2 GB
排查：
  free -h
  ps aux --sort=-%mem | grep webnovel | head -5
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

### 8. 知乎搜索失败

```
症状：smart_search 中 web_search 或 zhihu_search 返回错误
原因：ZHIHU_ACCESS_SECRET 未配置或已过期
排查：
  curl -s -H "Authorization: Bearer $ZHIHU_ACCESS_SECRET" \
    -H "X-Request-Timestamp: $(date +%s)" \
    "https://developer.zhihu.com/api/v1/content/zhihu_search?Query=test&Count=1"
解决：
  确认 systemd 服务中 ZHIHU_ACCESS_SECRET 已正确设置
  确认密钥未过期（知乎开发者平台查看）
```

### 9. ChromaDB 损坏（segfault / WAL 日志无法应用）

```
症状：工具调用段错误(SIGSEGV)、ingest_novel 删除旧 chunks 失败("Failed to apply logs to the hnsw segment writer")
原因：systemd 杀进程(systemctl stop) 时 ChromaDB 正在写入 HNSW segment，导致 WAL/Segment 文件损坏
解决：
  # 方案1（推荐）：从上次备份恢复 chroma_db
  sudo systemctl stop webnovel-mcp
  cd /home/your_username/webnovel-data
  mv chroma_db chroma_db.broken    # 保留坏库备用
  tar xzf /home/your_username/backups/webnovel-data-backup-20260530.tar.gz chroma_db state
  sudo systemctl start webnovel-mcp

  # 方案2（终极）：如果所有备份都不可用
  sudo systemctl stop webnovel-mcp
  cd /home/your_username/webnovel-data
  rm -rf chroma_db chromadb          # 清空向量库
  sudo systemctl start webnovel-mcp
  # 服务自动重建空库 → 需手动重新导入所有小说

  # 验证 chroma_db 是否正常
  find /home/your_username/webnovel-data/chroma_db -name '*.pma' -delete  # 清残留
  python3 -c "import chromadb; c=chromadb.PersistentClient(path='/home/your_username/webnovel-data/chroma_db'); print(c.list_collections())"
  # 不应 segfault，应正常打印 collection 列表
```

> **预防**：备份文件 tar.gz 而非直接 cp（避免递归陷阱）。备份时建议先 stop 服务。

### 10. 磁盘空间异常暴涨

```
症状：df -h / 显示使用率异常高（>50%），du 定位到大目录
常见元凶：
  1. chroma_db.broken — 服务重启/重导时 mv 旧库后忘记删除（可能 >100G）
  2. BrowserMetrics/*.pma — Chromium snap 自动化测试泄漏（单个 4MB，累计可达 100G+）
  3. .npm 缓存 — npm/i 缓存文件（可达数百 MB）
排查：
  du -sh /home/your_username/*/ | sort -rh | head -10
  du -sh /home/your_username/snap/chromium/common/chromium/BrowserMetrics/ 2>/dev/null
  du -sh /home/your_username/.npm 2>/dev/null
解决：
  # chroma_db.broken
  rm -rf /home/your_username/webnovel-data/chroma_db.broken

  # BrowserMetrics（Chrome/Chromium 自动化测试 bug）
  rm -rf /home/your_username/snap/chromium/common/chromium/BrowserMetrics
  ln -s /dev/null /home/your_username/snap/chromium/common/chromium/BrowserMetrics  # 永绝后患

  # .npm
  rm -rf /home/your_username/.npm
```

> **主动监控**：建议每月执行一次 `du -sh /home/your_username/*/ | sort -rh | head -5`。

***

## MCP 工具参考（v1.12.0，共 12 个）

### 浏览类

| 工具                  | 说明                           | 关键参数                              |
| ------------------- | ---------------------------- | --------------------------------- |
| `stats`             | 全局/单本统计（含 tantivy\_ready 指标） | scope（可选）, novel\_title（可选）     |
| `read_chapter`      | 读取章节完整正文                     | novel\_title, chapter             |
| `get_chapter_edges` | 提取章前章末段落（学习开头/结尾写法）          | novel\_title, chapter, paragraphs |

### 搜索类

| 工具                     | 说明                            | 关键参数                                                                      |
| ---------------------- | ----------------------------- | ------------------------------------------------------------------------- |
| `search`               | 全文统一检索（语义+BM25混合/纯语义/纯关键词/精排） | query, scope, mode, n\_results, novel\_filter, genre\_filter, alpha, use\_rerank |
| `smart_search`         | 智能搜索（LLM函数调用，5个内部工具，支持知乎搜索）   | query, n\_results, novel\_filter, genre\_filter                           |

### 分析类

| 工具               | 说明                      | 关键参数          |
| ---------------- | ----------------------- | ------------- |
| `style_analysis` | 风格分析/对比（单书名=分析，逗号分隔=对比） | novel\_titles |

### 章纲类（v1.10.0 新增）

| 工具                | 说明           | 关键参数                                                              |
| ----------------- | ------------ | ----------------------------------------------------------------- |
| `save_outline`    | 批量保存章纲（兼容单条，默认不覆盖） | novel\_title, outlines, overwrite（可选）                              |
| `get_outline`     | 获取章纲         | novel\_title, chapter（0=列表 / >0=单章 / "full"=全量文本）                |
| `search`               | 全文统一检索（含章纲 scope="outlines"） | query, scope, mode, n\_results, novel\_filter |

> **章纲写入流程**：IDE 侧 Agent 分析章节 → 调用 `save_outline` 批量存入知识库 → 后续通过 `get_outline` 或 `search(scope="outlines")` 检索参考。默认不覆盖已存在章纲（返回 skipped），设置 overwrite=True 可强制覆盖。

### 提取类（v1.11.0 新增）

| 工具                | 说明                     | 关键参数                                    |
| ----------------- | ---------------------- | --------------------------------------- |
| `extract_outline` | 提取章纲（单章或批量，服务端LLM自动存入） | novel\_title, chapter, end\_chapter（可选） |
| `manage_task`     | 管理异步任务（查询进度或取消）        | task\_id, action（"status"/"cancel"）     |

> **章纲提取流程**：`extract_outline("书名", chapter=100)` 提取单章，同步返回章纲正文。`extract_outline("书名", chapter=1, end_chapter=20)` 启动批量异步提取（上限 20 章），返回 task_id，通过 `manage_task(task_id)` 查询进度，`manage_task(task_id, action="cancel")` 取消任务。提取完成后自动封存。

### 工具详细参数

#### `stats` — 知识库统计

| 参数           | 类型  | 默认值 | 说明               |
| ------------ | --- | --- | ---------------- |
| novel\_title | str | ""  | 书名（模糊匹配），留空=全局统计 |

返回字段含 `tantivy_ready`，指示 BM25 关键词搜索是否可用。

#### `read_chapter` — 读取章节

| 参数           | 类型  | 默认值 | 说明           |
| ------------ | --- | --- | ------------ |
| novel\_title | str | 必填  | 书名（模糊匹配）     |
| chapter      | int | 1   | 章节号（1-based） |

#### `get_chapter_edges` — 章前章末段落

| 参数           | 类型  | 默认值 | 说明           |
| ------------ | --- | --- | ------------ |
| novel\_title | str | 必填  | 书名（模糊匹配）     |
| chapter      | int | 1   | 章节号（1-based） |
| paragraphs   | int | 2   | 章前/章末各提取几段   |

#### `search` — 全文检索

| 参数                   | 类型    | 默认值       | 说明                            |
| -------------------- | ----- | --------- | ----------------------------- |
| query                | str   | 必填        | 搜索关键词或自然语言描述                  |
| mode                 | str   | "hybrid"  | hybrid/semantic/bm25/rerank   |
| n\_results           | int   | 10        | 返回结果数量                        |
| novel\_filter        | str   | ""        | 限定书名（模糊匹配）                    |
| genre\_filter        | str   | ""        | 限定类型（修仙/科幻/悬疑/奇幻/赛博朋克/克苏鲁/高武） |
| chapter\_filter      | str   | ""        | 限定章节（按标题匹配）                   |
| alpha                | float | 0.6       | hybrid 语义权重（0\~1，越大越偏语义）      |
| use\_rerank          | bool  | False     | 是否启用 Cross-encoder 精排         |
| output\_format       | str   | "compact" | compact/clean/raw             |
| max\_content\_length | int   | 0         | 每条结果字数上限（0=不限）                |
| dedupe               | bool  | True      | 是否去重                          |

#### `smart_search` — 智能搜索

| 参数             | 类型  | 默认值       | 说明                |
| -------------- | --- | --------- | ----------------- |
| query          | str | 必填        | 自然语言描述（可模糊、口语化）   |
| n\_results     | int | 5         | 每次工具搜索返回数         |
| novel\_filter  | str | ""        | 限定书名（模糊匹配）        |
| genre\_filter  | str | ""        | 限定类型              |
| output\_format | str | "compact" | compact/clean/raw |

> smart\_search 调用 MiMo 模型进行函数调用模式搜索，响应时间约 30-120 秒。模型自主调用 5 个内部工具：
>
> - **内部知识库**：search\_text / search\_patterns / search\_entities
> - **外部知识**：web\_search（知乎全网搜索）/ zhihu\_search（知乎站内搜索）
>
> 最多 200 轮迭代，支持并行工具调用。LLM 调用失败时自动降级为 hybrid 搜索。

#### `style_analysis` — 风格分析

| 参数            | 类型  | 默认值 | 说明                 |
| ------------- | --- | --- | ------------------ |
| novel\_titles | str | 必填  | 书名或逗号分隔的书名列表（模糊匹配） |

> **注意**：首次分析较慢（约 30-120 秒），结果会缓存。

#### `save_outline` — 批量保存章纲

| 参数           | 类型          | 默认值 | 说明                                                                               |
| ------------ | ----------- | --- | -------------------------------------------------------------------------------- |
| novel\_title | str         | 必填  | 书名（模糊匹配）                                                                         |
| outlines     | List\[dict] | 必填  | 章纲列表，每项：chapter(int), content(str), outline\_type(str, 默认"章纲"), tags(List\[str]) |

> 单条示例：`outlines=[{"chapter": 100, "content": "..."}]`
> 批量示例：`outlines=[{"chapter": 1, "content": "..."}, {"chapter": 2, "content": "..."}]`
> 已有章纲会被覆盖更新。IDE 侧产出章纲后调用此工具持久化。

#### `get_outline` — 获取章纲

| 参数           | 类型               | 默认值 | 说明                                                        |
| ------------ | ---------------- | --- | --------------------------------------------------------- |
| novel\_title | str              | 必填  | 书名（模糊匹配）                                                  |
| chapter      | int 或 str        | 0   | 0=章节列表（仅章节号和类型）；>0=该章完整章纲；"full"=全书全量章纲文本（按章节号排序串联） |

#### `extract_outline` — 提取章纲（v1.11.0 新增）

| 参数           | 类型  | 默认值 | 说明                                               |
| ------------ | --- | --- | ------------------------------------------------ |
| novel\_title | str | 必填  | 书名（模糊匹配）                                         |
| chapter      | int | 必填  | 章节号（1-based），批量模式作起始章节                           |
| end\_chapter | int | 0   | 结束章节号。为 0 时只提取单章；>0 时批量提取 chapter 到 end\_chapter |

> 单章同步返回章纲正文，批量异步返回 task\_id。单章约 5-15 秒，批量串行处理。

#### `manage_task` — 管理异步任务（v1.11.2 新增）

| 参数       | 类型  | 默认值     | 说明                         |
| -------- | --- | ------- | -------------------------- |
| task\_id | str | 必填      | 任务ID（由 `extract_outline` 返回） |
| action   | str | "status" | "status"=查询进度 / "cancel"=取消任务 |

> 查询返回 status（running/completed/cancelled/error）、progress（0-100）。取消是异步的，任务会在当前章节完成后停止，用 `action="status"` 确认最终状态。

***

## 已知问题与限制

### 1. `save_outline` 不支持并行调用

多个 Agent 同时调用 `save_outline` 时存在竞态条件。`_save_state()` 序列化全部内存状态写入 7 个 JSON 文件，没有锁保护。并发写入会导致后者覆盖前者。

**临时规避**：一次只让一个 Agent 调用章纲工具。后续版本将添加 `threading.Lock` 串行化状态写入。

### 2. 章节号 `first_chapter`/`last_chapter` 可能为 0

部分小说（如永夜君王）的 stats 返回 `first_chapter: 0, last_chapter: 0`。原因是识别到第 0 章（前言/序章），但不影响 `read_chapter` 按 1-based 索引读取正文章节。

### 3. `stats` 的 `chunk_count` 为空需 `chunks_indexed`

`stats` 返回两个计数：

- `chunk_count` — 小说元数据中的预估分块数（可能为0）
- `chunks_indexed` — ChromaDB 中实际存储的分块数（准确值）

使用 `chunks_indexed` 判断实际数据量。

***


