# WebNovel Knowledge Base

网文知识库 MCP 服务器 - 用于小说分析、知识提取和智能搜索。

## 功能

- **小说导入**：支持 TXT 格式小说导入和分块索引
- **常规搜索**：语义搜索、BM25 关键词搜索、混合搜索、LLM 重排序搜索
- **智能搜索**：通用LLM模型驱动的多轮迭代深度搜索，支持外部知识扩展
- **章纲管理**：章纲保存、获取、自动提取（单章/批量）
- **知识提取**：实体提取、情节模式提取、写作模板提取、场景模式提取
- **风格分析**：写作风格分析、风格对比、幽默场景提取
- **章节浏览**：章节列表、完整章节内容、章前章末段落提取
- **范例生成**：基于经典章节的模仿改写（minimal/light/moderate 三级）
- **Agent 知识层**：自动存储搜索洞察，支持自定义知识写入
- **OAuth 2.0 PKCE 认证**：内置授权服务器，保护公网端点
- **异步任务**：支持后台异步提取任务（批量章纲提取等）

## 安装

```bash
pip install -r requirements.txt
```

## 配置

复制 `.env.example` 为 `.env` 并填入配置：

```bash
cp .env.example .env
```

### 环境变量说明

| 变量名 | 说明 | 默认值 |
| --- | --- | --- |
| `WEBNOVEL_KB_DATA` | 数据目录路径 | `./webnovel_data` |
| `LLM_API_KEY` | LLM API 密钥 | - |
| `LLM_BASE_URL` | Embedding API 地址 | - |
| `LLM_CHAT_BASE_URL` | Chat API 地址 | - |
| `LLM_CHAT_API_KEY` | Chat API 密钥（可选，默认同 LLM_API_KEY） | - |
| `LLM_EMBEDDING_MODEL` | Embedding 模型名 | - |
| `LLM_RERANK_MODEL` | Rerank 模型名 | - |
| `LLM_CHAT_MODEL` | Chat 模型名 | - |
| `LLM_EMBEDDING_DIMENSIONS` | Embedding 维度 | `4096` |
| `MCP_HOST` | 监听地址 | `0.0.0.0` |
| `MCP_PORT` | 监听端口 | `8765` |
| `MCP_TRANSPORT` | 传输方式：`http`(双协议) / `streamable-http` / `sse` / `stdio` | `http` |
| `MCP_OAUTH_ISSUER_URL` | OAuth 签发者 URL（可选，留空不启用 OAuth） | - |
| `OAUTH_JWT_SECRET` | JWT 签名密钥（可选） | - |
| `OAUTH_TOKEN_EXPIRY` | Token 有效期（秒） | `86400` |
| `ZHIHU_ACCESS_SECRET` | 外部搜索 API 密钥（可选，smart_search 外部搜索用） | - |

### 兼容性

支持任何 OpenAI 兼容的 API 服务，包括但不限于：

- 智谱 AI
- 月之暗面
- 深度求索
- 阿里通义
- 自建 OpenAI 兼容服务

## 运行

### stdio 模式（用于 MCP 客户端）

```bash
python -m webnovel_kb
```

### 双协议模式（推荐，同时支持 SSE 和 Streamable HTTP）

```bash
MCP_TRANSPORT=http MCP_HOST=0.0.0.0 MCP_PORT=8765 python -m webnovel_kb
```

### OAuth 2.0 PKCE 认证模式（公网部署）

```bash
MCP_TRANSPORT=http \
MCP_OAUTH_ISSUER_URL=https://your-domain.com \
OAUTH_JWT_SECRET=your-secret-key \
MCP_HOST=0.0.0.0 MCP_PORT=8765 \
python -m webnovel_kb
```

### systemd 服务

```bash
sudo cp webnovel-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable webnovel-mcp
sudo systemctl start webnovel-mcp
```

## MCP 工具（v1.12.1，共 12 个）

### 浏览类

| 工具名 | 说明 |
| --- | --- |
| `stats` | 全局/单本统计（scope: global/novels/knowledge/guide） |
| `read_chapter` | 读取章节完整内容 |
| `get_chapter_edges` | 提取章前/章末段落 |

### 搜索类

| 工具名 | 说明 |
| --- | --- |
| `search` | 统一搜索（semantic/bm25/hybrid/rerank，支持 scope: chunks/outlines/agent_knowledge/all） |
| `smart_search` | 智能搜索（LLM 多轮迭代，支持外部知识扩展） |

### 分析类

| 工具名 | 说明 |
| --- | --- |
| `style_analysis` | 写作风格分析/对比（支持 output_format: compact/clean/raw） |

### 章纲类

| 工具名 | 说明 |
| --- | --- |
| `save_outline` | 批量保存章纲 |
| `get_outline` | 获取章纲（chapter: 0=列表/>0=单章/"full"=全量） |
| `extract_outline` | 自动提取章纲（单章同步/批量异步，上限 20 章） |

### 知识与创作类

| 工具名 | 说明 |
| --- | --- |
| `add_knowledge` | 写入 Agent 知识（研究/套路/风格指南） |
| `generate_sample` | 范例生成（minimal/light/moderate 三级改写） |
| `manage_task` | 管理异步任务（status/cancel） |

> 使用 `stats(scope="guide")` 获取完整的工具使用手册。

## 目录结构

```
webnovel_kb/
├── __init__.py
├── __main__.py
├── config.py              # 环境变量配置
├── server.py              # MCP 服务入口（双协议支持）
├── oauth_auth.py          # OAuth 2.0 PKCE 授权服务器
├── data_models.py         # 数据模型
├── prompts.py             # LLM 提示词
├── worker.py              # 后台 Worker 进程
├── api/
│   ├── clients.py         # API 客户端（AsyncEmbedding/Rerank/Chat）
│   ├── mcp_tools.py       # MCP 工具门面
│   └── tools/             # 工具分组
│       ├── browse.py      # stats / read_chapter / get_chapter_edges
│       ├── search.py      # search / smart_search
│       ├── outline.py     # save_outline / get_outline / extract_outline
│       ├── analysis.py    # style_analysis
│       ├── knowledge.py   # add_knowledge
│       ├── creation.py    # generate_sample
│       └── task.py        # manage_task
├── core/
│   ├── knowledge_base.py  # 门面类（~711 行）
│   ├── knowledge_store.py # Agent 知识层
│   ├── novel_reader.py    # 章节读取
│   ├── outline_manager.py # 章纲管理
│   ├── task_manager.py    # 异步任务管理
│   ├── chunker.py         # 文本分块
│   ├── indexer.py         # 索引管理
│   └── state.py           # 状态持久化
├── search/
│   ├── smart.py           # 智能搜索引擎
│   ├── external.py        # 外部搜索 API
│   ├── semantic.py        # 语义搜索（ChromaDB）
│   ├── bm25_search.py     # BM25 搜索（Tantivy）
│   ├── hybrid.py          # 混合搜索（RRF）
│   ├── rerank.py          # LLM 重排序
│   └── unified.py         # 统一搜索入口
├── extraction/
│   ├── entities.py        # 实体提取
│   ├── outlines.py        # 章纲自动提取
│   ├── plot_patterns.py   # 情节模式
│   ├── scene_patterns.py  # 场景模式
│   └── writing_templates.py # 写作模板
├── analysis/
│   ├── humor.py           # 幽默分析
│   └── style.py           # 风格分析
├── creation/
│   └── sample_generator.py # 范例生成
├── scripts/
│   └── build_optimized_indexes.py
└── utils/
    ├── novel_resolver.py  # 小说模糊匹配
    ├── chinese_numbers.py # 中文数字转换
    ├── dedupe.py          # 去重
    ├── format.py          # 格式化
    ├── exceptions.py      # 自定义异常
    ├── logging_config.py  # 日志配置
    └── query_cache.py     # 查询缓存
```

## 认证说明

### 内网访问

- 直接访问 `http://localhost:8765/mcp`
- 无需认证

### 公网访问（OAuth 2.0 PKCE）

1. 配置 `MCP_OAUTH_ISSUER_URL` 和 `OAUTH_JWT_SECRET`
2. 客户端通过 PKCE 流程获取 access token
3. 后续请求携带 `Authorization: Bearer <token>`

## 版本

当前版本：1.12.1

详见 [CHANGELOG.md](CHANGELOG.md)

## 许可证

MIT
