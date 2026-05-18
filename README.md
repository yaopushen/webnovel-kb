# WebNovel Knowledge Base

网文知识库 MCP 服务器 - 用于小说分析、知识提取和智能搜索。

## 功能

- **小说导入**：支持 TXT 格式小说导入和分块索引
- **常规搜索**：语义搜索、BM25 关键词搜索、混合搜索、LLM 重排序搜索
- **智能搜索**：通用LLM模型驱动的多轮迭代深度搜索
- **知识提取**：实体提取、情节模式提取、写作模板提取、场景模式提取
- **风格分析**：写作风格分析、风格对比、幽默场景提取
- **章节浏览**：章节列表、完整章节内容、章前章末段落提取
- **OAuth 2.0 PKCE 认证**：内置授权服务器，保护公网端点
- **异步任务**：支持后台异步提取任务

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

| 变量名                        | 说明                               | 默认值               |
| -------------------------- | -------------------------------- | ----------------- |
| `WEBNOVEL_KB_DATA`         | 数据目录路径                           | `./webnovel_data` |
| `LLM_API_KEY`              | LLM API 密钥                       | -                 |
| `LLM_BASE_URL`             | Embedding API 地址                 | -                 |
| `LLM_CHAT_BASE_URL`        | Chat API 地址                      | -                 |
| `LLM_EMBEDDING_MODEL`      | Embedding 模型名                    | -                 |
| `LLM_RERANK_MODEL`         | Rerank 模型名                       | -                 |
| `LLM_CHAT_MODEL`           | Chat 模型名                         | -                 |
| `LLM_EMBEDDING_DIMENSIONS` | Embedding 维度                     | `4096`            |
| `MCP_HOST`                 | 监听地址                             | `0.0.0.0`         |
| `MCP_PORT`                 | 监听端口                             | `8765`            |
| `MCP_TRANSPORT`            | 传输方式 (`stdio`/`streamable-http`) | `streamable-http` |
| `MCP_OAUTH_ISSUER_URL`     | OAuth 签发者 URL（可选）                | -                 |
| `OAUTH_JWT_SECRET`         | JWT 签名密钥（可选）                     | -                 |
| `OAUTH_TOKEN_EXPIRY`       | Token 有效期（秒）                     | `86400`           |

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

### Streamable-HTTP 模式（推荐，用于远程访问）

```bash
MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8765 python -m webnovel_kb
```

### OAuth 2.0 PKCE 认证模式（公网部署）

```bash
MCP_TRANSPORT=streamable-http \
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

## MCP 工具

### 浏览类（5个）

| 工具名                 | 说明        |
| ------------------- | --------- |
| `stats`             | 全局/单本统计   |
| `list_novels`       | 列出所有小说    |
| `list_chapters`     | 列出章节列表    |
| `read_chapter`      | 读取章节完整内容  |
| `get_chapter_edges` | 提取章前/章末段落 |

### 搜索类（5个）

| 工具名                    | 说明                                   |
| ---------------------- | ------------------------------------ |
| `search`               | 统一搜索（支持 semantic/bm25/hybrid/rerank） |
| `smart_search`         | 通用LLM模型智能搜索（多轮迭代）                    |
| `search_knowledge`     | 搜索情节模式/写作模板                          |
| `search_entities`      | 搜索实体（角色/功法/地点等）                      |
| `get_entity_relations` | 查询实体关系网                              |

### 分析类（1个）

| 工具名              | 说明        |
| ---------------- | --------- |
| `style_analysis` | 写作风格分析/对比 |

## 目录结构

```
webnovel_kb/
├── __init__.py
├── __main__.py
├── config.py
├── server.py
├── oauth_auth.py          # OAuth 2.0 PKCE 授权服务器
├── data_models.py
├── prompts.py
├── requirements.txt
├── api/
│   ├── clients.py         # API 客户端
│   └── mcp_tools.py       # MCP 工具定义（11个）
├── core/
│   ├── chunker.py         # 文本分块
│   ├── indexer.py         # 索引管理（ChromaDB）
│   ├── knowledge_base.py  # 核心知识库
│   └── state.py           # 状态管理
├── extraction/
│   ├── entities.py        # 实体提取
│   ├── plot_patterns.py   # 情节模式
│   ├── scene_patterns.py  # 场景模式
│   └── writing_templates.py # 写作模板
├── search/
│   ├── bm25_search.py     # BM25 搜索
│   ├── hybrid.py          # 混合搜索（RRF 融合）
│   ├── rerank.py          # LLM 重排序
│   ├── semantic.py        # 语义搜索
│   └── unified.py         # 统一搜索入口
├── analysis/
│   ├── humor.py           # 幽默分析
│   └── style.py           # 风格分析
├── scripts/
│   └── build_optimized_indexes.py
└── utils/
    ├── dedupe.py          # 去重
    ├── format.py          # 格式化
    ├── exceptions.py
    ├── logging_config.py
    └── query_cache.py
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

当前版本：1.9.1

## 许可证

MIT
