# WebNovel Knowledge Base

网文知识库 MCP 服务器 - 用于小说分析、知识提取和智能搜索。

## 功能特性

- **语义检索** - 基于向量嵌入 + ChromaDB 向量数据库
- **混合搜索** - BM25 关键词 + 语义向量融合，支持 Rerank 精排
- **降噪输出** - 默认输出干净文本，无需手动清洗结构化数据
- **情节模式提取** - 自动识别跨章节长线叙事模式
- **写作模板** - 提取可复用的场景写作结构
- **风格分析** - 分析小说的节奏、对话比、钩子密度等
- **实体关系图谱** - 角色、地点、组织及其关系网络
- **异步任务** - 支持后台异步提取，不阻塞主服务

## MCP 工具列表

| 工具 | 说明 |
|---|---|
| `ingest_novel` | 导入小说到知识库 |
| `search` | 统一检索（semantic/bm25/hybrid/rerank） |
| `search_knowledge` | 搜索情节模式/写作模板 |
| `search_entities` | 语义搜索实体（角色、地点等） |
| `analyze_style` | 分析小说写作风格 |
| `compare_styles` | 对比多本小说风格 |
| `novel_stats` | 获取小说统计信息 |
| `get_entity_relations` | 查询实体关系网络 |
| `extract` | 提取情节模式/实体/模板（支持异步） |
| `get_task_status` | 查询异步任务状态 |
| `list_novels` | 列出已导入小说 |
| `get_stats` | 获取知识库统计 |

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 API Key
```

### 3. 启动服务

**stdio 模式**（Claude Desktop 等）：
```bash
python -m webnovel_kb
```

**SSE 模式**（Trae、Cursor 等）：
```bash
MCP_TRANSPORT=sse MCP_HOST=0.0.0.0 MCP_PORT=8765 python -m webnovel_kb
```

### 4. 导入小说

```python
ingest_novel(
    file_path="/path/to/novel.txt",
    title="小说标题",
    author="作者",
    genre="类型"
)
```

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   MCP Server                        │
│  (FastMCP + SSE/stdio transport)                   │
├─────────────────────────────────────────────────────┤
│  Search Engine                                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ Semantic│ │  BM25   │ │ Hybrid  │ │ Rerank  │  │
│  │(ChromaDB)│ │(jieba)  │ │  融合   │ │ (LLM)   │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
├─────────────────────────────────────────────────────┤
│  Storage                                            │
│  ┌─────────────────────┐ ┌─────────────────────┐   │
│  │ ChromaDB            │ │  JSON 文件           │   │
│  │ (向量存储 + 索引)   │ │ (实体/关系/模式)     │   │
│  └─────────────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────┤
│  Knowledge Graph (NetworkX)                         │
│  实体 · 关系 · 情节模式 · 写作模板                  │
└─────────────────────────────────────────────────────┘
```

## 配置说明

| 环境变量 | 说明 | 必需 |
|---|---|---|
| `LLM_API_KEY` | LLM API Key（支持 OpenAI 兼容 API） | 提取功能需要 |
| `LLM_BASE_URL` | Embedding API 地址 | 否 |
| `LLM_CHAT_BASE_URL` | Chat API 地址 | 否 |
| `LLM_EMBEDDING_MODEL` | 嵌入模型名称 | 否 |
| `LLM_EMBEDDING_DIMENSIONS` | 向量维度 | 否（默认 512） |
| `WEBNOVEL_KB_DATA` | 数据存储路径 | 否（默认 `./data`） |
| `MCP_TRANSPORT` | 传输模式：stdio/sse | 否（默认 stdio） |
| `MCP_HOST` | SSE 监听地址 | 否 |
| `MCP_PORT` | SSE 监听端口 | 否 |

## 目录结构

```
webnovel_kb/
├── __init__.py
├── __main__.py
├── config.py
├── server.py
├── data_models.py
├── prompts.py
├── requirements.txt
├── api/
│   ├── clients.py      # API 客户端
│   └── mcp_tools.py    # MCP 工具定义
├── core/
│   ├── chunker.py      # 文本分块
│   ├── indexer.py      # 索引管理
│   ├── knowledge_base.py # 核心知识库
│   └── state.py        # 状态管理
├── extraction/
│   ├── entities.py     # 实体提取
│   ├── plot_patterns.py # 情节模式
│   ├── scene_patterns.py # 场景模式
│   └── writing_templates.py # 写作模板
├── search/
│   ├── bm25_search.py  # BM25 搜索
│   ├── hybrid.py       # 混合搜索
│   ├── rerank.py       # 重排序
│   ├── semantic.py     # 语义搜索
│   └── unified.py      # 统一搜索
├── analysis/
│   ├── humor.py        # 幽默分析
│   └── style.py        # 风格分析
└── utils/
    ├── dedupe.py       # 去重
    └── format.py       # 格式化
```

## 维护指南

详见 [MAINTENANCE.md](./MAINTENANCE.md)

## 版本

当前版本：1.5

## 许可证

MIT
