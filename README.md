# WebNovel Knowledge Base

一个专为中文网络小说设计的知识库 MCP 服务器，支持语义检索、情节模式提取、风格分析等功能。

## 功能特性

- **语义检索** - 基于 BGE-small-zh 向量模型 + FAISS/Tantivy 高性能索引
- **混合搜索** - BM25 关键词 + 语义向量融合，支持 Rerank 精排
- **降噪输出** - 默认输出干净文本，无需手动清洗结构化数据
- **情节模式提取** - 自动识别常见网文套路（退婚流、系统流、扮猪吃虎等）
- **写作模板** - 提取可复用的场景写作结构
- **风格分析** - 分析小说的节奏、对话比、钩子密度等
- **实体关系图谱** - 角色、地点、组织及其关系网络

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

## 搜索降噪

三个搜索工具统一支持降噪参数：

```python
search(
    query="灵能催化技术",
    mode="hybrid",           # semantic/bm25/hybrid/rerank
    output_format="compact", # raw/compact/clean (默认 compact)
    max_content_length=200,  # 每条最大字数 (0=不限制)
    dedupe=True              # 去重 (默认 True)
)
```

**输出格式对比**：

| 格式 | 示例输出 |
|---|---|
| `raw` | `{"text": "...", "metadata": {...}, "relevance": 0.84}` |
| `compact` | `["[《隐秘死角》 第3章] 联邦政府明令禁止..."]` |
| `clean` | `["联邦政府明令禁止..."]` |

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
    title="隐秘死角",
    author="滚开",
    genre="奇幻"
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
│  │ (FAISS) │ │(Tantivy)│ │  融合   │ │ (LLM)   │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
├─────────────────────────────────────────────────────┤
│  Storage                                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
│  │ ChromaDB    │ │  FAISS      │ │  Tantivy    │  │
│  │ (向量存储)  │ │ (向量索引)  │ │ (全文索引)  │  │
│  └─────────────┘ └─────────────┘ └─────────────┘  │
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
| `LLM_EMBEDDING_MODEL` | 嵌入模型名称 | 否（默认 bge-small-zh） |
| `LLM_EMBEDDING_DIMENSIONS` | 向量维度 | 否（默认 512） |
| `WEBNOVEL_KB_DATA` | 数据存储路径 | 否（默认 `./data`） |
| `MCP_TRANSPORT` | 传输模式：stdio/sse | 否（默认 stdio） |
| `MCP_HOST` | SSE 监听地址 | 否 |
| `MCP_PORT` | SSE 监听端口 | 否 |

## 性能指标

- 向量索引：42,000+ 文档，检索延迟 < 50ms
- BM25 索引：44,000+ 文档，检索延迟 < 10ms
- 内存占用：约 2GB（含模型）

## 开发计划

详见 [ROADMAP.md](./ROADMAP.md)

## License

MIT
