# WebNovel Knowledge Base - 项目维护指南

> 版本：v1.5 | 更新日期：2026-05-01

## 目录
1. [架构概览](#架构概览)
2. [服务部署](#服务部署)
3. [日常运维](#日常运维)
4. [数据备份与恢复](#数据备份与恢复)
5. [故障排查](#故障排查)
6. [MCP 工具参考](#mcp-工具参考)

---

## 架构概览

### 模块化架构（v1.5）
```
webnovel_kb/
├── server.py          # MCP 服务入口（精简版，~40行）
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
│   └── mcp_tools.py   # MCP 工具注册（12个工具）
└── utils/
    ├── dedupe.py      # 去重逻辑
    └── format.py      # 输出格式化
```

### 数据流向
```
MCP Client (Trae IDE)
       │ SSE/HTTP (port 8765)
       ▼
  MCP Server (server.py)
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

---

## 服务部署

### 环境变量

所有配置通过环境变量管理，参考 `.env.example`：

| 变量名 | 说明 |
|--------|------|
| `MCP_TRANSPORT` | 传输方式：`sse` 或 `stdio` |
| `MCP_HOST` | 监听地址，公网部署建议 `0.0.0.0` |
| `MCP_PORT` | 监听端口，默认 `8765` |
| `WEBNOVEL_KB_DATA` | 数据目录绝对路径 |
| `LLM_API_KEY` | LLM 服务 API 密钥 |
| `LLM_BASE_URL` | Embedding API 地址 |
| `LLM_CHAT_BASE_URL` | Chat API 地址 |
| `LLM_EMBEDDING_MODEL` | Embedding 模型名称 |
| `LLM_RERANK_MODEL` | Rerank 模型名称 |
| `LLM_CHAT_MODEL` | Chat 模型名称 |
| `LLM_EMBEDDING_DIMENSIONS` | Embedding 向量维度 |

> **兼容性说明**：旧版部署可能使用供应商特定的变量名前缀。
> v1.5 已统一为 `LLM_*` 前缀，部署时需同步更新 systemd 服务中的环境变量名称。

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
journalctl -u webnovel-mcp --since '1h' # 最近1小时
```

### 手动启动（调试用）

```bash
cd /path/to/webnovel-kb
source venv/bin/activate
MCP_TRANSPORT=sse MCP_HOST=0.0.0.0 MCP_PORT=8765 python -m webnovel_kb
```

### 更新部署

```bash
# 1. 拉取最新代码
cd /path/to/webnovel-kb
git pull origin main

# 2. 重启服务
sudo systemctl restart webnovel-mcp

# 3. 验证
sudo systemctl status webnovel-mcp
```

---

## 日常运维

### 查看知识库统计
通过 MCP 工具调用：
```
mcp_webnovel-kb_get_stats()
```

### 监控异步任务
```
mcp_webnovel-kb_get_task_status(task_id="xxx")
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

# 磁盘占用（替换为实际数据目录）
du -sh /path/to/data/
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
| `bm25_index.pkl` | BM25 索引（旧版，可删除） |
| `faiss_index.faiss` | FAISS 索引（旧版，可删除） |
| `faiss_index.meta.json` | FAISS 元数据（旧版，可删除） |
| `tantivy_index/` | Tantivy 索引（旧版，可删除） |

> **注意**：`bm25_index.pkl`、`faiss_index.*`、`tantivy_index/` 是旧版优化索引，v1.5 使用 ChromaDB 内置搜索，这些文件可安全删除以释放磁盘空间。

---

## 故障排查

### 1. MCP 连接失败
```
症状：客户端显示 "list tools failed"
原因：服务未运行或端口被占用
排查：
  ps aux | grep webnovel_kb
  netstat -tlnp | grep 8765
解决：
  sudo systemctl restart webnovel-mcp
```

### 2. 搜索返回空结果
```
症状：search/search_knowledge 返回空
原因：ChromaDB 数据损坏或维度不匹配
排查：
  # 检查 ChromaDB 数据目录大小
  du -sh /path/to/data/chroma_db/

  # 通过 MCP 检查统计
  mcp_webnovel-kb_get_stats()
解决：
  # 如果 total_chunks 为 0，需要重新导入小说
  # 如果 total_chunks 正常但搜索为空，检查 embedding 维度配置
```

### 3. Embedding 维度不匹配
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

### 4. 异步任务卡住
```
症状：get_task_status 长时间 running
原因：LLM API 响应慢或超时
排查：
  journalctl -u webnovel-mcp --since '30min' | grep -i error
解决：
  sudo systemctl restart webnovel-mcp
```

### 5. 内存不足
```
症状：服务崩溃或 OOM
原因：ChromaDB + embedding 缓存占用较大
排查：
  free -h
  ps aux --sort=-%mem | head -5
解决：
  # 短期：重启服务
  sudo systemctl restart webnovel-mcp

  # 长期：清理旧索引文件
  rm /path/to/data/bm25_index.pkl
  rm /path/to/data/faiss_index.*
  rm -rf /path/to/data/tantivy_index/
```

### 6. API 连接问题
```
症状：embedding 或 rerank 调用失败
原因：API 服务不可用或密钥过期
排查：
  curl -s -H "Authorization: Bearer $LLM_API_KEY" $LLM_BASE_URL/models
解决：
  检查 systemd 服务中的 LLM_API_KEY 是否有效
```

---

## MCP 工具参考

### 搜索类
| 工具 | 说明 | 关键参数 |
|------|------|----------|
| `search` | 统一搜索（自动选择策略） | query, n_results, mode |
| `search_knowledge` | 知识搜索 | query, n_results |
| `search_entities` | 实体搜索 | query, n_results |
| `get_entity_relations` | 实体关系查询 | entity_name |

### 提取类
| 工具 | 说明 | 关键参数 |
|------|------|----------|
| `extract` | 知识提取 | novel_title, extract_type |
| `ingest_novel` | 导入小说 | file_path, title, author, genre |

### 分析类
| 工具 | 说明 | 关键参数 |
|------|------|----------|
| `analyze_style` | 风格分析 | novel_title |
| `compare_styles` | 风格对比 | novel_titles（逗号分隔） |
| `novel_stats` | 小说统计 | novel_title |

### 系统类
| 工具 | 说明 |
|------|------|
| `list_novels` | 列出所有小说 |
| `get_stats` | 系统统计 |
| `get_task_status` | 异步任务状态 |

---

## 项目历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2025-04-30 | 初始版本，单体架构（server.py 2200+ 行） |
| v1.5 | 2026-05-01 | 模块化重构，修复语义搜索维度问题，清理旧文件 |