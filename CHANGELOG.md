# WebNovel Knowledge Base - 版本更新日志 (Changelog)

> 记录 WebNovel KB 项目的版本演进、重构与功能迭代历史。

---

## 版本历史总览

| 版本 | 日期 | 说明 |
| :--- | :--- | :--- |
| **v1.12.1** | **2026-06-10** | **内存SQLite优化 + 工具Docstring瘦身 + 智能搜索返回过滤**（详见下方） |
| v1.12.0 | 2026-06-10 | 代码职责拆分 + SSE/HTTP 双协议支持（详见下方） |
| v1.11.2 | 2026-06-03 | get_outline 全量 + manage_task + 删除 delete_outline（详见下方） |
| v1.11.1 | 2026-06-03 | cancel_task + 批量上限 20 章（详见下方） |
| v1.11.0 | 2026-06-30 | 章纲自动提取（extract_outline/batch）+ get_task_status（详见下方） |
| v1.10.0 | 2026-05-30 | 章纲工具（save/get/search/delete）+ ChromaDB 修复 + 空间清理（详见下方） |
| v1.9.2 | 2026-05-19 | smart_search 接入知乎搜索（详见下方） |
| v1.9.1 | 2026-05-18 | 维护文档更新 |
| v1.9 | 2026-05-17 | 架构瘦身 + 技术债务清理（移除 FAISS/rank_bm25，统一 Tantivy+ChromaDB） |
| v1.8 | 2026-05-15 | 工具参考全面更新（10→11，新增 get_chapter_edges），smart_search 描述修正 |
| v1.7 | 2026-05-15 | SSE→Streamable-HTTP，新增 OAuth 2.0 PKCE 认证，公网端点保护 |
| v1.6 | 2026-05-11 | 工具精简（15→10），新增 smart_search，移除管理工具，合并 stats/style_analysis |
| v1.5 | 2026-05-01 | 模块化重构，修复语义搜索维度问题，清理旧文件 |
| v1.0 | 2025-04-30 | 初始版本，单体架构（server.py 2200+ 行） |

---

## 详细变更记录

### v1.12.1 变更详记

**后台内存优化（Pickle to SQLite3）：**
- ✅ **SQLite 缓存替换** — 在 `api/clients.py` 中引入 `SQLiteEmbeddingCache` 封装了 SQLite3 的按需查写操作，替换了原有一次性反序列化加载 965MB `embeddings_cache.pkl` 到内存的设计。
- ✅ **内存大幅降低** — 主服务（`webnovel-mcp`）和 Worker 进程（`webnovel-mcp-worker`）的运行内存从约 3.5GB 降至每个服务仅约 150MB~250MB，整体内存开销降低了 85% 以上。
- ✅ **平滑数据迁移** — 在服务启动时自动检测是否存在旧的 `pkl` 文件，若存在且 SQLite 为空，则在后台守护线程自动批量迁移（每批 1000 条），迁移完成后将原 pickle 文件重命名为 `.pkl.migrated`。

**工具描述精简（Token 节省）：**
- ✅ **Docstring 极限精简** — 对 12 个 MCP 工具中挂载函数的 docstrings 进行了极限精简，均缩短至 1 行简短英文摘要，极大地节省了每次会话客户端加载的 System Prompt Token。
- ✅ **动态工具指南** — 新建了专用的 `api/tools/tools_guide.py` 模块，将这 12 个工具的原有详细入参、出参和最佳实践等整理成 Markdown 格式的完整手册。
- ✅ **按需提取** — 拓展了 `stats` 工具，使用 `stats(scope="guide")` 即可动态获取完整的工具使用手册，供 Agent 或开发者随时提取。

**返回过滤（智能搜索返回精简）：**
- ✅ **思考链与快照阶梯控制** — 在 `search/smart.py` 中，对 `smart_search` 接口的思考链返回结构根据 `output_format` 实施了阶梯式控制：
  - **`clean` 模式**：彻底移除中间思考链与结果快照，仅输出最终深度思考答案，提升 Token 效率。
  - **`compact` 模式（默认）**：保留每轮的 thoughts 与 tool calls 轨迹，但彻底屏蔽具体返回的大段文本内容快照（将其统计化为 `"(已精简，共命中 X 条记录)"`），既防爆上下文又保留了可观测性。
  - **`raw` 模式**：保留原始思考链和包含前 200 字正文的结果快照，用作详细调试。

---

### v1.12.0 变更详记

**异步化重构（architecture-refactor Phase 1）：**
- ✅ **clients.py 全面异步化** — `RemoteChatClient`/`RemoteEmbeddingFunction`/`RemoteReranker` 的 HTTP 调用从 `requests` 改为 `httpx.AsyncClient`，方法签名改为 `async def`。
- ✅ **knowledge_base.py 异步化** — `smart_search`/`_tavily_search`/`_zhihu_search`/`_zhida`/`extract_outline`/`analyze_style`/`compare_styles` 改为 `async def`。
- ✅ **mcp_tools.py 异步适配** — 异步工具直接 `await`，同步工具保持 `asyncio.to_thread` 包装，新增 `_safe_tool_async` 包装器。
- ✅ **smart_search 并行优化** — 内部 `ThreadPoolExecutor` 替换为 `asyncio.gather`，解决多并发超时问题。
- ✅ **extraction 模块异步化** — outlines/entities/plot_patterns/writing_templates/scene_patterns/style/humor 中 LLM 调用全部改为 async。

**后台 Worker 进程（architecture-refactor Phase 2.5）：**
- ✅ **`worker.py`** — 文件系统任务队列消费者，支持 extract_outline_batch / knowledge_cleanup / generate_sample。
- ✅ **长时任务迁移** — 批量章纲提取、知识自动整理（`knowledge_cleanup`）以及任务管理（`manage_task`）均已全部迁移至 Worker 后台处理，避免主进程中的事件循环锁冲突与内存泄露。

**Agent 知识层（architecture-refactor Phase 2）：**
- ✅ **`core/knowledge_store.py`** — 引入 `KnowledgeStore` 类（管理 `agent_knowledge` ChromaDB 集合）。
- ✅ **`add_knowledge` 工具** — 支持写入研究成果、套路分析与写作风格，自动整理模块已适配 Worker 机制。
- ✅ **`stats` 与 `search` 支持 `scope` 检索** — 覆盖 novels / outlines / agent_knowledge 范围。
- ✅ **智能搜索自动存入洞察** — `smart_search` 自动在搜索完成后存储洞察结果，复用历史研究。

**素材范例生成（architecture-refactor Phase 3）：**
- ✅ **`creation/sample_generator.py`** — `SampleGenerator` 类，支持 minimal/light/moderate 三级重写。
- ✅ **`generate_sample` MCP 工具** — 指定章节最小限度重写，替换人名地名保留原文骨架。

**死代码清理（architecture-refactor Phase 4）：**
- 🗑️ **删除** `api_clients.py`（旧版客户端）、`batch_ingest.py`、`run_ingest.py`。
- 🧹 **config.py** — 移除 `TAVILY_API_KEY`、`XFYUN_*` 系列未使用变量。
- 🔧 **writing_templates.py** — 修复过时的 XFYUN 引用。
- 🗑️ **移除低命中率工具代码** — `search_entities`/`search_entities_semantic`/`search_knowledge`/`get_entity_relations` 方法及注册。

**工具返回过滤（architecture-refactor）：**
- ✅ **输出格式化** — 搜索类工具全面支持 `output_format` 参数（`compact`/`clean`/`raw`），支持在 compact 模式下裁剪冗余元数据，显著降低传输体积和 LLM 调用的 Context 损耗。

**代码职责拆分（code-split-refactor Phase 1 + Phase 2）：**
- ✅ **knowledge_base.py 门面化** — 从 1866 行缩减至 711 行（-62%），提取 7 个子模块：
  - `search/smart.py` — SmartSearchEngine（智能搜索，~456 行）
  - `search/external.py` — ExternalSearch（知乎/Tavily，~141 行）
  - `core/novel_reader.py` — NovelReader（章节读取，~136 行）
  - `core/outline_manager.py` — OutlineManager（章纲管理，~359 行）
  - `core/task_manager.py` — TaskManager（异步任务，~139 行）
  - `utils/novel_resolver.py` — 统一小说模糊匹配
  - `utils/chinese_numbers.py` — 中文数字转换
- ✅ **mcp_tools.py 分组** — 从 518 行缩减至 72 行（-86%），拆分为 7 个工具分组文件：
  - `api/tools/browse.py` — stats / read_chapter / get_chapter_edges
  - `api/tools/search.py` — search / smart_search
  - `api/tools/outline.py` — save_outline / get_outline / extract_outline
  - `api/tools/analysis.py` — style_analysis
  - `api/tools/knowledge.py` — add_knowledge
  - `api/tools/creation.py` — generate_sample
  - `api/tools/task.py` — manage_task
- ✅ **工具精简** — 从 17 个减至 12 个（移除 list_novels / search_knowledge / search_entities / get_entity_relations / search_outlines，功能合并到 stats 和 search 的 scope 参数）。

**SSE/HTTP 双协议支持：**
- ✅ **`MCP_TRANSPORT=http` 双协议模式** — 同一端口同时支持 SSE (`/sse`, `/messages`) 和 Streamable HTTP (`/mcp`)。
- ✅ **`_build_app()` 统一构建** — 支持 `http` / `sse` / `streamable-http` 三种模式。
- ✅ **Starlette lifespan 兼容** — 双协议模式通过自定义 `_dual_lifespan` 初始化 StreamableHTTP session manager。
- ✅ **OAuth 路由注入** — 所有协议模式均支持 OAuth 2.0 PKCE 路由注入。

**修复：**
- 🔧 **style.py `_analyze_perspective` 异步修复** — 方法使用了 `await` 但未声明为 `async def`，已修正。
- 🔧 **requirements.txt** — `fastmcp>=0.1.0` → `fastmcp>=2.14.0`（服务器实际运行 3.2.4）。

---

### v1.11.2 变更详记

**新增功能：**
- ✅ **`get_outline` 全量模式** — `chapter="full"` 返回全书全量章纲文本（按章节号排序串联）。
- ✅ **`manage_task`** — 合并 `get_task_status` + `cancel_task`，通过 `action` 参数区分（"status"/"cancel"）。

**工具删除：**
- ❌ **`delete_outline`** — 移除，简化章纲工具链。

**修复与优化：**
- 🔧 **`extract_outline_batch` 残留引用清理** — mcp_tools.py + MAINTENANCE.md 中全部更新为 `extract_outline`。
- 🔧 **`get_outline` 类型扩展** — chapter 参数支持 `int | str`，兼容全量文本和单章查询。

---

### v1.11.1 变更详记

**新增功能：**
- ✅ **`cancel_task`** — 取消正在运行的异步任务，任务会在当前章节完成后停止。
- ✅ **批量提取上限** — `extract_outline` 批量模式限制 20 章，防止资源滥用。

**修复与优化：**
- 🔧 **取消支持** — `extract_batch` 新增 `is_cancelled` 回调，每章提取前检查取消标志。
- 🔧 **任务状态扩展** — 新增 "cancelled" 状态，与 "completed"/"error" 并列。
- 🔧 **防呆设计（参数校验）** — 各工具新增入参合法性检查：
  - `extract_outline`: chapter ≤ 0、end_chapter < 0、end < start 均报错
  - `read_chapter` / `get_chapter_edges` / `delete_outline`: chapter ≤ 0 报错
  - `get_chapter_edges`: paragraphs ≤ 0 报错
- 🔧 **`save_outline` 重复检查** — 默认不覆盖已存在章纲（返回 `skipped`），设 `overwrite=True` 可强制覆盖。
- 🔧 **`extract_outline` 内部调用 `overwrite=True`** — 重复提取同一章自动覆盖更新。

---

### v1.11.0 变更详记

**新增功能：**
- ✅ **章纲自动提取** — `extract_outline` 单章/批量一体（`end_chapter` 参数控制），服务端 LLM 串行提取。
- ✅ **`extraction/outlines.py` 独立模块** — ChapterOutlineExtractor，封装读取→LLM提取→存储→封存全流程。
- ✅ **`get_task_status`** — 异步任务进度查询，支持批量提取和旧版 start_async_extraction。
- ✅ **`get_outline` 优化** — chapter=0 时只返回章节号和类型列表，不返回正文内容。

**架构变化：**
- 新增文件：`webnovel_kb/extraction/outlines.py`。
- `knowledge_base.py`：集成 ChapterOutlineExtractor，get_outline 返回格式优化。
- `mcp_tools.py`：extract_outline 合并单章/批量，工具总数 15→17。

---

### v1.10.0 变更详记

**新增功能：**
- ✅ **章纲 MCP 工具（4个）** — `save_outline`(批量/单条)、`get_outline`、`search_outlines`、`delete_outline`。
- ✅ **`stats` 补充章节信息** — 新增 `first_chapter`/`last_chapter` 字段，`chapter_count` 准确统计。
- ✅ **移除 `list_chapters` 工具** — 章节信息整合到 `stats`，精简工具数。
- ✅ **`read_chapter` 支持多格式章标题** — 兼容 `第N章`/`第N回`/`章N`/`N、` 等格式。

**修复与优化：**
- 🔧 **永夜君王章节修复** — 1349章全部标注为 `第N章` 标准格式，源文件 `fix_yongye.py` 留存。
- 🔧 **ChromaDB 修复** — WAL segment 损坏，从备份恢复（旧库 105G 已清理）。
- 🧹 **磁盘空间清理** — 回收 ~190G（chroma_db.broken 105G + BrowserMetrics 84G + .npm 644M）。
- 🧹 **BrowserMetrics 根治** — 软链接到 `/dev/null`，防止 Chromium snap 自动化测试再泄漏。

---

### v1.9.2 变更详记

**smart_search 外部搜索升级：**
- ✅ **新增 `zhihu_search` 内部工具** — 知乎站内搜索，获取网文作者实战经验、写作技巧讨论。
- ✅ **`web_search` 底层替换** — Tavily API → 知乎全网搜索 API（`global_search`），结果质量更高。
- ✅ **system prompt 优化** — 明确区分内部知识库工具与外部知识工具，引导模型合理并行调用。
- 新增配置：`ZHIHU_ACCESS_SECRET`、`ZHIHU_SEARCH_URL`、`ZHIHU_GLOBAL_SEARCH_URL`。

**smart_search 内部工具从 4 个增至 5 个：**
- `search_text` (内部): 知识库全文
- `search_patterns` (内部): 情节模式
- `search_entities` (内部): 实体数据
- `web_search` (外部): 知乎全网搜索
- `zhihu_search` (外部): 知乎站内搜索
