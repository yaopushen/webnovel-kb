"""WebNovel KB MCP tools documentation guide module."""

TOOLS_GUIDE_MD = """# WebNovel KB MCP 工具使用手册与指南

本服务共提供 12 个核心工具，涵盖浏览、搜索、章纲管理、Agent 知识库、创作辅助、风格分析及任务追踪等各个方面。

---

## 1. stats
- **功能**：获取知识库的统计信息，支持查看全局状态、已导入小说列表、已入库的 Agent 知识以及单本小说详情。
- **参数**：
  - `scope` (str, 默认 `"global"`): 统计范围：
    - `"global"`: 全局库统计（如小说数、分块数、ChromaDB 准备状态等）
    - `"novels"`: 列出所有已导入小说的列表与字数统计（等价于原 `list_novels`）
    - `"knowledge"`: 列出所有已入库的 Agent 知识条目
    - `"guide"`: 返回本工具使用手册
    - `书名` (或作为 scope 传入): 对指定小说进行专门的数据统计
  - `novel_title` (str, 可选): 书名（支持模糊匹配）。当 `scope` 设为书名时可省略。
- **示例调用**：`stats(scope="novels")`

---

## 2. read_chapter
- **功能**：读取指定小说的某一章节的完整正文内容。
- **参数**：
  - `novel_title` (str, 必填): 书名（支持模糊匹配）。
  - `chapter` (int, 默认 `1`): 章节号 (1-based)。
- **示例调用**：`read_chapter(novel_title="没钱修什么仙", chapter=12)`

---

## 3. get_chapter_edges
- **功能**：提取小说某一章节的开头前几段和结尾后几段。常用于学习和模仿小说的章前悬念铺垫与章末钩子写法。
- **参数**：
  - `novel_title` (str, 必填): 书名（支持模糊匹配）。
  - `chapter` (int, 默认 `1`): 章节号 (1-based)。
  - `paragraphs` (int, 默认 `2`): 开头和结尾各提取的段落数量。
- **示例调用**：`get_chapter_edges(novel_title="宿命之环", chapter=5, paragraphs=3)`

---

## 4. search
- **功能**：在小说原文、已提取章纲或 Agent 知识库中执行全文多模式搜索。
- **参数**：
  - `query` (str, 必填): 搜索关键词或自然语言描述（如 "主角金手指觉醒"）。
  - `scope` (str, 默认 `"chunks"`): 检索范围：
    - `"chunks"`: 搜索小说原文分块
    - `"outlines"`: 搜索保存的章纲
    - `"agent_knowledge"`: 搜索 Agent 知识层
    - `"all"`: 混合以上三个来源，按相关度融合
  - `mode` (str, 默认 `"hybrid"`): 检索模式：
    - `"hybrid"`: 语义向量+关键词混合检索，最推荐
    - `"semantic"`: 纯向量语义检索，适合概念模糊的查找
    - `"bm25"`: 纯文本关键字匹配，适合精确术语
    - `"rerank"`: 检索后启用 Cross-Encoder 模型精排（需配置模型）
  - `novel_filter` (str, 可选): 限定只搜索特定书名（支持模糊匹配）。
  - `genre_filter` (str, 可选): 限定类型（如 "修仙", "悬疑", "科幻" 等）。
  - `n_results` (int, 默认 `10`): 返回的结果数。
  - `output_format` (str, 默认 `"compact"`): 输出格式（`"compact"` 仅返回纯内容，`"clean"` 返回纯文本，`"raw"` 返回包含元数据的完整 JSON）。
- **示例调用**：`search(query="突破练气期", scope="all", mode="hybrid")`

---

## 5. smart_search
- **功能**：智能搜索。由底层大语言模型驱动，自主规划搜索策略，可进行多轮的内部工具调用，自动结合内部知识库、全网搜索引擎和知乎网文圈讨论，最终深度思考输出高度提炼的分析答案。
- **参数**：
  - `query` (str, 必填): 口语化、复杂或多意图的自然语言问题。
  - `novel_filter` (str, 可选): 限定只搜索某本小说。
  - `genre_filter` (str, 可选): 限定检索类型。
  - `n_results` (int, 默认 `5`): 每轮工具调用的返回数量。
  - `output_format` (str, 默认 `"compact"`): 
    - `"clean"`: 仅返回最终的深度思考分析答案，彻底隐藏中间步骤（极省 Token）。
    - `"compact"`: 返回答案，并附带不含正文片段的思考链路径（ thoughts + tool calls 动作概要）。
    - `"raw"`: 包含详细的思考过程和每步获取的前200字结果快照。
- **示例调用**：`smart_search(query="正面配角发现主角战力飙升时的处理描写", output_format="compact")`

---

## 6. save_outline
- **功能**：保存或批量保存已分析出的章节大纲到知识库中。
- **参数**：
  - `novel_title` (str, 必填): 书名。
  - `outlines` (dict 或 list, 必填): 单个字典或多个字典的列表。每个字典需包含：
    - `chapter` (int, 必填): 章节号
    - `content` (str, 必填): 章纲文本内容
    - `outline_type` (str, 可选): "章纲"/"细纲"/"简纲"
    - `tags` (list[str], 可选): 标签，如 `["伏笔", "打脸"]`
  - `overwrite` (bool, 默认 `False`): 若已存在该章纲，是否强制覆盖。

---

## 7. get_outline
- **功能**：提取已保存小说的章纲信息。
- **参数**：
  - `novel_title` (str, 必填): 书名。
  - `chapter` (int 或 str, 默认 `0`): 
    - `0` (默认): 列出全书所有已有的章纲目录（含章节、类型、标签），不含正文。
    - `>0`: 返回该章节对应的详细章纲正文。
    - `"full"`: 拼接并返回整本书的所有章纲正文全文。

---

## 8. extract_outline
- **功能**：一键智能章纲提取。读取指定章节的正文，调用 LLM 深度分析生成该章的精细叙事性大纲，并自动将其存入知识库。
- **参数**：
  - `novel_title` (str, 必填): 书名。
  - `chapter` (int, 必填): 起始章节号。
  - `end_chapter` (int, 默认 `0`): 结束章节号。为 `0` 时仅分析单章；`>0` 时以异步任务在后台批量串行提取（限制最大批量 20 章），并返回 `task_id`。
- **示例调用**：`extract_outline(novel_title="隐秘死角", chapter=1, end_chapter=10)`

---

## 9. add_knowledge
- **功能**：将行业调研、套路公式、风格指南等知识存入 Agent 知识层，并可选择是否对比分析它与库中已有知识的关系（冲突、重复、补充等）。
- **参数**：
  - `content` (str, 必填): 知识正文。
  - `title` (str, 必填): 知识条目标题。
  - `category` (str, 默认 `"research"`): 类别，如 `"market_analysis"`, `"style_guide"`, `"writing_technique"`, `"research"` 等。
  - `tags` (list, 可选): 标签列表。
  - `source` (str, 可选): 来源描述。
  - `analyze` (bool, 默认 `True`): 是否启动自动分析它与旧知识的关联。

---

## 10. generate_draft
- **功能**：初稿生成（模仿模式或创造模式）。
  - **⚠️调用者过度自信警示**：实机测试表明，写作中工具（创造模式）输出的质量通常比调用者（Agent）认为的要更高，请充分信任工具生成的初稿内容。
  - 模仿模式：针对指定的经典章节进行中度改写，生成可直接参考的改写初稿。
  - 创造模式：调用 mimo-v2.5-pro 思考链模型，加载硬编码的写作范例与指定章节正文作为双重风格种子，基于传入的大纲提示创作全新初稿并完整返回思考链。
- **参数**：
  - `novel_title` (str, 必填): 书名。
  - `chapter` (int, 必填): 章节号。在模仿模式下为待改写章节，在创造模式下为风格种子章节。
  - `mode` (str, 默认 `"imitate"`): 运行模式：
    - `"imitate"`: 模仿模式（基于中度改写算法）。
    - `"create"`: 创造模式（使用 mimo-v2.5-pro 思考链模型）。
  - `prompt` (str, 可选): 写作大纲或提示词（在创造模式下必填）。**注：调用此模式 of Agent 应当先参考该书的素材章纲提取出的信息作为大纲填入。**
  - `custom_instructions` (str, 可选): 自定义写作/改写指令。**注：创造模式下由于使用了高度精调的专属文风模板，此参数将被完全忽略。**

---

## 11. style_analysis
- **功能**：分析单本小说的句长分布、对话占比、幽默段落、核心视角以及不同情节部分的叙事节奏，或输入多个书名进行风格特征横向比对。
- **参数**：
  - `novel_titles` (str, 必填): 单个书名或以英文逗号分隔的多本书名（如 `"隐秘死角,宿命之环"`）。
- **注意**：首次分析可能需要 30-120 秒，分析结果会自动缓存。

---

## 12. manage_task
- **功能**：管理后台长时异步任务（如批量提取章纲任务）。
- **参数**：
  - `task_id` (str, 必填): 任务 ID。
  - `action` (str, 默认 `"status"`): `"status"` (查询当前进度与状态，包含 completed/running/cancelled/error)，`"cancel"` (尝试取消任务，将在当前章节处理完毕后异步终止)。
"""
