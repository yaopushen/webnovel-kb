"""MCP tool definitions for webnovel_kb — web novel knowledge base.

=== TYPICAL WORKFLOWS (新手 agent 请先读这里) ===

读小说全文:
  1. list_novels → 获取所有书名
  2. list_chapters(novel_title) → 获取章节列表
  3. read_chapter(novel_title, chapter=N) → 循环读取每一章

查剧情/手法:
  1. search(query, mode="hybrid") → 全文混合检索
  2. smart_search(query) → 智能分解搜索（适合模糊/复杂查询）
  3. search_knowledge(query, knowledge_type="plot_patterns") → 搜情节模式
  4. search_entities(query) → 找角色/功法/地点等实体
  5. get_entity_relations(entity_name) → 查某个实体的所有关系

分析风格:
  1. stats(novel_title) → 基础统计（留空获取全局统计）
  2. style_analysis(novel_titles) → 风格分析（单本）或风格对比（逗号分隔多本）

=== 重要约定 ===
- 所有书名参数 (novel_title, novel_filter, source_novel) 支持模糊匹配。
  例如 "没钱修什么仙" 能匹配到 "没钱修什么仙？"（全角问号会被自动处理）。
- 如果 read_chapter 返回 error，请先用 list_chapters 确认章节范围。
- 如果搜索无结果，尝试缩短 query 或换用 mode="bm25"（关键词匹配）。
"""
import asyncio
from functools import partial
from typing import Optional, List
from mcp.server.fastmcp import FastMCP

from webnovel_kb.utils.logging_config import get_logger
from webnovel_kb.utils.exceptions import WebNovelError

logger = get_logger("api.mcp_tools")


class MCPTools:

    def __init__(self, mcp: FastMCP, kb):
        self.mcp = mcp
        self.kb = kb
        self._register_tools()

    def _safe_tool(self, name: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except WebNovelError as e:
            logger.error(f"MCP tool '{name}' failed: {e}", exc_info=True)
            return {"error": str(e), "detail": e.detail, "tool": name, "type": type(e).__name__}
        except Exception as e:
            logger.error(f"MCP tool '{name}' failed: {e}", exc_info=True)
            return {"error": str(e), "tool": name}

    def _register_tools(self):
        kb = self.kb
        mcp = self.mcp
        safe = self._safe_tool

        @mcp.tool()
        async def stats(novel_title: str = "") -> dict:
            """获取知识库统计信息——全局或单本小说。

参数:
  novel_title — 书名（支持模糊匹配）。留空则获取全局统计。

返回 (novel_title 留空时):
  - total_novels, total_chunks, total_patterns, total_entities, total_relationships
  - bm25_ready, optimized_search, query_cache

返回 (指定 novel_title 时):
  - title, author, genre, word_count
  - chunk_count / chunks_indexed: 总分块数与已索引数
  - has_style_analysis: 是否已完成风格分析
  - entities_extracted: 已提取的实体数量
  - patterns_extracted: 已提取的情节模式数量

关联工具: 若 has_style_analysis=false，调用 style_analysis 生成分析。
"""
            if novel_title and novel_title.strip():
                return await asyncio.to_thread(safe, "novel_stats", kb.novel_stats, novel_title)
            return await asyncio.to_thread(safe, "get_stats", kb.get_stats)

        @mcp.tool()
        async def list_novels() -> list[dict]:
            """列出知识库中所有已导入的小说。

用途: 获取所有可用小说的书名、作者、类型和字数。后续会用书名调用其他工具。
每本书返回:
  - title: 书名 (用于 read_chapter/list_chapters/stats 等)
  - author: 作者
  - genre: 类型标签 (修仙/科幻/悬疑/奇幻/赛博朋克/克苏鲁/高武)
  - word_count: 总字数
  - chunk_count: 切分块数

建议: 在不清楚有哪些书、或需要精确书名时，先调此工具。
"""
            return await asyncio.to_thread(safe, "list_novels", kb.list_novels)

        @mcp.tool()
        async def list_chapters(novel_title: str) -> dict:
            """列出小说的所有章节标题、序号和分块范围。

参数:
  novel_title — 书名，支持模糊匹配。

返回:
  - novel: 精确书名
  - total_chapters: 章节总数
  - chapters[]: 每章 {number, title, first_chunk, last_chunk, chunk_count}

典型用法:
  1. 先调用此工具获取章节列表
  2. 再用 read_chapter(novel_title, chapter=N) 逐章读取正文
  3. 输出的 chunk_count 帮助预判章节长度
"""
            return await asyncio.to_thread(safe, "list_chapters", kb.list_chapters, novel_title)

        @mcp.tool()
        async def read_chapter(novel_title: str, chapter: int = 1) -> dict:
            """读取小说的指定章节完整正文。

参数:
  novel_title — 书名，支持模糊匹配。
  chapter — 章节号 (1-based, 第一章=1)。

返回:
  - novel: 精确书名
  - chapter_number: 章节号
  - chapter_title: 章节标题 (如 "第1章")
  - content: 完整正文内容
  - word_count: 本章字数
  - chunk_count: 本章占用分块数

错误时返回:
  - {"error": "...", "hint": "使用 list_chapters 查看可用章节"}

注意: 章节号超出范围时会返回 error，此时应调用 list_chapters 确认有效范围。
"""
            return await asyncio.to_thread(safe, "read_chapter", kb.read_chapter, novel_title, chapter)

        @mcp.tool()
        async def get_chapter_edges(novel_title: str, chapter: int = 1,
                              paragraphs: int = 2) -> dict:
            """提取章前章末段落——用于学习章节开头和结尾的写法。

参数:
  novel_title — 书名，支持模糊匹配。
  chapter — 章节号 (1-based, 第一章=1)。
  paragraphs — 章前/章末各提取几段，默认2段。

返回:
  - novel: 精确书名
  - chapter_number: 章节号
  - chapter_title: 章节标题
  - opening: 章前段落列表 (前N段)
  - closing: 章末段落列表 (后N段)
  - total_paragraphs: 本章总段落数
  - opening_word_count: 章前总字数
  - closing_word_count: 章末总字数

用途:
  - 学习不同类型的章节如何开头（悬念、动作、对话、描写等）
  - 学习章节结尾的钩子技巧（悬念、反转、情感铺垫等）
  - 批量提取20种类型的章前章末示例

注意: 返回的是纯文本段落，保留原文格式。
"""
            result = await asyncio.to_thread(safe, "read_chapter", kb.read_chapter, novel_title, chapter)
            if isinstance(result, dict) and "error" in result:
                return result

            content = result.get("content", "")
            if not content:
                return {"error": "章节内容为空", "novel": result.get("novel"), "chapter": chapter}

            all_paragraphs = [p.strip() for p in content.split("\n") if p.strip()]

            if not all_paragraphs:
                return {"error": "无法解析段落", "novel": result.get("novel"), "chapter": chapter}

            opening = all_paragraphs[:paragraphs]
            closing = all_paragraphs[-paragraphs:] if len(all_paragraphs) > paragraphs else all_paragraphs

            return {
                "novel": result.get("novel"),
                "chapter_number": chapter,
                "chapter_title": result.get("chapter_title", ""),
                "opening": opening,
                "closing": closing,
                "total_paragraphs": len(all_paragraphs),
                "opening_word_count": sum(len(p) for p in opening),
                "closing_word_count": sum(len(p) for p in closing)
            }

        @mcp.tool()
        async def search(query: str, mode: str = "hybrid", n_results: int = 10,
                   novel_filter: str = "", genre_filter: str = "",
                   chapter_filter: str = "", alpha: float = 0.6,
                   use_rerank: bool = False,
                   output_format: str = "compact", max_content_length: int = 0,
                   dedupe: bool = True) -> list:
            """全文统一检索——同时搜索所有已导入小说的文本内容。

参数:
  query — 搜索关键词或自然语言描述。例如: "主角获得金手指", "突破境界"。
  mode — 检索模式:
      "hybrid" (默认, 推荐): 语义+关键词混合，兼顾相关性与精确匹配
      "semantic": 纯语义搜索，适合模糊的概念查询
      "bm25": 纯关键词匹配，适合精确术语查找
      "rerank": 混合搜索后再用精排模型重排 (需要 LLM_RERANK_MODEL 已配置)
  n_results — 返回结果数量，默认 10。
  novel_filter — 限定只搜某本书。输入书名（支持模糊匹配），留空表示搜全部。
  genre_filter — 限定只搜某种类型。值: 修仙, 科幻, 悬疑, 奇幻, 赛博朋克, 克苏鲁, 高武。
  chapter_filter — 限定只搜某章节（按章节标题字符串匹配），通常不需要。
  alpha — hybrid 模式下语义搜索权重 (0~1)。越大越偏向语义，默认 0.6。
  use_rerank — 是否启用 Cross-encoder 精排 (需要 reranker 已配置)，默认 False。
  output_format — 输出格式:
      "compact" (默认): 精简格式 [书名 - 作者 [章节名]] 摘要...
      "clean": 纯文本
      "raw": 完整元数据
  max_content_length — 每条结果字数上限 (0=不限制)。
  dedupe — 是否去重 (默认 True)。

返回格式: [ "[书名 - 作者 [章节名]] 内容...", ... ]

如果无结果:
  - 尝试缩短 query 或去掉特殊字符
  - 换用 mode="bm25"
  - 检查 novel_filter 或 genre_filter 是否限制过严
"""
            resolved_novel = kb.resolve_novel_title(novel_filter) if novel_filter else None
            return await asyncio.to_thread(safe, "unified_search", kb.unified_search.search,
                query, mode=mode, n_results=n_results,
                novel_filter=resolved_novel,
                genre_filter=genre_filter or None,
                chapter_filter=chapter_filter or None,
                alpha=alpha,
                use_rerank=use_rerank,
                output_format=output_format,
                max_content_length=max_content_length,
                dedupe=dedupe
            )

        @mcp.tool()
        async def search_knowledge(query: str = "", knowledge_type: str = "plot_patterns",
                             n_results: int = 10, use_semantic: bool = True,
                             type_filter: str = "", source_novel: str = "",
                             output_format: str = "compact", max_content_length: int = 0,
                             dedupe: bool = True) -> list:
            """搜索已提取的结构化知识——情节模式或写法模板。

参数:
  query — 自然语言描述。如 "悬念设置手法", "金手指激活场景"。留空则不过滤。
  knowledge_type — 搜索目标:
      "plot_patterns" (默认): 情节模式——已被 AI 提取并归类的叙事套路
      "writing_templates": 写法模板——特定场景的写法结构
  n_results — 返回条数，默认 10。
  use_semantic — True=语义搜索 (推荐), False=关键词过滤。
  type_filter — 限定情节模式类型。常用值:
      悬念链, 跨距伏笔, 反转铺垫, 情感爆发点, 世界观展开, 力量体系引入,
      角色弧光, 高潮设计, 节奏控制, 对比映衬, 身份揭示
  source_novel — 限定出自某书。书名支持模糊匹配。
  output_format: raw/compact/clean，默认 compact。
  max_content_length — 每条字数上限 (0=不限制)。
  dedupe — 是否去重 (默认 True)。

前提条件: 需要先通过 extract(novel_title, extract_type="plot_patterns") 提取过知识。
           如果没有提取过，结果可能为空。
"""
            resolved_novel = kb.resolve_novel_title(source_novel) if source_novel else None
            return await asyncio.to_thread(safe, "search_knowledge", kb.search_knowledge,
                query, knowledge_type, n_results, use_semantic,
                type_filter=type_filter or None,
                source_novel=resolved_novel,
                output_format=output_format,
                max_content_length=max_content_length,
                dedupe=dedupe
            )

        @mcp.tool()
        async def search_entities(query: str, n_results: int = 10,
                            entity_type: str = "", source_novel: str = "",
                            output_format: str = "compact", max_content_length: int = 0,
                            dedupe: bool = True) -> list:
            """语义搜索实体——用自然语言描述查找角色、功法、地点、组织等。

参数:
  query — 自然语言描述。如 "反派角色", "修炼功法", "炼丹", "宗门"。
  n_results — 返回条数，默认 10。
  entity_type — 实体类型过滤。有效值:
      "角色", "功法", "组织", "地点", "物品", "概念", "事件", "种族"
      留空表示不限制类型。
  source_novel — 限定出自某书。书名支持模糊匹配。
  output_format: raw/compact/clean，默认 compact。
  max_content_length — 每条字数上限 (0=不限制)。
  dedupe — 是否去重 (默认 True)。

返回: 每条实体包含 name, entity_type, description, source_novel, semantic_score(0~1) 等。

前提条件: 需要先通过 extract(novel_title, extract_type="entities") 提取过实体。
"""
            resolved_novel = kb.resolve_novel_title(source_novel) if source_novel else None
            raw = await asyncio.to_thread(safe, "search_entities_semantic", kb.search_entities_semantic,
                query, n_results,
                entity_type=entity_type or None,
                source_novel=resolved_novel
            )
            if isinstance(raw, list) and not (len(raw) == 1 and isinstance(raw[0], dict) and "error" in raw[0]):
                return await asyncio.to_thread(kb._format_search_results, raw, output_format, max_content_length, dedupe)
            return raw

        @mcp.tool()
        async def get_entity_relations(entity_name: str, source_novel: str = "") -> list[dict]:
            """查询某个实体的所有关系——它与其他角色/组织/功法等的关联。

参数:
  entity_name — 实体名称 (需精确)。可通过 search_entities 获取名称。
  source_novel — 限定出自某书 (支持模糊匹配)，留空则查所有书。

返回: 每条关系包含:
  - source / target: 关系的两端实体名
  - rel_type: 关系类型 (如 "同伴->命运共同体", "从属->超越")
  - description: 关系描述
  - source_novel: 出自哪本书
  - evolution: 关系演进说明

典型用法:
  1. search_entities("主角") → 找到主角实体名
  2. get_entity_relations("主角名") → 查看主角的所有关系网
"""
            resolved_novel = kb.resolve_novel_title(source_novel) if source_novel else None
            return await asyncio.to_thread(safe, "get_entity_relations", kb.get_entity_relations, entity_name, resolved_novel)

        @mcp.tool()
        async def style_analysis(novel_titles: str) -> dict:
            """分析单本或多本小说的写作风格。

参数:
  novel_titles — 书名或逗号分隔的书名列表（支持模糊匹配）。如:
      单本: "没钱修什么仙？"
      对比: "没钱修什么仙？,隐秘死角"

返回 (单本):
  - avg_sentence_len: 全局平均句长
  - dialogue_ratio: 对话占比 (0~1)
  - narrative_perspective: 叙事视角 (如 "第三人称限知")
  - section_breakdown[]: 分段节奏分析 {开篇, 发展, 高潮, 收尾}
  - humor_scenes[]: 提取的幽默场景
  - sample_passages[]: 代表性段落

返回 (多本对比):
  - novels: 对比的书名列表
  - comparison: 每本书的完整风格 profile
  - summary: 风格对比摘要 (每本书一句)

注意: 首次分析较慢（约30-120秒），结果会缓存。如果某本书尚未做过风格分析，会自动触发。
"""
            if isinstance(novel_titles, str):
                titles = [t.strip() for t in novel_titles.split(",") if t.strip()]
            else:
                titles = novel_titles

            if len(titles) == 1:
                return await asyncio.to_thread(safe, "analyze_style", kb.analyze_style, titles[0])
            return await asyncio.to_thread(safe, "compare_styles", kb.compare_styles, titles)

        @mcp.tool()
        async def smart_search(query: str, n_results: int = 5,
                         novel_filter: str = "", genre_filter: str = "",
                         output_format: str = "compact") -> dict:
            """智能搜索——小米模型函数调用模式：多轮对话自主搜索知识库后深度思考回答。

适用场景: 模糊、复杂、多意图的查询，例如:
  - "正面配角发现主角战力飙升的处理"
  - "主角金手指激活时的反应描写"
  - "反派被主角打脸时的心理描写"

与普通 search 的区别:
  - search: 直接用原始 query 搜索，命中率依赖关键词选择
  - smart_search: 模型自主调用搜索工具，多轮迭代，深度思考后给出分析结果

参数:
  query — 自然语言描述，可以模糊、口语化。
  n_results — 每次工具搜索返回的结果数，默认 5。
  novel_filter — 限定只搜某本书。留空搜全部。
  genre_filter — 限定类型。留空不限制。
  output_format — 输出格式: "compact"/"clean"/"raw"。

返回:
  - query: 原始查询
  - 思考链: 每轮 {round, 思考, 调用} 的列表，调用为工具名+关键词摘要
  - 结果: 模型深度思考后的最终分析答案

注意: 此工具调用小米 MiMo 模型进行函数调用模式搜索，响应时间约 30-120 秒。
"""
            resolved_novel = kb.resolve_novel_title(novel_filter) if novel_filter else None
            return await asyncio.to_thread(safe, "smart_search", kb.smart_search,
                query, n_results=n_results,
                novel_filter=resolved_novel,
                genre_filter=genre_filter or None,
                output_format=output_format
            )
