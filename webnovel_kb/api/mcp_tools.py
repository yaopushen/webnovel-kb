"""MCP tool definitions for webnovel_kb."""
from typing import Optional, List
from mcp.server.fastmcp import FastMCP

from webnovel_kb.utils.logging_config import get_logger
from webnovel_kb.utils.exceptions import WebNovelError

logger = get_logger("api.mcp_tools")


class MCPTools:
    """MCP 工具定义类。"""

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
        def ingest_novel(file_path: str, title: str, author: str, genre: str) -> dict:
            """导入一本网文到知识库。file_path为小说txt文件路径，title为书名，author为作者，genre为类型（如奇幻/悬疑/赛博朋克）"""
            return safe("ingest_novel", kb.ingest_novel, file_path, title, author, genre)

        @mcp.tool()
        def search(query: str, mode: str = "hybrid", n_results: int = 10,
                   novel_filter: str = "", genre_filter: str = "",
                   chapter_filter: str = "", alpha: float = 0.6,
                   use_rerank: bool = False,
                   output_format: str = "compact", max_content_length: int = 0,
                   dedupe: bool = True) -> list:
            """统一检索接口。mode可选：semantic(语义), bm25(关键词), hybrid(混合), rerank(精排)。alpha控制hybrid模式语义权重。use_rerank对hybrid模式启用精排。output_format: raw/compact/clean(默认compact)。max_content_length每条字数上限(0=不限制)。dedupe去重(默认True)。"""
            return safe("unified_search", kb.unified_search.search,
                query, mode=mode, n_results=n_results,
                novel_filter=novel_filter or None,
                genre_filter=genre_filter or None,
                chapter_filter=chapter_filter or None,
                alpha=alpha,
                use_rerank=use_rerank,
                output_format=output_format,
                max_content_length=max_content_length,
                dedupe=dedupe
            )

        @mcp.tool()
        def search_knowledge(query: str = "", knowledge_type: str = "plot_patterns",
                             n_results: int = 10, use_semantic: bool = True,
                             type_filter: str = "", source_novel: str = "",
                             output_format: str = "compact", max_content_length: int = 0,
                             dedupe: bool = True) -> list:
            """统一知识搜索。knowledge_type: plot_patterns(情节模式)或writing_templates(写法模板)。use_semantic=True用语义搜索，False用关键字过滤。output_format: raw/compact/clean(默认compact)。max_content_length每条字数上限(0=不限制)。dedupe去重(默认True)。"""
            return safe("search_knowledge", kb.search_knowledge,
                query, knowledge_type, n_results, use_semantic,
                type_filter=type_filter or None,
                source_novel=source_novel or None,
                output_format=output_format,
                max_content_length=max_content_length,
                dedupe=dedupe
            )

        @mcp.tool()
        def analyze_style(novel_title: str) -> dict:
            """分析指定小说的写作风格，包括分段节奏变化（开篇/发展/高潮/收尾的句长和对话比）、原文幽默场景提取、叙事视角、章节钩子密度等"""
            return safe("analyze_style", kb.analyze_style, novel_title)

        @mcp.tool()
        def compare_styles(novel_titles: str) -> dict:
            """对比多本小说的写作风格。novel_titles为逗号分隔的书名列表，如'隐秘死角,没钱修什么仙'"""
            if isinstance(novel_titles, str):
                titles = [t.strip() for t in novel_titles.split(",") if t.strip()]
            else:
                titles = novel_titles
            return safe("compare_styles", kb.compare_styles, titles)

        @mcp.tool()
        def novel_stats(novel_title: str) -> dict:
            """获取单本小说的细粒度统计：章节数、平均分块长度、对话占比、章节列表等"""
            return safe("novel_stats", kb.novel_stats, novel_title)

        @mcp.tool()
        def list_novels() -> list[dict]:
            """列出知识库中所有已导入的小说"""
            return safe("list_novels", kb.list_novels)

        @mcp.tool()
        def get_stats() -> dict:
            """获取知识库统计信息：小说数量、分块数、实体数、关系数、情节模式数等"""
            return safe("get_stats", kb.get_stats)

        @mcp.tool()
        def search_entities(query: str, n_results: int = 10,
                            entity_type: str = "", source_novel: str = "",
                            output_format: str = "compact", max_content_length: int = 0,
                            dedupe: bool = True) -> list:
            """语义搜索实体。用自然语言描述查找相关角色、地点、组织等，如'反派角色'、'修炼功法'。output_format: raw/compact/clean(默认compact)。"""
            raw = safe("search_entities_semantic", kb.search_entities_semantic,
                query, n_results,
                entity_type=entity_type or None,
                source_novel=source_novel or None
            )
            if isinstance(raw, list) and not (len(raw) == 1 and isinstance(raw[0], dict) and "error" in raw[0]):
                return kb._format_search_results(raw, output_format, max_content_length, dedupe)
            return raw

        @mcp.tool()
        def get_entity_relations(entity_name: str, source_novel: str = "") -> list[dict]:
            """查询实体的所有关系。返回该实体的入边和出边关系"""
            return safe("get_entity_relations", kb.get_entity_relations, entity_name, source_novel or None)

        @mcp.tool()
        def extract(novel_title: str, extract_type: str = "plot_patterns",
                    max_chunks: int = 20, cross_chunk: bool = False,
                    async_mode: bool = False) -> dict:
            """自动提取小说中的结构化知识。extract_type可选：entities(实体)/plot_patterns(情节模式)/writing_templates(写法模板)/scene_patterns(场景模式)/all(全部)。cross_chunk=True启用跨章节深度提取（较慢但更全面）。async_mode=True时后台运行返回task_id"""
            if async_mode:
                return safe("start_async_extraction", kb.start_async_extraction, novel_title, max_chunks, extract_type)
            return safe("extract", kb.extract, novel_title, extract_type, max_chunks, cross_chunk)

        @mcp.tool()
        def get_task_status(task_id: str) -> dict:
            """查询异步任务状态和结果。task_id从extract(async_mode=True)返回"""
            return safe("get_task_status", kb.get_task_status, task_id)

        @mcp.tool()
        def read_chapter(novel_title: str, chapter: int = 1) -> dict:
            """读取小说的指定章节完整正文。novel_title为书名，chapter为章节号(1-based，第一章=1)。返回完整的章节正文内容。"""
            return safe("read_chapter", kb.read_chapter, novel_title, chapter)

        @mcp.tool()
        def list_chapters(novel_title: str) -> dict:
            """列出小说的所有章节标题、序号和分块范围。用于章节导航，获取章节列表后可配合read_chapter使用。"""
            return safe("list_chapters", kb.list_chapters, novel_title)
