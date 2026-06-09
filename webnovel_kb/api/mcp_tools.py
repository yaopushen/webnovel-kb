"""MCP tool definitions — 精简版，工具注册委托给 api/tools/ 各分组。

=== TYPICAL WORKFLOWS (新手 agent 请先读这里) ===

读小说全文:
  1. stats(scope="novels") → 获取所有书名
  2. stats(novel_title) → 获取章节数量（chapter_count）
  3. read_chapter(novel_title, chapter=N) → 循环读取每一章

查剧情/手法:
  1. search(query, mode="hybrid") → 全文混合检索
  2. smart_search(query) → 智能分解搜索（适合模糊/复杂查询）

分析风格:
  1. stats(novel_title) → 基础统计（留空获取全局统计）
  2. style_analysis(novel_titles) → 风格分析（单本）或风格对比（逗号分隔多本）

章纲写作（IDE 创作 → MCP 存储）:
  1. read_chapter(novel_title, chapter=N) → 读取章节正文
  2. save_outline(novel_title, outlines=[{chapter, content, ...}]) → IDE 产出章纲后存入
  3. get_outline(novel_title) → 查看已有章纲列表
  4. search(query, scope="outlines") → 语义搜索参考章纲

章纲自动化（服务端 LLM 提取 → 自动封存）:
  1. extract_outline(novel_title, chapter=N) → 单章提取
  2. extract_outline(novel_title, chapter=1, end_chapter=100) → 批量异步提取
  3. manage_task(task_id) → 查询进度

=== 重要约定 ===
- 所有书名参数 (novel_title, novel_filter, source_novel) 支持模糊匹配。
- read_chapter 的 chapter 参数超出范围时会返回 error，此时用 stats 查看 chapter_count。
- 如果搜索无结果，尝试缩短 query 或换用 mode="bm25"。
- save_outline 支持更新：同一小说同一章节重复调用会覆盖旧章纲。
"""
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

    async def _safe_tool_async(self, name: str, func, *args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except WebNovelError as e:
            logger.error(f"MCP tool '{name}' failed: {e}", exc_info=True)
            return {"error": str(e), "detail": e.detail, "tool": name, "type": type(e).__name__}
        except Exception as e:
            logger.error(f"MCP tool '{name}' failed: {e}", exc_info=True)
            return {"error": str(e), "tool": name}

    def _register_tools(self):
        from webnovel_kb.api.tools import register_all_tools
        register_all_tools(self.mcp, self.kb, self._safe_tool, self._safe_tool_async)
