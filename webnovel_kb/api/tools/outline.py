"""章纲类工具：save_outline / get_outline / extract_outline。"""
import asyncio
from typing import Union


def register_outline_tools(mcp, kb, safe, safe_async):

    @mcp.tool()
    async def save_outline(novel_title: str, outlines: Union[list, dict],
                     overwrite: bool = False) -> dict:
        """Save single or batch chapter outlines to the knowledge base."""
        return await safe_async("save_outline", kb.save_outline,
                               novel_title, outlines, overwrite=overwrite)

    @mcp.tool()
    async def get_outline(novel_title: str, chapter: Union[int, str] = 0) -> dict:
        """Retrieve stored chapter outlines (individual chapter or full book outline list)."""
        if isinstance(chapter, int) and chapter <= 0:
            chapter = None
        return await asyncio.to_thread(safe, "get_outline", kb.get_outline, novel_title, chapter)

    @mcp.tool()
    async def extract_outline(novel_title: str, chapter: int, end_chapter: int = 0) -> dict:
        """Extract and save outlines from novel chapters using LLM (single or async batch)."""
        if chapter <= 0:
            return {"error": f"章节号必须大于 0，当前值: {chapter}"}
        if end_chapter < 0:
            return {"error": f"结束章节号不能为负数，当前值: {end_chapter}"}
        if end_chapter > 0 and end_chapter < chapter:
            return {"error": f"结束章节号({end_chapter})不能小于起始章节号({chapter})"}
        if end_chapter > 0 and end_chapter >= chapter:
            total = end_chapter - chapter + 1
            if total > 20:
                return {"error": f"批量提取上限 20 章，当前请求 {total} 章。请缩小范围。"}
            return await safe_async(
                "extract_outline", kb.start_outline_extraction,
                novel_title, chapter, end_chapter
            )
        return await safe_async("extract_outline", kb.extract_outline, novel_title, chapter)
