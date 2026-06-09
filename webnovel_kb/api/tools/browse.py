"""浏览类工具：stats / read_chapter / get_chapter_edges。"""
import asyncio


def register_browse_tools(mcp, kb, safe):

    @mcp.tool()
    async def stats(scope: str = "global", novel_title: str = "") -> dict:
        """Get store statistics (or scope="guide" for complete tools reference)."""
        if scope == "guide":
            from webnovel_kb.api.tools.tools_guide import TOOLS_GUIDE_MD
            return {"tools_guide": TOOLS_GUIDE_MD}
        if novel_title and novel_title.strip():
            return await asyncio.to_thread(safe, "novel_stats", kb.novel_stats, novel_title)
        if scope == "novels":
            return await asyncio.to_thread(safe, "list_novels", kb.list_novels)
        if scope == "knowledge":
            return {"knowledge": kb.knowledge_store.list_all()}
        if scope != "global":
            return await asyncio.to_thread(safe, "novel_stats", kb.novel_stats, scope)
        return await asyncio.to_thread(safe, "get_stats", kb.get_stats)

    @mcp.tool()
    async def read_chapter(novel_title: str, chapter: int = 1) -> dict:
        """Read the complete text of a specific chapter from a novel."""
        if chapter <= 0:
            return {"error": f"章节号必须大于 0，当前值: {chapter}"}
        return await asyncio.to_thread(safe, "read_chapter", kb.read_chapter, novel_title, chapter)

    @mcp.tool()
    async def get_chapter_edges(novel_title: str, chapter: int = 1,
                          paragraphs: int = 2) -> dict:
        """Extract the opening and closing paragraphs of a specific chapter."""
        if chapter <= 0:
            return {"error": f"章节号必须大于 0，当前值: {chapter}"}
        if paragraphs <= 0:
            return {"error": f"段落数必须大于 0，当前值: {paragraphs}"}
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
