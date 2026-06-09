"""搜索类工具：search / smart_search。"""


def register_search_tools(mcp, kb, safe_async):

    @mcp.tool()
    async def search(query: str, scope: str = "chunks", mode: str = "hybrid",
               n_results: int = 10, novel_filter: str = "", genre_filter: str = "",
               chapter_filter: str = "", alpha: float = 0.6,
               use_rerank: bool = False,
               output_format: str = "compact", max_content_length: int = 0,
               dedupe: bool = True) -> list:
        """Search text, outlines, or knowledge using keyword/semantic queries."""
        resolved_novel = kb.resolve_novel_title(novel_filter) if novel_filter else None
        return await safe_async("search_with_scope", kb.search_with_scope,
            query, scope=scope, n_results=n_results,
            novel_filter=resolved_novel or "",
            genre_filter=genre_filter or "",
            mode=mode, alpha=alpha, use_rerank=use_rerank,
            output_format=output_format,
            max_content_length=max_content_length,
            dedupe=dedupe
        )

    @mcp.tool()
    async def smart_search(query: str, n_results: int = 5,
                     novel_filter: str = "", genre_filter: str = "",
                     output_format: str = "compact") -> dict:
        """Execute multi-round intelligent search combined with web and deep reasoning."""
        resolved_novel = kb.resolve_novel_title(novel_filter) if novel_filter else None
        return await safe_async("smart_search", kb.smart_search,
            query, n_results=n_results,
            novel_filter=resolved_novel,
            genre_filter=genre_filter or None,
            output_format=output_format
        )
