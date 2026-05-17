"""Unified search interface with output formatting and query cache."""
from typing import Optional, List, Any

from webnovel_kb.utils.dedupe import dedupe_results
from webnovel_kb.utils.format import format_search_results
from webnovel_kb.utils.logging_config import get_logger
from webnovel_kb.utils.query_cache import QueryCache

logger = get_logger("search.unified")


class UnifiedSearch:
    """统一搜索接口，整合所有搜索模式。"""

    def __init__(self, semantic_search, bm25_search, hybrid_search, rerank_search,
                 reranker, query_cache: QueryCache = None):
        self.semantic_search = semantic_search
        self.bm25_search = bm25_search
        self.hybrid_search = hybrid_search
        self.rerank_search = rerank_search
        self.reranker = reranker
        self._query_cache = query_cache

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        n_results: int = 10,
        novel_filter: Optional[str] = None,
        genre_filter: Optional[str] = None,
        chapter_filter: Optional[str] = None,
        alpha: float = 0.6,
        use_rerank: bool = False,
        output_format: str = "compact",
        max_content_length: int = 0,
        dedupe: bool = True
    ) -> List[Any]:
        """统一搜索接口。"""
        if self._query_cache is not None:
            cache_key = self._query_cache.make_key(
                query, mode=mode, n_results=n_results,
                novel_filter=novel_filter, genre_filter=genre_filter,
                chapter_filter=chapter_filter, alpha=alpha,
                use_rerank=use_rerank, output_format=output_format,
                max_content_length=max_content_length, dedupe=dedupe
            )
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Query cache hit: {query[:50]}...")
                return cached

        if mode == "semantic":
            raw = self.semantic_search.search(query, n_results, novel_filter, genre_filter, chapter_filter)
        elif mode == "bm25":
            raw = self.bm25_search.search(query, n_results, novel_filter, genre_filter)
        elif mode == "rerank":
            raw = self.rerank_search.search(query, n_results, novel_filter, genre_filter)
        else:
            if use_rerank and self.reranker:
                raw = self.rerank_search.search(query, n_results, novel_filter, genre_filter)
            else:
                raw = self.hybrid_search.search(query, n_results, alpha, novel_filter, genre_filter)

        result = format_search_results(raw, output_format, max_content_length, dedupe, dedupe_results)

        if self._query_cache is not None:
            self._query_cache.put(cache_key, result)

        return result
