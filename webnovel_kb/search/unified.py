"""Unified search interface with output formatting."""
from typing import Optional, List, Any

from webnovel_kb.utils.dedupe import dedupe_results
from webnovel_kb.utils.format import format_search_results


class UnifiedSearch:
    """统一搜索接口，整合所有搜索模式。"""
    
    def __init__(self, semantic_search, bm25_search, hybrid_search, rerank_search, reranker):
        self.semantic_search = semantic_search
        self.bm25_search = bm25_search
        self.hybrid_search = hybrid_search
        self.rerank_search = rerank_search
        self.reranker = reranker
    
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
        """
        统一搜索接口。
        
        Args:
            query: 搜索查询
            mode: 搜索模式 - semantic/bm25/hybrid/rerank
            n_results: 返回结果数量
            novel_filter: 小说标题过滤
            genre_filter: 类型过滤
            chapter_filter: 章节过滤
            alpha: hybrid模式下语义权重
            use_rerank: 是否使用rerank精排
            output_format: 输出格式 - raw/compact/clean
            max_content_length: 内容最大长度
            dedupe: 是否去重
        
        Returns:
            搜索结果列表
        """
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
        
        return format_search_results(raw, output_format, max_content_length, dedupe, dedupe_results)
