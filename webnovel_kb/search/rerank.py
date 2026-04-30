"""Rerank search using cross-encoder models."""
from typing import Optional, List, Dict, Any


class RerankSearch:
    """重排序搜索，使用 rerank 模型对结果进行精排。"""
    
    def __init__(self, reranker, hybrid_search):
        self.reranker = reranker
        self.hybrid_search = hybrid_search
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        novel_filter: Optional[str] = None,
        genre_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """执行重排序搜索。"""
        if not self.reranker:
            return self.hybrid_search.search(query, n_results=n_results,
                                              novel_filter=novel_filter, genre_filter=genre_filter)
        
        candidates = self.hybrid_search.search(query, n_results=n_results * 5,
                                                novel_filter=novel_filter, genre_filter=genre_filter)
        if not candidates:
            return []
        
        documents = [c["text"] for c in candidates]
        rerank_results = self.reranker.rerank(query, documents, top_n=n_results)
        
        output = []
        for item in rerank_results:
            idx = item.get("index", 0)
            if idx < len(candidates):
                c = candidates[idx].copy()
                c["rerank_score"] = item.get("relevance_score", 0)
                c["search_method"] = "hybrid+rerank"
                output.append(c)
        return output
