"""Hybrid search combining semantic and BM25."""
from typing import Optional, List, Dict, Any
import numpy as np


class HybridSearch:
    """混合搜索，结合语义搜索和 BM25。"""
    
    def __init__(self, index_manager, semantic_search, bm25_search, embedding_fn):
        self.index_manager = index_manager
        self.semantic_search = semantic_search
        self.bm25_search = bm25_search
        self.embedding_fn = embedding_fn
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        alpha: float = 0.6,
        novel_filter: Optional[str] = None,
        genre_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """执行混合搜索，使用 RRF 融合。"""
        if self.index_manager.use_optimized_search and self.index_manager.hybrid_engine and self.index_manager.faiss_store:
            if self.index_manager.faiss_store.count == 0:
                self.index_manager._build_faiss_index()
            if self.index_manager.faiss_store.count > 0:
                query_vector = np.array(self.embedding_fn([query])[0], dtype=np.float32)
                return self.index_manager.hybrid_engine.search(
                    query, query_vector, n_results, alpha, novel_filter, genre_filter
                )
        
        sem_results = self.semantic_search.search(query, n_results=n_results * 3,
                                                   novel_filter=novel_filter, genre_filter=genre_filter)
        bm25_results = self.bm25_search.search(query, n_results=n_results * 3,
                                                novel_filter=novel_filter, genre_filter=genre_filter)
        
        k = 60
        rrf_scores: Dict[str, Dict] = {}
        
        for rank, r in enumerate(sem_results):
            if "status" in r:
                continue
            key = f"{r['metadata'].get('novel_id', '')}_{r['metadata'].get('chunk_index', '')}"
            if key not in rrf_scores:
                rrf_scores[key] = {"data": r, "sem_score": r["relevance"], "bm25_score": 0}
            rrf_scores[key]["sem_score"] = r["relevance"]
            rrf_scores[key]["sem_rank"] = rank + 1
        
        for rank, r in enumerate(bm25_results):
            if "status" in r:
                continue
            key = f"{r['metadata'].get('novel_id', '')}_{r['metadata'].get('chunk_index', '')}"
            if key not in rrf_scores:
                rrf_scores[key] = {"data": r, "sem_score": 0, "bm25_score": r.get("bm25_score", 0)}
            rrf_scores[key]["bm25_score"] = r.get("bm25_score", 0)
            rrf_scores[key]["bm25_rank"] = rank + 1
        
        for key in rrf_scores:
            item = rrf_scores[key]
            sem_rrf = 1.0 / (k + item.get("sem_rank", k * 3)) if "sem_rank" in item else 0
            bm25_rrf = 1.0 / (k + item.get("bm25_rank", k * 3)) if "bm25_rank" in item else 0
            item["hybrid_score"] = round(alpha * sem_rrf + (1 - alpha) * bm25_rrf, 6)
        
        sorted_items = sorted(rrf_scores.values(), key=lambda x: x["hybrid_score"], reverse=True)
        
        output = []
        for item in sorted_items[:n_results]:
            result = item["data"].copy()
            result["sem_score"] = item["sem_score"]
            result["bm25_score"] = item["bm25_score"]
            result["hybrid_score"] = item["hybrid_score"]
            output.append(result)
        return output
