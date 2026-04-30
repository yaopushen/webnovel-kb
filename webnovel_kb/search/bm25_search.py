"""BM25 keyword search."""
from typing import Optional, List, Dict, Any


class BM25Search:
    """BM25 关键词搜索。"""
    
    def __init__(self, index_manager, collection, novels: dict):
        self.index_manager = index_manager
        self.collection = collection
        self.novels = novels
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        novel_filter: Optional[str] = None,
        genre_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """执行 BM25 搜索。"""
        if self.index_manager.use_optimized_search and self.index_manager.tantivy_index:
            results = self.index_manager.tantivy_index.search(query, n_results, novel_filter, genre_filter)
            if results:
                return [{
                    "text": r.text,
                    "metadata": r.metadata,
                    "bm25_score": round(r.score, 4),
                    "source": r.source
                } for r in results]
        
        self.index_manager.ensure_bm25()
        if not self.index_manager.bm25:
            return [{"status": "index_not_ready", "query": query, "hint": "BM25索引未就绪，请等待索引构建完成或重启服务"}]
        
        tokens = self.index_manager.tokenize(query)
        if not tokens:
            return [{"status": "no_results", "query": query, "hint": "分词结果为空，请尝试其他关键词"}]
        
        scores = self.index_manager.bm25.get_scores(tokens)
        scored_indices = []
        
        for idx in range(len(scores)):
            if scores[idx] <= 0:
                continue
            meta = self.index_manager.bm25_metadata[idx] if idx < len(self.index_manager.bm25_metadata) else {}
            if novel_filter and meta.get("title", "") != novel_filter:
                continue
            if genre_filter:
                novel_id = meta.get("novel_id", "")
                if novel_id in self.novels and self.novels[novel_id].genre != genre_filter:
                    continue
            scored_indices.append((idx, scores[idx]))
        
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = scored_indices[:n_results]
        
        results = []
        for idx, score in top_indices:
            meta = self.index_manager.bm25_metadata[idx] if idx < len(self.index_manager.bm25_metadata) else {}
            chunk_id = f"{meta.get('novel_id', '')}_{meta.get('chunk_index', 0)}"
            try:
                doc = self.collection.get(ids=[chunk_id], include=["documents"])
                text = doc["documents"][0] if doc["documents"] else ""
            except Exception:
                text = ""
            results.append({
                "text": text,
                "metadata": meta,
                "bm25_score": round(float(score), 4),
                "source": f"{meta.get('title', '')} - {meta.get('author', '')} [{meta.get('chapter_title', '') or 'chunk ' + str(meta.get('chunk_index', ''))}]"
            })
        return results
