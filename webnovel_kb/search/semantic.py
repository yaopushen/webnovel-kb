"""Semantic search using vector embeddings."""
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("webnovel-kb")


class SemanticSearch:
    """语义搜索，基于向量嵌入。"""
    
    def __init__(self, collection, embedding_fn=None):
        self.collection = collection
        self.embedding_fn = embedding_fn
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        novel_filter: Optional[str] = None,
        genre_filter: Optional[str] = None,
        chapter_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """执行语义搜索。"""
        where_filter = {}
        if novel_filter:
            where_filter["title"] = novel_filter
        if genre_filter:
            where_filter["genre"] = genre_filter
        if chapter_filter:
            where_filter["chapter_title"] = chapter_filter
        
        query_params = {"n_results": n_results}
        if where_filter:
            query_params["where"] = where_filter
        
        if self.embedding_fn:
            try:
                query_vec = self.embedding_fn([query])[0]
                query_params["query_embeddings"] = [query_vec]
            except Exception as e:
                logger.warning(f"Embedding failed, falling back to query_texts: {e}")
                query_params["query_texts"] = [query]
        else:
            query_params["query_texts"] = [query]
        
        try:
            results = self.collection.query(**query_params)
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return [{"status": "error", "error": str(e)[:200],
                     "hint": "Semantic search unavailable. Try mode='bm25'."}]
        
        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0
                source_parts = []
                if meta.get("title"):
                    source_parts.append(f"《{meta['title']}》")
                if meta.get("chapter_title"):
                    source_parts.append(meta["chapter_title"])
                source = " ".join(source_parts)
                output.append({
                    "text": doc,
                    "metadata": meta,
                    "relevance": round(max(0, 1 - dist / 2), 4),
                    "source": source
                })
        return output
