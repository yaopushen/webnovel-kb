"""Semantic search using vector embeddings."""
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger("webnovel-kb")


class SemanticSearch:
    """语义搜索，使用向量嵌入进行相似度检索。"""
    
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
            logger.warning(f"Semantic search failed: {e}")
            return [{"status": "semantic_unavailable", "error": str(e)[:100],
                     "hint": "Vector dimension mismatch. Try hybrid or BM25 mode instead."}]
        
        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0
                output.append({
                    "text": doc,
                    "metadata": meta,
                    "relevance": round(max(0, 1 - dist / 2), 4),
                    "source": f"{meta.get('title', '')} - {meta.get('author', '')} [{meta.get('chapter_title', '') or 'chunk ' + str(meta.get('chunk_index', ''))}]"
                })
        
        if not output:
            return [{"status": "no_results", "query": query, "hint": "未找到相关结果，尝试更换关键词或减少过滤条件"}]
        return output
