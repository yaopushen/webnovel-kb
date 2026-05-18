"""BM25 keyword search via TantivyBM25."""
from typing import Optional, List, Dict, Any

from webnovel_kb.utils.logging_config import get_logger

logger = get_logger("search.bm25")


class BM25Search:
    """BM25 关键词搜索（TantivyBM25 实现，无内存冗余）。"""

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
                # BM25 索引存的是分词后文本，回 ChromaDB 取原始文本
                chunk_ids = [r.chunk_id for r in results if r.chunk_id]
                original_texts = {}
                if chunk_ids and self.collection:
                    try:
                        unique_ids = list(set(chunk_ids))
                        fetched = self.collection.get(ids=unique_ids, include=["documents"])
                        if fetched and fetched.get("ids"):
                            for cid, doc in zip(fetched["ids"], fetched["documents"]):
                                original_texts[cid] = doc
                    except Exception:
                        pass
                output = []
                for r in results:
                    text = original_texts.get(r.chunk_id, r.text)
                    output.append({
                        "text": text,
                        "metadata": r.metadata,
                        "bm25_score": round(r.score, 4),
                        "source": r.source
                    })
                return output

        return [{"status": "index_not_ready", "query": query, "hint": "Tantivy索引未就绪，请等待索引构建完成或重启服务"}]
