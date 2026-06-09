"""Index management for ChromaDB and Tantivy search engine."""
import threading
from pathlib import Path
from typing import Optional, Dict, List, Any

import jieba

try:
    from webnovel_kb.search_engines import (
        TantivyBM25, HybridSearchEngine,
        TANTIVY_AVAILABLE
    )
except ImportError:
    TANTIVY_AVAILABLE = False
    TantivyBM25 = None
    HybridSearchEngine = None

from webnovel_kb.utils.logging_config import get_logger

logger = get_logger("core.indexer")

STOPWORDS = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
             "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
             "自己", "这", "他", "她", "它", "们", "那", "被", "从", "把", "让", "对", "而",
             "但", "又", "么", "吗", "呢", "吧", "啊", "哦", "嗯", "呀", "啦", "哈"}


class IndexManager:
    """索引管理器，管理 ChromaDB 和 Tantivy BM25 索引。"""

    def __init__(self, data_dir: Path, collection, patterns_collection,
                 entities_collection, embedding_fn):
        self.data_dir = data_dir
        self.collection = collection
        self.patterns_collection = patterns_collection
        self.entities_collection = entities_collection
        self.embedding_fn = embedding_fn

        self._tantivy_index: Optional[TantivyBM25] = None
        self._hybrid_engine: Optional[HybridSearchEngine] = None
        self._use_optimized_search = TANTIVY_AVAILABLE

    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokens = list(jieba.cut(text))
        return [t.strip() for t in tokens if t.strip() and t.strip() not in STOPWORDS]

    def init_optimized_search(self) -> bool:
        """初始化优化搜索引擎（仅 TantivyBM25）。"""
        if not self._use_optimized_search:
            return False

        try:
            tantivy_dir = self.data_dir / "tantivy_index"
            self._tantivy_index = TantivyBM25(tantivy_dir)

            tantivy_exists = tantivy_dir.exists() and any(tantivy_dir.iterdir())

            self._hybrid_engine = HybridSearchEngine(
                self._tantivy_index,
                self.collection,
                self.embedding_fn,
                cache_ttl=60
            )

            logger.info("Optimized search engines initialized (Tantivy + ChromaDB)")
            return not tantivy_exists
        except Exception as e:
            logger.error(f"Failed to init optimized search: {e}")
            self._use_optimized_search = False
            return False

    def build_all_indexes(self, novels: dict) -> None:
        """构建所有索引。"""
        if self._use_optimized_search:
            self._build_tantivy_index()

    def _build_tantivy_index(self) -> None:
        if not self._use_optimized_search or not self._tantivy_index:
            return

        logger.info("Building Tantivy index from ChromaDB...")
        total = self.collection.count()
        if total == 0:
            return

        batch_size = 500
        documents = []

        for offset in range(0, total, batch_size):
            batch = self.collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset
            )
            if batch and batch.get("ids"):
                for i, cid in enumerate(batch["ids"]):
                    documents.append({
                        "chunk_id": cid,
                        "text": batch["documents"][i] if batch.get("documents") else "",
                        "metadata": batch["metadatas"][i] if batch.get("metadatas") else {}
                    })

            if offset % 5000 == 0:
                logger.info(f"Tantivy build progress: {len(documents)}/{total}")

        if documents:
            self._tantivy_index.build_index(documents)
            logger.info(f"Tantivy index built: {self._tantivy_index.doc_count} documents")

    @property
    def use_optimized_search(self) -> bool:
        return self._use_optimized_search

    @property
    def tantivy_index(self):
        return self._tantivy_index

    @property
    def hybrid_engine(self):
        return self._hybrid_engine

    @property
    def bm25_ready(self) -> bool:
        return self._use_optimized_search and self._tantivy_index is not None and self._tantivy_index.doc_count > 0
