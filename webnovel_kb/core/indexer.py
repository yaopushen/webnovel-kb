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

    def index_plot_patterns(self, plot_patterns: list) -> None:
        """索引情节模式到向量存储。"""
        if not plot_patterns:
            return
        try:
            existing_ids = set()
            offset = 0
            batch = 1000
            while True:
                result = self.patterns_collection.get(limit=batch, offset=offset, include=[])
                ids_chunk = result.get("ids", []) if result else []
                if not ids_chunk:
                    break
                existing_ids.update(ids_chunk)
                if len(ids_chunk) < batch:
                    break
                offset += batch

            ids = []
            documents = []
            metadatas = []
            for i, p in enumerate(plot_patterns):
                pattern_id = f"pattern_{i}"
                if pattern_id in existing_ids:
                    continue
                text = f"{p.pattern_type}: {p.description}"
                if p.pattern_text:
                    text += f"\n{p.pattern_text}"
                ids.append(pattern_id)
                documents.append(text)
                metadatas.append({
                    "pattern_type": p.pattern_type or "",
                    "source_novel": p.source_novel or "",
                    "effectiveness": p.effectiveness or "",
                    "index": i
                })
            if ids:
                batch_size = 100
                for j in range(0, len(ids), batch_size):
                    self.patterns_collection.upsert(
                        ids=ids[j:j+batch_size],
                        documents=documents[j:j+batch_size],
                        metadatas=metadatas[j:j+batch_size]
                    )
                logger.info(f"Indexed {len(ids)} plot patterns into vector store")
        except Exception as e:
            logger.warning(f"Failed to index plot patterns: {e}")

    def index_entities(self, entities: dict) -> None:
        """索引实体到向量存储。"""
        if not entities:
            return
        try:
            existing_ids = set()
            offset = 0
            batch = 1000
            while True:
                result = self.entities_collection.get(limit=batch, offset=offset, include=[])
                ids_chunk = result.get("ids", []) if result else []
                if not ids_chunk:
                    break
                existing_ids.update(ids_chunk)
                if len(ids_chunk) < batch:
                    break
                offset += batch

            ids = []
            documents = []
            metadatas = []
            for eid, e in entities.items():
                entity_id = f"entity_{eid}"
                if entity_id in existing_ids:
                    continue
                text = f"{e.entity_type}: {e.name}"
                if e.description:
                    text += f" - {e.description}"
                ids.append(entity_id)
                documents.append(text)
                metadatas.append({
                    "entity_type": e.entity_type or "",
                    "name": e.name,
                    "source_novel": e.source_novel or "",
                    "entity_id": eid
                })
            if ids:
                batch_size = 100
                for j in range(0, len(ids), batch_size):
                    self.entities_collection.upsert(
                        ids=ids[j:j+batch_size],
                        documents=documents[j:j+batch_size],
                        metadatas=metadatas[j:j+batch_size]
                    )
                logger.info(f"Indexed {len(ids)} entities into vector store")
        except Exception as e:
            logger.warning(f"Failed to index entities: {e}")

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
