"""Index management for ChromaDB and optimized search engines."""
import pickle
import threading
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import jieba

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from webnovel_kb.search_engines import (
        TantivyBM25, FAISSVectorStore, HybridSearchEngine,
        TANTIVY_AVAILABLE, FAISS_AVAILABLE
    )
except ImportError:
    TANTIVY_AVAILABLE = False
    FAISS_AVAILABLE = False
    TantivyBM25 = None
    FAISSVectorStore = None
    HybridSearchEngine = None

from webnovel_kb.config import LLM_EMBEDDING_DIMENSIONS
from webnovel_kb.utils.logging_config import get_logger
from webnovel_kb.utils.exceptions import IndexError_

logger = get_logger("core.indexer")

STOPWORDS = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
             "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
             "自己", "这", "他", "她", "它", "们", "那", "被", "从", "把", "让", "对", "而",
             "但", "又", "么", "吗", "呢", "吧", "啊", "哦", "嗯", "呀", "啦", "哈"}


class IndexManager:
    """索引管理器，管理 ChromaDB、BM25、FAISS、Tantivy 索引。"""

    def __init__(self, data_dir: Path, collection, patterns_collection,
                 entities_collection, embedding_fn):
        self.data_dir = data_dir
        self.collection = collection
        self.patterns_collection = patterns_collection
        self.entities_collection = entities_collection
        self.embedding_fn = embedding_fn

        self.bm25_corpus: List[List[str]] = []
        self.bm25_metadata: List[dict] = []
        self.bm25: Optional[Any] = None
        self._bm25_ready = False

        self._tantivy_index: Optional[TantivyBM25] = None
        self._faiss_store: Optional[FAISSVectorStore] = None
        self._hybrid_engine: Optional[HybridSearchEngine] = None
        self._use_optimized_search = TANTIVY_AVAILABLE and FAISS_AVAILABLE

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """分词函数。"""
        tokens = list(jieba.cut(text))
        return [t.strip() for t in tokens if t.strip() and t.strip() not in STOPWORDS]

    def init_optimized_search(self) -> bool:
        """初始化优化搜索引擎。"""
        if not self._use_optimized_search:
            return False

        try:
            tantivy_dir = self.data_dir / "tantivy_index"
            self._tantivy_index = TantivyBM25(tantivy_dir)

            faiss_path = self.data_dir / "faiss_index.faiss"
            self._faiss_store = FAISSVectorStore(
                faiss_path,
                dimensions=LLM_EMBEDDING_DIMENSIONS
            )

            faiss_loaded = False
            if faiss_path.exists():
                faiss_loaded = self._faiss_store.load_index()
                if faiss_loaded:
                    logger.info(f"FAISS index loaded: {self._faiss_store.count} vectors")

            tantivy_dir = self.data_dir / "tantivy_index"
            tantivy_exists = tantivy_dir.exists() and any(tantivy_dir.iterdir())

            self._hybrid_engine = HybridSearchEngine(
                self._tantivy_index,
                self._faiss_store,
                cache_ttl=60
            )

            logger.info("Optimized search engines initialized")
            return not faiss_loaded or not tantivy_exists
        except Exception as e:
            logger.error(f"Failed to init optimized search: {e}")
            self._use_optimized_search = False
            return False

    def build_all_indexes(self, novels: dict) -> None:
        """构建所有索引。"""
        if self._use_optimized_search:
            self._build_tantivy_index()
            self._build_faiss_index()
        self._rebuild_bm25()

    def _build_faiss_index(self) -> None:
        if not self._use_optimized_search or not self._faiss_store:
            return

        logger.info("Building FAISS index from ChromaDB...")
        total = self.collection.count()
        if total == 0:
            logger.warning("No documents in ChromaDB, cannot build FAISS index")
            return

        batch_size = 500
        all_vectors = []
        all_ids = []
        all_texts = []
        all_metas = []

        for offset in range(0, total, batch_size):
            batch = self.collection.get(
                include=["documents", "metadatas", "embeddings"],
                limit=batch_size,
                offset=offset
            )
            if batch and batch.get("ids"):
                for i, cid in enumerate(batch["ids"]):
                    if batch.get("embeddings") and batch["embeddings"][i]:
                        all_vectors.append(batch["embeddings"][i])
                        all_ids.append(cid)
                        all_texts.append(batch["documents"][i] if batch.get("documents") else "")
                        all_metas.append(batch["metadatas"][i] if batch.get("metadatas") else {})

            if offset % 5000 == 0:
                logger.info(f"FAISS build progress: {len(all_vectors)}/{total}")

        if all_vectors:
            vectors = np.array(all_vectors, dtype=np.float32)
            self._faiss_store.build_index(vectors, all_ids, all_texts, all_metas)
            logger.info(f"FAISS index built: {self._faiss_store.count} vectors")

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

    def _rebuild_bm25(self) -> None:
        """从缓存加载或重建 BM25 索引。"""
        bm25_cache = self.data_dir / "bm25_index.pkl"
        if bm25_cache.exists():
            try:
                with open(bm25_cache, "rb") as f:
                    cache = pickle.load(f)
                self.bm25_corpus = cache["corpus"]
                self.bm25_metadata = cache["metadata"]
                logger.info(f"BM25 corpus loaded: {len(self.bm25_corpus)} documents (index deferred)")
                self._bm25_ready = False
                return
            except Exception as e:
                logger.warning(f"BM25 cache load failed: {e}, will rebuild on demand")
        self._bm25_ready = False

    def ensure_bm25(self) -> None:
        """确保 BM25 索引可用。"""
        if self._bm25_ready and self.bm25 is not None:
            return
        if not BM25_AVAILABLE:
            logger.warning("rank_bm25 not available, BM25 search disabled")
            return
        if self.bm25_corpus:
            logger.info("Building BM25 index on demand...")
            self.bm25 = BM25Okapi(self.bm25_corpus)
            self._bm25_ready = True
            logger.info(f"BM25 index ready: {len(self.bm25_corpus)} documents")
        else:
            self._rebuild_bm25_from_chroma()

    def _rebuild_bm25_from_chroma(self) -> None:
        """从 ChromaDB 重建 BM25 索引。"""
        try:
            total = self.collection.count()
            if total == 0:
                logger.warning("ChromaDB is empty, BM25 not available")
                return
            self.bm25_corpus = []
            self.bm25_metadata = []
            batch_size = 500
            for offset in range(0, total, batch_size):
                limit = min(batch_size, total - offset)
                batch_ids = self.collection.get(
                    include=["documents", "metadatas"],
                    limit=limit,
                    offset=offset
                )
                if batch_ids and batch_ids["documents"]:
                    for i, doc in enumerate(batch_ids["documents"]):
                        meta = batch_ids["metadatas"][i] if batch_ids["metadatas"] else {}
                        tokens = self.tokenize(doc)
                        self.bm25_corpus.append(tokens)
                        self.bm25_metadata.append({
                            "novel_id": meta.get("novel_id", ""),
                            "title": meta.get("title", ""),
                            "chunk_index": meta.get("chunk_index", 0)
                        })
                logger.info(f"BM25 rebuild progress: {min(offset + batch_size, total)}/{total}")
            if self.bm25_corpus:
                if BM25_AVAILABLE:
                    self.bm25 = BM25Okapi(self.bm25_corpus)
                    logger.info(f"BM25 rebuilt from ChromaDB: {len(self.bm25_corpus)} documents")
                self._save_bm25_cache()
            else:
                logger.warning("No documents found, BM25 not available")
        except Exception as e:
            logger.error(f"Failed to rebuild BM25 from ChromaDB: {e}")

    def _save_bm25_cache(self) -> None:
        """保存 BM25 缓存。"""
        bm25_cache = self.data_dir / "bm25_index.pkl"
        try:
            with open(bm25_cache.with_suffix('.pkl.tmp'), "wb") as f:
                pickle.dump({
                    "corpus": self.bm25_corpus,
                    "metadata": self.bm25_metadata,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            bm25_cache.with_suffix('.pkl.tmp').replace(bm25_cache)
            logger.info(f"BM25 cache saved: {len(self.bm25_corpus)} documents")
        except Exception as e:
            logger.error(f"BM25 cache save failed: {e}")

    def preload_bm25_background(self) -> None:
        """后台预加载 BM25 索引。"""
        if not BM25_AVAILABLE:
            return
        if self.bm25_corpus and not self._bm25_ready:
            def _build():
                try:
                    logger.info("Pre-building BM25 index in background...")
                    self.bm25 = BM25Okapi(self.bm25_corpus)
                    self._bm25_ready = True
                    logger.info(f"BM25 index pre-built: {len(self.bm25_corpus)} documents")
                except Exception as e:
                    logger.warning(f"BM25 background build failed: {e}")
            t = threading.Thread(target=_build, daemon=True)
            t.start()

    def add_to_bm25(self, text: str, metadata: dict) -> None:
        """添加文档到 BM25 索引。"""
        tokens = self.tokenize(text)
        self.bm25_corpus.append(tokens)
        self.bm25_metadata.append(metadata)

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
                    "name": e.name or "",
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
    def faiss_store(self):
        return self._faiss_store

    @property
    def hybrid_engine(self):
        return self._hybrid_engine

    @property
    def bm25_ready(self) -> bool:
        return self._bm25_ready
