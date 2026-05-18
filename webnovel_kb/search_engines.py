"""
Optimized search engines: Tantivy for BM25, ChromaDB for vectors.
v1.9: Removed FAISS (GPU incompatible with GTX 1060) and rank_bm25 (memory hog).
"""
from __future__ import annotations
import os
import time
import threading
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    import tantivy

import jieba

try:
    import tantivy
    TANTIVY_AVAILABLE = True
except ImportError:
    tantivy = None
    TANTIVY_AVAILABLE = False


STOPWORDS = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
             "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
             "自己", "这", "他", "她", "它", "们", "那", "被", "从", "把", "让", "对", "而",
             "但", "又", "么", "吗", "呢", "吧", "啊", "哦", "嗯", "呀", "啦", "哈"}


def tokenize(text: str) -> List[str]:
    tokens = list(jieba.cut(text))
    return [t.strip() for t in tokens if t.strip() and t.strip() not in STOPWORDS]


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    metadata: dict
    score: float
    source: str


class TantivyBM25:
    def __init__(self, index_dir: Path, tokenizer: Callable[[str], List[str]] = tokenize):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self._index: Optional[tantivy.Index] = None
        self._searcher: Optional[tantivy.Searcher] = None
        self._doc_count = 0
        self._id_to_meta: Dict[str, dict] = {}
        self._id_to_text: Dict[str, str] = {}
        self._lock = threading.RLock()
        
        if TANTIVY_AVAILABLE:
            self._try_load_existing_index()
        
    def _try_load_existing_index(self):
        """尝试加载已存在的索引。"""
        try:
            if any(self.index_dir.iterdir()):
                schema = self._build_schema()
                self._index = tantivy.Index(schema, path=str(self.index_dir))
                self._index.reload()
                self._searcher = self._index.searcher()
                
                self._doc_count = self._searcher.num_docs
                
                self._load_metadata_cache()
        except Exception as e:
            from webnovel_kb.utils.logging_config import get_logger
            get_logger("engines.tantivy").warning(f"Failed to load existing index: {e}")
    
    def _load_metadata_cache(self):
        """从索引中加载元数据缓存。"""
        if not self._searcher or self._doc_count == 0:
            return
        try:
            for doc_id in range(self._doc_count):
                try:
                    doc = self._searcher.doc(doc_id)
                    if doc:
                        chunk_id = doc.get_first("chunk_id")
                        if chunk_id:
                            self._id_to_meta[chunk_id] = {
                                "title": doc.get_first("title") or "",
                                "author": doc.get_first("author") or "",
                                "genre": doc.get_first("genre") or "",
                                "chapter_title": doc.get_first("chapter_title") or "",
                                "novel_id": doc.get_first("novel_id") or "",
                                "chunk_index": doc.get_first("chunk_index") or 0
                            }
                            self._id_to_text[chunk_id] = doc.get_first("text") or ""
                except Exception:
                    continue
        except Exception as e:
            from webnovel_kb.utils.logging_config import get_logger
            get_logger("engines.tantivy").warning(f"Failed to load metadata cache: {e}")
        
    def _build_schema(self) -> tantivy.Schema:
        builder = tantivy.SchemaBuilder()
        builder.add_text_field("chunk_id", stored=True, tokenizer_name="raw")
        builder.add_text_field("text", stored=True)
        builder.add_text_field("title", stored=True)
        builder.add_text_field("author", stored=True)
        builder.add_text_field("genre", stored=True)
        builder.add_text_field("chapter_title", stored=True)
        builder.add_text_field("novel_id", stored=True)
        builder.add_integer_field("chunk_index", stored=True)
        return builder.build()
    
    def build_index(self, documents: List[dict], clear_existing: bool = True):
        if not TANTIVY_AVAILABLE:
            raise RuntimeError("Tantivy not installed. Run: pip install tantivy")
        
        with self._lock:
            if clear_existing and self._index is not None:
                import shutil
                for f in self.index_dir.iterdir():
                    if f.is_file():
                        f.unlink()
                self._index = None
                self._searcher = None
                self._id_to_meta.clear()
                self._id_to_text.clear()
            
            schema = self._build_schema()
            self._index = tantivy.Index(schema, path=str(self.index_dir))
            writer = self._index.writer(50_000_000, num_threads=2)
            
            for doc in documents:
                chunk_id = doc.get("chunk_id", "")
                text = doc.get("text", "")
                meta = doc.get("metadata", {})
                
                tantivy_doc = tantivy.Document(
                    chunk_id=chunk_id,
                    text=" ".join(self.tokenizer(text)),
                    title=meta.get("title", ""),
                    author=meta.get("author", ""),
                    genre=meta.get("genre", ""),
                    chapter_title=meta.get("chapter_title", ""),
                    novel_id=meta.get("novel_id", ""),
                    chunk_index=meta.get("chunk_index", 0)
                )
                writer.add_document(tantivy_doc)
                self._id_to_meta[chunk_id] = meta
                self._id_to_text[chunk_id] = text
            
            writer.commit()
            self._index.reload()
            self._searcher = self._index.searcher()
            self._doc_count = len(documents)
    
    def search(self, query: str, n_results: int = 10, 
               novel_filter: Optional[str] = None,
               genre_filter: Optional[str] = None) -> List[SearchResult]:
        if self._index is None:
            return []
        
        tokens = self.tokenizer(query)
        if not tokens:
            return []
        
        query_str = " ".join(tokens)
        parsed = self._index.parse_query(query_str)
        
        results = self._searcher.search(parsed, limit=n_results * 3)
        
        output = []
        for hit in results.hits:
            doc_addr = hit.doc_address if hasattr(hit, 'doc_address') else (hit[1] if isinstance(hit, tuple) and len(hit) > 1 else None)
            hit_score = hit.score if hasattr(hit, 'score') else (hit[0] if isinstance(hit, tuple) else 0.0)
            if doc_addr is None:
                continue
            doc = self._searcher.doc(doc_addr)
            chunk_id = doc.get_first("chunk_id")
            
            meta = self._id_to_meta.get(chunk_id, {
                "title": doc.get_first("title") or "",
                "author": doc.get_first("author") or "",
                "genre": doc.get_first("genre") or "",
                "chapter_title": doc.get_first("chapter_title") or "",
                "novel_id": doc.get_first("novel_id") or "",
                "chunk_index": doc.get_first("chunk_index") or 0
            })
            
            if novel_filter and meta.get("title") != novel_filter:
                continue
            if genre_filter and meta.get("genre") != genre_filter:
                continue
            
            text = self._id_to_text.get(chunk_id, doc.get_first("text") or "")
            source = f"{meta.get('title', '')} - {meta.get('author', '')} [{meta.get('chapter_title', '') or 'chunk ' + str(meta.get('chunk_index', ''))}]"
            
            output.append(SearchResult(
                chunk_id=chunk_id,
                text=text,
                metadata=meta,
                score=hit_score,
                source=source
            ))
            
            if len(output) >= n_results:
                break
        
        return output
    
    def add_document(self, chunk_id: str, text: str, metadata: dict):
        with self._lock:
            if self._index is None:
                schema = self._build_schema()
                self._index = tantivy.Index(schema, path=str(self.index_dir))
            
            writer = self._index.writer(50_000_000, num_threads=1)
            tantivy_doc = tantivy.Document(
                chunk_id=chunk_id,
                text=" ".join(self.tokenizer(text)),
                title=metadata.get("title", ""),
                author=metadata.get("author", ""),
                genre=metadata.get("genre", ""),
                chapter_title=metadata.get("chapter_title", ""),
                novel_id=metadata.get("novel_id", ""),
                chunk_index=metadata.get("chunk_index", 0)
            )
            writer.add_document(tantivy_doc)
            writer.commit()
            self._index.reload()
            self._searcher = self._index.searcher()
            self._id_to_meta[chunk_id] = metadata
            self._id_to_text[chunk_id] = text
            self._doc_count += 1
    
    @property
    def doc_count(self) -> int:
        return self._doc_count


class QueryCache:
    def __init__(self, ttl_seconds: int = 60, max_size: int = 1000):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, tuple[float, Any]] = {}
        self._lock = threading.RLock()
    
    def _key(self, query: str, search_type: str, filters: dict) -> str:
        filter_str = json.dumps(filters, sort_keys=True)
        return hashlib.md5(f"{search_type}:{query}:{filter_str}".encode()).hexdigest()
    
    def get(self, query: str, search_type: str, filters: dict) -> Optional[Any]:
        with self._lock:
            key = self._key(query, search_type, filters)
            if key in self._cache:
                ts, result = self._cache[key]
                if time.time() - ts < self.ttl:
                    return result
                del self._cache[key]
        return None
    
    def set(self, query: str, search_type: str, filters: dict, result: Any):
        with self._lock:
            if len(self._cache) >= self.max_size:
                oldest = min(self._cache.items(), key=lambda x: x[1][0])
                del self._cache[oldest[0]]
            
            key = self._key(query, search_type, filters)
            self._cache[key] = (time.time(), result)
    
    def clear(self):
        with self._lock:
            self._cache.clear()


class HybridSearchEngine:
    """Hybrid search using Tantivy BM25 + ChromaDB semantic (HNSW)."""

    def __init__(self, bm25: TantivyBM25, collection,
                 embedding_fn, cache_ttl: int = 60):
        self.bm25 = bm25
        self.collection = collection
        self.embedding_fn = embedding_fn
        self.cache = QueryCache(ttl_seconds=cache_ttl)
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _sem_search(self, query: str, n_results: int,
                    novel_filter: Optional[str], genre_filter: Optional[str]) -> List[Dict[str, Any]]:
        if not self.embedding_fn:
            return []
        try:
            where = {}
            if novel_filter:
                where["title"] = novel_filter
            if genre_filter:
                where["genre"] = genre_filter
            params = {"query_texts": [query], "n_results": n_results * 3,
                      "include": ["documents", "metadatas", "distances"]}
            if where:
                params["where"] = where
            results = self.collection.query(**params)
            output = []
            if results and results["documents"] and results["documents"][0]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
                dists = results["distances"][0] if results["distances"] else [1.0] * len(docs)
                for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                    output.append({
                        "text": doc,
                        "metadata": meta,
                        "relevance": round(1.0 - float(dist), 4),
                        "source": f"{meta.get('title','')} - {meta.get('author','')} [{meta.get('chapter_title','')}]"
                    })
            return output[:n_results * 3]
        except Exception as e:
            from webnovel_kb.utils.logging_config import get_logger
            get_logger("engines.hybrid").warning(f"Semantic search failed: {e}")
            return []

    def _bm25_search(self, query: str, n_results: int,
                     novel_filter: Optional[str], genre_filter: Optional[str]) -> List[Dict[str, Any]]:
        results = self.bm25.search(query, n_results * 3, novel_filter, genre_filter)
        # BM25 索引存的是分词后文本，需要回 ChromaDB 取原始文本
        output = []
        chunk_ids = [r.chunk_id for r in results if r.chunk_id]
        original_texts = {}
        if chunk_ids and self.collection:
            try:
                unique_ids = list(set(chunk_ids))
                fetched = self.collection.get(ids=unique_ids, include=["documents"])
                if fetched and fetched.get("ids"):
                    for cid, doc in zip(fetched["ids"], fetched["documents"]):
                        original_texts[cid] = doc
            except Exception as e:
                from webnovel_kb.utils.logging_config import get_logger
                get_logger("engines.hybrid").warning(f"Failed to fetch original texts from ChromaDB: {e}")
        for r in results:
            text = original_texts.get(r.chunk_id, r.text)
            output.append({
                "text": text,
                "metadata": r.metadata,
                "bm25_score": r.score,
                "source": r.source
            })
        return output

    def search(self, query: str, query_vector=None, n_results: int = 10,
               alpha: float = 0.6, novel_filter: Optional[str] = None,
               genre_filter: Optional[str] = None) -> List[dict]:
        filters = {"novel": novel_filter, "genre": genre_filter}
        
        cached = self.cache.get(query, "hybrid", filters)
        if cached:
            return cached
        
        sem_future = self._executor.submit(
            self._sem_search, query, n_results * 3, novel_filter, genre_filter
        )
        bm25_future = self._executor.submit(
            self._bm25_search, query, n_results * 3, novel_filter, genre_filter
        )
        
        sem_results = sem_future.result()
        bm25_results = bm25_future.result()
        
        k = 60
        rrf_scores: Dict[str, dict] = {}
        
        for rank, r in enumerate(sem_results):
            key = r["metadata"].get("chunk_id", "") or \
                  f"{r['metadata'].get('novel_id','')}_{r['metadata'].get('chunk_index',0)}"
            if key not in rrf_scores:
                rrf_scores[key] = {"data": r, "sem_score": r.get("relevance", 0), "bm25_score": 0}
            rrf_scores[key]["sem_rank"] = rank + 1
        
        for rank, r in enumerate(bm25_results):
            key = r["metadata"].get("chunk_id", "") or \
                  f"{r['metadata'].get('novel_id','')}_{r['metadata'].get('chunk_index',0)}"
            if key not in rrf_scores:
                rrf_scores[key] = {"data": r, "sem_score": 0, "bm25_score": r.get("bm25_score", 0)}
            rrf_scores[key]["bm25_rank"] = rank + 1
        
        for key in rrf_scores:
            item = rrf_scores[key]
            sem_rrf = 1.0 / (k + item.get("sem_rank", k * 3)) if "sem_rank" in item else 0
            bm25_rrf = 1.0 / (k + item.get("bm25_rank", k * 3)) if "bm25_rank" in item else 0
            item["hybrid_score"] = round(alpha * sem_rrf + (1 - alpha) * bm25_rrf, 6)
        
        sorted_items = sorted(rrf_scores.values(), key=lambda x: x["hybrid_score"], reverse=True)
        
        output = []
        for item in sorted_items[:n_results]:
            r = item["data"]
            output.append({
                "text": r["text"],
                "metadata": r["metadata"],
                "relevance": item.get("sem_score", 0),
                "bm25_score": item.get("bm25_score", 0),
                "hybrid_score": item["hybrid_score"],
                "source": r.get("source", "")
            })
        
        self.cache.set(query, "hybrid", filters, output)
        return output
