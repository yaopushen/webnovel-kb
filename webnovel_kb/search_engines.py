"""
Optimized search engines: Tantivy for BM25, FAISS for vectors.
"""
import os
import time
import threading
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import jieba

try:
    import tantivy
    TANTIVY_AVAILABLE = True
except ImportError:
    TANTIVY_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


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
            doc = self._searcher.doc(hit.doc_address)
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
                score=hit.score,
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


class FAISSVectorStore:
    def __init__(self, index_path: Path, dimensions: int = 4096, use_mmap: bool = True):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.use_mmap = use_mmap
        self._index: Optional[faiss.Index] = None
        self._id_map: Dict[int, str] = {}
        self._id_to_meta: Dict[str, dict] = {}
        self._id_to_text: Dict[str, str] = {}
        self._next_id = 0
        self._lock = threading.RLock()
        
        self._meta_path = self.index_path.with_suffix(".meta.json")
    
    def build_index(self, vectors: np.ndarray, chunk_ids: List[str],
                    texts: List[str], metas: List[dict]):
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not installed. Run: pip install faiss-cpu")
        
        with self._lock:
            n = len(chunk_ids)
            assert vectors.shape[0] == n
            assert vectors.shape[1] == self.dimensions
            
            quantizer = faiss.IndexFlatIP(self.dimensions)
            self._index = faiss.IndexIDMap(quantizer)
            
            ids = np.arange(n, dtype=np.int64)
            self._index.add_with_ids(vectors.astype(np.float32), ids)
            
            self._id_map = {i: cid for i, cid in enumerate(chunk_ids)}
            self._id_to_meta = {cid: m for cid, m in zip(chunk_ids, metas)}
            self._id_to_text = {cid: t for cid, t in zip(chunk_ids, texts)}
            self._next_id = n
            
            self._save_index()
    
    def _save_index(self):
        faiss.write_index(self._index, str(self.index_path))
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "id_map": self._id_map,
                "next_id": self._next_id,
                "id_to_meta": self._id_to_meta,
                "id_to_text": self._id_to_text
            }, f, ensure_ascii=False)
    
    def load_index(self):
        if not self.index_path.exists():
            return False
        
        with self._lock:
            flags = faiss.IO_FLAG_MMAP if self.use_mmap else 0
            self._index = faiss.read_index(str(self.index_path), flags)
            
            if self._meta_path.exists():
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._id_map = {int(k): v for k, v in data.get("id_map", {}).items()}
                    self._next_id = data.get("next_id", 0)
                    self._id_to_meta = data.get("id_to_meta", {})
                    self._id_to_text = data.get("id_to_text", {})
            return True
    
    def search(self, query_vector: np.ndarray, n_results: int = 10,
               novel_filter: Optional[str] = None,
               genre_filter: Optional[str] = None) -> List[SearchResult]:
        if self._index is None:
            return []
        
        q = query_vector.reshape(1, -1).astype(np.float32)
        D, I = self._index.search(q, n_results * 3)
        
        output = []
        for i, (score, idx) in enumerate(zip(D[0], I[0])):
            if idx < 0:
                continue
            
            chunk_id = self._id_map.get(int(idx))
            if not chunk_id:
                continue
            
            meta = self._id_to_meta.get(chunk_id, {})
            if novel_filter and meta.get("title") != novel_filter:
                continue
            if genre_filter and meta.get("genre") != genre_filter:
                continue
            
            text = self._id_to_text.get(chunk_id, "")
            source = f"{meta.get('title', '')} - {meta.get('author', '')} [{meta.get('chapter_title', '') or 'chunk ' + str(meta.get('chunk_index', ''))}]"
            
            output.append(SearchResult(
                chunk_id=chunk_id,
                text=text,
                metadata=meta,
                score=float(score),
                source=source
            ))
            
            if len(output) >= n_results:
                break
        
        return output
    
    def add_vector(self, chunk_id: str, vector: np.ndarray, text: str, metadata: dict):
        with self._lock:
            if self._index is None:
                quantizer = faiss.IndexFlatIP(self.dimensions)
                self._index = faiss.IndexIDMap(quantizer)
            
            vec = vector.reshape(1, -1).astype(np.float32)
            idx = np.array([self._next_id], dtype=np.int64)
            self._index.add_with_ids(vec, idx)
            
            self._id_map[self._next_id] = chunk_id
            self._id_to_meta[chunk_id] = metadata
            self._id_to_text[chunk_id] = text
            self._next_id += 1
    
    def get_vector(self, chunk_id: str) -> Optional[np.ndarray]:
        for idx, cid in self._id_map.items():
            if cid == chunk_id:
                vec = np.zeros(self.dimensions, dtype=np.float32)
                self._index.reconstruct(int(idx), vec)
                return vec
        return None
    
    @property
    def count(self) -> int:
        return self._index.ntotal if self._index else 0


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
    def __init__(self, bm25: TantivyBM25, vector_store: FAISSVectorStore,
                 cache_ttl: int = 60):
        self.bm25 = bm25
        self.vector_store = vector_store
        self.cache = QueryCache(ttl_seconds=cache_ttl)
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def _sem_search(self, query_vector: np.ndarray, n_results: int,
                    novel_filter: Optional[str], genre_filter: Optional[str]) -> List[SearchResult]:
        return self.vector_store.search(query_vector, n_results, novel_filter, genre_filter)
    
    def _bm25_search(self, query: str, n_results: int,
                     novel_filter: Optional[str], genre_filter: Optional[str]) -> List[SearchResult]:
        return self.bm25.search(query, n_results, novel_filter, genre_filter)
    
    def search(self, query: str, query_vector: np.ndarray, n_results: int = 10,
               alpha: float = 0.6, novel_filter: Optional[str] = None,
               genre_filter: Optional[str] = None) -> List[dict]:
        filters = {"novel": novel_filter, "genre": genre_filter}
        
        cached = self.cache.get(query, "hybrid", filters)
        if cached:
            return cached
        
        sem_future = self._executor.submit(
            self._sem_search, query_vector, n_results * 3, novel_filter, genre_filter
        )
        bm25_future = self._executor.submit(
            self._bm25_search, query, n_results * 3, novel_filter, genre_filter
        )
        
        sem_results = sem_future.result()
        bm25_results = bm25_future.result()
        
        k = 60
        rrf_scores: Dict[str, dict] = {}
        
        for rank, r in enumerate(sem_results):
            key = r.chunk_id
            if key not in rrf_scores:
                rrf_scores[key] = {"data": r, "sem_score": r.score, "bm25_score": 0}
            rrf_scores[key]["sem_rank"] = rank + 1
        
        for rank, r in enumerate(bm25_results):
            key = r.chunk_id
            if key not in rrf_scores:
                rrf_scores[key] = {"data": r, "sem_score": 0, "bm25_score": r.score}
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
                "text": r.text,
                "metadata": r.metadata,
                "relevance": item.get("sem_score", 0),
                "bm25_score": item.get("bm25_score", 0),
                "hybrid_score": item["hybrid_score"],
                "source": r.source
            })
        
        self.cache.set(query, "hybrid", filters, output)
        return output
