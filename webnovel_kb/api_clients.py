"""API client wrappers for LLM embedding, rerank, and chat services."""
import json
import hashlib
import time
import logging
import pickle
import array
import threading
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction

from webnovel_kb.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_CHAT_BASE_URL,
    LLM_EMBEDDING_MODEL, LLM_RERANK_MODEL,
    LLM_EMBEDDING_DIMENSIONS, LLM_CHAT_MODEL,
    EMBEDDING_CACHE_PATH
)

logger = logging.getLogger("webnovel-kb")


class RemoteEmbeddingFunction(EmbeddingFunction):
    _api_semaphore = threading.Semaphore(4)

    def __init__(self, api_key: str = LLM_API_KEY, base_url: str = LLM_BASE_URL,
                 model: str = LLM_EMBEDDING_MODEL, dimensions: int = LLM_EMBEDDING_DIMENSIONS,
                 batch_size: int = 20, max_retries: int = 3, cache_path: str = ""):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._cache: dict[str, array.array] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache_dirty = False
        self._cache_lock = threading.Lock()
        self._flush_timer: threading.Timer | None = None
        self._http_client = None
        if self._cache_path and self._cache_path.exists():
            self._load_cache()

    def _get_http_client(self):
        if self._http_client is None:
            import httpx
            self._http_client = httpx.Client(
                timeout=httpx.Timeout(connect=10, read=120, write=10, pool=30),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=3),
                http2=True
            )
        return self._http_client

    def _load_cache(self):
        try:
            with open(self._cache_path, "rb") as f:
                raw = pickle.load(f)
            count = 0
            for k, v in raw.items():
                if isinstance(v, list):
                    raw[k] = array.array('f', v)
                    count += 1
            self._cache = raw
            logger.info(f"Embedding cache loaded: {len(self._cache)} entries ({count} compact)")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")

    def _save_cache(self):
        if not self._cache_path or not self._cache_dirty:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_data = {}
            for k, v in self._cache.items():
                save_data[k] = v.tolist() if isinstance(v, array.array) else v
            tmp_path = self._cache_path.with_suffix('.pkl.tmp')
            with open(tmp_path, "wb") as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(self._cache_path)
            self._cache_dirty = False
            logger.info(f"Embedding cache saved: {len(self._cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    def _schedule_flush(self):
        if self._flush_timer is not None:
            self._flush_timer.cancel()
        self._flush_timer = threading.Timer(30.0, self._do_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _do_flush(self):
        with self._cache_lock:
            if self._cache_dirty:
                self._save_cache()

    def flush(self):
        with self._cache_lock:
            self._save_cache()
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("LLM_API_KEY not set")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions
        }
        url = f"{self.base_url}/embeddings"
        with self._api_semaphore:
            client = self._get_http_client()
            for attempt in range(self.max_retries):
                try:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    if "data" in data:
                        sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
                        return [item["embedding"] for item in sorted_data]
                    else:
                        raise RuntimeError(f"API error: {data}")
                except Exception as e:
                    status = getattr(getattr(e, 'response', None), 'status_code', None)
                    if status == 429:
                        if attempt == self.max_retries - 1:
                            raise RuntimeError(
                                f"Embedding API rate-limited after {self.max_retries} retries"
                            ) from e
                        time.sleep(2 ** attempt)
                    elif status and status >= 500:
                        if attempt == self.max_retries - 1:
                            raise
                        time.sleep(2 ** attempt)
                    else:
                        if attempt == self.max_retries - 1:
                            raise
                        time.sleep(1)
        return []

    def __call__(self, input: list[str]) -> list[list[float]]:
        all_embeddings = []
        uncached_texts = []
        uncached_indices = []
        with self._cache_lock:
            for i, text in enumerate(input):
                cache_key = hashlib.md5(text.encode()).hexdigest()
                if cache_key in self._cache:
                    emb = self._cache[cache_key]
                    if isinstance(emb, array.array):
                        all_embeddings.append((i, emb.tolist()))
                    else:
                        all_embeddings.append((i, emb))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        if uncached_texts:
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[batch_start:batch_start + self.batch_size]
                batch_indices = uncached_indices[batch_start:batch_start + self.batch_size]
                embeddings = self._call_api(batch)
                with self._cache_lock:
                    for j, emb in enumerate(embeddings):
                        idx = batch_indices[j]
                        cache_key = hashlib.md5(batch[j].encode()).hexdigest()
                        self._cache[cache_key] = array.array('f', emb)
                        all_embeddings.append((idx, emb))
                self._cache_dirty = True
            self._schedule_flush()
        all_embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in all_embeddings]


class RemoteReranker:
    def __init__(self, api_key: str = LLM_API_KEY, base_url: str = LLM_BASE_URL,
                 model: str = LLM_RERANK_MODEL):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http_client = None

    def _get_http_client(self):
        if self._http_client is None:
            import httpx
            self._http_client = httpx.Client(
                timeout=httpx.Timeout(connect=10, read=30, write=10, pool=10),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                http2=True
            )
        return self._http_client

    def rerank(self, query: str, documents: list[str], top_n: int = 10) -> list[dict]:
        if not self.api_key:
            return []
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n
        }
        url = f"{self.base_url}/rerank"
        try:
            client = self._get_http_client()
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if "results" in data:
                return data["results"]
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Rerank API error: {e}")
            return []


class RemoteChatClient:
    def __init__(self, api_key: str = LLM_API_KEY, base_url: str = LLM_CHAT_BASE_URL,
                 model: str = LLM_CHAT_MODEL):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http_client = None

    def _get_http_client(self):
        if self._http_client is None:
            import httpx
            self._http_client = httpx.Client(
                timeout=httpx.Timeout(connect=10, read=240, write=10, pool=10),
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
                http2=True
            )
        return self._http_client

    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 4096) -> str:
        if not self.api_key:
            return ""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        url = f"{self.base_url}/chat/completions"
        try:
            client = self._get_http_client()
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0].get("message", {}).get("content", "")
            return ""
        except Exception as e:
            logger.warning(f"Chat API error: {e}")
            return ""


class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._use_local = True
        except Exception:
            self._use_local = False

    def __call__(self, input: list[str]) -> list[list[float]]:
        if self._use_local:
            return self.model.encode(input, show_progress_bar=False).tolist()
        else:
            dim = 512
            return [[(int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 100) / 100.0 for _ in range(dim)] for s in input]


def _create_embedding_function(cache_path: str = EMBEDDING_CACHE_PATH) -> EmbeddingFunction:
    if LLM_API_KEY and LLM_BASE_URL:
        return RemoteEmbeddingFunction(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            model=LLM_EMBEDDING_MODEL,
            dimensions=LLM_EMBEDDING_DIMENSIONS,
            cache_path=cache_path
        )
    else:
        return LocalEmbeddingFunction()
