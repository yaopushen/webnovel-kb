"""API clients for remote services — async with httpx."""
import asyncio
import hashlib
import pickle
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Dict

import httpx
import numpy as np

from webnovel_kb.utils.logging_config import get_logger
from webnovel_kb.utils.exceptions import APIError, CacheError

logger = get_logger("api.clients")


def create_embedding_function(cache_path: str = ""):
    """创建嵌入函数实例。"""
    from webnovel_kb.config import (
        LLM_API_KEY, LLM_BASE_URL,
        LLM_EMBEDDING_MODEL, LLM_EMBEDDING_DIMENSIONS
    )
    if LLM_API_KEY:
        return RemoteEmbeddingFunction(
            api_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=LLM_EMBEDDING_MODEL,
            dimensions=LLM_EMBEDDING_DIMENSIONS,
            cache_path=cache_path
        )
    else:
        return None


class SQLiteEmbeddingCache:
    """基于 SQLite3 存储的 Embedding 缓存，消除大 pkl 文件导致的内存开销。"""

    def __init__(self, cache_path: Path):
        # 兼容传入 pkl 格式的文件路径，自动转换成 .db 格式
        if cache_path.suffix == ".pkl":
            self.pkl_path = cache_path
            self.db_path = cache_path.with_suffix(".db")
        else:
            self.pkl_path = cache_path.with_suffix(".pkl")
            self.db_path = cache_path

        self._init_db()
        self._trigger_migration_if_needed()

    def _init_db(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    key TEXT PRIMARY KEY,
                    vector BLOB
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize SQLite embedding cache database: {e}")

    def _trigger_migration_if_needed(self):
        # 检查是否需要从旧的 pickle 进行迁移
        if self.pkl_path and self.pkl_path.exists():
            try:
                # 如果数据库为空，才进行迁移
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT count(*) FROM embedding_cache")
                count = cursor.fetchone()[0]
                conn.close()

                if count == 0:
                    # 在后台线程中执行迁移，避免阻塞主进程初始化
                    threading.Thread(
                        target=self._migrate_data,
                        args=(self.pkl_path, self.db_path),
                        daemon=True
                    ).start()
                else:
                    # 如果数据库不为空，但旧 pickle 还存在，直接重命名它
                    migrated_path = self.pkl_path.with_suffix(".pkl.migrated")
                    if migrated_path.exists():
                        migrated_path.unlink()
                    self.pkl_path.rename(migrated_path)
                    logger.info(f"Old pickle found but db is not empty. Renamed {self.pkl_path} to {migrated_path}")
            except Exception as e:
                logger.warning(f"Error checking migration state: {e}")

    def _migrate_data(self, pkl_path: Path, db_path: Path):
        try:
            logger.info(f"Starting embedding cache migration from {pkl_path} to {db_path}...")
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded {len(data)} cache entries from old pickle. Migrating to SQLite...")
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            batch = []
            for k, v in data.items():
                if isinstance(v, list):
                    vec = np.array(v, dtype=np.float32)
                elif isinstance(v, np.ndarray):
                    vec = v.astype(np.float32)
                else:
                    continue
                batch.append((k, vec.tobytes()))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT OR REPLACE INTO embedding_cache (key, vector) VALUES (?, ?)", batch)
                    conn.commit()
                    batch = []
            
            if batch:
                cursor.executemany("INSERT OR REPLACE INTO embedding_cache (key, vector) VALUES (?, ?)", batch)
                conn.commit()
            
            conn.close()
            logger.info("Embedding cache migration completed successfully.")
            
            migrated_path = pkl_path.with_suffix(".pkl.migrated")
            if migrated_path.exists():
                migrated_path.unlink()
            pkl_path.rename(migrated_path)
            logger.info(f"Renamed migrated pickle file to {migrated_path}")
        except Exception as e:
            logger.error(f"Failed to migrate embedding cache: {e}", exc_info=True)

    def get_many(self, keys: List[str]) -> Dict[str, np.ndarray]:
        if not keys:
            return {}
        results = {}
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in keys)
            cursor.execute(f"SELECT key, vector FROM embedding_cache WHERE key IN ({placeholders})", keys)
            for row in cursor.fetchall():
                k, v_bytes = row
                results[k] = np.frombuffer(v_bytes, dtype=np.float32)
            conn.close()
        except Exception as e:
            logger.error(f"Error reading from SQLite embedding cache: {e}")
        return results

    def set_many(self, key_vector_pairs: Dict[str, np.ndarray]):
        if not key_vector_pairs:
            return
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            data = [(k, v.tobytes()) for k, v in key_vector_pairs.items()]
            cursor.executemany("INSERT OR REPLACE INTO embedding_cache (key, vector) VALUES (?, ?)", data)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error writing to SQLite embedding cache: {e}")


class RemoteEmbeddingFunction:
    """远程嵌入服务客户端（httpx 异步）。使用 SQLite 作为轻量按需缓存。"""

    def __init__(self, api_url: str, api_key: str, model: str = "BAAI/bge-small-zh-v1.5",
                 dimensions: int = 512, cache_path: Optional[str] = None):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
        )
        self.db_cache = None
        if cache_path:
            try:
                self.db_cache = SQLiteEmbeddingCache(Path(cache_path))
            except Exception as e:
                logger.warning(f"Failed to create SQLiteEmbeddingCache: {e}")

    async def close(self):
        """关闭 httpx 客户端。"""
        await self._client.aclose()

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        """异步嵌入调用。"""
        if not texts:
            return []
        
        result = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        keys = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts]

        cached_map = {}
        if self.db_cache:
            try:
                cached_map = await asyncio.to_thread(self.db_cache.get_many, keys)
            except Exception as e:
                logger.warning(f"Failed to get cache in __call__: {e}")

        for i, (text, key) in enumerate(zip(texts, keys)):
            if key in cached_map:
                emb = cached_map[key]
                result[i] = emb.tolist()
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            new_embeddings = await self._batch_embed(uncached_texts)
            to_save = {}
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                result[idx] = emb
                key = hashlib.md5(text.encode("utf-8")).hexdigest()
                to_save[key] = np.array(emb, dtype=np.float32)

            if self.db_cache and to_save:
                try:
                    await asyncio.to_thread(self.db_cache.set_many, to_save)
                except Exception as e:
                    logger.warning(f"Failed to save cache in __call__: {e}")

        return result

    async def _batch_embed(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        url = f"{self.api_url.rstrip('/')}/embeddings"
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                resp = await self._client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={"input": batch, "model": self.model, "dimensions": self.dimensions},
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    data.sort(key=lambda x: x.get("index", 0))
                    all_embeddings.extend(item["embedding"] for item in data)
                else:
                    logger.error(f"Embedding API error: {resp.status_code} - {resp.text}")
                    all_embeddings.extend([[0.0] * self.dimensions] * len(batch))
            except Exception as e:
                logger.error(f"Embedding request failed: {e}")
                all_embeddings.extend([[0.0] * self.dimensions] * len(batch))
        return all_embeddings


class RemoteReranker:
    """远程重排序服务客户端（httpx 异步）。"""

    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
        )

    async def close(self):
        """关闭 httpx 客户端。"""
        await self._client.aclose()

    async def rerank(self, query: str, documents: List[str], top_n: int = 10) -> List[dict]:
        if not documents:
            return []
        url = f"{self.api_url.rstrip('/')}/rerank"
        try:
            resp = await self._client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n
                },
            )
            if resp.status_code == 200:
                return resp.json().get("results", [])
            else:
                logger.error(f"Rerank API error: {resp.status_code} - {resp.text}")
                return [{"index": i, "relevance_score": 0.0} for i in range(min(top_n, len(documents)))]
        except Exception as e:
            logger.error(f"Rerank request failed: {e}")
            return [{"index": i, "relevance_score": 0.0} for i in range(min(top_n, len(documents)))]


class RemoteChatClient:
    """远程对话服务客户端（httpx 异步）。"""

    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
        )

    async def close(self):
        """关闭 httpx 客户端。"""
        await self._client.aclose()

    async def _request(self, messages: List[dict], temperature: float = 0.7,
                 max_tokens: int = 4096, tools: Optional[List[dict]] = None,
                 tool_choice: str = "auto") -> Optional[dict]:
        url = f"{self.api_url.rstrip('/')}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        try:
            resp = await self._client.post(
                url,
                headers={
                    "api-key": self.api_key,
                    "Content-Type": "application/json"
                },
                json=body,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                error_detail = resp.text[:500] if resp.text else ""
                logger.error(f"Chat API error: {resp.status_code} - {error_detail}")
                return {
                    "_error": True,
                    "status_code": resp.status_code,
                    "message": f"Chat API HTTP {resp.status_code}",
                    "detail": error_detail,
                    "retry_after": resp.headers.get("Retry-After"),
                }
        except httpx.TimeoutException as e:
            logger.error(f"Chat request timeout: {e}")
            return {
                "_error": True,
                "status_code": 0,
                "message": f"Chat API 请求超时",
                "detail": str(e),
                "retry_after": None,
            }
        except httpx.ConnectError as e:
            logger.error(f"Chat request connection error: {e}")
            return {
                "_error": True,
                "status_code": 0,
                "message": f"Chat API 连接失败",
                "detail": str(e),
                "retry_after": None,
            }
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            return {
                "_error": True,
                "status_code": 0,
                "message": f"Chat API 内部错误: {type(e).__name__}",
                "detail": str(e),
                "retry_after": None,
            }

    async def chat(self, messages: List[dict], temperature: float = 0.7,
             max_tokens: int = 4096, tools: Optional[List[dict]] = None,
             tool_choice: str = "auto") -> Optional[str]:
        resp = await self._request(messages, temperature, max_tokens, tools, tool_choice)
        if resp and not resp.get("_error"):
            choices = resp.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        return None

    async def chat_raw(self, messages: List[dict], temperature: float = 0.7,
                 max_tokens: int = 4096, tools: Optional[List[dict]] = None,
                 tool_choice: str = "auto") -> Optional[dict]:
        return await self._request(messages, temperature, max_tokens, tools, tool_choice)
