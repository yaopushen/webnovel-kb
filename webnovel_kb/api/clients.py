"""API clients for remote services."""
import hashlib
import pickle
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests

logger = logging.getLogger("webnovel-kb")


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


class RemoteEmbeddingFunction:
    """远程嵌入服务客户端。"""
    
    def __init__(self, api_url: str, api_key: str, model: str = "BAAI/bge-small-zh-v1.5",
                 dimensions: int = 512, cache_path: Optional[str] = None):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = {}
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self._cache = pickle.load(f)
                for k, v in list(self._cache.items()):
                    if isinstance(v, list):
                        self._cache[k] = np.array(v, dtype=np.float32)
                logger.info(f"Loaded embedding cache: {len(self._cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._cache:
                emb = self._cache[cache_key]
                if isinstance(emb, np.ndarray):
                    result.append(emb.tolist())
                else:
                    result.append(emb)
            else:
                result.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if uncached_texts:
            new_embeddings = self._batch_embed(uncached_texts)
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                result[idx] = emb
                cache_key = hashlib.md5(text.encode()).hexdigest()
                self._cache[cache_key] = np.array(emb, dtype=np.float32)
            
            if self.cache_path:
                try:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.cache_path.with_suffix('.pkl.tmp'), "wb") as f:
                        pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                    self.cache_path.with_suffix('.pkl.tmp').replace(self.cache_path)
                except Exception as e:
                    logger.warning(f"Failed to save embedding cache: {e}")
        
        return result
    
    def _batch_embed(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        url = f"{self.api_url.rstrip('/')}/embeddings"
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                resp = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={"input": batch, "model": self.model, "dimensions": self.dimensions},
                    timeout=60
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    data.sort(key=lambda x: x.get("index", 0))
                    all_embeddings.extend(item["embedding"] for item in data)
                else:
                    logger.error(f"Embedding API error: {resp.status_code} - {resp.text}")
                    all_extensions.extend([[0.0] * self.dimensions] * len(batch))
            except Exception as e:
                logger.error(f"Embedding request failed: {e}")
                all_embeddings.extend([[0.0] * self.dimensions] * len(batch))
        return all_embeddings


class RemoteReranker:
    """远程重排序服务客户端。"""
    
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
    
    def rerank(self, query: str, documents: List[str], top_n: int = 10) -> List[dict]:
        if not documents:
            return []
        url = f"{self.api_url.rstrip('/')}/rerank"
        try:
            resp = requests.post(
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
                timeout=30
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
    """远程对话服务客户端。"""
    
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
    
    def chat(self, messages: List[dict], temperature: float = 0.7, 
             max_tokens: int = 4096) -> Optional[str]:
        url = f"{self.api_url.rstrip('/')}/chat/completions"
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=120
            )
            if resp.status_code == 200:
                choices = resp.json().get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
            else:
                logger.error(f"Chat API error: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            return None
