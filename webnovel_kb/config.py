"""Configuration management for webnovel_kb project."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("WEBNOVEL_KB_DATA", str(Path(__file__).parent.parent / "data")))

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_CHAT_BASE_URL = os.environ.get("LLM_CHAT_BASE_URL", "")
LLM_EMBEDDING_MODEL = os.environ.get("LLM_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
LLM_RERANK_MODEL = os.environ.get("LLM_RERANK_MODEL", "")
LLM_EMBEDDING_DIMENSIONS = int(os.environ.get("LLM_EMBEDDING_DIMENSIONS", "512"))
LLM_CHAT_MODEL = os.environ.get("LLM_CHAT_MODEL", "")

EMBEDDING_CACHE_PATH = os.environ.get("EMBEDDING_CACHE_PATH", str(DATA_DIR / "embeddings_cache.pkl"))
