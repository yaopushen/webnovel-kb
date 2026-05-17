"""Configuration management for webnovel_kb project."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("WEBNOVEL_KB_DATA", str(Path(__file__).parent.parent / "webnovel_data")))

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_CHAT_BASE_URL = os.environ.get("LLM_CHAT_BASE_URL", "")
LLM_EMBEDDING_MODEL = os.environ.get("LLM_EMBEDDING_MODEL", "")
LLM_RERANK_MODEL = os.environ.get("LLM_RERANK_MODEL", "")
LLM_EMBEDDING_DIMENSIONS = int(os.environ.get("LLM_EMBEDDING_DIMENSIONS", "512"))
LLM_CHAT_MODEL = os.environ.get("LLM_CHAT_MODEL", "")

EMBEDDING_CACHE_PATH = os.environ.get("EMBEDDING_CACHE_PATH", str(DATA_DIR / "embeddings_cache.pkl"))

LOG_LEVEL = os.environ.get("WEBNOVEL_KB_LOG_LEVEL", "INFO")
LOG_DIR = os.environ.get("WEBNOVEL_KB_LOG_DIR", str(DATA_DIR / "logs"))
LOG_FILE = os.environ.get("WEBNOVEL_KB_LOG_FILE", "webnovel-kb.log")
LOG_MAX_BYTES = int(os.environ.get("WEBNOVEL_KB_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.environ.get("WEBNOVEL_KB_LOG_BACKUP_COUNT", "5"))
LOG_CONSOLE_LEVEL = os.environ.get("WEBNOVEL_KB_LOG_CONSOLE_LEVEL", "")
LOG_FILE_LEVEL = os.environ.get("WEBNOVEL_KB_LOG_FILE_LEVEL", "")

QUERY_CACHE_SIZE = int(os.environ.get("WEBNOVEL_KB_QUERY_CACHE_SIZE", "256"))
QUERY_CACHE_TTL = int(os.environ.get("WEBNOVEL_KB_QUERY_CACHE_TTL", "300"))
