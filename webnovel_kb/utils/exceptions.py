"""Custom exception hierarchy for webnovel_kb."""


class WebNovelError(Exception):
    """Base exception for all webnovel-kb errors."""

    def __init__(self, message: str = "", detail: str = ""):
        self.detail = detail
        super().__init__(message)


class ConfigError(WebNovelError):
    """Configuration error (missing API keys, invalid settings)."""


class IngestError(WebNovelError):
    """Novel ingestion error (file not found, parse failure)."""


class SearchError(WebNovelError):
    """Search execution error (index not ready, query failure)."""


class ExtractionError(WebNovelError):
    """LLM extraction error (API failure, parse error)."""


class IndexError_(WebNovelError):
    """Index build/load error (BM25, FAISS, ChromaDB)."""


class CacheError(WebNovelError):
    """Cache read/write error (pickle, file I/O)."""


class APIError(WebNovelError):
    """Remote API call error (embedding, rerank, chat)."""

    def __init__(self, message: str = "", detail: str = "", status_code: int = 0):
        self.status_code = status_code
        super().__init__(message, detail)
