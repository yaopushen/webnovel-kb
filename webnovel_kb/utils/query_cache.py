"""TTL-based LRU query result cache."""
import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class QueryCache:
    """Thread-safe LRU cache with TTL for search results.

    Args:
        max_size: Maximum number of cached entries.
        ttl_seconds: Time-to-live for each entry in seconds (0 = no expiry).
    """

    def __init__(self, max_size: int = 256, ttl_seconds: int = 300):
        self.max_size = max(1, max_size)
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(query: str, **kwargs) -> str:
        """Generate a deterministic cache key from query and parameters."""
        parts = [query]
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if v is None or v == "" or v is False or v == 0:
                continue
            parts.append(f"{k}={v}")
        raw = "|".join(parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get a cached result by key. Returns None on miss or expiry."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            ts, value = entry
            if self.ttl > 0 and (time.time() - ts) > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        """Store a result in the cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), value)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }
