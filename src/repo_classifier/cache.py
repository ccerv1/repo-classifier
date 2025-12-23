"""
TTL-based in-memory cache for GitHub data and embeddings.
"""

import time
from typing import Any


class TTLCache:
    """
    Simple time-to-live cache implementation.
    
    Keys expire after a configurable TTL (default 1 hour).
    Thread-safe for basic operations.
    """

    def __init__(self, default_ttl: int = 3600):
        """
        Initialize cache with default TTL.
        
        Args:
            default_ttl: Default time-to-live in seconds (default: 3600 = 1 hour)
        """
        self._cache: dict[str, tuple[Any, float]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """
        Get value from cache if it exists and hasn't expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None

        value, expires_at = self._cache[key]
        
        if time.time() > expires_at:
            # Entry has expired, remove it
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Store value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + ttl
        self._cache[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        """
        Remove key from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if key was removed, False if it didn't exist
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries from cache."""
        self._cache.clear()

    def clear_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, (_, expires_at) in self._cache.items()
            if now > expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def size(self) -> int:
        """Return number of entries in cache (including expired)."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None


# Global cache instances
github_cache = TTLCache()  # For GitHub API responses
embedding_cache = TTLCache()  # For OpenAI embeddings

