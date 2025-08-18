"""
Cache Strategy Pattern Implementation for Pokemon Gym Factory.

Provides production-grade caching strategies with clean dependency injection,
following SOLID principles for testable and maintainable architecture.

Author: Bot Dean - Production-First Engineering
"""

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class CacheStrategy(ABC):
    """
    Abstract base class for caching strategies.

    Defines the contract for all cache implementations, enabling clean
    dependency injection and strategy pattern usage throughout the system.

    Production Requirements:
    - Thread-safe operations for concurrent access
    - Configurable TTL (Time To Live) support
    - Observable cache performance metrics
    - Graceful failure handling
    """

    @abstractmethod
    def get(self, key: str) -> dict[str, Any] | None:
        """
        Retrieve value from cache.

        Args:
            key: Cache key identifier

        Returns:
            Cached value or None if not found/expired

        Raises:
            CacheError: On cache operation failure
        """
        pass

    @abstractmethod
    def set(self, key: str, value: dict[str, Any], ttl_seconds: float | None = None) -> bool:
        """
        Store value in cache with optional TTL.

        Args:
            key: Cache key identifier
            value: Value to cache (must be serializable)
            ttl_seconds: Time to live in seconds (None = default TTL)

        Returns:
            True if successfully stored, False otherwise

        Raises:
            CacheError: On cache operation failure
        """
        pass

    @abstractmethod
    def clear(self, key: str | None = None) -> bool:
        """
        Clear cache entries.

        Args:
            key: Specific key to clear, or None to clear all

        Returns:
            True if successfully cleared, False otherwise

        Raises:
            CacheError: On cache operation failure
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache metrics for monitoring
        """
        pass


class CacheError(Exception):
    """
    Exception raised for cache operation failures.

    Provides actionable error messages for production debugging.
    """

    pass


class InMemoryCache(CacheStrategy):
    """
    Thread-safe in-memory cache implementation.

    Production features:
    - Thread-safe using RLock for high-performance concurrent access
    - Configurable default TTL with per-key override support
    - Automatic expiration handling
    - Performance metrics for monitoring
    - Memory-efficient cleanup of expired entries
    """

    def __init__(self, default_ttl_seconds: float = 300, max_size: int = 1000):
        """
        Initialize in-memory cache.

        Args:
            default_ttl_seconds: Default TTL for cache entries (5 minutes)
            max_size: Maximum number of entries before LRU eviction
        """
        self.default_ttl = default_ttl_seconds
        self.max_size = max_size

        # Thread-safe storage with expiration metadata
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()  # Reentrant for nested operations

        # Performance metrics
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0

        logger.info(
            f"Initialized InMemoryCache with TTL={default_ttl_seconds}s, max_size={max_size}"
        )

    def get(self, key: str) -> dict[str, Any] | None:
        """
        Get value from in-memory cache with automatic expiration.

        Thread-safe implementation with cleanup of expired entries.
        """
        try:
            with self._lock:
                if key not in self._cache:
                    self._misses += 1
                    logger.debug(f"Cache miss for key: {key}")
                    return None

                entry = self._cache[key]
                current_time = time.time()

                # Check expiration
                if current_time >= entry["expires_at"]:
                    # Expired - clean up and return None
                    del self._cache[key]
                    self._misses += 1
                    logger.debug(f"Cache expired for key: {key}")
                    return None

                # Valid entry - update access time for LRU
                entry["last_accessed"] = current_time
                self._hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return entry["value"]

        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            raise CacheError(f"Cache get operation failed: {e}") from e

    def set(self, key: str, value: dict[str, Any], ttl_seconds: float | None = None) -> bool:
        """
        Set value in in-memory cache with TTL and size management.

        Implements LRU eviction when max_size is exceeded.
        """
        try:
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            current_time = time.time()
            expires_at = current_time + ttl

            with self._lock:
                # Check if we need to evict entries
                if len(self._cache) >= self.max_size and key not in self._cache:
                    self._evict_lru_entry()

                # Store the entry with metadata
                self._cache[key] = {
                    "value": value,
                    "expires_at": expires_at,
                    "created_at": current_time,
                    "last_accessed": current_time,
                }

                self._sets += 1
                logger.debug(f"Cached key: {key} with TTL={ttl}s")
                return True

        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            raise CacheError(f"Cache set operation failed: {e}") from e

    def clear(self, key: str | None = None) -> bool:
        """
        Clear cache entries with thread safety.
        """
        try:
            with self._lock:
                if key is None:
                    # Clear all entries
                    entries_cleared = len(self._cache)
                    self._cache.clear()
                    logger.info(f"Cleared all cache entries ({entries_cleared} items)")
                else:
                    # Clear specific key
                    if key in self._cache:
                        del self._cache[key]
                        logger.info(f"Cleared cache entry for key: {key}")
                    else:
                        logger.debug(f"Key not found for clearing: {key}")

                return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise CacheError(f"Cache clear operation failed: {e}") from e

    def get_stats(self) -> dict[str, Any]:
        """
        Get comprehensive cache performance statistics.
        """
        try:
            with self._lock:
                total_requests = self._hits + self._misses
                hit_ratio = self._hits / max(total_requests, 1)

                # Count valid vs expired entries
                current_time = time.time()
                valid_entries = 0
                expired_entries = 0

                for entry in self._cache.values():
                    if current_time < entry["expires_at"]:
                        valid_entries += 1
                    else:
                        expired_entries += 1

                return {
                    "strategy": "InMemoryCache",
                    "total_entries": len(self._cache),
                    "valid_entries": valid_entries,
                    "expired_entries": expired_entries,
                    "hits": self._hits,
                    "misses": self._misses,
                    "sets": self._sets,
                    "evictions": self._evictions,
                    "hit_ratio": hit_ratio,
                    "max_size": self.max_size,
                    "default_ttl": self.default_ttl,
                }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"strategy": "InMemoryCache", "error": str(e)}

    def _evict_lru_entry(self) -> None:
        """
        Evict least recently used entry to make space.

        Internal method called when max_size is reached.
        """
        if not self._cache:
            return

        # Find LRU entry by last_accessed timestamp
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k]["last_accessed"])

        del self._cache[lru_key]
        self._evictions += 1
        logger.debug(f"Evicted LRU cache entry: {lru_key}")


class NullCache(CacheStrategy):
    """
    No-op cache implementation for testing and cache-disabled scenarios.

    Provides the same interface as other cache strategies but performs
    no caching operations. Useful for:
    - Testing scenarios where caching should be disabled
    - Performance testing to isolate cache impact
    - Debugging cache-related issues
    - Development environments
    """

    def __init__(self):
        """Initialize null cache with no storage."""
        self._operations = 0
        logger.info("Initialized NullCache (no-op)")

    def get(self, key: str) -> dict[str, Any] | None:
        """Always return None (cache miss)."""
        self._operations += 1
        logger.debug(f"NullCache get (no-op): {key}")
        return None

    def set(self, key: str, value: dict[str, Any], ttl_seconds: float | None = None) -> bool:
        """Always return True but store nothing."""
        self._operations += 1
        logger.debug(f"NullCache set (no-op): {key}")
        return True

    def clear(self, key: str | None = None) -> bool:
        """Always return True but clear nothing."""
        self._operations += 1
        logger.debug("NullCache clear (no-op)")
        return True

    def get_stats(self) -> dict[str, Any]:
        """Return stats showing null cache behavior."""
        return {
            "strategy": "NullCache",
            "operations": self._operations,
            "note": "No caching performed - all operations are no-ops",
        }


class RedisCache(CacheStrategy):
    """
    Redis-backed cache implementation for production distributed systems.

    Provides persistent, distributed caching with Redis backend.
    Currently a stub implementation for future development.

    Production features (planned):
    - Redis cluster support for high availability
    - Connection pooling for performance
    - Automatic serialization/deserialization
    - Circuit breaker pattern for Redis failures
    - Fallback to local cache on Redis unavailability
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl_seconds: float = 300):
        """
        Initialize Redis cache (stub implementation).

        Args:
            redis_url: Redis connection URL
            default_ttl_seconds: Default TTL for cache entries
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl_seconds
        self._operations = 0

        # TODO: Initialize Redis connection with production patterns:
        # - Connection pooling
        # - Circuit breaker
        # - Health monitoring
        # - Failover to local cache

        logger.warning("RedisCache is a stub implementation - not production ready")
        logger.info(f"Initialized RedisCache stub with URL={redis_url}, TTL={default_ttl_seconds}s")

    def get(self, key: str) -> dict[str, Any] | None:
        """Get value from Redis (stub - always returns None)."""
        self._operations += 1
        logger.debug(f"RedisCache get (stub): {key}")
        # TODO: Implement Redis GET with error handling
        return None

    def set(self, key: str, value: dict[str, Any], ttl_seconds: float | None = None) -> bool:
        """Set value in Redis (stub - always returns True)."""
        self._operations += 1
        logger.debug(f"RedisCache set (stub): {key}")
        # TODO: Implement Redis SET with TTL and error handling
        return True

    def clear(self, key: str | None = None) -> bool:
        """Clear Redis entries (stub - always returns True)."""
        self._operations += 1
        logger.debug("RedisCache clear (stub)")
        # TODO: Implement Redis DEL/FLUSHDB with error handling
        return True

    def get_stats(self) -> dict[str, Any]:
        """Return stats showing Redis stub behavior."""
        return {
            "strategy": "RedisCache",
            "status": "stub_implementation",
            "operations": self._operations,
            "redis_url": self.redis_url,
            "default_ttl": self.default_ttl,
            "note": "Stub implementation - not production ready",
        }


def create_cache_strategy(strategy_type: str = "auto", **kwargs) -> CacheStrategy:
    """
    Factory function to create appropriate cache strategy.

    Provides environment-aware cache strategy selection with
    production defaults and configuration flexibility.

    Args:
        strategy_type: Strategy type - "auto", "memory", "null", "redis"
        **kwargs: Strategy-specific configuration parameters

    Returns:
        Configured cache strategy instance

    Raises:
        ValueError: On invalid strategy type or configuration

    Examples:
        # Auto-detection based on environment
        cache = create_cache_strategy()

        # Explicit memory cache with custom TTL
        cache = create_cache_strategy("memory", default_ttl_seconds=600)

        # Null cache for testing
        cache = create_cache_strategy("null")

        # Redis cache with custom URL
        cache = create_cache_strategy("redis", redis_url="redis://prod-cluster:6379")
    """
    logger.info(f"Creating cache strategy: {strategy_type}")

    if strategy_type == "auto":
        # Environment-aware auto-selection
        cache_strategy_env = os.getenv("CACHE_STRATEGY")
        if cache_strategy_env:
            strategy_type = cache_strategy_env.lower()
        elif os.getenv("REDIS_URL"):
            strategy_type = "redis"
        elif os.getenv("DISABLE_CACHE"):
            strategy_type = "null"
        else:
            # Default to in-memory for most use cases
            strategy_type = "memory"

        logger.info(f"Auto-selected cache strategy: {strategy_type}")

    if strategy_type == "memory":
        return InMemoryCache(**kwargs)
    elif strategy_type == "null":
        return NullCache()
    elif strategy_type == "redis":
        # Pass REDIS_URL from environment if not explicitly provided
        if "redis_url" not in kwargs and os.getenv("REDIS_URL"):
            kwargs["redis_url"] = os.getenv("REDIS_URL")
        return RedisCache(**kwargs)
    else:
        raise ValueError(f"Unknown cache strategy: {strategy_type}")
