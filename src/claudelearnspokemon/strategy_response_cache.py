"""
Strategy Response Cache

High-performance caching system for Opus strategy responses.
Implements LRU eviction, TTL expiration, and thread-safe operations.
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .opus_strategist_exceptions import CacheError
from .strategy_response import StrategyResponse

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL and access tracking."""

    response: StrategyResponse
    created_at: float
    last_accessed: float
    ttl_seconds: float
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has exceeded TTL."""
        return (time.time() - self.created_at) > self.ttl_seconds

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class ResponseCache:
    """
    Thread-safe LRU cache with TTL for strategy responses.

    Designed for production use with performance monitoring,
    bounded memory usage, and automatic cleanup.
    
    Important: Call shutdown() when the cache is no longer needed
    to prevent thread leaks. The background cleanup thread will
    continue running until explicitly stopped.
    
    Example:
        cache = ResponseCache(max_size=100)
        # ... use cache ...
        cache.shutdown()  # Important: prevents thread leak
    """

    def __init__(
        self,
        max_size: int = 100,
        default_ttl: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0,  # 1 minute
    ):
        """
        Initialize cache with production configuration.

        Args:
            max_size: Maximum number of cached responses
            default_ttl: Default TTL for cached entries (seconds)
            cleanup_interval: How often to run TTL cleanup (seconds)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()  # Simple lock for better performance (Issue #190)
        self._shutdown_event = threading.Event()  # Signal for clean shutdown
        self._metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "ttl_evictions": 0,
            "size_evictions": 0,
            "put_operations": 0,
            "cleanup_runs": 0,
        }

        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._background_cleanup, daemon=True, name="strategy-cache-cleanup"
        )
        self._cleanup_thread.start()

        logger.info(f"ResponseCache initialized: max_size={max_size}, ttl={default_ttl}s")

    def put(self, cache_key: str, response: StrategyResponse, ttl: float | None = None) -> None:
        """
        Store response in cache with optional TTL override.

        Args:
            cache_key: Unique key for caching
            response: Strategy response to cache
            ttl: Optional TTL override (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl

        try:
            with self._lock:
                current_time = time.time()

                # Create cache entry
                entry = CacheEntry(
                    response=response,
                    created_at=current_time,
                    last_accessed=current_time,
                    ttl_seconds=ttl,
                )

                # If key exists, update it (moves to end in OrderedDict)
                if cache_key in self._cache:
                    del self._cache[cache_key]

                self._cache[cache_key] = entry
                self._metrics["put_operations"] += 1

                # Evict LRU entries if over size limit
                while len(self._cache) > self.max_size:
                    lru_key, _ = self._cache.popitem(last=False)  # Remove first (LRU)
                    self._metrics["evictions"] += 1
                    self._metrics["size_evictions"] += 1
                    logger.debug(f"Evicted LRU entry: {lru_key}")

        except Exception as e:
            logger.error(f"Cache put failed for key {cache_key}: {str(e)}")
            raise CacheError(f"Failed to store response in cache: {str(e)}", "PUT") from e

    def get(self, cache_key: str) -> StrategyResponse | None:
        """
        Retrieve response from cache if present and not expired.

        Args:
            cache_key: Cache key to retrieve

        Returns:
            Cached response or None if not found/expired
        """
        try:
            with self._lock:
                entry = self._cache.get(cache_key)

                if entry is None:
                    self._metrics["misses"] += 1
                    return None

                # Check TTL expiration
                if entry.is_expired:
                    del self._cache[cache_key]
                    self._metrics["misses"] += 1
                    self._metrics["ttl_evictions"] += 1
                    logger.debug(f"Cache entry expired: {cache_key}")
                    return None

                # Update access time and move to end (most recently used)
                entry.touch()
                self._cache.move_to_end(cache_key)

                self._metrics["hits"] += 1
                return entry.response

        except Exception as e:
            logger.error(f"Cache get failed for key {cache_key}: {str(e)}")
            # Don't raise on get failures - just return None
            self._metrics["misses"] += 1
            return None

    def invalidate(self, cache_key: str) -> bool:
        """
        Remove specific entry from cache.

        Args:
            cache_key: Key to invalidate

        Returns:
            True if entry was present and removed
        """
        try:
            with self._lock:
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    logger.debug(f"Invalidated cache entry: {cache_key}")
                    return True
                return False

        except Exception as e:
            logger.error(f"Cache invalidation failed for key {cache_key}: {str(e)}")
            raise CacheError(f"Failed to invalidate cache entry: {str(e)}", "INVALIDATE") from e

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        try:
            with self._lock:
                count = len(self._cache)
                self._cache.clear()
                logger.info(f"Cache cleared: {count} entries removed")
                return count

        except Exception as e:
            logger.error(f"Cache clear failed: {str(e)}")
            raise CacheError(f"Failed to clear cache: {str(e)}", "CLEAR") from e

    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._metrics["hits"] + self._metrics["misses"]
            hit_rate = (self._metrics["hits"] / total_requests) if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "hits": self._metrics["hits"],
                "misses": self._metrics["misses"],
                "total_requests": total_requests,
                "evictions": self._metrics["evictions"],
                "ttl_evictions": self._metrics["ttl_evictions"],
                "size_evictions": self._metrics["size_evictions"],
                "put_operations": self._metrics["put_operations"],
                "cleanup_runs": self._metrics["cleanup_runs"],
            }

    def shutdown(self) -> None:
        """
        Gracefully shutdown the cache and background thread.
        
        This method should be called when the cache is no longer needed
        to ensure proper resource cleanup and prevent thread leaks.
        """
        logger.debug("Shutting down ResponseCache...")
        self._shutdown_event.set()
        
        if self._cleanup_thread.is_alive():
            # Give the thread a reasonable time to finish
            self._cleanup_thread.join(timeout=1.0)
            if self._cleanup_thread.is_alive():
                logger.warning("Background cleanup thread did not shutdown gracefully")
        
        logger.debug("ResponseCache shutdown complete")

    def __del__(self) -> None:
        """Ensure cleanup thread is stopped when cache is destroyed."""
        try:
            self.shutdown()
        except Exception:
            # Avoid raising exceptions in __del__
            pass

    def _background_cleanup(self) -> None:
        """Background thread for periodic TTL cleanup."""
        while not self._shutdown_event.is_set():
            try:
                # Use event.wait() instead of time.sleep() for responsive shutdown
                if self._shutdown_event.wait(timeout=self.cleanup_interval):
                    # Shutdown was requested
                    break
                
                # Only run cleanup if not shutting down
                if not self._shutdown_event.is_set():
                    self._cleanup_expired_entries()

            except Exception as e:
                logger.error(f"Background cleanup error: {str(e)}")
                # Continue running even if cleanup fails, but respect shutdown
                continue
        
        logger.debug("Background cleanup thread shutting down")

    def _cleanup_expired_entries(self) -> None:
        """Remove expired entries from cache."""
        try:
            with self._lock:
                expired_keys = []
                current_time = time.time()

                for key, entry in list(self._cache.items()):  # Safe iteration snapshot (Issue #239)
                    if (current_time - entry.created_at) > entry.ttl_seconds:
                        expired_keys.append(key)

                # Remove expired entries
                for key in expired_keys:
                    if key in self._cache:  # TOCTOU protection (Issue #239)
                        del self._cache[key]
                        self._metrics["ttl_evictions"] += 1
                        self._metrics["evictions"] += 1  # Fix: Update total evictions counter

                if expired_keys:
                    logger.debug(f"Cleanup removed {len(expired_keys)} expired entries")

                self._metrics["cleanup_runs"] += 1

        except Exception as e:
            logger.error(f"TTL cleanup failed: {str(e)}")

    @staticmethod
    def generate_cache_key(game_state: dict[str, Any]) -> str:
        """
        Generate deterministic cache key from game state.

        Args:
            game_state: Current game state dictionary

        Returns:
            SHA-256 hash of normalized game state
        """
        # Normalize game state for consistent hashing
        normalized_state = {
            "location": game_state.get("location", "unknown"),
            "health": game_state.get("health", 0),
            "level": game_state.get("level", 1),
            "pokemon_count": game_state.get("pokemon_count", 0),
            "badges": game_state.get("badges", 0),
            "items": sorted(game_state.get("items", [])),  # Sort for consistency
        }

        # Create deterministic string representation
        state_str = json.dumps(normalized_state, sort_keys=True, separators=(",", ":"))

        # Generate SHA-256 hash
        return hashlib.sha256(state_str.encode("utf-8")).hexdigest()[
            :32
        ]  # 32 chars for readability
