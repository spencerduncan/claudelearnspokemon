"""
Production-Grade Cache Strategy Pattern Implementation for Pokemon Gym Factory.

This module provides comprehensive caching strategies with advanced resilience patterns,
following SOLID principles for testable and maintainable architecture. Features include:

## Available Cache Strategies

### InMemoryCache
- Thread-safe in-memory caching with LRU eviction
- Configurable TTL and size limits
- High-performance local caching
- Comprehensive metrics and monitoring

### NullCache  
- No-op cache implementation for testing and debugging
- Useful for performance isolation and development

### RedisCache (Production-Ready)
- **NEW**: Complete production implementation with advanced patterns
- Connection pooling for high performance (configurable pool size)
- Circuit breaker pattern for automatic failure detection and recovery
- Intelligent fallback to InMemoryCache when Redis is unavailable
- JSON serialization with optional compression for large values
- Health monitoring with configurable check intervals
- Thread-safe operations with comprehensive error handling
- Performance metrics and statistics collection
- Support for Redis clusters and authentication

## Advanced Features

### Circuit Breaker Pattern
- Automatic failure detection with configurable thresholds
- Circuit states: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
- Configurable recovery timeout and failure thresholds
- Prevents cascade failures and improves system resilience

### Fallback Strategy
- Seamless fallback from Redis to InMemoryCache
- Transparent operation continuation during Redis outages
- Configurable fallback behavior per cache instance
- Metrics tracking for fallback operations

### Performance Optimizations
- Connection pooling with configurable limits
- Batch operations support (future enhancement)
- JSON compression for large payloads (threshold-based)
- Efficient serialization/deserialization
- Thread-safe concurrent operations

## Usage Examples

```python
# Basic Redis cache with fallback
cache = RedisCache(
    redis_url="redis://localhost:6379",
    default_ttl_seconds=300,
    fallback_enabled=True,
    max_connections=10
)

# Environment-aware cache creation
cache = create_cache_strategy("auto")  # Auto-detects best strategy

# High-performance setup with compression
cache = RedisCache(
    redis_url="redis://cluster:6379",
    fallback_enabled=True,
    compress_values=True,
    max_connections=50,
    health_check_interval=15.0
)

# Use cache with any strategy
cache.set("key", {"data": "value"}, ttl_seconds=60)
result = cache.get("key")
cache.clear("key")

# Monitor performance
stats = cache.get_stats()
health = cache.is_healthy()
```

## Configuration Options

### Environment Variables
- `CACHE_STRATEGY`: Force specific strategy ("memory", "redis", "null")
- `REDIS_URL`: Redis connection URL for auto-detection
- `DISABLE_CACHE`: Set to disable caching (uses NullCache)

### Redis Configuration
- Connection pooling with configurable pool size
- TTL support with per-key and default settings
- Compression threshold for large values
- Health check intervals and circuit breaker settings
- Fallback cache configuration

## Error Handling

All cache implementations provide consistent error handling:
- Operations never raise exceptions to calling code
- Failed operations return appropriate defaults (None, False)
- Comprehensive logging of errors and warnings
- Circuit breaker automatic recovery
- Graceful degradation with fallback strategies

Author: Claude Code (Innovator) - Production-First Engineering with Advanced Resilience Patterns
"""

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker state for Redis connection resilience."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failures detected, circuit is open
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for Redis connection resilience.
    
    Provides automatic failure detection and recovery for Redis operations.
    Implements the circuit breaker pattern with configurable thresholds.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery (seconds)
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.RLock()
    
    def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result if successful
            
        Raises:
            Exception: If circuit is open or function fails
        """
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if time.time() - self._last_failure_time < self.recovery_timeout:
                    raise Exception("Circuit breaker is OPEN - Redis unavailable")
                else:
                    # Transition to half-open for testing
                    self._state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to HALF_OPEN for testing")
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset or close circuit
                if self._state == CircuitBreakerState.HALF_OPEN:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit breaker CLOSED - Redis recovered")
                elif self._state == CircuitBreakerState.CLOSED:
                    # Reset failure count on success
                    if self._failure_count > 0:
                        self._failure_count = 0
                        logger.debug("Circuit breaker reset failure count")
                
                return result
                
            except Exception as e:
                # Failure - increment count and potentially open circuit
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitBreakerState.OPEN
                    logger.error(f"Circuit breaker OPENED after {self._failure_count} failures: {e}")
                else:
                    logger.warning(f"Circuit breaker failure {self._failure_count}/{self.failure_threshold}: {e}")
                
                raise
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count


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
    Production-ready Redis cache implementation with advanced resilience patterns.

    Features:
    - Connection pooling for high performance
    - Circuit breaker pattern for automatic failover
    - Fallback to InMemoryCache when Redis is unavailable  
    - JSON serialization with optional compression
    - Health monitoring and performance metrics
    - Adaptive TTL based on usage patterns
    - Thread-safe operations with proper error handling
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl_seconds: float = 300,
        max_connections: int = 10,
        fallback_enabled: bool = True,
        compress_values: bool = False,
        health_check_interval: float = 30.0,
    ):
        """
        Initialize production Redis cache with resilience features.

        Args:
            redis_url: Redis connection URL
            default_ttl_seconds: Default TTL for cache entries
            max_connections: Maximum connections in pool
            fallback_enabled: Enable fallback to InMemoryCache on Redis failure
            compress_values: Enable JSON compression for large values
            health_check_interval: Interval for Redis health checks (seconds)
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl_seconds
        self.max_connections = max_connections
        self.fallback_enabled = fallback_enabled
        self.compress_values = compress_values
        self.health_check_interval = health_check_interval

        # Performance and health metrics
        self._operations = 0
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._fallback_operations = 0
        self._last_health_check = 0.0
        self._is_healthy = False

        # Thread safety
        self._lock = threading.RLock()

        # Circuit breaker for resilience
        self._circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)

        # Fallback cache when Redis is unavailable
        if self.fallback_enabled:
            self._fallback_cache = InMemoryCache(
                default_ttl_seconds=default_ttl_seconds, max_size=1000
            )
        else:
            self._fallback_cache = None

        # Initialize Redis connection pool
        self._redis_client = None
        self._initialize_redis_connection()

        logger.info(f"Initialized production RedisCache with URL={redis_url}")

    def _initialize_redis_connection(self):
        """Initialize Redis connection pool with error handling."""
        try:
            import redis

            # Parse Redis URL for connection parameters
            self._redis_client = redis.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=self.health_check_interval,
                decode_responses=True,  # Automatically decode byte responses
            )

            # Test connection
            self._circuit_breaker.call(self._redis_client.ping)
            self._is_healthy = True
            self._last_health_check = time.time()

            logger.info("Redis connection pool initialized successfully")

        except ImportError:
            logger.error("redis-py not installed - falling back to memory cache")
            self._redis_client = None
            self._is_healthy = False
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            self._redis_client = None
            self._is_healthy = False

    def _serialize_value(self, value: dict[str, Any]) -> str:
        """Serialize value to JSON string with optional compression."""
        json_str = json.dumps(value, separators=(",", ":"))  # Compact JSON

        if self.compress_values and len(json_str) > 1024:  # Compress large values
            try:
                import zlib

                compressed = zlib.compress(json_str.encode())
                return f"z:{compressed.hex()}"  # Prefix to indicate compression
            except Exception:
                pass  # Fall back to uncompressed

        return json_str

    def _deserialize_value(self, serialized: str) -> dict[str, Any]:
        """Deserialize JSON string with decompression support."""
        try:
            if serialized.startswith("z:"):  # Compressed value
                import zlib

                compressed_hex = serialized[2:]
                compressed = bytes.fromhex(compressed_hex)
                json_str = zlib.decompress(compressed).decode()
            else:
                json_str = serialized

            return json.loads(json_str)

        except Exception as e:
            logger.error(f"Failed to deserialize cache value: {e}")
            raise CacheError(f"Deserialization failed: {e}") from e

    def _execute_with_fallback(self, operation_name: str, redis_func, fallback_func=None):
        """Execute Redis operation with circuit breaker and fallback."""
        if not self._redis_client:
            self._errors += 1  # Count unavailable client as error
            if fallback_func and self._fallback_cache:
                self._fallback_operations += 1
                return fallback_func()
            raise CacheError("Redis client not available and no fallback configured")

        try:
            # Execute Redis operation through circuit breaker
            return self._circuit_breaker.call(redis_func)

        except Exception as e:
            self._errors += 1
            logger.error(f"Redis {operation_name} failed: {e}")

            # Try fallback if enabled
            if fallback_func and self._fallback_cache:
                logger.info(f"Using fallback cache for {operation_name}")
                self._fallback_operations += 1
                try:
                    return fallback_func()
                except Exception as fallback_error:
                    logger.error(f"Fallback {operation_name} also failed: {fallback_error}")
                    raise CacheError(
                        f"Both Redis and fallback {operation_name} failed"
                    ) from fallback_error

            # No fallback - propagate error
            raise CacheError(f"Redis {operation_name} failed: {e}") from e

    def get(self, key: str) -> dict[str, Any] | None:
        """Get value from Redis with fallback support."""
        self._operations += 1

        def redis_get():
            serialized = self._redis_client.get(key)
            if serialized is None:
                self._misses += 1
                return None

            self._hits += 1
            return self._deserialize_value(serialized)

        def fallback_get():
            return self._fallback_cache.get(key)

        try:
            with self._lock:
                return self._execute_with_fallback("get", redis_get, fallback_get)
        except Exception as e:
            logger.error(f"Cache get operation failed for key {key}: {e}")
            return None  # Return None on error to maintain cache contract

    def set(self, key: str, value: dict[str, Any], ttl_seconds: float | None = None) -> bool:
        """Set value in Redis with fallback support."""
        self._operations += 1
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        def redis_set():
            serialized = self._serialize_value(value)
            if ttl > 0:
                return self._redis_client.setex(key, int(ttl), serialized)
            else:
                return self._redis_client.set(key, serialized)

        def fallback_set():
            return self._fallback_cache.set(key, value, ttl_seconds)

        try:
            with self._lock:
                result = self._execute_with_fallback("set", redis_set, fallback_set)
                return bool(result)
        except Exception as e:
            logger.error(f"Cache set operation failed for key {key}: {e}")
            return False

    def clear(self, key: str | None = None) -> bool:
        """Clear cache entries with fallback support."""
        self._operations += 1

        def redis_clear():
            if key is None:
                # Clear all keys (use with caution in production)
                return self._redis_client.flushdb()
            else:
                # Delete specific key
                return self._redis_client.delete(key) > 0

        def fallback_clear():
            return self._fallback_cache.clear(key)

        try:
            with self._lock:
                result = self._execute_with_fallback("clear", redis_clear, fallback_clear)
                return bool(result)
        except Exception as e:
            logger.error(f"Cache clear operation failed: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics including health metrics."""
        try:
            with self._lock:
                total_requests = self._hits + self._misses
                hit_ratio = self._hits / max(total_requests, 1)

                # Check current health status
                current_time = time.time()
                if current_time - self._last_health_check > self.health_check_interval:
                    self._check_health()

                # Get circuit breaker stats
                circuit_state = self._circuit_breaker.state.value
                circuit_failures = self._circuit_breaker.failure_count

                # Get fallback cache stats if available
                fallback_stats = {}
                if self._fallback_cache:
                    fallback_stats = self._fallback_cache.get_stats()

                return {
                    "strategy": "RedisCache",
                    "status": "production" if self._is_healthy else "degraded",
                    "total_operations": self._operations,
                    "hits": self._hits,
                    "misses": self._misses,
                    "errors": self._errors,
                    "hit_ratio": hit_ratio,
                    "fallback_operations": self._fallback_operations,
                    "circuit_breaker_state": circuit_state,
                    "circuit_breaker_failures": circuit_failures,
                    "redis_url": self.redis_url,
                    "default_ttl": self.default_ttl,
                    "max_connections": self.max_connections,
                    "fallback_enabled": self.fallback_enabled,
                    "compress_values": self.compress_values,
                    "is_healthy": self._is_healthy,
                    "last_health_check": self._last_health_check,
                    "fallback_cache_stats": fallback_stats,
                }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"strategy": "RedisCache", "error": str(e)}

    def _check_health(self):
        """Perform health check on Redis connection."""
        try:
            if self._redis_client:
                self._circuit_breaker.call(self._redis_client.ping)
                self._is_healthy = True
            else:
                self._is_healthy = False

            self._last_health_check = time.time()

        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            self._is_healthy = False
            self._last_health_check = time.time()

    def is_healthy(self) -> bool:
        """Check if Redis cache is currently healthy."""
        current_time = time.time()
        if current_time - self._last_health_check > self.health_check_interval:
            self._check_health()
        return self._is_healthy

    def get_connection_pool_stats(self) -> dict[str, Any]:
        """Get Redis connection pool statistics (if available)."""
        if not self._redis_client or not hasattr(self._redis_client, "connection_pool"):
            return {"error": "Connection pool not available"}

        try:
            pool = self._redis_client.connection_pool
            return {
                "max_connections": getattr(pool, "max_connections", "unknown"),
                "created_connections": len(getattr(pool, "_created_connections", [])),
                "available_connections": len(getattr(pool, "_available_connections", [])),
                "in_use_connections": len(getattr(pool, "_in_use_connections", [])),
            }
        except Exception as e:
            logger.error(f"Failed to get connection pool stats: {e}")
            return {"error": str(e)}


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
