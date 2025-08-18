"""
Test suite for Cache Strategy Pattern implementations.

Tests all cache strategy implementations including thread safety, TTL behavior,
LRU eviction, and production patterns for dependency injection.

Author: Bot Dean - Production-First Engineering
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

from src.claudelearnspokemon.compatibility.cache_strategies import (
    CacheError,
    CacheStrategy,
    InMemoryCache,
    NullCache,
    RedisCache,
    create_cache_strategy,
)


@pytest.mark.fast
class TestCacheStrategyInterface:
    """Test the abstract base class interface."""

    def test_cache_strategy_is_abstract(self):
        """Test CacheStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CacheStrategy()

    def test_cache_strategy_subclass_must_implement_methods(self):
        """Test subclasses must implement all abstract methods."""

        class IncompleteCache(CacheStrategy):
            """Cache missing some required methods."""

            def get(self, key):
                return None

            # Missing set, clear, get_stats

        with pytest.raises(TypeError):
            IncompleteCache()


@pytest.mark.fast
class TestInMemoryCacheBasics:
    """Test basic InMemoryCache functionality."""

    def setup_method(self):
        """Set up cache for each test."""
        self.cache = InMemoryCache(default_ttl_seconds=60, max_size=100)

    def test_cache_initialization(self):
        """Test cache initializes with correct parameters."""
        assert self.cache.default_ttl == 60
        assert self.cache.max_size == 100
        assert len(self.cache._cache) == 0

        # Test metrics start at zero
        stats = self.cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0

    def test_set_and_get_basic(self):
        """Test basic set and get operations."""
        key = "test_key"
        value = {"type": "benchflow", "timestamp": time.time()}

        # Set value
        result = self.cache.set(key, value)
        assert result is True

        # Get value
        retrieved = self.cache.get(key)
        assert retrieved == value

        # Check metrics
        stats = self.cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["sets"] == 1

    def test_get_nonexistent_key(self):
        """Test getting non-existent key returns None."""
        result = self.cache.get("nonexistent")
        assert result is None

        # Check metrics
        stats = self.cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_set_with_custom_ttl(self):
        """Test setting with custom TTL."""
        key = "test_key"
        value = {"data": "test"}

        # Set with 0.1 second TTL
        self.cache.set(key, value, ttl_seconds=0.1)

        # Should be available immediately
        assert self.cache.get(key) == value

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired
        assert self.cache.get(key) is None

    def test_clear_specific_key(self):
        """Test clearing specific key."""
        # Set multiple keys
        self.cache.set("key1", {"data": "1"})
        self.cache.set("key2", {"data": "2"})

        # Clear specific key
        result = self.cache.clear("key1")
        assert result is True

        # Check key1 is gone but key2 remains
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") == {"data": "2"}

    def test_clear_all_keys(self):
        """Test clearing all keys."""
        # Set multiple keys
        self.cache.set("key1", {"data": "1"})
        self.cache.set("key2", {"data": "2"})

        # Clear all
        result = self.cache.clear()
        assert result is True

        # Check all keys are gone
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None

        # Check cache is empty
        stats = self.cache.get_stats()
        assert stats["total_entries"] == 0


@pytest.mark.medium
class TestInMemoryCacheTTLAndExpiration:
    """Test TTL behavior and expiration handling."""

    def setup_method(self):
        """Set up cache for each test."""
        self.cache = InMemoryCache(default_ttl_seconds=0.2)  # Short TTL for testing

    def test_ttl_expiration(self):
        """Test entries expire after TTL."""
        key = "expiring_key"
        value = {"data": "expires"}

        # Set entry
        self.cache.set(key, value)

        # Should be available immediately
        assert self.cache.get(key) == value

        # Wait for expiration
        time.sleep(0.25)

        # Should be expired and automatically cleaned up
        assert self.cache.get(key) is None

        # Cache should have cleaned up the entry
        stats = self.cache.get_stats()
        assert stats["total_entries"] == 0

    def test_stats_differentiate_valid_and_expired(self):
        """Test stats correctly identify valid vs expired entries."""
        # Set entries with different expiration times
        current_time = time.time()

        # Valid entry
        self.cache.set("valid_key", {"data": "valid"})

        # Manually inject expired entry (for testing stats calculation)
        expired_key = "expired_key"
        self.cache._cache[expired_key] = {
            "value": {"data": "expired"},
            "expires_at": current_time - 1,  # Already expired
            "created_at": current_time - 2,
            "last_accessed": current_time - 2,
        }

        stats = self.cache.get_stats()
        assert stats["valid_entries"] == 1
        assert stats["expired_entries"] == 1
        assert stats["total_entries"] == 2

    def test_custom_ttl_overrides_default(self):
        """Test custom TTL overrides default TTL."""
        # Default TTL is 0.2s, use custom 0.5s
        self.cache.set("long_key", {"data": "long"}, ttl_seconds=0.5)
        self.cache.set("short_key", {"data": "short"})  # Uses default 0.2s

        # Wait for default TTL to expire
        time.sleep(0.25)

        # Short key should be expired, long key should still be valid
        assert self.cache.get("short_key") is None
        assert self.cache.get("long_key") == {"data": "long"}


@pytest.mark.medium
class TestInMemoryCacheLRUEviction:
    """Test LRU eviction behavior."""

    def setup_method(self):
        """Set up cache with small max_size for testing."""
        self.cache = InMemoryCache(default_ttl_seconds=60, max_size=3)

    def test_lru_eviction_when_full(self):
        """Test LRU eviction when cache reaches max size."""
        # Fill cache to max size
        self.cache.set("key1", {"data": "1"})
        time.sleep(0.01)  # Ensure different timestamps
        self.cache.set("key2", {"data": "2"})
        time.sleep(0.01)
        self.cache.set("key3", {"data": "3"})

        # Cache should be full
        stats = self.cache.get_stats()
        assert stats["total_entries"] == 3
        assert stats["evictions"] == 0

        # Add one more entry - should evict key1 (oldest)
        self.cache.set("key4", {"data": "4"})

        # Check eviction occurred
        assert self.cache.get("key1") is None  # Evicted
        assert self.cache.get("key2") == {"data": "2"}
        assert self.cache.get("key3") == {"data": "3"}
        assert self.cache.get("key4") == {"data": "4"}

        # Check metrics
        stats = self.cache.get_stats()
        assert stats["evictions"] == 1
        assert stats["total_entries"] == 3

    def test_lru_access_updates_last_accessed(self):
        """Test accessing entry updates its position in LRU order."""
        # Fill cache
        self.cache.set("key1", {"data": "1"})
        time.sleep(0.01)
        self.cache.set("key2", {"data": "2"})
        time.sleep(0.01)
        self.cache.set("key3", {"data": "3"})

        # Access key1 to make it recently used
        self.cache.get("key1")

        # Add new entry - should evict key2 (now oldest accessed)
        self.cache.set("key4", {"data": "4"})

        # key1 should still be there (was accessed recently)
        assert self.cache.get("key1") == {"data": "1"}
        # key2 should be evicted
        assert self.cache.get("key2") is None
        assert self.cache.get("key3") == {"data": "3"}
        assert self.cache.get("key4") == {"data": "4"}

    def test_overwriting_existing_key_no_eviction(self):
        """Test overwriting existing key doesn't trigger eviction."""
        # Fill cache to max size
        self.cache.set("key1", {"data": "1"})
        self.cache.set("key2", {"data": "2"})
        self.cache.set("key3", {"data": "3"})

        # Overwrite existing key
        self.cache.set("key2", {"data": "2_updated"})

        # No eviction should occur
        assert self.cache.get("key1") == {"data": "1"}
        assert self.cache.get("key2") == {"data": "2_updated"}
        assert self.cache.get("key3") == {"data": "3"}

        stats = self.cache.get_stats()
        assert stats["evictions"] == 0
        assert stats["total_entries"] == 3


@pytest.mark.slow
class TestInMemoryCacheThreadSafety:
    """Test thread safety of InMemoryCache under concurrent access."""

    def setup_method(self):
        """Set up cache for threading tests."""
        self.cache = InMemoryCache(default_ttl_seconds=60, max_size=1000)
        self.num_threads = 10
        self.operations_per_thread = 100

    def test_concurrent_set_and_get(self):
        """Test concurrent set and get operations are thread-safe."""

        def worker(thread_id):
            """Worker function for concurrent operations."""
            results = []
            for i in range(self.operations_per_thread):
                key = f"thread_{thread_id}_key_{i}"
                value = {"thread": thread_id, "operation": i}

                # Set value
                self.cache.set(key, value)

                # Get value back
                retrieved = self.cache.get(key)
                results.append((key, retrieved == value))

                # Also try to get other threads' values occasionally
                if i % 10 == 0:
                    other_thread = (thread_id + 1) % self.num_threads
                    other_key = f"thread_{other_thread}_key_{i}"
                    self.cache.get(other_key)  # May or may not exist

            return results

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(self.num_threads)]

            all_results = []
            for future in as_completed(futures):
                thread_results = future.result()
                all_results.extend(thread_results)

        # Verify all operations succeeded
        for key, success in all_results:
            assert success, f"Thread safety violation for key: {key}"

        # Verify final cache state is consistent
        stats = self.cache.get_stats()
        expected_entries = self.num_threads * self.operations_per_thread

        # Should have all entries (or less due to eviction, but not more)
        assert stats["total_entries"] <= expected_entries
        assert stats["total_entries"] > 0

    def test_concurrent_clear_operations(self):
        """Test concurrent clear operations don't cause corruption."""
        # Pre-populate cache
        for i in range(100):
            self.cache.set(f"key_{i}", {"data": i})

        def worker(thread_id):
            """Worker that performs mixed operations including clears."""
            for i in range(50):
                if i % 5 == 0:
                    # Clear specific key
                    self.cache.clear(f"key_{thread_id * 10 + i % 10}")
                elif i % 10 == 0:
                    # Clear all (occasionally)
                    self.cache.clear()
                else:
                    # Regular set/get operations
                    key = f"worker_{thread_id}_{i}"
                    self.cache.set(key, {"worker": thread_id, "i": i})
                    self.cache.get(key)

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]

            # Wait for completion without exceptions
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    pytest.fail(f"Thread safety violation in clear operations: {e}")

        # Cache should still be functional after concurrent operations
        self.cache.set("final_test", {"test": "final"})
        assert self.cache.get("final_test") == {"test": "final"}

        # Stats should be consistent
        stats = self.cache.get_stats()
        assert isinstance(stats, dict)
        assert all(isinstance(v, int | float | str) for v in stats.values() if v is not None)


@pytest.mark.fast
class TestNullCache:
    """Test NullCache no-op behavior."""

    def setup_method(self):
        """Set up null cache for each test."""
        self.cache = NullCache()

    def test_get_always_returns_none(self):
        """Test get always returns None."""
        assert self.cache.get("any_key") is None
        assert self.cache.get("") is None
        assert self.cache.get("123") is None

    def test_set_always_returns_true(self):
        """Test set always returns True but stores nothing."""
        assert self.cache.set("key", {"data": "value"}) is True
        assert self.cache.set("key", {"data": "value"}, ttl_seconds=60) is True

        # But get still returns None
        assert self.cache.get("key") is None

    def test_clear_always_returns_true(self):
        """Test clear always returns True."""
        assert self.cache.clear() is True
        assert self.cache.clear("specific_key") is True

    def test_get_stats_shows_null_behavior(self):
        """Test stats reflect null cache behavior."""
        # Perform operations
        self.cache.get("key1")
        self.cache.set("key2", {"data": "test"})
        self.cache.clear("key3")

        stats = self.cache.get_stats()
        assert stats["strategy"] == "NullCache"
        assert stats["operations"] == 3
        assert "no-ops" in stats["note"].lower()


@pytest.mark.fast
class TestRedisCacheStub:
    """Test RedisCache stub implementation."""

    def setup_method(self):
        """Set up Redis cache stub for each test."""
        self.cache = RedisCache("redis://test:6379", default_ttl_seconds=300)

    def test_redis_cache_initialization(self):
        """Test Redis cache initializes with correct parameters."""
        assert self.cache.redis_url == "redis://test:6379"
        assert self.cache.default_ttl == 300

    def test_redis_stub_behavior(self):
        """Test Redis stub returns expected responses."""
        # All operations should work as no-ops in stub
        assert self.cache.get("key") is None
        assert self.cache.set("key", {"data": "value"}) is True
        assert self.cache.clear() is True
        assert self.cache.clear("key") is True

    def test_redis_stats_show_stub_status(self):
        """Test Redis stats indicate stub implementation."""
        stats = self.cache.get_stats()
        assert stats["strategy"] == "RedisCache"
        assert stats["status"] == "stub_implementation"
        assert "stub" in stats["note"].lower()
        assert stats["redis_url"] == "redis://test:6379"


@pytest.mark.fast
class TestCacheStrategyFactory:
    """Test cache strategy factory function."""

    def test_create_memory_strategy(self):
        """Test creating in-memory cache strategy."""
        cache = create_cache_strategy("memory")
        assert isinstance(cache, InMemoryCache)
        assert cache.default_ttl == 300  # Default

    def test_create_memory_with_custom_params(self):
        """Test creating memory cache with custom parameters."""
        cache = create_cache_strategy("memory", default_ttl_seconds=600, max_size=50)
        assert isinstance(cache, InMemoryCache)
        assert cache.default_ttl == 600
        assert cache.max_size == 50

    def test_create_null_strategy(self):
        """Test creating null cache strategy."""
        cache = create_cache_strategy("null")
        assert isinstance(cache, NullCache)

    def test_create_redis_strategy(self):
        """Test creating Redis cache strategy."""
        cache = create_cache_strategy("redis", redis_url="redis://custom:6379")
        assert isinstance(cache, RedisCache)
        assert cache.redis_url == "redis://custom:6379"

    def test_invalid_strategy_type(self):
        """Test invalid strategy type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_cache_strategy("invalid_strategy")
        assert "Unknown cache strategy" in str(exc_info.value)

    @patch.dict(os.environ, {}, clear=True)
    def test_auto_strategy_defaults_to_memory(self):
        """Test auto strategy defaults to memory when no environment vars set."""
        cache = create_cache_strategy("auto")
        assert isinstance(cache, InMemoryCache)

    @patch.dict(os.environ, {"CACHE_STRATEGY": "null"})
    def test_auto_strategy_uses_env_var(self):
        """Test auto strategy uses CACHE_STRATEGY environment variable."""
        cache = create_cache_strategy("auto")
        assert isinstance(cache, NullCache)

    @patch.dict(os.environ, {"REDIS_URL": "redis://env-redis:6379"})
    def test_auto_strategy_detects_redis_url(self):
        """Test auto strategy detects REDIS_URL environment variable."""
        cache = create_cache_strategy("auto")
        assert isinstance(cache, RedisCache)
        assert cache.redis_url == "redis://env-redis:6379"

    @patch.dict(os.environ, {"DISABLE_CACHE": "1"})
    def test_auto_strategy_detects_disable_cache(self):
        """Test auto strategy uses null cache when DISABLE_CACHE is set."""
        cache = create_cache_strategy("auto")
        assert isinstance(cache, NullCache)

    @patch.dict(os.environ, {"CACHE_STRATEGY": "memory", "REDIS_URL": "redis://test:6379"})
    def test_explicit_env_overrides_detection(self):
        """Test explicit CACHE_STRATEGY overrides other environment detection."""
        cache = create_cache_strategy("auto")
        assert isinstance(cache, InMemoryCache)  # Should use explicit memory, not Redis


@pytest.mark.fast
class TestCacheErrorHandling:
    """Test error handling in cache implementations."""

    def test_cache_error_exception(self):
        """Test CacheError exception can be raised."""
        with pytest.raises(CacheError) as exc_info:
            raise CacheError("Test cache error")
        assert "Test cache error" in str(exc_info.value)

    def test_in_memory_cache_handles_invalid_operations(self):
        """Test InMemoryCache handles edge cases gracefully."""
        cache = InMemoryCache()

        # Test operations with various invalid inputs
        # These should not crash but may return None/False appropriately
        result = cache.get("")  # Empty key
        assert result is None

        result = cache.get(None)  # None key - may raise TypeError, which is acceptable
        # This is implementation-dependent behavior

    def test_null_cache_handles_all_inputs(self):
        """Test NullCache handles any input without errors."""
        cache = NullCache()

        # Should handle any input gracefully
        assert cache.get("") is None
        assert cache.get(None) is None
        assert cache.set("", {}) is True
        assert cache.set(None, None) is True
        assert cache.clear("") is True
        assert cache.clear(None) is True


@pytest.mark.medium
class TestCacheStrategiesIntegration:
    """Test cache strategies work together and can be swapped."""

    def test_different_strategies_same_interface(self):
        """Test all strategies implement the same interface consistently."""
        strategies = [InMemoryCache(), NullCache(), RedisCache()]

        test_key = "integration_test"
        test_value = {"type": "benchflow", "timestamp": time.time()}

        for strategy in strategies:
            # All should implement the interface
            result = strategy.set(test_key, test_value)
            assert isinstance(result, bool)

            retrieved = strategy.get(test_key)
            # NullCache and RedisCache (stub) will return None
            # InMemoryCache should return the value
            if isinstance(strategy, InMemoryCache):
                assert retrieved == test_value
            else:
                assert retrieved is None

            result = strategy.clear(test_key)
            assert isinstance(result, bool)

            stats = strategy.get_stats()
            assert isinstance(stats, dict)
            assert "strategy" in stats

    def test_strategies_can_be_substituted(self):
        """Test strategies can be substituted without changing client code."""

        def cache_user(cache_strategy: CacheStrategy):
            """Function that uses any cache strategy."""
            # Use the cache without knowing its implementation
            cache_strategy.set("test", {"data": "test_data"})
            result = cache_strategy.get("test")
            stats = cache_strategy.get_stats()
            cache_strategy.clear("test")

            return result, stats

        # Test with different strategies
        memory_result, memory_stats = cache_user(InMemoryCache())
        null_result, null_stats = cache_user(NullCache())
        redis_result, redis_stats = cache_user(RedisCache())

        # Results should be consistent with each strategy's behavior
        assert memory_result == {"data": "test_data"}  # InMemory returns data
        assert null_result is None  # Null returns None
        assert redis_result is None  # Redis stub returns None

        # All should return stats
        assert all("strategy" in stats for stats in [memory_stats, null_stats, redis_stats])
