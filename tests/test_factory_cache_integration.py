"""
Integration tests for Pokemon Gym Factory with Cache Strategy dependency injection.

Tests the complete dependency injection pattern between factory functions and
cache strategies, ensuring clean architecture and backward compatibility.

Author: Bot Dean - Production-First Engineering
"""

import time
from unittest.mock import Mock, patch

import pytest

from src.claudelearnspokemon.compatibility.cache_strategies import (
    InMemoryCache,
    NullCache,
    RedisCache,
    create_cache_strategy,
)
from src.claudelearnspokemon.emulator_pool import PokemonGymClient
from src.claudelearnspokemon.pokemon_gym_adapter import PokemonGymAdapter
from src.claudelearnspokemon.pokemon_gym_factory import (
    clear_detection_cache,
    create_pokemon_client,
    detect_server_type,
    get_detection_cache_stats,
    get_factory_metrics,
    set_default_cache_strategy,
)


@pytest.mark.fast
class TestFactoryCacheIntegrationBasics:
    """Test basic integration between factory functions and cache strategies."""

    def setup_method(self):
        """Set up clean state for each test."""
        # Reset to default cache strategy
        set_default_cache_strategy(create_cache_strategy("memory"))

    def test_factory_uses_injected_cache_strategy(self):
        """Test factory uses explicitly injected cache strategy."""
        # Create custom cache strategy
        custom_cache = NullCache()

        with patch(
            "src.claudelearnspokemon.pokemon_gym_factory.requests.Session"
        ) as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            # Mock server response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"session_active": True}
            mock_session.get.return_value = mock_response

            # Create client with explicit cache strategy
            client = create_pokemon_client(
                port=8081,
                container_id="test_container",
                adapter_type="auto",
                cache_strategy=custom_cache,
            )

            # Should create appropriate client
            assert isinstance(client, PokemonGymAdapter)

            # Verify custom cache was used (NullCache never stores anything)
            # So a second detection should make another HTTP call
            create_pokemon_client(
                port=8081,
                container_id="test_container2",
                adapter_type="auto",
                cache_strategy=custom_cache,
            )

            # With NullCache, should make HTTP call each time
            assert mock_session.get.call_count == 2

    def test_factory_uses_default_cache_when_none_provided(self):
        """Test factory falls back to default cache when none provided."""
        # Set up default cache
        default_cache = InMemoryCache(default_ttl_seconds=60)
        set_default_cache_strategy(default_cache)

        with patch(
            "src.claudelearnspokemon.pokemon_gym_factory.requests.Session"
        ) as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            # Mock server response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"session_active": True}
            mock_session.get.return_value = mock_response

            # Create client without explicit cache strategy
            client1 = create_pokemon_client(
                port=8081,
                container_id="test_container",
                adapter_type="auto",
                # No cache_strategy parameter - should use default
            )

            # Create second client - should use cached result
            client2 = create_pokemon_client(
                port=8081, container_id="test_container2", adapter_type="auto"
            )

            # Should create appropriate clients
            assert isinstance(client1, PokemonGymAdapter)
            assert isinstance(client2, PokemonGymAdapter)

            # With InMemoryCache, second call should use cache
            assert mock_session.get.call_count == 1

    def test_backward_compatibility_no_cache_parameter(self):
        """Test backward compatibility - existing code works without cache_strategy parameter."""
        # This tests that old code like:
        # client = create_pokemon_client(8081, "container_id", "direct")
        # still works without modification

        client = create_pokemon_client(
            port=8081, container_id="test_container", adapter_type="direct"
        )

        assert isinstance(client, PokemonGymClient)
        assert client.port == 8081
        assert client.container_id == "test_container"


@pytest.mark.fast
class TestCacheStrategySubstitution:
    """Test different cache strategies can be substituted transparently."""

    def test_memory_cache_caching_behavior(self):
        """Test InMemoryCache provides actual caching."""
        memory_cache = InMemoryCache(default_ttl_seconds=60)

        with patch(
            "src.claudelearnspokemon.pokemon_gym_factory.requests.Session"
        ) as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            # Mock server response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"session_active": True}
            mock_session.get.return_value = mock_response

            # First detection
            result1 = detect_server_type(8081, cache_strategy=memory_cache)
            assert result1 == "benchflow"

            # Second detection should use cache
            result2 = detect_server_type(8081, cache_strategy=memory_cache)
            assert result2 == "benchflow"

            # Only one HTTP call should be made
            assert mock_session.get.call_count == 1

            # Verify cache stats
            stats = memory_cache.get_stats()
            assert stats["hits"] == 1  # Second call was cache hit
            assert stats["misses"] == 1  # First call was cache miss
            assert stats["sets"] == 1  # One value was set

    def test_null_cache_no_caching_behavior(self):
        """Test NullCache provides no caching."""
        null_cache = NullCache()

        with patch(
            "src.claudelearnspokemon.pokemon_gym_factory.requests.Session"
        ) as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            # Mock server response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"session_active": True}
            mock_session.get.return_value = mock_response

            # Multiple detections
            result1 = detect_server_type(8081, cache_strategy=null_cache)
            result2 = detect_server_type(8081, cache_strategy=null_cache)
            result3 = detect_server_type(8081, cache_strategy=null_cache)

            assert all(r == "benchflow" for r in [result1, result2, result3])

            # Each call should make HTTP request (no caching)
            assert mock_session.get.call_count == 3

            # Verify null cache stats
            stats = null_cache.get_stats()
            assert stats["strategy"] == "NullCache"
            assert "no-ops" in stats["note"].lower()

    def test_redis_cache_production_behavior(self):
        """Test RedisCache production behavior with fallback caching."""
        # Use fallback_enabled=True (default) to test production behavior
        redis_cache = RedisCache("redis://test:6379", fallback_enabled=True)

        with patch(
            "src.claudelearnspokemon.pokemon_gym_factory.requests.Session"
        ) as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            # Mock server response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"session_active": True}
            mock_session.get.return_value = mock_response

            # Detection with Redis cache (will use fallback)
            result = detect_server_type(8081, cache_strategy=redis_cache)
            assert result == "benchflow"

            # Second call should use cached value from fallback cache
            result2 = detect_server_type(8081, cache_strategy=redis_cache)
            assert result2 == "benchflow"

            # Only one HTTP request should be made (second is cached)
            assert mock_session.get.call_count == 1

            # Verify Redis cache stats show fallback usage
            stats = redis_cache.get_stats()
            assert stats["strategy"] == "RedisCache"
            assert stats["status"] == "degraded"  # Redis unavailable, using fallback
            assert stats["fallback_operations"] > 0

    def test_redis_cache_no_fallback_behavior(self):
        """Test RedisCache without fallback behaves like old stub."""
        # Disable fallback to test "no caching" behavior
        redis_cache = RedisCache("redis://test:6379", fallback_enabled=False)

        with patch(
            "src.claudelearnspokemon.pokemon_gym_factory.requests.Session"
        ) as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            # Mock server response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"session_active": True}
            mock_session.get.return_value = mock_response

            # Detection with Redis cache (no fallback)
            result = detect_server_type(8081, cache_strategy=redis_cache)
            assert result == "benchflow"

            # Second call should also make HTTP request (no caching without fallback)
            result2 = detect_server_type(8081, cache_strategy=redis_cache)
            assert result2 == "benchflow"

            # Each call makes HTTP request (no caching without Redis or fallback)
            assert mock_session.get.call_count == 2

            # Verify Redis cache stats show errors but no fallback
            stats = redis_cache.get_stats()
            assert stats["strategy"] == "RedisCache"
            assert stats["status"] == "degraded"
            assert stats["fallback_operations"] == 0  # No fallback configured
            assert stats["errors"] > 0


@pytest.mark.fast
class TestCacheOperationIntegration:
    """Test cache operation functions work with different strategies."""

    def test_clear_cache_with_different_strategies(self):
        """Test clear_detection_cache works with different cache strategies."""
        strategies = [
            ("InMemoryCache", InMemoryCache()),
            ("NullCache", NullCache()),
            ("RedisCache", RedisCache()),
        ]

        for strategy_name, strategy in strategies:
            # Add some data to cache (if it supports it)
            strategy.set("port_8081", {"type": "benchflow", "timestamp": time.time()})
            strategy.set("port_8082", {"type": "direct", "timestamp": time.time()})

            # Clear all
            clear_detection_cache(cache_strategy=strategy)
            # Function doesn't return anything, but shouldn't crash

            # Clear specific port
            strategy.set("port_8081", {"type": "benchflow", "timestamp": time.time()})
            clear_detection_cache(8081, cache_strategy=strategy)
            # Should not crash

            print(f"✓ {strategy_name} clear operations work correctly")

    def test_get_cache_stats_with_different_strategies(self):
        """Test get_detection_cache_stats works with different strategies."""
        strategies = [
            ("InMemoryCache", InMemoryCache()),
            ("NullCache", NullCache()),
            ("RedisCache", RedisCache()),
        ]

        for strategy_name, strategy in strategies:
            stats = get_detection_cache_stats(cache_strategy=strategy)

            # All should return dict with strategy info
            assert isinstance(stats, dict)
            assert "strategy" in stats
            assert stats["strategy"] == strategy_name

            print(f"✓ {strategy_name} stats: {stats}")

    def test_get_factory_metrics_includes_cache_info(self):
        """Test factory metrics include cache strategy information."""
        test_cache = InMemoryCache()

        metrics = get_factory_metrics(cache_strategy=test_cache)

        assert isinstance(metrics, dict)
        assert "detection_cache" in metrics
        assert "cache_strategy_type" in metrics
        assert metrics["cache_strategy_type"] == "InMemoryCache"

        # Should include cache-specific stats
        cache_stats = metrics["detection_cache"]
        assert cache_stats["strategy"] == "InMemoryCache"
        assert "hits" in cache_stats
        assert "misses" in cache_stats


@pytest.mark.medium
class TestProductionScenarios:
    """Test production-like scenarios with cache strategies."""

    def test_mixed_cache_strategies_in_parallel(self):
        """Test different cache strategies can work in parallel without interference."""
        memory_cache = InMemoryCache()
        null_cache = NullCache()

        # Simulate parallel usage with different cache strategies
        with patch(
            "src.claudelearnspokemon.pokemon_gym_factory.requests.Session"
        ) as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            # Mock server response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"session_active": True}
            mock_session.get.return_value = mock_response

            # Detections with different caches
            result1 = detect_server_type(8081, cache_strategy=memory_cache)
            result2 = detect_server_type(8081, cache_strategy=null_cache)
            result3 = detect_server_type(8081, cache_strategy=memory_cache)  # Should use cache
            result4 = detect_server_type(8081, cache_strategy=null_cache)  # Should make HTTP call

            assert all(r == "benchflow" for r in [result1, result2, result3, result4])

            # Memory cache should cache (2 calls), null cache shouldn't (2 calls) = 3 total
            # (memory: 1 call + 1 cache hit, null: 2 calls)
            assert mock_session.get.call_count == 3

            # Verify each cache strategy maintains its own state
            memory_stats = memory_cache.get_stats()
            null_stats = null_cache.get_stats()

            assert memory_stats["hits"] == 1  # One cache hit
            assert memory_stats["misses"] == 1  # One cache miss
            # NullCache counts all operations (get, set for each detection)
            assert null_stats["operations"] == 4  # Two detections = 4 operations (2 get + 2 set)

    def test_cache_strategy_hot_swap(self):
        """Test cache strategies can be swapped at runtime."""
        # Start with memory cache
        memory_cache = InMemoryCache()
        set_default_cache_strategy(memory_cache)

        # Warm up cache
        with patch(
            "src.claudelearnspokemon.pokemon_gym_factory.requests.Session"
        ) as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"session_active": True}
            mock_session.get.return_value = mock_response

            # Fill cache
            detect_server_type(8081)  # Uses default memory cache
            assert mock_session.get.call_count == 1

            # Verify cache is populated
            stats = get_detection_cache_stats()
            assert stats["total_entries"] == 1

            # Hot swap to null cache
            null_cache = NullCache()
            set_default_cache_strategy(null_cache)

            # Verify new default is in effect
            new_stats = get_detection_cache_stats()
            assert new_stats["strategy"] == "NullCache"
            assert new_stats["operations"] == 0  # Fresh null cache

            # New detections use null cache (no caching)
            detect_server_type(8082)  # Should make HTTP call
            detect_server_type(8082)  # Should make another HTTP call

            # Total calls: 1 (memory cache) + 2 (null cache) = 3
            assert mock_session.get.call_count == 3


@pytest.mark.fast
class TestErrorHandlingIntegration:
    """Test error handling in cache strategy integration."""

    def test_factory_handles_cache_errors_gracefully(self):
        """Test factory handles cache operation failures gracefully."""
        # Create a mock cache that raises errors
        error_cache = Mock()
        error_cache.get.side_effect = Exception("Cache read failed")
        error_cache.set.side_effect = Exception("Cache write failed")

        with patch(
            "src.claudelearnspokemon.pokemon_gym_factory.requests.Session"
        ) as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"session_active": True}
            mock_session.get.return_value = mock_response

            # Detection should still work despite cache errors
            result = detect_server_type(8081, cache_strategy=error_cache)
            assert result == "benchflow"

            # Should have made HTTP call (cache failed)
            assert mock_session.get.call_count == 1

    def test_factory_validates_cache_strategy_interface(self):
        """Test factory gracefully handles invalid cache strategy objects."""
        # Create object that doesn't implement CacheStrategy interface
        invalid_cache = object()

        with patch(
            "src.claudelearnspokemon.pokemon_gym_factory.requests.Session"
        ) as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"session_active": True}
            mock_session.get.return_value = mock_response

            # Factory should handle invalid cache gracefully and continue with detection
            result = detect_server_type(8081, cache_strategy=invalid_cache)

            # Should still return valid result despite cache errors
            assert result == "benchflow"

            # Should have made HTTP call since caching failed
            assert mock_session.get.call_count == 1


@pytest.mark.fast
class TestEnvironmentIntegration:
    """Test cache strategy selection based on environment."""

    @patch.dict("os.environ", {"CACHE_STRATEGY": "memory"})
    def test_auto_strategy_respects_environment(self):
        """Test auto strategy selection respects environment variables."""
        cache = create_cache_strategy("auto")
        assert isinstance(cache, InMemoryCache)

    @patch.dict("os.environ", {"DISABLE_CACHE": "1"})
    def test_auto_strategy_disables_cache_when_requested(self):
        """Test auto strategy uses NullCache when caching is disabled."""
        cache = create_cache_strategy("auto")
        assert isinstance(cache, NullCache)

    @patch.dict("os.environ", {"REDIS_URL": "redis://test:6379"})
    def test_auto_strategy_detects_redis(self):
        """Test auto strategy detects Redis configuration."""
        cache = create_cache_strategy("auto")
        assert isinstance(cache, RedisCache)
        assert cache.redis_url == "redis://test:6379"
