"""
Test suite for Pokemon Gym Factory compatibility layer.

Tests the factory pattern with auto-detection, caching, and production
error handling patterns for transparent adapter selection.

Author: Bot Dean - Production-First Engineering
"""

import time
from unittest.mock import Mock, patch

import pytest
from requests.exceptions import RequestException

from src.claudelearnspokemon.compatibility.cache_strategies import (
    InMemoryCache,
    NullCache,
)
from src.claudelearnspokemon.emulator_pool import PokemonGymClient
from src.claudelearnspokemon.pokemon_gym_adapter import PokemonGymAdapter
from src.claudelearnspokemon.pokemon_gym_factory import (
    FactoryError,
    _is_benchflow_server,
    clear_detection_cache,
    create_pokemon_client,
    detect_server_type,
    get_detection_cache_stats,
    get_factory_metrics,
    set_default_cache_strategy,
    validate_client_compatibility,
)


@pytest.mark.fast
class TestPokemonGymFactoryBasics:
    """Test basic factory functionality and parameter validation."""

    def setup_method(self):
        """Set up clean cache strategy for each test."""
        # Use NullCache for predictable test behavior
        self.test_cache = NullCache()
        set_default_cache_strategy(self.test_cache)
        clear_detection_cache()

    def test_factory_with_explicit_direct_type(self):
        """Test factory creates PokemonGymClient when explicitly requested."""
        client = create_pokemon_client(
            port=8081, container_id="test_container", adapter_type="direct"
        )

        assert isinstance(client, PokemonGymClient)
        assert client.port == 8081
        assert client.container_id == "test_container"

    def test_factory_with_explicit_benchflow_type(self):
        """Test factory creates PokemonGymAdapter when explicitly requested."""
        client = create_pokemon_client(
            port=8081, container_id="test_container", adapter_type="benchflow"
        )

        assert isinstance(client, PokemonGymAdapter)
        assert client.port == 8081
        assert client.container_id == "test_container"

    def test_factory_with_fallback_type(self):
        """Test factory creates PokemonGymClient in fallback mode."""
        client = create_pokemon_client(
            port=8081, container_id="test_container", adapter_type="fallback"
        )

        assert isinstance(client, PokemonGymClient)

    def test_factory_with_custom_input_delay(self):
        """Test factory passes custom input delay to adapter."""
        client = create_pokemon_client(
            port=8081, container_id="test_container", adapter_type="benchflow", input_delay=0.1
        )

        assert isinstance(client, PokemonGymAdapter)
        assert client.config["input_delay"] == 0.1

    def test_factory_parameter_validation(self):
        """Test factory validates input parameters."""
        # Invalid port
        with pytest.raises(FactoryError) as exc_info:
            create_pokemon_client(port=-1, container_id="test")
        assert "Invalid port" in str(exc_info.value)

        # Empty container ID
        with pytest.raises(FactoryError) as exc_info:
            create_pokemon_client(port=8081, container_id="")
        assert "Container ID cannot be empty" in str(exc_info.value)

        # Invalid adapter type
        with pytest.raises(FactoryError) as exc_info:
            create_pokemon_client(port=8081, container_id="test", adapter_type="invalid")
        assert "Invalid adapter_type" in str(exc_info.value)


@pytest.mark.fast
class TestServerTypeDetection:
    """Test server type auto-detection logic."""

    def setup_method(self):
        """Set up cache strategy for testing."""
        # Use InMemoryCache for testing actual caching behavior
        self.test_cache = InMemoryCache(default_ttl_seconds=300)
        clear_detection_cache(cache_strategy=self.test_cache)

    @patch("src.claudelearnspokemon.pokemon_gym_factory.requests.Session")
    def test_benchflow_server_detection_success(self, mock_session_class):
        """Test successful detection of benchflow-ai server."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock benchflow-ai status endpoint success
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"session_active": True, "location": {"map": "town"}}
        mock_session.get.return_value = mock_response

        result = detect_server_type(8081, cache_strategy=self.test_cache)

        assert result == "benchflow"
        mock_session.get.assert_called_once_with("http://localhost:8081/status", timeout=3.0)

    @patch("src.claudelearnspokemon.pokemon_gym_factory.requests.Session")
    def test_direct_server_detection_fallback(self, mock_session_class):
        """Test fallback to direct server when benchflow detection fails."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock benchflow-ai status endpoint failure
        mock_session.get.side_effect = RequestException("Not found")

        result = detect_server_type(8081, cache_strategy=self.test_cache)

        assert result == "direct"

    @patch("src.claudelearnspokemon.pokemon_gym_factory.requests.Session")
    def test_benchflow_detection_with_json_response(self, mock_session_class):
        """Test benchflow detection recognizes JSON response."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock status endpoint with generic JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"some": "data"}
        mock_session.get.return_value = mock_response

        assert _is_benchflow_server("http://localhost:8081", mock_session, 3.0)

    @patch("src.claudelearnspokemon.pokemon_gym_factory.requests.Session")
    def test_benchflow_detection_with_specific_fields(self, mock_session_class):
        """Test benchflow detection recognizes specific field patterns."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock status endpoint with benchflow-specific fields
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"game_state": {"level": 1}}
        mock_session.get.return_value = mock_response

        assert _is_benchflow_server("http://localhost:8081", mock_session, 3.0)

    @patch("src.claudelearnspokemon.pokemon_gym_factory.requests.Session")
    def test_benchflow_detection_handles_non_json(self, mock_session_class):
        """Test benchflow detection handles non-JSON responses."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock status endpoint that returns 200 but non-JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = Exception("Not JSON")
        mock_session.get.return_value = mock_response

        # Should still detect as benchflow-ai if /status returns 200
        assert _is_benchflow_server("http://localhost:8081", mock_session, 3.0)

    @patch("src.claudelearnspokemon.pokemon_gym_factory.requests.Session")
    def test_benchflow_detection_handles_404(self, mock_session_class):
        """Test benchflow detection handles 404 responses."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_session.get.return_value = mock_response

        assert not _is_benchflow_server("http://localhost:8081", mock_session, 3.0)


@pytest.mark.fast
class TestDetectionCaching:
    """Test detection result caching functionality."""

    def setup_method(self):
        """Set up cache strategy for testing."""
        # Use InMemoryCache for testing actual caching behavior
        self.test_cache = InMemoryCache(default_ttl_seconds=300)
        clear_detection_cache(cache_strategy=self.test_cache)

    @patch("src.claudelearnspokemon.pokemon_gym_factory.requests.Session")
    def test_detection_caching_works(self, mock_session_class):
        """Test detection results are cached properly."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock successful benchflow detection
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"session_active": True}
        mock_session.get.return_value = mock_response

        # First detection
        result1 = detect_server_type(8081, cache_strategy=self.test_cache)
        assert result1 == "benchflow"

        # Second detection should use cache
        result2 = detect_server_type(8081, cache_strategy=self.test_cache)
        assert result2 == "benchflow"

        # Should only make one HTTP call due to caching
        assert mock_session.get.call_count == 1

    def test_cache_expiration(self):
        """Test cache entries expire correctly."""
        # Use cache with very short TTL for testing expiration
        short_ttl_cache = InMemoryCache(default_ttl_seconds=0.1)

        # Add entry that will expire
        cache_key = "port_8081"
        short_ttl_cache.set(cache_key, {"type": "benchflow", "timestamp": time.time()})

        # Wait for expiration
        time.sleep(0.15)

        with patch(
            "src.claudelearnspokemon.pokemon_gym_factory.requests.Session"
        ) as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            # Mock new detection
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}
            mock_session.get.return_value = mock_response

            # Should not use expired cache
            _ = detect_server_type(8081, cache_strategy=short_ttl_cache)

            # Should make new HTTP call
            mock_session.get.assert_called_once()

    def test_clear_detection_cache_all(self):
        """Test clearing all cache entries."""
        # Add cache entries
        self.test_cache.set("port_8081", {"type": "benchflow", "timestamp": time.time()})
        self.test_cache.set("port_8082", {"type": "direct", "timestamp": time.time()})

        clear_detection_cache(cache_strategy=self.test_cache)

        # Verify cache is cleared
        stats = self.test_cache.get_stats()
        assert stats["total_entries"] == 0

    def test_clear_detection_cache_specific_port(self):
        """Test clearing cache for specific port."""
        # Add cache entries
        self.test_cache.set("port_8081", {"type": "benchflow", "timestamp": time.time()})
        self.test_cache.set("port_8082", {"type": "direct", "timestamp": time.time()})

        clear_detection_cache(8081, cache_strategy=self.test_cache)

        # Verify specific port cleared but other remains
        assert self.test_cache.get("port_8081") is None
        assert self.test_cache.get("port_8082") is not None

    def test_get_detection_cache_stats(self):
        """Test cache statistics reporting."""
        current_time = time.time()

        # Add cache entries
        self.test_cache.set("port_8081", {"type": "benchflow", "timestamp": current_time})
        self.test_cache.set("port_8082", {"type": "direct", "timestamp": current_time})

        stats = get_detection_cache_stats(cache_strategy=self.test_cache)

        assert "strategy" in stats
        assert stats["strategy"] == "InMemoryCache"
        assert stats["total_entries"] == 2
        # InMemoryCache provides detailed stats
        assert "hits" in stats
        assert "misses" in stats
        assert "sets" in stats


@pytest.mark.fast
class TestAutoDetectionWithFactory:
    """Test auto-detection integration with factory pattern."""

    def setup_method(self):
        """Set up cache strategy for testing."""
        # Use NullCache for predictable test behavior
        self.test_cache = NullCache()
        set_default_cache_strategy(self.test_cache)
        clear_detection_cache(cache_strategy=self.test_cache)

    @patch("src.claudelearnspokemon.pokemon_gym_factory.detect_server_type")
    def test_auto_detection_creates_adapter_for_benchflow(self, mock_detect):
        """Test auto-detection creates adapter for benchflow servers."""
        mock_detect.return_value = "benchflow"

        client = create_pokemon_client(
            port=8081, container_id="test_container", adapter_type="auto"
        )

        assert isinstance(client, PokemonGymAdapter)
        # Should be called with cache strategy
        mock_detect.assert_called_once()
        args = mock_detect.call_args[0]
        assert args[0] == 8081  # port
        assert args[1] == 3.0  # timeout

    @patch("src.claudelearnspokemon.pokemon_gym_factory.detect_server_type")
    def test_auto_detection_creates_client_for_direct(self, mock_detect):
        """Test auto-detection creates direct client for direct servers."""
        mock_detect.return_value = "direct"

        client = create_pokemon_client(
            port=8081, container_id="test_container", adapter_type="auto"
        )

        assert isinstance(client, PokemonGymClient)
        # Should be called with cache strategy
        mock_detect.assert_called_once()
        args = mock_detect.call_args[0]
        assert args[0] == 8081  # port
        assert args[1] == 3.0  # timeout

    @patch("src.claudelearnspokemon.pokemon_gym_factory.detect_server_type")
    def test_auto_detection_passes_custom_timeout(self, mock_detect):
        """Test auto-detection passes custom detection timeout."""
        mock_detect.return_value = "direct"

        create_pokemon_client(
            port=8081, container_id="test_container", adapter_type="auto", detection_timeout=5.0
        )

        # Should be called with custom timeout
        mock_detect.assert_called_once()
        args = mock_detect.call_args[0]
        assert args[0] == 8081  # port
        assert args[1] == 5.0  # timeout


@pytest.mark.fast
class TestClientCompatibilityValidation:
    """Test client compatibility validation functionality."""

    def setup_method(self):
        """Set up cache strategy for testing."""
        # Use NullCache for predictable test behavior
        self.test_cache = NullCache()
        set_default_cache_strategy(self.test_cache)
        clear_detection_cache(cache_strategy=self.test_cache)

    def test_validate_pokemon_gym_client(self):
        """Test validation of PokemonGymClient compatibility."""
        client = PokemonGymClient(8081, "test_container")

        assert validate_client_compatibility(client)

    def test_validate_pokemon_gym_adapter(self):
        """Test validation of PokemonGymAdapter compatibility."""
        adapter = PokemonGymAdapter(8081, "test_container")

        assert validate_client_compatibility(adapter)

    def test_validate_incompatible_client(self):
        """Test validation fails for incompatible client."""
        # Create mock client missing required methods
        mock_client = Mock()
        mock_client.port = 8081
        mock_client.container_id = "test"
        del mock_client.send_input  # Remove required method

        assert not validate_client_compatibility(mock_client)

    def test_validate_client_missing_attributes(self):
        """Test validation fails for client missing required attributes."""
        # Create mock client with methods but missing attributes
        mock_client = Mock()
        mock_client.send_input = Mock()
        mock_client.get_state = Mock()
        mock_client.reset_game = Mock()
        mock_client.is_healthy = Mock()
        mock_client.close = Mock()

        # Explicitly make port and container_id missing by deleting them
        if hasattr(mock_client, "port"):
            delattr(mock_client, "port")
        if hasattr(mock_client, "container_id"):
            delattr(mock_client, "container_id")

        # Use spec_set to prevent auto-creation of attributes
        mock_client = Mock(
            spec_set=["send_input", "get_state", "reset_game", "is_healthy", "close"]
        )

        assert not validate_client_compatibility(mock_client)


@pytest.mark.fast
class TestFactoryErrorHandling:
    """Test factory error handling and edge cases."""

    def setup_method(self):
        """Set up cache strategy for testing."""
        # Use NullCache for predictable test behavior
        self.test_cache = NullCache()
        set_default_cache_strategy(self.test_cache)
        clear_detection_cache(cache_strategy=self.test_cache)

    @patch("src.claudelearnspokemon.pokemon_gym_factory.detect_server_type")
    def test_factory_handles_detection_failure(self, mock_detect):
        """Test factory handles detection failures gracefully."""
        # Mock detection failure
        mock_detect.side_effect = Exception("Detection failed")

        with pytest.raises(FactoryError) as exc_info:
            create_pokemon_client(port=8081, container_id="test_container", adapter_type="auto")

        assert "Client creation failed" in str(exc_info.value)

    @patch("src.claudelearnspokemon.pokemon_gym_factory.PokemonGymAdapter")
    def test_factory_handles_adapter_creation_failure(self, mock_adapter_class):
        """Test factory handles adapter creation failures."""
        # Mock adapter creation failure
        mock_adapter_class.side_effect = Exception("Adapter creation failed")

        with pytest.raises(FactoryError) as exc_info:
            create_pokemon_client(
                port=8081, container_id="test_container", adapter_type="benchflow"
            )

        assert "Client creation failed" in str(exc_info.value)


@pytest.mark.fast
class TestFactoryMetrics:
    """Test factory metrics and monitoring functionality."""

    def setup_method(self):
        """Set up cache strategy for testing."""
        # Use NullCache for predictable test behavior
        self.test_cache = NullCache()
        set_default_cache_strategy(self.test_cache)
        clear_detection_cache(cache_strategy=self.test_cache)

    def test_get_factory_metrics(self):
        """Test factory metrics reporting."""
        metrics = get_factory_metrics()

        assert "detection_cache" in metrics
        assert "supported_types" in metrics
        assert "default_detection_timeout" in metrics
        assert "default_input_delay" in metrics
        assert "cache_strategy_type" in metrics

        # Verify supported types
        expected_types = ["direct", "benchflow", "auto", "fallback"]
        assert metrics["supported_types"] == expected_types

        # Verify defaults
        assert metrics["default_detection_timeout"] == 3.0
        assert metrics["default_input_delay"] == 0.05


@pytest.mark.fast
class TestFactoryProductionScenarios:
    """Test factory behavior in production-like scenarios."""

    def setup_method(self):
        """Set up cache strategy for testing."""
        # Use InMemoryCache for thread safety testing
        self.test_cache = InMemoryCache()
        set_default_cache_strategy(self.test_cache)
        clear_detection_cache(cache_strategy=self.test_cache)

    @patch("src.claudelearnspokemon.pokemon_gym_factory.detect_server_type")
    def test_factory_with_mixed_server_types(self, mock_detect):
        """Test factory handles mixed server types correctly."""

        # Mock different server types for different ports
        def mock_detection(port, timeout=3.0, cache_strategy=None):
            if port == 8081:
                return "benchflow"
            else:
                return "direct"

        mock_detect.side_effect = mock_detection

        # Create clients for different ports
        client1 = create_pokemon_client(8081, "container1", "auto")
        client2 = create_pokemon_client(8082, "container2", "auto")

        # Should create appropriate client types
        assert isinstance(client1, PokemonGymAdapter)
        assert isinstance(client2, PokemonGymClient)

    def test_factory_concurrent_usage_simulation(self):
        """Test factory behavior under concurrent usage patterns."""
        # This would be expanded for actual concurrent testing
        # For now, test rapid sequential calls

        clients = []
        for i in range(10):
            client = create_pokemon_client(
                port=8081 + i,
                container_id=f"container_{i}",
                adapter_type="direct",  # Use explicit type to avoid detection overhead
            )
            clients.append(client)

        # All should be direct clients
        assert all(isinstance(client, PokemonGymClient) for client in clients)

        # All should have correct configuration
        for i, client in enumerate(clients):
            assert client.port == 8081 + i
            assert client.container_id == f"container_{i}"

    def test_cache_thread_safety(self):
        """Test cache operations are thread-safe under concurrent access."""
        import concurrent.futures

        # Clear cache before test
        clear_detection_cache()

        def concurrent_detection(port):
            """Function to run in multiple threads"""
            try:
                return detect_server_type(port, cache_strategy=self.test_cache)
            except Exception as e:
                return f"error: {e}"

        def concurrent_cache_operations(thread_id):
            """Mixed cache operations to stress test thread safety"""
            results = []

            # Each thread performs various cache operations
            for i in range(5):
                port = 8080 + (thread_id * 10 + i) % 20  # Spread ports across threads

                # Detection (read and potentially write cache)
                result = concurrent_detection(port)
                results.append(("detect", port, result))

                # Get stats (read cache)
                stats = get_detection_cache_stats(cache_strategy=self.test_cache)
                results.append(("stats", port, stats.get("total_entries", 0)))

                # Clear specific port occasionally (write cache)
                if i % 3 == 0:
                    clear_detection_cache(port, cache_strategy=self.test_cache)
                    results.append(("clear", port, "cleared"))

            return results

        # Run concurrent operations with multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit work to all threads simultaneously
            futures = [
                executor.submit(concurrent_cache_operations, thread_id) for thread_id in range(10)
            ]

            # Wait for all threads to complete
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    thread_results = future.result(timeout=30)
                    all_results.extend(thread_results)
                except Exception as e:
                    # Thread safety violations would cause exceptions
                    pytest.fail(f"Thread safety violation detected: {e}")

        # Verify no thread safety violations occurred
        assert len(all_results) > 0, "No operations completed - possible deadlock"

        # Check that detection results are valid
        detection_results = [r for r in all_results if r[0] == "detect"]
        for _op_type, _port, result in detection_results:
            if isinstance(result, str) and result.startswith("error:"):
                # Allow network errors but not thread safety errors
                error_msg = result.lower()
                thread_safety_indicators = [
                    "dictionary changed size during iteration",
                    "dictionary keys changed during iteration",
                    "dictionary changed during iteration",
                    "keyerror",  # If cache corruption causes missing keys
                    "runtimeerror",  # General thread safety errors
                ]
                for indicator in thread_safety_indicators:
                    assert indicator not in error_msg, f"Thread safety violation: {result}"
            else:
                # Valid detection results
                assert result in ["benchflow", "direct"], f"Invalid detection result: {result}"

        # Verify cache stats are consistent (no negative values or other corruption)
        stats_results = [r for r in all_results if r[0] == "stats"]
        for _op_type, _port, total_entries in stats_results:
            assert isinstance(total_entries, int), "Cache stats corrupted"
            assert total_entries >= 0, "Negative cache entries indicate corruption"

        # Final verification: cache should still be functional
        final_stats = get_detection_cache_stats(cache_strategy=self.test_cache)
        assert isinstance(final_stats, dict)
        assert final_stats.get("total_entries", 0) >= 0
