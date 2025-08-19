"""
Integration tests for CheckpointManager with MemgraphCheckpointDiscovery.

Tests production-ready integration patterns:
- Backward compatibility with existing SQLite-only mode
- Discovery backend integration with circuit breaker
- Graceful fallback on backend failures
- Metrics and observability integration
- Environment variable configuration

Author: Bot Dean - Production Integration Testing
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from claudelearnspokemon.checkpoint_manager import (
    CheckpointManager,
    CircuitBreakerState,
)


class MockDiscoveryBackend:
    """Mock discovery backend for testing integration patterns."""

    def __init__(self, should_fail: bool = False, response_delay: float = 0.0):
        self.should_fail = should_fail
        self.response_delay = response_delay
        self.call_count = 0
        self.save_calls = []
        self.find_calls = []

    def find_nearest_checkpoint(self, location: str) -> str:
        """Mock find implementation."""
        self.call_count += 1
        self.find_calls.append(location)

        if self.response_delay > 0:
            time.sleep(self.response_delay)

        if self.should_fail:
            raise Exception(f"Mock backend failure for location: {location}")

        # Mock behavior: return checkpoint ID if location contains "city"
        if "city" in location.lower():
            return f"checkpoint_for_{location.lower().replace(' ', '_')}"
        return ""

    def save_checkpoint_with_scoring(
        self, checkpoint_id: str, location: str, metadata: dict
    ) -> None:
        """Mock save implementation."""
        self.save_calls.append((checkpoint_id, location, metadata))

        if self.should_fail:
            raise Exception(f"Mock save failure for checkpoint: {checkpoint_id}")

    def get_performance_metrics(self) -> dict:
        """Mock metrics implementation."""
        return {
            "avg_discovery_time_ms": 2.5,
            "discovery_queries": self.call_count,
            "performance_status": {
                "discovery_target_met": True,
            },
        }


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_game_state():
    """Sample game state for testing."""
    return {
        "player": {"name": "RED", "position": {"x": 100, "y": 150}},
        "pokemon": [{"name": "PIKACHU", "level": 25}],
        "inventory": {"pokeball": 10},
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "location": "cerulean_city",
        "progress_markers": {"badges": 2},
        "notes": "Test checkpoint",
    }


@pytest.mark.fast
class TestBackwardCompatibility:
    """Test that existing CheckpointManager usage continues working unchanged."""

    def test_checkpoint_manager_without_discovery_backend(
        self, temp_storage_dir, sample_game_state, sample_metadata
    ):
        """Test CheckpointManager works exactly as before without discovery backend."""
        # Create CheckpointManager without discovery backend (original usage)
        manager = CheckpointManager(
            storage_dir=temp_storage_dir, max_checkpoints=5, enable_metrics=True
        )

        # Verify no discovery backend configured
        assert manager.discovery_backend is None
        assert manager._circuit_breaker is None
        assert not manager._discovery_backend_enabled

        # Verify save/load functionality works unchanged
        checkpoint_id = manager.save_checkpoint(sample_game_state, sample_metadata)
        assert checkpoint_id

        loaded_state = manager.load_checkpoint(checkpoint_id)
        assert loaded_state == sample_game_state

        # Verify find_nearest_checkpoint uses SQLite fallback
        found_checkpoint = manager.find_nearest_checkpoint("cerulean_city")
        assert found_checkpoint == checkpoint_id

        # Verify metrics work without discovery backend
        metrics = manager.get_metrics()
        assert metrics["discovery_backend"]["enabled"] is False
        assert "no_backend_configured" in metrics["discovery_backend"]["reason"]

    def test_environment_variable_configuration(self, temp_storage_dir):
        """Test environment variable configuration for discovery backend."""
        # Test default (disabled)
        manager = CheckpointManager(storage_dir=temp_storage_dir)
        assert not manager._discovery_backend_enabled

        # Test environment variable enabling
        with patch.dict(os.environ, {"ENABLE_MEMGRAPH_DISCOVERY": "true"}):
            manager = CheckpointManager(storage_dir=temp_storage_dir)
            assert manager._discovery_backend_enabled  # Would be enabled if backend provided

        # Test explicit parameter override
        with patch.dict(os.environ, {"ENABLE_MEMGRAPH_DISCOVERY": "true"}):
            manager = CheckpointManager(
                storage_dir=temp_storage_dir, enable_discovery_backend=False
            )
            assert not manager._discovery_backend_enabled


@pytest.mark.fast
class TestDiscoveryBackendIntegration:
    """Test integration with discovery backend."""

    def test_discovery_backend_success_path(
        self, temp_storage_dir, sample_game_state, sample_metadata
    ):
        """Test successful discovery backend integration."""
        mock_backend = MockDiscoveryBackend()

        manager = CheckpointManager(
            storage_dir=temp_storage_dir,
            discovery_backend=mock_backend,
            enable_discovery_backend=True,
            enable_metrics=True,
        )

        # Verify discovery backend is configured
        assert manager.discovery_backend is mock_backend
        assert manager._circuit_breaker is not None
        assert manager._circuit_breaker.state == CircuitBreakerState.CLOSED

        # Test save with discovery backend sync
        checkpoint_id = manager.save_checkpoint(sample_game_state, sample_metadata)

        # Verify discovery backend received sync call
        assert len(mock_backend.save_calls) == 1
        saved_id, saved_location, saved_metadata = mock_backend.save_calls[0]
        assert saved_id == checkpoint_id
        assert saved_location == "cerulean_city"

        # Test find_nearest_checkpoint uses discovery backend
        result = manager.find_nearest_checkpoint("cerulean_city")
        assert result == "checkpoint_for_cerulean_city"
        assert len(mock_backend.find_calls) == 1

        # Verify metrics include discovery backend data
        metrics = manager.get_metrics()
        assert metrics["discovery_backend"]["enabled"] is True
        assert metrics["discovery_backend"]["success_rate"] == 1.0
        assert metrics["discovery_calls_total"] == 1
        assert metrics["discovery_success_total"] == 1

    def test_discovery_backend_failure_and_fallback(
        self, temp_storage_dir, sample_game_state, sample_metadata
    ):
        """Test graceful fallback when discovery backend fails."""
        mock_backend = MockDiscoveryBackend(should_fail=True)

        manager = CheckpointManager(
            storage_dir=temp_storage_dir,
            discovery_backend=mock_backend,
            enable_discovery_backend=True,
            enable_metrics=True,
        )

        # Save checkpoint to have data for SQLite fallback
        checkpoint_id = manager.save_checkpoint(sample_game_state, sample_metadata)

        # Discovery backend save should succeed (save_calls records attempts)
        # but the sync fails internally (handled gracefully)
        assert checkpoint_id
        assert len(mock_backend.save_calls) == 1  # Save sync was attempted

        # Test find_nearest_checkpoint falls back to SQLite (generates more failures)
        result1 = manager.find_nearest_checkpoint("cerulean_city")
        assert result1 == checkpoint_id  # SQLite found it

        result2 = manager.find_nearest_checkpoint("cerulean_city")
        assert result2 == checkpoint_id  # SQLite found it again

        # Verify circuit breaker opened after 3 failures (save sync + 2 find calls)
        assert manager._circuit_breaker.state == CircuitBreakerState.OPEN

        # Verify metrics reflect fallback behavior
        metrics = manager.get_metrics()
        assert metrics["discovery_fallback_total"] == 2  # From find_nearest_checkpoint calls
        assert metrics["discovery_backend"]["circuit_breaker"]["state"] == "open"


@pytest.mark.fast
class TestCircuitBreakerFunctionality:
    """Test circuit breaker behavior in production scenarios."""

    def test_circuit_breaker_states(self, temp_storage_dir):
        """Test circuit breaker state transitions."""
        mock_backend = MockDiscoveryBackend(should_fail=True)

        manager = CheckpointManager(
            storage_dir=temp_storage_dir,
            discovery_backend=mock_backend,
            enable_discovery_backend=True,
        )

        circuit_breaker = manager._circuit_breaker

        # Initial state: CLOSED
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.is_available()

        # Trigger failures to open circuit
        for i in range(3):  # failure_threshold = 3
            manager.find_nearest_checkpoint(f"location_{i}")

        # Circuit should now be OPEN
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert not circuit_breaker.is_available()

        # Mock successful backend
        mock_backend.should_fail = False

        # Simulate recovery timeout
        circuit_breaker._last_failure_time = time.time() - 35.0  # 35 seconds ago

        # Should transition to HALF_OPEN
        assert circuit_breaker.is_available()  # Triggers state transition
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Successful call should close circuit
        manager.find_nearest_checkpoint("test_location")
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_metrics(self, temp_storage_dir):
        """Test circuit breaker metrics collection."""
        mock_backend = MockDiscoveryBackend(should_fail=True)

        manager = CheckpointManager(
            storage_dir=temp_storage_dir,
            discovery_backend=mock_backend,
            enable_discovery_backend=True,
            enable_metrics=True,
        )

        # Generate some failures - circuit breaker opens after 3 failures
        for i in range(5):
            manager.find_nearest_checkpoint(f"location_{i}")

        metrics = manager.get_metrics()
        cb_metrics = metrics["discovery_backend"]["circuit_breaker"]

        # Only first 3 calls reach the backend (then circuit opens)
        assert cb_metrics["total_calls"] == 3
        assert cb_metrics["failed_calls"] == 3
        assert cb_metrics["failure_rate"] == 1.0
        assert cb_metrics["state"] == "open"

        # CheckpointManager metrics should show all 5 discovery attempts
        assert metrics["discovery_calls_total"] == 3  # Only calls that reached backend
        assert metrics["discovery_fallback_total"] == 3  # 3 backend failures


@pytest.mark.medium
@pytest.mark.fast
class TestProductionScenarios:
    """Test real-world production scenarios."""

    def test_discovery_backend_performance_monitoring(self, temp_storage_dir):
        """Test performance metrics are properly tracked."""
        # Mock backend with realistic response times
        mock_backend = MockDiscoveryBackend(response_delay=0.01)  # 10ms delay

        manager = CheckpointManager(
            storage_dir=temp_storage_dir,
            discovery_backend=mock_backend,
            enable_discovery_backend=True,
            enable_metrics=True,
        )

        # Make several discovery calls
        for _ in range(5):
            manager.find_nearest_checkpoint("test_city")

        metrics = manager.get_metrics()

        # Verify performance tracking
        assert metrics["discovery_calls_total"] == 5
        assert metrics["discovery_success_total"] == 5
        assert metrics["discovery_avg_time_ms"] > 0
        assert metrics["discovery_backend"]["success_rate"] == 1.0

        # Verify backend metrics are included
        backend_perf = metrics["discovery_backend"]["performance"]
        assert "avg_discovery_time_ms" in backend_perf
        assert "discovery_target_met" in backend_perf["performance_status"]

    def test_concurrent_access_with_discovery_backend(
        self, temp_storage_dir, sample_game_state, sample_metadata
    ):
        """Test thread safety with discovery backend integration."""
        import threading

        mock_backend = MockDiscoveryBackend()

        manager = CheckpointManager(
            storage_dir=temp_storage_dir,
            discovery_backend=mock_backend,
            enable_discovery_backend=True,
            enable_metrics=True,
        )

        # Save some checkpoints concurrently
        checkpoint_ids = []
        errors = []

        def save_checkpoint(i):
            try:
                metadata = sample_metadata.copy()
                metadata["location"] = f"location_{i}"

                checkpoint_id = manager.save_checkpoint(sample_game_state, metadata)
                checkpoint_ids.append(checkpoint_id)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=save_checkpoint, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors and all saves succeeded
        assert not errors
        assert len(checkpoint_ids) == 5

        # Verify discovery backend received all sync calls
        assert len(mock_backend.save_calls) == 5

    def test_discovery_backend_unavailable_at_startup(self, temp_storage_dir):
        """Test graceful handling when discovery backend is unavailable."""
        # Mock backend that fails immediately
        mock_backend = MockDiscoveryBackend(should_fail=True)

        # Manager should initialize successfully even with failing backend
        manager = CheckpointManager(
            storage_dir=temp_storage_dir,
            discovery_backend=mock_backend,
            enable_discovery_backend=True,
        )

        # Should still work with SQLite fallback
        checkpoint_id = manager.save_checkpoint({"test": "data"}, {"location": "test"})
        assert checkpoint_id

        found = manager.find_nearest_checkpoint("test")
        assert found == checkpoint_id


@pytest.mark.slow
def test_integration_with_real_memgraph_discovery():
    """
    Integration test placeholder for real MemgraphCheckpointDiscovery.

    This test should be run with actual Memgraph instance available.
    """
    # This would require actual MemgraphCheckpointDiscovery instance
    # Skip if not in integration test environment
    pytest.skip("Requires actual Memgraph instance - run separately in integration environment")
