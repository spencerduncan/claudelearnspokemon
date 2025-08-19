#!/usr/bin/env python3
"""
Simple integration test demonstrating the enhanced CheckpointManager.

This validates that the integration is working without requiring Memgraph.
"""

import tempfile

import pytest

from claudelearnspokemon.checkpoint_manager import CheckpointManager


@pytest.mark.medium
def test_backward_compatibility():
    """Test that existing usage continues working."""
    print("Testing backward compatibility...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Original usage - no discovery backend
        manager = CheckpointManager(storage_dir=temp_dir, max_checkpoints=5, enable_metrics=True)

        # Verify no discovery backend
        assert manager.discovery_backend is None
        assert manager._circuit_breaker is None
        print("✅ No discovery backend configured")

        # Standard operations
        game_state = {"player": {"name": "RED", "level": 25}}
        metadata = {"location": "cerulean_city"}

        checkpoint_id = manager.save_checkpoint(game_state, metadata)
        loaded_state = manager.load_checkpoint(checkpoint_id)
        found_id = manager.find_nearest_checkpoint("cerulean_city")

        assert loaded_state == game_state
        assert found_id == checkpoint_id
        print("✅ All operations work exactly as before")

        # Metrics include discovery backend info
        metrics = manager.get_metrics()
        assert metrics["discovery_backend"]["enabled"] is False
        print("✅ Metrics correctly show no discovery backend")


@pytest.mark.medium
def test_mock_discovery_backend():
    """Test with mock discovery backend."""
    print("\nTesting with mock discovery backend...")

    class MockBackend:
        def find_nearest_checkpoint(self, location: str) -> str:
            return f"mock_checkpoint_for_{location}"

        def save_checkpoint_with_scoring(
            self, checkpoint_id: str, location: str, metadata: dict
        ) -> None:
            pass

        def get_performance_metrics(self) -> dict:
            return {"avg_discovery_time_ms": 2.5}

    with tempfile.TemporaryDirectory() as temp_dir:
        mock_backend = MockBackend()

        manager = CheckpointManager(
            storage_dir=temp_dir,
            discovery_backend=mock_backend,
            enable_discovery_backend=True,
            enable_metrics=True,
        )

        # Verify discovery backend configured
        assert manager.discovery_backend is mock_backend
        assert manager._circuit_breaker is not None
        print("✅ Discovery backend and circuit breaker configured")

        # Test discovery
        result = manager.find_nearest_checkpoint("test_location")
        assert result == "mock_checkpoint_for_test_location"
        print("✅ Discovery backend used for find_nearest_checkpoint")

        # Test metrics
        metrics = manager.get_metrics()
        assert metrics["discovery_backend"]["enabled"] is True
        assert metrics["discovery_backend"]["backend_type"] == "MockBackend"
        print("✅ Enhanced metrics include discovery backend data")


@pytest.mark.medium
def test_circuit_breaker_behavior():
    """Test circuit breaker functionality."""
    print("\nTesting circuit breaker behavior...")

    class FailingBackend:
        def find_nearest_checkpoint(self, location: str) -> str:
            raise Exception("Mock failure")

        def save_checkpoint_with_scoring(
            self, checkpoint_id: str, location: str, metadata: dict
        ) -> None:
            raise Exception("Mock save failure")

        def get_performance_metrics(self) -> dict:
            return {"error": "unavailable"}

    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(
            storage_dir=temp_dir,
            discovery_backend=FailingBackend(),
            enable_discovery_backend=True,
        )

        # Create checkpoint for SQLite fallback
        game_state = {"player": {"name": "RED"}}
        metadata = {"location": "test_city"}
        checkpoint_id = manager.save_checkpoint(game_state, metadata)
        print("✅ Checkpoint saved despite backend failures")

        # Trigger circuit breaker
        for i in range(4):
            result = manager.find_nearest_checkpoint("test_city")
            if i < 3:
                # First 3 calls fail but fallback to SQLite
                assert result == checkpoint_id
            else:
                # After circuit opens, still works via SQLite
                assert result == checkpoint_id

        # Verify circuit opened
        from claudelearnspokemon.checkpoint_manager import CircuitBreakerState

        assert manager._circuit_breaker.state == CircuitBreakerState.OPEN
        print("✅ Circuit breaker opened after failures, system still operational")


if __name__ == "__main__":
    print("Simple Integration Test")
    print("======================")

    test_backward_compatibility()
    test_mock_discovery_backend()
    test_circuit_breaker_behavior()

    print("\n🎉 All integration tests passed!")
    print("\nProduction-ready features validated:")
    print("• Backward compatibility maintained")
    print("• Discovery backend integration working")
    print("• Circuit breaker protecting system")
    print("• Graceful fallback to SQLite")
    print("• Comprehensive metrics and observability")
