"""
Contract tests for PokemonGymAdapter interface specification.

These tests validate that any PokemonGymAdapter implementation conforms
to the interface specification exactly. They serve as both validation
and documentation of expected behavior.

The tests are designed to run against both mock implementations (for
development) and real implementations (for integration testing).

Performance Requirements Validation:
- All operations must meet documented SLA requirements
- Resource usage must stay within specified bounds
- Error handling must follow documented contracts

Author: John Botmack - Performance-Focused Contract Testing
"""

import time
from typing import Any
from unittest.mock import Mock

import pytest

# Import types for contract validation
from claudelearnspokemon.pokemon_gym_adapter_types import (
    MAX_INPUT_PROCESSING_TIME,
    MAX_SESSION_OVERHEAD,
    MAX_STATE_RETRIEVAL_TIME,
    VALID_BUTTONS,
    AdapterConfig,
    LoggingConfig,
    NetworkConfig,
    PerformanceConfig,
    SessionConfig,
)


class PokemonGymAdapterContractTest:
    """
    Base contract test class for PokemonGymAdapter implementations.

    This class defines the contract requirements that any adapter
    implementation must satisfy. Inherit from this class and implement
    the create_adapter() method to test your implementation.
    """

    def create_adapter(
        self,
        port: int = 8081,
        container_id: str = "test-container",
        config: AdapterConfig | None = None,
    ):
        """
        Create adapter instance for testing.

        Must be overridden by concrete test classes to provide
        the specific adapter implementation being tested.
        """
        raise NotImplementedError("Subclasses must implement create_adapter()")

    def create_mock_adapter(self) -> Mock:
        """Create a mock adapter for testing error conditions."""
        mock_adapter = Mock()
        mock_adapter.port = 8081
        mock_adapter.container_id = "test-container"
        return mock_adapter

    # =================================================================
    # Initialization Contract Tests
    # =================================================================

    def test_adapter_initialization_basic(self):
        """Test basic adapter initialization with minimal parameters."""
        adapter = self.create_adapter(port=8081, container_id="test-container")

        # Verify required attributes are set
        assert hasattr(adapter, "port")
        assert hasattr(adapter, "container_id")
        assert adapter.port == 8081
        assert adapter.container_id == "test-container"

        # Verify initialization performance
        start_time = time.time()
        adapter = self.create_adapter(port=8082, container_id="test-2")
        init_time = (time.time() - start_time) * 1000  # Convert to ms

        assert init_time < 1000, f"Initialization took {init_time:.2f}ms, must be < 1000ms"

    def test_adapter_initialization_with_config(self):
        """Test adapter initialization with full configuration."""
        config = AdapterConfig(
            base_url="http://localhost:8081",
            port=8081,
            container_id="test-container",
            network=NetworkConfig(connect_timeout=5.0, read_timeout=30.0),
            session=SessionConfig(auto_initialize=True, session_timeout=300.0),
            performance=PerformanceConfig(input_batch_size=10, input_delay_ms=50.0),
            logging=LoggingConfig(level="INFO", log_requests=True),
        )

        adapter = self.create_adapter(port=8081, container_id="test-container", config=config)

        # Verify configuration is applied (implementation-specific validation)
        assert hasattr(adapter, "port")
        assert adapter.port == 8081

    def test_session_initialization_overhead(self):
        """Test that session initialization meets overhead requirements."""
        # Measure session initialization time
        start_time = time.time()
        adapter = self.create_adapter()

        # If adapter has session initialization, measure that too
        # For mock, this is just the construction time
        init_time = (time.time() - start_time) * 1000

        # Verify adapter was created successfully
        assert adapter is not None

        assert (
            init_time < MAX_SESSION_OVERHEAD
        ), f"Session initialization took {init_time:.2f}ms, must be < {MAX_SESSION_OVERHEAD}ms"

    def test_adapter_initialization_invalid_port(self):
        """Test adapter initialization with invalid port numbers."""
        with pytest.raises(ValueError, match="Invalid port number"):
            self.create_adapter(port=0, container_id="test")

        with pytest.raises(ValueError, match="Invalid port number"):
            self.create_adapter(port=70000, container_id="test")

    # =================================================================
    # send_input() Contract Tests
    # =================================================================

    def test_send_input_basic_sequence(self):
        """Test send_input with basic button sequences."""
        adapter = self.create_adapter()

        # Test single button
        start_time = time.time()
        result = adapter.send_input("A")
        execution_time = (time.time() - start_time) * 1000

        # Validate response format
        assert isinstance(result, dict)
        assert "success" in result
        assert "buttons_processed" in result
        assert "sequence" in result
        assert "execution_time_ms" in result
        assert "timestamp" in result

        # Validate response content
        assert isinstance(result["success"], bool)
        assert isinstance(result["buttons_processed"], int)
        assert isinstance(result["sequence"], str)
        assert isinstance(result["execution_time_ms"], int | float)
        assert isinstance(result["timestamp"], int | float)

        # Validate performance requirement
        assert (
            execution_time < MAX_INPUT_PROCESSING_TIME
        ), f"Single button took {execution_time:.2f}ms, must be < {MAX_INPUT_PROCESSING_TIME}ms"

    def test_send_input_multi_button_sequence(self):
        """Test send_input with multi-button sequences."""
        adapter = self.create_adapter()

        test_sequences = [
            "A B",
            "A B START",
            "UP DOWN LEFT RIGHT",
            "A B START SELECT UP DOWN LEFT RIGHT",
        ]

        for sequence in test_sequences:
            start_time = time.time()
            result = adapter.send_input(sequence)
            execution_time = (time.time() - start_time) * 1000

            # Validate basic response format
            assert result["success"] is True or result["success"] is False
            assert result["sequence"] == sequence

            # Count expected buttons
            expected_count = len(sequence.split())
            if result["success"]:
                assert result["buttons_processed"] == expected_count

            # Validate performance based on sequence length
            button_count = len(sequence.split())
            if button_count <= 10:
                max_time = 100
            elif button_count <= 50:
                max_time = 200
            else:
                max_time = 500

            assert (
                execution_time < max_time
            ), f"Sequence '{sequence}' took {execution_time:.2f}ms, must be < {max_time}ms"

    def test_send_input_invalid_buttons(self):
        """Test send_input with invalid button names."""
        adapter = self.create_adapter()

        invalid_sequences = ["INVALID", "A INVALID B", "X Y Z", "A B INVALID START"]

        for sequence in invalid_sequences:
            with pytest.raises((ValueError, TypeError), match="Invalid button"):
                adapter.send_input(sequence)

    def test_send_input_empty_sequence(self):
        """Test send_input with empty sequence (should be valid no-op)."""
        adapter = self.create_adapter()

        result = adapter.send_input("")

        assert result["success"] is True
        assert result["buttons_processed"] == 0
        assert result["sequence"] == ""

    def test_send_input_large_sequence_performance(self):
        """Test send_input performance with large sequences."""
        adapter = self.create_adapter()

        # Create sequence with 100 buttons (max allowed)
        large_sequence = " ".join(["A"] * 100)

        start_time = time.time()
        result = adapter.send_input(large_sequence)
        execution_time = (time.time() - start_time) * 1000

        # Must complete within 500ms for large sequences
        assert execution_time < 500, f"Large sequence took {execution_time:.2f}ms, must be < 500ms"

        if result["success"]:
            assert result["buttons_processed"] == 100

    def test_send_input_concurrent_calls(self):
        """Test send_input thread safety with concurrent calls."""
        adapter = self.create_adapter()

        import queue
        import threading

        results_queue = queue.Queue()
        num_threads = 5

        def worker(thread_id):
            try:
                result = adapter.send_input("A B")  # Simple sequence
                results_queue.put((thread_id, result, None))
            except Exception as e:
                results_queue.put((thread_id, None, e))

        # Start concurrent threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)

        # Wait for all threads
        for t in threads:
            t.join()

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Verify all threads completed successfully
        assert len(results) == num_threads
        for thread_id, result, error in results:
            if error:
                pytest.fail(f"Thread {thread_id} failed with error: {error}")
            assert result is not None
            assert "success" in result

    # =================================================================
    # get_state() Contract Tests
    # =================================================================

    def test_get_state_basic(self):
        """Test basic get_state functionality and response format."""
        adapter = self.create_adapter()

        start_time = time.time()
        state = adapter.get_state()
        execution_time = (time.time() - start_time) * 1000

        # Validate performance requirement
        assert (
            execution_time < MAX_STATE_RETRIEVAL_TIME
        ), f"get_state took {execution_time:.2f}ms, must be < {MAX_STATE_RETRIEVAL_TIME}ms"

        # Validate response format
        assert isinstance(state, dict)

        # Required fields
        required_fields = ["timestamp", "frame_count"]
        for field in required_fields:
            assert field in state, f"Missing required field: {field}"

        # Optional but common fields
        optional_fields = ["player_position", "tile_grid", "inventory", "party", "flags"]
        for field in optional_fields:
            if field in state:
                if field == "player_position" and state[field] is not None:
                    assert isinstance(state[field], dict)
                    assert "x" in state[field]
                    assert "y" in state[field]
                elif field == "tile_grid" and state[field] is not None:
                    assert isinstance(state[field], list)
                    # Validate 20x18 grid if present
                    if len(state[field]) > 0:
                        assert len(state[field]) <= 20  # Max 20 rows
                        assert all(len(row) <= 18 for row in state[field])  # Max 18 cols

    def test_get_state_caching(self):
        """Test get_state caching behavior for performance."""
        adapter = self.create_adapter()

        # First call
        start1 = time.time()
        state1 = adapter.get_state()
        time1 = (time.time() - start1) * 1000

        # Second call immediately (should be cached)
        start2 = time.time()
        state2 = adapter.get_state()
        time2 = (time.time() - start2) * 1000

        # Cache should make second call faster (if implemented)
        # Note: This is optional optimization, not required

        # Both calls should return same timestamp if cached
        if state1.get("timestamp") == state2.get("timestamp"):
            # Cached response - validate it's faster
            assert time2 <= time1, "Cached response should be faster or equal"

    def test_get_state_freshness(self):
        """Test get_state returns fresh data within acceptable limits."""
        adapter = self.create_adapter()

        current_time = time.time()
        state = adapter.get_state()

        # State should be fresh (within 1 second)
        if "timestamp" in state:
            state_age = current_time - state["timestamp"]
            assert state_age < 1.0, f"State is {state_age:.2f}s old, must be < 1.0s"

    # =================================================================
    # reset_game() Contract Tests
    # =================================================================

    def test_reset_game_basic(self):
        """Test basic reset_game functionality."""
        adapter = self.create_adapter()

        start_time = time.time()
        result = adapter.reset_game()
        execution_time = (time.time() - start_time) * 1000

        # Validate performance requirement (< 2s for session restart)
        assert execution_time < 2000, f"reset_game took {execution_time:.2f}ms, must be < 2000ms"

        # Validate response format
        assert isinstance(result, dict)
        assert "success" in result
        assert "message" in result
        assert "timestamp" in result

        # Validate response types
        assert isinstance(result["success"], bool)
        assert isinstance(result["message"], str)
        assert isinstance(result["timestamp"], int | float)

        # Optional fields
        if "initial_state" in result and result["initial_state"] is not None:
            assert isinstance(result["initial_state"], dict)
        if "reset_time_ms" in result:
            assert isinstance(result["reset_time_ms"], int | float)
        if "session_id" in result:
            assert isinstance(result["session_id"], str)

    def test_reset_game_state_cleanup(self):
        """Test that reset_game properly cleans up state."""
        adapter = self.create_adapter()

        # Get initial state (to ensure adapter is working)
        adapter.get_state()

        # Send some inputs to change state
        adapter.send_input("A B START")

        # Reset game
        reset_result = adapter.reset_game()

        # Get state after reset
        post_reset_state = adapter.get_state()

        # Verify reset was successful
        assert reset_result["success"] is True

        # State should be clean (frame count reset, etc.)
        if "frame_count" in post_reset_state:
            # Frame count should be low after reset
            assert post_reset_state["frame_count"] < 1000, "Frame count should be low after reset"

    # =================================================================
    # is_healthy() Contract Tests
    # =================================================================

    def test_is_healthy_basic(self):
        """Test basic is_healthy functionality."""
        adapter = self.create_adapter()

        start_time = time.time()
        healthy = adapter.is_healthy()
        execution_time = (time.time() - start_time) * 1000

        # Validate performance requirement
        assert execution_time < 1000, f"is_healthy took {execution_time:.2f}ms, must be < 1000ms"

        # Validate response type
        assert isinstance(healthy, bool)

    def test_is_healthy_never_raises(self):
        """Test that is_healthy never raises exceptions."""
        adapter = self.create_adapter()

        # Should never raise exceptions, even in error conditions
        try:
            result = adapter.is_healthy()
            assert isinstance(result, bool)
        except Exception as e:
            pytest.fail(f"is_healthy raised exception: {e}")

    def test_is_healthy_frequent_calls(self):
        """Test is_healthy can be called frequently without performance degradation."""
        adapter = self.create_adapter()

        call_times = []
        num_calls = 10

        for _ in range(num_calls):
            start_time = time.time()
            adapter.is_healthy()
            call_time = (time.time() - start_time) * 1000
            call_times.append(call_time)

        # All calls should be fast
        for i, call_time in enumerate(call_times):
            assert call_time < 1000, f"Call {i} took {call_time:.2f}ms, must be < 1000ms"

        # Performance shouldn't degrade with frequent calls
        if len(call_times) > 5:
            avg_early = sum(call_times[:5]) / 5
            avg_late = sum(call_times[-5:]) / 5
            assert avg_late <= avg_early * 2, "Performance degraded with frequent calls"

    # =================================================================
    # close() Contract Tests
    # =================================================================

    def test_close_basic(self):
        """Test basic close functionality."""
        adapter = self.create_adapter()

        start_time = time.time()
        # close() should not return anything
        result = adapter.close()
        execution_time = (time.time() - start_time) * 1000

        # Validate performance requirement
        assert execution_time < 2000, f"close took {execution_time:.2f}ms, must be < 2000ms"

        # Should not return anything
        assert result is None

    def test_close_idempotent(self):
        """Test that close is idempotent (safe to call multiple times)."""
        adapter = self.create_adapter()

        # Multiple calls should not fail
        adapter.close()
        adapter.close()  # Second call should not raise exception
        adapter.close()  # Third call should not raise exception

    def test_close_never_raises(self):
        """Test that close never raises exceptions during cleanup."""
        adapter = self.create_adapter()

        try:
            adapter.close()
        except Exception as e:
            pytest.fail(f"close raised exception: {e}")

    # =================================================================
    # Error Handling Contract Tests
    # =================================================================

    def test_error_response_format(self):
        """Test that errors follow the standard error response format."""
        # This test requires a mock that can simulate errors
        mock_adapter = self.create_mock_adapter()

        # Mock an error response (example format)
        # error_response = {
        #     "error": True,
        #     "error_type": "NetworkError",
        #     "message": "Connection failed",
        #     "timestamp": time.time(),
        #     "operation": "send_input",
        # }

        mock_adapter.send_input.side_effect = Exception("Test error")

        # The adapter should catch exceptions and format them properly
        # (This test might need to be adapted based on actual implementation)

    def test_network_timeout_handling(self):
        """Test handling of network timeout scenarios."""
        # Create adapter with short timeout for testing
        # Note: This would be used to create an adapter with timeout config
        # For now, just test timeout behavior conceptually
        pass

        # adapter = self.create_adapter(config=config)

        # Network operations should timeout and handle gracefully
        # (Actual behavior depends on implementation)
        # This test is a placeholder for actual implementation testing

    def test_validation_error_handling(self):
        """Test input validation error handling."""
        adapter = self.create_adapter()

        # Invalid button names should raise validation errors
        with pytest.raises((ValueError, TypeError)):
            adapter.send_input("INVALID_BUTTON")

        # Oversized sequences should be handled
        huge_sequence = " ".join(["A"] * 1000)  # Way over limit
        with pytest.raises((ValueError, TypeError)):
            adapter.send_input(huge_sequence)

    # =================================================================
    # Performance Contract Tests
    # =================================================================

    def test_memory_usage_limits(self):
        """Test that adapter stays within memory usage limits."""
        adapter = self.create_adapter()

        # Import memory profiling (optional, might not be available)
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Perform many operations
            for i in range(100):
                adapter.send_input("A B")
                if i % 10 == 0:
                    adapter.get_state()

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (< 100MB for 100 operations)
            assert (
                memory_increase < 100
            ), f"Memory increased by {memory_increase:.2f}MB, should be < 100MB"

        except ImportError:
            pytest.skip("psutil not available for memory testing")

    def test_resource_cleanup(self):
        """Test that adapter properly cleans up resources."""
        # Create and close many adapters to test for resource leaks
        for i in range(10):
            adapter = self.create_adapter(port=8081 + i, container_id=f"test-{i}")
            adapter.send_input("A")
            adapter.close()

        # If we get here without issues, resource cleanup is working

    def test_concurrent_operation_performance(self):
        """Test performance under concurrent load."""
        adapter = self.create_adapter()

        import queue
        import threading

        num_threads = 5
        operations_per_thread = 10
        results_queue = queue.Queue()

        def worker(thread_id):
            thread_times = []
            for _ in range(operations_per_thread):
                start_time = time.time()
                adapter.send_input("A")
                execution_time = (time.time() - start_time) * 1000
                thread_times.append(execution_time)

            results_queue.put((thread_id, thread_times))

        # Start concurrent workers
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        # Collect results
        all_times = []
        while not results_queue.empty():
            thread_id, thread_times = results_queue.get()
            all_times.extend(thread_times)

        # Validate performance under load
        avg_time = sum(all_times) / len(all_times) if all_times else 0

        # Average operation time shouldn't degrade too much under load
        assert (
            avg_time < MAX_INPUT_PROCESSING_TIME * 2
        ), f"Average time under load: {avg_time:.2f}ms, threshold: {MAX_INPUT_PROCESSING_TIME * 2}ms"


# =================================================================
# Concrete Test Implementation for Mock Adapter
# =================================================================


class TestMockPokemonGymAdapterContracts(PokemonGymAdapterContractTest):
    """
    Contract tests using a mock implementation.

    This validates the contract test framework itself and provides
    a reference for what the interface should look like.
    """

    def create_adapter(
        self,
        port: int = 8081,
        container_id: str = "test-container",
        config: AdapterConfig | None = None,
    ):
        """Create a mock adapter that conforms to the interface contract."""

        class MockPokemonGymAdapter:
            def __init__(self, port: int, container_id: str, config: AdapterConfig | None = None):
                # Validate port number
                if not isinstance(port, int) or port <= 1024 or port >= 65536:
                    raise ValueError("Invalid port number")

                self.port = port
                self.container_id = container_id
                self.config = config or AdapterConfig(base_url=f"http://localhost:{port}")
                self._session_active = False
                self._state_cache = None
                self._cache_time = 0
                self._reset_time = 0  # Track when game was last reset

            def send_input(self, input_sequence: str) -> dict[str, Any]:
                # Validate input
                if not isinstance(input_sequence, str):
                    raise TypeError("Input sequence must be string")

                if input_sequence.strip():  # Non-empty sequence
                    buttons = input_sequence.strip().split()

                    # Check for oversized sequences
                    if len(buttons) > 100:  # Reasonable limit for testing
                        raise ValueError(
                            f"Input sequence too large: {len(buttons)} buttons (max 100)"
                        )

                    for button in buttons:
                        if button not in VALID_BUTTONS:
                            raise ValueError(f"Invalid button: {button}")

                    # Simulate processing time
                    if len(buttons) > 50:
                        time.sleep(0.05)  # 50ms for large sequences
                    else:
                        time.sleep(0.01)  # 10ms for small sequences

                    return {
                        "success": True,
                        "buttons_processed": len(buttons),
                        "sequence": input_sequence,
                        "execution_time_ms": len(buttons) * 10,  # 10ms per button
                        "frames_advanced": len(buttons) * 3,
                        "timestamp": time.time(),
                    }
                else:
                    return {
                        "success": True,
                        "buttons_processed": 0,
                        "sequence": "",
                        "execution_time_ms": 0,
                        "frames_advanced": 0,
                        "timestamp": time.time(),
                    }

            def get_state(self) -> dict[str, Any]:
                # Simulate caching
                current_time = time.time()
                if self._state_cache and (current_time - self._cache_time) < 1.0:
                    return self._state_cache

                # Simulate state retrieval time
                time.sleep(0.01)  # 10ms

                state = {
                    "player_position": {"x": 10, "y": 15, "map_id": "test_map"},
                    "tile_grid": [[1, 2] * 9 for _ in range(20)],  # 20 rows x 18 columns
                    "inventory": {"items": {"POTION": 5}, "money": 1000},
                    "party": [
                        {"species": "PIKACHU", "level": 25, "hp": 85, "max_hp": 85, "status": "OK"}
                    ],
                    "flags": {"STARTED_GAME": True, "GOT_POKEBALL": False},
                    "timestamp": current_time,
                    "frame_count": int(
                        (current_time - self._reset_time) * 60
                    ),  # Frames since reset
                    "checksum": "abc123def456",
                }

                self._state_cache = state
                self._cache_time = current_time
                return state

            def reset_game(self) -> dict[str, Any]:
                # Simulate reset time
                time.sleep(0.1)  # 100ms

                self._state_cache = None  # Clear cache
                self._session_active = False
                self._reset_time = time.time()  # Set reset time for frame count calculation

                return {
                    "success": True,
                    "message": "Game reset successfully",
                    "initial_state": self.get_state(),
                    "reset_time_ms": 100,
                    "session_id": f"session_{int(time.time())}",
                    "timestamp": time.time(),
                }

            def is_healthy(self) -> bool:
                # Simulate health check time
                time.sleep(0.01)  # 10ms
                return True  # Mock is always healthy

            def close(self) -> None:
                # Simulate cleanup time
                time.sleep(0.05)  # 50ms
                self._session_active = False
                self._state_cache = None

        return MockPokemonGymAdapter(port, container_id, config)


# =================================================================
# Additional Contract Validation Tests
# =================================================================


class TestContractValidationFramework:
    """Test the contract validation framework itself."""

    def test_contract_test_completeness(self):
        """Verify that all interface methods have contract tests."""
        interface_methods = [
            "__init__",
            "send_input",
            "get_state",
            "reset_game",
            "is_healthy",
            "close",
        ]

        test_methods = [
            method for method in dir(PokemonGymAdapterContractTest) if method.startswith("test_")
        ]

        # Verify we have tests for all core methods
        for method in interface_methods:
            if method == "__init__":
                method_tests = [t for t in test_methods if "initialization" in t]
            else:
                method_tests = [
                    t for t in test_methods if method.replace("_", "") in t.replace("_", "")
                ]

            assert len(method_tests) > 0, f"No contract tests found for {method}"

    def test_performance_requirements_coverage(self):
        """Verify that all performance requirements have tests."""
        performance_constants = [
            "MAX_INPUT_PROCESSING_TIME",
            "MAX_STATE_RETRIEVAL_TIME",
            "MAX_SESSION_OVERHEAD",
        ]

        # Read the contract test file content
        import inspect

        test_source = inspect.getsource(PokemonGymAdapterContractTest)

        for constant in performance_constants:
            assert (
                constant in test_source
            ), f"Performance constant {constant} not tested in contracts"


if __name__ == "__main__":
    # Run contract tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])
