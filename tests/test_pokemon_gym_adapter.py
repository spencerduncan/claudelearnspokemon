"""
Comprehensive unit tests for PokemonGymAdapter.

Tests the adapter pattern implementation that bridges our EmulatorPool interface
with the benchflow-ai/pokemon-gym server API.

Following Uncle Bob's TDD principles:
- Write failing tests first (RED)
- Implement minimal code to pass (GREEN)
- Refactor for clean code (REFACTOR)

Author: Uncle Bot - Software Craftsmanship Applied
"""

import json
import time
from unittest.mock import patch

import pytest
import responses

from claudelearnspokemon.pokemon_gym_adapter import (
    PokemonGymAdapter,
    PokemonGymAdapterError,
    SessionManager,
)


class TestPokemonGymAdapterInitialization:
    """Test adapter initialization and configuration."""

    def test_adapter_initializes_with_port_and_container_id(self):
        """Test adapter initializes correctly with required parameters."""
        adapter = PokemonGymAdapter(8080, "test-container-123")

        assert adapter.port == 8080
        assert adapter.container_id == "test-container-123"
        assert adapter.base_url == "http://localhost:8080"
        assert adapter.session is not None
        assert isinstance(adapter.session_manager, SessionManager)

    def test_adapter_initializes_with_custom_config(self):
        """Test adapter accepts custom configuration parameters."""
        config = {"rom_path": "/path/to/pokemon.gb", "save_state": "initial.save", "headless": True}

        adapter = PokemonGymAdapter(8081, "custom-container", config=config)

        assert adapter.config == config
        assert adapter.port == 8081

    def test_adapter_sets_reasonable_timeouts(self):
        """Test adapter configures appropriate HTTP timeouts."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Verify timeout configuration
        assert hasattr(adapter, "input_timeout")
        assert hasattr(adapter, "state_timeout")
        assert adapter.input_timeout <= 10.0  # Reasonable input timeout
        assert adapter.state_timeout <= 5.0  # Fast state queries


class TestInputTranslation:
    """Test input sequence translation from batch to sequential actions."""

    @responses.activate
    def test_send_input_translates_single_button(self):
        """Test single button input translates to one API call."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock session initialization
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "test-session-123", "status": "initialized"},
            status=200,
        )

        # Mock button press action
        responses.add(
            responses.POST,
            "http://localhost:8080/action",
            json={"status": "success", "frame_count": 1},
            status=200,
        )

        result = adapter.send_input("A")

        assert len(responses.calls) == 2  # initialize + action
        action_call = responses.calls[1]
        action_body = json.loads(action_call.request.body)

        assert action_body == {"action_type": "press_key", "keys": ["A"]}
        assert result["status"] == "success"

    @responses.activate
    def test_send_input_translates_multiple_buttons(self):
        """Test multiple button sequence translates to sequential API calls."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock session initialization
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "test-session-456", "status": "initialized"},
            status=200,
        )

        # Mock each button press
        for _ in range(3):  # A, B, START
            responses.add(
                responses.POST,
                "http://localhost:8080/action",
                json={"status": "success", "frame_count": 1},
                status=200,
            )

        adapter.send_input("A B START")

        # Should be 1 initialize + 3 actions = 4 total calls
        assert len(responses.calls) == 4

        # Verify each action call
        expected_buttons = ["A", "B", "START"]
        for i, expected_button in enumerate(expected_buttons):
            action_call = responses.calls[i + 1]  # Skip initialize call
            action_body = json.loads(action_call.request.body)
            assert action_body == {"action_type": "press_key", "keys": [expected_button]}

    @responses.activate
    def test_send_input_handles_direction_inputs(self):
        """Test directional inputs are translated correctly."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock session and actions
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "test", "status": "initialized"},
        )
        responses.add(responses.POST, "http://localhost:8080/action", json={"status": "success"})
        responses.add(responses.POST, "http://localhost:8080/action", json={"status": "success"})

        adapter.send_input("UP DOWN")

        # Verify directional inputs
        up_call = json.loads(responses.calls[1].request.body)
        down_call = json.loads(responses.calls[2].request.body)

        assert up_call["keys"] == ["UP"]
        assert down_call["keys"] == ["DOWN"]

    def test_input_parsing_handles_empty_input(self):
        """Test empty input sequence is handled gracefully."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Should not make any action calls for empty input
        with responses.RequestsMock() as rsps:
            adapter.send_input("")
            assert len(rsps.calls) == 0

    def test_input_parsing_normalizes_whitespace(self):
        """Test input parsing handles extra whitespace."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Test internal parsing logic
        parsed = adapter._parse_input_sequence("  A   B  START  ")
        expected = ["A", "B", "START"]
        assert parsed == expected


class TestStateMapping:
    """Test state mapping from benchflow-ai format to our expected format."""

    @responses.activate
    def test_get_state_maps_benchflow_response(self):
        """Test get_state maps benchflow-ai status to our expected format."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock session initialization
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "test-session", "status": "initialized"},
            status=200,
        )

        # Mock benchflow-ai status response
        benchflow_response = {
            "game_status": "running",
            "player": {"x": 10, "y": 15, "map_id": "pallet_town"},
            "screen": {"tiles": [[1, 2, 3], [4, 5, 6]], "width": 20, "height": 18},
            "frame_count": 1205,
        }

        responses.add(
            responses.GET, "http://localhost:8080/status", json=benchflow_response, status=200
        )

        result = adapter.get_state()

        # Verify mapping to our expected format
        assert "game_status" in result
        assert "player_position" in result
        assert "screen_data" in result
        assert "frame_count" in result

        # Verify specific mappings
        assert result["game_status"] == "running"
        assert result["player_position"] == {"x": 10, "y": 15, "map_id": "pallet_town"}
        assert result["screen_data"]["tiles"] == [[1, 2, 3], [4, 5, 6]]
        assert result["frame_count"] == 1205

    @responses.activate
    def test_get_state_handles_missing_fields(self):
        """Test state mapping gracefully handles missing fields."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock session initialization
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "test-session", "status": "initialized"},
            status=200,
        )

        # Minimal benchflow response
        minimal_response = {"game_status": "paused"}

        responses.add(
            responses.GET, "http://localhost:8080/status", json=minimal_response, status=200
        )

        result = adapter.get_state()

        # Should provide sensible defaults for missing fields
        assert result["game_status"] == "paused"
        assert result["player_position"] == {}
        assert result["screen_data"] == {}
        assert result["frame_count"] == 0

    def test_state_mapping_defensive_parsing(self):
        """Test state mapping handles malformed data defensively."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Test internal mapping logic
        malformed_data = {
            "player": "not_a_dict",
            "screen": {"tiles": "not_a_list"},
            "frame_count": "not_a_number",
        }

        result = adapter._map_state_response(malformed_data)

        # Should not crash and provide safe defaults
        assert isinstance(result, dict)
        assert "player_position" in result
        assert "screen_data" in result


class TestSessionManagement:
    """Test session lifecycle management."""

    @responses.activate
    def test_session_auto_initialization(self):
        """Test session is automatically initialized on first API call."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock successful initialization
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "auto-session-789", "status": "initialized"},
            status=200,
        )
        responses.add(
            responses.GET, "http://localhost:8080/status", json={"game_status": "running"}
        )

        # First call should trigger initialization
        adapter.get_state()

        assert adapter.session_manager.session_id == "auto-session-789"
        assert adapter.session_manager.is_initialized

    @responses.activate
    def test_reset_game_stops_and_reinitializes(self):
        """Test reset_game implements stop/initialize sequence."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock session lifecycle
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "session-1", "status": "initialized"},
        )
        responses.add(
            responses.GET, "http://localhost:8080/status", json={"game_status": "running"}
        )
        responses.add(responses.POST, "http://localhost:8080/stop", json={"status": "stopped"})
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "session-2", "status": "initialized"},
        )

        # Initialize first session
        adapter.get_state()
        old_session = adapter.session_manager.session_id

        # Reset should stop and reinitialize
        result = adapter.reset_game()

        assert result["status"] == "initialized"
        assert adapter.session_manager.session_id == "session-2"
        assert adapter.session_manager.session_id != old_session

    @responses.activate
    def test_session_timeout_recovery(self):
        """Test automatic reconnection on session timeout."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # First initialization
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "session-1", "status": "initialized"},
        )

        # Simulate session timeout on action
        responses.add(
            responses.POST,
            "http://localhost:8080/action",
            json={"error": "session_expired"},
            status=400,
        )

        # Recovery initialization
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "session-2", "status": "initialized"},
        )
        responses.add(responses.POST, "http://localhost:8080/action", json={"status": "success"})

        # Should recover automatically
        result = adapter.send_input("A")

        assert result["status"] == "success"
        assert adapter.session_manager.session_id == "session-2"

    def test_session_manager_tracks_state(self):
        """Test SessionManager properly tracks session state."""
        session_manager = SessionManager("http://localhost:8080")

        assert not session_manager.is_initialized
        assert session_manager.session_id is None

        # Mock initialization
        with patch("requests.Session.post") as mock_post:
            mock_post.return_value.json.return_value = {
                "session_id": "test-session",
                "status": "initialized",
            }
            mock_post.return_value.raise_for_status.return_value = None

            session_manager.initialize_session()

            assert session_manager.is_initialized
            assert session_manager.session_id == "test-session"


class TestResetFunctionalityEnhancements:
    """Test enhanced reset functionality for Issue #143."""

    @responses.activate
    def test_reset_performance_under_500ms(self):
        """Test reset operation completes within 500ms performance target."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock fast responses for performance test
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "perf-session-1", "status": "initialized"},
        )
        responses.add(responses.POST, "http://localhost:8080/stop", json={"status": "stopped"})
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "perf-session-2", "status": "initialized"},
        )
        responses.add(
            responses.GET, "http://localhost:8080/status", json={"game_status": "running"}
        )

        # Initialize session first
        adapter._ensure_session_initialized()

        # Measure reset performance
        start_time = time.time()
        result = adapter.reset_game()
        elapsed_ms = (time.time() - start_time) * 1000

        # Should meet performance requirement
        assert elapsed_ms < 500, f"Reset took {elapsed_ms}ms, exceeds 500ms target"
        assert "operation_time_ms" in result
        assert result["operation_time_ms"] < 500
        assert "sla_exceeded" not in result or not result["sla_exceeded"]

    @responses.activate
    def test_reset_configuration_preservation(self):
        """Test configuration is preserved across resets."""
        original_config = {
            "rom_path": "/test/pokemon.gb",
            "save_state": "test.save",
            "headless": True,
        }
        adapter = PokemonGymAdapter(8080, "test-container", config=original_config)

        # Mock session lifecycle
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "config-session-1", "status": "initialized"},
        )
        responses.add(responses.POST, "http://localhost:8080/stop", json={"status": "stopped"})
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "config-session-2", "status": "initialized"},
        )
        responses.add(
            responses.GET, "http://localhost:8080/status", json={"game_status": "running"}
        )

        # Initialize and reset
        adapter._ensure_session_initialized()
        result = adapter.reset_game()

        # Verify configuration preservation
        assert adapter.config == original_config
        assert result["configuration_preserved"] is True
        assert "reset_details" in result

    @responses.activate
    def test_concurrent_reset_handling(self):
        """Test graceful handling of concurrent reset operations."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock session lifecycle with delays to simulate concurrent access
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "concurrent-session-1", "status": "initialized"},
        )
        responses.add(responses.POST, "http://localhost:8080/stop", json={"status": "stopped"})
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "concurrent-session-2", "status": "initialized"},
        )
        responses.add(
            responses.GET, "http://localhost:8080/status", json={"game_status": "running"}
        )

        # Initialize session
        adapter._ensure_session_initialized()

        import threading

        results = []
        errors = []

        def reset_worker():
            try:
                result = adapter.reset_game()
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start concurrent reset operations
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=reset_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should handle concurrent resets gracefully
        assert len(results) > 0, "At least one reset should succeed"
        assert len(errors) == 0, f"No errors expected, got: {errors}"

        # All successful results should have valid session IDs
        for result in results:
            assert result["status"] in [
                "initialized",
                "initialized",
            ]  # May include concurrent message
            assert "session_id" in result

    @responses.activate
    def test_reset_failure_recovery(self):
        """Test emergency recovery when reset fails."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock initial session
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "failure-session-1", "status": "initialized"},
        )

        # Mock stop success but initialization failure in reset
        responses.add(responses.POST, "http://localhost:8080/stop", json={"status": "stopped"})
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"error": "initialization_failed"},
            status=500,
        )

        # Mock emergency recovery success
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "recovery-session", "status": "initialized"},
        )

        # Initialize session first
        adapter._ensure_session_initialized()

        # Reset should trigger emergency recovery
        result = adapter.reset_game()

        assert result["status"] == "recovered"
        assert result["recovery_applied"] is True
        assert "original_error" in result
        assert adapter.session_manager.session_id == "recovery-session"

    @responses.activate
    def test_reset_complete_failure_cleanup(self):
        """Test clean state after complete reset failure."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock initial session
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "cleanup-session", "status": "initialized"},
        )

        # Mock stop success but all subsequent initialization attempts fail
        responses.add(responses.POST, "http://localhost:8080/stop", json={"status": "stopped"})

        # Mock initialization failure in reset
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"error": "initialization_failed"},
            status=500,
        )

        # Mock emergency recovery failure
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"error": "recovery_failed"},
            status=500,
        )

        # Initialize session first
        adapter._ensure_session_initialized()

        # Reset should fail but clean up state
        with pytest.raises(PokemonGymAdapterError) as exc_info:
            adapter.reset_game()

        # Should contain the final cleanup message
        assert "Session state has been reset to clean state" in str(exc_info.value)
        assert adapter.session_manager.is_initialized is False
        assert adapter.session_manager.session_id is None

    @responses.activate
    def test_reset_detailed_response_format(self):
        """Test reset response contains all required production details."""
        adapter = PokemonGymAdapter(8080, "test-container", config={"test": "value"})

        # Mock session lifecycle
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "detailed-session-1", "status": "initialized"},
        )
        responses.add(responses.POST, "http://localhost:8080/stop", json={"status": "stopped"})
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "detailed-session-2", "status": "initialized"},
        )
        responses.add(
            responses.GET, "http://localhost:8080/status", json={"game_status": "running"}
        )

        # Initialize and reset
        adapter._ensure_session_initialized()
        result = adapter.reset_game()

        # Verify comprehensive response format
        required_fields = [
            "status",
            "session_id",
            "message",
            "emulator_port",
            "container_id",
            "operation_time_ms",
            "configuration_preserved",
            "reset_details",
        ]

        for field in required_fields:
            assert field in result, f"Required field '{field}' missing from response"

        assert result["status"] == "initialized"
        assert result["emulator_port"] == 8080
        assert result["container_id"] == "test-container"
        assert result["configuration_preserved"] is True
        assert isinstance(result["operation_time_ms"], int)
        assert result["operation_time_ms"] >= 0

    def test_session_manager_thread_safety(self):
        """Test SessionManager thread safety with concurrent operations."""
        session_manager = SessionManager("http://localhost:8080")

        # Verify thread safety attributes exist
        assert hasattr(session_manager, "_lock")
        assert hasattr(session_manager, "_reset_in_progress")
        assert session_manager._reset_in_progress is False

        # Test that concurrent initialization is handled
        with patch("requests.Session.post") as mock_post:
            mock_post.return_value.json.return_value = {
                "session_id": "thread-safety-test",
                "status": "initialized",
            }
            mock_post.return_value.raise_for_status.return_value = None

            import threading

            results = []

            def init_worker():
                try:
                    result = session_manager.initialize_session()
                    results.append(result)
                except Exception as e:
                    results.append(e)

            # Start concurrent initialization
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=init_worker)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # All should succeed or be safely handled
            assert len(results) == 5
            # At least one should succeed
            successful_results = [r for r in results if isinstance(r, dict)]
            assert len(successful_results) > 0


class TestErrorConditionsAndRecovery:
    """Test error handling and recovery mechanisms."""

    @responses.activate
    def test_network_timeout_handling(self):
        """Test adapter handles network timeouts gracefully."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock session initialization
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "test-session", "status": "initialized"},
            status=200,
        )

        # Simulate timeout
        responses.add(responses.GET, "http://localhost:8080/status", body=Exception("Timeout"))

        with pytest.raises(PokemonGymAdapterError) as exc_info:
            adapter.get_state()

        assert "timeout" in str(exc_info.value).lower()

    @responses.activate
    def test_server_error_handling(self):
        """Test adapter handles server errors appropriately."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Simulate server error
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"error": "server_error"},
            status=500,
        )

        with pytest.raises(PokemonGymAdapterError) as exc_info:
            adapter.send_input("A")

        assert "server error" in str(exc_info.value).lower()

    @responses.activate
    def test_malformed_response_handling(self):
        """Test adapter handles malformed JSON responses."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock session initialization
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "test-session", "status": "initialized"},
            status=200,
        )

        # Return invalid JSON
        responses.add(
            responses.GET, "http://localhost:8080/status", body="invalid json{", status=200
        )

        with pytest.raises(PokemonGymAdapterError) as exc_info:
            adapter.get_state()

        assert "invalid response" in str(exc_info.value).lower()

    def test_retry_mechanism_with_exponential_backoff(self):
        """Test retry mechanism implements exponential backoff."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Test retry logic (would need to mock actual retries)
        retry_delays = adapter._calculate_retry_delays(max_retries=3)

        # Should implement exponential backoff
        assert len(retry_delays) == 3
        assert retry_delays[0] < retry_delays[1] < retry_delays[2]
        assert all(delay >= 0.1 for delay in retry_delays)  # Minimum delay

    @responses.activate
    def test_health_check_detects_session_problems(self):
        """Test is_healthy detects session and connectivity issues."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Healthy response
        responses.add(
            responses.GET,
            "http://localhost:8080/status",
            json={"game_status": "running"},
            status=200,
        )

        assert adapter.is_healthy() is True

        # Reset responses for unhealthy test
        responses.reset()
        responses.add(
            responses.GET,
            "http://localhost:8080/status",
            json={"error": "session_expired"},
            status=400,
        )

        assert adapter.is_healthy() is False


class TestPerformanceBenchmarks:
    """Test performance requirements and benchmarks."""

    @responses.activate
    def test_reset_performance_benchmark(self):
        """Production benchmark test for reset performance - must be <500ms."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock fast responses to ensure we're testing adapter logic, not network
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "bench-session-1", "status": "initialized"},
        )
        responses.add(responses.POST, "http://localhost:8080/stop", json={"status": "stopped"})
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "bench-session-2", "status": "initialized"},
        )
        responses.add(
            responses.GET, "http://localhost:8080/status", json={"game_status": "running"}
        )

        # Initialize session first
        adapter._ensure_session_initialized()

        # Run multiple reset operations to get reliable timing
        reset_times = []
        for i in range(5):
            start_time = time.time()
            result = adapter.reset_game()
            elapsed_ms = (time.time() - start_time) * 1000
            reset_times.append(elapsed_ms)

            # Validate each reset
            assert result["status"] == "initialized"
            assert "operation_time_ms" in result

            # Reset responses for next iteration
            responses.reset()
            responses.add(responses.POST, "http://localhost:8080/stop", json={"status": "stopped"})
            responses.add(
                responses.POST,
                "http://localhost:8080/initialize",
                json={"session_id": f"bench-session-{i+3}", "status": "initialized"},
            )
            responses.add(
                responses.GET, "http://localhost:8080/status", json={"game_status": "running"}
            )

        # Performance validation
        avg_reset_time = sum(reset_times) / len(reset_times)
        max_reset_time = max(reset_times)
        min_reset_time = min(reset_times)

        print("\nReset Performance Benchmark Results:")
        print(f"Average reset time: {avg_reset_time:.2f}ms")
        print(f"Min reset time: {min_reset_time:.2f}ms")
        print(f"Max reset time: {max_reset_time:.2f}ms")

        # Production SLA requirement
        assert avg_reset_time < 500, f"Average reset time {avg_reset_time:.2f}ms exceeds 500ms SLA"
        assert (
            max_reset_time < 1000
        ), f"Max reset time {max_reset_time:.2f}ms exceeds reasonable upper bound"

        # All individual operations should meet SLA
        for i, reset_time in enumerate(reset_times):
            assert reset_time < 500, f"Reset {i+1} took {reset_time:.2f}ms, exceeds 500ms SLA"

    @responses.activate
    def test_input_translation_performance(self):
        """Test batch input processing completes within 100ms."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock fast responses
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "perf-test", "status": "initialized"},
        )

        for _ in range(10):  # 10 button sequence
            responses.add(
                responses.POST, "http://localhost:8080/action", json={"status": "success"}
            )

        # Measure performance
        start_time = time.time()
        adapter.send_input("A B START SELECT UP DOWN LEFT RIGHT A B")
        elapsed = time.time() - start_time

        # Should complete within performance requirement
        assert elapsed < 0.1  # 100ms requirement

    @responses.activate
    def test_state_retrieval_performance(self):
        """Test state retrieval completes within 50ms."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock session initialization
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json={"session_id": "test-session", "status": "initialized"},
            status=200,
        )

        # Mock fast state response
        responses.add(
            responses.GET, "http://localhost:8080/status", json={"game_status": "running"}
        )

        # Measure performance
        start_time = time.time()
        adapter.get_state()
        elapsed = time.time() - start_time

        # Should complete within performance requirement
        assert elapsed < 0.05  # 50ms requirement

    def test_memory_usage_efficiency(self):
        """Test adapter maintains efficient memory usage."""
        import gc
        import tracemalloc

        tracemalloc.start()

        # Create and use adapter
        PokemonGymAdapter(8080, "test-container")

        # Force garbage collection
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (less than 1MB for basic adapter)
        assert peak < 1024 * 1024  # 1MB limit


class TestCompatibilityLayer:
    """Test compatibility with existing EmulatorPool interface."""

    def test_adapter_implements_pokemongymclient_interface(self):
        """Test adapter provides same interface as PokemonGymClient."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Verify all required methods exist
        assert hasattr(adapter, "send_input")
        assert hasattr(adapter, "get_state")
        assert hasattr(adapter, "reset_game")
        assert hasattr(adapter, "is_healthy")
        assert hasattr(adapter, "close")

        # Verify method signatures match
        import inspect

        # send_input should accept string and return dict
        sig = inspect.signature(adapter.send_input)
        assert len(sig.parameters) == 1  # input_sequence parameter

        # get_state should return dict
        sig = inspect.signature(adapter.get_state)
        assert len(sig.parameters) == 0  # no parameters

    def test_adapter_works_with_emulator_pool(self):
        """Test adapter can be used as drop-in replacement in EmulatorPool."""
        # This would be an integration test, but we can test interface compatibility
        adapter = PokemonGymAdapter(8080, "test-container")

        # Should have same attributes as PokemonGymClient
        assert hasattr(adapter, "port")
        assert hasattr(adapter, "container_id")
        assert hasattr(adapter, "base_url")

        # Should work with string formatting (used in EmulatorPool logging)
        adapter_str = str(adapter)
        assert "8080" in adapter_str
        assert "test-container" in adapter_str


class TestFactoryPattern:
    """Test factory methods for adapter creation."""

    def test_create_adapter_factory_method(self):
        """Test factory method creates appropriate adapter type."""
        # This would test a factory that chooses between direct client and adapter
        from claudelearnspokemon.pokemon_gym_adapter import create_pokemon_gym_client

        # Should create adapter for benchflow-ai API
        client = create_pokemon_gym_client(8080, "test-container", api_type="benchflow")
        assert isinstance(client, PokemonGymAdapter)

        # Should create direct client for legacy API
        client = create_pokemon_gym_client(8080, "test-container", api_type="legacy")
        # Would return PokemonGymClient in real implementation
        assert hasattr(client, "send_input")

    def test_factory_unknown_api_type_raises_error(self):
        """Test factory raises error for unknown API type."""
        from claudelearnspokemon.pokemon_gym_adapter import create_pokemon_gym_client

        with pytest.raises(ValueError) as exc_info:
            create_pokemon_gym_client(8080, "test-container", api_type="unknown")

        assert "unknown api_type" in str(exc_info.value).lower()


# Additional test fixtures and utilities for comprehensive testing
@pytest.fixture
def mock_adapter():
    """Fixture providing a mocked adapter for testing."""
    return PokemonGymAdapter(8080, "test-container-fixture")


@pytest.fixture
def benchflow_response():
    """Fixture providing sample benchflow-ai response data."""
    return {
        "game_status": "running",
        "player": {"x": 10, "y": 15, "map_id": "test_map"},
        "screen": {"tiles": [[1, 2], [3, 4]], "width": 20, "height": 18},
        "frame_count": 100,
    }


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_input_sequences(self):
        """Test adapter handles very long input sequences."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # 100 button sequence
        long_sequence = " ".join(["A"] * 100)
        parsed = adapter._parse_input_sequence(long_sequence)

        assert len(parsed) == 100
        assert all(button == "A" for button in parsed)

    def test_invalid_button_names(self):
        """Test adapter handles invalid button names gracefully."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Should handle invalid buttons gracefully (filter or error)
        with pytest.raises(PokemonGymAdapterError) as exc_info:
            adapter._parse_input_sequence("A INVALID_BUTTON B")

        assert "invalid button" in str(exc_info.value).lower()

    def test_concurrent_adapter_usage(self):
        """Test multiple adapters can be used concurrently."""
        adapter1 = PokemonGymAdapter(8081, "container-1")
        adapter2 = PokemonGymAdapter(8082, "container-2")

        # Should have independent session managers
        assert adapter1.session_manager is not adapter2.session_manager
        assert adapter1.port != adapter2.port
        assert adapter1.container_id != adapter2.container_id

    def test_adapter_close_cleanup(self):
        """Test adapter properly closes resources."""
        adapter = PokemonGymAdapter(8080, "test-container")

        # Mock session manager to test close behavior
        with patch.object(adapter.session_manager, "close") as mock_close:
            with patch.object(adapter.session, "close") as mock_session_close:
                adapter.close()

                # Should close both session manager and HTTP session
                mock_close.assert_called_once()
                mock_session_close.assert_called_once()


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v"])