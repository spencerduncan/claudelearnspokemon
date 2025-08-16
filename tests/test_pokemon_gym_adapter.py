"""
Unit tests for PokemonGymAdapter benchflow-ai integration.

Tests cover production failure scenarios, performance requirements, and API compatibility.
Following Bot Dean's philosophy: test all failure modes first, optimize for <100ms performance.
"""

import time
from unittest.mock import Mock, patch

import httpx
import pytest

from claudelearnspokemon.emulator_pool import EmulatorPoolError, PokemonGymClient
from claudelearnspokemon.pokemon_gym_adapter import PokemonGymAdapter, PokemonGymAdapterError


class TestPokemonGymAdapterInitialization:
    """Test PokemonGymAdapter initialization and configuration."""

    def setup_method(self) -> None:
        """Set up test environment for each test."""
        self.port = 8081
        self.container_id = "test_container_123"

    def teardown_method(self) -> None:
        """Clean up after each test - production hygiene."""
        # Clean up any adapters that weren't properly closed
        pass

    @pytest.mark.unit
    def test_initialization_defaults(self) -> None:
        """Test adapter initializes with production defaults."""
        adapter = PokemonGymAdapter(self.port, self.container_id)

        # Verify inheritance from PokemonGymClient
        assert isinstance(adapter, PokemonGymClient)
        assert adapter.port == 8081
        assert adapter.container_id == "test_container_123"
        assert adapter.base_url == "http://localhost:8081"

        # Verify adapter-specific initialization
        assert not adapter._session_initialized
        assert adapter._last_action_time is None

        # Verify timeout configuration for production performance
        expected_timeouts = {
            "initialize": 5.0,
            "action": 0.1,  # Critical: <100ms requirement
            "status": 0.05,  # Critical: fast health checks
            "stop": 2.0,
            "default": 0.1,
        }
        assert adapter.timeout_config == expected_timeouts

        # Verify HTTP client configuration
        assert isinstance(adapter.http_client, httpx.Client)
        assert not adapter.http_client.is_closed
        assert adapter.http_client.base_url == "http://localhost:8081"

        # Clean up
        adapter.close()

    @pytest.mark.unit
    def test_initialization_custom_config(self) -> None:
        """Test adapter accepts custom configuration."""
        custom_timeouts = {
            "initialize": 10.0,
            "action": 0.05,  # Even faster for high-performance mode
            "status": 0.02,
            "stop": 1.0,
            "default": 0.05,
        }

        custom_limits = {
            "max_keepalive_connections": 20,
            "max_connections": 50,
            "keepalive_expiry": 60.0,
        }

        adapter = PokemonGymAdapter(
            self.port,
            self.container_id,
            server_url="https://benchflow.example.com",
            timeout_config=custom_timeouts,
            connection_limits=custom_limits,
        )

        # Verify custom configuration applied
        assert adapter.base_url == "https://benchflow.example.com"
        assert adapter.timeout_config == custom_timeouts

        # Clean up
        adapter.close()

    @pytest.mark.unit
    def test_inheritance_compatibility(self) -> None:
        """Test adapter maintains full PokemonGymClient compatibility."""
        adapter = PokemonGymAdapter(self.port, self.container_id)

        # Verify all parent methods are accessible
        assert hasattr(adapter, "send_input")
        assert hasattr(adapter, "get_state")
        assert hasattr(adapter, "reset_game")
        assert hasattr(adapter, "is_healthy")
        assert hasattr(adapter, "close")

        # Verify new adapter methods
        assert hasattr(adapter, "initialize_session")
        assert hasattr(adapter, "execute_action")
        assert hasattr(adapter, "get_session_status")
        assert hasattr(adapter, "stop_session")
        assert hasattr(adapter, "is_session_active")

        adapter.close()


class TestPokemonGymAdapterSessionManagement:
    """Test session lifecycle management."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.port = 8081
        self.container_id = "test_container_123"
        self.adapter = PokemonGymAdapter(self.port, self.container_id)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if hasattr(self, "adapter"):
            self.adapter.close()

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_initialize_session_success(self, mock_post) -> None:
        """Test successful session initialization."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "session_id": "test_session_123",
            "status": "initialized",
            "initial_state": {"game": "pokemon_red"},
        }
        mock_post.return_value = mock_response

        # Test initialization
        config = {"game_type": "pokemon_red", "difficulty": "normal"}
        result = self.adapter.initialize_session(config)

        # Verify request was made correctly
        mock_post.assert_called_once_with(
            "/initialize", json=config, timeout=5.0  # Initialize timeout
        )

        # Verify response processing
        assert result["session_id"] == "test_session_123"
        assert result["status"] == "initialized"
        assert self.adapter._session_initialized is True

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_initialize_session_timeout(self, mock_post) -> None:
        """Test session initialization timeout handling."""
        mock_post.side_effect = httpx.TimeoutException("Request timeout")

        with pytest.raises(PokemonGymAdapterError) as exc_info:
            self.adapter.initialize_session()

        assert "Session initialization timeout" in str(exc_info.value)
        assert "Consider increasing timeout" in str(exc_info.value)
        assert not self.adapter._session_initialized

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_initialize_session_http_error(self, mock_post) -> None:
        """Test session initialization HTTP error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.side_effect = httpx.HTTPStatusError(
            "Server error", request=Mock(), response=mock_response
        )

        with pytest.raises(PokemonGymAdapterError) as exc_info:
            self.adapter.initialize_session()

        assert "Session initialization failed" in str(exc_info.value)
        assert "500" in str(exc_info.value)
        assert not self.adapter._session_initialized

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_stop_session_success(self, mock_post) -> None:
        """Test successful session stop."""
        # Set up initialized session
        self.adapter._session_initialized = True

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "stopped",
            "final_metrics": {"actions": 150, "time": 300.5},
        }
        mock_post.return_value = mock_response

        # Test session stop
        result = self.adapter.stop_session(save_state=True)

        # Verify request
        mock_post.assert_called_once_with("/stop", json={"save_state": True}, timeout=2.0)

        # Verify state update
        assert result["status"] == "stopped"
        assert not self.adapter._session_initialized


class TestPokemonGymAdapterActions:
    """Test action execution - critical performance path."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.port = 8081
        self.container_id = "test_container_123"
        self.adapter = PokemonGymAdapter(self.port, self.container_id)
        self.adapter._session_initialized = True  # Mock initialized state

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.adapter.close()

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_execute_action_string_input(self, mock_post) -> None:
        """Test action execution with string input."""
        # Mock fast response for performance test
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "state": {"player_x": 10, "player_y": 5},
            "reward": 0.1,
            "done": False,
        }
        mock_post.return_value = mock_response

        # Execute action
        result = self.adapter.execute_action("A B START")
        # Note: In production we'd measure execution time here for performance validation
        # but for unit tests we just verify the timeout configuration is correct

        # Verify API call
        mock_post.assert_called_once_with(
            "/action", json={"action": "A B START"}, timeout=0.1  # Critical: <100ms requirement
        )

        # Verify response
        assert result["reward"] == 0.1
        assert not result["done"]

        # Verify performance tracking
        assert self.adapter._last_action_time is not None

        # Note: Actual performance test would require real network
        # This tests the timeout configuration is correct

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_execute_action_dict_input(self, mock_post) -> None:
        """Test action execution with structured input."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_post.return_value = mock_response

        # Test with dictionary action
        action_dict = {"type": "button_sequence", "buttons": ["A", "B"], "duration": 0.5}

        self.adapter.execute_action(action_dict)

        mock_post.assert_called_once_with("/action", json=action_dict, timeout=0.1)

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_execute_action_timeout_violation(self, mock_post) -> None:
        """Test action timeout handling - critical for <100ms requirement."""
        mock_post.side_effect = httpx.TimeoutException("Action timeout")

        with pytest.raises(PokemonGymAdapterError) as exc_info:
            self.adapter.execute_action("A")

        error_msg = str(exc_info.value)
        assert "Action execution timeout" in error_msg
        assert "violates <100ms performance requirement" in error_msg

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_execute_action_uninitialized_session_warning(self, mock_post) -> None:
        """Test action on uninitialized session logs warning."""
        self.adapter._session_initialized = False

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_post.return_value = mock_response

        with patch("claudelearnspokemon.pokemon_gym_adapter.logger") as mock_logger:
            self.adapter.execute_action("A")

            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "uninitialized session" in warning_call


class TestPokemonGymAdapterStatus:
    """Test status and health checking functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.port = 8081
        self.container_id = "test_container_123"
        self.adapter = PokemonGymAdapter(self.port, self.container_id)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.adapter.close()

    @pytest.mark.unit
    @patch("httpx.Client.get")
    def test_get_session_status_success(self, mock_get) -> None:
        """Test successful status retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "active": True,
            "uptime": 125.5,
            "actions_processed": 42,
            "performance": {"avg_response_time": 0.045},
        }
        mock_get.return_value = mock_response

        result = self.adapter.get_session_status()

        # Verify API call with fast timeout
        mock_get.assert_called_once_with("/status", timeout=0.05)  # Very fast for status checks

        # Verify response
        assert result["active"] is True
        assert result["uptime"] == 125.5
        assert result["performance"]["avg_response_time"] == 0.045

    @pytest.mark.unit
    @patch("httpx.Client.get")
    def test_get_session_status_timeout(self, mock_get) -> None:
        """Test status timeout handling."""
        mock_get.side_effect = httpx.TimeoutException("Status timeout")

        with pytest.raises(PokemonGymAdapterError) as exc_info:
            self.adapter.get_session_status()

        assert "Status retrieval timeout" in str(exc_info.value)
        assert "Expected <50ms response time" in str(exc_info.value)

    @pytest.mark.unit
    @patch.object(PokemonGymAdapter, "get_session_status")
    def test_is_session_active_true(self, mock_status) -> None:
        """Test session activity check when active."""
        self.adapter._session_initialized = True
        mock_status.return_value = {"active": True}

        assert self.adapter.is_session_active() is True
        mock_status.assert_called_once()

    @pytest.mark.unit
    @patch.object(PokemonGymAdapter, "get_session_status")
    def test_is_session_active_false_uninitialized(self, mock_status) -> None:
        """Test session activity check when not initialized."""
        self.adapter._session_initialized = False

        # Should return False without calling server
        assert self.adapter.is_session_active() is False
        mock_status.assert_not_called()

    @pytest.mark.unit
    @patch.object(PokemonGymAdapter, "get_session_status")
    def test_is_session_active_server_error(self, mock_status) -> None:
        """Test session activity check handles server errors gracefully."""
        self.adapter._session_initialized = True
        mock_status.side_effect = PokemonGymAdapterError("Server error")

        # Should return False on any error
        assert self.adapter.is_session_active() is False


class TestPokemonGymAdapterFactoryMethod:
    """Test factory method and preset configurations."""

    @pytest.mark.unit
    def test_create_adapter_benchflow_preset(self) -> None:
        """Test benchflow preset configuration."""
        adapter = PokemonGymAdapter.create_adapter(
            port=8081, container_id="test123", adapter_type="benchflow"
        )

        # Verify benchflow-specific timeouts
        expected_timeouts = {
            "initialize": 5.0,
            "action": 0.1,
            "status": 0.05,
            "stop": 2.0,
            "default": 0.1,
        }
        assert adapter.timeout_config == expected_timeouts

        adapter.close()

    @pytest.mark.unit
    def test_create_adapter_high_performance_preset(self) -> None:
        """Test high performance preset for production scale."""
        adapter = PokemonGymAdapter.create_adapter(
            port=8081, container_id="test123", adapter_type="high_performance"
        )

        # Verify aggressive timeouts for high performance
        assert adapter.timeout_config["action"] == 0.05  # Even faster
        assert adapter.timeout_config["status"] == 0.025

        adapter.close()

    @pytest.mark.unit
    def test_create_adapter_development_preset(self) -> None:
        """Test development preset with relaxed timeouts."""
        adapter = PokemonGymAdapter.create_adapter(
            port=8081, container_id="test123", adapter_type="development"
        )

        # Verify relaxed timeouts for development
        assert adapter.timeout_config["action"] == 1.0
        assert adapter.timeout_config["initialize"] == 10.0

        adapter.close()

    @pytest.mark.unit
    def test_create_adapter_unknown_type(self) -> None:
        """Test factory with unknown adapter type."""
        with pytest.raises(ValueError) as exc_info:
            PokemonGymAdapter.create_adapter(
                port=8081, container_id="test123", adapter_type="unknown_type"
            )

        assert "Unknown adapter type: unknown_type" in str(exc_info.value)
        assert "benchflow" in str(exc_info.value)

    @pytest.mark.unit
    def test_create_adapter_with_overrides(self) -> None:
        """Test factory method with configuration overrides."""
        custom_timeouts = {"action": 0.2, "status": 0.1}  # Override the preset

        adapter = PokemonGymAdapter.create_adapter(
            port=8081,
            container_id="test123",
            adapter_type="benchflow",
            timeout_config=custom_timeouts,
        )

        # Verify override was applied
        assert adapter.timeout_config == custom_timeouts

        adapter.close()


class TestPokemonGymAdapterPerformance:
    """Test performance requirements and monitoring."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.adapter = PokemonGymAdapter(8081, "test_container")

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.adapter.close()

    @pytest.mark.unit
    def test_performance_metrics(self) -> None:
        """Test performance metrics collection."""
        # Initially no metrics
        metrics = self.adapter.get_performance_metrics()
        assert metrics["session_initialized"] is False
        assert metrics["last_action_time"] is None

        # Mock some activity
        self.adapter._session_initialized = True
        self.adapter._last_action_time = 0.045  # 45ms - good performance

        metrics = self.adapter.get_performance_metrics()
        assert metrics["session_initialized"] is True
        assert metrics["last_action_time"] == 0.045
        assert "timeout_config" in metrics
        assert "connection_pool_info" in metrics

    @pytest.mark.unit
    @patch("httpx.Client.post")
    @patch("claudelearnspokemon.pokemon_gym_adapter.logger")
    def test_action_performance_warning(self, mock_logger, mock_post) -> None:
        """Test performance warning when approaching timeout."""
        # Mock slow response (80ms - approaching 100ms limit)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}

        def slow_response(*args, **kwargs):
            time.sleep(0.08)  # Simulate 80ms response
            return mock_response

        mock_post.side_effect = slow_response

        self.adapter._session_initialized = True
        self.adapter.execute_action("A")

        # Verify warning was logged
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "approaching timeout" in warning_msg
        assert "target: <0.1s" in warning_msg


class TestPokemonGymAdapterCleanup:
    """Test resource cleanup and lifecycle management."""

    @pytest.mark.unit
    def test_close_cleanup(self) -> None:
        """Test proper resource cleanup on close."""
        adapter = PokemonGymAdapter(8081, "test_container")

        # Verify initial state
        assert not adapter.http_client.is_closed

        # Close adapter
        adapter.close()

        # Verify cleanup
        assert adapter.http_client.is_closed

    @pytest.mark.unit
    def test_context_manager_cleanup(self) -> None:
        """Test adapter can be used as context manager."""
        # This would be useful for production resource management
        adapter = PokemonGymAdapter(8081, "test_container")

        try:
            # Use adapter
            assert not adapter.http_client.is_closed
        finally:
            # Always clean up
            adapter.close()
            assert adapter.http_client.is_closed

    @pytest.mark.unit
    @patch("claudelearnspokemon.pokemon_gym_adapter.logger")
    def test_close_error_handling(self, mock_logger) -> None:
        """Test close handles errors gracefully."""
        adapter = PokemonGymAdapter(8081, "test_container")

        # Mock HTTP client close to raise error
        adapter.http_client.close = Mock(side_effect=Exception("Close error"))

        # Close should not raise, but should log error
        adapter.close()

        mock_logger.error.assert_called()
        error_msg = mock_logger.error.call_args[0][0]
        assert "Error closing HTTP client" in error_msg


class TestPokemonGymAdapterErrorHandling:
    """Test error handling and exception hierarchy."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.adapter = PokemonGymAdapter(8081, "test_container")

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.adapter.close()

    @pytest.mark.unit
    def test_error_inheritance(self) -> None:
        """Test PokemonGymAdapterError inherits from EmulatorPoolError."""
        error = PokemonGymAdapterError("Test error")

        # Verify inheritance chain
        assert isinstance(error, EmulatorPoolError)
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_unexpected_error_handling(self, mock_post) -> None:
        """Test handling of unexpected errors during operations."""
        mock_post.side_effect = ValueError("Unexpected error")

        with pytest.raises(PokemonGymAdapterError) as exc_info:
            self.adapter.initialize_session()

        assert "Unexpected error during session initialization" in str(exc_info.value)

    @pytest.mark.unit
    def test_string_representation(self) -> None:
        """Test string representations for logging."""
        adapter = PokemonGymAdapter(8081, "test_container_123456789")

        # Test __str__ method
        str_repr = str(adapter)
        assert "PokemonGymAdapter" in str_repr
        assert "port=8081" in str_repr
        assert "test_contain" in str_repr  # Truncated container ID (first 12 chars)
        assert "session=inactive" in str_repr

        # Test __repr__ method
        repr_str = repr(adapter)
        assert "PokemonGymAdapter(" in repr_str
        assert "container_id='test_container_123456789'" in repr_str
        assert "session_initialized=False" in repr_str

        adapter.close()


class TestPokemonGymAdapterIntegration:
    """Integration tests for full adapter lifecycle."""

    @pytest.mark.integration
    @patch("httpx.Client.post")
    @patch("httpx.Client.get")
    def test_full_session_lifecycle(self, mock_get, mock_post) -> None:
        """Test complete session lifecycle with mocked HTTP."""
        adapter = PokemonGymAdapter(8081, "integration_test")

        try:
            # Mock initialize response
            init_response = Mock()
            init_response.status_code = 200
            init_response.json.return_value = {"session_id": "test123", "status": "ready"}

            # Mock action response
            action_response = Mock()
            action_response.status_code = 200
            action_response.json.return_value = {"reward": 1.0, "done": False}

            # Mock status response
            status_response = Mock()
            status_response.status_code = 200
            status_response.json.return_value = {"active": True, "uptime": 10.5}

            # Mock stop response
            stop_response = Mock()
            stop_response.status_code = 200
            stop_response.json.return_value = {"status": "stopped", "total_actions": 5}

            mock_post.side_effect = [init_response, action_response, stop_response]
            mock_get.return_value = status_response

            # Full lifecycle test

            # 1. Initialize session
            init_result = adapter.initialize_session({"game": "pokemon"})
            assert init_result["session_id"] == "test123"
            assert adapter._session_initialized

            # 2. Execute actions
            action_result = adapter.execute_action("A B")
            assert action_result["reward"] == 1.0

            # 3. Check status
            status_result = adapter.get_session_status()
            assert status_result["active"]
            assert adapter.is_session_active()  # This calls get_session_status() again

            # 4. Stop session
            stop_result = adapter.stop_session()
            assert stop_result["status"] == "stopped"
            assert not adapter._session_initialized

            # Verify all API calls were made correctly
            assert mock_post.call_count == 3  # init, action, stop
            assert mock_get.call_count == 2  # status called twice (direct + via is_session_active)

        finally:
            adapter.close()
