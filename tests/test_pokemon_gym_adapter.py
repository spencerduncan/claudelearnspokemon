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

        # Verify all parent attributes are accessible
        assert hasattr(adapter, "port")
        assert hasattr(adapter, "container_id")
        assert hasattr(adapter, "base_url")

        # Verify parent methods are inherited
        assert hasattr(adapter, "close")

        # Clean up
        adapter.close()

    @pytest.mark.unit
    def test_string_representations(self) -> None:
        """Test string representations for logging."""
        adapter = PokemonGymAdapter(self.port, self.container_id)

        # Test __str__ for logging
        str_repr = str(adapter)
        assert "PokemonGymAdapter" in str_repr
        assert "8081" in str_repr
        assert "test_container" in str_repr
        assert "inactive" in str_repr

        # Test __repr__ for debugging
        repr_str = repr(adapter)
        assert "PokemonGymAdapter" in repr_str
        assert "test_container_123" in repr_str
        assert "session_initialized=False" in repr_str

        # Clean up
        adapter.close()


class TestPokemonGymAdapterSession:
    """Test benchflow-ai session management."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.port = 8082
        self.container_id = "session_test_456"
        self.adapter = PokemonGymAdapter(self.port, self.container_id)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if hasattr(self, "adapter"):
            self.adapter.close()

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_initialize_session_success(self, mock_post: Mock) -> None:
        """Test successful session initialization."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "status": "initialized",
            "session_id": "test_session_123",
            "initial_state": {"frame_count": 0},
        }
        mock_post.return_value = mock_response

        # Test initialization
        start_time = time.perf_counter()
        result = self.adapter.initialize_session({"difficulty": "normal"})
        duration = time.perf_counter() - start_time

        # Verify performance requirement
        assert duration < 5.0  # Initialize timeout

        # Verify response
        assert result["status"] == "initialized"
        assert result["session_id"] == "test_session_123"
        assert self.adapter._session_initialized

        # Verify HTTP call
        mock_post.assert_called_once_with(
            "/initialize",
            json={"difficulty": "normal"},
            timeout=5.0,
        )

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_initialize_session_timeout(self, mock_post: Mock) -> None:
        """Test session initialization timeout handling."""
        # Mock timeout exception
        mock_post.side_effect = httpx.TimeoutException("Request timeout")

        # Test timeout handling
        with pytest.raises(PokemonGymAdapterError) as exc_info:
            self.adapter.initialize_session()

        # Verify error details
        assert "timeout" in str(exc_info.value).lower()
        assert "consider increasing timeout" in str(exc_info.value).lower()
        assert not self.adapter._session_initialized

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_initialize_session_http_error(self, mock_post: Mock) -> None:
        """Test session initialization HTTP error handling."""
        # Mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=Mock(), response=mock_response
        )

        # Test error handling
        with pytest.raises(PokemonGymAdapterError) as exc_info:
            self.adapter.initialize_session()

        # Verify error details
        assert "500" in str(exc_info.value)
        assert "Internal Server Error" in str(exc_info.value)
        assert not self.adapter._session_initialized


class TestPokemonGymAdapterActions:
    """Test action execution with performance monitoring."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.port = 8083
        self.container_id = "action_test_789"
        self.adapter = PokemonGymAdapter(self.port, self.container_id)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if hasattr(self, "adapter"):
            self.adapter.close()

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_execute_action_string_input(self, mock_post: Mock) -> None:
        """Test action execution with string input."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "status": "success",
            "frame_count": 150,
            "reward": 10,
        }
        mock_post.return_value = mock_response

        # Test action execution
        start_time = time.perf_counter()
        result = self.adapter.execute_action("A")
        duration = time.perf_counter() - start_time

        # Verify performance requirement (<100ms)
        assert duration < 0.1
        assert self.adapter._last_action_time == duration

        # Verify response
        assert result["status"] == "success"
        assert result["frame_count"] == 150

        # Verify HTTP call with string normalization
        mock_post.assert_called_once_with(
            "/action",
            json={"action": "A"},
            timeout=0.1,
        )

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_execute_action_dict_input(self, mock_post: Mock) -> None:
        """Test action execution with dictionary input."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "success", "frames_advanced": 5}
        mock_post.return_value = mock_response

        # Test dictionary action
        action_dict = {"action_type": "press_key", "keys": ["A", "B"]}
        result = self.adapter.execute_action(action_dict)

        # Verify response
        assert result["status"] == "success"
        assert result["frames_advanced"] == 5

        # Verify HTTP call with dictionary passthrough
        mock_post.assert_called_once_with(
            "/action",
            json=action_dict,
            timeout=0.1,
        )

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_execute_action_timeout_violation(self, mock_post: Mock) -> None:
        """Test action execution timeout handling."""
        # Mock timeout exception
        mock_post.side_effect = httpx.TimeoutException("Action timeout")

        # Test timeout handling
        with pytest.raises(PokemonGymAdapterError) as exc_info:
            self.adapter.execute_action("START")

        # Verify error details
        assert "timeout" in str(exc_info.value).lower()
        assert "violates <100ms performance requirement" in str(exc_info.value)

    @pytest.mark.unit
    @patch("httpx.Client.post")
    @patch("time.perf_counter")
    def test_execute_action_performance_warning(self, mock_time: Mock, mock_post: Mock) -> None:
        """Test performance warning for slow actions."""
        # Mock slow execution time (90ms > 80ms warning threshold)
        mock_time.side_effect = [0.0, 0.09]  # 90ms execution

        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value = mock_response

        # Test action execution with logging
        with patch("claudelearnspokemon.pokemon_gym_adapter.logger") as mock_logger:
            result = self.adapter.execute_action("B")

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "approaching timeout" in warning_call
            assert "0.090s" in warning_call

        # Verify successful response despite warning
        assert result["status"] == "success"
        assert self.adapter._last_action_time == 0.09


class TestPokemonGymAdapterStatus:
    """Test session status monitoring."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.port = 8084
        self.container_id = "status_test_012"
        self.adapter = PokemonGymAdapter(self.port, self.container_id)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if hasattr(self, "adapter"):
            self.adapter.close()

    @pytest.mark.unit
    @patch("httpx.Client.get")
    def test_get_session_status_success(self, mock_get: Mock) -> None:
        """Test successful status retrieval."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "active": True,
            "frame_count": 500,
            "player_position": {"x": 100, "y": 150},
        }
        mock_get.return_value = mock_response

        # Test status retrieval
        start_time = time.perf_counter()
        result = self.adapter.get_session_status()
        duration = time.perf_counter() - start_time

        # Verify performance requirement (<50ms)
        assert duration < 0.05

        # Verify response
        assert result["active"] is True
        assert result["frame_count"] == 500

        # Verify HTTP call
        mock_get.assert_called_once_with("/status", timeout=0.05)

    @pytest.mark.unit
    @patch("httpx.Client.get")
    def test_get_session_status_timeout(self, mock_get: Mock) -> None:
        """Test status retrieval timeout handling."""
        # Mock timeout exception
        mock_get.side_effect = httpx.TimeoutException("Status timeout")

        # Test timeout handling
        with pytest.raises(PokemonGymAdapterError) as exc_info:
            self.adapter.get_session_status()

        # Verify error details
        assert "timeout" in str(exc_info.value).lower()
        assert "expected <50ms response time" in str(exc_info.value).lower()

    @pytest.mark.unit
    @patch("httpx.Client.get")
    def test_is_session_active_true(self, mock_get: Mock) -> None:
        """Test session activity check - active session."""
        # Setup adapter as initialized
        self.adapter._session_initialized = True

        # Mock successful status response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"active": True}
        mock_get.return_value = mock_response

        # Test activity check
        assert self.adapter.is_session_active() is True

    @pytest.mark.unit
    def test_is_session_active_false_not_initialized(self) -> None:
        """Test session activity check - not initialized."""
        # Ensure adapter is not initialized
        self.adapter._session_initialized = False

        # Test activity check (should return False without HTTP call)
        assert self.adapter.is_session_active() is False

    @pytest.mark.unit
    @patch("httpx.Client.get")
    def test_is_session_active_false_server_error(self, mock_get: Mock) -> None:
        """Test session activity check - server error."""
        # Setup adapter as initialized
        self.adapter._session_initialized = True

        # Mock server error
        mock_get.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=Mock(), response=Mock()
        )

        # Test activity check (should return False on error)
        assert self.adapter.is_session_active() is False


class TestPokemonGymAdapterSessionStop:
    """Test session termination."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.port = 8085
        self.container_id = "stop_test_345"
        self.adapter = PokemonGymAdapter(self.port, self.container_id)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if hasattr(self, "adapter"):
            self.adapter.close()

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_stop_session_success(self, mock_post: Mock) -> None:
        """Test successful session termination."""
        # Setup adapter as initialized
        self.adapter._session_initialized = True

        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "status": "stopped",
            "final_score": 1500,
            "session_duration": 300.5,
        }
        mock_post.return_value = mock_response

        # Test session stop
        result = self.adapter.stop_session(save_state=True)

        # Verify response
        assert result["status"] == "stopped"
        assert result["final_score"] == 1500
        assert not self.adapter._session_initialized

        # Verify HTTP call
        mock_post.assert_called_once_with(
            "/stop",
            json={"save_state": True},
            timeout=2.0,
        )

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_stop_session_timeout(self, mock_post: Mock) -> None:
        """Test session stop timeout handling."""
        # Mock timeout exception
        mock_post.side_effect = httpx.TimeoutException("Stop timeout")

        # Test timeout handling
        with pytest.raises(PokemonGymAdapterError) as exc_info:
            self.adapter.stop_session()

        # Verify error details
        assert "timeout" in str(exc_info.value).lower()
        assert "server may be unresponsive" in str(exc_info.value).lower()


class TestPokemonGymAdapterMetrics:
    """Test performance metrics and monitoring."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.port = 8086
        self.container_id = "metrics_test_678"
        self.adapter = PokemonGymAdapter(self.port, self.container_id)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if hasattr(self, "adapter"):
            self.adapter.close()

    @pytest.mark.unit
    def test_get_performance_metrics_initial(self) -> None:
        """Test initial performance metrics."""
        metrics = self.adapter.get_performance_metrics()

        # Verify initial state
        assert metrics["session_initialized"] is False
        assert metrics["last_action_time"] is None
        assert "timeout_config" in metrics
        assert "connection_pool_info" in metrics

        # Verify timeout configuration is included
        assert metrics["timeout_config"]["action"] == 0.1
        assert metrics["timeout_config"]["initialize"] == 5.0

    @pytest.mark.unit
    def test_get_performance_metrics_after_action(self) -> None:
        """Test performance metrics after action execution."""
        # Simulate action timing
        self.adapter._session_initialized = True
        self.adapter._last_action_time = 0.075

        metrics = self.adapter.get_performance_metrics()

        # Verify updated state
        assert metrics["session_initialized"] is True
        assert metrics["last_action_time"] == 0.075


class TestPokemonGymAdapterFactoryMethods:
    """Test factory methods and presets."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        # Clean up any created adapters
        pass

    @pytest.mark.unit
    def test_create_adapter_benchflow_preset(self) -> None:
        """Test benchflow preset configuration."""
        adapter = PokemonGymAdapter.create_adapter(
            port=8087,
            container_id="factory_test_901",
            adapter_type="benchflow",
        )

        try:
            # Verify benchflow preset configuration
            assert adapter.timeout_config["action"] == 0.1
            assert adapter.timeout_config["initialize"] == 5.0
            assert adapter.timeout_config["status"] == 0.05

        finally:
            adapter.close()

    @pytest.mark.unit
    def test_create_adapter_high_performance_preset(self) -> None:
        """Test high_performance preset configuration."""
        adapter = PokemonGymAdapter.create_adapter(
            port=8088,
            container_id="hp_test_234",
            adapter_type="high_performance",
        )

        try:
            # Verify high_performance preset configuration
            assert adapter.timeout_config["action"] == 0.05  # Faster than benchflow
            assert adapter.timeout_config["initialize"] == 3.0  # Faster than benchflow
            assert adapter.timeout_config["status"] == 0.025  # Faster than benchflow

        finally:
            adapter.close()

    @pytest.mark.unit
    def test_create_adapter_development_preset(self) -> None:
        """Test development preset configuration."""
        adapter = PokemonGymAdapter.create_adapter(
            port=8089,
            container_id="dev_test_567",
            adapter_type="development",
        )

        try:
            # Verify development preset configuration
            assert adapter.timeout_config["action"] == 1.0  # Much slower for debugging
            assert adapter.timeout_config["initialize"] == 10.0  # Slower for debugging
            assert adapter.timeout_config["status"] == 0.5  # Slower for debugging

        finally:
            adapter.close()

    @pytest.mark.unit
    def test_create_adapter_unknown_type(self) -> None:
        """Test error handling for unknown adapter type."""
        with pytest.raises(ValueError) as exc_info:
            PokemonGymAdapter.create_adapter(
                port=8090,
                container_id="unknown_test_890",
                adapter_type="unknown_type",
            )

        # Verify error message
        assert "Unknown adapter type: unknown_type" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    @pytest.mark.unit
    def test_create_adapter_with_overrides(self) -> None:
        """Test factory method with configuration overrides."""
        custom_timeouts = {
            "initialize": 15.0,
            "action": 0.2,
            "status": 0.1,
            "stop": 3.0,
            "default": 0.2,
        }

        adapter = PokemonGymAdapter.create_adapter(
            port=8091,
            container_id="override_test_123",
            adapter_type="benchflow",
            timeout_config=custom_timeouts,
        )

        try:
            # Verify overrides applied
            assert adapter.timeout_config == custom_timeouts

        finally:
            adapter.close()


class TestPokemonGymAdapterResourceManagement:
    """Test resource management and cleanup."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.port = 8092
        self.container_id = "cleanup_test_456"

    @pytest.mark.unit
    def test_close_cleans_http_client(self) -> None:
        """Test close() method cleans up HTTP client."""
        adapter = PokemonGymAdapter(self.port, self.container_id)

        # Verify client is open initially
        assert not adapter.http_client.is_closed

        # Close adapter
        adapter.close()

        # Verify client is closed
        assert adapter.http_client.is_closed

    @pytest.mark.unit
    @patch("claudelearnspokemon.pokemon_gym_adapter.logger")
    def test_close_handles_http_client_error(self, mock_logger: Mock) -> None:
        """Test close() handles HTTP client errors gracefully."""
        adapter = PokemonGymAdapter(self.port, self.container_id)

        # Mock HTTP client close to raise exception
        with patch.object(adapter.http_client, "close", side_effect=Exception("Close error")):
            # Should not raise exception
            adapter.close()

        # Verify error was logged
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        assert "Error closing HTTP client" in error_call

    @pytest.mark.unit
    def test_context_manager_usage(self) -> None:
        """Test adapter can be used as context manager."""
        with PokemonGymAdapter(self.port, self.container_id) as adapter:
            # Verify adapter is usable
            assert isinstance(adapter, PokemonGymAdapter)
            assert not adapter.http_client.is_closed

        # Context manager should handle cleanup
        # Note: This test assumes context manager is implemented in parent class


class TestPokemonGymAdapterErrorHierarchy:
    """Test error hierarchy and exception handling."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.port = 8093
        self.container_id = "error_test_789"

    @pytest.mark.unit
    def test_error_hierarchy(self) -> None:
        """Test PokemonGymAdapterError inherits correctly."""
        # Verify error hierarchy
        assert issubclass(PokemonGymAdapterError, EmulatorPoolError)
        assert issubclass(PokemonGymAdapterError, Exception)

        # Test error instantiation
        error = PokemonGymAdapterError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, EmulatorPoolError)

    @pytest.mark.unit
    @patch("httpx.Client.post")
    def test_error_context_preservation(self, mock_post: Mock) -> None:
        """Test error context is preserved in exception chain."""
        adapter = PokemonGymAdapter(self.port, self.container_id)

        try:
            # Mock an HTTP error with specific details
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.text = "Session not found"
            mock_post.side_effect = httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=mock_response
            )

            # This should raise PokemonGymAdapterError with context
            adapter.initialize_session()

        except PokemonGymAdapterError as e:
            # Verify error context is preserved
            assert "404" in str(e)
            assert "Session not found" in str(e)
            assert e.__cause__ is not None  # Verify exception chaining

        finally:
            adapter.close()


class TestPokemonGymAdapterIntegration:
    """Integration tests for end-to-end scenarios."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.port = 8094
        self.container_id = "integration_test_012"
        self.adapter = PokemonGymAdapter(self.port, self.container_id)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if hasattr(self, "adapter"):
            self.adapter.close()

    @pytest.mark.integration
    @patch("httpx.Client.post")
    @patch("httpx.Client.get")
    def test_full_session_lifecycle(self, mock_get: Mock, mock_post: Mock) -> None:
        """Test complete session lifecycle: init -> action -> status -> stop."""

        # Mock initialize session
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"status": "initialized", "session_id": "test_123"}

        # Mock execute action
        action_response = Mock()
        action_response.raise_for_status.return_value = None
        action_response.json.return_value = {"status": "success", "frame_count": 100}

        # Mock stop session
        stop_response = Mock()
        stop_response.raise_for_status.return_value = None
        stop_response.json.return_value = {"status": "stopped"}

        # Mock get status
        status_response = Mock()
        status_response.raise_for_status.return_value = None
        status_response.json.return_value = {"active": True, "frame_count": 100}

        # Configure mocks
        mock_post.side_effect = [init_response, action_response, stop_response]
        mock_get.return_value = status_response

        # Execute full lifecycle
        init_result = self.adapter.initialize_session({"game": "pokemon_red"})
        assert init_result["status"] == "initialized"
        assert self.adapter._session_initialized

        action_result = self.adapter.execute_action("A")
        assert action_result["status"] == "success"

        status_result = self.adapter.get_session_status()
        assert status_result["active"] is True

        stop_result = self.adapter.stop_session(save_state=True)
        assert stop_result["status"] == "stopped"
        assert not self.adapter._session_initialized

        # Verify all HTTP calls were made
        assert mock_post.call_count == 3
        assert mock_get.call_count == 1

    @pytest.mark.integration
    @patch("httpx.Client.post")
    def test_performance_under_load(self, mock_post: Mock) -> None:
        """Test adapter performance under sequential load."""
        # Mock fast responses for all actions
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "success", "frame_count": 1}
        mock_post.return_value = mock_response

        # Execute multiple actions quickly
        action_count = 100
        start_time = time.perf_counter()

        for i in range(action_count):
            result = self.adapter.execute_action(f"action_{i}")
            assert result["status"] == "success"

        total_duration = time.perf_counter() - start_time

        # Verify performance: should average well under 100ms per action
        avg_duration = total_duration / action_count
        assert avg_duration < 0.01  # 10ms average (well under 100ms requirement)

        # Verify last action time is tracked
        assert self.adapter._last_action_time is not None
        assert self.adapter._last_action_time < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
