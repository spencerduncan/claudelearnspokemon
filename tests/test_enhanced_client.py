"""
Tests for EnhancedPokemonGymClient: Kernel-quality client with advanced error recovery.

Comprehensive test coverage for enhanced client functionality:
- Integration of retry manager, circuit breaker, and session recovery
- End-to-end error recovery scenarios
- Performance and reliability validation
- Comprehensive metrics and monitoring

Author: Linus Torbot - Kernel Quality Standards Applied
"""

from unittest.mock import Mock, patch

import pytest
from requests.exceptions import ConnectionError, HTTPError

from claudelearnspokemon.circuit_breaker import CircuitState
from claudelearnspokemon.emulator_pool import EmulatorPoolError
from claudelearnspokemon.enhanced_client import EnhancedPokemonGymClient
from claudelearnspokemon.retry_manager import RetryConfig
from claudelearnspokemon.session_recovery import SessionConfig, SessionState


@pytest.mark.medium
class TestEnhancedPokemonGymClient:
    """Test suite for EnhancedPokemonGymClient functionality."""

    def setup_method(self):
        """Setup for each test method."""
        # Fast configurations for testing
        retry_config = RetryConfig(max_retries=2, base_delay=0.1, jitter=False)
        session_config = SessionConfig(session_timeout=60.0, initialization_timeout=5.0)

        self.client = EnhancedPokemonGymClient(
            port=8081,
            container_id="test-container-123",
            retry_config=retry_config,
            session_config=session_config,
        )

    def test_initialization(self):
        """Test enhanced client initialization."""
        assert self.client.port == 8081
        assert self.client.container_id == "test-container-123"
        assert self.client.retry_manager is not None
        assert self.client.circuit_breaker is not None
        assert self.client.session_recovery is not None

    @patch("claudelearnspokemon.enhanced_client.super")
    @patch("requests.Session.post")
    def test_send_input_success(self, mock_post, mock_super):
        """Test successful send_input with session management."""
        # Mock session initialization
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"session_id": "test-session"}
        mock_post.return_value = init_response

        # Mock successful send_input call
        mock_super_instance = Mock()
        mock_super_instance.send_input.return_value = {"status": "success"}
        mock_super.return_value = mock_super_instance

        result = self.client.send_input("A B START")

        assert result == {"status": "success"}
        mock_super_instance.send_input.assert_called_once_with("A B START")

    @patch("claudelearnspokemon.enhanced_client.super")
    @patch("requests.Session.post")
    def test_send_input_with_retry_recovery(self, mock_post, mock_super):
        """Test send_input with retry and recovery."""
        # Mock session initialization
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"session_id": "test-session"}
        mock_post.return_value = init_response

        # Mock send_input with initial failure, then success
        mock_super_instance = Mock()
        mock_super_instance.send_input.side_effect = [
            ConnectionError("Temporary network issue"),
            {"status": "success"},
        ]
        mock_super.return_value = mock_super_instance

        result = self.client.send_input("A B")

        assert result == {"status": "success"}
        assert mock_super_instance.send_input.call_count == 2

    @patch("claudelearnspokemon.enhanced_client.super")
    @patch("requests.Session.post")
    def test_session_error_handling_and_recovery(self, mock_post, mock_super):
        """Test session error detection and recovery."""
        # Mock session initialization (called twice for recovery)
        init_responses = [
            Mock(raise_for_status=Mock(), json=Mock(return_value={"session_id": "session-1"})),
            Mock(raise_for_status=Mock(), json=Mock(return_value={"session_id": "session-2"})),
        ]
        mock_post.side_effect = init_responses

        # Mock operation with session error, then success
        mock_super_instance = Mock()
        session_error = HTTPError("401 Unauthorized - Session expired")
        session_error.response = Mock(status_code=401)

        mock_super_instance.send_input.side_effect = [session_error, {"status": "success"}]
        mock_super.return_value = mock_super_instance

        result = self.client.send_input("START")

        assert result == {"status": "success"}
        # Should have two session initializations (original + recovery)
        assert mock_post.call_count == 2

    @patch("claudelearnspokemon.enhanced_client.super")
    @patch("requests.Session.post")
    def test_circuit_breaker_integration(self, mock_post, mock_super):
        """Test circuit breaker protection."""
        # Mock session initialization
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"session_id": "test-session"}
        mock_post.return_value = init_response

        # Mock persistent failures to trip circuit breaker
        mock_super_instance = Mock()
        mock_super_instance.send_input.side_effect = ConnectionError("Service down")
        mock_super.return_value = mock_super_instance

        # Make several attempts to trip the circuit breaker
        # Each attempt should fail with EmulatorPoolError until circuit trips
        for _attempt in range(6):  # More than circuit breaker threshold (5)
            with pytest.raises(EmulatorPoolError):
                self.client.send_input("A")

        # Circuit should now be open
        assert self.client.circuit_breaker.get_state() == CircuitState.OPEN

        # Next attempt should be immediately rejected due to open circuit
        with pytest.raises(EmulatorPoolError) as exc_info:
            self.client.send_input("B")

        assert "CircuitBreaker" in str(exc_info.value) and "is OPEN" in str(exc_info.value)

    @patch("claudelearnspokemon.enhanced_client.super")
    @patch("requests.Session.post")
    def test_get_state_success(self, mock_post, mock_super):
        """Test successful get_state operation."""
        # Mock session initialization
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"session_id": "test-session"}
        mock_post.return_value = init_response

        # Mock successful get_state
        mock_super_instance = Mock()
        mock_super_instance.get_state.return_value = {"location": "pallet_town"}
        mock_super.return_value = mock_super_instance

        result = self.client.get_state()

        assert result == {"location": "pallet_town"}

    @patch("claudelearnspokemon.enhanced_client.super")
    @patch("requests.Session.post")
    def test_reset_game_success(self, mock_post, mock_super):
        """Test successful reset_game operation."""
        # Mock session initialization
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"session_id": "test-session"}
        mock_post.return_value = init_response

        # Mock successful reset_game
        mock_super_instance = Mock()
        mock_super_instance.reset_game.return_value = {"reset": "complete"}
        mock_super.return_value = mock_super_instance

        result = self.client.reset_game()

        assert result == {"reset": "complete"}

    @patch("claudelearnspokemon.enhanced_client.super")
    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_enhanced_health_check(self, mock_get, mock_post, mock_super):
        """Test enhanced health check with session validation."""
        # Mock session initialization
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"session_id": "test-session"}
        mock_post.return_value = init_response

        # Mock session health check
        health_response = Mock()
        health_response.status_code = 200
        health_response.json.return_value = {"session_active": True, "session_id": "test-session"}
        mock_get.return_value = health_response

        # Mock parent health check
        mock_super_instance = Mock()
        mock_super_instance.is_healthy.return_value = True
        mock_super.return_value = mock_super_instance

        is_healthy = self.client.is_healthy()

        assert is_healthy is True

    def test_health_check_circuit_breaker_open(self):
        """Test health check returns False when circuit breaker is open."""
        # Force circuit breaker open
        self.client.circuit_breaker.force_open()

        is_healthy = self.client.is_healthy()

        assert is_healthy is False

    def test_force_session_refresh(self):
        """Test forced session refresh."""
        with patch("requests.Session.post") as mock_post:
            # Mock two different session responses
            responses = [
                Mock(raise_for_status=Mock(), json=Mock(return_value={"session_id": "session-1"})),
                Mock(raise_for_status=Mock(), json=Mock(return_value={"session_id": "session-2"})),
            ]
            mock_post.side_effect = responses

            session_id = self.client.force_session_refresh()

            assert session_id == "session-1"
            assert mock_post.call_count == 1

    def test_invalidate_session(self):
        """Test session invalidation."""
        self.client.invalidate_session()

        assert self.client.session_recovery.get_session_state() == SessionState.FAILED

    @patch("claudelearnspokemon.enhanced_client.super")
    @patch("requests.Session.post")
    def test_max_retries_exceeded_error(self, mock_post, mock_super):
        """Test error when max retries are exceeded."""
        # Mock session initialization
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"session_id": "test-session"}
        mock_post.return_value = init_response

        # Mock persistent failure
        mock_super_instance = Mock()
        mock_super_instance.send_input.side_effect = ConnectionError("Persistent failure")
        mock_super.return_value = mock_super_instance

        with pytest.raises(EmulatorPoolError) as exc_info:
            self.client.send_input("A")

        assert "Max retries exceeded" in str(exc_info.value)

    @patch("claudelearnspokemon.enhanced_client.super")
    @patch("requests.Session.post")
    def test_non_retriable_error_handling(self, mock_post, mock_super):
        """Test handling of non-retriable errors."""
        # Mock session initialization
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"session_id": "test-session"}
        mock_post.return_value = init_response

        # Mock non-retriable error (404)
        response_404 = Mock()
        response_404.status_code = 404
        error_404 = HTTPError("Not found", response=response_404)

        mock_super_instance = Mock()
        mock_super_instance.send_input.side_effect = error_404
        mock_super.return_value = mock_super_instance

        with pytest.raises(EmulatorPoolError) as exc_info:
            self.client.send_input("A")

        assert "Non-retriable error" in str(exc_info.value)

    def test_recovery_metrics_collection(self):
        """Test comprehensive recovery metrics collection."""
        metrics = self.client.get_recovery_metrics()

        assert "retry_metrics" in metrics
        assert "circuit_breaker" in metrics
        assert "session_info" in metrics
        assert "client_info" in metrics

        client_info = metrics["client_info"]
        assert client_info["port"] == 8081
        assert client_info["container_id"] == "test-container-123"

    def test_health_status_reporting(self):
        """Test comprehensive health status reporting."""
        with patch("claudelearnspokemon.enhanced_client.super") as mock_super:
            # Mock parent health check
            mock_super_instance = Mock()
            mock_super_instance.is_healthy.return_value = True
            mock_super.return_value = mock_super_instance

            health_status = self.client.get_health_status()

            assert "overall_healthy" in health_status
            assert "client_healthy" in health_status
            assert "circuit_breaker" in health_status
            assert "session_state" in health_status
            assert "recovery_metrics" in health_status

    def test_reset_recovery_state(self):
        """Test resetting all recovery components."""
        # Generate some state first
        with patch("requests.Session.post") as mock_post:
            init_response = Mock()
            init_response.raise_for_status.return_value = None
            init_response.json.return_value = {"session_id": "test-session"}
            mock_post.return_value = init_response

            try:
                self.client.session_recovery.ensure_session()
            except Exception:
                pass

        # Reset recovery state
        self.client.reset_recovery_state()

        # Verify reset
        assert self.client.circuit_breaker.get_state() == CircuitState.CLOSED
        assert self.client.session_recovery.get_session_state() == SessionState.FAILED
        retry_metrics = self.client.retry_manager.get_metrics()
        assert retry_metrics.total_attempts == 0

    def test_enhanced_close_cleanup(self):
        """Test enhanced cleanup on close."""
        with patch("claudelearnspokemon.enhanced_client.super") as mock_super:
            mock_super_instance = Mock()
            mock_super.return_value = mock_super_instance

            self.client.close()

            # Verify parent close was called
            mock_super_instance.close.assert_called_once()

    def test_string_representation(self):
        """Test enhanced string representation."""
        str_repr = str(self.client)

        assert "EnhancedPokemonGymClient" in str_repr
        assert "port=8081" in str_repr
        assert "container=test-contain" in str_repr  # Truncated to 12 chars from test-container-123
        assert "session_state=" in str_repr
        assert "circuit=" in str_repr

    @patch("claudelearnspokemon.enhanced_client.super")
    @patch("requests.Session.post")
    def test_operation_success_tracking(self, mock_post, mock_super):
        """Test tracking of successful operations."""
        # Mock session and operation
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"session_id": "test-session"}
        mock_post.return_value = init_response

        mock_super_instance = Mock()
        mock_super_instance.send_input.return_value = {"status": "success"}
        mock_super.return_value = mock_super_instance

        # Perform successful operation
        self.client.send_input("A")

        # Check that successful operation was tracked
        assert self.client._last_successful_operation is not None

    @patch("claudelearnspokemon.enhanced_client.super")
    @patch("requests.Session.post")
    def test_session_error_invalidates_session(self, mock_post, mock_super):
        """Test that session errors properly invalidate the session."""
        # Mock session initialization
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"session_id": "test-session"}
        mock_post.return_value = init_response

        # Mock session error
        session_error = Exception("Session expired")
        mock_super_instance = Mock()
        mock_super_instance.send_input.side_effect = session_error
        mock_super.return_value = mock_super_instance

        # Simulate session error detection
        self.client.session_recovery.handle_session_error(session_error)

        # Session should be invalidated
        assert self.client.session_recovery.get_session_state() == SessionState.FAILED
