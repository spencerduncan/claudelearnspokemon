"""
Tests for SessionRecovery: Kernel-quality session management and recovery.

Comprehensive test coverage for session lifecycle management:
- Session initialization and recovery
- Session expiration detection and handling
- Connection state validation and restoration
- Health monitoring and metrics collection

Author: Linus Torbot - Kernel Quality Standards Applied
"""

import time
from unittest.mock import Mock, patch

import pytest
from requests.exceptions import ConnectionError

from claudelearnspokemon.session_recovery import (
    SessionConfig,
    SessionRecovery,
    SessionRecoveryError,
    SessionState,
)


@pytest.mark.fast
class TestSessionRecovery:
    """Test suite for SessionRecovery functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.base_url = "http://localhost:8080"
        self.config = SessionConfig(
            session_timeout=10.0,  # Short timeout for testing
            health_check_interval=2.0,
            initialization_timeout=5.0,
            max_recovery_attempts=2,
        )
        self.session_recovery = SessionRecovery(self.base_url, self.config)

    def test_initialization_default_config(self):
        """Test SessionRecovery initializes with default configuration."""
        sr = SessionRecovery("http://localhost:8080")

        assert sr.base_url == "http://localhost:8080"
        assert sr.config.session_timeout == 3600.0
        assert sr.get_session_state() == SessionState.UNINITIALIZED

    def test_initialization_custom_config(self):
        """Test SessionRecovery initializes with custom configuration."""
        assert self.session_recovery.base_url == "http://localhost:8080"
        assert self.session_recovery.config.session_timeout == 10.0
        assert self.session_recovery.get_session_state() == SessionState.UNINITIALIZED

    @patch("requests.Session.post")
    def test_successful_session_initialization(self, mock_post):
        """Test successful session initialization."""
        # Mock successful initialization response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"session_id": "test-session-123"}
        mock_post.return_value = mock_response

        session_id = self.session_recovery.ensure_session()

        assert session_id == "test-session-123"
        assert self.session_recovery.get_session_id() == "test-session-123"
        assert self.session_recovery.get_session_state() == SessionState.ACTIVE

        # Verify initialization call
        mock_post.assert_called_once_with(
            f"{self.base_url}/initialize",
            json=self.config.initialization_payload,
            timeout=self.config.initialization_timeout,
        )

    @patch("requests.Session.post")
    def test_session_initialization_failure(self, mock_post):
        """Test session initialization failure handling."""
        mock_post.side_effect = ConnectionError("Connection failed")

        with pytest.raises(SessionRecoveryError) as exc_info:
            self.session_recovery.ensure_session()

        assert "Session initialization failed" in str(exc_info.value)
        assert self.session_recovery.get_session_state() == SessionState.FAILED

    @patch("requests.Session.post")
    def test_session_initialization_no_session_id(self, mock_post):
        """Test session initialization with missing session_id in response."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "initialized"}  # No session_id
        mock_post.return_value = mock_response

        with pytest.raises(SessionRecoveryError) as exc_info:
            self.session_recovery.ensure_session()

        assert "No session_id in initialization response" in str(exc_info.value)

    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_session_health_check_success(self, mock_get, mock_post):
        """Test successful session health check."""
        # Initialize session first
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"session_id": "test-session-123"}
        mock_post.return_value = init_response

        # Mock health check response
        health_response = Mock()
        health_response.status_code = 200
        health_response.json.return_value = {
            "session_active": True,
            "session_id": "test-session-123",
        }
        mock_get.return_value = health_response

        session_id = self.session_recovery.ensure_session()

        # Second call should use existing session (health check passes)
        session_id2 = self.session_recovery.ensure_session()

        assert session_id == session_id2
        assert mock_post.call_count == 1  # Only one initialization
        mock_get.assert_called_with(f"{self.base_url}/status", timeout=5.0)

    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_session_health_check_failure_triggers_recovery(self, mock_get, mock_post):
        """Test session recovery when health check fails."""
        # Initialize session first
        init_response = Mock()
        init_response.raise_for_status.return_value = None
        init_response.json.return_value = {"session_id": "original-session"}
        mock_post.return_value = init_response

        self.session_recovery.ensure_session()

        # Health check fails
        mock_get.side_effect = ConnectionError("Health check failed")

        # Recovery initialization
        recovery_response = Mock()
        recovery_response.raise_for_status.return_value = None
        recovery_response.json.return_value = {"session_id": "recovered-session"}
        mock_post.return_value = recovery_response

        session_id = self.session_recovery.ensure_session()

        assert session_id == "recovered-session"
        assert mock_post.call_count == 2  # Original + recovery

    def test_session_timeout_detection(self):
        """Test session timeout detection."""
        # Manually set session state to simulate initialized session
        self.session_recovery._session_id = "test-session"
        self.session_recovery._session_start_time = time.time() - 15.0  # 15 seconds ago
        self.session_recovery._state = SessionState.ACTIVE

        # Session should be considered expired
        needs_refresh = self.session_recovery._needs_session_refresh()
        assert needs_refresh is True
        assert self.session_recovery.get_session_state() == SessionState.EXPIRED

    def test_force_session_refresh(self):
        """Test forced session refresh."""
        with patch("requests.Session.post") as mock_post:
            # Mock initialization responses
            mock_response1 = Mock()
            mock_response1.raise_for_status.return_value = None
            mock_response1.json.return_value = {"session_id": "session-1"}

            mock_response2 = Mock()
            mock_response2.raise_for_status.return_value = None
            mock_response2.json.return_value = {"session_id": "session-2"}

            mock_post.side_effect = [mock_response1, mock_response2]

            # First initialization
            session1 = self.session_recovery.ensure_session()
            assert session1 == "session-1"

            # Force refresh
            session2 = self.session_recovery.ensure_session(force_refresh=True)
            assert session2 == "session-2"
            assert mock_post.call_count == 2

    def test_session_error_detection(self):
        """Test detection of session-related errors."""
        test_cases = [
            (Exception("Session expired"), True),
            (Exception("Unauthorized access"), True),
            (ValueError("Invalid session token"), True),
            (ConnectionError("Network timeout"), False),
            (Exception("Random error"), False),
        ]

        for error, expected in test_cases:
            result = self.session_recovery.is_session_error(error)
            assert result == expected, f"Failed for error: {error}"

    def test_session_error_detection_http_status(self):
        """Test session error detection with HTTP status codes."""
        # Mock HTTP errors with different status codes
        response_401 = Mock()
        response_401.status_code = 401
        error_401 = Exception()
        error_401.response = response_401

        response_500 = Mock()
        response_500.status_code = 500
        error_500 = Exception()
        error_500.response = response_500

        assert self.session_recovery.is_session_error(error_401) is True
        assert self.session_recovery.is_session_error(error_500) is False

    def test_handle_session_error(self):
        """Test session error handling."""
        # Set up active session
        self.session_recovery._session_id = "test-session"
        self.session_recovery._state = SessionState.ACTIVE

        error = Exception("Session expired")
        self.session_recovery.handle_session_error(error)

        assert self.session_recovery.get_session_state() == SessionState.FAILED
        assert self.session_recovery.get_session_id() is None

    def test_invalidate_session(self):
        """Test manual session invalidation."""
        # Set up active session
        self.session_recovery._session_id = "test-session"
        self.session_recovery._state = SessionState.ACTIVE

        self.session_recovery.invalidate_session()

        assert self.session_recovery.get_session_state() == SessionState.FAILED
        assert self.session_recovery.get_session_id() is None

    @patch("requests.Session.post")
    def test_max_recovery_attempts_exceeded(self, mock_post):
        """Test behavior when max recovery attempts are exceeded."""
        mock_post.side_effect = ConnectionError("Persistent failure")

        # Make max_recovery_attempts calls, each should fail
        for _ in range(self.config.max_recovery_attempts):
            with pytest.raises(SessionRecoveryError):
                self.session_recovery.ensure_session()

        # Next attempt should raise with max attempts message
        with pytest.raises(SessionRecoveryError) as exc_info:
            self.session_recovery.ensure_session()

        assert "Max recovery attempts" in str(exc_info.value)

    def test_metrics_collection(self):
        """Test session metrics collection."""
        with patch("requests.Session.post") as mock_post:
            # Mock successful initialization
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"session_id": "test-session"}
            mock_post.return_value = mock_response

            self.session_recovery.ensure_session()

            metrics = self.session_recovery.get_metrics()
            assert metrics.total_initializations == 1
            assert metrics.last_initialization_time is not None

    def test_session_info_reporting(self):
        """Test comprehensive session information reporting."""
        info = self.session_recovery.get_session_info()

        assert "session_id" in info
        assert "state" in info
        assert "session_age" in info
        assert "recovery_attempts" in info
        assert "base_url" in info
        assert "is_healthy" in info
        assert "metrics" in info

    @patch("requests.Session.get")
    def test_health_check_session_id_mismatch(self, mock_get):
        """Test health check with session ID mismatch."""
        # Set up session
        self.session_recovery._session_id = "original-session"
        self.session_recovery._state = SessionState.ACTIVE

        # Mock health response with different session ID
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "session_active": True,
            "session_id": "different-session",
        }
        mock_get.return_value = mock_response

        is_healthy = self.session_recovery._is_session_healthy()
        assert is_healthy is False

    @patch("requests.Session.get")
    def test_health_check_session_inactive(self, mock_get):
        """Test health check when session is reported as inactive."""
        # Set up session
        self.session_recovery._session_id = "test-session"
        self.session_recovery._state = SessionState.ACTIVE

        # Mock health response with inactive session
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"session_active": False, "session_id": "test-session"}
        mock_get.return_value = mock_response

        is_healthy = self.session_recovery._is_session_healthy()
        assert is_healthy is False

    @patch("requests.Session.post")
    def test_graceful_cleanup_on_close(self, mock_post):
        """Test graceful session cleanup on close."""
        # Set up active session
        with patch("requests.Session.post") as mock_init:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"session_id": "test-session"}
            mock_init.return_value = mock_response

            self.session_recovery.ensure_session()

        # Mock stop response
        mock_stop_response = Mock()
        mock_stop_response.raise_for_status.return_value = None
        mock_post.return_value = mock_stop_response

        self.session_recovery.close()

        # Verify stop was called
        mock_post.assert_called_with(
            f"{self.base_url}/stop", json={"session_id": "test-session"}, timeout=5.0
        )

        assert self.session_recovery.get_session_state() == SessionState.UNINITIALIZED

    def test_string_representation(self):
        """Test SessionRecovery string representation."""
        str_repr = str(self.session_recovery)
        assert "SessionRecovery" in str_repr
        assert "UNINITIALIZED" in str_repr

    def test_base_url_normalization(self):
        """Test base URL normalization (trailing slash removal)."""
        sr = SessionRecovery("http://localhost:8080/")
        assert sr.base_url == "http://localhost:8080"

    @patch("requests.Session.post")
    def test_recovery_success_metrics(self, mock_post):
        """Test recovery success metrics tracking."""
        # First call fails, second succeeds
        success_response = Mock()
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = {"session_id": "recovered-session"}

        mock_post.side_effect = [ConnectionError("First attempt fails"), success_response]

        # First call should fail and trigger recovery attempt
        try:
            self.session_recovery.ensure_session()
        except SessionRecoveryError:
            pass  # Expected on first attempt

        # Second call should succeed as recovery
        session_id = self.session_recovery.ensure_session()
        assert session_id == "recovered-session"

        metrics = self.session_recovery.get_metrics()
        assert metrics.successful_recoveries == 1
        assert metrics.failed_recoveries == 1  # One failure before success

    @patch("requests.Session.post")
    def test_recovery_failure_metrics(self, mock_post):
        """Test recovery failure metrics tracking."""
        mock_post.side_effect = ConnectionError("Persistent failure")

        # Attempt recovery until max attempts
        for _ in range(self.config.max_recovery_attempts):
            try:
                self.session_recovery.ensure_session()
            except SessionRecoveryError:
                pass

        metrics = self.session_recovery.get_metrics()
        assert metrics.failed_recoveries == self.config.max_recovery_attempts
