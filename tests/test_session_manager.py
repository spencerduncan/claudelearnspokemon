"""
Tests for Simple Session Lifecycle Management.

Tests basic session lifecycle as specified in Issue #142:
- Auto-initialize session on first API call
- Track session ID and state per adapter instance
- Proper session cleanup on shutdown
- Handle session expiration gracefully
- Support session configuration options

Author: Bot Dean - Production-Ready Testing
"""

import time
import unittest
from unittest.mock import Mock, patch

import pytest
import requests

from claudelearnspokemon.session_manager import (
    SessionConfig,
    SessionInfo,
    SessionManager,
    SessionState,
    create_session_manager,
)


@pytest.mark.unit
@pytest.mark.fast
class TestSessionConfig(unittest.TestCase):
    """Test SessionConfig data class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SessionConfig()

        assert config.session_timeout_seconds == 3600.0  # 1 hour
        assert config.headless is True
        assert config.sound is False
        assert config.auto_initialize is True
        assert config.base_url == "http://localhost:8080"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SessionConfig(
            session_timeout_seconds=1800.0,  # 30 minutes
            headless=False,
            sound=True,
            base_url="http://localhost:9090",
        )

        assert config.session_timeout_seconds == 1800.0
        assert config.headless is False
        assert config.sound is True
        assert config.base_url == "http://localhost:9090"


@pytest.mark.unit
@pytest.mark.fast
class TestSessionInfo(unittest.TestCase):
    """Test SessionInfo data class."""

    def test_default_session_info(self):
        """Test default session info values."""
        info = SessionInfo()

        assert info.session_id is None
        assert info.state == SessionState.UNINITIALIZED
        assert info.created_at <= time.time()
        assert info.last_activity <= time.time()
        assert isinstance(info.config, SessionConfig)


@pytest.mark.unit
@pytest.mark.fast
class TestSessionManager(unittest.TestCase):
    """Test SessionManager core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SessionConfig(
            base_url="http://localhost:8080",
            session_timeout_seconds=10.0,  # Short timeout for testing
            headless=True,
            sound=False,
        )
        self.session_manager = SessionManager(self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        self.session_manager.shutdown()

    def test_initialization(self):
        """Test session manager initialization."""
        assert self.session_manager.config == self.config
        assert self.session_manager.session_info.state == SessionState.UNINITIALIZED
        assert self.session_manager.session_info.session_id is None
        assert self.session_manager.is_initialized() is False

    @patch("requests.Session.post")
    def test_ensure_session_auto_initialize(self, mock_post):
        """Test auto-initialization on first ensure_session call."""
        # Mock successful initialization response
        mock_response = Mock()
        mock_response.json.return_value = {"session_id": "test-session-123"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # First call should initialize
        session_id = self.session_manager.ensure_session()

        assert session_id == "test-session-123"
        assert self.session_manager.session_info.session_id == "test-session-123"
        assert self.session_manager.session_info.state == SessionState.ACTIVE
        assert self.session_manager.is_initialized() is True

        # Verify initialize endpoint was called
        mock_post.assert_called_once_with(
            "http://localhost:8080/initialize",
            json={"headless": True, "sound": False},
            timeout=30.0,
        )

    @patch("requests.Session.post")
    def test_ensure_session_reuse_active(self, mock_post):
        """Test session reuse when already active."""
        # Mock successful initialization
        mock_response = Mock()
        mock_response.json.return_value = {"session_id": "test-session-456"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # First call initializes
        session_id1 = self.session_manager.ensure_session()
        assert session_id1 == "test-session-456"

        # Second call should reuse existing session
        session_id2 = self.session_manager.ensure_session()
        assert session_id2 == "test-session-456"
        assert session_id1 == session_id2

        # Initialize should only be called once
        assert mock_post.call_count == 1

    @patch("requests.Session.post")
    def test_session_expiration_handling(self, mock_post):
        """Test graceful handling of session expiration."""
        # Mock first successful initialization
        first_response = Mock()
        first_response.json.return_value = {"session_id": "test-session-789"}
        first_response.raise_for_status.return_value = None

        # Mock second successful initialization
        second_response = Mock()
        second_response.json.return_value = {"session_id": "test-session-new"}
        second_response.raise_for_status.return_value = None

        mock_post.side_effect = [first_response, second_response]

        # Initialize session
        session_id1 = self.session_manager.ensure_session()
        assert session_id1 == "test-session-789"

        # Force expiration by manipulating created_at
        self.session_manager.session_info.created_at = time.time() - 20.0  # 20 seconds ago

        # Check that session is detected as expired
        assert self.session_manager.get_session_state() == SessionState.EXPIRED

        # Ensure session should reinitialize
        session_id2 = self.session_manager.ensure_session()
        assert session_id2 == "test-session-new"
        assert session_id1 != session_id2

        # Should have called initialize twice
        assert mock_post.call_count == 2

    @patch("requests.Session.post")
    def test_initialization_failure(self, mock_post):
        """Test handling of initialization failures."""
        # Mock failed initialization
        mock_post.side_effect = requests.RequestException("Connection failed")

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to initialize session"):
            self.session_manager.ensure_session()

        # Session should remain uninitialized
        assert self.session_manager.session_info.state == SessionState.UNINITIALIZED
        assert self.session_manager.session_info.session_id is None
        assert self.session_manager.is_initialized() is False

    @patch("requests.Session.post")
    def test_missing_session_id_in_response(self, mock_post):
        """Test handling when initialize response lacks session_id."""
        # Mock response without session_id
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok"}  # No session_id
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="No session_id in initialize response"):
            self.session_manager.ensure_session()

        assert self.session_manager.session_info.state == SessionState.UNINITIALIZED

    def test_get_session_id_uninitialized(self):
        """Test getting session ID when uninitialized."""
        session_id = self.session_manager.get_session_id()
        assert session_id is None

    @patch("requests.Session.post")
    def test_get_session_id_initialized(self, mock_post):
        """Test getting session ID when initialized."""
        # Mock successful initialization
        mock_response = Mock()
        mock_response.json.return_value = {"session_id": "test-session-get"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Initialize
        self.session_manager.ensure_session()

        # Get session ID
        session_id = self.session_manager.get_session_id()
        assert session_id == "test-session-get"

    @patch("requests.Session.post")
    def test_shutdown_with_active_session(self, mock_post):
        """Test proper cleanup during shutdown."""
        # Mock successful initialization
        init_response = Mock()
        init_response.json.return_value = {"session_id": "test-session-shutdown"}
        init_response.raise_for_status.return_value = None

        # Mock successful termination
        term_response = Mock()
        term_response.status_code = 200

        mock_post.side_effect = [init_response, term_response]

        # Initialize session
        self.session_manager.ensure_session()
        assert self.session_manager.session_info.state == SessionState.ACTIVE

        # Shutdown
        self.session_manager.shutdown()

        # Should have called terminate
        assert mock_post.call_count == 2
        terminate_call = mock_post.call_args_list[1]
        assert terminate_call[0][0] == "http://localhost:8080/terminate"
        assert terminate_call[1]["json"] == {"session_id": "test-session-shutdown"}

        # Session should be terminated
        assert self.session_manager.session_info.state == SessionState.TERMINATED
        assert self.session_manager.session_info.session_id is None

    @patch("requests.Session.post")
    def test_shutdown_termination_failure(self, mock_post):
        """Test shutdown handles termination failures gracefully."""
        # Mock successful initialization
        init_response = Mock()
        init_response.json.return_value = {"session_id": "test-session-term-fail"}
        init_response.raise_for_status.return_value = None

        # Mock termination failure
        mock_post.side_effect = [init_response, requests.RequestException("Termination failed")]

        # Initialize session
        self.session_manager.ensure_session()

        # Shutdown should handle termination failure gracefully
        self.session_manager.shutdown()  # Should not raise

        # Session should still be marked as terminated
        assert self.session_manager.session_info.state == SessionState.TERMINATED

    def test_ensure_session_after_shutdown(self):
        """Test ensure_session fails after shutdown."""
        self.session_manager.shutdown()

        with pytest.raises(RuntimeError, match="Session manager has been shut down"):
            self.session_manager.ensure_session()

    def test_multiple_shutdowns(self):
        """Test multiple shutdown calls are safe."""
        self.session_manager.shutdown()
        self.session_manager.shutdown()  # Should not raise

        # Should still be shut down
        with pytest.raises(RuntimeError, match="Session manager has been shut down"):
            self.session_manager.ensure_session()


@pytest.mark.unit
@pytest.mark.fast
class TestSessionManagerFactory(unittest.TestCase):
    """Test factory function."""

    def test_create_session_manager_defaults(self):
        """Test factory with default parameters."""
        manager = create_session_manager()

        assert manager.config.base_url == "http://localhost:8080"
        assert manager.config.session_timeout_seconds == 3600.0
        assert manager.config.headless is True
        assert manager.config.sound is False

        manager.shutdown()

    def test_create_session_manager_custom(self):
        """Test factory with custom parameters."""
        manager = create_session_manager(
            base_url="http://localhost:9090",
            session_timeout_seconds=1800.0,
            headless=False,
            sound=True,
        )

        assert manager.config.base_url == "http://localhost:9090"
        assert manager.config.session_timeout_seconds == 1800.0
        assert manager.config.headless is False
        assert manager.config.sound is True

        manager.shutdown()


if __name__ == "__main__":
    # Configure logging for tests
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run tests
    unittest.main()
