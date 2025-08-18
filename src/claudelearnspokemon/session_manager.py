"""
Simple Session Lifecycle Management for Pokemon Gym Adapter.

Implements basic session lifecycle as specified in Issue #142:
- Auto-initialize session on first API call
- Track session ID and state per adapter instance
- Proper session cleanup on shutdown
- Handle session expiration gracefully
- Support session configuration options

Author: Bot Dean - Production-Ready Simplicity
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import requests

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Simple session states matching Issue #142 requirements."""

    UNINITIALIZED = auto()
    ACTIVE = auto()
    EXPIRED = auto()
    TERMINATED = auto()


@dataclass
class SessionConfig:
    """Session configuration options."""

    session_timeout_seconds: float = 3600.0  # 1 hour default
    headless: bool = True
    sound: bool = False
    auto_initialize: bool = True
    base_url: str = "http://localhost:8080"


@dataclass
class SessionInfo:
    """Simple session information tracking."""

    session_id: str | None = None
    state: SessionState = SessionState.UNINITIALIZED
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    config: SessionConfig = field(default_factory=SessionConfig)


class SessionManager:
    """
    Simple session lifecycle management for Pokemon Gym Adapter.

    Implements only the requirements from Issue #142:
    - Auto-initialize session on first API call
    - Track session ID and state per adapter instance
    - Proper session cleanup on shutdown
    - Handle session expiration gracefully
    - Support session configuration options
    """

    def __init__(self, config: SessionConfig | None = None):
        """Initialize session manager with configuration."""
        self.config = config or SessionConfig()
        self.session_info = SessionInfo(config=self.config)
        self.http_session = requests.Session()
        self._shutdown = False

        logger.info("SessionManager initialized with config: %s", self.config)

    def ensure_session(self) -> str:
        """
        Ensure session is active, auto-initializing if needed.

        Returns:
            str: Active session ID

        Raises:
            RuntimeError: If session initialization fails
        """
        if self._shutdown:
            raise RuntimeError("Session manager has been shut down")

        # Check if we need to initialize or reinitialize
        if self.session_info.state == SessionState.UNINITIALIZED or self._is_session_expired():
            self._initialize_session()

        # Update last activity
        self.session_info.last_activity = time.time()

        # After initialization, session_id should never be None
        assert self.session_info.session_id is not None
        return self.session_info.session_id

    def get_session_id(self) -> str | None:
        """Get current session ID without auto-initialization."""
        return self.session_info.session_id

    def get_session_state(self) -> SessionState:
        """Get current session state."""
        if self._is_session_expired() and self.session_info.state == SessionState.ACTIVE:
            self.session_info.state = SessionState.EXPIRED
            logger.info("Session %s marked as expired", self.session_info.session_id)

        return self.session_info.state

    def is_initialized(self) -> bool:
        """Check if session is initialized and active."""
        return self.session_info.state == SessionState.ACTIVE and not self._is_session_expired()

    def shutdown(self, timeout_seconds: float = 10.0) -> None:
        """
        Proper session cleanup on shutdown.

        Args:
            timeout_seconds: Maximum time to wait for cleanup
        """
        if self._shutdown:
            return

        self._shutdown = True
        logger.info("Shutting down session manager")

        # Clean up active session
        if self.session_info.state == SessionState.ACTIVE:
            try:
                self._terminate_session()
            except Exception as e:
                logger.error("Error during session termination: %s", e)

        # Close HTTP session
        self.http_session.close()

        logger.info("Session manager shutdown complete")

    def _initialize_session(self) -> None:
        """Initialize new session with Pokemon Gym endpoint."""
        logger.info("Initializing new session")

        try:
            # Call Pokemon Gym initialize endpoint
            response = self.http_session.post(
                f"{self.config.base_url}/initialize",
                json={"headless": self.config.headless, "sound": self.config.sound},
                timeout=30.0,
            )
            response.raise_for_status()

            data = response.json()
            session_id = data.get("session_id")

            if not session_id:
                raise RuntimeError("No session_id in initialize response")

            # Update session info
            self.session_info.session_id = session_id
            self.session_info.state = SessionState.ACTIVE
            self.session_info.created_at = time.time()
            self.session_info.last_activity = time.time()

            logger.info("Session initialized successfully: %s", session_id)

        except requests.RequestException as e:
            self.session_info.state = SessionState.UNINITIALIZED
            error_msg = f"Failed to initialize session: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _terminate_session(self) -> None:
        """Terminate active session."""
        if not self.session_info.session_id:
            return

        logger.info("Terminating session: %s", self.session_info.session_id)

        try:
            # Call Pokemon Gym terminate endpoint if available
            response = self.http_session.post(
                f"{self.config.base_url}/terminate",
                json={"session_id": self.session_info.session_id},
                timeout=10.0,
            )
            # Don't raise for status - termination is best effort
            if response.status_code == 200:
                logger.info("Session terminated successfully")
            else:
                logger.warning("Session termination returned status %d", response.status_code)

        except requests.RequestException as e:
            logger.warning("Error terminating session (continuing): %s", e)
        finally:
            self.session_info.state = SessionState.TERMINATED
            self.session_info.session_id = None

    def _is_session_expired(self) -> bool:
        """Check if current session has expired."""
        # If already marked as expired, return True
        if self.session_info.state == SessionState.EXPIRED:
            return True

        # Only active sessions can expire
        if self.session_info.state != SessionState.ACTIVE:
            return False

        if not self.session_info.session_id:
            return True

        # Check timeout
        elapsed = time.time() - self.session_info.created_at
        if elapsed > self.config.session_timeout_seconds:
            logger.info(
                "Session %s expired (%.1fs > %.1fs)",
                self.session_info.session_id,
                elapsed,
                self.config.session_timeout_seconds,
            )
            return True

        return False


# Factory function for easy instantiation
def create_session_manager(
    base_url: str = "http://localhost:8080",
    session_timeout_seconds: float = 3600.0,
    headless: bool = True,
    sound: bool = False,
) -> SessionManager:
    """
    Create a simple session manager with common configuration.

    Args:
        base_url: Pokemon Gym service URL
        session_timeout_seconds: Session timeout (default 1 hour)
        headless: Run in headless mode
        sound: Enable sound

    Returns:
        SessionManager: Configured session manager
    """
    config = SessionConfig(
        base_url=base_url,
        session_timeout_seconds=session_timeout_seconds,
        headless=headless,
        sound=sound,
    )
    return SessionManager(config)
