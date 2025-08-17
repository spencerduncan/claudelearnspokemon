"""
SessionRecovery: Kernel-quality session management and recovery.

Implements production-grade session lifecycle management:
- Automatic session initialization and recovery
- Session expiration detection and handling
- Connection state validation and restoration
- Integration with retry and circuit breaker patterns

Author: Linus Torbot - Kernel Quality Standards Applied
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import requests
from requests.exceptions import RequestException

# Configure logging for production observability
logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session lifecycle states."""

    UNINITIALIZED = auto()  # Session not yet created
    ACTIVE = auto()  # Session active and healthy
    EXPIRED = auto()  # Session expired, needs refresh
    FAILED = auto()  # Session failed, needs recreation
    RECOVERING = auto()  # Session recovery in progress


@dataclass
class SessionConfig:
    """Configuration for session management."""

    session_timeout: float = 3600.0  # Session timeout in seconds (1 hour)
    health_check_interval: float = 300.0  # Health check every 5 minutes
    initialization_timeout: float = 30.0  # Max time for session init
    max_recovery_attempts: int = 3  # Max recovery attempts before giving up
    initialization_payload: dict[str, Any] | None = None  # Payload for session init

    def __post_init__(self) -> None:
        """Set default initialization payload if not provided."""
        if self.initialization_payload is None:
            self.initialization_payload = {
                "headless": True,
                "sound": False,
                "timeout": self.initialization_timeout,
            }


@dataclass
class SessionMetrics:
    """Metrics for session operations."""

    total_initializations: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    session_timeouts: int = 0
    health_checks: int = 0
    session_duration_total: float = 0.0
    last_initialization_time: float | None = None
    last_health_check_time: float | None = None

    def record_initialization(self) -> None:
        """Record session initialization."""
        self.total_initializations += 1
        self.last_initialization_time = time.time()

    def record_successful_recovery(self) -> None:
        """Record successful session recovery."""
        self.successful_recoveries += 1

    def record_failed_recovery(self) -> None:
        """Record failed session recovery."""
        self.failed_recoveries += 1

    def record_session_timeout(self) -> None:
        """Record session timeout."""
        self.session_timeouts += 1

    def record_health_check(self) -> None:
        """Record health check attempt."""
        self.health_checks += 1
        self.last_health_check_time = time.time()

    @property
    def recovery_success_rate(self) -> float:
        """Calculate recovery success rate percentage."""
        total_recoveries = self.successful_recoveries + self.failed_recoveries
        if total_recoveries == 0:
            return 100.0
        return (self.successful_recoveries / total_recoveries) * 100.0


class SessionRecoveryError(Exception):
    """Custom exception for session recovery operations."""

    pass


class SessionRecovery:
    """
    Production-grade session recovery manager.

    Implements kernel engineering principles:
    - Robust session lifecycle management
    - Automatic recovery from session failures
    - Health monitoring and validation
    - Minimal overhead session tracking
    - Comprehensive operational metrics
    """

    def __init__(self, base_url: str, config: SessionConfig | None = None):
        """
        Initialize session recovery manager.

        Args:
            base_url: Base URL for the service
            config: Session configuration (uses defaults if None)
        """
        self.base_url = base_url.rstrip("/")
        self.config = config or SessionConfig()
        self.metrics = SessionMetrics()

        # Session state tracking
        self._state = SessionState.UNINITIALIZED
        self._session_id: str | None = None
        self._session_start_time: float | None = None
        self._recovery_attempts = 0

        # HTTP session for persistent connections
        self._http_session = requests.Session()

        logger.info(
            f"SessionRecovery initialized for {base_url}, "
            f"timeout={self.config.session_timeout}s"
        )

    def ensure_session(self, force_refresh: bool = False) -> str:
        """
        Ensure valid session exists, initializing or recovering as needed.

        Args:
            force_refresh: Force session refresh even if current session appears valid

        Returns:
            Valid session ID

        Raises:
            SessionRecoveryError: If session cannot be established or recovered
        """
        if force_refresh or self._needs_session_refresh():
            return self._initialize_or_recover_session()

        # Validate current session is still healthy
        if self._session_id and self._is_session_healthy():
            return self._session_id

        # Session validation failed, attempt recovery
        logger.warning("Session health check failed, attempting recovery")
        return self._initialize_or_recover_session()

    def get_session_id(self) -> str | None:
        """
        Get current session ID without validation.

        Returns:
            Current session ID or None if no session
        """
        return self._session_id

    def get_session_state(self) -> SessionState:
        """
        Get current session state.

        Returns:
            Current session state
        """
        return self._state

    def invalidate_session(self) -> None:
        """
        Invalidate current session, forcing reinitialization on next use.

        This is useful when external components detect session issues.
        """
        logger.info("Session manually invalidated")
        self._state = SessionState.FAILED
        self._session_id = None
        self._session_start_time = None
        self._recovery_attempts = 0

    def _needs_session_refresh(self) -> bool:
        """
        Determine if session needs refresh based on current state and time.

        Returns:
            True if session needs refresh
        """
        if self._state in {SessionState.UNINITIALIZED, SessionState.FAILED}:
            return True

        if self._state == SessionState.EXPIRED:
            return True

        # Check session timeout
        if self._session_start_time:
            session_age = time.time() - self._session_start_time
            if session_age >= self.config.session_timeout:
                logger.info(f"Session expired after {session_age:.1f}s")
                self._state = SessionState.EXPIRED
                self.metrics.record_session_timeout()
                return True

        return False

    def _initialize_or_recover_session(self) -> str:
        """
        Initialize new session or recover from failure.

        Returns:
            Valid session ID

        Raises:
            SessionRecoveryError: If session cannot be established
        """
        if self._recovery_attempts >= self.config.max_recovery_attempts:
            raise SessionRecoveryError(
                f"Max recovery attempts ({self.config.max_recovery_attempts}) exceeded. "
                f"Service may be unavailable."
            )

        try:
            self._state = SessionState.RECOVERING
            self._recovery_attempts += 1

            logger.info(f"Initializing session, attempt {self._recovery_attempts}")

            # Call session initialization endpoint
            response = self._http_session.post(
                f"{self.base_url}/initialize",
                json=self.config.initialization_payload,
                timeout=self.config.initialization_timeout,
            )

            response.raise_for_status()
            session_data = response.json()

            # Extract session ID from response
            session_id = session_data.get("session_id")
            if not session_id:
                raise SessionRecoveryError("No session_id in initialization response")

            # Update session state
            self._session_id = session_id
            self._session_start_time = time.time()
            self._state = SessionState.ACTIVE

            self.metrics.record_initialization()
            # Record successful recovery if this was a recovery attempt (attempts > 0 means recovery)
            if self._recovery_attempts > 0:
                self.metrics.record_successful_recovery()

            self._recovery_attempts = 0  # Reset on success

            logger.info(f"Session initialized successfully: {session_id}")
            return session_id

        except RequestException as e:
            logger.error(f"Session initialization failed: {e}")
            self._state = SessionState.FAILED
            self.metrics.record_failed_recovery()

            # If this was a recovery attempt, re-raise with context
            if self._recovery_attempts > 1:
                raise SessionRecoveryError(
                    f"Session recovery failed on attempt {self._recovery_attempts}: {e}"
                ) from e
            else:
                raise SessionRecoveryError(f"Session initialization failed: {e}") from e

    def _is_session_healthy(self) -> bool:
        """
        Check if current session is healthy and responsive.

        Returns:
            True if session is healthy
        """
        if not self._session_id:
            return False

        try:
            self.metrics.record_health_check()

            # Check service status endpoint
            response = self._http_session.get(f"{self.base_url}/status", timeout=5.0)

            if response.status_code != 200:
                logger.warning(f"Session health check failed: HTTP {response.status_code}")
                return False

            status_data = response.json()

            # Check if session is still active
            session_active = status_data.get("session_active", False)
            current_session = status_data.get("session_id")

            # Validate session ID matches
            if current_session != self._session_id:
                logger.warning(
                    f"Session ID mismatch: expected {self._session_id}, " f"got {current_session}"
                )
                return False

            if not session_active:
                logger.warning("Session is no longer active according to service")
                return False

            logger.debug("Session health check passed")
            return True

        except RequestException as e:
            logger.warning(f"Session health check failed: {e}")
            return False

    def is_session_error(self, exception: Exception) -> bool:
        """
        Determine if exception indicates a session-related error.

        Args:
            exception: Exception to analyze

        Returns:
            True if exception is session-related
        """
        # Check for explicit session error indicators
        error_message = str(exception).lower()
        session_indicators = [
            "session",
            "unauthorized",
            "authentication",
            "expired",
            "invalid",
            "401",
            "403",
        ]

        for indicator in session_indicators:
            if indicator in error_message:
                return True

        # Check HTTP status codes for session errors
        if hasattr(exception, "response") and exception.response:
            status_code = exception.response.status_code
            if status_code in {401, 403}:
                return True

        return False

    def handle_session_error(self, exception: Exception) -> None:
        """
        Handle session-related error by invalidating current session.

        Args:
            exception: Session-related exception
        """
        logger.warning(f"Session error detected: {exception}")
        self._state = SessionState.FAILED
        self._session_id = None
        self._session_start_time = None
        # Don't reset recovery attempts - let them accumulate

    def get_metrics(self) -> SessionMetrics:
        """
        Get current session metrics for monitoring.

        Returns:
            Copy of current session metrics
        """
        return SessionMetrics(
            total_initializations=self.metrics.total_initializations,
            successful_recoveries=self.metrics.successful_recoveries,
            failed_recoveries=self.metrics.failed_recoveries,
            session_timeouts=self.metrics.session_timeouts,
            health_checks=self.metrics.health_checks,
            session_duration_total=self.metrics.session_duration_total,
            last_initialization_time=self.metrics.last_initialization_time,
            last_health_check_time=self.metrics.last_health_check_time,
        )

    def get_session_info(self) -> dict[str, Any]:
        """
        Get comprehensive session information for monitoring.

        Returns:
            Dictionary with session information
        """
        session_age = None
        if self._session_start_time:
            session_age = time.time() - self._session_start_time

        return {
            "session_id": self._session_id,
            "state": self._state.name,
            "session_age": session_age,
            "recovery_attempts": self._recovery_attempts,
            "base_url": self.base_url,
            "is_healthy": (self._session_id is not None and self._state == SessionState.ACTIVE),
            "metrics": self.get_metrics(),
        }

    def close(self) -> None:
        """Clean up session and HTTP resources."""
        logger.info("SessionRecovery closing")

        # Attempt graceful session cleanup if possible
        if self._session_id and self._state == SessionState.ACTIVE:
            try:
                self._http_session.post(
                    f"{self.base_url}/stop", json={"session_id": self._session_id}, timeout=5.0
                )
                logger.info("Session stopped gracefully")
            except Exception as e:
                logger.warning(f"Failed to stop session gracefully: {e}")

        # Close HTTP session
        self._http_session.close()

        # Reset session state
        self._state = SessionState.UNINITIALIZED
        self._session_id = None
        self._session_start_time = None

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"SessionRecovery(state={self._state.name}, "
            f"session_id={self._session_id}, "
            f"attempts={self._recovery_attempts})"
        )
