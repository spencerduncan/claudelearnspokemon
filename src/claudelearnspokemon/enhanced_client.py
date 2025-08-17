"""
EnhancedPokemonGymClient: Kernel-quality client with advanced error recovery.

Extends PokemonGymClient with production-grade reliability features:
- Automatic retry with exponential backoff
- Circuit breaker protection for persistent failures
- Session lifecycle management and recovery
- Comprehensive failure analysis and metrics

Author: Linus Torbot - Kernel Quality Standards Applied
"""

import logging
from collections.abc import Callable
from typing import Any

from .circuit_breaker import CircuitBreaker, CircuitConfig
from .emulator_pool import EmulatorPoolError, PokemonGymClient
from .retry_manager import RetryConfig, RetryManager, RetryOutcome
from .session_recovery import SessionConfig, SessionRecovery, SessionRecoveryError

# Configure logging for production observability
logger = logging.getLogger(__name__)


class EnhancedPokemonGymClient(PokemonGymClient):
    """
    Production-grade Pokemon-gym client with comprehensive error recovery.

    Extends PokemonGymClient with kernel engineering principles:
    - Never fail silently - all errors are logged and handled
    - Automatic recovery from transient failures
    - Circuit breaker protection from cascading failures
    - Session management with automatic recovery
    - Comprehensive operational metrics
    """

    def __init__(
        self,
        port: int,
        container_id: str,
        retry_config: RetryConfig | None = None,
        circuit_config: CircuitConfig | None = None,
        session_config: SessionConfig | None = None,
    ):
        """
        Initialize enhanced client with error recovery capabilities.

        Args:
            port: HTTP port for emulator communication
            container_id: Docker container ID for this emulator
            retry_config: Retry behavior configuration
            circuit_config: Circuit breaker configuration
            session_config: Session management configuration
        """
        super().__init__(port, container_id)

        # Initialize error recovery components
        self.retry_manager = RetryManager(retry_config or RetryConfig())
        self.circuit_breaker = CircuitBreaker(
            name=f"emulator-{port}", config=circuit_config or CircuitConfig()
        )
        self.session_recovery = SessionRecovery(
            base_url=self.base_url, config=session_config or SessionConfig()
        )

        # Enhanced client state
        self._is_initialized = False
        self._last_successful_operation: float | None = None

        logger.info(
            f"EnhancedPokemonGymClient initialized for port {port} " f"with advanced error recovery"
        )

    def send_input(self, input_sequence: str) -> dict[str, Any]:
        """
        Send input sequence with automatic retry and recovery.

        Args:
            input_sequence: Button inputs (A, B, START, etc.)

        Returns:
            Response data from emulator

        Raises:
            EmulatorPoolError: On unrecoverable failure
        """

        def _send_input_operation() -> dict[str, Any]:
            """Internal operation wrapper for retry logic."""
            # Ensure valid session before attempting operation
            self._ensure_session()

            # Use parent implementation for actual HTTP call
            return super(EnhancedPokemonGymClient, self).send_input(input_sequence)

        return self._execute_with_recovery(
            operation=_send_input_operation, operation_name=f"send_input({input_sequence[:20]}...)"
        )

    def get_state(self) -> dict[str, Any]:
        """
        Get current game state with automatic retry and recovery.

        Returns:
            Current game state data

        Raises:
            EmulatorPoolError: On unrecoverable failure
        """

        def _get_state_operation() -> dict[str, Any]:
            """Internal operation wrapper for retry logic."""
            # Ensure valid session before attempting operation
            self._ensure_session()

            # Use parent implementation for actual HTTP call
            return super(EnhancedPokemonGymClient, self).get_state()

        return self._execute_with_recovery(
            operation=_get_state_operation, operation_name="get_state"
        )

    def reset_game(self) -> dict[str, Any]:
        """
        Reset the game with automatic retry and recovery.

        Returns:
            Reset confirmation from emulator

        Raises:
            EmulatorPoolError: On unrecoverable failure
        """

        def _reset_game_operation() -> dict[str, Any]:
            """Internal operation wrapper for retry logic."""
            # Ensure valid session before attempting operation
            self._ensure_session()

            # Use parent implementation for actual HTTP call
            return super(EnhancedPokemonGymClient, self).reset_game()

        return self._execute_with_recovery(
            operation=_reset_game_operation, operation_name="reset_game"
        )

    def is_healthy(self) -> bool:
        """
        Enhanced health check with circuit breaker integration.

        Returns:
            True if emulator is healthy and circuit is closed
        """
        # Quick circuit breaker check first
        if not self.circuit_breaker.is_available():
            logger.debug(f"Health check failed: circuit breaker is OPEN for {self}")
            return False

        def _health_check_operation() -> bool:
            """Internal health check wrapper."""
            # Check session health
            if not self.session_recovery.get_session_id():
                # Try to ensure session for health check
                try:
                    self._ensure_session()
                except Exception:
                    return False

            # Use parent implementation for actual health check
            return super(EnhancedPokemonGymClient, self).is_healthy()

        try:
            # Execute health check through circuit breaker
            result = self.circuit_breaker.call(
                func=_health_check_operation, operation_name="health_check"
            )
            return result

        except Exception as e:
            logger.warning(f"Health check failed for {self}: {e}")
            return False

    def _execute_with_recovery(self, operation: Callable[[], Any], operation_name: str) -> Any:
        """
        Execute operation with full error recovery pipeline.

        Args:
            operation: Function to execute
            operation_name: Descriptive name for logging

        Returns:
            Operation result

        Raises:
            EmulatorPoolError: On unrecoverable failure
        """

        def _circuit_protected_operation() -> Any:
            """Wrap operation with circuit breaker protection."""
            return self.circuit_breaker.call(operation, operation_name)

        # Execute through retry manager with circuit breaker
        outcome, result, exception = self.retry_manager.execute_with_retry(
            func=_circuit_protected_operation, operation_name=operation_name
        )

        # Handle different outcomes
        if outcome == RetryOutcome.SUCCESS:
            self._record_successful_operation()
            return result

        elif outcome == RetryOutcome.CIRCUIT_BREAKER_OPEN:
            raise EmulatorPoolError(
                f"Circuit breaker is OPEN for {self}. "
                f"Service may be experiencing persistent failures."
            )

        elif outcome == RetryOutcome.NON_RETRIABLE_ERROR:
            # Check if it's a session error we can handle
            if exception and self.session_recovery.is_session_error(exception):
                logger.warning(f"Session error detected during {operation_name}: {exception}")
                self.session_recovery.handle_session_error(exception)
                raise EmulatorPoolError(
                    f"Session error in {operation_name}: {exception}. "
                    f"Session invalidated for recovery."
                ) from exception
            else:
                raise EmulatorPoolError(
                    f"Non-retriable error in {operation_name}: {exception}"
                ) from exception

        elif outcome == RetryOutcome.MAX_RETRIES_EXCEEDED:
            raise EmulatorPoolError(
                f"Max retries exceeded for {operation_name}: {exception}. "
                f"Service may be unavailable."
            ) from exception

        else:
            # Should not reach here, but safety net
            raise EmulatorPoolError(
                f"Unexpected retry outcome {outcome} for {operation_name}: {exception}"
            ) from exception

    def _ensure_session(self) -> str:
        """
        Ensure valid session exists for operations.

        Returns:
            Valid session ID

        Raises:
            SessionRecoveryError: If session cannot be established
        """
        try:
            return self.session_recovery.ensure_session()
        except SessionRecoveryError as e:
            logger.error(f"Failed to ensure session for {self}: {e}")
            raise e

    def _record_successful_operation(self) -> None:
        """Record successful operation for health tracking."""
        import time

        self._last_successful_operation = time.time()

    def force_session_refresh(self) -> str:
        """
        Force session refresh even if current session appears valid.

        Returns:
            New session ID

        Raises:
            SessionRecoveryError: If session cannot be refreshed
        """
        logger.info(f"Forcing session refresh for {self}")
        return self.session_recovery.ensure_session(force_refresh=True)

    def invalidate_session(self) -> None:
        """
        Invalidate current session, forcing reinitialization on next use.

        Useful when external components detect session issues.
        """
        logger.info(f"Session invalidated for {self}")
        self.session_recovery.invalidate_session()

    def get_recovery_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive error recovery metrics for monitoring.

        Returns:
            Dictionary with all recovery metrics
        """
        return {
            "retry_metrics": self.retry_manager.get_metrics(),
            "circuit_breaker": self.circuit_breaker.get_health_status(),
            "session_info": self.session_recovery.get_session_info(),
            "client_info": {
                "port": self.port,
                "container_id": self.container_id,
                "last_successful_operation": self._last_successful_operation,
                "is_initialized": self._is_initialized,
            },
        }

    def get_health_status(self) -> dict[str, Any]:
        """
        Get comprehensive health status including all recovery components.

        Returns:
            Dictionary with detailed health status
        """
        return {
            "overall_healthy": self.is_healthy(),
            "client_healthy": super().is_healthy(),
            "circuit_breaker": self.circuit_breaker.get_health_status(),
            "session_state": self.session_recovery.get_session_state().name,
            "session_id": self.session_recovery.get_session_id(),
            "recovery_metrics": self.get_recovery_metrics(),
        }

    def reset_recovery_state(self) -> None:
        """
        Reset all recovery components to initial state.

        Useful for testing or after resolving persistent issues.
        """
        logger.info(f"Resetting recovery state for {self}")

        self.retry_manager.reset_metrics()
        self.circuit_breaker.reset()
        self.session_recovery.invalidate_session()

        self._is_initialized = False
        self._last_successful_operation = None

    def close(self) -> None:
        """
        Enhanced cleanup including all recovery components.
        """
        logger.info(f"Closing enhanced client for {self}")

        # Clean up recovery components
        self.session_recovery.close()

        # Call parent cleanup
        super().close()

    def __str__(self) -> str:
        """Enhanced string representation for debugging."""
        session_id = self.session_recovery.get_session_id()
        session_state = self.session_recovery.get_session_state().name
        circuit_state = self.circuit_breaker.get_state().name

        return (
            f"EnhancedPokemonGymClient(port={self.port}, "
            f"container={self.container_id[:12] if len(self.container_id) > 12 else self.container_id}, "
            f"session={session_id}, "
            f"session_state={session_state}, "
            f"circuit={circuit_state})"
        )
