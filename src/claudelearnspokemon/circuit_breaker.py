"""
CircuitBreaker: Kernel-quality circuit breaker pattern implementation.

Implements production-grade fault tolerance patterns:
- Three-state circuit breaker (CLOSED, OPEN, HALF_OPEN)
- Failure threshold detection with sliding window
- Automatic recovery attempts with configurable timeout
- Health monitoring and operational visibility

Author: Linus Torbot - Kernel Quality Standards Applied
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, TypeVar

# Configure logging for production observability
logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation - requests allowed
    OPEN = auto()  # Circuit tripped - requests rejected immediately
    HALF_OPEN = auto()  # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Custom exception for circuit breaker operations."""

    pass


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Consecutive failures to trip circuit
    recovery_timeout: float = 60.0  # Seconds to wait before attempting recovery
    success_threshold: int = 2  # Successes needed to close circuit from HALF_OPEN
    sliding_window_size: int = 10  # Size of sliding window for failure tracking
    expected_exception_types: tuple[Any, ...] | None = (
        None  # Exception types that count as failures
    )

    def __post_init__(self) -> None:
        """Set default expected exception types if not provided."""
        if self.expected_exception_types is None:
            # Default exceptions that should trip the circuit
            from requests.exceptions import RequestException

            self.expected_exception_types = (RequestException, ConnectionError, TimeoutError)


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker operations."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_trips: int = 0
    rejected_requests: int = 0
    recovery_attempts: int = 0
    current_consecutive_failures: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None

    def record_success(self) -> None:
        """Record successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.current_consecutive_failures = 0
        self.last_success_time = time.time()

    def record_failure(self) -> None:
        """Record failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.current_consecutive_failures += 1
        self.last_failure_time = time.time()

    def record_circuit_trip(self) -> None:
        """Record circuit breaker trip."""
        self.circuit_trips += 1

    def record_rejected_request(self) -> None:
        """Record rejected request due to open circuit."""
        self.rejected_requests += 1

    def record_recovery_attempt(self) -> None:
        """Record recovery attempt."""
        self.recovery_attempts += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0


class CircuitBreaker:
    """
    Production-grade circuit breaker implementation.

    Implements kernel engineering principles:
    - Fail fast when service is unavailable
    - Automatic recovery detection
    - Thread-safe operation with minimal locking
    - Comprehensive metrics for operational visibility
    - Configurable behavior for different services
    """

    def __init__(self, name: str = "default", config: CircuitConfig | None = None):
        """
        Initialize circuit breaker with configuration.

        Args:
            name: Descriptive name for logging and monitoring
            config: Circuit breaker configuration (uses defaults if None)
        """
        self.name = name
        self.config = config or CircuitConfig()
        self.metrics = CircuitMetrics()

        # Circuit breaker state
        self._state = CircuitState.CLOSED
        self._last_trip_time: float | None = None

        # Thread safety - minimize lock contention
        self._lock = threading.Lock()  # Simple lock sufficient - no reentrancy needed

        # Sliding window for failure tracking (simple implementation)
        self._recent_results: list[bool] = []  # True = success, False = failure

        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"recovery_timeout={self.config.recovery_timeout}s"
        )

    def call(self, func: Callable[[], T], operation_name: str = "operation") -> T:
        """
        Execute function through circuit breaker protection.

        Args:
            func: Function to execute
            operation_name: Descriptive name for logging

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Original exception: If function fails
        """
        # Check if circuit should allow request
        if not self._allow_request():
            self.metrics.record_rejected_request()
            raise CircuitBreakerError(
                f"CircuitBreaker '{self.name}' is OPEN. "
                f"Service calls rejected for {self.config.recovery_timeout}s"
            )

        try:
            # Execute the function
            result = func()
            self._record_success()
            logger.debug(f"CircuitBreaker '{self.name}' - {operation_name} succeeded")
            return result

        except Exception as e:
            # Only record failure if it's an expected exception type
            if self.config.expected_exception_types and isinstance(
                e, self.config.expected_exception_types
            ):
                self._record_failure()
                logger.warning(f"CircuitBreaker '{self.name}' - {operation_name} failed: {e}")
            else:
                # Unexpected exception - pass through without affecting circuit state
                logger.debug(f"CircuitBreaker '{self.name}' - unexpected exception: {e}")

            raise e

    def _allow_request(self) -> bool:
        """
        Determine if request should be allowed based on circuit state.

        Returns:
            True if request should be allowed
        """
        with self._lock:
            current_time = time.time()

            if self._state == CircuitState.CLOSED:
                return True

            elif self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if (
                    self._last_trip_time
                    and current_time - self._last_trip_time >= self.config.recovery_timeout
                ):
                    logger.info(f"CircuitBreaker '{self.name}' attempting recovery (HALF_OPEN)")
                    self._state = CircuitState.HALF_OPEN
                    self.metrics.record_recovery_attempt()
                    return True
                return False

            elif self._state == CircuitState.HALF_OPEN:
                # Allow limited requests to test if service has recovered
                return True

            return False

    def _record_success(self) -> None:
        """Record successful operation and update circuit state."""
        with self._lock:
            self.metrics.record_success()
            self._recent_results.append(True)
            self._trim_sliding_window()

            if self._state == CircuitState.HALF_OPEN:
                # Check if we have enough successes to close the circuit
                recent_successes = sum(
                    1 for result in self._recent_results[-self.config.success_threshold :] if result
                )

                if recent_successes >= self.config.success_threshold:
                    logger.info(f"CircuitBreaker '{self.name}' closing circuit after recovery")
                    self._state = CircuitState.CLOSED

    def _record_failure(self) -> None:
        """Record failed operation and update circuit state."""
        with self._lock:
            self.metrics.record_failure()
            self._recent_results.append(False)
            self._trim_sliding_window()

            # Check if we should trip the circuit
            if self._should_trip_circuit():
                self._trip_circuit()

    def _should_trip_circuit(self) -> bool:
        """
        Determine if circuit should be tripped based on failure patterns.

        Returns:
            True if circuit should be tripped
        """
        # Only trip if we're currently in CLOSED or HALF_OPEN state
        if self._state not in {CircuitState.CLOSED, CircuitState.HALF_OPEN}:
            return False

        # In HALF_OPEN state, trip immediately on any failure
        if self._state == CircuitState.HALF_OPEN:
            return True

        # In CLOSED state, check consecutive failures
        if self.metrics.current_consecutive_failures >= self.config.failure_threshold:
            return True

        # Check sliding window failure rate (additional heuristic)
        if len(self._recent_results) >= self.config.sliding_window_size:
            recent_failures = sum(1 for result in self._recent_results if not result)
            failure_rate = recent_failures / len(self._recent_results)

            # Trip if failure rate is above 80% and we have minimum failures
            if failure_rate > 0.8 and recent_failures >= self.config.failure_threshold:
                return True

        return False

    def _trip_circuit(self) -> None:
        """Trip the circuit breaker to OPEN state."""
        logger.warning(
            f"CircuitBreaker '{self.name}' tripping circuit! "
            f"Consecutive failures: {self.metrics.current_consecutive_failures}"
        )

        self._state = CircuitState.OPEN
        self._last_trip_time = time.time()
        self.metrics.record_circuit_trip()

    def _trim_sliding_window(self) -> None:
        """Trim sliding window to configured size."""
        if len(self._recent_results) > self.config.sliding_window_size:
            self._recent_results = self._recent_results[-self.config.sliding_window_size :]

    def get_state(self) -> CircuitState:
        """Get current circuit breaker state (thread-safe)."""
        with self._lock:
            return self._state

    def get_metrics(self) -> CircuitMetrics:
        """
        Get current circuit breaker metrics for monitoring.

        Returns:
            Copy of current metrics
        """
        with self._lock:
            return CircuitMetrics(
                total_requests=self.metrics.total_requests,
                successful_requests=self.metrics.successful_requests,
                failed_requests=self.metrics.failed_requests,
                circuit_trips=self.metrics.circuit_trips,
                rejected_requests=self.metrics.rejected_requests,
                recovery_attempts=self.metrics.recovery_attempts,
                current_consecutive_failures=self.metrics.current_consecutive_failures,
                last_failure_time=self.metrics.last_failure_time,
                last_success_time=self.metrics.last_success_time,
            )

    def reset(self) -> None:
        """Reset circuit breaker to CLOSED state and clear metrics."""
        with self._lock:
            logger.info(f"CircuitBreaker '{self.name}' manually reset")
            self._state = CircuitState.CLOSED
            self._last_trip_time = None
            self._recent_results.clear()
            self.metrics = CircuitMetrics()

    def force_open(self) -> None:
        """Force circuit breaker to OPEN state (for testing/emergency)."""
        with self._lock:
            logger.warning(f"CircuitBreaker '{self.name}' manually forced OPEN")
            self._state = CircuitState.OPEN
            self._last_trip_time = time.time()
            self.metrics.record_circuit_trip()

    def is_available(self) -> bool:
        """
        Check if circuit breaker would allow a request without making one.

        Returns:
            True if requests would be allowed
        """
        return self._allow_request()

    def get_health_status(self) -> dict[str, Any]:
        """
        Get comprehensive health status for monitoring.

        Returns:
            Dictionary with health status information
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.name,
                "is_available": self.is_available(),
                "success_rate": self.metrics.success_rate,
                "consecutive_failures": self.metrics.current_consecutive_failures,
                "total_requests": self.metrics.total_requests,
                "circuit_trips": self.metrics.circuit_trips,
                "rejected_requests": self.metrics.rejected_requests,
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time,
                "time_since_last_trip": (
                    time.time() - self._last_trip_time if self._last_trip_time else None
                ),
            }

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"CircuitBreaker(name='{self.name}', state={self._state.name}, "
            f"failures={self.metrics.current_consecutive_failures}/"
            f"{self.config.failure_threshold})"
        )
