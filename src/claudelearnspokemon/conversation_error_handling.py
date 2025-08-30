"""
Error handling and retry logic for Claude CLI conversation initialization.

This module provides production-ready error handling with exponential backoff,
circuit breaker patterns, and graceful degradation following Clean Code principles
and John Botmack's performance-first optimization approach.
"""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConversationErrorType(Enum):
    """Types of conversation errors for categorized handling."""

    SUBPROCESS_FAILED = "subprocess_failed"
    COMMUNICATION_TIMEOUT = "communication_timeout"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SYSTEM_PROMPT_FAILED = "system_prompt_failed"
    TURN_LIMIT_EXCEEDED = "turn_limit_exceeded"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RetryConfig:
    """Configuration for retry behavior with performance optimization."""

    max_attempts: int = 3
    initial_delay_ms: float = 100.0  # Start with 100ms like frame time
    max_delay_ms: float = 5000.0  # Cap at 5s like system startup
    backoff_multiplier: float = 2.0  # Exponential backoff
    jitter_factor: float = 0.1  # Add randomization to prevent thundering herd

    # Circuit breaker settings
    failure_threshold: int = 5  # Trip after 5 failures
    recovery_timeout_ms: float = 30000.0  # 30s circuit breaker timeout


class ConversationError(Exception):
    """Base exception for conversation-related errors."""

    def __init__(
        self,
        message: str,
        error_type: ConversationErrorType,
        process_id: int | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.process_id = process_id
        self.original_error = original_error
        self.timestamp = time.time()


class ConversationInitializationError(ConversationError):
    """Specific error for conversation initialization failures."""

    pass


class ConversationCommunicationError(ConversationError):
    """Specific error for communication failures during conversation."""

    pass


class CircuitBreakerState(Enum):
    """Circuit breaker states for conversation error handling."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, not attempting operations
    HALF_OPEN = "half_open"  # Testing if service has recovered


class ConversationCircuitBreaker:
    """
    Circuit breaker for conversation operations with performance monitoring.

    Implements the Circuit Breaker pattern to prevent cascading failures
    and reduce latency during system degradation.
    """

    def __init__(self, config: RetryConfig):
        """
        Initialize circuit breaker with configuration.

        Args:
            config: Retry configuration including circuit breaker settings
        """
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.next_attempt_time = 0.0

        logger.debug(
            f"CircuitBreaker initialized with {config.failure_threshold} failure threshold"
        )

    def can_execute(self) -> bool:
        """
        Check if operation can be executed based on circuit breaker state.

        Returns:
            True if operation should be attempted, False to fail fast
        """
        current_time = time.time() * 1000  # Convert to milliseconds

        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if current_time >= self.next_attempt_time:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN for testing")
                return True
            else:
                return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True

        return False

    def record_success(self):
        """Record successful operation and reset circuit breaker if needed."""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("Circuit breaker closed after successful operation")

    def record_failure(self):
        """Record failed operation and update circuit breaker state."""
        self.failure_count += 1
        self.last_failure_time = time.time() * 1000

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = self.last_failure_time + self.config.recovery_timeout_ms
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Next attempt in {self.config.recovery_timeout_ms}ms"
            )


class ConversationRetryHandler:
    """
    High-performance retry handler for conversation operations.

    Implements exponential backoff with jitter, circuit breaker pattern,
    and error categorization for optimal performance under failure conditions.
    """

    def __init__(self, config: RetryConfig | None = None):
        """
        Initialize retry handler with performance-optimized configuration.

        Args:
            config: Optional retry configuration, uses defaults if None
        """
        self.config = config or RetryConfig()
        self.circuit_breaker = ConversationCircuitBreaker(self.config)

        # Performance metrics
        self.total_attempts = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_retry_time_ms = 0.0

        logger.info(f"ConversationRetryHandler initialized with config: {self.config}")

    def execute_with_retry(
        self,
        operation: Callable[[], T],
        error_context: str = "conversation_operation",
        process_id: int | None = None,
    ) -> T:
        """
        Execute operation with retry logic and performance monitoring.

        Args:
            operation: Function to execute with retry logic
            error_context: Context description for error reporting
            process_id: Optional process ID for error tracking

        Returns:
            Result of successful operation execution

        Raises:
            ConversationError: If all retry attempts fail
        """
        if not self.circuit_breaker.can_execute():
            raise ConversationError(
                f"Circuit breaker open for {error_context}",
                ConversationErrorType.UNKNOWN_ERROR,
                process_id,
            )

        start_time = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self.config.max_attempts):
            self.total_attempts += 1

            try:
                # Execute the operation
                result = operation()

                # Record success metrics
                execution_time = (time.perf_counter() - start_time) * 1000
                self.total_successes += 1
                self.circuit_breaker.record_success()

                if attempt > 0:
                    logger.info(
                        f"Operation '{error_context}' succeeded on attempt {attempt + 1} "
                        f"after {execution_time:.1f}ms"
                    )

                return result

            except Exception as e:
                last_error = e
                attempt_time = (time.perf_counter() - start_time) * 1000

                # Categorize error type
                error_type = self._categorize_error(e)

                logger.warning(
                    f"Operation '{error_context}' failed on attempt {attempt + 1}/{self.config.max_attempts} "
                    f"after {attempt_time:.1f}ms: {str(e)}"
                )

                # Don't retry on certain error types
                if not self._should_retry(error_type):
                    logger.info(f"Not retrying {error_type.value} error")
                    break

                # Apply exponential backoff before next attempt
                if attempt < self.config.max_attempts - 1:
                    delay_ms = self._calculate_retry_delay(attempt)
                    logger.debug(f"Retrying in {delay_ms:.1f}ms")
                    time.sleep(delay_ms / 1000.0)
                    self.total_retry_time_ms += delay_ms

        # All attempts failed
        self.total_failures += 1
        self.circuit_breaker.record_failure()

        # Determine final error type and raise appropriate exception
        if last_error:
            error_type = self._categorize_error(last_error)
            final_error = ConversationError(
                f"Operation '{error_context}' failed after {self.config.max_attempts} attempts. "
                f"Last error: {str(last_error)}",
                error_type,
                process_id,
                last_error,
            )
        else:
            final_error = ConversationError(
                f"Operation '{error_context}' failed after {self.config.max_attempts} attempts",
                ConversationErrorType.UNKNOWN_ERROR,
                process_id,
            )

        raise final_error

    def _categorize_error(self, error: Exception) -> ConversationErrorType:
        """
        Categorize error for appropriate handling strategy.

        Args:
            error: Exception to categorize

        Returns:
            Categorized error type for handling logic
        """
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()

        # Subprocess and process errors
        if "subprocess" in error_str or "popen" in error_str or "oserror" in error_type_name:
            return ConversationErrorType.SUBPROCESS_FAILED

        # Timeout errors
        if "timeout" in error_str or "timeoutexpired" in error_type_name:
            return ConversationErrorType.COMMUNICATION_TIMEOUT

        # Authentication errors
        if "auth" in error_str or "permission" in error_str or "unauthorized" in error_str:
            return ConversationErrorType.AUTHENTICATION_ERROR

        # Rate limiting
        if "rate" in error_str and "limit" in error_str:
            return ConversationErrorType.RATE_LIMIT_EXCEEDED

        # Network errors
        if "network" in error_str or "connection" in error_str or "socket" in error_str:
            return ConversationErrorType.NETWORK_ERROR

        return ConversationErrorType.UNKNOWN_ERROR

    def _should_retry(self, error_type: ConversationErrorType) -> bool:
        """
        Determine if error type should be retried.

        Args:
            error_type: Categorized error type

        Returns:
            True if retry should be attempted, False for non-retryable errors
        """
        # Don't retry these error types
        non_retryable = {
            ConversationErrorType.AUTHENTICATION_ERROR,
            ConversationErrorType.TURN_LIMIT_EXCEEDED,
        }

        return error_type not in non_retryable

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in milliseconds before next retry
        """
        # Exponential backoff: delay = initial * (multiplier ^ attempt)
        base_delay = self.config.initial_delay_ms * (self.config.backoff_multiplier**attempt)

        # Apply maximum delay cap
        base_delay = min(base_delay, self.config.max_delay_ms)

        # Add jitter to prevent thundering herd (±jitter_factor%)
        jitter_range = base_delay * self.config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)

        final_delay = max(base_delay + jitter, self.config.initial_delay_ms / 2)
        return final_delay

    def get_performance_metrics(self) -> dict:
        """
        Get performance metrics for monitoring and optimization.

        Returns:
            Dictionary with retry handler performance metrics
        """
        success_rate = self.total_successes / max(1, self.total_attempts)
        avg_retry_time = self.total_retry_time_ms / max(1, self.total_failures)

        return {
            "total_attempts": self.total_attempts,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": success_rate,
            "total_retry_time_ms": self.total_retry_time_ms,
            "average_retry_time_ms": avg_retry_time,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "circuit_breaker_failure_count": self.circuit_breaker.failure_count,
        }

    def reset_metrics(self):
        """Reset performance metrics for fresh monitoring period."""
        self.total_attempts = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_retry_time_ms = 0.0

        logger.debug("ConversationRetryHandler metrics reset")


# Convenience functions for common retry scenarios
def retry_conversation_initialization(
    initialization_func: Callable[[], T],
    process_id: int | None = None,
    config: RetryConfig | None = None,
) -> T:
    """
    Retry conversation initialization with specialized error handling.

    Args:
        initialization_func: Function that initializes conversation
        process_id: Process ID for error context
        config: Optional retry configuration

    Returns:
        Result of successful initialization

    Raises:
        ConversationInitializationError: If initialization fails after retries
    """
    retry_handler = ConversationRetryHandler(config)

    try:
        return retry_handler.execute_with_retry(
            initialization_func, f"conversation_initialization_process_{process_id}", process_id
        )
    except ConversationError as e:
        raise ConversationInitializationError(
            f"Failed to initialize conversation for process {process_id}: {e}",
            e.error_type,
            process_id,
            e.original_error,
        ) from e


def retry_conversation_communication(
    communication_func: Callable[[], T],
    process_id: int | None = None,
    config: RetryConfig | None = None,
) -> T:
    """
    Retry conversation communication with specialized error handling.

    Args:
        communication_func: Function that performs communication
        process_id: Process ID for error context
        config: Optional retry configuration

    Returns:
        Result of successful communication

    Raises:
        ConversationCommunicationError: If communication fails after retries
    """
    retry_handler = ConversationRetryHandler(config)

    try:
        return retry_handler.execute_with_retry(
            communication_func, f"conversation_communication_process_{process_id}", process_id
        )
    except ConversationError as e:
        raise ConversationCommunicationError(
            f"Failed to communicate with process {process_id}: {e}",
            e.error_type,
            process_id,
            e.original_error,
        ) from e
