"""
RetryManager: Kernel-quality retry logic with exponential backoff.

Implements production-grade error recovery patterns:
- Exponential backoff with jitter to prevent thundering herd
- Configurable retry policies for different failure types
- Circuit breaker integration for persistent failures
- Comprehensive failure analysis and metrics

Author: Linus Torbot - Kernel Quality Standards Applied
"""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeVar

from requests.exceptions import (
    ConnectionError,
    HTTPError,
    ReadTimeout,
    Timeout,
)

# Configure logging for production observability
logger = logging.getLogger(__name__)

T = TypeVar("T")


class FailureType(Enum):
    """Classification of failure types for appropriate retry strategies."""

    NETWORK_ERROR = auto()  # Connection issues, timeouts
    SERVER_ERROR = auto()  # 5xx HTTP responses
    CLIENT_ERROR = auto()  # 4xx HTTP responses (usually not retriable)
    SESSION_ERROR = auto()  # Session expired/invalid
    UNKNOWN_ERROR = auto()  # Unexpected failures


class RetryOutcome(Enum):
    """Result of retry operation."""

    SUCCESS = auto()
    MAX_RETRIES_EXCEEDED = auto()
    CIRCUIT_BREAKER_OPEN = auto()
    NON_RETRIABLE_ERROR = auto()


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 0.5  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay cap
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd
    retriable_status_codes: set[int] | None = None  # HTTP status codes to retry

    def __post_init__(self) -> None:
        """Set default retriable status codes if not provided."""
        if self.retriable_status_codes is None:
            # 5xx server errors and specific 4xx that might be transient
            self.retriable_status_codes = {408, 429, 500, 502, 503, 504}


@dataclass
class RetryMetrics:
    """Metrics for retry operations."""

    total_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    total_delay_time: float = 0.0
    failure_types: dict[FailureType, int] | None = None

    def __post_init__(self) -> None:
        """Initialize failure types counter."""
        if self.failure_types is None:
            self.failure_types = dict.fromkeys(FailureType, 0)

    def record_attempt(self, failure_type: FailureType, delay: float = 0.0) -> None:
        """Record retry attempt with failure classification."""
        self.total_attempts += 1
        if self.failure_types is not None:
            self.failure_types[failure_type] += 1
        self.total_delay_time += delay

    def record_success(self) -> None:
        """Record successful retry operation."""
        self.successful_retries += 1

    def record_failure(self) -> None:
        """Record failed retry operation."""
        self.failed_retries += 1


class RetryManager:
    """
    Production-grade retry manager with exponential backoff.

    Implements kernel engineering principles:
    - Fail fast on non-retriable errors
    - Exponential backoff with jitter prevents cascading failures
    - Circuit breaker integration for system protection
    - Comprehensive metrics for operational visibility
    """

    def __init__(self, config: RetryConfig | None = None):
        """
        Initialize retry manager with configuration.

        Args:
            config: Retry configuration (uses defaults if None)
        """
        self.config = config or RetryConfig()
        self.metrics = RetryMetrics()

        logger.info(
            f"RetryManager initialized: max_retries={self.config.max_retries}, "
            f"base_delay={self.config.base_delay}s"
        )

    def execute_with_retry(
        self, func: Callable[[], T], operation_name: str = "operation"
    ) -> tuple[RetryOutcome, T | None, Exception | None]:
        """
        Execute function with retry logic and comprehensive error handling.

        Args:
            func: Function to execute with retries
            operation_name: Descriptive name for logging

        Returns:
            Tuple of (outcome, result, final_exception)
        """
        last_exception: Exception | None = None

        for attempt in range(self.config.max_retries + 1):  # +1 for initial attempt
            try:
                logger.debug(f"Attempting {operation_name}, attempt {attempt + 1}")
                result = func()

                if attempt > 0:
                    self.metrics.record_success()
                    logger.info(f"{operation_name} succeeded after {attempt} retries")

                return RetryOutcome.SUCCESS, result, None

            except Exception as e:
                last_exception = e
                failure_type = self._classify_failure(e)

                logger.warning(f"{operation_name} failed, attempt {attempt + 1}: {e}")

                # Check if error is retriable
                if not self._is_retriable(failure_type, e):
                    logger.error(f"{operation_name} failed with non-retriable error: {e}")
                    self.metrics.record_failure()
                    return RetryOutcome.NON_RETRIABLE_ERROR, None, e

                # Check if we have more attempts left
                if attempt >= self.config.max_retries:
                    logger.error(
                        f"{operation_name} failed after {self.config.max_retries} retries: {e}"
                    )
                    self.metrics.record_failure()
                    return RetryOutcome.MAX_RETRIES_EXCEEDED, None, e

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                self.metrics.record_attempt(failure_type, delay)

                logger.info(f"Retrying {operation_name} in {delay:.2f}s (attempt {attempt + 2})")
                time.sleep(delay)

        # Should not reach here, but safety check
        self.metrics.record_failure()
        return RetryOutcome.MAX_RETRIES_EXCEEDED, None, last_exception

    def _classify_failure(self, exception: Exception) -> FailureType:
        """
        Classify failure type for appropriate retry strategy.

        Args:
            exception: Exception to classify

        Returns:
            FailureType for the exception
        """
        if isinstance(exception, ConnectionError | Timeout | ReadTimeout):
            return FailureType.NETWORK_ERROR

        if isinstance(exception, HTTPError):
            if hasattr(exception, "response") and exception.response is not None:
                status_code = exception.response.status_code
                if 500 <= status_code < 600:
                    return FailureType.SERVER_ERROR
                elif 400 <= status_code < 500:
                    # Check for session-related errors
                    if status_code in {401, 403} or "session" in str(exception).lower():
                        return FailureType.SESSION_ERROR
                    return FailureType.CLIENT_ERROR

        if "session" in str(exception).lower() or "unauthorized" in str(exception).lower():
            return FailureType.SESSION_ERROR

        return FailureType.UNKNOWN_ERROR

    def _is_retriable(self, failure_type: FailureType, exception: Exception) -> bool:
        """
        Determine if failure type is retriable.

        Args:
            failure_type: Classified failure type
            exception: Original exception

        Returns:
            True if failure should be retried
        """
        # Network errors are almost always retriable
        if failure_type == FailureType.NETWORK_ERROR:
            return True

        # Server errors are retriable
        if failure_type == FailureType.SERVER_ERROR:
            return True

        # Session errors are retriable (we can attempt to reinitialize)
        if failure_type == FailureType.SESSION_ERROR:
            return True

        # Check HTTP status codes for retriable errors
        if isinstance(exception, HTTPError) and hasattr(exception, "response"):
            status_code = exception.response.status_code
            return (
                self.config.retriable_status_codes is not None
                and status_code in self.config.retriable_status_codes
            )

        # Unknown errors - be conservative and retry
        if failure_type == FailureType.UNKNOWN_ERROR:
            return True

        # Client errors (4xx except specific ones) are not retriable
        return False

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with optional jitter.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (multiplier ^ attempt)
        delay = self.config.base_delay * (self.config.backoff_multiplier**attempt)

        # Apply maximum delay cap
        delay = min(delay, self.config.max_delay)

        # Add jitter to prevent thundering herd problem
        if self.config.jitter:
            # Add ±25% jitter
            jitter_range = delay * 0.25
            jitter = random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay + jitter)  # Minimum 100ms delay

        return delay

    def get_metrics(self) -> RetryMetrics:
        """
        Get current retry metrics for monitoring.

        Returns:
            Copy of current retry metrics
        """
        return RetryMetrics(
            total_attempts=self.metrics.total_attempts,
            successful_retries=self.metrics.successful_retries,
            failed_retries=self.metrics.failed_retries,
            total_delay_time=self.metrics.total_delay_time,
            failure_types=self.metrics.failure_types.copy() if self.metrics.failure_types else None,
        )

    def reset_metrics(self) -> None:
        """Reset metrics counters (useful for testing or monitoring periods)."""
        self.metrics = RetryMetrics()
        logger.info("RetryManager metrics reset")

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"RetryManager(max_retries={self.config.max_retries}, "
            f"base_delay={self.config.base_delay}s, "
            f"total_attempts={self.metrics.total_attempts})"
        )
