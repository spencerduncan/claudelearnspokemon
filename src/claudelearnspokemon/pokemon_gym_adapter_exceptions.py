"""
Exception hierarchy for PokemonGymAdapter interface.

This module defines all exception types used by the PokemonGymAdapter
to ensure consistent error handling and proper error classification.

All exceptions include performance considerations and provide actionable
error information for debugging and recovery.

Author: John Botmack - Error Handling and Performance Engineering
"""

import time
from typing import Any


class AdapterError(Exception):
    """
    Base exception for all PokemonGymAdapter errors.

    Provides structured error information with performance metadata
    and actionable error details for debugging.

    Performance Requirements:
    - Exception creation overhead: < 1ms
    - Error message formatting: < 5ms
    - Memory usage per exception: < 1KB
    """

    def __init__(
        self,
        message: str,
        error_type: str | None = None,
        details: dict[str, Any] | None = None,
        operation: str | None = None,
        retry_after: float | None = None,
        performance_context: dict[str, Any] | None = None,
    ):
        """
        Initialize adapter error with comprehensive context.

        Args:
            message: Human-readable error description
            error_type: Error classification for programmatic handling
            details: Additional error context and debugging information
            operation: Name of the operation that failed
            retry_after: Suggested retry delay in seconds
            performance_context: Performance metrics at time of error
        """
        super().__init__(message)

        self.message = message
        self.error_type = error_type or self.__class__.__name__
        self.details = details or {}
        self.operation = operation
        self.retry_after = retry_after
        self.timestamp = time.time()
        self.performance_context = performance_context or {}

        # Generate unique request ID for error tracking
        self.request_id = f"req_{int(self.timestamp * 1000000) % 1000000:06d}"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert exception to dictionary format for serialization.

        Returns:
            Dictionary with all error information for API responses
        """
        return {
            "error": True,
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "request_id": self.request_id,
            "retry_after": self.retry_after,
            "performance_context": self.performance_context,
        }

    def __str__(self) -> str:
        """Enhanced string representation with performance context."""
        base_msg = f"{self.error_type}: {self.message}"

        if self.operation:
            base_msg = f"{self.operation} - {base_msg}"

        if self.performance_context.get("duration_ms"):
            duration = self.performance_context["duration_ms"]
            base_msg = f"{base_msg} (took {duration:.2f}ms)"

        if self.request_id:
            base_msg = f"{base_msg} [req:{self.request_id}]"

        return base_msg


class NetworkError(AdapterError):
    """
    Network communication errors with benchflow-ai Pokemon-gym server.

    These errors indicate problems with HTTP communication, connection
    timeouts, or server connectivity issues.

    Recovery Strategy:
    - Automatic retry with exponential backoff
    - Check network connectivity
    - Verify server availability
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        endpoint: str | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        details.update(
            {
                "status_code": status_code,
                "response_body": response_body[:1000] if response_body else None,  # Limit size
                "endpoint": endpoint,
            }
        )

        super().__init__(
            message=message,
            error_type="NetworkError",
            details=details,
            retry_after=kwargs.get("retry_after", 5.0),  # Default 5s retry
            **kwargs,
        )

        self.status_code = status_code
        self.response_body = response_body
        self.endpoint = endpoint


class SessionError(AdapterError):
    """
    Session management errors with Pokemon-gym emulator.

    These errors occur when session initialization, maintenance, or
    cleanup fails. Often recoverable through session restart.

    Recovery Strategy:
    - Stop existing session
    - Initialize new session
    - Retry original operation
    """

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        session_state: str | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        details.update({"session_id": session_id, "session_state": session_state})

        super().__init__(
            message=message,
            error_type="SessionError",
            details=details,
            retry_after=kwargs.get("retry_after", 2.0),  # Default 2s retry
            **kwargs,
        )

        self.session_id = session_id
        self.session_state = session_state


class ValidationError(AdapterError):
    """
    Input validation errors for adapter operations.

    These errors indicate invalid parameters, malformed requests,
    or constraint violations. Not recoverable through retry.

    Recovery Strategy:
    - Fix input parameters
    - Check API documentation
    - No automatic retry
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        constraint: str | None = None,
        valid_options: list[str] | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        details.update(
            {
                "field": field,
                "invalid_value": str(value) if value is not None else None,
                "constraint": constraint,
                "valid_options": valid_options,
            }
        )

        super().__init__(
            message=message,
            error_type="ValidationError",
            details=details,
            retry_after=None,  # No retry for validation errors
            **kwargs,
        )

        self.field = field
        self.value = value
        self.constraint = constraint
        self.valid_options = valid_options


class TimeoutError(AdapterError):
    """
    Operation timeout errors for adapter requests.

    These errors occur when operations exceed configured timeout limits.
    Often recoverable through retry with longer timeout.

    Recovery Strategy:
    - Increase timeout configuration
    - Check system performance
    - Retry with exponential backoff
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        operation_duration: float | None = None,
        timeout_type: str | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        details.update(
            {
                "timeout_seconds": timeout_seconds,
                "operation_duration": operation_duration,
                "timeout_type": timeout_type,  # "connect", "read", "total"
            }
        )

        super().__init__(
            message=message,
            error_type="TimeoutError",
            details=details,
            retry_after=kwargs.get("retry_after", timeout_seconds * 2 if timeout_seconds else 10.0),
            **kwargs,
        )

        self.timeout_seconds = timeout_seconds
        self.operation_duration = operation_duration
        self.timeout_type = timeout_type


class ContainerError(AdapterError):
    """
    Docker container errors for Pokemon-gym emulator.

    These errors indicate problems with the underlying Docker container
    running the Pokemon emulator. May require container restart.

    Recovery Strategy:
    - Check container status
    - Restart container if needed
    - Verify container health
    """

    def __init__(
        self,
        message: str,
        container_id: str | None = None,
        container_status: str | None = None,
        exit_code: int | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        details.update(
            {
                "container_id": container_id,
                "container_status": container_status,
                "exit_code": exit_code,
            }
        )

        super().__init__(
            message=message,
            error_type="ContainerError",
            details=details,
            retry_after=kwargs.get("retry_after", 10.0),  # Default 10s retry
            **kwargs,
        )

        self.container_id = container_id
        self.container_status = container_status
        self.exit_code = exit_code


class StateError(AdapterError):
    """
    Game state consistency and integrity errors.

    These errors occur when game state data is corrupted, inconsistent,
    or fails validation checks.

    Recovery Strategy:
    - Reset game state
    - Verify state integrity
    - Clear state caches
    """

    def __init__(
        self,
        message: str,
        state_checksum: str | None = None,
        expected_checksum: str | None = None,
        corrupted_fields: list[str] | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        details.update(
            {
                "state_checksum": state_checksum,
                "expected_checksum": expected_checksum,
                "corrupted_fields": corrupted_fields,
            }
        )

        super().__init__(
            message=message,
            error_type="StateError",
            details=details,
            retry_after=kwargs.get("retry_after", 1.0),  # Default 1s retry
            **kwargs,
        )

        self.state_checksum = state_checksum
        self.expected_checksum = expected_checksum
        self.corrupted_fields = corrupted_fields or []


class PerformanceError(AdapterError):
    """
    Performance threshold violations and SLA breaches.

    These errors occur when operations exceed performance requirements
    or resource usage limits are violated.

    Recovery Strategy:
    - Reduce operation complexity
    - Check system resources
    - Optimize configuration
    """

    def __init__(
        self,
        message: str,
        actual_duration_ms: float | None = None,
        max_duration_ms: float | None = None,
        resource_usage: dict[str, float] | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        details.update(
            {
                "actual_duration_ms": actual_duration_ms,
                "max_duration_ms": max_duration_ms,
                "resource_usage": resource_usage,
            }
        )

        super().__init__(
            message=message,
            error_type="PerformanceError",
            details=details,
            retry_after=kwargs.get("retry_after", 1.0),  # Quick retry for perf issues
            **kwargs,
        )

        self.actual_duration_ms = actual_duration_ms
        self.max_duration_ms = max_duration_ms
        self.resource_usage = resource_usage or {}


class InitializationError(AdapterError):
    """
    Adapter initialization and configuration errors.

    These errors occur during adapter startup or when configuration
    parameters are invalid or incompatible.

    Recovery Strategy:
    - Fix configuration
    - Check dependencies
    - Verify system requirements
    """

    def __init__(
        self,
        message: str,
        config_field: str | None = None,
        config_value: Any | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        details.update(
            {
                "config_field": config_field,
                "config_value": str(config_value) if config_value is not None else None,
            }
        )

        super().__init__(
            message=message,
            error_type="InitializationError",
            details=details,
            retry_after=None,  # No retry for init errors
            **kwargs,
        )

        self.config_field = config_field
        self.config_value = config_value


# =============================================================================
# Error Classification Utilities
# =============================================================================


def classify_error(exception: Exception) -> str:
    """
    Classify generic exceptions into adapter error types.

    Args:
        exception: Exception to classify

    Returns:
        Error type string for consistent handling
    """
    if isinstance(exception, AdapterError):
        return exception.error_type

    exception_type = type(exception).__name__

    # Network-related exceptions
    if exception_type in [
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "HTTPError",
        "RequestException",
    ]:
        return "NetworkError"

    # Timeout-related exceptions
    if exception_type in ["TimeoutError", "socket.timeout"]:
        return "TimeoutError"

    # Validation-related exceptions
    if exception_type in ["ValueError", "TypeError", "KeyError"]:
        return "ValidationError"

    # Container-related exceptions
    if exception_type in ["DockerException", "ContainerError"]:
        return "ContainerError"

    # Default to generic adapter error
    return "AdapterError"


def is_retryable_error(error: AdapterError) -> bool:
    """
    Determine if an error is retryable.

    Args:
        error: AdapterError instance

    Returns:
        True if error should be retried, False otherwise
    """
    non_retryable_types = ["ValidationError", "InitializationError"]

    return error.error_type not in non_retryable_types and error.retry_after is not None


def calculate_retry_delay(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, backoff_factor: float = 2.0
) -> float:
    """
    Calculate retry delay with exponential backoff.

    Args:
        attempt: Retry attempt number (1, 2, 3, ...)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Exponential backoff multiplier

    Returns:
        Calculated delay in seconds
    """
    delay = base_delay * (backoff_factor ** (attempt - 1))
    return min(delay, max_delay)


# =============================================================================
# Error Formatting Utilities
# =============================================================================


def format_error_for_logging(error: AdapterError) -> str:
    """
    Format error for structured logging.

    Args:
        error: AdapterError instance

    Returns:
        Formatted log message with all relevant context
    """
    log_parts = [
        f"[{error.error_type}]",
        f"Operation: {error.operation or 'unknown'}",
        f"Message: {error.message}",
        f"RequestID: {error.request_id}",
    ]

    if error.performance_context.get("duration_ms"):
        duration = error.performance_context["duration_ms"]
        log_parts.append(f"Duration: {duration:.2f}ms")

    if error.retry_after:
        log_parts.append(f"RetryAfter: {error.retry_after}s")

    if error.details:
        detail_strs = [f"{k}={v}" for k, v in error.details.items() if v is not None]
        if detail_strs:
            log_parts.append(f"Details: {', '.join(detail_strs)}")

    return " | ".join(log_parts)


def create_error_response(error: AdapterError) -> dict[str, Any]:
    """
    Create standardized error response dictionary.

    Args:
        error: AdapterError instance

    Returns:
        Error response dictionary for API responses
    """
    return {
        "error": True,
        "error_type": error.error_type,
        "message": error.message,
        "details": error.details,
        "timestamp": error.timestamp,
        "request_id": error.request_id,
        "retry_after": error.retry_after,
        "operation": error.operation,
    }


# =============================================================================
# Performance-Aware Error Handling
# =============================================================================


class ErrorHandler:
    """
    Performance-optimized error handler for adapter operations.

    Provides consistent error handling with minimal performance overhead
    and proper error classification for recovery strategies.
    """

    def __init__(self):
        self.error_counts = {}
        self.last_error_time = {}

    def handle_exception(
        self,
        exception: Exception,
        operation: str,
        performance_context: dict[str, Any] | None = None,
    ) -> AdapterError:
        """
        Handle exception and convert to appropriate AdapterError.

        Args:
            exception: Original exception
            operation: Name of operation that failed
            performance_context: Performance metrics at time of error

        Returns:
            Properly formatted AdapterError
        """
        # Track error frequency for circuit breaker patterns
        error_type = classify_error(exception)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_error_time[error_type] = time.time()

        if isinstance(exception, AdapterError):
            # Already properly formatted
            if not exception.operation:
                exception.operation = operation
            return exception

        # Convert generic exception to AdapterError
        if error_type == "NetworkError":
            return NetworkError(
                message=str(exception), operation=operation, performance_context=performance_context
            )
        elif error_type == "TimeoutError":
            return TimeoutError(
                message=str(exception), operation=operation, performance_context=performance_context
            )
        elif error_type == "ValidationError":
            return ValidationError(
                message=str(exception), operation=operation, performance_context=performance_context
            )
        else:
            return AdapterError(
                message=str(exception),
                error_type=error_type,
                operation=operation,
                performance_context=performance_context,
            )

    def should_circuit_break(
        self, error_type: str, threshold: int = 10, time_window: float = 60.0
    ) -> bool:
        """
        Determine if circuit breaker should activate.

        Args:
            error_type: Type of error to check
            threshold: Error count threshold
            time_window: Time window in seconds

        Returns:
            True if circuit should break, False otherwise
        """
        if error_type not in self.error_counts:
            return False

        error_count = self.error_counts[error_type]
        last_error = self.last_error_time.get(error_type, 0)

        return bool(error_count >= threshold and (time.time() - last_error) < time_window)
