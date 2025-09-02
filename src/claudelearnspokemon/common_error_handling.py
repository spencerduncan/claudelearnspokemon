"""
Common error handling utilities and mixins for consistent exception management.

This module provides standardized error handling patterns to reduce code duplication
across the Pokemon speedrun learning system.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class ErrorSeverity(Enum):
    """Error severity levels for consistent handling."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ErrorContext:
    """Context information for error handling."""

    operation_name: str
    component_name: str
    start_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Get operation duration in milliseconds if start_time was set."""
        if self.start_time:
            return (time.time() - self.start_time) * 1000
        return None


class BaseExceptionHandler(ABC):
    """
    Base class for consistent exception handling across components.

    Provides standard patterns for logging, metrics, and fallback behavior.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._error_metrics = {
            "total_errors": 0,
            "errors_by_severity": {severity.name.lower(): 0 for severity in ErrorSeverity},
            "errors_by_operation": {},
        }
        self._error_metrics_lock = threading.Lock()

    def handle_exception(
        self,
        exception: Exception,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        fallback_value: T | None = None,
        reraise_as: type[Exception] | None = None,
        include_metrics: bool = True,
    ) -> T | None:
        """
        Handle exception with standardized logging, metrics, and fallback behavior.

        Args:
            exception: The caught exception
            context: Error context information
            severity: Error severity level
            fallback_value: Value to return instead of re-raising
            reraise_as: Exception type to re-raise as (chains original)
            include_metrics: Whether to record error metrics

        Returns:
            fallback_value if provided, otherwise re-raises

        Raises:
            reraise_as: If specified, re-raises as this exception type
            Exception: Original exception if no fallback or reraise specified
        """
        # Record metrics if requested
        if include_metrics:
            self._record_error_metric(context.operation_name, severity)

        # Build error message with context
        error_msg = self._build_error_message(exception, context)

        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error_msg)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(error_msg)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error_msg)
        else:
            self.logger.debug(error_msg)

        # Handle circuit breaker updates if applicable
        self._update_circuit_breaker_on_error(exception, context)

        # Re-raise as different exception type if specified
        if reraise_as:
            raise reraise_as(f"{context.operation_name} failed: {str(exception)}") from exception

        # Return fallback value if provided
        if fallback_value is not None:
            return fallback_value

        # Otherwise re-raise original exception
        raise exception

    def safe_execute(
        self,
        operation: Callable[[], T],
        context: ErrorContext,
        fallback_value: T | None = None,
        allowed_exceptions: tuple[type[Exception], ...] = (Exception,),
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        reraise_as: type[Exception] | None = None,
    ) -> T:
        """
        Safely execute an operation with standardized error handling.

        Args:
            operation: Function to execute
            context: Error context
            fallback_value: Value to return on error
            allowed_exceptions: Tuple of exception types to catch
            severity: Error severity level
            reraise_as: Exception type to re-raise as

        Returns:
            Operation result or fallback_value

        Raises:
            Exception: If not in allowed_exceptions or if no fallback provided
        """
        context.start_time = time.time()

        try:
            result = operation()

            # Log successful completion for high-severity operations
            if severity >= ErrorSeverity.HIGH:
                duration = context.duration_ms or 0
                self.logger.info(
                    f"{context.operation_name} completed successfully in {duration:.2f}ms"
                )

            return result

        except allowed_exceptions as e:
            return self.handle_exception(e, context, severity, fallback_value, reraise_as)

    def _build_error_message(self, exception: Exception, context: ErrorContext) -> str:
        """Build comprehensive error message with context."""
        parts = [f"{context.component_name}.{context.operation_name} failed:", str(exception)]

        if context.duration_ms:
            parts.append(f"(after {context.duration_ms:.2f}ms)")

        if context.metadata:
            metadata_str = ", ".join(f"{k}={v}" for k, v in context.metadata.items())
            parts.append(f"[{metadata_str}]")

        return " ".join(parts)

    def _record_error_metric(self, operation_name: str, severity: ErrorSeverity) -> None:
        """Record error metrics for monitoring."""
        with self._error_metrics_lock:
            self._error_metrics["total_errors"] += 1
            self._error_metrics["errors_by_severity"][severity.name.lower()] += 1

            if operation_name not in self._error_metrics["errors_by_operation"]:
                self._error_metrics["errors_by_operation"][operation_name] = 0
            self._error_metrics["errors_by_operation"][operation_name] += 1

    def _update_circuit_breaker_on_error(self, exception: Exception, context: ErrorContext) -> None:
        """Update circuit breaker state on error if applicable."""
        # Check if component has circuit breaker
        if hasattr(self, "circuit_breaker") and self.circuit_breaker:
            self.circuit_breaker.metrics.record_failure()

    def get_error_metrics(self) -> dict[str, Any]:
        """Get error metrics for monitoring."""
        with self._error_metrics_lock:
            return self._error_metrics.copy()

    @abstractmethod
    def create_fallback_response(self, operation_name: str, error: Exception) -> Any:
        """Create component-specific fallback response."""
        pass


class RetryableExceptionHandler(BaseExceptionHandler):
    """Exception handler with retry logic for transient failures."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def create_fallback_response(self, operation_name: str, error: Exception) -> Any:
        """Create generic fallback response for retryable operations."""
        return {
            "success": False,
            "error": str(error),
            "fallback": True,
            "operation": operation_name,
            "timestamp": time.time(),
        }

    def execute_with_retry(
        self,
        operation: Callable[[], T],
        context: ErrorContext,
        retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
        non_retryable_exceptions: tuple[type[Exception], ...] = (),
        fallback_value: T | None = None,
        reraise_as: type[Exception] | None = None,
    ) -> T:
        """
        Execute operation with retry logic for transient failures.

        Args:
            operation: Function to execute
            context: Error context
            retryable_exceptions: Exceptions that should trigger retry
            non_retryable_exceptions: Exceptions that should not be retried
            fallback_value: Value to return after all retries exhausted
            reraise_as: Exception type to re-raise as

        Returns:
            Operation result or fallback_value
        """
        context.start_time = time.time()
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return operation()

            except non_retryable_exceptions as e:
                # Don't retry these exceptions
                return self.handle_exception(
                    e, context, ErrorSeverity.HIGH, fallback_value, reraise_as
                )

            except retryable_exceptions as e:
                last_exception = e

                if attempt < self.max_retries:
                    self.logger.warning(
                        f"{context.operation_name} failed on attempt {attempt + 1}/{self.max_retries + 1}: {e}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    # All retries exhausted
                    context.metadata["attempts"] = attempt + 1
                    return self.handle_exception(
                        e, context, ErrorSeverity.HIGH, fallback_value, reraise_as
                    )

        # Should never reach here, but just in case
        if last_exception:
            raise last_exception

        return fallback_value  # type: ignore


class ComponentErrorHandler(BaseExceptionHandler):
    """
    Error handler mixin for specific component types.

    Provides component-specific error handling patterns and fallback strategies.
    """

    def __init__(self, component_name: str):
        super().__init__()
        self.component_name = component_name

    def create_context(
        self, operation_name: str, metadata: dict[str, Any] | None = None
    ) -> ErrorContext:
        """Create error context for this component."""
        return ErrorContext(
            operation_name=operation_name,
            component_name=self.component_name,
            metadata=metadata or {},
        )

    @abstractmethod
    def create_fallback_response(self, operation_name: str, error: Exception) -> Any:
        """Create component-specific fallback response."""
        pass


class StrategyErrorHandler(ComponentErrorHandler):
    """Error handler for strategy-related components."""

    def create_fallback_response(self, operation_name: str, error: Exception) -> dict[str, Any]:
        """Create fallback strategy response."""
        return {
            "strategy_id": f"fallback_{int(time.time())}",
            "experiments": [],
            "directives": ["Wait and observe"],
            "confidence": 0.1,
            "fallback_reason": str(error),
            "timestamp": time.time(),
        }


class EmulatorErrorHandler(ComponentErrorHandler):
    """Error handler for emulator-related components."""

    def create_fallback_response(self, operation_name: str, error: Exception) -> dict[str, Any]:
        """Create fallback emulator response."""
        return {
            "status": "error",
            "error_message": str(error),
            "fallback": True,
            "timestamp": time.time(),
        }


class CheckpointErrorHandler(ComponentErrorHandler):
    """Error handler for checkpoint-related components."""

    def create_fallback_response(self, operation_name: str, error: Exception) -> dict[str, Any]:
        """Create fallback checkpoint response."""
        return {
            "success": False,
            "error": str(error),
            "fallback_used": True,
            "timestamp": time.time(),
        }
