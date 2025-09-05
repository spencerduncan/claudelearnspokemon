"""
Common logging utilities for consistent logging across the Pokemon speedrun learning system.

This module provides standardized logging patterns to reduce code duplication
and ensure consistent log formatting, context, and metrics integration.
"""

import functools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union, TypeVar

T = TypeVar("T")


class LogLevel(Enum):
    """Log levels with consistent naming."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogContext:
    """Context information for structured logging."""

    component: str
    operation: str
    execution_id: str | None = None
    timing_info: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_timing(self, name: str, value: float) -> None:
        """Add timing information to context."""
        if self.timing_info is None:
            self.timing_info = {}
        self.timing_info[name] = value

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to context."""
        self.metadata[key] = value


class StructuredLogger:
    """
    Structured logger that provides consistent formatting and context management.

    Reduces logging duplication by providing standard patterns for:
    - Operation start/completion logging
    - Error logging with context
    - Performance metrics integration
    - Circuit breaker state logging
    """

    def __init__(self, name: str, default_context: LogContext | None = None):
        self.logger = logging.getLogger(name)
        self.component_name = name
        self.default_context = default_context or LogContext(name, "unknown")

        # Metrics for logging patterns
        self._log_metrics: Dict[str, Any] = {
            "total_logs": 0,
            "logs_by_level": {level.name.lower(): 0 for level in LogLevel},
            "operations_logged": set(),
        }
        self._metrics_lock = threading.Lock()

    def log_operation_start(
        self,
        operation: str,
        context: LogContext | None = None,
        level: LogLevel = LogLevel.INFO,
        **metadata,
    ) -> LogContext:
        """
        Log the start of an operation with context.

        Args:
            operation: Name of the operation
            context: Optional context (creates new one if None)
            level: Log level
            **metadata: Additional metadata to include

        Returns:
            LogContext for continued use in the operation
        """
        if context is None:
            context = LogContext(self.component_name, operation)

        # Add metadata
        for key, value in metadata.items():
            context.add_metadata(key, value)

        # Record start time
        context.add_timing("start_time", time.time())

        # Log operation start
        msg = f"Starting {operation}"
        if metadata:
            metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
            msg += f" [{metadata_str}]"

        self._log_with_context(level, msg, context)

        return context

    def log_operation_success(
        self,
        context: LogContext,
        level: LogLevel = LogLevel.INFO,
        result_summary: str | None = None,
        **metadata,
    ) -> None:
        """
        Log successful completion of an operation.

        Args:
            context: Context from operation start
            level: Log level
            result_summary: Optional summary of results
            **metadata: Additional metadata
        """
        # Calculate duration
        if context.timing_info is not None:
            start_time = context.timing_info.get("start_time")
            if start_time:
                duration_ms = (time.time() - start_time) * 1000
                context.add_timing("duration_ms", duration_ms)

        # Add metadata
        for key, value in metadata.items():
            context.add_metadata(key, value)

        # Build message
        msg = f"Completed {context.operation}"
        if context.timing_info is not None and context.timing_info.get("duration_ms"):
            msg += f" in {context.timing_info['duration_ms']:.2f}ms"

        if result_summary:
            msg += f": {result_summary}"

        self._log_with_context(level, msg, context)

    def log_operation_failure(
        self, context: LogContext, error: Exception, level: LogLevel = LogLevel.ERROR, **metadata
    ) -> None:
        """
        Log operation failure with error details.

        Args:
            context: Context from operation start
            error: Exception that caused the failure
            level: Log level
            **metadata: Additional metadata
        """
        # Calculate duration if available
        if context.timing_info is not None:
            start_time = context.timing_info.get("start_time")
            if start_time:
                duration_ms = (time.time() - start_time) * 1000
                context.add_timing("duration_ms", duration_ms)

        # Add error details to metadata
        context.add_metadata("error_type", type(error).__name__)
        context.add_metadata("error_message", str(error))

        for key, value in metadata.items():
            context.add_metadata(key, value)

        # Build message
        msg = f"Failed {context.operation}: {error}"
        if context.timing_info is not None and context.timing_info.get("duration_ms"):
            msg += f" (after {context.timing_info['duration_ms']:.2f}ms)"

        self._log_with_context(level, msg, context)

    def log_circuit_breaker_state(
        self, state: str, operation: str, level: LogLevel = LogLevel.WARNING, **metadata
    ) -> None:
        """
        Log circuit breaker state changes.

        Args:
            state: Circuit breaker state (open, closed, half_open)
            operation: Operation that triggered the state change
            level: Log level
            **metadata: Additional context
        """
        context = LogContext(self.component_name, "circuit_breaker")
        context.add_metadata("circuit_breaker_state", state)
        context.add_metadata("triggering_operation", operation)

        for key, value in metadata.items():
            context.add_metadata(key, value)

        msg = f"Circuit breaker {state} for {operation}"
        self._log_with_context(level, msg, context)

    def log_cache_event(
        self,
        event_type: str,
        cache_key: str,
        operation: str,
        level: LogLevel = LogLevel.DEBUG,
        **metadata,
    ) -> None:
        """
        Log cache-related events (hits, misses, invalidations).

        Args:
            event_type: Type of cache event (hit, miss, put, invalidate)
            cache_key: Cache key (truncated for logging)
            operation: Operation that used the cache
            level: Log level
            **metadata: Additional context
        """
        context = LogContext(self.component_name, "cache")
        context.add_metadata("cache_event", event_type)
        context.add_metadata(
            "cache_key", cache_key[:16] + "..." if len(cache_key) > 16 else cache_key
        )
        context.add_metadata("operation", operation)

        for key, value in metadata.items():
            context.add_metadata(key, value)

        msg = f"Cache {event_type} for {operation}: {cache_key[:16]}..."
        self._log_with_context(level, msg, context)

    def log_performance_metric(
        self,
        metric_name: str,
        value: int | float,
        operation: str,
        level: LogLevel = LogLevel.INFO,
        **metadata,
    ) -> None:
        """
        Log performance metrics.

        Args:
            metric_name: Name of the metric
            value: Metric value
            operation: Operation being measured
            level: Log level
            **metadata: Additional context
        """
        context = LogContext(self.component_name, "metrics")
        context.add_metadata("metric_name", metric_name)
        context.add_metadata("metric_value", value)
        context.add_metadata("operation", operation)

        for key, value in metadata.items():
            context.add_metadata(key, value)

        msg = f"Metric {metric_name}={value} for {operation}"
        self._log_with_context(level, msg, context)

    def log_retry_attempt(
        self,
        operation: str,
        attempt: int,
        max_attempts: int,
        error: Exception,
        level: LogLevel = LogLevel.WARNING,
        **metadata,
    ) -> None:
        """
        Log retry attempts for failed operations.

        Args:
            operation: Operation being retried
            attempt: Current attempt number
            max_attempts: Maximum retry attempts
            error: Error that caused the retry
            level: Log level
            **metadata: Additional context
        """
        context = LogContext(self.component_name, "retry")
        context.add_metadata("operation", operation)
        context.add_metadata("attempt", attempt)
        context.add_metadata("max_attempts", max_attempts)
        context.add_metadata("error", str(error))

        for key, value in metadata.items():
            context.add_metadata(key, value)

        msg = f"Retry {attempt}/{max_attempts} for {operation}: {error}"
        self._log_with_context(level, msg, context)

    def log_initialization(self, config: dict[str, Any], level: LogLevel = LogLevel.INFO) -> None:
        """
        Log component initialization with configuration.

        Args:
            config: Configuration parameters
            level: Log level
        """
        context = LogContext(self.component_name, "initialization")

        # Add key config items to metadata
        for key, value in config.items():
            context.add_metadata(f"config_{key}", value)

        config_summary = ", ".join(f"{k}={v}" for k, v in config.items())
        msg = f"{self.component_name} initialized with config: {config_summary}"

        self._log_with_context(level, msg, context)

    def _log_with_context(self, level: LogLevel, message: str, context: LogContext) -> None:
        """
        Internal method to log with structured context.

        Args:
            level: Log level
            message: Message to log
            context: Logging context
        """
        # Record metrics
        with self._metrics_lock:
            self._log_metrics["total_logs"] += 1
            self._log_metrics["logs_by_level"][level.name.lower()] += 1
            self._log_metrics["operations_logged"].add(context.operation)

        # Build structured message
        structured_msg = message

        if context.execution_id:
            structured_msg += f" [exec_id={context.execution_id}]"

        if context.metadata:
            metadata_items = []
            for key, value in context.metadata.items():
                # Truncate long values for readability
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                metadata_items.append(f"{key}={value}")

            if metadata_items:
                structured_msg += f" [{', '.join(metadata_items)}]"

        # Log using the underlying logger
        self.logger.log(level.value, structured_msg)

    def get_log_metrics(self) -> dict[str, Any]:
        """Get logging metrics for monitoring."""
        with self._metrics_lock:
            return {
                **self._log_metrics,
                "operations_logged": list(self._log_metrics["operations_logged"]),
            }


def logged_operation(
    operation_name: str | None = None,
    log_level: LogLevel = LogLevel.INFO,
    log_results: bool = False,
    log_errors: bool = True,
):
    """
    Decorator to automatically log operation start, completion, and errors.

    Args:
        operation_name: Override operation name (defaults to function name)
        log_level: Log level for success logging
        log_results: Whether to log return values
        log_errors: Whether to log exceptions
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            # Get logger from self if it has one
            logger = getattr(self, "logger", None)
            if not isinstance(logger, StructuredLogger):
                # Fallback to creating a basic structured logger
                logger = StructuredLogger(self.__class__.__name__)

            # Determine operation name
            op_name = operation_name or func.__name__

            # Start logging
            context = logger.log_operation_start(
                op_name,
                level=log_level,
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )

            try:
                result = func(self, *args, **kwargs)

                # Log success
                result_summary = None
                if log_results:
                    if hasattr(result, "__dict__"):
                        result_summary = f"{type(result).__name__} object"
                    else:
                        result_summary = str(result)[:100]

                logger.log_operation_success(context, log_level, result_summary)
                return result

            except Exception as e:
                if log_errors:
                    logger.log_operation_failure(context, e)
                raise

        return wrapper

    return decorator


class LoggerFactory:
    """Factory for creating consistent loggers across components."""

    _loggers: dict[str, StructuredLogger] = {}
    _lock = threading.Lock()

    @classmethod
    def get_logger(cls, name: str, context: LogContext | None = None) -> StructuredLogger:
        """Get or create a structured logger for a component."""
        with cls._lock:
            if name not in cls._loggers:
                cls._loggers[name] = StructuredLogger(name, context)
            return cls._loggers[name]

    @classmethod
    def get_all_metrics(cls) -> dict[str, dict[str, Any]]:
        """Get metrics from all created loggers."""
        with cls._lock:
            return {name: logger.get_log_metrics() for name, logger in cls._loggers.items()}


class ComponentLogger:
    """
    Mixin class for components that need structured logging.

    Provides easy access to structured logging with component context.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

    def log_start(self, operation: str, **metadata) -> LogContext:
        """Log operation start."""
        return self.logger.log_operation_start(operation, **metadata)

    def log_success(
        self, context: LogContext, result_summary: str | None = None, **metadata
    ) -> None:
        """Log operation success."""
        self.logger.log_operation_success(context, result_summary=result_summary, **metadata)

    def log_failure(self, context: LogContext, error: Exception, **metadata) -> None:
        """Log operation failure."""
        self.logger.log_operation_failure(context, error, **metadata)

    def log_circuit_breaker(self, state: str, operation: str, **metadata) -> None:
        """Log circuit breaker state."""
        self.logger.log_circuit_breaker_state(state, operation, **metadata)

    def log_cache(self, event_type: str, cache_key: str, operation: str, **metadata) -> None:
        """Log cache event."""
        self.logger.log_cache_event(event_type, cache_key, operation, **metadata)

    def log_metric(self, metric_name: str, value: int | float, operation: str, **metadata) -> None:
        """Log performance metric."""
        self.logger.log_performance_metric(metric_name, value, operation, **metadata)

    def log_retry(
        self, operation: str, attempt: int, max_attempts: int, error: Exception, **metadata
    ) -> None:
        """Log retry attempt."""
        self.logger.log_retry_attempt(operation, attempt, max_attempts, error, **metadata)

    def log_initialization(self, config: dict[str, Any], level: LogLevel = LogLevel.INFO) -> None:
        """Log component initialization with configuration."""
        self.logger.log_initialization(config, level)
