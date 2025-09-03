"""
Common circuit breaker utilities and patterns for consistent fault tolerance.

This module provides standardized circuit breaker integration patterns to reduce
code duplication and ensure consistent fault tolerance across components.
"""

import functools
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, TypeVar

from .circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitConfig

T = TypeVar("T")
logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Strategies for handling circuit breaker open states."""

    RAISE_EXCEPTION = "raise"
    RETURN_FALLBACK = "fallback"
    RETURN_NONE = "none"
    CALL_FALLBACK_FUNC = "callback"


@dataclass
class CircuitBreakerConfig:
    """Enhanced circuit breaker configuration with component-specific defaults."""

    name: str | None = None
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    sliding_window_size: int = 10
    expected_exceptions: tuple[type[Exception], ...] = ()
    fallback_strategy: FallbackStrategy = FallbackStrategy.RAISE_EXCEPTION
    fallback_value: Any = None
    fallback_function: Callable[..., Any] | None = None

    def to_circuit_config(self) -> CircuitConfig:
        """Convert to standard CircuitConfig."""
        return CircuitConfig(
            failure_threshold=self.failure_threshold,
            recovery_timeout=self.recovery_timeout,
            success_threshold=self.success_threshold,
            sliding_window_size=self.sliding_window_size,
            expected_exception_types=self.expected_exceptions or None,
        )


class CircuitBreakerMixin:
    """
    Mixin class that provides standardized circuit breaker functionality.

    Provides common patterns:
    - Automatic circuit breaker initialization
    - Standard operation protection
    - Fallback handling
    - Health status reporting
    """

    def __init__(self, circuit_config: CircuitBreakerConfig | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize circuit breaker with component-specific config
        if circuit_config is None:
            circuit_config = self._get_default_circuit_config()

        # Set name if not provided
        if circuit_config.name is None:
            circuit_config.name = f"{self.__class__.__name__}"

        self._circuit_config = circuit_config
        self.circuit_breaker = CircuitBreaker(
            name=circuit_config.name, config=circuit_config.to_circuit_config()
        )

        logger.debug(f"Circuit breaker initialized for {circuit_config.name}")

    def _get_default_circuit_config(self) -> CircuitBreakerConfig:
        """Override in subclasses to provide component-specific defaults."""
        return CircuitBreakerConfig()

    def execute_with_circuit_breaker(
        self,
        operation: Callable[[], T],
        operation_name: str,
        fallback_strategy: FallbackStrategy | None = None,
        fallback_value: T | None = None,
        fallback_function: Callable[[], T] | None = None,
    ) -> T | None:
        """
        Execute operation with circuit breaker protection and fallback handling.

        Args:
            operation: Function to execute
            operation_name: Name for logging and metrics
            fallback_strategy: Override default fallback strategy
            fallback_value: Override default fallback value
            fallback_function: Override default fallback function

        Returns:
            Operation result or fallback value

        Raises:
            CircuitBreakerError: If circuit is open and no fallback
            Exception: Original operation exceptions
        """
        try:
            # Use circuit breaker call method
            return self.circuit_breaker.call(operation, operation_name)

        except CircuitBreakerError as e:
            # Handle circuit breaker open state
            strategy = fallback_strategy or self._circuit_config.fallback_strategy

            logger.warning(f"Circuit breaker open for {operation_name}: {e}")

            if strategy == FallbackStrategy.RAISE_EXCEPTION:
                raise
            elif strategy == FallbackStrategy.RETURN_FALLBACK:
                fallback = (
                    fallback_value
                    if fallback_value is not None
                    else self._circuit_config.fallback_value
                )
                logger.info(f"Using fallback value for {operation_name}: {fallback}")
                return fallback
            elif strategy == FallbackStrategy.RETURN_NONE:
                logger.info(f"Returning None for {operation_name} due to circuit breaker")
                return None
            elif strategy == FallbackStrategy.CALL_FALLBACK_FUNC:
                fallback_func = fallback_function or self._circuit_config.fallback_function
                if fallback_func:
                    logger.info(f"Using fallback function for {operation_name}")
                    return fallback_func()
                else:
                    logger.error(f"No fallback function provided for {operation_name}")
                    raise
            else:
                raise

    def is_circuit_available(self) -> bool:
        """Check if circuit breaker is allowing requests."""
        return self.circuit_breaker.is_available()

    def get_circuit_health_status(self) -> dict[str, Any]:
        """Get circuit breaker health status."""
        return self.circuit_breaker.get_health_status()

    def force_circuit_open(self) -> None:
        """Force circuit breaker to open state (for testing)."""
        self.circuit_breaker.force_open()

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state."""
        self.circuit_breaker.reset()


class StrategistCircuitBreakerMixin(CircuitBreakerMixin):
    """Circuit breaker mixin optimized for strategy components."""

    def _get_default_circuit_config(self) -> CircuitBreakerConfig:
        """Strategy-specific circuit breaker configuration."""
        from .opus_strategist_exceptions import (
            MalformedResponseError,
            ResponseTimeoutError,
        )

        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            expected_exceptions=(ResponseTimeoutError, MalformedResponseError, ConnectionError),
            fallback_strategy=FallbackStrategy.CALL_FALLBACK_FUNC,
        )


class EmulatorCircuitBreakerMixin(CircuitBreakerMixin):
    """Circuit breaker mixin optimized for emulator components."""

    def _get_default_circuit_config(self) -> CircuitBreakerConfig:
        """Emulator-specific circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=15.0,  # Shorter timeout for emulator recovery
            success_threshold=1,
            expected_exceptions=(ConnectionError, TimeoutError),
            fallback_strategy=FallbackStrategy.RETURN_FALLBACK,
            fallback_value={"status": "error", "fallback": True},
        )


class CheckpointCircuitBreakerMixin(CircuitBreakerMixin):
    """Circuit breaker mixin optimized for checkpoint components."""

    def _get_default_circuit_config(self) -> CircuitBreakerConfig:
        """Checkpoint-specific circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=2,  # Strict for data integrity
            recovery_timeout=10.0,  # Quick recovery for checkpoints
            success_threshold=1,
            expected_exceptions=(IOError, OSError, FileNotFoundError),
            fallback_strategy=FallbackStrategy.RAISE_EXCEPTION,  # Don't hide checkpoint errors
        )


def with_circuit_breaker(
    circuit_breaker: CircuitBreaker,
    operation_name: str | None = None,
    fallback_strategy: FallbackStrategy = FallbackStrategy.RAISE_EXCEPTION,
    fallback_value: Any = None,
    fallback_function: Callable[..., Any] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    """
    Decorator to add circuit breaker protection to methods.

    Args:
        circuit_breaker: CircuitBreaker instance to use
        operation_name: Override operation name (defaults to function name)
        fallback_strategy: How to handle circuit breaker open state
        fallback_value: Value to return if fallback strategy is RETURN_FALLBACK
        fallback_function: Function to call if fallback strategy is CALL_FALLBACK_FUNC
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T | None:
            op_name = operation_name or func.__name__

            def operation() -> T:
                return func(*args, **kwargs)

            try:
                return circuit_breaker.call(operation, op_name)

            except CircuitBreakerError as e:
                logger.warning(f"Circuit breaker open for {op_name}: {e}")

                if fallback_strategy == FallbackStrategy.RAISE_EXCEPTION:
                    raise
                elif fallback_strategy == FallbackStrategy.RETURN_FALLBACK:
                    logger.info(f"Using fallback value for {op_name}")
                    return fallback_value
                elif fallback_strategy == FallbackStrategy.RETURN_NONE:
                    logger.info(f"Returning None for {op_name}")
                    return None
                elif fallback_strategy == FallbackStrategy.CALL_FALLBACK_FUNC:
                    if fallback_function:
                        logger.info(f"Using fallback function for {op_name}")
                        return fallback_function(*args, **kwargs)
                    else:
                        logger.error(f"No fallback function for {op_name}")
                        raise
                else:
                    raise

        return wrapper

    return decorator


def protected_operation(
    circuit_config: CircuitBreakerConfig | None = None,
    operation_name: str | None = None,
    fallback_strategy: FallbackStrategy = FallbackStrategy.RAISE_EXCEPTION,
    fallback_value: Any = None,
) -> Callable[[type], type]:
    """
    Class decorator to add circuit breaker protection to all public methods.

    Args:
        circuit_config: Circuit breaker configuration
        operation_name: Override operation name pattern
        fallback_strategy: How to handle circuit breaker open states
        fallback_value: Default fallback value for all operations
    """

    def class_decorator(cls: type) -> type:
        # Store original init
        original_init = cls.__init__

        def new_init(self, *args, **kwargs) -> None:
            # Initialize with circuit breaker
            if circuit_config:
                config = circuit_config
                if config.name is None:
                    config.name = cls.__name__
            else:
                config = CircuitBreakerConfig(name=cls.__name__)

            self._circuit_breaker_config = config
            self.circuit_breaker = CircuitBreaker(
                name=config.name, config=config.to_circuit_config()
            )

            # Call original init
            original_init(self, *args, **kwargs)

        # Replace init
        cls.__init__ = new_init

        # Wrap public methods
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and callable(getattr(cls, attr_name)):
                original_method = getattr(cls, attr_name)
                if hasattr(original_method, "__func__"):  # Skip static methods
                    wrapped_method = with_circuit_breaker(
                        # Circuit breaker will be available after init
                        lambda self: self.circuit_breaker,
                        operation_name or attr_name,
                        fallback_strategy,
                        fallback_value,
                    )(original_method)
                    setattr(cls, attr_name, wrapped_method)

        return cls

    return class_decorator


class CircuitBreakerRegistry:
    """
    Registry for managing circuit breakers across components.

    Provides centralized monitoring and management of all circuit breakers.
    """

    _instance: Optional["CircuitBreakerRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CircuitBreakerRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._registry_lock = threading.RLock()
        self._initialized = True

    def register(self, circuit_breaker: CircuitBreaker) -> None:
        """Register a circuit breaker for monitoring."""
        with self._registry_lock:
            self._circuit_breakers[circuit_breaker.name] = circuit_breaker

    def get_all_health_status(self) -> dict[str, dict[str, Any]]:
        """Get health status from all registered circuit breakers."""
        with self._registry_lock:
            return {name: cb.get_health_status() for name, cb in self._circuit_breakers.items()}

    def get_open_circuits(self) -> dict[str, CircuitBreaker]:
        """Get all circuit breakers in open state."""
        with self._registry_lock:
            return {
                name: cb for name, cb in self._circuit_breakers.items() if not cb.is_available()
            }

    def force_all_open(self) -> None:
        """Force all circuit breakers to open state (for testing)."""
        with self._registry_lock:
            for cb in self._circuit_breakers.values():
                cb.force_open()

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        with self._registry_lock:
            for cb in self._circuit_breakers.values():
                cb.reset()


# Create global registry instance
_circuit_registry = CircuitBreakerRegistry()


def get_circuit_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return _circuit_registry
