"""
ConversationErrorHandler: Robust error handling for Claude conversation operations.

This module provides conversation-level error handling that integrates with the existing
RetryManager and CircuitBreaker infrastructure to ensure reliable communication with
Claude CLI processes.

Key Features:
- RetryManager integration for transient conversation failures
- CircuitBreaker protection for persistent Claude CLI failures
- Conversation-specific error classification and handling
- Performance monitoring and metrics collection
- Configurable retry policies and failure thresholds

Performance Targets:
- Message retry operation: <100ms additional overhead per retry
- Circuit breaker check: <1ms per operation
- Conversation error classification: <5ms per operation
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, TypeVar

from .circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitConfig
from .retry_manager import RetryConfig, RetryManager, RetryOutcome

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConversationError(Exception):
    """Base exception for conversation-level errors."""
    pass


class ConversationFailureType(Enum):
    """Classification of conversation failure types."""
    
    CLAUDE_CLI_ERROR = auto()  # Claude CLI process communication errors
    MESSAGE_TIMEOUT = auto()   # Message processing timeout
    INVALID_RESPONSE = auto()  # Invalid or malformed response from Claude
    SESSION_ERROR = auto()     # Session-related failures
    RESOURCE_EXHAUSTION = auto()  # Resource limits exceeded
    UNKNOWN_ERROR = auto()     # Unexpected conversation failures


@dataclass
class ConversationErrorConfig:
    """Configuration for conversation error handling."""
    
    # Retry configuration
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    
    # Circuit breaker configuration
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    
    # Conversation-specific settings
    message_timeout: float = 30.0
    response_validation: bool = True
    
    def to_retry_config(self) -> RetryConfig:
        """Convert to RetryManager configuration."""
        return RetryConfig(
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            backoff_multiplier=self.backoff_multiplier,
            jitter=True,
        )
    
    def to_circuit_config(self) -> CircuitConfig:
        """Convert to CircuitBreaker configuration."""
        return CircuitConfig(
            failure_threshold=self.failure_threshold,
            recovery_timeout=self.recovery_timeout,
            success_threshold=self.success_threshold,
            expected_exception_types=(ConversationError, ConnectionError, TimeoutError),
        )


@dataclass
class ConversationMetrics:
    """Metrics for conversation error handling operations."""
    
    total_conversation_attempts: int = 0
    successful_conversations: int = 0
    failed_conversations: int = 0
    retry_attempts: int = 0
    circuit_breaker_trips: int = 0
    avg_conversation_time_ms: float = 0.0
    failure_types: dict[ConversationFailureType, int] | None = None
    
    def __post_init__(self) -> None:
        """Initialize failure types counter."""
        if self.failure_types is None:
            self.failure_types = dict.fromkeys(ConversationFailureType, 0)
    
    def record_conversation_attempt(self) -> None:
        """Record conversation attempt."""
        self.total_conversation_attempts += 1
    
    def record_success(self, duration_ms: float) -> None:
        """Record successful conversation."""
        self.successful_conversations += 1
        # Update running average
        if self.successful_conversations == 1:
            self.avg_conversation_time_ms = duration_ms
        else:
            self.avg_conversation_time_ms = (
                self.avg_conversation_time_ms * (self.successful_conversations - 1) + duration_ms
            ) / self.successful_conversations
    
    def record_failure(self, failure_type: ConversationFailureType) -> None:
        """Record conversation failure."""
        self.failed_conversations += 1
        if self.failure_types:
            self.failure_types[failure_type] += 1
    
    def record_retry(self) -> None:
        """Record retry attempt."""
        self.retry_attempts += 1
    
    def record_circuit_trip(self) -> None:
        """Record circuit breaker trip."""
        self.circuit_breaker_trips += 1


class ConversationErrorHandler:
    """
    Comprehensive error handling for Claude conversation operations.
    
    Integrates RetryManager and CircuitBreaker to provide robust conversation-level
    error handling with appropriate retry strategies and failure protection.
    """
    
    def __init__(self, config: ConversationErrorConfig | None = None, name: str = "conversation"):
        """
        Initialize conversation error handler.
        
        Args:
            config: Error handling configuration (uses defaults if None)
            name: Descriptive name for logging and metrics
        """
        self.config = config or ConversationErrorConfig()
        self.name = name
        self.metrics = ConversationMetrics()
        
        # Initialize underlying error handling components
        self.retry_manager = RetryManager(self.config.to_retry_config())
        self.circuit_breaker = CircuitBreaker(
            name=f"{name}_circuit_breaker",
            config=self.config.to_circuit_config()
        )
        
        logger.info(
            f"ConversationErrorHandler '{name}' initialized: "
            f"max_retries={self.config.max_retries}, "
            f"failure_threshold={self.config.failure_threshold}"
        )
    
    def send_message_with_retry(
        self,
        send_func: Callable[[], T],
        message_content: str,
        operation_name: str = "send_message"
    ) -> tuple[bool, T | None, Exception | None]:
        """
        Send message with comprehensive error handling and retry logic.
        
        Args:
            send_func: Function that sends message and returns response
            message_content: Content being sent (for logging/metrics)
            operation_name: Descriptive name for the operation
            
        Returns:
            Tuple of (success, result, exception)
        """
        start_time = time.time()
        self.metrics.record_conversation_attempt()
        
        try:
            # Execute through circuit breaker protection
            def protected_send():
                return self.circuit_breaker.call(send_func, operation_name)
            
            # Execute with retry logic
            retry_outcome, result, exception = self.retry_manager.execute_with_retry(
                protected_send, f"{self.name}_{operation_name}"
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if retry_outcome == RetryOutcome.SUCCESS and result is not None:
                self.metrics.record_success(duration_ms)
                logger.debug(f"Conversation {operation_name} succeeded in {duration_ms:.1f}ms")
                return True, result, None
            else:
                # Handle different failure outcomes
                failure_type = self._classify_failure_outcome(retry_outcome, exception)
                self.metrics.record_failure(failure_type)
                
                if retry_outcome == RetryOutcome.MAX_RETRIES_EXCEEDED:
                    self.metrics.retry_attempts += self.config.max_retries
                elif retry_outcome == RetryOutcome.CIRCUIT_BREAKER_OPEN:
                    self.metrics.record_circuit_trip()
                
                logger.warning(
                    f"Conversation {operation_name} failed after {duration_ms:.1f}ms: "
                    f"{retry_outcome.name} - {exception}"
                )
                return False, None, exception
                
        except CircuitBreakerError as e:
            # Circuit breaker blocked the operation
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_failure(ConversationFailureType.CLAUDE_CLI_ERROR)
            self.metrics.record_circuit_trip()
            
            logger.error(f"Conversation {operation_name} blocked by circuit breaker: {e}")
            return False, None, e
            
        except Exception as e:
            # Unexpected error in error handling itself
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_failure(ConversationFailureType.UNKNOWN_ERROR)
            
            logger.error(f"Unexpected error in conversation error handling: {e}")
            return False, None, e
    
    def _classify_failure_outcome(
        self, 
        outcome: RetryOutcome, 
        exception: Exception | None
    ) -> ConversationFailureType:
        """
        Classify failure outcome into appropriate conversation failure type.
        
        Args:
            outcome: Retry outcome from RetryManager
            exception: Original exception that caused failure
            
        Returns:
            ConversationFailureType for the failure
        """
        if outcome == RetryOutcome.CIRCUIT_BREAKER_OPEN:
            return ConversationFailureType.CLAUDE_CLI_ERROR
        
        if exception is None:
            return ConversationFailureType.UNKNOWN_ERROR
        
        # Classify based on exception type and message
        if isinstance(exception, TimeoutError) or "timeout" in str(exception).lower():
            return ConversationFailureType.MESSAGE_TIMEOUT
        
        if isinstance(exception, ConnectionError) or "connection" in str(exception).lower():
            return ConversationFailureType.CLAUDE_CLI_ERROR
        
        if "session" in str(exception).lower() or "authentication" in str(exception).lower():
            return ConversationFailureType.SESSION_ERROR
        
        if "invalid" in str(exception).lower() or "malformed" in str(exception).lower():
            return ConversationFailureType.INVALID_RESPONSE
        
        if "resource" in str(exception).lower() or "memory" in str(exception).lower():
            return ConversationFailureType.RESOURCE_EXHAUSTION
        
        return ConversationFailureType.UNKNOWN_ERROR
    
    def validate_response(self, response: Any) -> bool:
        """
        Validate Claude response format and content.
        
        Args:
            response: Response from Claude to validate
            
        Returns:
            True if response is valid
        """
        if not self.config.response_validation:
            return True
        
        # Basic response validation
        if response is None:
            return False
        
        # Check for common invalid response patterns
        response_str = str(response)
        if not response_str.strip():
            return False
        
        if "error" in response_str.lower() and "internal" in response_str.lower():
            return False
        
        return True
    
    def get_health_status(self) -> dict[str, Any]:
        """
        Get comprehensive health status for monitoring.
        
        Returns:
            Dictionary with health status information
        """
        circuit_health = self.circuit_breaker.get_health_status()
        retry_metrics = self.retry_manager.get_metrics()
        
        success_rate = 0.0
        if self.metrics.total_conversation_attempts > 0:
            success_rate = (
                self.metrics.successful_conversations / self.metrics.total_conversation_attempts
            ) * 100.0
        
        return {
            "name": self.name,
            "is_healthy": circuit_health["is_available"],
            "circuit_breaker": circuit_health,
            "retry_metrics": {
                "total_attempts": retry_metrics.total_attempts,
                "successful_retries": retry_metrics.successful_retries,
                "failed_retries": retry_metrics.failed_retries,
                "total_delay_time": retry_metrics.total_delay_time,
            },
            "conversation_metrics": {
                "total_attempts": self.metrics.total_conversation_attempts,
                "successful_conversations": self.metrics.successful_conversations,
                "failed_conversations": self.metrics.failed_conversations,
                "success_rate_percent": success_rate,
                "avg_conversation_time_ms": self.metrics.avg_conversation_time_ms,
                "retry_attempts": self.metrics.retry_attempts,
                "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
                "failure_breakdown": self.metrics.failure_types.copy() if self.metrics.failure_types else {},
            },
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics and error handling state."""
        self.metrics = ConversationMetrics()
        self.retry_manager.reset_metrics()
        self.circuit_breaker.reset()
        logger.info(f"ConversationErrorHandler '{self.name}' metrics reset")
    
    def force_circuit_open(self) -> None:
        """Force circuit breaker open (for testing/emergency)."""
        self.circuit_breaker.force_open()
        logger.warning(f"ConversationErrorHandler '{self.name}' circuit breaker forced open")
    
    def is_available(self) -> bool:
        """
        Check if error handler would allow operations.
        
        Returns:
            True if conversations would be allowed
        """
        return self.circuit_breaker.is_available()
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"ConversationErrorHandler(name='{self.name}', "
            f"circuit_state={self.circuit_breaker.get_state().name}, "
            f"success_rate={self.metrics.successful_conversations}/{self.metrics.total_conversation_attempts})"
        )