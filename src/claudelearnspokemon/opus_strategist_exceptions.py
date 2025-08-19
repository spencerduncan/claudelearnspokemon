"""
OpusStrategist Exception Hierarchy

Production-ready exception classes for strategy response processing.
Follows Google SRE principles for actionable error messages and
proper error categorization for monitoring and alerting.
"""

from typing import Any


class OpusStrategistError(Exception):
    """Base exception for all OpusStrategist-related errors."""

    def __init__(self, message: str, error_code: str = "OPUS_GENERAL_ERROR"):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class StrategyValidationError(OpusStrategistError):
    """
    Raised when strategy response fails validation rules.

    Indicates structural problems with Opus response format.
    Recovery: retry with different prompt or use fallback strategy.
    """

    def __init__(self, message: str, validation_failures: list | None = None):
        super().__init__(message, "STRATEGY_VALIDATION_FAILED")
        self.validation_failures = validation_failures or []


class MalformedResponseError(OpusStrategistError):
    """
    Raised when response cannot be parsed as valid JSON.

    Indicates communication or encoding issues with Opus.
    Recovery: retry request, check Claude connection, use cached response.
    """

    def __init__(self, message: str, raw_response: str | None = None):
        super().__init__(message, "MALFORMED_RESPONSE")
        self.raw_response = raw_response


class ResponseTimeoutError(OpusStrategistError):
    """
    Raised when response processing exceeds timeout limits.

    Indicates performance degradation or oversized responses.
    Recovery: increase timeout, implement response truncation, use cached strategy.
    """

    def __init__(self, message: str, timeout_seconds: float):
        super().__init__(message, "RESPONSE_TIMEOUT")
        self.timeout_seconds = timeout_seconds


class CacheError(OpusStrategistError):
    """
    Raised when response caching operations fail.

    Indicates memory pressure or cache corruption.
    Recovery: clear cache, reduce cache size, continue without caching.
    """

    def __init__(self, message: str, cache_operation: str):
        super().__init__(message, "CACHE_ERROR")
        self.cache_operation = cache_operation


class DirectiveExtractionError(OpusStrategistError):
    """
    Raised when strategic directives cannot be extracted from response.

    Indicates semantic parsing issues or unexpected response format.
    Recovery: use default directives, retry with different extraction rules.
    """

    def __init__(self, message: str, response_data: dict[str, Any] | None = None):
        super().__init__(message, "DIRECTIVE_EXTRACTION_FAILED")
        self.response_data = response_data


class CircuitBreakerError(OpusStrategistError):
    """
    Raised when circuit breaker is open due to repeated failures.

    Indicates systematic issues with Opus communication.
    Recovery: use fallback strategies, wait for circuit breaker reset.
    """

    def __init__(self, message: str, failure_count: int, reset_time: float):
        super().__init__(message, "CIRCUIT_BREAKER_OPEN")
        self.failure_count = failure_count
        self.reset_time = reset_time
