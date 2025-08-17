"""
Tests for RetryManager: Kernel-quality retry logic with exponential backoff.

Comprehensive test coverage for all failure scenarios and recovery mechanisms:
- Exponential backoff behavior validation
- Failure classification testing
- Retry policy configuration
- Metrics collection verification

Author: Linus Torbot - Kernel Quality Standards Applied
"""

import time
from unittest.mock import Mock, patch

from requests.exceptions import ConnectionError, HTTPError, ReadTimeout, Timeout

from claudelearnspokemon.retry_manager import (
    FailureType,
    RetryConfig,
    RetryManager,
    RetryMetrics,
    RetryOutcome,
)


class TestRetryManager:
    """Test suite for RetryManager functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.config = RetryConfig(
            max_retries=3,
            base_delay=0.1,  # Fast tests
            max_delay=1.0,
            backoff_multiplier=2.0,
            jitter=False,  # Disable for predictable testing
        )
        self.retry_manager = RetryManager(self.config)

    def test_initialization_default_config(self):
        """Test RetryManager initializes with default configuration."""
        manager = RetryManager()

        assert manager.config.max_retries == 3
        assert manager.config.base_delay == 0.5
        assert manager.config.backoff_multiplier == 2.0
        assert manager.config.jitter is True
        assert isinstance(manager.metrics, RetryMetrics)

    def test_initialization_custom_config(self):
        """Test RetryManager initializes with custom configuration."""
        assert self.retry_manager.config.max_retries == 3
        assert self.retry_manager.config.base_delay == 0.1
        assert self.retry_manager.config.jitter is False

    def test_successful_operation_no_retry(self):
        """Test successful operation requires no retries."""
        mock_func = Mock(return_value="success")

        outcome, result, exception = self.retry_manager.execute_with_retry(
            mock_func, "test_operation"
        )

        assert outcome == RetryOutcome.SUCCESS
        assert result == "success"
        assert exception is None
        assert mock_func.call_count == 1

    def test_retry_on_connection_error(self):
        """Test retry behavior on ConnectionError."""
        mock_func = Mock(
            side_effect=[
                ConnectionError("Connection failed"),
                ConnectionError("Still failing"),
                "success",  # Third attempt succeeds
            ]
        )

        start_time = time.time()
        outcome, result, exception = self.retry_manager.execute_with_retry(
            mock_func, "test_connection"
        )
        elapsed = time.time() - start_time

        assert outcome == RetryOutcome.SUCCESS
        assert result == "success"
        assert exception is None
        assert mock_func.call_count == 3

        # Should have some delay for retries
        assert elapsed >= 0.3  # 0.1 + 0.2 = 0.3s minimum delay

        # Check metrics
        metrics = self.retry_manager.get_metrics()
        assert metrics.total_attempts == 2  # Only failed attempts counted
        assert metrics.successful_retries == 1

    def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        error = ConnectionError("Persistent failure")
        mock_func = Mock(side_effect=error)

        outcome, result, exception = self.retry_manager.execute_with_retry(
            mock_func, "persistent_failure"
        )

        assert outcome == RetryOutcome.MAX_RETRIES_EXCEEDED
        assert result is None
        assert exception == error
        assert mock_func.call_count == 4  # Initial + 3 retries

    def test_non_retriable_error(self):
        """Test non-retriable errors are not retried."""
        # 404 is not in retriable status codes by default
        response_mock = Mock()
        response_mock.status_code = 404
        error = HTTPError(response=response_mock)
        mock_func = Mock(side_effect=error)

        outcome, result, exception = self.retry_manager.execute_with_retry(mock_func, "not_found")

        assert outcome == RetryOutcome.NON_RETRIABLE_ERROR
        assert result is None
        assert exception == error
        assert mock_func.call_count == 1  # No retries

    def test_retriable_http_errors(self):
        """Test that 5xx and specific 4xx errors are retried."""
        response_mock = Mock()
        response_mock.status_code = 503  # Service Unavailable - retriable

        mock_func = Mock(side_effect=[HTTPError(response=response_mock), "success"])

        outcome, result, exception = self.retry_manager.execute_with_retry(
            mock_func, "service_unavailable"
        )

        assert outcome == RetryOutcome.SUCCESS
        assert result == "success"
        assert mock_func.call_count == 2

    def test_failure_classification_network_errors(self):
        """Test classification of network-related errors."""
        test_cases = [
            (ConnectionError("Connection failed"), FailureType.NETWORK_ERROR),
            (Timeout("Request timeout"), FailureType.NETWORK_ERROR),
            (ReadTimeout("Read timeout"), FailureType.NETWORK_ERROR),
        ]

        for error, expected_type in test_cases:
            classified_type = self.retry_manager._classify_failure(error)
            assert classified_type == expected_type

    def test_failure_classification_http_errors(self):
        """Test classification of HTTP errors."""
        # Server error (5xx)
        server_response = Mock()
        server_response.status_code = 500
        server_error = HTTPError(response=server_response)
        assert self.retry_manager._classify_failure(server_error) == FailureType.SERVER_ERROR

        # Client error (4xx)
        client_response = Mock()
        client_response.status_code = 404
        client_error = HTTPError(response=client_response)
        assert self.retry_manager._classify_failure(client_error) == FailureType.CLIENT_ERROR

        # Session error (401/403)
        session_response = Mock()
        session_response.status_code = 401
        session_error = HTTPError(response=session_response)
        assert self.retry_manager._classify_failure(session_error) == FailureType.SESSION_ERROR

    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        # Test with jitter disabled for predictable results
        delays = []
        for attempt in range(4):
            delay = self.retry_manager._calculate_delay(attempt)
            delays.append(delay)

        # Expected: 0.1, 0.2, 0.4, 0.8
        assert delays[0] == 0.1
        assert delays[1] == 0.2
        assert delays[2] == 0.4
        assert delays[3] == 0.8

    def test_exponential_backoff_with_max_delay(self):
        """Test exponential backoff respects max_delay cap."""
        delay = self.retry_manager._calculate_delay(10)  # Very high attempt
        assert delay <= self.config.max_delay

    def test_exponential_backoff_with_jitter(self):
        """Test exponential backoff with jitter adds randomization."""
        manager = RetryManager(RetryConfig(base_delay=0.5, jitter=True))

        delays = []
        for _ in range(10):
            delay = manager._calculate_delay(1)  # Expected base: 1.0s
            delays.append(delay)

        # All delays should be different due to jitter
        assert len(set(delays)) > 5  # Most should be unique
        # All should be around 1.0s ± 25%
        for delay in delays:
            assert 0.75 <= delay <= 1.25

    def test_session_error_identification(self):
        """Test identification of session-related errors."""
        session_errors = [
            Exception("Session expired"),
            Exception("Unauthorized access"),
            ValueError("Invalid session token"),
        ]

        for error in session_errors:
            failure_type = self.retry_manager._classify_failure(error)
            assert failure_type == FailureType.SESSION_ERROR

    def test_metrics_collection(self):
        """Test retry metrics are collected correctly."""
        # Simulate multiple operations with different outcomes
        mock_func1 = Mock(return_value="success")
        self.retry_manager.execute_with_retry(mock_func1)

        mock_func2 = Mock(side_effect=[ConnectionError("fail"), "success"])
        self.retry_manager.execute_with_retry(mock_func2)

        metrics = self.retry_manager.get_metrics()

        assert metrics.total_attempts == 1  # Only the failed attempt
        assert metrics.successful_retries == 1
        assert metrics.failed_retries == 0
        assert FailureType.NETWORK_ERROR in metrics.failure_types
        assert metrics.failure_types[FailureType.NETWORK_ERROR] == 1

    def test_metrics_reset(self):
        """Test metrics can be reset."""
        # Generate some metrics
        mock_func = Mock(side_effect=[ConnectionError("fail"), "success"])
        self.retry_manager.execute_with_retry(mock_func)

        # Verify metrics exist
        metrics_before = self.retry_manager.get_metrics()
        assert metrics_before.total_attempts > 0

        # Reset and verify
        self.retry_manager.reset_metrics()
        metrics_after = self.retry_manager.get_metrics()

        assert metrics_after.total_attempts == 0
        assert metrics_after.successful_retries == 0
        assert metrics_after.failed_retries == 0

    def test_string_representation(self):
        """Test RetryManager string representation."""
        str_repr = str(self.retry_manager)
        assert "RetryManager" in str_repr
        assert "max_retries=3" in str_repr
        assert "base_delay=0.1" in str_repr

    @patch("time.sleep")
    def test_delay_timing_accuracy(self, mock_sleep):
        """Test that delays are applied as calculated."""
        mock_func = Mock(side_effect=[ConnectionError("fail"), ConnectionError("fail"), "success"])

        self.retry_manager.execute_with_retry(mock_func)

        # Check that sleep was called with correct delays
        expected_calls = [0.1, 0.2]  # First and second retry delays
        actual_calls = [call[0][0] for call in mock_sleep.call_args_list]

        assert actual_calls == expected_calls

    def test_custom_retriable_status_codes(self):
        """Test custom retriable status codes configuration."""
        config = RetryConfig(retriable_status_codes={418, 420})  # Custom codes
        manager = RetryManager(config)

        # 418 should be retriable
        response_418 = Mock()
        response_418.status_code = 418
        error_418 = HTTPError(response=response_418)
        assert manager._is_retriable(FailureType.CLIENT_ERROR, error_418)

        # 404 should not be retriable
        response_404 = Mock()
        response_404.status_code = 404
        error_404 = HTTPError(response=response_404)
        assert not manager._is_retriable(FailureType.CLIENT_ERROR, error_404)
