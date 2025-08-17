"""
Tests for CircuitBreaker: Kernel-quality circuit breaker pattern implementation.

Comprehensive test coverage for circuit breaker functionality:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure threshold detection and sliding window
- Recovery timeout and automatic retry
- Thread safety and concurrent operations

Author: Linus Torbot - Kernel Quality Standards Applied
"""

import threading
import time
from unittest.mock import Mock

import pytest
from requests.exceptions import ConnectionError

from claudelearnspokemon.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitConfig,
    CircuitState,
)


@pytest.mark.slow
class TestCircuitBreaker:
    """Test suite for CircuitBreaker functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.config = CircuitConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Fast recovery for testing
            success_threshold=2,
            sliding_window_size=5,
        )
        self.circuit_breaker = CircuitBreaker("test-circuit", self.config)

    def test_initialization_default_config(self):
        """Test CircuitBreaker initializes with default configuration."""
        cb = CircuitBreaker()

        assert cb.name == "default"
        assert cb.config.failure_threshold == 5
        assert cb.config.recovery_timeout == 60.0
        assert cb.get_state() == CircuitState.CLOSED

    def test_initialization_custom_config(self):
        """Test CircuitBreaker initializes with custom configuration."""
        assert self.circuit_breaker.name == "test-circuit"
        assert self.circuit_breaker.config.failure_threshold == 3
        assert self.circuit_breaker.get_state() == CircuitState.CLOSED

    def test_successful_operation_closed_state(self):
        """Test successful operation in CLOSED state."""
        mock_func = Mock(return_value="success")

        result = self.circuit_breaker.call(mock_func, "test_op")

        assert result == "success"
        assert self.circuit_breaker.get_state() == CircuitState.CLOSED
        assert mock_func.call_count == 1

        # Check metrics
        metrics = self.circuit_breaker.get_metrics()
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0

    def test_circuit_trip_on_consecutive_failures(self):
        """Test circuit trips after reaching failure threshold."""
        mock_func = Mock(side_effect=ConnectionError("Connection failed"))

        # Execute failures up to threshold
        for i in range(self.config.failure_threshold):
            with pytest.raises(ConnectionError):
                self.circuit_breaker.call(mock_func, f"fail_{i}")

            if i < self.config.failure_threshold - 1:
                assert self.circuit_breaker.get_state() == CircuitState.CLOSED
            else:
                assert self.circuit_breaker.get_state() == CircuitState.OPEN

        # Check metrics
        metrics = self.circuit_breaker.get_metrics()
        assert metrics.failed_requests == self.config.failure_threshold
        assert metrics.circuit_trips == 1

    def test_circuit_rejects_requests_when_open(self):
        """Test circuit breaker rejects requests when OPEN."""
        # Trip the circuit
        mock_func = Mock(side_effect=ConnectionError("Connection failed"))
        for _ in range(self.config.failure_threshold):
            with pytest.raises(ConnectionError):
                self.circuit_breaker.call(mock_func, "fail")

        assert self.circuit_breaker.get_state() == CircuitState.OPEN

        # Now all requests should be rejected immediately
        with pytest.raises(CircuitBreakerError) as exc_info:
            self.circuit_breaker.call(Mock(return_value="should_not_execute"), "rejected")

        assert "is OPEN" in str(exc_info.value)

        # Check rejected request metric
        metrics = self.circuit_breaker.get_metrics()
        assert metrics.rejected_requests == 1

    def test_circuit_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to HALF_OPEN after recovery timeout."""
        # Trip the circuit
        mock_func = Mock(side_effect=ConnectionError("Connection failed"))
        for _ in range(self.config.failure_threshold):
            with pytest.raises(ConnectionError):
                self.circuit_breaker.call(mock_func, "fail")

        assert self.circuit_breaker.get_state() == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(self.config.recovery_timeout + 0.1)

        # Next request should transition to HALF_OPEN
        successful_func = Mock(return_value="success")
        result = self.circuit_breaker.call(successful_func, "recovery_test")

        assert result == "success"
        assert self.circuit_breaker.get_state() == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success_threshold(self):
        """Test circuit closes from HALF_OPEN after success threshold."""
        # Trip circuit and transition to HALF_OPEN
        self._trip_circuit_and_recover()

        assert self.circuit_breaker.get_state() == CircuitState.HALF_OPEN

        # Execute successful operations to reach success threshold
        successful_func = Mock(return_value="success")
        for i in range(self.config.success_threshold):
            result = self.circuit_breaker.call(successful_func, f"success_{i}")
            assert result == "success"

        # Circuit should now be CLOSED
        assert self.circuit_breaker.get_state() == CircuitState.CLOSED

    def test_half_open_trips_on_failure(self):
        """Test circuit trips from HALF_OPEN on failure."""
        # Trip circuit and transition to HALF_OPEN
        self._trip_circuit_and_recover()

        assert self.circuit_breaker.get_state() == CircuitState.HALF_OPEN

        # Fail during half-open testing
        mock_func = Mock(side_effect=ConnectionError("Still failing"))

        with pytest.raises(ConnectionError):
            self.circuit_breaker.call(mock_func, "half_open_fail")

        # Should go back to OPEN
        assert self.circuit_breaker.get_state() == CircuitState.OPEN

    def test_sliding_window_failure_rate(self):
        """Test sliding window failure rate detection."""
        # Create pattern that avoids consecutive failure threshold but hits sliding window rate
        # Window size = 5, need >80% failure rate with >=3 failures
        # Pattern: F-S-F-S-F-F-F (4 failures out of 5 = 80% in final window, >=3 failures)
        operations = [
            Mock(side_effect=ConnectionError("Fail 0")),  # F
            Mock(return_value="Success 1"),  # S
            Mock(side_effect=ConnectionError("Fail 2")),  # F
            Mock(return_value="Success 3"),  # S (break consecutive)
            Mock(side_effect=ConnectionError("Fail 4")),  # F
            Mock(side_effect=ConnectionError("Fail 5")),  # F (2 consecutive, below threshold of 3)
            Mock(
                side_effect=ConnectionError("Fail 6")
            ),  # F - window [S,F,F,F] = 4/5 = 80%, >= 3 failures
        ]

        # Execute operations - should not trip until final operation
        for i, op in enumerate(operations[:-1]):
            try:
                self.circuit_breaker.call(op, f"op_{i}")
            except ConnectionError:
                pass

        # Circuit should still be closed (2 consecutive < threshold of 3)
        assert self.circuit_breaker.get_state() == CircuitState.CLOSED

        # This final failure should trigger sliding window logic
        # The sliding window will be [S,F,F,F] = 4/5 = 80% failure rate with 4 failures
        try:
            self.circuit_breaker.call(operations[-1], "op_final")
        except ConnectionError:
            pass

        # Circuit should now trip due to high failure rate in sliding window
        assert self.circuit_breaker.get_state() == CircuitState.OPEN

    def test_non_expected_exceptions_do_not_trip_circuit(self):
        """Test that non-expected exceptions don't affect circuit state."""
        # Configure circuit with specific exception types
        config = CircuitConfig(
            failure_threshold=2, expected_exception_types=(ConnectionError, TimeoutError)
        )
        cb = CircuitBreaker("selective-circuit", config)

        # ValueError is not in expected_exception_types
        mock_func = Mock(side_effect=ValueError("Not a network error"))

        # This should pass through without affecting circuit state
        with pytest.raises(ValueError):
            cb.call(mock_func, "unexpected_error")

        assert cb.get_state() == CircuitState.CLOSED

        # But ConnectionError should affect circuit
        conn_error_func = Mock(side_effect=ConnectionError("Network error"))
        with pytest.raises(ConnectionError):
            cb.call(conn_error_func, "network_error")

        metrics = cb.get_metrics()
        assert metrics.failed_requests == 1  # Only ConnectionError counted

    def test_metrics_collection(self):
        """Test comprehensive metrics collection."""
        # Execute various operations
        success_func = Mock(return_value="success")
        self.circuit_breaker.call(success_func, "success")

        fail_func = Mock(side_effect=ConnectionError("fail"))
        with pytest.raises(ConnectionError):
            self.circuit_breaker.call(fail_func, "fail")

        metrics = self.circuit_breaker.get_metrics()

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.success_rate == 50.0
        assert metrics.current_consecutive_failures == 1

    def test_health_status_reporting(self):
        """Test comprehensive health status reporting."""
        health = self.circuit_breaker.get_health_status()

        assert health["name"] == "test-circuit"
        assert health["state"] == "CLOSED"
        assert health["is_available"] is True
        assert health["success_rate"] == 100.0
        assert "total_requests" in health
        assert "circuit_trips" in health

    def test_manual_reset(self):
        """Test manual circuit reset functionality."""
        # Trip the circuit
        self._trip_circuit()
        assert self.circuit_breaker.get_state() == CircuitState.OPEN

        # Manual reset
        self.circuit_breaker.reset()

        assert self.circuit_breaker.get_state() == CircuitState.CLOSED
        metrics = self.circuit_breaker.get_metrics()
        assert metrics.total_requests == 0
        assert metrics.circuit_trips == 0

    def test_force_open(self):
        """Test manual force open functionality."""
        assert self.circuit_breaker.get_state() == CircuitState.CLOSED

        self.circuit_breaker.force_open()

        assert self.circuit_breaker.get_state() == CircuitState.OPEN

        # Should reject requests
        with pytest.raises(CircuitBreakerError):
            self.circuit_breaker.call(Mock(return_value="test"), "test")

    def test_is_available_check(self):
        """Test is_available check without making requests."""
        assert self.circuit_breaker.is_available() is True

        # Trip circuit
        self._trip_circuit()
        assert self.circuit_breaker.is_available() is False

        # Wait for recovery
        time.sleep(self.config.recovery_timeout + 0.1)
        assert self.circuit_breaker.is_available() is True

    def test_thread_safety(self):
        """Test circuit breaker thread safety."""
        results = []
        errors = []

        def worker():
            """Worker function for concurrent testing."""
            try:
                # Mix of successful and failing operations
                success_func = Mock(return_value="success")
                result = self.circuit_breaker.call(success_func, "concurrent_success")
                results.append(result)

                fail_func = Mock(side_effect=ConnectionError("concurrent_fail"))
                self.circuit_breaker.call(fail_func, "concurrent_fail")

            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify we got results and the circuit breaker handled concurrent access
        assert len(results) > 0
        assert len(errors) > 0

        # Circuit should be in a consistent state
        state = self.circuit_breaker.get_state()
        assert state in [CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN]

    def test_string_representation(self):
        """Test CircuitBreaker string representation."""
        str_repr = str(self.circuit_breaker)
        assert "CircuitBreaker" in str_repr
        assert "test-circuit" in str_repr
        assert "CLOSED" in str_repr

    def test_recovery_attempts_metric(self):
        """Test recovery attempts are tracked."""
        # Trip circuit
        self._trip_circuit()

        # Wait and make recovery attempt
        time.sleep(self.config.recovery_timeout + 0.1)
        success_func = Mock(return_value="success")
        self.circuit_breaker.call(success_func, "recovery")

        metrics = self.circuit_breaker.get_metrics()
        assert metrics.recovery_attempts == 1

    def test_consecutive_failures_reset_on_success(self):
        """Test consecutive failures counter resets on success."""
        # Have some failures (but not enough to trip)
        fail_func = Mock(side_effect=ConnectionError("fail"))
        for _ in range(2):  # Less than threshold
            with pytest.raises(ConnectionError):
                self.circuit_breaker.call(fail_func, "partial_fail")

        metrics = self.circuit_breaker.get_metrics()
        assert metrics.current_consecutive_failures == 2

        # Now succeed
        success_func = Mock(return_value="success")
        self.circuit_breaker.call(success_func, "reset_counter")

        metrics = self.circuit_breaker.get_metrics()
        assert metrics.current_consecutive_failures == 0

    def _trip_circuit(self):
        """Helper method to trip the circuit."""
        mock_func = Mock(side_effect=ConnectionError("Connection failed"))
        for _ in range(self.config.failure_threshold):
            with pytest.raises(ConnectionError):
                self.circuit_breaker.call(mock_func, "trip")

    def _trip_circuit_and_recover(self):
        """Helper method to trip circuit and transition to HALF_OPEN."""
        self._trip_circuit()
        time.sleep(self.config.recovery_timeout + 0.1)

        # Make one successful call to transition to HALF_OPEN
        success_func = Mock(return_value="recovery")
        self.circuit_breaker.call(success_func, "recover")
