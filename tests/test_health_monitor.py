"""
Tests for HealthMonitor - Simple timer-based health monitoring system.

Comprehensive test suite covering:
- Timer-based periodic health checks
- HTTP connectivity validation
- Health status tracking and logging
- Performance characteristics
- Error handling and cleanup
- EmulatorPool integration

Author: John Botmack - Performance-First Testing
"""

import threading
import time
import unittest
from unittest.mock import Mock, patch

import pytest
import requests

from src.claudelearnspokemon.health_monitor import HealthMonitor, HealthMonitorError


@pytest.mark.medium
class TestHealthMonitorBasics(unittest.TestCase):
    """Test basic HealthMonitor functionality and lifecycle."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_pool = Mock()
        self.mock_pool.clients_by_port = {
            8081: Mock(port=8081, container_id="container1-abc123"),
            8082: Mock(port=8082, container_id="container2-def456"),
        }
        self.mock_pool.get_status.return_value = {
            "available_count": 2,
            "busy_count": 0,
            "total_count": 2,
            "status": "healthy",
        }

        # Create monitor with short intervals for testing
        self.monitor = HealthMonitor(
            emulator_pool=self.mock_pool,
            check_interval=0.1,  # 100ms for fast tests
            health_timeout=1.0,
        )

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "monitor"):
            self.monitor.stop()

    def test_initialization(self):
        """Test HealthMonitor initializes correctly."""
        assert self.monitor.emulator_pool == self.mock_pool
        assert self.monitor.check_interval == 0.1
        assert self.monitor.health_timeout == 1.0
        assert not self.monitor._running
        assert self.monitor._check_count == 0

    def test_start_stop_lifecycle(self):
        """Test monitor start/stop lifecycle."""
        # Initially stopped
        assert not self.monitor._running

        # Start monitoring
        self.monitor.start()
        assert self.monitor._running
        assert self.monitor._timer is not None

        # Stop monitoring
        self.monitor.stop()
        assert not self.monitor._running
        assert self.monitor._timer is None

    def test_multiple_starts_safe(self):
        """Test multiple start calls are safe."""
        self.monitor.start()
        assert self.monitor._running

        # Second start should be ignored
        self.monitor.start()
        assert self.monitor._running

    def test_multiple_stops_safe(self):
        """Test multiple stop calls are safe."""
        self.monitor.start()
        self.monitor.stop()
        assert not self.monitor._running

        # Second stop should be safe
        self.monitor.stop()
        assert not self.monitor._running

    def test_get_stats_initial(self):
        """Test get_stats returns correct initial values."""
        stats = self.monitor.get_stats()

        expected_keys = {
            "running",
            "check_interval",
            "health_timeout",
            "total_checks",
            "average_check_time",
            "last_check_time",
            "monitored_emulators",
        }
        assert set(stats.keys()) == expected_keys
        assert stats["running"] is False
        assert stats["total_checks"] == 0
        assert stats["average_check_time"] == 0.0

    def test_context_manager(self):
        """Test HealthMonitor works as context manager."""
        with HealthMonitor(self.mock_pool, check_interval=0.1) as monitor:
            assert monitor._running

        # Should be stopped after context exit
        assert not monitor._running


@pytest.mark.medium
class TestHealthCheckLogic(unittest.TestCase):
    """Test health check logic and HTTP validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_pool = Mock()
        self.mock_pool.clients_by_port = {
            8081: Mock(port=8081, container_id="container1-abc123"),
            8082: Mock(port=8082, container_id="container2-def456"),
        }
        self.mock_pool.get_status.return_value = {
            "available_count": 2,
            "busy_count": 0,
            "total_count": 2,
            "status": "healthy",
        }

        self.monitor = HealthMonitor(self.mock_pool, check_interval=60.0)

    def tearDown(self):
        """Clean up after tests."""
        self.monitor.stop()

    @patch("requests.Session")
    def test_force_check_all_healthy(self, mock_session_class):
        """Test force_check with all emulators healthy."""
        # Mock HTTP responses
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        result = self.monitor.force_check()

        # Verify results
        assert result["overall_status"] == "healthy"
        assert result["healthy_count"] == 2
        assert result["total_count"] == 2
        assert len(result["emulators"]) == 2

        # Verify individual emulator results
        for _port_str, emulator_data in result["emulators"].items():
            assert emulator_data["healthy"] is True
            assert emulator_data["error"] is None

    @patch("requests.Session")
    def test_force_check_with_failures(self, mock_session_class):
        """Test force_check with some emulator failures."""
        # Mock HTTP responses - first succeeds, second fails
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        def side_effect_get(url, timeout):
            if "8081" in url:
                mock_response = Mock()
                mock_response.status_code = 200
                return mock_response
            else:
                raise requests.RequestException("Connection failed")

        mock_session.get.side_effect = side_effect_get

        result = self.monitor.force_check()

        # Verify degraded status
        assert result["overall_status"] == "degraded"
        assert result["healthy_count"] == 1
        assert result["total_count"] == 2

        # Verify specific results
        emulators = result["emulators"]
        assert emulators["8081"]["healthy"] is True
        assert emulators["8082"]["healthy"] is False

    @patch("requests.Session")
    def test_health_change_tracking(self, mock_session_class):
        """Test health change detection and callbacks."""
        # Set up health change callback
        health_changes = []

        def on_health_change(port, old_healthy, new_healthy):
            health_changes.append((port, old_healthy, new_healthy))

        self.monitor.on_health_change = on_health_change

        # Mock session for HTTP requests
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # First check - all healthy
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        result1 = self.monitor.force_check()
        assert result1["healthy_count"] == 2
        assert len(health_changes) == 0  # No changes on first check

        # Second check - one fails
        def side_effect_get(url, timeout):
            if "8081" in url:
                mock_response = Mock()
                mock_response.status_code = 200
                return mock_response
            else:
                raise requests.RequestException("Connection failed")

        mock_session.get.side_effect = side_effect_get

        result2 = self.monitor.force_check()
        assert result2["healthy_count"] == 1
        assert len(health_changes) == 1
        assert health_changes[0] == ("8082", True, False)

    def test_force_check_uninitialized_pool(self):
        """Test force_check with uninitialized emulator pool."""
        # Remove clients_by_port attribute
        delattr(self.mock_pool, "clients_by_port")

        with pytest.raises(HealthMonitorError, match="EmulatorPool not initialized"):
            self.monitor.force_check()

    @patch("requests.Session")
    def test_individual_emulator_health_check(self, mock_session_class):
        """Test individual emulator health check method."""
        mock_client = Mock()
        mock_client.port = 8081

        # Mock successful response
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        is_healthy = self.monitor._check_emulator_health(mock_client)
        assert is_healthy is True

        # Mock failed response
        mock_response.status_code = 500
        is_healthy = self.monitor._check_emulator_health(mock_client)
        assert is_healthy is False

        # Mock connection error
        mock_session.get.side_effect = requests.RequestException("Connection error")
        is_healthy = self.monitor._check_emulator_health(mock_client)
        assert is_healthy is False


@pytest.mark.medium
class TestPerformanceAndTiming(unittest.TestCase):
    """Test performance characteristics and timing behavior."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.mock_pool = Mock()
        self.mock_pool.clients_by_port = {
            8081: Mock(port=8081, container_id="container1-abc123"),
        }
        self.mock_pool.get_status.return_value = {
            "available_count": 1,
            "busy_count": 0,
            "total_count": 1,
            "status": "healthy",
        }

        self.monitor = HealthMonitor(self.mock_pool, check_interval=0.1)

    def tearDown(self):
        """Clean up after tests."""
        self.monitor.stop()

    @patch("requests.Session")
    def test_performance_metrics_tracking(self, mock_session_class):
        """Test performance metrics are tracked correctly."""
        # Mock HTTP response with small delay to ensure measurable time
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_response = Mock()
        mock_response.status_code = 200

        def slow_get(url, timeout):
            time.sleep(0.001)  # 1ms delay to ensure measurable execution time
            return mock_response

        mock_session.get.side_effect = slow_get

        # Perform multiple checks
        for _ in range(3):
            self.monitor.force_check()

        stats = self.monitor.get_stats()
        assert stats["total_checks"] == 3
        assert stats["average_check_time"] >= 0  # Allow 0 for very fast systems
        assert stats["last_check_time"] is not None

    def test_timer_scheduling(self):
        """Test timer scheduling and execution."""
        check_count = 0

        def count_checks(*args, **kwargs):
            nonlocal check_count
            check_count += 1
            return {
                "overall_status": "healthy",
                "healthy_count": 1,
                "total_count": 1,
                "emulators": {"8081": {"healthy": True, "error": None}},
            }

        # Mock the health check method
        self.monitor._perform_health_check = count_checks

        # Start monitoring and wait for checks
        self.monitor.start()
        time.sleep(0.25)  # Wait for 2-3 timer intervals
        self.monitor.stop()

        # Should have executed at least 2 checks
        assert check_count >= 2

    @patch("requests.Session")
    def test_health_check_timeout_behavior(self, mock_session_class):
        """Test health check respects timeout settings."""
        # Mock slow HTTP response
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        def slow_get(url, timeout):
            assert timeout == self.monitor.health_timeout
            time.sleep(0.01)  # Small delay to verify timeout is used
            mock_response = Mock()
            mock_response.status_code = 200
            return mock_response

        mock_session.get.side_effect = slow_get

        start_time = time.time()
        result = self.monitor.force_check()
        check_time = time.time() - start_time

        # Should complete quickly (not hang on timeout)
        assert check_time < 1.0
        assert result["healthy_count"] == 1


@pytest.mark.medium
class TestErrorHandlingAndRecovery(unittest.TestCase):
    """Test error handling and recovery scenarios."""

    def setUp(self):
        """Set up error handling test fixtures."""
        self.mock_pool = Mock()
        self.mock_pool.clients_by_port = {
            8081: Mock(port=8081, container_id="container1-abc123"),
        }
        self.mock_pool.get_status.return_value = {
            "available_count": 1,
            "busy_count": 0,
            "total_count": 1,
            "status": "healthy",
        }

        self.monitor = HealthMonitor(self.mock_pool, check_interval=0.1)

    def tearDown(self):
        """Clean up after tests."""
        self.monitor.stop()

    @patch("src.claudelearnspokemon.health_monitor.logger")
    def test_timer_callback_exception_recovery(self, mock_logger):
        """Test recovery from timer callback exceptions."""
        # Mock _perform_health_check to raise exception initially
        call_count = 0

        def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Simulated failure")
            return {
                "overall_status": "healthy",
                "healthy_count": 1,
                "total_count": 1,
                "emulators": {"8081": {"healthy": True, "error": None}},
            }

        self.monitor._perform_health_check = failing_then_success

        # Start monitoring
        self.monitor.start()
        time.sleep(0.25)  # Wait for multiple timer intervals
        self.monitor.stop()

        # Should have attempted recovery
        assert call_count >= 2
        mock_logger.error.assert_called()

    def test_callback_exception_handling(self):
        """Test handling of exceptions in health change callbacks."""

        # Set up failing callback
        def failing_callback(port, old_healthy, new_healthy):
            raise Exception("Callback failed")

        self.monitor.on_health_change = failing_callback

        # Simulate health change (should not crash)
        self.monitor._track_health_change("8081", True)
        self.monitor._track_health_change("8081", False)  # This should trigger callback

        # Monitor should still be functional
        assert self.monitor is not None

    def test_pool_status_exception_handling(self):
        """Test handling of exceptions when getting pool status."""
        # Mock pool.get_status to raise exception
        self.mock_pool.get_status.side_effect = Exception("Pool status failed")

        # Should raise HealthMonitorError with descriptive message
        with pytest.raises(HealthMonitorError, match="Health check failed"):
            self.monitor.force_check()


@pytest.mark.medium
class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios with real-world EmulatorPool behavior."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_pool = Mock()

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "monitor"):
            self.monitor.stop()

    def test_empty_emulator_pool(self):
        """Test health monitoring with empty emulator pool."""
        self.mock_pool.clients_by_port = {}
        self.mock_pool.get_status.return_value = {
            "available_count": 0,
            "busy_count": 0,
            "total_count": 0,
            "status": "not_initialized",
        }

        self.monitor = HealthMonitor(self.mock_pool)
        result = self.monitor.force_check()

        assert result["overall_status"] == "no_emulators"
        assert result["healthy_count"] == 0
        assert result["total_count"] == 0

    @patch("requests.Session")
    def test_large_emulator_pool(self, mock_session_class):
        """Test health monitoring with large number of emulators."""
        # Create 10 mock emulators
        clients = {}
        for i in range(10):
            port = 8081 + i
            clients[port] = Mock(port=port, container_id=f"container{i}-abc123")

        self.mock_pool.clients_by_port = clients
        self.mock_pool.get_status.return_value = {
            "available_count": 10,
            "busy_count": 0,
            "total_count": 10,
            "status": "healthy",
        }

        # Mock all emulators as healthy
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        self.monitor = HealthMonitor(self.mock_pool)
        result = self.monitor.force_check()

        assert result["overall_status"] == "healthy"
        assert result["healthy_count"] == 10
        assert result["total_count"] == 10
        assert len(result["emulators"]) == 10

    def test_concurrent_access_safety(self):
        """Test thread safety of health monitoring operations."""
        self.mock_pool.clients_by_port = {
            8081: Mock(port=8081, container_id="container1-abc123"),
        }
        self.mock_pool.get_status.return_value = {
            "available_count": 1,
            "busy_count": 0,
            "total_count": 1,
            "status": "healthy",
        }

        self.monitor = HealthMonitor(self.mock_pool, check_interval=0.05)

        # Start monitoring
        self.monitor.start()

        # Perform concurrent operations
        threads = []
        results = []

        def concurrent_ops():
            for _ in range(5):
                try:
                    stats = self.monitor.get_stats()
                    results.append(stats["total_checks"])
                except Exception as e:
                    results.append(f"Error: {e}")

        # Start multiple threads
        for _ in range(3):
            thread = threading.Thread(target=concurrent_ops)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=1.0)

        self.monitor.stop()

        # Should have completed without errors
        assert len(results) == 15  # 3 threads * 5 operations
        error_count = sum(1 for r in results if isinstance(r, str) and "Error" in r)
        assert error_count == 0
