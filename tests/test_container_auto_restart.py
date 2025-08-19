"""
Tests for ContainerAutoRestart - Basic container auto-restart functionality.

Comprehensive test suite covering:
- Health change callback integration
- Restart attempt limiting and cooldown enforcement
- Manual restart operations
- Statistics and monitoring
- Error handling and edge cases
- Integration with EmulatorPool and HealthMonitor

Author: Bot Dean - Workstation Engineering Testing
"""

import time
import unittest
from unittest.mock import Mock

import pytest

from src.claudelearnspokemon.container_auto_restart import (
    ContainerAutoRestart,
    ContainerAutoRestartError,
)


@pytest.mark.fast
@pytest.mark.medium
class TestContainerAutoRestartBasics(unittest.TestCase):
    """Test basic ContainerAutoRestart functionality and lifecycle."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_pool = Mock()
        self.mock_monitor = Mock()

        self.auto_restart = ContainerAutoRestart(
            emulator_pool=self.mock_pool,
            health_monitor=self.mock_monitor,
            max_restart_attempts=3,
            restart_cooldown=1.0,  # Short cooldown for testing
        )

    def test_initialization_with_health_monitor(self):
        """Test ContainerAutoRestart initializes correctly with health monitor."""
        assert self.auto_restart.emulator_pool == self.mock_pool
        assert self.auto_restart.health_monitor == self.mock_monitor
        assert self.auto_restart.max_restart_attempts == 3
        assert self.auto_restart.restart_cooldown == 1.0
        assert self.auto_restart.enabled is True

        # Should have registered as health change callback
        assert self.mock_monitor.on_health_change == self.auto_restart.on_health_change

    def test_initialization_without_health_monitor(self):
        """Test initialization without health monitor."""
        auto_restart = ContainerAutoRestart(
            emulator_pool=self.mock_pool,
            health_monitor=None,
        )

        assert auto_restart.health_monitor is None
        assert auto_restart.enabled is True

    def test_initialization_disabled(self):
        """Test initialization with auto-restart disabled."""
        auto_restart = ContainerAutoRestart(
            emulator_pool=self.mock_pool,
            enable_auto_restart=False,
        )

        assert auto_restart.enabled is False

    def test_set_health_monitor(self):
        """Test setting health monitor after initialization."""
        auto_restart = ContainerAutoRestart(
            emulator_pool=self.mock_pool,
            health_monitor=None,
        )

        new_monitor = Mock()
        auto_restart.set_health_monitor(new_monitor)

        assert auto_restart.health_monitor == new_monitor
        assert new_monitor.on_health_change == auto_restart.on_health_change

    def test_register_with_health_monitor_no_monitor(self):
        """Test registration fails when no health monitor available."""
        auto_restart = ContainerAutoRestart(
            emulator_pool=self.mock_pool,
            health_monitor=None,
        )

        with pytest.raises(ContainerAutoRestartError, match="Health monitor not available"):
            auto_restart.register_with_health_monitor()

    def test_enable_disable(self):
        """Test enable/disable functionality."""
        assert self.auto_restart.enabled is True

        self.auto_restart.disable()
        assert self.auto_restart.enabled is False

        self.auto_restart.enable()
        assert self.auto_restart.enabled is True


@pytest.mark.fast
@pytest.mark.medium
class TestHealthChangeIntegration(unittest.TestCase):
    """Test integration with HealthMonitor health change callbacks."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_pool = Mock()
        self.mock_monitor = Mock()

        self.auto_restart = ContainerAutoRestart(
            emulator_pool=self.mock_pool,
            health_monitor=self.mock_monitor,
            max_restart_attempts=3,
            restart_cooldown=0.1,  # Very short cooldown for testing
        )

    def test_health_change_container_recovery(self):
        """Test health change callback handles container recovery."""
        port_str = "8081"

        # Set up some restart attempts
        self.auto_restart.restart_attempts[port_str] = 2
        self.auto_restart.last_restart_time[port_str] = time.time()

        # Container recovers (False -> True)
        self.auto_restart.on_health_change(port_str, False, True)

        # Restart counters should be reset
        assert port_str not in self.auto_restart.restart_attempts
        assert port_str not in self.auto_restart.last_restart_time

    def test_health_change_container_failure_enabled(self):
        """Test health change callback handles container failure when enabled."""
        port_str = "8081"
        self.mock_pool.replace_failed_container.return_value = True

        # Container fails (True -> False)
        self.auto_restart.on_health_change(port_str, True, False)

        # Should have attempted restart
        self.mock_pool.replace_failed_container.assert_called_once_with(8081)

        # Should track the attempt
        assert self.auto_restart.restart_attempts[port_str] == 1
        assert port_str in self.auto_restart.last_restart_time

    def test_health_change_container_failure_disabled(self):
        """Test health change callback when auto-restart is disabled."""
        port_str = "8081"
        self.auto_restart.disable()

        # Container fails (True -> False)
        self.auto_restart.on_health_change(port_str, True, False)

        # Should not have attempted restart
        self.mock_pool.replace_failed_container.assert_not_called()

        # Should not track attempts
        assert port_str not in self.auto_restart.restart_attempts

    def test_health_change_callback_exception_handling(self):
        """Test health change callback handles exceptions gracefully."""
        port_str = "8081"
        self.mock_pool.replace_failed_container.side_effect = Exception("Test error")

        # Should not raise exception
        self.auto_restart.on_health_change(port_str, True, False)

        # Attempt should still be recorded
        assert self.auto_restart.restart_attempts[port_str] == 1


@pytest.mark.fast
@pytest.mark.medium
class TestRestartLogic(unittest.TestCase):
    """Test restart attempt logic and policy enforcement."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_pool = Mock()
        self.auto_restart = ContainerAutoRestart(
            emulator_pool=self.mock_pool,
            max_restart_attempts=3,
            restart_cooldown=0.1,  # Short for testing
        )

    def test_attempt_restart_success(self):
        """Test successful restart attempt."""
        port_str = "8081"
        self.mock_pool.replace_failed_container.return_value = True

        result = self.auto_restart.attempt_restart(port_str)

        assert result is True
        assert self.auto_restart.restart_attempts[port_str] == 1
        assert self.auto_restart.successful_restarts[port_str] == 1
        assert self.auto_restart.total_successful_restarts == 1
        assert port_str in self.auto_restart.last_restart_time

    def test_attempt_restart_failure(self):
        """Test failed restart attempt."""
        port_str = "8081"
        self.mock_pool.replace_failed_container.return_value = False

        result = self.auto_restart.attempt_restart(port_str)

        assert result is False
        assert self.auto_restart.restart_attempts[port_str] == 1
        assert self.auto_restart.failed_restarts[port_str] == 1
        assert self.auto_restart.total_failed_restarts == 1

    def test_attempt_restart_max_attempts_reached(self):
        """Test restart is blocked when max attempts reached."""
        port_str = "8081"
        self.auto_restart.restart_attempts[port_str] = 3  # At limit

        result = self.auto_restart.attempt_restart(port_str)

        assert result is False
        self.mock_pool.replace_failed_container.assert_not_called()
        # Attempt count shouldn't increase
        assert self.auto_restart.restart_attempts[port_str] == 3

    def test_attempt_restart_cooldown_active(self):
        """Test restart is blocked during cooldown period."""
        port_str = "8081"
        self.auto_restart.last_restart_time[port_str] = time.time()  # Just restarted

        result = self.auto_restart.attempt_restart(port_str)

        assert result is False
        self.mock_pool.replace_failed_container.assert_not_called()

    def test_attempt_restart_after_cooldown(self):
        """Test restart is allowed after cooldown expires."""
        port_str = "8081"
        self.auto_restart.restart_attempts[port_str] = 1  # Has previous attempts
        self.auto_restart.last_restart_time[port_str] = time.time() - 0.2  # Past cooldown
        self.mock_pool.replace_failed_container.return_value = True

        result = self.auto_restart.attempt_restart(port_str)

        assert result is True
        assert self.auto_restart.restart_attempts[port_str] == 2  # Incremented

    def test_attempt_restart_disabled(self):
        """Test restart is blocked when auto-restart disabled."""
        port_str = "8081"
        self.auto_restart.disable()

        result = self.auto_restart.attempt_restart(port_str)

        assert result is False
        self.mock_pool.replace_failed_container.assert_not_called()

    def test_attempt_restart_exception_handling(self):
        """Test restart handles exceptions from pool operations."""
        port_str = "8081"
        self.mock_pool.replace_failed_container.side_effect = Exception("Test error")

        result = self.auto_restart.attempt_restart(port_str)

        assert result is False
        assert self.auto_restart.failed_restarts[port_str] == 1
        assert self.auto_restart.total_failed_restarts == 1


@pytest.mark.fast
@pytest.mark.medium
class TestManualRestart(unittest.TestCase):
    """Test manual restart functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_pool = Mock()
        self.auto_restart = ContainerAutoRestart(
            emulator_pool=self.mock_pool,
            max_restart_attempts=3,
            restart_cooldown=10.0,  # Long cooldown to test bypass
        )

    def test_manual_restart_success(self):
        """Test successful manual restart."""
        port_str = "8081"
        self.mock_pool.replace_failed_container.return_value = True

        result = self.auto_restart.manual_restart(port_str)

        assert result is True
        assert self.auto_restart.successful_restarts[port_str] == 1
        assert self.auto_restart.total_successful_restarts == 1

    def test_manual_restart_bypasses_cooldown(self):
        """Test manual restart bypasses cooldown period."""
        port_str = "8081"
        self.auto_restart.last_restart_time[port_str] = time.time()  # Just restarted
        self.mock_pool.replace_failed_container.return_value = True

        result = self.auto_restart.manual_restart(port_str)

        # Should succeed despite recent restart
        assert result is True
        self.mock_pool.replace_failed_container.assert_called_once()

    def test_manual_restart_bypasses_attempt_limit(self):
        """Test manual restart bypasses attempt limit."""
        port_str = "8081"
        self.auto_restart.restart_attempts[port_str] = 3  # At limit
        self.mock_pool.replace_failed_container.return_value = True

        result = self.auto_restart.manual_restart(port_str, reset_attempts=True)

        # Should succeed despite being at attempt limit
        assert result is True
        self.mock_pool.replace_failed_container.assert_called_once()

        # Attempt counter should be reset
        assert port_str not in self.auto_restart.restart_attempts

    def test_manual_restart_no_reset_attempts(self):
        """Test manual restart without resetting attempt counters."""
        port_str = "8081"
        self.auto_restart.restart_attempts[port_str] = 2
        self.mock_pool.replace_failed_container.return_value = True

        result = self.auto_restart.manual_restart(port_str, reset_attempts=False)

        assert result is True
        # Attempt counter should remain
        assert self.auto_restart.restart_attempts[port_str] == 2

    def test_manual_restart_failure(self):
        """Test failed manual restart."""
        port_str = "8081"
        self.mock_pool.replace_failed_container.return_value = False

        result = self.auto_restart.manual_restart(port_str)

        assert result is False
        assert self.auto_restart.failed_restarts[port_str] == 1


@pytest.mark.fast
@pytest.mark.medium
class TestCounterManagement(unittest.TestCase):
    """Test restart counter management and reset functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_pool = Mock()
        self.auto_restart = ContainerAutoRestart(self.mock_pool)

    def test_reset_restart_counters_specific_port(self):
        """Test resetting counters for specific port."""
        # Set up some counters
        self.auto_restart.restart_attempts["8081"] = 2
        self.auto_restart.restart_attempts["8082"] = 1
        self.auto_restart.last_restart_time["8081"] = time.time()
        self.auto_restart.last_restart_time["8082"] = time.time()

        # Reset specific port
        self.auto_restart.reset_restart_counters("8081")

        # Only 8081 should be reset
        assert "8081" not in self.auto_restart.restart_attempts
        assert "8081" not in self.auto_restart.last_restart_time
        assert self.auto_restart.restart_attempts["8082"] == 1
        assert "8082" in self.auto_restart.last_restart_time

    def test_reset_restart_counters_all_ports(self):
        """Test resetting counters for all ports."""
        # Set up some counters
        self.auto_restart.restart_attempts["8081"] = 2
        self.auto_restart.restart_attempts["8082"] = 1
        self.auto_restart.last_restart_time["8081"] = time.time()
        self.auto_restart.last_restart_time["8082"] = time.time()

        # Reset all ports
        self.auto_restart.reset_restart_counters()

        # All should be reset
        assert len(self.auto_restart.restart_attempts) == 0
        assert len(self.auto_restart.last_restart_time) == 0


@pytest.mark.fast
@pytest.mark.medium
class TestStatisticsAndMonitoring(unittest.TestCase):
    """Test statistics collection and monitoring functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_pool = Mock()
        self.auto_restart = ContainerAutoRestart(
            emulator_pool=self.mock_pool,
            max_restart_attempts=3,
            restart_cooldown=30.0,
        )

    def test_get_restart_stats_initial(self):
        """Test restart statistics initial state."""
        stats = self.auto_restart.get_restart_stats()

        expected_keys = {
            "enabled",
            "max_restart_attempts",
            "restart_cooldown",
            "total_restart_attempts",
            "total_successful_restarts",
            "total_failed_restarts",
            "success_rate",
            "containers_with_attempts",
            "current_restart_attempts",
            "successful_restarts_by_port",
            "failed_restarts_by_port",
        }
        assert set(stats.keys()) == expected_keys
        assert stats["enabled"] is True
        assert stats["max_restart_attempts"] == 3
        assert stats["restart_cooldown"] == 30.0
        assert stats["total_restart_attempts"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_restart_stats_with_activity(self):
        """Test restart statistics with some activity."""
        # Simulate some restart activity
        self.auto_restart.restart_attempts["8081"] = 2
        self.auto_restart.successful_restarts["8081"] = 1
        self.auto_restart.failed_restarts["8081"] = 1
        self.auto_restart.total_restart_attempts = 4
        self.auto_restart.total_successful_restarts = 3
        self.auto_restart.total_failed_restarts = 1

        stats = self.auto_restart.get_restart_stats()

        assert stats["total_restart_attempts"] == 4
        assert stats["total_successful_restarts"] == 3
        assert stats["total_failed_restarts"] == 1
        assert stats["success_rate"] == 75.0
        assert stats["containers_with_attempts"] == 1
        assert stats["current_restart_attempts"]["8081"] == 2

    def test_get_container_status_no_history(self):
        """Test container status for container with no restart history."""
        status = self.auto_restart.get_container_status("8081")

        expected_keys = {
            "port",
            "restart_attempts",
            "max_attempts",
            "attempts_remaining",
            "successful_restarts",
            "failed_restarts",
            "last_restart_time",
            "time_since_last_restart",
            "cooldown_remaining",
            "can_restart",
        }
        assert set(status.keys()) == expected_keys
        assert status["port"] == "8081"
        assert status["restart_attempts"] == 0
        assert status["attempts_remaining"] == 3
        assert status["last_restart_time"] is None
        assert status["can_restart"] is True

    def test_get_container_status_with_history(self):
        """Test container status for container with restart history."""
        port_str = "8081"
        current_time = time.time()

        # Set up some history
        self.auto_restart.restart_attempts[port_str] = 2
        self.auto_restart.successful_restarts[port_str] = 1
        self.auto_restart.failed_restarts[port_str] = 1
        self.auto_restart.last_restart_time[port_str] = current_time - 10  # 10 seconds ago

        status = self.auto_restart.get_container_status(port_str)

        assert status["restart_attempts"] == 2
        assert status["attempts_remaining"] == 1  # 3 - 2
        assert status["successful_restarts"] == 1
        assert status["failed_restarts"] == 1
        assert status["last_restart_time"] == current_time - 10
        assert abs(status["time_since_last_restart"] - 10) < 1  # Allow small timing variance
        assert abs(status["cooldown_remaining"] - 20) < 1  # Allow small timing variance
        assert status["can_restart"] is False  # Still in cooldown

    def test_get_container_status_cooldown_expired(self):
        """Test container status when cooldown has expired."""
        port_str = "8081"

        # Set up expired cooldown
        self.auto_restart.restart_attempts[port_str] = 1
        self.auto_restart.last_restart_time[port_str] = time.time() - 40  # 40 seconds ago

        status = self.auto_restart.get_container_status(port_str)

        assert status["cooldown_remaining"] == 0
        assert status["can_restart"] is True

    def test_get_container_status_max_attempts_reached(self):
        """Test container status when max attempts reached."""
        port_str = "8081"

        # Set up max attempts
        self.auto_restart.restart_attempts[port_str] = 3
        self.auto_restart.last_restart_time[port_str] = time.time() - 40  # Past cooldown

        status = self.auto_restart.get_container_status(port_str)

        assert status["attempts_remaining"] == 0
        assert status["can_restart"] is False  # Max attempts reached

    def test_str_representation(self):
        """Test string representation for debugging."""
        str_repr = str(self.auto_restart)

        assert "ContainerAutoRestart" in str_repr
        assert "enabled=True" in str_repr
        assert "max_attempts=3" in str_repr
        assert "cooldown=30.0s" in str_repr
        assert "total_attempts=0" in str_repr


@pytest.mark.fast
@pytest.mark.medium
class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_pool = Mock()
        self.auto_restart = ContainerAutoRestart(self.mock_pool)

    def test_invalid_port_string(self):
        """Test handling of invalid port strings."""
        # Should handle gracefully without crashing
        try:
            self.auto_restart.on_health_change("invalid_port", True, False)
            # Should not raise exception, but restart won't work
        except Exception as e:
            # If it does raise, should be a reasonable error
            assert "invalid literal" in str(e).lower()

    def test_negative_cooldown(self):
        """Test initialization with negative cooldown."""
        auto_restart = ContainerAutoRestart(
            emulator_pool=self.mock_pool,
            restart_cooldown=-5.0,
        )
        # Should initialize but cooldown logic may behave unexpectedly
        assert auto_restart.restart_cooldown == -5.0

    def test_zero_max_attempts(self):
        """Test initialization with zero max attempts."""
        auto_restart = ContainerAutoRestart(
            emulator_pool=self.mock_pool,
            max_restart_attempts=0,
        )

        # Should never be able to restart
        result = auto_restart.attempt_restart("8081")
        assert result is False
