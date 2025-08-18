"""
Integration tests for ContainerAutoRestart with HealthMonitor and EmulatorPool.

Tests the complete auto-restart pipeline from health monitoring through
container replacement in a realistic scenario.

Author: Bot Dean - Workstation Engineering Integration Testing
"""

import time
from unittest.mock import Mock, patch

import pytest

from src.claudelearnspokemon.container_auto_restart import ContainerAutoRestart
from src.claudelearnspokemon.emulator_pool import EmulatorPool
from src.claudelearnspokemon.health_monitor import HealthMonitor


@pytest.mark.fast
@pytest.mark.medium
class TestAutoRestartIntegration:
    """Test complete auto-restart integration scenarios."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.mock_pool = Mock(spec=EmulatorPool)
        self.health_monitor = Mock(spec=HealthMonitor)

        # Create auto-restart component
        self.auto_restart = ContainerAutoRestart(
            emulator_pool=self.mock_pool,
            health_monitor=self.health_monitor,
            max_restart_attempts=2,
            restart_cooldown=0.1,  # Short cooldown for testing
        )

    @pytest.mark.integration
    def test_complete_auto_restart_flow_success(self):
        """Test complete flow: health failure -> auto-restart -> success."""
        port_str = "8081"

        # Mock successful container replacement
        self.mock_pool.replace_failed_container.return_value = True

        # Simulate health monitor detecting failure
        # (This would normally be called by HealthMonitor when container fails)
        self.auto_restart.on_health_change(port_str, True, False)

        # Verify restart was attempted
        self.mock_pool.replace_failed_container.assert_called_once_with(8081)

        # Verify state tracking
        assert self.auto_restart.restart_attempts[port_str] == 1
        assert self.auto_restart.successful_restarts[port_str] == 1
        assert self.auto_restart.total_successful_restarts == 1

        # Simulate container recovery
        self.auto_restart.on_health_change(port_str, False, True)

        # Restart counters should be reset after recovery
        assert port_str not in self.auto_restart.restart_attempts

    @pytest.mark.integration
    def test_complete_auto_restart_flow_with_retries(self):
        """Test auto-restart with multiple failures and eventual success."""
        port_str = "8081"

        # First attempt fails, second succeeds
        self.mock_pool.replace_failed_container.side_effect = [False, True]

        # First failure - restart fails
        self.auto_restart.on_health_change(port_str, True, False)

        assert self.auto_restart.restart_attempts[port_str] == 1
        assert self.auto_restart.failed_restarts[port_str] == 1

        # Wait for cooldown
        time.sleep(0.12)

        # Second failure - simulate another health transition
        # Need to simulate recovery first, then failure again
        self.auto_restart.on_health_change(port_str, False, True)  # Recovery (resets counters)
        self.auto_restart.on_health_change(port_str, True, False)  # Failure again

        # After recovery, restart attempts reset to 1 for the new failure
        assert self.auto_restart.restart_attempts[port_str] == 1
        assert self.auto_restart.successful_restarts[port_str] == 1
        assert self.mock_pool.replace_failed_container.call_count == 2

    @pytest.mark.integration
    def test_auto_restart_reaches_max_attempts(self):
        """Test auto-restart stops after reaching max attempts within a failure session."""
        port_str = "8081"

        # All restart attempts fail
        self.mock_pool.replace_failed_container.return_value = False

        # Initial failure
        self.auto_restart.on_health_change(port_str, True, False)

        # Wait for cooldown and attempt manual second restart to simulate repeated failures
        time.sleep(0.12)
        result = self.auto_restart.attempt_restart(port_str)
        assert result is False  # Should fail

        # At this point we should have 2 failed attempts
        assert self.auto_restart.restart_attempts[port_str] == 2

        # Try a third restart - should be blocked due to max attempts
        time.sleep(0.12)
        result = self.auto_restart.attempt_restart(port_str)
        assert result is False  # Should be blocked

        # Should have attempted only 2 restarts due to max limit
        assert self.mock_pool.replace_failed_container.call_count == 2
        assert self.auto_restart.restart_attempts[port_str] == 2

    @pytest.mark.integration
    def test_health_monitor_integration_callback_registration(self):
        """Test that ContainerAutoRestart properly registers with HealthMonitor."""
        # Verify that the auto_restart callback was set on the health monitor
        assert self.health_monitor.on_health_change == self.auto_restart.on_health_change

    @pytest.mark.integration
    def test_multiple_containers_independent_restart_tracking(self):
        """Test that multiple containers are tracked independently."""
        self.mock_pool.replace_failed_container.return_value = True

        # Container 1 fails
        self.auto_restart.on_health_change("8081", True, False)

        # Container 2 fails
        self.auto_restart.on_health_change("8082", True, False)

        # Both should have restart attempts tracked
        assert self.auto_restart.restart_attempts["8081"] == 1
        assert self.auto_restart.restart_attempts["8082"] == 1

        # Container 1 recovers
        self.auto_restart.on_health_change("8081", False, True)

        # Only container 1 should have counters reset
        assert "8081" not in self.auto_restart.restart_attempts
        assert self.auto_restart.restart_attempts["8082"] == 1

    @pytest.mark.integration
    @patch("src.claudelearnspokemon.container_auto_restart.logger")
    def test_exception_handling_in_integration_flow(self, mock_logger):
        """Test that exceptions in the integration flow are handled gracefully."""
        port_str = "8081"

        # Mock pool method to raise exception
        self.mock_pool.replace_failed_container.side_effect = Exception("Docker error")

        # Should not raise exception despite pool error
        self.auto_restart.on_health_change(port_str, True, False)

        # Should still track the attempt
        assert self.auto_restart.restart_attempts[port_str] == 1
        assert self.auto_restart.failed_restarts[port_str] == 1

    @pytest.mark.integration
    def test_manual_restart_integration(self):
        """Test manual restart functionality in integration scenario."""
        port_str = "8081"

        # Set up some automatic restart history
        self.auto_restart.restart_attempts[port_str] = 2  # At max
        self.auto_restart.last_restart_time[port_str] = time.time()  # In cooldown

        # Mock successful manual restart
        self.mock_pool.replace_failed_container.return_value = True

        # Manual restart should bypass limits
        result = self.auto_restart.manual_restart(port_str, reset_attempts=True)

        assert result is True
        self.mock_pool.replace_failed_container.assert_called_once_with(8081)

        # Counters should be reset
        assert port_str not in self.auto_restart.restart_attempts

    @pytest.mark.integration
    def test_statistics_integration(self):
        """Test statistics collection across multiple integration scenarios."""
        self.mock_pool.replace_failed_container.side_effect = [True, False, True]

        # Multiple container failures and restarts
        self.auto_restart.on_health_change("8081", True, False)  # Success
        time.sleep(0.12)
        self.auto_restart.on_health_change("8082", True, False)  # Failure
        time.sleep(0.12)
        # Third attempt on 8081 - need to simulate recovery first
        self.auto_restart.on_health_change("8081", False, True)  # Recovery
        self.auto_restart.on_health_change("8081", True, False)  # Failure again

        # Check comprehensive statistics
        stats = self.auto_restart.get_restart_stats()

        assert stats["total_restart_attempts"] == 3
        assert stats["total_successful_restarts"] == 2
        assert stats["total_failed_restarts"] == 1
        assert stats["success_rate"] == pytest.approx(66.67, rel=0.1)
        assert stats["containers_with_attempts"] == 2

    @pytest.mark.integration
    def test_disabled_auto_restart_integration(self):
        """Test that disabling auto-restart prevents automatic restarts."""
        port_str = "8081"

        # Disable auto-restart
        self.auto_restart.disable()

        # Container failure should not trigger restart
        self.auto_restart.on_health_change(port_str, True, False)

        # No restart should have been attempted
        self.mock_pool.replace_failed_container.assert_not_called()
        assert port_str not in self.auto_restart.restart_attempts

        # Manual restart should still work
        self.mock_pool.replace_failed_container.return_value = True
        result = self.auto_restart.manual_restart(port_str)

        assert result is True
        self.mock_pool.replace_failed_container.assert_called_once()
