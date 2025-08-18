"""
ContainerAutoRestart: Basic container auto-restart functionality for EmulatorPool.

Integrates with HealthMonitor to automatically restart failed containers using
workstation-appropriate restart policies with attempt limits and cooldowns.

Designed for reliable operation in development environments with clear logging
and simple error handling patterns.

Author: Bot Dean - Workstation Engineering
"""

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .emulator_pool import EmulatorPool
    from .health_monitor import HealthMonitor
else:
    # Graceful handling for runtime imports
    try:
        from .emulator_pool import EmulatorPool
        from .health_monitor import HealthMonitor
    except ImportError:
        # For testing or when components aren't available
        EmulatorPool = Any
        HealthMonitor = Any

# Configure logging for auto-restart operations
logger = logging.getLogger(__name__)


class ContainerAutoRestartError(Exception):
    """Custom exception for container auto-restart operations."""

    pass


class ContainerAutoRestart:
    """
    Basic container auto-restart functionality for workstation environments.

    Integrates with HealthMonitor to automatically restart failed containers
    using simple, reliable restart policies appropriate for development use:
    - Attempt limits per container to prevent infinite restart loops
    - Cooldown periods between restart attempts
    - Clear logging of all restart operations
    - Integration with EmulatorPool container lifecycle
    """

    def __init__(
        self,
        emulator_pool: "EmulatorPool",
        health_monitor: "HealthMonitor | None" = None,
        max_restart_attempts: int = 3,
        restart_cooldown: float = 30.0,
        enable_auto_restart: bool = True,
    ):
        """
        Initialize container auto-restart functionality.

        Args:
            emulator_pool: EmulatorPool instance to manage
            health_monitor: HealthMonitor to integrate with (optional - can be set later)
            max_restart_attempts: Maximum restart attempts per container (default: 3)
            restart_cooldown: Seconds between restart attempts (default: 30s)
            enable_auto_restart: Whether to enable automatic restarts (default: True)
        """
        self.emulator_pool = emulator_pool
        self.health_monitor = health_monitor
        self.max_restart_attempts = max_restart_attempts
        self.restart_cooldown = restart_cooldown
        self.enabled = enable_auto_restart

        # Restart attempt tracking per container
        self.restart_attempts: dict[str, int] = {}  # port_str -> attempt_count
        self.last_restart_time: dict[str, float] = {}  # port_str -> timestamp
        self.successful_restarts: dict[str, int] = {}  # port_str -> success_count
        self.failed_restarts: dict[str, int] = {}  # port_str -> failure_count

        # Performance and monitoring metrics
        self.total_restart_attempts = 0
        self.total_successful_restarts = 0
        self.total_failed_restarts = 0

        # Register with health monitor if provided
        if self.health_monitor:
            self.register_with_health_monitor()

        logger.info(
            f"ContainerAutoRestart initialized: max_attempts={max_restart_attempts}, "
            f"cooldown={restart_cooldown}s, enabled={enable_auto_restart}"
        )

    def register_with_health_monitor(self) -> None:
        """
        Register this component as the health change callback for the health monitor.

        Raises:
            ContainerAutoRestartError: If health monitor is not available
        """
        if not self.health_monitor:
            raise ContainerAutoRestartError("Health monitor not available for registration")

        # Set ourselves as the health change callback
        self.health_monitor.on_health_change = self.on_health_change
        logger.info("ContainerAutoRestart registered with HealthMonitor")

    def set_health_monitor(self, health_monitor: "HealthMonitor") -> None:
        """
        Set the health monitor and register for callbacks.

        Args:
            health_monitor: HealthMonitor instance to integrate with
        """
        self.health_monitor = health_monitor
        self.register_with_health_monitor()

    def on_health_change(self, port_str: str, old_healthy: bool, new_healthy: bool) -> None:
        """
        Health change callback from HealthMonitor.

        Called when container health status changes. Implements auto-restart logic
        for containers that become unhealthy.

        Args:
            port_str: Port identifier as string
            old_healthy: Previous health status
            new_healthy: Current health status
        """
        try:
            if new_healthy and not old_healthy:
                # Container recovered - reset restart tracking
                logger.info(f"Container on port {port_str} recovered, resetting restart counters")
                self.restart_attempts.pop(port_str, None)
                self.last_restart_time.pop(port_str, None)

            elif not new_healthy and old_healthy:
                # Container failed - attempt restart if enabled
                if self.enabled:
                    logger.warning(f"Container on port {port_str} failed, attempting auto-restart")
                    self.attempt_restart(port_str)
                else:
                    logger.info(f"Container on port {port_str} failed but auto-restart disabled")

        except Exception as e:
            logger.error(f"Error in health change callback for port {port_str}: {e}")

    def attempt_restart(self, port_str: str) -> bool:
        """
        Attempt to restart a failed container with policy enforcement.

        Implements restart attempt limits, cooldown periods, and proper error handling
        appropriate for workstation development environments.

        Args:
            port_str: Port identifier as string

        Returns:
            True if restart was attempted and succeeded, False otherwise
        """
        if not self.enabled:
            logger.info(f"Auto-restart disabled, skipping restart for port {port_str}")
            return False

        port = int(port_str)
        current_time = time.time()

        # Check restart attempt limits
        attempts = self.restart_attempts.get(port_str, 0)
        if attempts >= self.max_restart_attempts:
            logger.error(
                f"Max restart attempts ({self.max_restart_attempts}) reached for "
                f"container on port {port}. Manual intervention required."
            )
            return False

        # Check cooldown period
        last_restart = self.last_restart_time.get(port_str, 0)
        time_since_last = current_time - last_restart
        if time_since_last < self.restart_cooldown:
            remaining_cooldown = self.restart_cooldown - time_since_last
            logger.warning(
                f"Restart cooldown active for port {port} "
                f"({remaining_cooldown:.1f}s remaining), skipping restart"
            )
            return False

        # Attempt container restart
        attempt_num = attempts + 1
        logger.info(
            f"Attempting restart of container on port {port} "
            f"(attempt {attempt_num}/{self.max_restart_attempts})"
        )

        # Update tracking before attempt
        self.restart_attempts[port_str] = attempt_num
        self.last_restart_time[port_str] = current_time
        self.total_restart_attempts += 1

        try:
            # Use EmulatorPool's container replacement functionality
            success = self.emulator_pool.replace_failed_container(port)

            if success:
                # Restart succeeded
                self.successful_restarts[port_str] = self.successful_restarts.get(port_str, 0) + 1
                self.total_successful_restarts += 1

                logger.info(
                    f"Successfully restarted container on port {port} "
                    f"(attempt {attempt_num}/{self.max_restart_attempts})"
                )
                return True
            else:
                # Restart failed
                self.failed_restarts[port_str] = self.failed_restarts.get(port_str, 0) + 1
                self.total_failed_restarts += 1

                logger.error(
                    f"Failed to restart container on port {port} "
                    f"(attempt {attempt_num}/{self.max_restart_attempts}). "
                    f"Will retry after cooldown if attempts remain."
                )
                return False

        except Exception as e:
            # Unexpected error during restart
            self.failed_restarts[port_str] = self.failed_restarts.get(port_str, 0) + 1
            self.total_failed_restarts += 1

            logger.error(
                f"Exception during container restart on port {port}: {e} "
                f"(attempt {attempt_num}/{self.max_restart_attempts})"
            )
            return False

    def manual_restart(self, port_str: str, reset_attempts: bool = True) -> bool:
        """
        Manually trigger a container restart, bypassing normal policies.

        Useful for development and debugging scenarios where you need to force
        a restart regardless of attempt limits or cooldowns.

        Args:
            port_str: Port identifier as string
            reset_attempts: Whether to reset attempt counters (default: True)

        Returns:
            True if restart succeeded, False otherwise
        """
        port = int(port_str)

        if reset_attempts:
            # Reset counters for fresh start
            self.restart_attempts.pop(port_str, None)
            self.last_restart_time.pop(port_str, None)
            logger.info(f"Reset restart counters for port {port} before manual restart")

        logger.info(f"Manual restart requested for container on port {port}")

        try:
            success = self.emulator_pool.replace_failed_container(port)

            if success:
                self.successful_restarts[port_str] = self.successful_restarts.get(port_str, 0) + 1
                self.total_successful_restarts += 1
                logger.info(f"Manual restart successful for container on port {port}")
            else:
                self.failed_restarts[port_str] = self.failed_restarts.get(port_str, 0) + 1
                self.total_failed_restarts += 1
                logger.error(f"Manual restart failed for container on port {port}")

            return success

        except Exception as e:
            self.failed_restarts[port_str] = self.failed_restarts.get(port_str, 0) + 1
            self.total_failed_restarts += 1
            logger.error(f"Exception during manual restart on port {port}: {e}")
            return False

    def reset_restart_counters(self, port_str: str | None = None) -> None:
        """
        Reset restart attempt counters for debugging and recovery.

        Args:
            port_str: Specific port to reset, or None to reset all ports
        """
        if port_str:
            # Reset specific port
            self.restart_attempts.pop(port_str, None)
            self.last_restart_time.pop(port_str, None)
            logger.info(f"Reset restart counters for port {port_str}")
        else:
            # Reset all ports
            self.restart_attempts.clear()
            self.last_restart_time.clear()
            logger.info("Reset restart counters for all ports")

    def get_restart_stats(self) -> dict[str, Any]:
        """
        Get comprehensive restart statistics for monitoring and debugging.

        Returns:
            Dictionary with restart statistics and current state
        """
        return {
            "enabled": self.enabled,
            "max_restart_attempts": self.max_restart_attempts,
            "restart_cooldown": self.restart_cooldown,
            "total_restart_attempts": self.total_restart_attempts,
            "total_successful_restarts": self.total_successful_restarts,
            "total_failed_restarts": self.total_failed_restarts,
            "success_rate": (
                (self.total_successful_restarts / self.total_restart_attempts * 100)
                if self.total_restart_attempts > 0
                else 0.0
            ),
            "containers_with_attempts": len(self.restart_attempts),
            "current_restart_attempts": dict(self.restart_attempts),
            "successful_restarts_by_port": dict(self.successful_restarts),
            "failed_restarts_by_port": dict(self.failed_restarts),
        }

    def get_container_status(self, port_str: str) -> dict[str, Any]:
        """
        Get restart status for a specific container.

        Args:
            port_str: Port identifier as string

        Returns:
            Dictionary with container-specific restart status
        """
        current_time = time.time()
        last_restart = self.last_restart_time.get(port_str, 0)
        time_since_last = current_time - last_restart if last_restart > 0 else None

        return {
            "port": port_str,
            "restart_attempts": self.restart_attempts.get(port_str, 0),
            "max_attempts": self.max_restart_attempts,
            "attempts_remaining": max(
                0, self.max_restart_attempts - self.restart_attempts.get(port_str, 0)
            ),
            "successful_restarts": self.successful_restarts.get(port_str, 0),
            "failed_restarts": self.failed_restarts.get(port_str, 0),
            "last_restart_time": last_restart if last_restart > 0 else None,
            "time_since_last_restart": time_since_last,
            "cooldown_remaining": (
                max(0, self.restart_cooldown - time_since_last)
                if time_since_last is not None and time_since_last < self.restart_cooldown
                else 0
            ),
            "can_restart": (
                self.enabled
                and self.restart_attempts.get(port_str, 0) < self.max_restart_attempts
                and (time_since_last is None or time_since_last >= self.restart_cooldown)
            ),
        }

    def enable(self) -> None:
        """Enable automatic container restarts."""
        self.enabled = True
        logger.info("ContainerAutoRestart enabled")

    def disable(self) -> None:
        """Disable automatic container restarts."""
        self.enabled = False
        logger.info("ContainerAutoRestart disabled")

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"ContainerAutoRestart(enabled={self.enabled}, "
            f"max_attempts={self.max_restart_attempts}, "
            f"cooldown={self.restart_cooldown}s, "
            f"total_attempts={self.total_restart_attempts})"
        )
