"""
HealthMonitor: Simple timer-based health monitoring for EmulatorPool containers.

Provides periodic health checks with HTTP connectivity validation and status logging.
Designed for workstation development with fail-fast approach and minimal overhead.

Performance characteristics:
- Timer-based checks every 30 seconds (configurable)
- HTTP ping validation with 3-second timeout
- Minimal resource usage for development environments
- Clean shutdown handling with proper timer cleanup

Author: John Botmack - Performance-First Development
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import requests

# Configure logging for health monitoring
logger = logging.getLogger(__name__)


class HealthMonitorError(Exception):
    """Custom exception for health monitoring operations."""

    pass


class HealthMonitor:
    """
    Timer-based health monitoring system for EmulatorPool containers.

    Implements simple, reliable health checks with minimal resource overhead:
    - Periodic HTTP ping validation
    - Health status logging and reporting
    - Clean timer management and shutdown
    - Integration with existing EmulatorPool infrastructure
    """

    def __init__(
        self,
        emulator_pool: Any,  # EmulatorPool type - avoiding circular import
        check_interval: float = 30.0,
        health_timeout: float = 3.0,
        on_health_change: Callable[[str, bool, bool], None] | None = None,
    ):
        """
        Initialize health monitoring for EmulatorPool.

        Args:
            emulator_pool: EmulatorPool instance to monitor
            check_interval: Seconds between health checks (default: 30s)
            health_timeout: Timeout for individual health checks (default: 3s)
            on_health_change: Optional callback for health state changes
                             Called with (port, old_healthy, new_healthy)
        """
        self.emulator_pool = emulator_pool
        self.check_interval = check_interval
        self.health_timeout = health_timeout
        self.on_health_change = on_health_change

        # Health monitoring state
        self._timer: threading.Timer | None = None
        self._running = False
        self._lock = threading.RLock()  # RLock required: timer callback holds lock during health check

        # Track health history for change detection
        self._last_health_status: dict[str, bool] = {}

        # Performance metrics
        self._check_count = 0
        self._total_check_time = 0.0
        self._last_check_time: float | None = None

        logger.info(
            f"HealthMonitor initialized: interval={check_interval}s, timeout={health_timeout}s"
        )

    def start(self) -> None:
        """
        Start periodic health monitoring.

        Thread-safe operation that can be called multiple times safely.
        """
        with self._lock:
            if self._running:
                logger.warning("HealthMonitor already running - ignoring start request")
                return

            logger.info("Starting health monitoring")
            self._running = True
            self._schedule_next_check()

    def stop(self) -> None:
        """
        Stop health monitoring and clean up resources.

        Thread-safe operation with proper timer cleanup.
        Safe to call multiple times.
        """
        with self._lock:
            if not self._running:
                logger.info("HealthMonitor already stopped")
                return

            logger.info("Stopping health monitoring")
            self._running = False

            # Cancel pending timer
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

            logger.info("Health monitoring stopped")

    def force_check(self) -> dict[str, Any]:
        """
        Perform immediate health check outside of timer schedule.

        Returns:
            Health check results for all emulators

        Raises:
            HealthMonitorError: If emulator pool not initialized
        """
        logger.info("Performing forced health check")
        return self._perform_health_check()

    def get_stats(self) -> dict[str, Any]:
        """
        Get health monitoring performance statistics.

        Returns:
            Dictionary with monitoring metrics and status
        """
        with self._lock:
            avg_check_time = (
                self._total_check_time / self._check_count if self._check_count > 0 else 0.0
            )

            return {
                "running": self._running,
                "check_interval": self.check_interval,
                "health_timeout": self.health_timeout,
                "total_checks": self._check_count,
                "average_check_time": round(avg_check_time, 3),
                "last_check_time": self._last_check_time,
                "monitored_emulators": len(self._last_health_status),
            }

    def _schedule_next_check(self) -> None:
        """Schedule the next health check timer."""
        if not self._running:
            return

        self._timer = threading.Timer(self.check_interval, self._timer_callback)
        self._timer.daemon = True  # Don't block program exit
        self._timer.start()

        logger.debug(f"Next health check scheduled in {self.check_interval}s")

    def _timer_callback(self) -> None:
        """Timer callback that performs health check and reschedules."""
        try:
            with self._lock:
                if not self._running:
                    return

                # Perform health check
                self._perform_health_check()

                # Schedule next check
                self._schedule_next_check()

        except Exception as e:
            logger.error(f"Health check timer callback failed: {e}")

            # Attempt to reschedule if still running
            with self._lock:
                if self._running:
                    logger.info("Attempting to recover from timer callback failure")
                    self._schedule_next_check()

    def _perform_health_check(self) -> dict[str, Any]:
        """
        Perform health check on all emulators with performance tracking.

        Returns:
            Health check results dictionary

        Raises:
            HealthMonitorError: If emulator pool not available
        """
        start_time = time.time()

        try:
            # Get current emulator pool status
            if not hasattr(self.emulator_pool, "clients_by_port"):
                raise HealthMonitorError("EmulatorPool not initialized or invalid")

            pool_status = self.emulator_pool.get_status()

            # Perform individual health checks
            health_results = {}
            healthy_count = 0
            total_count = 0

            for port, client in self.emulator_pool.clients_by_port.items():
                port_str = str(port)
                total_count += 1

                try:
                    # HTTP-based health check with timeout
                    is_healthy = self._check_emulator_health(client)
                    health_results[port_str] = {
                        "healthy": is_healthy,
                        "port": port,
                        "container_id": client.container_id[:12],
                        "error": None,
                    }

                    if is_healthy:
                        healthy_count += 1

                    # Track health changes and notify
                    self._track_health_change(port_str, is_healthy)

                except Exception as e:
                    error_msg = str(e)
                    health_results[port_str] = {
                        "healthy": False,
                        "port": port,
                        "container_id": client.container_id[:12],
                        "error": error_msg,
                    }

                    logger.error(f"Health check failed for emulator on port {port}: {error_msg}")
                    self._track_health_change(port_str, False)

            # Calculate overall health status
            overall_status = "healthy" if healthy_count == total_count else "degraded"
            if total_count == 0:
                overall_status = "no_emulators"

            # Performance tracking
            check_time = time.time() - start_time
            self._update_performance_metrics(check_time)

            # Create comprehensive health report
            health_report = {
                "timestamp": time.time(),
                "overall_status": overall_status,
                "healthy_count": healthy_count,
                "total_count": total_count,
                "check_duration": round(check_time, 3),
                "pool_status": pool_status,
                "emulators": health_results,
            }

            # Log health summary
            self._log_health_summary(health_report)

            return health_report

        except Exception as e:
            check_time = time.time() - start_time
            self._update_performance_metrics(check_time)

            logger.error(f"Health check failed: {e}")
            raise HealthMonitorError(f"Health check failed: {e}") from e

    def _check_emulator_health(self, client: Any) -> bool:
        """
        Check individual emulator health via HTTP ping.

        Args:
            client: PokemonGymClient to health check

        Returns:
            True if emulator is healthy, False otherwise
        """
        try:
            # Use existing health check method with our timeout
            session = requests.Session()
            response = session.get(
                f"http://localhost:{client.port}/health", timeout=self.health_timeout
            )

            return response.status_code == 200

        except requests.RequestException:
            return False
        except Exception as e:
            logger.debug(f"Unexpected error in health check for port {client.port}: {e}")
            return False

    def _track_health_change(self, port_str: str, current_healthy: bool) -> None:
        """
        Track health state changes and trigger callbacks.

        Args:
            port_str: Port as string identifier
            current_healthy: Current health status
        """
        previous_healthy = self._last_health_status.get(port_str)

        if previous_healthy is not None and previous_healthy != current_healthy:
            # Health state changed
            status_change = "recovered" if current_healthy else "failed"
            logger.info(f"Emulator on port {port_str} {status_change}")

            # Trigger callback if provided
            if self.on_health_change:
                try:
                    self.on_health_change(port_str, previous_healthy, current_healthy)
                except Exception as e:
                    logger.error(f"Health change callback failed for port {port_str}: {e}")

        # Update tracked status
        self._last_health_status[port_str] = current_healthy

    def _update_performance_metrics(self, check_time: float) -> None:
        """Update performance tracking metrics."""
        with self._lock:
            self._check_count += 1
            self._total_check_time += check_time
            self._last_check_time = time.time()

    def _log_health_summary(self, health_report: dict[str, Any]) -> None:
        """
        Log concise health summary for operational visibility.

        Args:
            health_report: Complete health check results
        """
        healthy_count = health_report["healthy_count"]
        total_count = health_report["total_count"]
        check_duration = health_report["check_duration"]
        overall_status = health_report["overall_status"]

        # Log level based on health status
        if overall_status == "healthy":
            logger.info(
                f"Health check complete: {healthy_count}/{total_count} healthy "
                f"(duration: {check_duration}s)"
            )
        elif overall_status == "degraded":
            unhealthy_count = total_count - healthy_count
            logger.warning(
                f"Health check DEGRADED: {healthy_count}/{total_count} healthy, "
                f"{unhealthy_count} failing (duration: {check_duration}s)"
            )
        else:
            logger.error(f"Health check FAILED: {overall_status} " f"(duration: {check_duration}s)")

        # Log individual failures for debugging
        for port_str, emulator_data in health_report.get("emulators", {}).items():
            if not emulator_data["healthy"] and emulator_data.get("error"):
                logger.debug(f"Port {port_str} health failure: {emulator_data['error']}")

    def __enter__(self):
        """Context manager entry - start monitoring."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop monitoring."""
        self.stop()
