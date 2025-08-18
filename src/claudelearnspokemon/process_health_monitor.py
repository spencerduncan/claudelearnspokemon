"""
Health monitoring for Claude CLI processes.

This module handles health checking, state management, and backoff timing,
following the Single Responsibility Principle by separating health monitoring
concerns from process lifecycle and communication.
"""

import logging
import subprocess
import threading
import time
from enum import Enum

from .process_metrics_collector import ProcessMetricsCollector

logger = logging.getLogger(__name__)


class ProcessState(Enum):
    """Process lifecycle states."""

    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    TERMINATED = "terminated"


class ProcessHealthMonitor:
    """
    Monitors health and manages state for Claude CLI processes.

    This class is responsible for:
    - Performing health checks on subprocess instances
    - Managing process state transitions
    - Implementing exponential backoff for health check timing
    - Recording health metrics through the metrics collector
    """

    def __init__(self, process_id: int, metrics_collector: ProcessMetricsCollector):
        """
        Initialize health monitor for a specific process.

        Args:
            process_id: Unique identifier for the process
            metrics_collector: Collector for recording health metrics
        """
        self.process_id = process_id
        self.metrics_collector = metrics_collector
        self.state = ProcessState.INITIALIZING
        self._lock = threading.Lock()
        self._health_check_backoff = 1.0  # Start at 1ms, exponential backoff

        # Health check configuration
        self.min_backoff = 0.001  # 1ms minimum
        self.max_backoff = 0.1  # 100ms maximum
        self.excellent_threshold = 0.005  # 5ms is excellent
        self.slow_threshold = 0.010  # 10ms needs backoff
        self.backoff_decrease_factor = 0.9
        self.backoff_increase_factor = 1.1

    def check_health(self, process: subprocess.Popen | None) -> bool:
        """
        Perform health check on the subprocess with performance timing.

        Args:
            process: The subprocess.Popen instance to check

        Returns:
            True if process is healthy, False otherwise
        """
        start_time = time.time()

        try:
            with self._lock:
                if process is None:
                    self._transition_to_state(ProcessState.TERMINATED)
                    return False

                # Check if process is still running
                return_code = process.poll()
                if return_code is not None:
                    logger.warning(f"Process {self.process_id} terminated with code {return_code}")
                    self._transition_to_state(ProcessState.FAILED)
                    return False

                # Record health check timing
                duration = time.time() - start_time
                self.metrics_collector.record_health_check(duration)

                # Adjust backoff timing based on performance
                self._adjust_backoff_timing(duration)

                # If we were previously failed, mark as healthy
                if self.state == ProcessState.FAILED:
                    self._transition_to_state(ProcessState.HEALTHY)

                # Ensure we're in a healthy state if check passed
                if self.state == ProcessState.INITIALIZING:
                    self._transition_to_state(ProcessState.HEALTHY)

                return True

        except Exception as e:
            logger.error(f"Health check failed for process {self.process_id}: {e}")
            self._transition_to_state(ProcessState.DEGRADED)
            return False

    def _adjust_backoff_timing(self, duration: float):
        """
        Adjust exponential backoff based on health check performance.

        Args:
            duration: Duration of the health check in seconds
        """
        if duration < self.excellent_threshold:
            # Excellent performance - reduce backoff
            self._health_check_backoff = max(
                self.min_backoff, self._health_check_backoff * self.backoff_decrease_factor
            )
        elif duration > self.slow_threshold:
            # Slow performance - increase backoff
            self._health_check_backoff = min(
                self.max_backoff, self._health_check_backoff * self.backoff_increase_factor
            )

    def _transition_to_state(self, new_state: ProcessState):
        """
        Transition to a new process state with logging.

        Args:
            new_state: The state to transition to
        """
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            logger.debug(
                f"Process {self.process_id} state transition: {old_state.value} -> {new_state.value}"
            )

            # Record failure events for metrics
            if new_state in [ProcessState.FAILED, ProcessState.DEGRADED]:
                self.metrics_collector.record_failure()

    def get_current_state(self) -> ProcessState:
        """
        Get the current process state in a thread-safe manner.

        Returns:
            Current ProcessState
        """
        with self._lock:
            return self.state

    def get_backoff_delay(self) -> float:
        """
        Get current backoff delay for health checks.

        Returns:
            Current backoff delay in seconds
        """
        with self._lock:
            return self._health_check_backoff

    def is_healthy(self) -> bool:
        """
        Check if process is currently in a healthy state.

        Returns:
            True if process state is HEALTHY
        """
        return self.get_current_state() == ProcessState.HEALTHY

    def is_failed(self) -> bool:
        """
        Check if process is in a failed state.

        Returns:
            True if process state is FAILED or DEGRADED
        """
        state = self.get_current_state()
        return state in [ProcessState.FAILED, ProcessState.DEGRADED]

    def mark_as_healthy(self):
        """Mark the process as healthy (used after successful startup)."""
        with self._lock:
            self._transition_to_state(ProcessState.HEALTHY)

    def mark_as_failed(self):
        """Mark the process as failed (used when startup fails)."""
        with self._lock:
            self._transition_to_state(ProcessState.FAILED)

    def mark_as_terminated(self):
        """Mark the process as terminated (used during shutdown)."""
        with self._lock:
            self._transition_to_state(ProcessState.TERMINATED)

    def reset_backoff(self):
        """Reset backoff timing to initial value (useful for testing)."""
        with self._lock:
            self._health_check_backoff = 1.0


class HealthCheckScheduler:
    """
    Schedules periodic health checks for multiple processes.

    This class provides centralized scheduling of health checks
    while maintaining individual process monitoring isolation.
    """

    def __init__(self, check_interval: float = 5.0):
        """
        Initialize health check scheduler.

        Args:
            check_interval: Default interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.monitors: dict[int, ProcessHealthMonitor] = {}
        self.process_refs: dict[int, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._running = False
        self._scheduler_thread: threading.Thread | None = None

    def add_monitor(self, monitor: ProcessHealthMonitor, process: subprocess.Popen):
        """
        Add a health monitor to the scheduler.

        Args:
            monitor: ProcessHealthMonitor to schedule
            process: subprocess.Popen instance to monitor
        """
        with self._lock:
            self.monitors[monitor.process_id] = monitor
            self.process_refs[monitor.process_id] = process

    def remove_monitor(self, process_id: int):
        """
        Remove a health monitor from the scheduler.

        Args:
            process_id: ID of the process to stop monitoring
        """
        with self._lock:
            self.monitors.pop(process_id, None)
            self.process_refs.pop(process_id, None)

    def start_monitoring(self):
        """Start the background health check scheduler."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._scheduler_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self._scheduler_thread.start()
            logger.info("Health check scheduler started")

    def stop_monitoring(self):
        """Stop the background health check scheduler."""
        with self._lock:
            self._running = False

        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=1.0)

        logger.info("Health check scheduler stopped")

    def _health_check_loop(self):
        """Main loop for periodic health checks."""
        while self._running:
            try:
                with self._lock:
                    monitors_copy = dict(self.monitors)
                    processes_copy = dict(self.process_refs)

                # Perform health checks outside of lock to avoid blocking
                for process_id, monitor in monitors_copy.items():
                    if process_id in processes_copy:
                        process = processes_copy[process_id]
                        monitor.check_health(process)

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(1.0)  # Brief pause before retrying

    def get_health_summary(self) -> dict[int, bool]:
        """
        Get health status for all monitored processes.

        Returns:
            Dictionary mapping process_id to health status
        """
        with self._lock:
            return {
                process_id: monitor.is_healthy() for process_id, monitor in self.monitors.items()
            }
