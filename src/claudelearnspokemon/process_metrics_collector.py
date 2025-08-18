"""
Process metrics collection for Claude CLI processes.

This module handles all performance metrics collection and aggregation,
following the Single Responsibility Principle by separating metrics
concerns from process lifecycle management.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcessMetrics:
    """Performance and health metrics for a Claude process."""

    process_id: int
    startup_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_health_check: float = 0.0
    health_check_duration: float = 0.0
    failure_count: int = 0
    restart_count: int = 0
    total_uptime: float = 0.0
    _start_time: float = field(default_factory=time.time, init=False)

    def get_uptime(self) -> float:
        """Calculate current uptime in seconds."""
        return time.time() - self._start_time

    def reset_uptime(self):
        """Reset the uptime counter (typically called on restart)."""
        self._start_time = time.time()


class ProcessMetricsCollector:
    """
    Collects and manages performance metrics for Claude CLI processes.

    This class is responsible for:
    - Recording timing data (startup, health checks, etc.)
    - Tracking process events (failures, restarts)
    - Providing metrics aggregation and reporting
    - Thread-safe metrics updates
    """

    def __init__(self, process_id: int):
        """
        Initialize metrics collector for a specific process.

        Args:
            process_id: Unique identifier for the process
        """
        self.process_id = process_id
        self.metrics = ProcessMetrics(process_id=process_id)
        self._lock = threading.Lock()

    def record_startup_time(self, duration: float):
        """
        Record process startup timing.

        Args:
            duration: Startup time in seconds
        """
        with self._lock:
            self.metrics.startup_time = duration

    def record_health_check(self, duration: float):
        """
        Record health check timing and update last check time.

        Args:
            duration: Health check duration in seconds
        """
        with self._lock:
            self.metrics.health_check_duration = duration
            self.metrics.last_health_check = time.time()

    def record_failure(self):
        """Record a process failure event."""
        with self._lock:
            self.metrics.failure_count += 1

    def record_restart(self):
        """Record a process restart event and reset uptime."""
        with self._lock:
            self.metrics.restart_count += 1
            self.metrics.reset_uptime()

    def update_resource_usage(self, memory_mb: float, cpu_percent: float):
        """
        Update resource usage metrics.

        Args:
            memory_mb: Memory usage in megabytes
            cpu_percent: CPU usage as percentage (0-100)
        """
        with self._lock:
            self.metrics.memory_usage_mb = memory_mb
            self.metrics.cpu_usage_percent = cpu_percent

    def get_metrics_snapshot(self) -> ProcessMetrics:
        """
        Get a thread-safe snapshot of current metrics.

        Returns:
            Copy of current ProcessMetrics
        """
        with self._lock:
            # Create a copy to avoid concurrent modification
            return ProcessMetrics(
                process_id=self.metrics.process_id,
                startup_time=self.metrics.startup_time,
                memory_usage_mb=self.metrics.memory_usage_mb,
                cpu_usage_percent=self.metrics.cpu_usage_percent,
                last_health_check=self.metrics.last_health_check,
                health_check_duration=self.metrics.health_check_duration,
                failure_count=self.metrics.failure_count,
                restart_count=self.metrics.restart_count,
                total_uptime=self.metrics.get_uptime(),
            )

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Get performance metrics formatted for reporting.

        Returns:
            Dictionary with performance metrics in readable format
        """
        snapshot = self.get_metrics_snapshot()

        return {
            "process_id": snapshot.process_id,
            "startup_time_ms": round(snapshot.startup_time * 1000, 2),
            "health_check_time_ms": round(snapshot.health_check_duration * 1000, 2),
            "memory_usage_mb": round(snapshot.memory_usage_mb, 2),
            "cpu_usage_percent": round(snapshot.cpu_usage_percent, 2),
            "failure_count": snapshot.failure_count,
            "restart_count": snapshot.restart_count,
            "uptime_seconds": round(snapshot.total_uptime, 1),
            "last_health_check_ago": round(time.time() - snapshot.last_health_check, 1),
        }

    def is_performing_well(self) -> bool:
        """
        Evaluate if the process is performing within acceptable parameters.

        Returns:
            True if performance is acceptable, False otherwise
        """
        snapshot = self.get_metrics_snapshot()

        # Performance criteria based on original requirements
        startup_ok = snapshot.startup_time < 0.5  # <500ms for strategic, <100ms for tactical
        health_check_ok = snapshot.health_check_duration < 0.01  # <10ms
        memory_ok = snapshot.memory_usage_mb < 100  # <100MB (original was 50MB baseline)
        failure_rate_ok = snapshot.failure_count < 5  # Reasonable failure threshold

        return startup_ok and health_check_ok and memory_ok and failure_rate_ok

    def reset_metrics(self):
        """Reset all metrics to initial state (useful for testing)."""
        with self._lock:
            self.metrics = ProcessMetrics(process_id=self.process_id)


class AggregatedMetricsCollector:
    """
    Aggregates metrics across multiple processes for system-wide reporting.

    This class provides system-level metrics aggregation while maintaining
    individual process metric isolation.
    """

    def __init__(self):
        """Initialize the aggregated metrics collector."""
        self.collectors: dict[int, ProcessMetricsCollector] = {}
        self._lock = threading.Lock()

    def add_collector(self, collector: ProcessMetricsCollector):
        """
        Add a process metrics collector to the aggregation.

        Args:
            collector: ProcessMetricsCollector to include in aggregation
        """
        with self._lock:
            self.collectors[collector.process_id] = collector

    def remove_collector(self, process_id: int):
        """
        Remove a process metrics collector from aggregation.

        Args:
            process_id: ID of the process to remove
        """
        with self._lock:
            self.collectors.pop(process_id, None)

    def get_system_metrics(self) -> dict[str, Any]:
        """
        Get aggregated metrics across all tracked processes.

        Returns:
            Dictionary with system-wide performance metrics
        """
        with self._lock:
            if not self.collectors:
                return {
                    "total_processes": 0,
                    "healthy_processes": 0,
                    "failed_processes": 0,
                    "average_startup_time_ms": 0.0,
                    "average_health_check_time_ms": 0.0,
                    "total_restarts": 0,
                    "total_failures": 0,
                    "process_details": {},
                }

            startup_times = []
            health_check_times = []
            total_restarts = 0
            total_failures = 0
            healthy_count = 0
            process_details = {}

            for collector in self.collectors.values():
                metrics = collector.get_metrics_snapshot()

                startup_times.append(metrics.startup_time)
                health_check_times.append(metrics.health_check_duration)
                total_restarts += metrics.restart_count
                total_failures += metrics.failure_count

                # Consider a process healthy if it's performing well
                if collector.is_performing_well():
                    healthy_count += 1

                process_details[metrics.process_id] = collector.get_performance_summary()

            return {
                "total_processes": len(self.collectors),
                "healthy_processes": healthy_count,
                "failed_processes": len(self.collectors) - healthy_count,
                "average_startup_time_ms": round(sum(startup_times) / len(startup_times) * 1000, 2),
                "average_health_check_time_ms": round(
                    sum(health_check_times) / len(health_check_times) * 1000, 2
                ),
                "total_restarts": total_restarts,
                "total_failures": total_failures,
                "process_details": process_details,
            }
