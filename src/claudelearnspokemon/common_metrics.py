"""
Common metrics recording utilities for consistent performance monitoring.

This module provides standardized metrics collection patterns to reduce code duplication
and ensure consistent monitoring across the Pokemon speedrun learning system.
"""

import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MetricType(Enum):
    """Types of metrics supported."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Container for metric values with metadata."""

    value: int | float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ThreadSafeMetrics:
    """
    Thread-safe metrics collection class with common patterns.

    Provides standardized methods for recording counters, gauges, timers,
    and histograms with automatic thread safety and aggregation.
    """

    def __init__(self, component_name: str, max_histogram_samples: int = 1000):
        self.component_name = component_name
        self.max_histogram_samples = max_histogram_samples

        # Thread-safe storage
        self._lock = threading.RLock()

        # Different metric types
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._timers: dict[str, list[float]] = defaultdict(list)
        self._histograms: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_histogram_samples)
        )

        # Metadata storage
        self._metric_metadata: dict[str, dict[str, Any]] = {}

        # Derived metrics for performance
        self._timer_stats: dict[str, dict[str, float]] = {}

    def increment_counter(self, name: str, value: int = 1, **metadata) -> None:
        """
        Increment a counter metric.

        Args:
            name: Counter name
            value: Amount to increment by
            **metadata: Additional metadata to store
        """
        with self._lock:
            self._counters[name] += value
            if metadata:
                self._metric_metadata[f"counter_{name}"] = metadata

    def set_gauge(self, name: str, value: int | float, **metadata) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Gauge name
            value: Current value
            **metadata: Additional metadata to store
        """
        with self._lock:
            self._gauges[name] = float(value)
            if metadata:
                self._metric_metadata[f"gauge_{name}"] = metadata

    def record_time(self, name: str, duration_ms: float, **metadata) -> None:
        """
        Record a timing measurement.

        Args:
            name: Timer name
            duration_ms: Duration in milliseconds
            **metadata: Additional metadata to store
        """
        with self._lock:
            self._timers[name].append(duration_ms)

            # Keep only recent samples for performance
            if len(self._timers[name]) > self.max_histogram_samples:
                self._timers[name] = self._timers[name][-self.max_histogram_samples :]

            # Update derived statistics
            self._update_timer_stats(name)

            if metadata:
                self._metric_metadata[f"timer_{name}"] = metadata

    def record_histogram_value(self, name: str, value: int | float, **metadata) -> None:
        """
        Record a value in a histogram.

        Args:
            name: Histogram name
            value: Value to record
            **metadata: Additional metadata to store
        """
        with self._lock:
            self._histograms[name].append(float(value))
            if metadata:
                self._metric_metadata[f"histogram_{name}"] = metadata

    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        with self._lock:
            return self._counters[name]

    def get_gauge(self, name: str) -> float | None:
        """Get current gauge value."""
        with self._lock:
            return self._gauges.get(name)

    def get_timer_stats(self, name: str) -> dict[str, float] | None:
        """Get timer statistics."""
        with self._lock:
            return self._timer_stats.get(name, {}).copy()

    def get_histogram_stats(self, name: str) -> dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            values = list(self._histograms.get(name, []))
            if not values:
                return {}

            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "p95": self._percentile(values, 0.95),
                "p99": self._percentile(values, 0.99),
            }

    def get_all_metrics(self) -> dict[str, Any]:
        """Get snapshot of all metrics."""
        with self._lock:
            return {
                "component": self.component_name,
                "timestamp": time.time(),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timer_stats": {name: stats.copy() for name, stats in self._timer_stats.items()},
                "histogram_stats": {
                    name: self.get_histogram_stats(name) for name in self._histograms.keys()
                },
                "metadata": self._metric_metadata.copy(),
            }

    def reset_metrics(self, metric_names: list[str] | None = None) -> None:
        """
        Reset specific metrics or all metrics.

        Args:
            metric_names: List of metric names to reset (all if None)
        """
        with self._lock:
            if metric_names is None:
                # Reset all metrics
                self._counters.clear()
                self._gauges.clear()
                self._timers.clear()
                self._histograms.clear()
                self._timer_stats.clear()
                self._metric_metadata.clear()
            else:
                # Reset specific metrics
                for name in metric_names:
                    self._counters.pop(name, None)
                    self._gauges.pop(name, None)
                    if name in self._timers:
                        self._timers[name].clear()
                        self._timer_stats.pop(name, None)
                    if name in self._histograms:
                        self._histograms[name].clear()

                    # Reset metadata
                    for key in list(self._metric_metadata.keys()):
                        if key.endswith(f"_{name}"):
                            del self._metric_metadata[key]

    def _update_timer_stats(self, name: str) -> None:
        """Update derived timer statistics."""
        values = self._timers[name]
        if not values:
            return

        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "latest": values[-1],
        }

        if len(values) > 1:
            stats["std_dev"] = statistics.stdev(values)
            stats["p95"] = self._percentile(values, 0.95)
            stats["p99"] = self._percentile(values, 0.99)

        self._timer_stats[name] = stats

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]


class MetricsRegistry:
    """
    Global registry for component metrics.

    Provides centralized access to metrics across all components
    and aggregation capabilities.
    """

    _instance: Optional["MetricsRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._metrics: dict[str, ThreadSafeMetrics] = {}
        self._registry_lock = threading.RLock()
        self._initialized = True

    def get_metrics(self, component_name: str) -> ThreadSafeMetrics:
        """Get or create metrics instance for a component."""
        with self._registry_lock:
            if component_name not in self._metrics:
                self._metrics[component_name] = ThreadSafeMetrics(component_name)
            return self._metrics[component_name]

    def get_all_component_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics from all registered components."""
        with self._registry_lock:
            return {name: metrics.get_all_metrics() for name, metrics in self._metrics.items()}

    def get_aggregated_counters(self) -> dict[str, int]:
        """Get aggregated counter values across all components."""
        aggregated: Dict[str, int] = defaultdict(int)

        with self._registry_lock:
            for metrics in self._metrics.values():
                component_metrics = metrics.get_all_metrics()
                for counter_name, value in component_metrics["counters"].items():
                    aggregated[counter_name] += value

        return dict(aggregated)

    def reset_all_metrics(self) -> None:
        """Reset metrics for all components."""
        with self._registry_lock:
            for metrics in self._metrics.values():
                metrics.reset_metrics()


class ComponentMetrics:
    """
    Mixin class for components that need metrics recording.

    Provides easy access to thread-safe metrics with component context.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._component_metrics = MetricsRegistry().get_metrics(self.__class__.__name__)

    def record_metric(self, metric_name: str, value: int) -> None:
        """Record a simple counter metric (backward compatibility)."""
        self._component_metrics.increment_counter(metric_name, value)

    def increment(self, metric_name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self._component_metrics.increment_counter(metric_name, value)

    def set_gauge(self, metric_name: str, value: int | float) -> None:
        """Set a gauge metric."""
        self._component_metrics.set_gauge(metric_name, value)

    def record_timing(self, metric_name: str, duration_ms: float) -> None:
        """Record a timing metric."""
        self._component_metrics.record_time(metric_name, duration_ms)

    def record_histogram(self, metric_name: str, value: int | float) -> None:
        """Record a histogram value."""
        self._component_metrics.record_histogram_value(metric_name, value)

    def get_metrics(self) -> dict[str, Any]:
        """Get all metrics for this component."""
        return self._component_metrics.get_all_metrics()


class PerformanceTimer:
    """
    Context manager for timing operations with automatic metrics recording.

    Usage:
        with PerformanceTimer(self._component_metrics, "operation_name"):
            # do work
            pass
    """

    def __init__(
        self, metrics: ThreadSafeMetrics, operation_name: str, record_histogram: bool = False
    ):
        self.metrics = metrics
        self.operation_name = operation_name
        self.record_histogram = record_histogram
        self.start_time: float | None = None

    def __enter__(self) -> "PerformanceTimer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            self.metrics.record_time(self.operation_name, duration_ms)

            if self.record_histogram:
                self.metrics.record_histogram_value(f"{self.operation_name}_histogram", duration_ms)


class StandardMetricsMixin:
    """
    Standard metrics mixin that provides common metric patterns.

    Includes patterns commonly used across the codebase:
    - Request/response counting
    - Success/failure rates
    - Response time tracking
    - Circuit breaker metrics
    - Cache hit/miss rates
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "_component_metrics"):
            self._component_metrics = MetricsRegistry().get_metrics(self.__class__.__name__)

        # Initialize standard metrics
        self._init_standard_metrics()

    def _init_standard_metrics(self) -> None:
        """Initialize standard metric counters."""
        standard_counters = [
            "total_requests",
            "successful_responses",
            "failed_responses",
            "cache_hits",
            "cache_misses",
            "circuit_breaker_trips",
            "retry_attempts",
            "fallback_responses",
        ]

        for counter in standard_counters:
            self._component_metrics.increment_counter(counter, 0)

    def record_request(self) -> None:
        """Record a request."""
        self._component_metrics.increment_counter("total_requests")

    def record_success(self, response_time_ms: float | None = None) -> None:
        """Record a successful response."""
        self._component_metrics.increment_counter("successful_responses")
        if response_time_ms is not None:
            self.record_response_time(response_time_ms)

    def record_failure(self) -> None:
        """Record a failed response."""
        self._component_metrics.increment_counter("failed_responses")

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self._component_metrics.increment_counter("cache_hits")

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self._component_metrics.increment_counter("cache_misses")

    def record_circuit_breaker_trip(self) -> None:
        """Record a circuit breaker trip."""
        self._component_metrics.increment_counter("circuit_breaker_trips")

    def record_retry_attempt(self) -> None:
        """Record a retry attempt."""
        self._component_metrics.increment_counter("retry_attempts")

    def record_fallback_response(self) -> None:
        """Record use of fallback response."""
        self._component_metrics.increment_counter("fallback_responses")

    def record_response_time(self, duration_ms: float) -> None:
        """Record response time with automatic statistics."""
        self._component_metrics.record_time("response_time", duration_ms)

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        successful = self._component_metrics.get_counter("successful_responses")
        total = self._component_metrics.get_counter("total_requests")

        if total == 0:
            return 0.0

        return successful / total

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        hits = self._component_metrics.get_counter("cache_hits")
        misses = self._component_metrics.get_counter("cache_misses")
        total = hits + misses

        if total == 0:
            return 0.0

        return hits / total

    def record_metric(self, metric_name: str, value: int) -> None:
        """Record a simple counter metric (backward compatibility)."""
        self._component_metrics.increment_counter(metric_name, value)
