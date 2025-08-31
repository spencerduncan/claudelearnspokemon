"""
Prometheus metrics export for Pokemon speedrun learning agent system.

This module provides comprehensive metrics collection and export in Prometheus format,
integrating with existing monitoring infrastructure while maintaining performance targets.

Performance requirements:
- Metrics collection: <5ms per scrape
- Zero-disruption integration with existing collectors
- Thread-safe operations for concurrent access
- Backward compatibility with existing monitoring
"""

import threading
import time
from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server
from prometheus_client.core import CollectorRegistry

from .process_metrics_collector import AggregatedMetricsCollector, ProcessMetricsCollector


class PrometheusMetricsExporter:
    """
    Central coordinator for Prometheus metrics export.
    
    Integrates with existing metrics collectors to provide comprehensive
    system monitoring without disrupting current operations.
    """

    def __init__(
        self,
        aggregated_collector: Optional[AggregatedMetricsCollector] = None,
        registry: Optional[CollectorRegistry] = None,
        metric_prefix: str = "pokemon_speedrun"
    ):
        """
        Initialize Prometheus metrics exporter.

        Args:
            aggregated_collector: Existing aggregated metrics collector
            registry: Prometheus collector registry (None for default)
            metric_prefix: Prefix for all metric names
        """
        self.aggregated_collector = aggregated_collector
        self.registry = registry
        self.prefix = metric_prefix
        self._lock = threading.Lock()
        self._http_server = None
        
        # Initialize Prometheus metrics
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize all Prometheus metric objects."""
        # Process metrics
        self.process_startup_time = Histogram(
            f"{self.prefix}_process_startup_seconds",
            "Process startup time in seconds",
            ["process_type"],
            registry=self.registry
        )
        
        self.process_memory_usage = Gauge(
            f"{self.prefix}_process_memory_bytes",
            "Process memory usage in bytes",
            ["process_id", "process_type"],
            registry=self.registry
        )
        
        self.process_cpu_usage = Gauge(
            f"{self.prefix}_process_cpu_percent",
            "Process CPU usage percentage",
            ["process_id", "process_type"],
            registry=self.registry
        )
        
        self.process_health_check_duration = Histogram(
            f"{self.prefix}_health_check_duration_seconds",
            "Health check duration in seconds",
            ["process_id"],
            registry=self.registry
        )
        
        self.process_failures_total = Counter(
            f"{self.prefix}_process_failures_total",
            "Total number of process failures",
            ["process_id", "failure_type"],
            registry=self.registry
        )
        
        self.process_restarts_total = Counter(
            f"{self.prefix}_process_restarts_total", 
            "Total number of process restarts",
            ["process_id"],
            registry=self.registry
        )
        
        # System-level metrics
        self.healthy_processes = Gauge(
            f"{self.prefix}_healthy_processes",
            "Number of healthy processes",
            registry=self.registry
        )
        
        self.total_processes = Gauge(
            f"{self.prefix}_total_processes",
            "Total number of managed processes",
            registry=self.registry
        )
        
        # Performance metrics
        self.average_startup_time = Gauge(
            f"{self.prefix}_average_startup_seconds",
            "Average process startup time",
            registry=self.registry
        )
        
        self.average_health_check_time = Gauge(
            f"{self.prefix}_average_health_check_seconds", 
            "Average health check duration",
            registry=self.registry
        )
        
        # System information
        self.system_info = Info(
            f"{self.prefix}_system",
            "System information",
            registry=self.registry
        )

    def update_metrics(self):
        """
        Update all Prometheus metrics from existing collectors.
        
        This method should be called periodically to refresh metrics.
        Performance target: <5ms execution time.
        """
        start_time = time.time()
        
        try:
            with self._lock:
                if self.aggregated_collector:
                    self._update_from_aggregated_collector()
                    
            # Performance validation
            update_duration = time.time() - start_time
            if update_duration > 0.005:  # 5ms threshold
                import logging
                logging.warning(
                    f"Prometheus metrics update took {update_duration:.3f}s (>{0.005:.3f}s threshold)"
                )
                
        except Exception as e:
            import logging
            logging.error(f"Failed to update Prometheus metrics: {e}")

    def _update_from_aggregated_collector(self):
        """Update metrics from the aggregated collector."""
        system_metrics = self.aggregated_collector.get_system_metrics()
        
        # System-level metrics
        self.healthy_processes.set(system_metrics["healthy_processes"])
        self.total_processes.set(system_metrics["total_processes"])
        self.average_startup_time.set(system_metrics["average_startup_time_ms"] / 1000.0)
        self.average_health_check_time.set(system_metrics["average_health_check_time_ms"] / 1000.0)
        
        # Process-specific metrics
        for process_id, details in system_metrics.get("process_details", {}).items():
            process_id_str = str(process_id)
            
            # Memory usage (convert MB to bytes)
            self.process_memory_usage.labels(
                process_id=process_id_str,
                process_type="claude_process"
            ).set(details["memory_usage_mb"] * 1024 * 1024)
            
            # CPU usage
            self.process_cpu_usage.labels(
                process_id=process_id_str,
                process_type="claude_process"
            ).set(details["cpu_usage_percent"])
            
            # Record startup time
            if details["startup_time_ms"] > 0:
                self.process_startup_time.labels(
                    process_type="claude_process"
                ).observe(details["startup_time_ms"] / 1000.0)
            
            # Record health check duration
            if details["health_check_time_ms"] > 0:
                self.process_health_check_duration.labels(
                    process_id=process_id_str
                ).observe(details["health_check_time_ms"] / 1000.0)

    def record_process_failure(self, process_id: int, failure_type: str = "unknown"):
        """
        Record a process failure event.

        Args:
            process_id: ID of the failed process
            failure_type: Type of failure (startup, health_check, runtime, etc.)
        """
        self.process_failures_total.labels(
            process_id=str(process_id),
            failure_type=failure_type
        ).inc()

    def record_process_restart(self, process_id: int):
        """
        Record a process restart event.

        Args:
            process_id: ID of the restarted process
        """
        self.process_restarts_total.labels(
            process_id=str(process_id)
        ).inc()

    def set_system_info(self, info_dict: Dict[str, str]):
        """
        Set system information metrics.

        Args:
            info_dict: Dictionary of system information key-value pairs
        """
        self.system_info.info(info_dict)

    def start_http_server(self, port: int = 8000, addr: str = "localhost"):
        """
        Start HTTP server for Prometheus metrics scraping.

        Args:
            port: Port to serve metrics on
            addr: Address to bind server to
        """
        with self._lock:
            if self._http_server is not None:
                return  # Already running
                
            self._http_server = start_http_server(port, addr, registry=self.registry)
            import logging
            logging.info(f"Prometheus metrics server started on {addr}:{port}")

    def stop_http_server(self):
        """Stop the Prometheus metrics HTTP server."""
        with self._lock:
            if self._http_server is not None:
                self._http_server.shutdown()
                self._http_server = None
                import logging
                logging.info("Prometheus metrics server stopped")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of current metric values.

        Returns:
            Dictionary with current metric values for debugging/monitoring
        """
        if not self.aggregated_collector:
            return {"error": "No aggregated collector available"}
            
        return {
            "last_update": time.time(),
            "system_metrics": self.aggregated_collector.get_system_metrics(),
            "prometheus_server_running": self._http_server is not None,
        }


class MetricsUpdateScheduler:
    """
    Scheduler for periodic Prometheus metrics updates.
    
    Provides automatic metrics refresh to keep Prometheus data current
    while maintaining performance requirements.
    """
    
    def __init__(self, exporter: PrometheusMetricsExporter, update_interval: float = 15.0):
        """
        Initialize metrics update scheduler.

        Args:
            exporter: PrometheusMetricsExporter to update
            update_interval: Interval between updates in seconds
        """
        self.exporter = exporter
        self.update_interval = update_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Start periodic metrics updates."""
        with self._lock:
            if self._running:
                return
                
            self._running = True
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()
            
            import logging
            logging.info(f"Metrics update scheduler started (interval: {self.update_interval}s)")

    def stop(self):
        """Stop periodic metrics updates."""
        with self._lock:
            self._running = False
            
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            
        import logging
        logging.info("Metrics update scheduler stopped")

    def _update_loop(self):
        """Main update loop."""
        while self._running:
            try:
                self.exporter.update_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                import logging
                logging.error(f"Error in metrics update loop: {e}")
                time.sleep(1.0)  # Brief pause before retry