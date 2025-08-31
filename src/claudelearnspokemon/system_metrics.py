"""
System-level metrics collection for Pokemon speedrun learning agent.

Provides comprehensive OS-level monitoring including CPU, memory, disk, and network metrics
with efficient caching to minimize performance overhead.

Performance requirements:
- Metrics collection: <3ms with 5-second caching
- Memory efficient: <10MB additional memory usage
- Thread-safe operations for concurrent access
"""

import logging
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Container for system-level performance metrics."""
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: tuple = field(default_factory=tuple)
    
    # Memory metrics  
    memory_total: int = 0
    memory_available: int = 0
    memory_used: int = 0
    memory_percent: float = 0.0
    swap_total: int = 0
    swap_used: int = 0
    swap_percent: float = 0.0
    
    # Disk metrics
    disk_total: int = 0
    disk_used: int = 0
    disk_free: int = 0
    disk_percent: float = 0.0
    
    # Network metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0
    
    # Process metrics
    process_count: int = 0
    
    # System info
    boot_time: float = 0.0
    uptime_seconds: float = 0.0
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "cpu": {
                "percent": self.cpu_percent,
                "count": self.cpu_count,
                "load_average": list(self.load_average) if self.load_average else [],
            },
            "memory": {
                "total": self.memory_total,
                "available": self.memory_available,
                "used": self.memory_used,
                "percent": self.memory_percent,
            },
            "swap": {
                "total": self.swap_total,
                "used": self.swap_used,
                "percent": self.swap_percent,
            },
            "disk": {
                "total": self.disk_total,
                "used": self.disk_used,
                "free": self.disk_free,
                "percent": self.disk_percent,
            },
            "network": {
                "bytes_sent": self.network_bytes_sent,
                "bytes_recv": self.network_bytes_recv,
                "packets_sent": self.network_packets_sent,
                "packets_recv": self.network_packets_recv,
            },
            "process_count": self.process_count,
            "uptime_seconds": self.uptime_seconds,
            "timestamp": self.timestamp,
        }


class SystemMetricsCollector:
    """
    Efficient system metrics collector with caching.
    
    Collects comprehensive OS-level metrics while maintaining performance
    through intelligent caching and minimal system call overhead.
    """
    
    def __init__(self, cache_duration: float = 5.0, disk_path: str = "/"):
        """
        Initialize system metrics collector.

        Args:
            cache_duration: Cache duration in seconds for performance
            disk_path: Filesystem path to monitor for disk metrics
        """
        self.cache_duration = cache_duration
        self.disk_path = disk_path
        self._lock = threading.Lock()
        
        # Cached metrics and timing
        self._cached_metrics: Optional[SystemMetrics] = None
        self._last_collection_time: float = 0.0
        
        # System information (collected once)
        self._system_info = self._collect_system_info()
        self._boot_time = psutil.boot_time()
        
        # Network counters for delta calculation
        self._last_network_counters: Optional[psutil._common.snetio] = None
        
        logger.info(f"SystemMetricsCollector initialized with {cache_duration}s cache")

    def _collect_system_info(self) -> Dict[str, str]:
        """Collect static system information."""
        try:
            return {
                "platform": platform.platform(),
                "system": platform.system(), 
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            }
        except Exception as e:
            logger.warning(f"Failed to collect system info: {e}")
            return {"error": str(e)}

    def get_metrics(self) -> SystemMetrics:
        """
        Get current system metrics with caching.
        
        Uses caching to ensure <3ms performance target while providing
        accurate system monitoring data.
        
        Returns:
            SystemMetrics object with current system state
        """
        current_time = time.time()
        
        with self._lock:
            # Check if cached metrics are still valid
            if (self._cached_metrics and 
                current_time - self._last_collection_time < self.cache_duration):
                return self._cached_metrics
        
        # Collect fresh metrics
        start_time = time.time()
        metrics = self._collect_fresh_metrics()
        collection_duration = time.time() - start_time
        
        with self._lock:
            self._cached_metrics = metrics
            self._last_collection_time = current_time
            
        # Performance validation
        if collection_duration > 0.003:  # 3ms threshold
            logger.warning(
                f"System metrics collection took {collection_duration:.3f}s (>{0.003:.3f}s threshold)"
            )
            
        return metrics

    def _collect_fresh_metrics(self) -> SystemMetrics:
        """Collect fresh system metrics from OS."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            
            # Load average (Unix/Linux only)
            load_average = ()
            try:
                if hasattr(psutil, 'getloadavg'):
                    load_average = psutil.getloadavg()
            except (AttributeError, OSError):
                pass  # Windows doesn't support load average
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage(self.disk_path)
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Uptime calculation
            current_time = time.time()
            uptime = current_time - self._boot_time
            
            return SystemMetrics(
                # CPU
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                load_average=load_average,
                
                # Memory
                memory_total=memory.total,
                memory_available=memory.available,
                memory_used=memory.used,
                memory_percent=memory.percent,
                
                # Swap
                swap_total=swap.total,
                swap_used=swap.used,
                swap_percent=swap.percent,
                
                # Disk
                disk_total=disk.total,
                disk_used=disk.used,
                disk_free=disk.free,
                disk_percent=(disk.used / disk.total * 100) if disk.total > 0 else 0.0,
                
                # Network
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                network_packets_sent=network.packets_sent,
                network_packets_recv=network.packets_recv,
                
                # Process
                process_count=process_count,
                
                # System
                boot_time=self._boot_time,
                uptime_seconds=uptime,
                timestamp=current_time,
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return empty metrics with error indicator
            return SystemMetrics(timestamp=time.time())

    def get_system_info(self) -> Dict[str, str]:
        """
        Get static system information.
        
        Returns:
            Dictionary with system platform and configuration details
        """
        return self._system_info.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for monitoring overhead.
        
        Returns:
            Dictionary with collection timing and cache statistics
        """
        with self._lock:
            cache_age = (time.time() - self._last_collection_time
                        if self._cached_metrics else float('inf'))
            
            return {
                "cache_duration": self.cache_duration,
                "cache_age_seconds": cache_age,
                "cache_valid": cache_age < self.cache_duration,
                "last_collection": self._last_collection_time,
                "metrics_available": self._cached_metrics is not None,
            }

    def clear_cache(self):
        """Clear cached metrics to force fresh collection on next request."""
        with self._lock:
            self._cached_metrics = None
            self._last_collection_time = 0.0

    def is_healthy(self) -> bool:
        """
        Check if system metrics collection is functioning properly.
        
        Returns:
            True if metrics collection is working, False otherwise
        """
        try:
            metrics = self.get_metrics()
            # Basic sanity checks
            return (
                metrics.timestamp > 0 and
                metrics.cpu_count > 0 and
                metrics.memory_total > 0 and
                metrics.disk_total > 0
            )
        except Exception as e:
            logger.error(f"System metrics health check failed: {e}")
            return False

    def get_resource_usage_alerts(self, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Check current resource usage against configurable thresholds.
        
        Args:
            thresholds: Custom threshold values (defaults to production values)
            
        Returns:
            Dictionary with alert status for each resource type
        """
        if thresholds is None:
            thresholds = {
                "cpu_percent": 80.0,
                "memory_percent": 85.0,
                "disk_percent": 90.0,
                "swap_percent": 50.0,
            }
        
        metrics = self.get_metrics()
        alerts = {}
        
        # CPU alert
        if metrics.cpu_percent > thresholds.get("cpu_percent", 80.0):
            alerts["cpu_high"] = {
                "current": metrics.cpu_percent,
                "threshold": thresholds["cpu_percent"],
                "severity": "warning" if metrics.cpu_percent < 90.0 else "critical",
            }
        
        # Memory alert
        if metrics.memory_percent > thresholds.get("memory_percent", 85.0):
            alerts["memory_high"] = {
                "current": metrics.memory_percent,
                "threshold": thresholds["memory_percent"],
                "severity": "warning" if metrics.memory_percent < 95.0 else "critical",
            }
        
        # Disk alert
        if metrics.disk_percent > thresholds.get("disk_percent", 90.0):
            alerts["disk_high"] = {
                "current": metrics.disk_percent,
                "threshold": thresholds["disk_percent"],
                "severity": "warning" if metrics.disk_percent < 95.0 else "critical",
            }
        
        # Swap alert (if swap is configured)
        if metrics.swap_total > 0 and metrics.swap_percent > thresholds.get("swap_percent", 50.0):
            alerts["swap_high"] = {
                "current": metrics.swap_percent,
                "threshold": thresholds["swap_percent"],
                "severity": "warning" if metrics.swap_percent < 75.0 else "critical",
            }
        
        return {
            "timestamp": metrics.timestamp,
            "alerts": alerts,
            "alert_count": len(alerts),
            "has_critical": any(alert.get("severity") == "critical" for alert in alerts.values()),
        }