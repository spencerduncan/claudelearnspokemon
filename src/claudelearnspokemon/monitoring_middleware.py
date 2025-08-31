"""
HTTP monitoring middleware for Pokemon speedrun learning agent.

Provides comprehensive HTTP request/response instrumentation with minimal overhead,
tracking performance, status codes, and usage patterns across pokemon-gym endpoints.

Performance requirements:
- HTTP middleware overhead: <1ms per request
- Memory efficient: <5MB additional memory for instrumentation
- Thread-safe operations for concurrent requests
"""

import functools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class HTTPRequestMetrics:
    """Metrics for a single HTTP request."""
    
    method: str
    url: str
    status_code: int
    duration_seconds: float
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


@dataclass 
class EndpointStats:
    """Aggregated statistics for an HTTP endpoint."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    last_request_time: float = 0.0
    
    def add_request(self, metrics: HTTPRequestMetrics):
        """Add a request to the endpoint statistics."""
        self.total_requests += 1
        self.total_duration += metrics.duration_seconds
        self.min_duration = min(self.min_duration, metrics.duration_seconds)
        self.max_duration = max(self.max_duration, metrics.duration_seconds)
        self.last_request_time = metrics.timestamp
        
        if 200 <= metrics.status_code < 400:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
    
    @property
    def average_duration(self) -> float:
        """Calculate average request duration."""
        return self.total_duration / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.successful_requests / self.total_requests * 100.0 
                if self.total_requests > 0 else 0.0)


class HTTPMonitoringMiddleware:
    """
    HTTP request/response monitoring middleware.
    
    Provides comprehensive instrumentation of HTTP interactions with 
    pokemon-gym endpoints while maintaining <1ms overhead requirement.
    """
    
    def __init__(
        self, 
        enable_detailed_logging: bool = False,
        max_recorded_requests: int = 1000,
        endpoint_grouping: bool = True
    ):
        """
        Initialize HTTP monitoring middleware.

        Args:
            enable_detailed_logging: Enable detailed request/response logging
            max_recorded_requests: Maximum number of detailed requests to keep in memory
            endpoint_grouping: Group similar endpoints for aggregated statistics
        """
        self.enable_detailed_logging = enable_detailed_logging
        self.max_recorded_requests = max_recorded_requests
        self.endpoint_grouping = endpoint_grouping
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._endpoint_stats: Dict[str, EndpointStats] = {}
        self._recent_requests: list = []
        
        # Performance tracking
        self._instrumentation_overhead = []
        self._max_overhead_samples = 100
        
        logger.info(f"HTTP monitoring middleware initialized (detailed_logging={enable_detailed_logging})")

    def _normalize_endpoint(self, url: str) -> str:
        """
        Normalize URL to endpoint for grouping statistics.
        
        Args:
            url: Full URL to normalize
            
        Returns:
            Normalized endpoint string
        """
        if not self.endpoint_grouping:
            return url
            
        try:
            parsed = urlparse(url)
            path = parsed.path
            
            # Group common pokemon-gym endpoints
            if '/state' in path:
                return 'GET /state'
            elif '/reset' in path:
                return 'POST /reset'
            elif '/action' in path:
                return 'POST /action'
            elif '/health' in path:
                return 'GET /health'
            elif '/screen' in path:
                return 'GET /screen'
            else:
                return f"{parsed.path}"
                
        except Exception:
            return url

    @contextmanager
    def monitor_request(self, method: str, url: str, **kwargs):
        """
        Context manager for monitoring HTTP requests.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional request parameters
            
        Yields:
            Request monitoring context
        """
        start_time = time.time()
        instrumentation_start = time.time()
        
        request_metrics = HTTPRequestMetrics(
            method=method.upper(),
            url=url,
            status_code=0,
            duration_seconds=0.0
        )
        
        try:
            # Calculate instrumentation overhead
            instrumentation_overhead = time.time() - instrumentation_start
            self._record_instrumentation_overhead(instrumentation_overhead)
            
            yield request_metrics
            
        except Exception as e:
            request_metrics.error = str(e)
            logger.error(f"HTTP request failed: {method} {url} - {e}")
            raise
        finally:
            # Finalize metrics
            end_time = time.time()
            request_metrics.duration_seconds = end_time - start_time
            
            # Record metrics (thread-safe)
            self._record_request_metrics(request_metrics)

    def instrument_requests_function(self, requests_func: Callable) -> Callable:
        """
        Decorator to instrument a requests function (get, post, etc.).
        
        Args:
            requests_func: Requests library function to instrument
            
        Returns:
            Instrumented function
        """
        @functools.wraps(requests_func)
        def instrumented_request(*args, **kwargs):
            # Extract method and URL from arguments
            method = requests_func.__name__.upper()
            url = args[0] if args else kwargs.get('url', 'unknown')
            
            with self.monitor_request(method, url) as metrics:
                response = requests_func(*args, **kwargs)
                
                # Update metrics from response
                metrics.status_code = response.status_code
                metrics.response_size_bytes = len(response.content) if hasattr(response, 'content') else 0
                
                return response
                
        return instrumented_request

    def _record_request_metrics(self, metrics: HTTPRequestMetrics):
        """Record request metrics in thread-safe manner."""
        with self._lock:
            # Update endpoint statistics
            endpoint = self._normalize_endpoint(metrics.url)
            if endpoint not in self._endpoint_stats:
                self._endpoint_stats[endpoint] = EndpointStats()
            
            self._endpoint_stats[endpoint].add_request(metrics)
            
            # Store detailed request (with size limit)
            self._recent_requests.append(metrics)
            if len(self._recent_requests) > self.max_recorded_requests:
                self._recent_requests.pop(0)  # Remove oldest
                
            # Optional detailed logging
            if self.enable_detailed_logging:
                logger.info(
                    f"HTTP {metrics.method} {metrics.url} -> {metrics.status_code} "
                    f"({metrics.duration_seconds:.3f}s)"
                )

    def _record_instrumentation_overhead(self, overhead_seconds: float):
        """Record instrumentation overhead for performance monitoring."""
        with self._lock:
            self._instrumentation_overhead.append(overhead_seconds)
            if len(self._instrumentation_overhead) > self._max_overhead_samples:
                self._instrumentation_overhead.pop(0)
                
            # Alert if overhead exceeds target
            if overhead_seconds > 0.001:  # 1ms threshold
                logger.warning(
                    f"HTTP instrumentation overhead {overhead_seconds:.4f}s exceeds 1ms target"
                )

    def get_endpoint_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get aggregated statistics for all monitored endpoints.
        
        Returns:
            Dictionary mapping endpoints to their statistics
        """
        with self._lock:
            stats = {}
            for endpoint, endpoint_stats in self._endpoint_stats.items():
                stats[endpoint] = {
                    "total_requests": endpoint_stats.total_requests,
                    "successful_requests": endpoint_stats.successful_requests,
                    "failed_requests": endpoint_stats.failed_requests,
                    "success_rate_percent": round(endpoint_stats.success_rate, 2),
                    "average_duration_ms": round(endpoint_stats.average_duration * 1000, 2),
                    "min_duration_ms": round(endpoint_stats.min_duration * 1000, 2),
                    "max_duration_ms": round(endpoint_stats.max_duration * 1000, 2),
                    "last_request_ago_seconds": round(time.time() - endpoint_stats.last_request_time, 1),
                }
            return stats

    def get_recent_requests(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        """
        Get recent HTTP requests with details.
        
        Args:
            limit: Maximum number of requests to return
            
        Returns:
            List of recent request dictionaries
        """
        with self._lock:
            requests = self._recent_requests[-limit:] if limit else self._recent_requests[:]
            
            return [
                {
                    "method": req.method,
                    "url": req.url,
                    "status_code": req.status_code,
                    "duration_ms": round(req.duration_seconds * 1000, 2),
                    "timestamp": req.timestamp,
                    "error": req.error,
                }
                for req in requests
            ]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get middleware performance metrics.
        
        Returns:
            Dictionary with performance and overhead statistics
        """
        with self._lock:
            if not self._instrumentation_overhead:
                return {"error": "No overhead measurements available"}
                
            overhead_values = self._instrumentation_overhead[:]
            
            return {
                "instrumentation_overhead": {
                    "average_ms": round(sum(overhead_values) / len(overhead_values) * 1000, 3),
                    "max_ms": round(max(overhead_values) * 1000, 3),
                    "min_ms": round(min(overhead_values) * 1000, 3),
                    "samples": len(overhead_values),
                    "target_ms": 1.0,
                },
                "total_endpoints_monitored": len(self._endpoint_stats),
                "total_requests_recorded": sum(stats.total_requests for stats in self._endpoint_stats.values()),
                "recent_requests_stored": len(self._recent_requests),
            }

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get health summary for monitoring status.
        
        Returns:
            Dictionary with overall health status
        """
        try:
            performance = self.get_performance_metrics()
            endpoint_stats = self.get_endpoint_statistics()
            
            # Calculate overall success rate
            total_requests = sum(
                stats["total_requests"] for stats in endpoint_stats.values()
            )
            total_successful = sum(
                stats["successful_requests"] for stats in endpoint_stats.values()
            )
            overall_success_rate = (
                total_successful / total_requests * 100.0 if total_requests > 0 else 0.0
            )
            
            # Check performance health
            avg_overhead = (
                performance.get("instrumentation_overhead", {}).get("average_ms", 0.0)
            )
            performance_healthy = avg_overhead < 1.0  # <1ms target
            
            return {
                "healthy": performance_healthy and overall_success_rate >= 95.0,
                "total_requests": total_requests,
                "overall_success_rate": round(overall_success_rate, 2),
                "performance_healthy": performance_healthy,
                "average_overhead_ms": avg_overhead,
                "endpoints_monitored": len(endpoint_stats),
                "timestamp": time.time(),
            }
            
        except Exception as e:
            logger.error(f"Failed to generate health summary: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": time.time(),
            }

    def reset_statistics(self):
        """Reset all collected statistics and metrics."""
        with self._lock:
            self._endpoint_stats.clear()
            self._recent_requests.clear()
            self._instrumentation_overhead.clear()
            logger.info("HTTP monitoring statistics reset")

    def set_detailed_logging(self, enabled: bool):
        """Enable or disable detailed request logging."""
        self.enable_detailed_logging = enabled
        logger.info(f"HTTP detailed logging {'enabled' if enabled else 'disabled'}")


# Global middleware instance for easy integration
_global_middleware: Optional[HTTPMonitoringMiddleware] = None


def get_global_middleware() -> HTTPMonitoringMiddleware:
    """Get or create global HTTP monitoring middleware instance."""
    global _global_middleware
    if _global_middleware is None:
        _global_middleware = HTTPMonitoringMiddleware()
    return _global_middleware


def monitor_requests_session(session):
    """
    Instrument a requests.Session object with monitoring.
    
    Args:
        session: requests.Session to instrument
        
    Returns:
        Instrumented session
    """
    middleware = get_global_middleware()
    
    # Instrument common HTTP methods
    session.get = middleware.instrument_requests_function(session.get)
    session.post = middleware.instrument_requests_function(session.post)
    session.put = middleware.instrument_requests_function(session.put)
    session.delete = middleware.instrument_requests_function(session.delete)
    session.patch = middleware.instrument_requests_function(session.patch)
    session.head = middleware.instrument_requests_function(session.head)
    session.options = middleware.instrument_requests_function(session.options)
    
    return session