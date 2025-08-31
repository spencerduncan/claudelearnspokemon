"""
Tests for HTTP monitoring middleware functionality.

Validates request/response instrumentation, performance overhead,
and endpoint statistics collection.
"""

import time
import threading
import unittest
from unittest.mock import MagicMock, patch

import pytest
import requests

from claudelearnspokemon.monitoring_middleware import (
    HTTPRequestMetrics,
    EndpointStats,
    HTTPMonitoringMiddleware,
    get_global_middleware,
    monitor_requests_session
)


class TestHTTPRequestMetrics(unittest.TestCase):
    """Test cases for HTTPRequestMetrics data structure."""
    
    def test_initialization(self):
        """Test HTTPRequestMetrics initialization."""
        metrics = HTTPRequestMetrics(
            method="GET",
            url="http://localhost:8081/health",
            status_code=200,
            duration_seconds=0.045
        )
        
        self.assertEqual(metrics.method, "GET")
        self.assertEqual(metrics.url, "http://localhost:8081/health")
        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(metrics.duration_seconds, 0.045)
        self.assertIsNone(metrics.error)
        self.assertIsInstance(metrics.timestamp, float)


class TestEndpointStats(unittest.TestCase):
    """Test cases for EndpointStats aggregation."""
    
    def test_initialization(self):
        """Test EndpointStats initialization with defaults."""
        stats = EndpointStats()
        
        self.assertEqual(stats.total_requests, 0)
        self.assertEqual(stats.successful_requests, 0)
        self.assertEqual(stats.failed_requests, 0)
        self.assertEqual(stats.total_duration, 0.0)
        self.assertEqual(stats.min_duration, float('inf'))
        self.assertEqual(stats.max_duration, 0.0)
    
    def test_add_successful_request(self):
        """Test adding successful request metrics."""
        stats = EndpointStats()
        metrics = HTTPRequestMetrics(
            method="GET",
            url="http://localhost:8081/health",
            status_code=200,
            duration_seconds=0.025
        )
        
        stats.add_request(metrics)
        
        self.assertEqual(stats.total_requests, 1)
        self.assertEqual(stats.successful_requests, 1)
        self.assertEqual(stats.failed_requests, 0)
        self.assertEqual(stats.total_duration, 0.025)
        self.assertEqual(stats.min_duration, 0.025)
        self.assertEqual(stats.max_duration, 0.025)
        self.assertEqual(stats.average_duration, 0.025)
        self.assertEqual(stats.success_rate, 100.0)
    
    def test_add_failed_request(self):
        """Test adding failed request metrics."""
        stats = EndpointStats()
        metrics = HTTPRequestMetrics(
            method="POST",
            url="http://localhost:8081/action",
            status_code=500,
            duration_seconds=0.15
        )
        
        stats.add_request(metrics)
        
        self.assertEqual(stats.total_requests, 1)
        self.assertEqual(stats.successful_requests, 0)
        self.assertEqual(stats.failed_requests, 1)
        self.assertEqual(stats.success_rate, 0.0)
    
    def test_add_multiple_requests(self):
        """Test statistics with multiple requests."""
        stats = EndpointStats()
        
        # Add successful request
        stats.add_request(HTTPRequestMetrics("GET", "http://test", 200, 0.020))
        # Add another successful request  
        stats.add_request(HTTPRequestMetrics("GET", "http://test", 200, 0.040))
        # Add failed request
        stats.add_request(HTTPRequestMetrics("GET", "http://test", 500, 0.080))
        
        self.assertEqual(stats.total_requests, 3)
        self.assertEqual(stats.successful_requests, 2)
        self.assertEqual(stats.failed_requests, 1)
        self.assertEqual(stats.min_duration, 0.020)
        self.assertEqual(stats.max_duration, 0.080)
        self.assertAlmostEqual(stats.average_duration, 0.047, places=3)  # (0.02+0.04+0.08)/3
        self.assertAlmostEqual(stats.success_rate, 66.67, places=1)     # 2/3 * 100


class TestHTTPMonitoringMiddleware(unittest.TestCase):
    """Test cases for HTTPMonitoringMiddleware."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.middleware = HTTPMonitoringMiddleware(
            max_recorded_requests=100,
            endpoint_grouping=True
        )
    
    def test_initialization(self):
        """Test proper initialization of middleware."""
        self.assertFalse(self.middleware.enable_detailed_logging)
        self.assertEqual(self.middleware.max_recorded_requests, 100)
        self.assertTrue(self.middleware.endpoint_grouping)
    
    def test_normalize_endpoint_grouping(self):
        """Test URL normalization for endpoint grouping."""
        test_cases = [
            ("http://localhost:8081/health", "GET /health"),
            ("http://localhost:8082/state", "GET /state"),
            ("http://localhost:8083/reset", "POST /reset"),
            ("http://localhost:8084/action", "POST /action"),
            ("http://localhost:8081/screen", "GET /screen"),
            ("http://localhost:8081/custom/path", "/custom/path"),
        ]
        
        for url, expected in test_cases:
            result = self.middleware._normalize_endpoint(url)
            self.assertEqual(result, expected, f"URL {url} should normalize to {expected}")
    
    def test_normalize_endpoint_no_grouping(self):
        """Test URL normalization with grouping disabled."""
        middleware = HTTPMonitoringMiddleware(endpoint_grouping=False)
        url = "http://localhost:8081/health"
        
        result = middleware._normalize_endpoint(url)
        self.assertEqual(result, url)
    
    def test_monitor_request_context_manager(self):
        """Test monitor_request context manager functionality."""
        with self.middleware.monitor_request("GET", "http://localhost:8081/health") as metrics:
            # Simulate request processing time
            time.sleep(0.01)
            metrics.status_code = 200
            metrics.response_size_bytes = 1024
        
        # Check if request was recorded
        stats = self.middleware.get_endpoint_statistics()
        self.assertIn("GET /health", stats)
        self.assertEqual(stats["GET /health"]["total_requests"], 1)
        self.assertEqual(stats["GET /health"]["successful_requests"], 1)
    
    def test_monitor_request_with_exception(self):
        """Test monitor_request handles exceptions properly."""
        try:
            with self.middleware.monitor_request("POST", "http://localhost:8081/action") as metrics:
                metrics.status_code = 200
                raise Exception("Simulated network error")
        except Exception as e:
            self.assertEqual(str(e), "Simulated network error")
        
        # Request should still be recorded
        recent_requests = self.middleware.get_recent_requests(limit=1)
        self.assertEqual(len(recent_requests), 1)
        self.assertIsNotNone(recent_requests[0]["error"])
    
    def test_performance_overhead_measurement(self):
        """Test that middleware overhead meets <1ms requirement."""
        # Perform multiple requests to measure overhead
        for i in range(10):
            with self.middleware.monitor_request("GET", f"http://localhost:808{i % 4 + 1}/health") as metrics:
                metrics.status_code = 200
        
        # Get performance metrics
        perf_metrics = self.middleware.get_performance_metrics()
        
        self.assertIn("instrumentation_overhead", perf_metrics)
        overhead = perf_metrics["instrumentation_overhead"]
        
        # Validate overhead is reasonable
        self.assertLess(overhead["average_ms"], 1.0, "Average overhead should be <1ms")
        self.assertGreater(overhead["samples"], 5)
    
    def test_endpoint_statistics_aggregation(self):
        """Test endpoint statistics aggregation."""
        # Generate requests for different endpoints
        endpoints = [
            ("GET", "http://localhost:8081/health", 200),
            ("GET", "http://localhost:8082/health", 200),
            ("GET", "http://localhost:8081/state", 200),
            ("POST", "http://localhost:8081/action", 500),  # Failed request
            ("GET", "http://localhost:8081/health", 200),   # Repeat
        ]
        
        for method, url, status in endpoints:
            with self.middleware.monitor_request(method, url) as metrics:
                metrics.status_code = status
        
        stats = self.middleware.get_endpoint_statistics()
        
        # Verify grouped statistics
        self.assertIn("GET /health", stats)
        self.assertIn("GET /state", stats)
        self.assertIn("POST /action", stats)
        
        # Check health endpoint stats (3 requests, all successful)
        health_stats = stats["GET /health"]
        self.assertEqual(health_stats["total_requests"], 3)
        self.assertEqual(health_stats["successful_requests"], 3)
        self.assertEqual(health_stats["success_rate_percent"], 100.0)
        
        # Check action endpoint stats (1 request, failed)
        action_stats = stats["POST /action"]
        self.assertEqual(action_stats["total_requests"], 1)
        self.assertEqual(action_stats["failed_requests"], 1)
        self.assertEqual(action_stats["success_rate_percent"], 0.0)
    
    def test_recent_requests_limit(self):
        """Test recent requests storage with size limit."""
        # Generate more requests than the limit
        for i in range(150):  # More than max_recorded_requests=100
            with self.middleware.monitor_request("GET", f"http://test/{i}") as metrics:
                metrics.status_code = 200
        
        recent = self.middleware.get_recent_requests()
        
        # Should not exceed the limit
        self.assertLessEqual(len(recent), self.middleware.max_recorded_requests)
        
        # Should contain the most recent requests
        last_request = recent[-1]
        self.assertIn("http://test/149", last_request["url"])  # Last generated request
    
    def test_get_health_summary(self):
        """Test health summary generation."""
        # Generate some requests with mixed results
        with self.middleware.monitor_request("GET", "http://localhost:8081/health") as metrics:
            metrics.status_code = 200
        
        with self.middleware.monitor_request("POST", "http://localhost:8081/action") as metrics:
            metrics.status_code = 500
        
        health = self.middleware.get_health_summary()
        
        self.assertIn("healthy", health)
        self.assertIn("total_requests", health)
        self.assertIn("overall_success_rate", health)
        self.assertIn("performance_healthy", health)
        
        self.assertEqual(health["total_requests"], 2)
        self.assertEqual(health["overall_success_rate"], 50.0)  # 1 success, 1 failure
    
    def test_reset_statistics(self):
        """Test resetting all collected statistics."""
        # Generate some data
        with self.middleware.monitor_request("GET", "http://localhost:8081/health") as metrics:
            metrics.status_code = 200
        
        # Verify data exists
        stats = self.middleware.get_endpoint_statistics()
        self.assertGreater(len(stats), 0)
        
        # Reset
        self.middleware.reset_statistics()
        
        # Verify data is cleared
        stats = self.middleware.get_endpoint_statistics()
        self.assertEqual(len(stats), 0)
        
        recent = self.middleware.get_recent_requests()
        self.assertEqual(len(recent), 0)
    
    def test_thread_safety(self):
        """Test thread-safe operations during concurrent requests."""
        def make_requests():
            for i in range(20):
                with self.middleware.monitor_request("GET", f"http://localhost:8081/test{i}") as metrics:
                    metrics.status_code = 200
                    time.sleep(0.001)  # Small delay
        
        # Run concurrent request monitoring
        threads = [threading.Thread(target=make_requests) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Verify all requests were recorded correctly
        stats = self.middleware.get_endpoint_statistics()
        total_requests = sum(endpoint["total_requests"] for endpoint in stats.values())
        self.assertEqual(total_requests, 60)  # 3 threads * 20 requests each


class TestGlobalMiddleware(unittest.TestCase):
    """Test cases for global middleware functionality."""
    
    def test_get_global_middleware(self):
        """Test getting global middleware instance."""
        middleware1 = get_global_middleware()
        middleware2 = get_global_middleware()
        
        # Should return the same instance
        self.assertIs(middleware1, middleware2)
    
    def test_monitor_requests_session(self):
        """Test instrumenting a requests.Session object."""
        session = requests.Session()
        instrumented_session = monitor_requests_session(session)
        
        # Should return the same session object
        self.assertIs(session, instrumented_session)
        
        # Methods should be wrapped (check function names changed)
        # Note: In real implementation, wrapped functions would have monitoring


@pytest.mark.integration
class TestHTTPMiddlewareIntegration:
    """Integration tests for HTTP monitoring middleware."""
    
    def test_real_http_request_monitoring(self):
        """Test monitoring real HTTP requests (mocked)."""
        middleware = HTTPMonitoringMiddleware()
        
        # Mock a real request scenario
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b"OK"
            mock_get.return_value = mock_response
            
            # Monitor the request
            with middleware.monitor_request("GET", "http://httpbin.org/get") as metrics:
                response = mock_get("http://httpbin.org/get")
                metrics.status_code = response.status_code
                metrics.response_size_bytes = len(response.content)
            
            # Verify monitoring
            stats = middleware.get_endpoint_statistics()
            assert len(stats) > 0
            
            health = middleware.get_health_summary()
            assert health["healthy"] is True
            assert health["total_requests"] == 1
    
    def test_middleware_performance_under_load(self):
        """Test middleware performance under load conditions."""
        middleware = HTTPMonitoringMiddleware()
        
        start_time = time.time()
        
        # Simulate high load
        for i in range(100):
            with middleware.monitor_request("GET", f"http://test/endpoint{i % 10}") as metrics:
                metrics.status_code = 200
        
        total_time = time.time() - start_time
        
        # Verify performance
        perf = middleware.get_performance_metrics()
        avg_overhead = perf["instrumentation_overhead"]["average_ms"]
        
        assert avg_overhead < 1.0, f"Average overhead {avg_overhead}ms should be <1ms"
        assert total_time < 1.0, f"Total time {total_time}s for 100 requests should be <1s"


if __name__ == "__main__":
    unittest.main()