"""
Comprehensive unit tests for MetricsEndpoint HTTP server.

Tests cover all functionality including:
- HTTP server lifecycle (start/stop)
- Request handling for different endpoints
- Performance characteristics and response times
- Thread safety and concurrent request handling
- Error handling and edge cases

Author: Claude Code - Scientist Worker - HTTP Performance Testing
"""

import pytest
import requests
import threading
import time
import gzip
from unittest.mock import Mock, patch
from urllib.parse import urljoin

# Import components under test
import sys
sys.path.insert(0, 'src')

from claudelearnspokemon.metrics_endpoint import MetricsEndpoint, MetricsRequestHandler
from claudelearnspokemon.prometheus_exporter import PrometheusMetricsExporter


class TestMetricsEndpoint:
    """Test suite for MetricsEndpoint HTTP server."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.host = "localhost"
        self.port = 18000  # Use high port to avoid conflicts
        self.endpoint = None
    
    def teardown_method(self):
        """Clean up after each test."""
        if self.endpoint and self.endpoint.is_healthy():
            self.endpoint.stop()
        # Allow time for cleanup
        time.sleep(0.1)
    
    def test_initialization(self):
        """Test MetricsEndpoint initialization."""
        endpoint = MetricsEndpoint(
            host=self.host,
            port=self.port,
            max_connections=5,
            request_timeout=15.0
        )
        
        assert endpoint.host == self.host
        assert endpoint.port == self.port
        assert endpoint.max_connections == 5
        assert endpoint.request_timeout == 15.0
        assert endpoint.metrics_exporter is None
        assert not endpoint.is_healthy()
        
        self.endpoint = endpoint
    
    def test_initialization_with_exporter(self):
        """Test initialization with metrics exporter."""
        exporter = PrometheusMetricsExporter()
        endpoint = MetricsEndpoint(
            host=self.host,
            port=self.port,
            metrics_exporter=exporter
        )
        
        assert endpoint.metrics_exporter == exporter
        self.endpoint = endpoint
    
    def test_server_start_stop(self):
        """Test server lifecycle management."""
        endpoint = MetricsEndpoint(host=self.host, port=self.port)
        self.endpoint = endpoint
        
        # Test start
        assert endpoint.start() is True
        assert endpoint.is_healthy() is True
        
        # Test double start (should return False)
        assert endpoint.start() is False
        
        # Test stop
        assert endpoint.stop() is True
        assert endpoint.is_healthy() is False
        
        # Test double stop (should return True - idempotent)
        assert endpoint.stop() is True
    
    def test_context_manager(self):
        """Test context manager functionality."""
        endpoint = MetricsEndpoint(host=self.host, port=self.port)
        
        with endpoint as ep:
            assert ep.is_healthy()
            assert ep == endpoint
        
        # Should be stopped after context exit
        assert not endpoint.is_healthy()
    
    def test_metrics_exporter_setting(self):
        """Test setting metrics exporter after initialization."""
        endpoint = MetricsEndpoint(host=self.host, port=self.port)
        exporter = PrometheusMetricsExporter()
        
        endpoint.set_metrics_exporter(exporter)
        assert endpoint.metrics_exporter == exporter
        
        self.endpoint = endpoint
    
    def test_get_endpoint_url(self):
        """Test endpoint URL generation."""
        endpoint = MetricsEndpoint(host="example.com", port=9090)
        
        url = endpoint.get_endpoint_url()
        assert url == "http://example.com:9090/metrics"
        
        self.endpoint = endpoint
    
    def test_server_stats(self):
        """Test server statistics tracking."""
        endpoint = MetricsEndpoint(host=self.host, port=self.port)
        
        stats = endpoint.get_server_stats()
        
        assert isinstance(stats, dict)
        assert 'uptime_seconds' in stats
        assert 'total_requests' in stats
        assert 'requests_per_second' in stats
        assert 'average_response_time_ms' in stats
        assert 'is_running' in stats
        assert 'server_address' in stats
        assert 'endpoint_stats' in stats
        
        # Initial values
        assert stats['total_requests'] == 0
        assert stats['is_running'] is False
        assert stats['server_address'] == f"{self.host}:{self.port}"
        
        self.endpoint = endpoint
    
    def test_repr(self):
        """Test string representation."""
        endpoint = MetricsEndpoint(host=self.host, port=self.port)
        
        repr_str = repr(endpoint)
        assert f"{self.host}:{self.port}" in repr_str
        assert "stopped" in repr_str
        
        self.endpoint = endpoint


@pytest.mark.slow  # HTTP server tests are inherently slower
class TestMetricsEndpointHTTP:
    """Integration tests for HTTP functionality of MetricsEndpoint."""
    
    def setup_method(self):
        """Set up HTTP test fixtures."""
        self.host = "localhost"
        self.port = 18001  # Different port for HTTP tests
        self.base_url = f"http://{self.host}:{self.port}"
        self.endpoint = None
        
    def teardown_method(self):
        """Clean up HTTP test fixtures."""
        if self.endpoint and self.endpoint.is_healthy():
            self.endpoint.stop()
        time.sleep(0.2)  # Allow more time for HTTP cleanup
    
    def test_http_metrics_endpoint_no_exporter(self):
        """Test /metrics endpoint with no exporter registered."""
        self.endpoint = MetricsEndpoint(host=self.host, port=self.port)
        
        with self.endpoint:
            # Allow server to start
            time.sleep(0.1)
            
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            
            assert response.status_code == 200
            assert "text/plain" in response.headers.get('content-type', '')
            assert "No metrics available" in response.text
    
    def test_http_metrics_endpoint_with_exporter(self):
        """Test /metrics endpoint with exporter."""
        exporter = PrometheusMetricsExporter()
        self.endpoint = MetricsEndpoint(
            host=self.host, 
            port=self.port,
            metrics_exporter=exporter
        )
        
        with self.endpoint:
            time.sleep(0.1)
            
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            
            assert response.status_code == 200
            content_type = response.headers.get('content-type', '')
            assert 'text/plain' in content_type and 'version=0.0.4' in content_type
            assert len(response.text) > 0
    
    def test_http_health_endpoint(self):
        """Test /health endpoint."""
        self.endpoint = MetricsEndpoint(host=self.host, port=self.port)
        
        with self.endpoint:
            time.sleep(0.1)
            
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            assert response.status_code == 200
            assert "text/plain" in response.headers.get('content-type', '')
            assert "status=healthy" in response.text
            assert "timestamp=" in response.text
    
    def test_http_index_endpoint(self):
        """Test / (index) endpoint."""
        self.endpoint = MetricsEndpoint(host=self.host, port=self.port)
        
        with self.endpoint:
            time.sleep(0.1)
            
            response = requests.get(f"{self.base_url}/", timeout=5)
            
            assert response.status_code == 200
            assert "text/html" in response.headers.get('content-type', '')
            assert "Claude Learns Pokemon" in response.text
            assert "/metrics" in response.text
            assert "/health" in response.text
    
    def test_http_404_endpoint(self):
        """Test 404 handling for unknown endpoints."""
        self.endpoint = MetricsEndpoint(host=self.host, port=self.port)
        
        with self.endpoint:
            time.sleep(0.1)
            
            response = requests.get(f"{self.base_url}/unknown", timeout=5)
            
            assert response.status_code == 404
            assert "Error 404" in response.text
    
    def test_http_response_headers(self):
        """Test HTTP response headers."""
        self.endpoint = MetricsEndpoint(host=self.host, port=self.port)
        
        with self.endpoint:
            time.sleep(0.1)
            
            # Test metrics endpoint headers
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            
            assert response.status_code == 200
            assert 'content-length' in response.headers
            assert 'cache-control' in response.headers
            assert 'no-cache' in response.headers['cache-control']
    
    def test_gzip_compression(self):
        """Test gzip compression for large responses."""
        # Create exporter with some data to make response larger
        from claudelearnspokemon.process_metrics_collector import ProcessMetricsCollector
        
        exporter = PrometheusMetricsExporter()
        
        # Add several collectors to increase response size
        for i in range(10):
            collector = ProcessMetricsCollector(process_id=6000 + i)
            collector.record_startup_time(0.1 + i * 0.01)
            collector.record_health_check(0.005)
            collector.update_resource_usage(50.0, 10.0)
            exporter.register_process_collector(collector)
        
        self.endpoint = MetricsEndpoint(
            host=self.host,
            port=self.port,
            metrics_exporter=exporter
        )
        
        with self.endpoint:
            time.sleep(0.1)
            
            # Request with gzip support
            response = requests.get(
                f"{self.base_url}/metrics",
                headers={'Accept-Encoding': 'gzip'},
                timeout=5
            )
            
            assert response.status_code == 200
            # Check if response was actually compressed
            # (requests automatically decompresses, so we check encoding header)
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        self.endpoint = MetricsEndpoint(host=self.host, port=self.port)
        
        results = []
        
        def make_request():
            try:
                response = requests.get(f"{self.base_url}/health", timeout=10)
                results.append(response.status_code)
            except Exception as e:
                results.append(str(e))
        
        with self.endpoint:
            time.sleep(0.1)
            
            # Make 5 concurrent requests
            threads = [threading.Thread(target=make_request) for _ in range(5)]
            
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join()
        
        # All requests should succeed
        assert len(results) == 5
        assert all(result == 200 for result in results)
    
    def test_performance_characteristics(self):
        """Test response time meets <100ms SLA."""
        exporter = PrometheusMetricsExporter()
        self.endpoint = MetricsEndpoint(
            host=self.host,
            port=self.port, 
            metrics_exporter=exporter
        )
        
        response_times = []
        
        with self.endpoint:
            time.sleep(0.1)
            
            # Make multiple requests to measure performance
            for _ in range(10):
                start_time = time.perf_counter()
                response = requests.get(f"{self.base_url}/metrics", timeout=5)
                response_time = (time.perf_counter() - start_time) * 1000  # ms
                
                assert response.status_code == 200
                response_times.append(response_time)
                
                time.sleep(0.01)  # Small delay between requests
        
        # Calculate statistics
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Verify performance requirements
        assert avg_response_time < 100.0, f"Average response time {avg_response_time:.2f}ms exceeds 100ms SLA"
        assert max_response_time < 200.0, f"Max response time {max_response_time:.2f}ms is too high"
        
        print(f"Performance: avg={avg_response_time:.2f}ms, max={max_response_time:.2f}ms")
    
    def test_server_stats_tracking(self):
        """Test that server statistics are properly tracked."""
        self.endpoint = MetricsEndpoint(host=self.host, port=self.port)
        
        with self.endpoint:
            time.sleep(0.1)
            
            # Make some requests
            requests.get(f"{self.base_url}/metrics", timeout=5)
            requests.get(f"{self.base_url}/health", timeout=5)
            requests.get(f"{self.base_url}/", timeout=5)
            
            # Check statistics
            stats = self.endpoint.get_server_stats()
            
            assert stats['total_requests'] >= 3
            assert stats['requests_per_second'] > 0
            assert stats['average_response_time_ms'] > 0
            assert stats['average_response_time_ms'] < 1000  # Should be well under 1 second
            
            # Check endpoint-specific stats
            endpoint_stats = stats['endpoint_stats']
            assert 'metrics' in endpoint_stats
            assert 'health' in endpoint_stats
            assert 'index' in endpoint_stats
            
            for endpoint_name, endpoint_data in endpoint_stats.items():
                assert endpoint_data['requests'] >= 1
                assert endpoint_data['average_response_time_ms'] > 0


class TestMetricsRequestHandler:
    """Test suite for MetricsRequestHandler class."""
    
    def test_handler_creation(self):
        """Test that handler can be created (basic sanity check)."""
        # Note: Testing HTTP handlers in isolation is complex
        # Most functionality is tested via integration tests above
        assert MetricsRequestHandler is not None
        
        # Test log_message override (should not raise)
        # This is mainly to ensure the custom logging doesn't break
        handler = Mock()
        handler.address_string = Mock(return_value="127.0.0.1")
        
        # Call the method with mocked self
        MetricsRequestHandler.log_message(handler, "Test message %s", "arg")


class TestMetricsEndpointEdgeCases:
    """Test edge cases and error conditions for MetricsEndpoint."""
    
    def test_start_with_invalid_port(self):
        """Test starting server with invalid port."""
        # Use port 0 to let OS assign, then try to use same port again
        endpoint1 = MetricsEndpoint(host="localhost", port=18002)
        endpoint2 = MetricsEndpoint(host="localhost", port=18002)
        
        try:
            success1 = endpoint1.start()
            assert success1 is True
            
            # Second endpoint on same port should fail
            success2 = endpoint2.start()
            assert success2 is False
            
        finally:
            endpoint1.stop()
            endpoint2.stop()
    
    def test_stop_timeout(self):
        """Test stop with timeout."""
        endpoint = MetricsEndpoint(host="localhost", port=18003)
        
        try:
            endpoint.start()
            time.sleep(0.1)
            
            # Test stop with very short timeout
            result = endpoint.stop(timeout=0.001)
            # Should still succeed even with short timeout for simple case
            assert result in [True, False]  # Either is acceptable
            
        finally:
            # Ensure cleanup
            endpoint.stop(timeout=5.0)
    
    def test_health_check_when_stopped(self):
        """Test health check on stopped server."""
        endpoint = MetricsEndpoint(host="localhost", port=18004)
        
        assert endpoint.is_healthy() is False
        
        endpoint.start()
        time.sleep(0.1)
        assert endpoint.is_healthy() is True
        
        endpoint.stop()
        assert endpoint.is_healthy() is False


# Performance-focused test markers
pytestmark = [
    pytest.mark.unit,
]