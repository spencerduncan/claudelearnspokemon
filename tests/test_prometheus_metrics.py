"""
Tests for Prometheus metrics export functionality.

Validates comprehensive metrics collection, export performance,
and integration with existing monitoring infrastructure.
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry, REGISTRY

from claudelearnspokemon.prometheus_metrics import (
    PrometheusMetricsExporter,
    MetricsUpdateScheduler
)
from claudelearnspokemon.process_metrics_collector import (
    AggregatedMetricsCollector,
    ProcessMetricsCollector
)


class TestPrometheusMetricsExporter(unittest.TestCase):
    """Test cases for PrometheusMetricsExporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use separate registry for tests to avoid conflicts
        self.registry = CollectorRegistry()
        self.aggregated_collector = MagicMock(spec=AggregatedMetricsCollector)
        self.exporter = PrometheusMetricsExporter(
            aggregated_collector=self.aggregated_collector,
            registry=self.registry
        )
    
    def test_initialization(self):
        """Test proper initialization of metrics exporter."""
        self.assertEqual(self.exporter.prefix, "pokemon_speedrun")
        self.assertIsNotNone(self.exporter.process_startup_time)
        self.assertIsNotNone(self.exporter.process_memory_usage)
        self.assertIsNotNone(self.exporter.healthy_processes)
        
    def test_custom_prefix(self):
        """Test initialization with custom metric prefix."""
        custom_exporter = PrometheusMetricsExporter(
            registry=CollectorRegistry(),
            metric_prefix="custom_test"
        )
        self.assertEqual(custom_exporter.prefix, "custom_test")
    
    def test_update_metrics_performance(self):
        """Test that metrics updates meet performance requirements (<5ms)."""
        # Mock aggregated collector data
        mock_data = {
            "healthy_processes": 4,
            "total_processes": 4,
            "average_startup_time_ms": 45.5,
            "average_health_check_time_ms": 2.3,
            "process_details": {
                123: {
                    "memory_usage_mb": 125.5,
                    "cpu_usage_percent": 15.2,
                    "startup_time_ms": 45.5,
                    "health_check_time_ms": 2.3,
                }
            }
        }
        self.aggregated_collector.get_system_metrics.return_value = mock_data
        
        # Measure update performance
        start_time = time.time()
        self.exporter.update_metrics()
        update_duration = time.time() - start_time
        
        # Validate performance requirement
        self.assertLess(update_duration, 0.005, "Metrics update should be <5ms")
        
        # Verify metrics were updated
        self.aggregated_collector.get_system_metrics.assert_called_once()
    
    def test_update_metrics_with_no_collector(self):
        """Test metrics update when no aggregated collector is available."""
        exporter = PrometheusMetricsExporter(registry=CollectorRegistry())
        
        # Should not raise exception
        exporter.update_metrics()
    
    def test_record_process_failure(self):
        """Test recording process failure events."""
        # Record failure
        self.exporter.record_process_failure(123, "startup")
        
        # Verify metric was recorded (would need to check registry in real test)
        # This is a smoke test to ensure no exceptions
    
    def test_record_process_restart(self):
        """Test recording process restart events."""
        # Record restart
        self.exporter.record_process_restart(123)
        
        # Verify metric was recorded (smoke test)
    
    def test_set_system_info(self):
        """Test setting system information metrics."""
        info_dict = {
            "version": "1.0.0",
            "environment": "test",
            "hostname": "test-host"
        }
        
        # Should not raise exception
        self.exporter.set_system_info(info_dict)
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary for debugging."""
        mock_data = {"healthy_processes": 4, "total_processes": 4}
        self.aggregated_collector.get_system_metrics.return_value = mock_data
        
        summary = self.exporter.get_metrics_summary()
        
        self.assertIn("last_update", summary)
        self.assertIn("system_metrics", summary)
        self.assertIn("prometheus_server_running", summary)
        self.assertEqual(summary["system_metrics"], mock_data)
    
    def test_get_metrics_summary_no_collector(self):
        """Test metrics summary when no collector is available."""
        exporter = PrometheusMetricsExporter(registry=CollectorRegistry())
        
        summary = exporter.get_metrics_summary()
        
        self.assertIn("error", summary)
    
    def test_thread_safety(self):
        """Test thread-safe operations during concurrent updates."""
        mock_data = {
            "healthy_processes": 4,
            "total_processes": 4,
            "average_startup_time_ms": 45.5,
            "average_health_check_time_ms": 2.3,
            "process_details": {}
        }
        self.aggregated_collector.get_system_metrics.return_value = mock_data
        
        def update_metrics():
            for _ in range(10):
                self.exporter.update_metrics()
                time.sleep(0.001)  # Small delay
        
        # Run concurrent updates
        threads = [threading.Thread(target=update_metrics) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should complete without deadlocks or exceptions


class TestMetricsUpdateScheduler(unittest.TestCase):
    """Test cases for MetricsUpdateScheduler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exporter = MagicMock(spec=PrometheusMetricsExporter)
        self.scheduler = MetricsUpdateScheduler(
            exporter=self.exporter,
            update_interval=0.1  # 100ms for fast testing
        )
    
    def test_initialization(self):
        """Test proper initialization of scheduler."""
        self.assertEqual(self.scheduler.update_interval, 0.1)
        self.assertFalse(self.scheduler._running)
        self.assertIsNone(self.scheduler._thread)
    
    def test_start_stop_cycle(self):
        """Test starting and stopping the scheduler."""
        # Start scheduler
        self.scheduler.start()
        self.assertTrue(self.scheduler._running)
        self.assertIsNotNone(self.scheduler._thread)
        
        # Wait briefly for updates
        time.sleep(0.25)  # Allow 2-3 updates
        
        # Stop scheduler
        self.scheduler.stop()
        self.assertFalse(self.scheduler._running)
        
        # Verify exporter was called
        self.assertGreater(self.exporter.update_metrics.call_count, 1)
    
    def test_start_already_running(self):
        """Test starting scheduler when already running."""
        self.scheduler.start()
        initial_thread = self.scheduler._thread
        
        # Try to start again
        self.scheduler.start()
        
        # Should be same thread (no duplicate start)
        self.assertEqual(self.scheduler._thread, initial_thread)
        
        self.scheduler.stop()
    
    def test_stop_not_running(self):
        """Test stopping scheduler when not running."""
        # Should not raise exception
        self.scheduler.stop()
        self.assertFalse(self.scheduler._running)
    
    def test_scheduler_error_handling(self):
        """Test scheduler handles exporter errors gracefully."""
        # Make exporter raise exception
        self.exporter.update_metrics.side_effect = Exception("Test error")
        
        self.scheduler.start()
        time.sleep(0.15)  # Wait for error to occur
        self.scheduler.stop()
        
        # Should have attempted multiple calls despite errors
        self.assertGreater(self.exporter.update_metrics.call_count, 0)


class TestPrometheusIntegration(unittest.TestCase):
    """Integration tests for Prometheus metrics system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.registry = CollectorRegistry()
        
        # Create real aggregated collector with mock data
        self.process_collector = ProcessMetricsCollector(process_id=123)
        self.aggregated_collector = AggregatedMetricsCollector()
        self.aggregated_collector.add_collector(self.process_collector)
        
        self.exporter = PrometheusMetricsExporter(
            aggregated_collector=self.aggregated_collector,
            registry=self.registry
        )
    
    def test_end_to_end_metrics_flow(self):
        """Test complete metrics flow from collection to export."""
        # Simulate process metrics
        self.process_collector.record_startup_time(0.045)  # 45ms
        self.process_collector.record_health_check(0.003)  # 3ms
        self.process_collector.update_resource_usage(128.5, 15.2)  # 128MB, 15.2% CPU
        
        # Update Prometheus metrics
        self.exporter.update_metrics()
        
        # Verify metrics were exported (would check registry in real implementation)
        summary = self.exporter.get_metrics_summary()
        self.assertIn("system_metrics", summary)
        
        system_metrics = summary["system_metrics"]
        self.assertEqual(system_metrics["total_processes"], 1)
        self.assertGreater(system_metrics["average_startup_time_ms"], 40)
    
    def test_performance_under_load(self):
        """Test metrics performance under load conditions."""
        # Add multiple process collectors
        for i in range(10):
            collector = ProcessMetricsCollector(process_id=i + 200)
            collector.record_startup_time(0.050)  # 50ms
            collector.update_resource_usage(100.0, 10.0)
            self.aggregated_collector.add_collector(collector)
        
        # Measure update performance with more data
        start_time = time.time()
        for _ in range(5):
            self.exporter.update_metrics()
        total_time = time.time() - start_time
        
        avg_update_time = total_time / 5
        self.assertLess(avg_update_time, 0.005, 
                       f"Average update time {avg_update_time:.4f}s should be <5ms")


@pytest.mark.integration
class TestPrometheusServerIntegration:
    """Integration tests requiring actual HTTP server."""
    
    def test_http_server_lifecycle(self):
        """Test starting and stopping HTTP metrics server."""
        registry = CollectorRegistry()
        exporter = PrometheusMetricsExporter(registry=registry)
        
        try:
            # Start server on non-standard port for testing
            exporter.start_http_server(port=0)  # OS assigns available port
            
            # Server should be running
            assert exporter._http_server is not None
            
            # Stop server
            exporter.stop_http_server()
            assert exporter._http_server is None
            
        finally:
            # Cleanup
            if exporter._http_server:
                exporter.stop_http_server()


if __name__ == "__main__":
    unittest.main()