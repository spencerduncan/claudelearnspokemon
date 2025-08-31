"""
Tests for system metrics collection functionality.

Validates efficient OS-level monitoring with caching, performance requirements,
and resource usage alerting capabilities.
"""

import time
import threading
import unittest
from unittest.mock import MagicMock, patch

import pytest

from claudelearnspokemon.system_metrics import (
    SystemMetrics,
    SystemMetricsCollector
)


class TestSystemMetrics(unittest.TestCase):
    """Test cases for SystemMetrics data structure."""
    
    def test_initialization_defaults(self):
        """Test SystemMetrics initializes with reasonable defaults."""
        metrics = SystemMetrics()
        
        self.assertEqual(metrics.cpu_percent, 0.0)
        self.assertEqual(metrics.memory_total, 0)
        self.assertEqual(metrics.process_count, 0)
        self.assertIsInstance(metrics.timestamp, float)
        self.assertGreater(metrics.timestamp, 0)
    
    def test_to_dict_conversion(self):
        """Test converting SystemMetrics to dictionary format."""
        metrics = SystemMetrics(
            cpu_percent=25.5,
            memory_total=8589934592,  # 8GB
            memory_used=4294967296,   # 4GB
            memory_percent=50.0,
            disk_total=1099511627776, # 1TB
            disk_used=549755813888,   # 512GB
            process_count=125
        )
        
        result = metrics.to_dict()
        
        # Verify structure
        self.assertIn("cpu", result)
        self.assertIn("memory", result)
        self.assertIn("disk", result)
        self.assertIn("process_count", result)
        
        # Verify values
        self.assertEqual(result["cpu"]["percent"], 25.5)
        self.assertEqual(result["memory"]["total"], 8589934592)
        self.assertEqual(result["memory"]["percent"], 50.0)
        self.assertEqual(result["process_count"], 125)


class TestSystemMetricsCollector(unittest.TestCase):
    """Test cases for SystemMetricsCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = SystemMetricsCollector(cache_duration=1.0)  # 1 second cache
    
    @patch('claudelearnspokemon.system_metrics.psutil')
    def test_collect_fresh_metrics(self, mock_psutil):
        """Test collecting fresh system metrics from psutil."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 35.5
        mock_psutil.cpu_count.return_value = 8
        
        # Mock memory
        mock_memory = MagicMock()
        mock_memory.total = 17179869184  # 16GB
        mock_memory.available = 8589934592  # 8GB
        mock_memory.used = 8589934592   # 8GB
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock swap
        mock_swap = MagicMock()
        mock_swap.total = 2147483648  # 2GB
        mock_swap.used = 0
        mock_swap.percent = 0.0
        mock_psutil.swap_memory.return_value = mock_swap
        
        # Mock disk
        mock_disk = MagicMock()
        mock_disk.total = 1099511627776  # 1TB
        mock_disk.used = 549755813888    # 512GB
        mock_disk.free = 549755813888    # 512GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        # Mock network
        mock_network = MagicMock()
        mock_network.bytes_sent = 1073741824    # 1GB
        mock_network.bytes_recv = 2147483648    # 2GB
        mock_network.packets_sent = 1000000
        mock_network.packets_recv = 2000000
        mock_psutil.net_io_counters.return_value = mock_network
        
        # Mock other functions
        mock_psutil.pids.return_value = list(range(150))  # 150 processes
        mock_psutil.boot_time.return_value = time.time() - 3600  # 1 hour uptime
        
        # Collect metrics
        metrics = self.collector.get_metrics()
        
        # Verify collection
        self.assertEqual(metrics.cpu_percent, 35.5)
        self.assertEqual(metrics.cpu_count, 8)
        self.assertEqual(metrics.memory_total, 17179869184)
        self.assertEqual(metrics.memory_percent, 50.0)
        self.assertEqual(metrics.process_count, 150)
        self.assertGreater(metrics.uptime_seconds, 3500)  # Around 1 hour
    
    def test_metrics_caching_performance(self):
        """Test that caching meets performance requirements (<3ms)."""
        # First call will collect fresh metrics
        start_time = time.time()
        metrics1 = self.collector.get_metrics()
        first_duration = time.time() - start_time
        
        # Second call should use cache
        start_time = time.time()
        metrics2 = self.collector.get_metrics()
        cached_duration = time.time() - start_time
        
        # Cached call should be much faster
        self.assertLess(cached_duration, 0.001, "Cached metrics should be <1ms")
        self.assertEqual(metrics1.timestamp, metrics2.timestamp, "Should return cached data")
    
    def test_cache_expiration(self):
        """Test that cache expires after configured duration."""
        # Get initial metrics
        metrics1 = self.collector.get_metrics()
        initial_timestamp = metrics1.timestamp
        
        # Wait for cache to expire (1 second + margin)
        time.sleep(1.2)
        
        # Get metrics again - should be fresh
        metrics2 = self.collector.get_metrics()
        
        self.assertGreater(metrics2.timestamp, initial_timestamp, 
                          "Should collect fresh metrics after cache expiry")
    
    def test_clear_cache(self):
        """Test manually clearing the metrics cache."""
        # Get cached metrics
        metrics1 = self.collector.get_metrics()
        
        # Clear cache
        self.collector.clear_cache()
        
        # Next call should collect fresh metrics
        metrics2 = self.collector.get_metrics()
        
        self.assertGreater(metrics2.timestamp, metrics1.timestamp)
    
    def test_is_healthy_check(self):
        """Test system health validation."""
        with patch('claudelearnspokemon.system_metrics.psutil') as mock_psutil:
            # Mock healthy system
            mock_psutil.cpu_count.return_value = 4
            mock_memory = MagicMock()
            mock_memory.total = 8589934592  # 8GB
            mock_psutil.virtual_memory.return_value = mock_memory
            mock_disk = MagicMock()
            mock_disk.total = 1099511627776  # 1TB
            mock_psutil.disk_usage.return_value = mock_disk
            mock_psutil.net_io_counters.return_value = MagicMock()
            mock_psutil.pids.return_value = list(range(100))
            mock_psutil.boot_time.return_value = time.time() - 3600
            
            self.assertTrue(self.collector.is_healthy())
    
    def test_is_healthy_check_failure(self):
        """Test health check failure handling."""
        with patch('claudelearnspokemon.system_metrics.psutil') as mock_psutil:
            # Mock psutil failure
            mock_psutil.cpu_count.side_effect = Exception("System unavailable")
            
            self.assertFalse(self.collector.is_healthy())
    
    def test_get_resource_usage_alerts_normal(self):
        """Test resource usage alerts under normal conditions."""
        with patch('claudelearnspokemon.system_metrics.psutil') as mock_psutil:
            # Mock normal system usage
            mock_psutil.cpu_percent.return_value = 45.0  # Normal CPU
            
            mock_memory = MagicMock()
            mock_memory.total = 8589934592
            mock_memory.percent = 60.0  # Normal memory
            mock_psutil.virtual_memory.return_value = mock_memory
            
            mock_swap = MagicMock()
            mock_swap.total = 2147483648
            mock_swap.percent = 25.0  # Normal swap
            mock_psutil.swap_memory.return_value = mock_swap
            
            mock_disk = MagicMock()
            mock_disk.total = 1099511627776
            mock_disk.used = 549755813888  # 50% disk usage
            mock_psutil.disk_usage.return_value = mock_disk
            
            # Mock other required calls
            mock_psutil.net_io_counters.return_value = MagicMock()
            mock_psutil.pids.return_value = list(range(100))
            mock_psutil.boot_time.return_value = time.time() - 3600
            
            alerts = self.collector.get_resource_usage_alerts()
            
            self.assertEqual(alerts["alert_count"], 0)
            self.assertFalse(alerts["has_critical"])
    
    def test_get_resource_usage_alerts_high_usage(self):
        """Test resource usage alerts under high usage conditions."""
        with patch('claudelearnspokemon.system_metrics.psutil') as mock_psutil:
            # Mock high system usage
            mock_psutil.cpu_percent.return_value = 95.0  # Critical CPU
            
            mock_memory = MagicMock()
            mock_memory.total = 8589934592
            mock_memory.percent = 98.0  # Critical memory
            mock_psutil.virtual_memory.return_value = mock_memory
            
            mock_swap = MagicMock()
            mock_swap.total = 2147483648
            mock_swap.percent = 80.0  # High swap
            mock_psutil.swap_memory.return_value = mock_swap
            
            mock_disk = MagicMock()
            mock_disk.total = 1099511627776
            mock_disk.used = 1044266352025  # 95% disk usage
            mock_psutil.disk_usage.return_value = mock_disk
            
            # Mock other required calls
            mock_psutil.net_io_counters.return_value = MagicMock()
            mock_psutil.pids.return_value = list(range(100))
            mock_psutil.boot_time.return_value = time.time() - 3600
            
            alerts = self.collector.get_resource_usage_alerts()
            
            self.assertGreater(alerts["alert_count"], 0)
            self.assertTrue(alerts["has_critical"])
            
            # Check specific alerts
            alert_dict = alerts["alerts"]
            self.assertIn("cpu_high", alert_dict)
            self.assertIn("memory_high", alert_dict)
            self.assertEqual(alert_dict["cpu_high"]["severity"], "critical")
            self.assertEqual(alert_dict["memory_high"]["severity"], "critical")
    
    def test_custom_alert_thresholds(self):
        """Test resource alerts with custom thresholds."""
        custom_thresholds = {
            "cpu_percent": 50.0,      # Lower CPU threshold
            "memory_percent": 70.0,   # Lower memory threshold
            "disk_percent": 80.0,     # Lower disk threshold
        }
        
        with patch('claudelearnspokemon.system_metrics.psutil') as mock_psutil:
            # Mock moderate usage that exceeds custom thresholds
            mock_psutil.cpu_percent.return_value = 60.0
            
            mock_memory = MagicMock()
            mock_memory.total = 8589934592
            mock_memory.percent = 75.0
            mock_psutil.virtual_memory.return_value = mock_memory
            
            mock_swap = MagicMock()
            mock_swap.total = 0  # No swap configured
            mock_psutil.swap_memory.return_value = mock_swap
            
            mock_disk = MagicMock()
            mock_disk.total = 1099511627776
            mock_disk.used = 879609302220  # 85% disk usage
            mock_psutil.disk_usage.return_value = mock_disk
            
            # Mock other required calls
            mock_psutil.net_io_counters.return_value = MagicMock()
            mock_psutil.pids.return_value = list(range(100))
            mock_psutil.boot_time.return_value = time.time() - 3600
            
            alerts = self.collector.get_resource_usage_alerts(custom_thresholds)
            
            # Should trigger alerts with custom thresholds
            self.assertGreater(alerts["alert_count"], 0)
            alert_dict = alerts["alerts"]
            self.assertIn("cpu_high", alert_dict)
            self.assertIn("memory_high", alert_dict)
            self.assertIn("disk_high", alert_dict)
    
    def test_get_performance_summary(self):
        """Test getting performance summary for monitoring overhead."""
        # Get metrics to populate cache
        self.collector.get_metrics()
        
        summary = self.collector.get_performance_summary()
        
        self.assertIn("cache_duration", summary)
        self.assertIn("cache_valid", summary)
        self.assertIn("metrics_available", summary)
        self.assertEqual(summary["cache_duration"], 1.0)
        self.assertTrue(summary["metrics_available"])
    
    def test_thread_safety(self):
        """Test thread-safe operations during concurrent access."""
        def collect_metrics():
            for _ in range(5):
                self.collector.get_metrics()
                time.sleep(0.01)
        
        # Run concurrent collections
        threads = [threading.Thread(target=collect_metrics) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should complete without deadlocks
        self.assertTrue(self.collector.is_healthy())


@pytest.mark.integration
class TestSystemMetricsIntegration:
    """Integration tests with real system calls."""
    
    def test_real_system_metrics_collection(self):
        """Test collecting actual system metrics."""
        collector = SystemMetricsCollector(cache_duration=0.1)
        
        # Collect real metrics
        metrics = collector.get_metrics()
        
        # Validate realistic values
        assert metrics.cpu_count > 0
        assert metrics.memory_total > 0
        assert metrics.disk_total > 0
        assert metrics.process_count > 10  # Reasonable minimum
        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100
    
    def test_collection_performance_real_system(self):
        """Test collection performance on real system."""
        collector = SystemMetricsCollector(cache_duration=0.1)
        
        # First collection (fresh)
        start_time = time.time()
        collector.get_metrics()
        fresh_duration = time.time() - start_time
        
        # Second collection (cached)
        start_time = time.time()
        collector.get_metrics()
        cached_duration = time.time() - start_time
        
        # Validate performance requirements
        assert fresh_duration < 0.010, f"Fresh collection {fresh_duration:.4f}s should be <10ms"
        assert cached_duration < 0.001, f"Cached collection {cached_duration:.4f}s should be <1ms"


if __name__ == "__main__":
    unittest.main()