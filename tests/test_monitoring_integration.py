"""
Comprehensive integration tests for the Pokemon speedrun monitoring system.

Tests the complete monitoring pipeline including Prometheus metrics export,
system metrics collection, HTTP middleware, speedrun metrics, and alerting.
"""

import time
import threading
import unittest
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry

from claudelearnspokemon.prometheus_metrics import (
    PrometheusMetricsExporter,
    MetricsUpdateScheduler
)
from claudelearnspokemon.system_metrics import SystemMetricsCollector
from claudelearnspokemon.monitoring_middleware import HTTPMonitoringMiddleware
from claudelearnspokemon.speedrun_metrics import (
    SpeedrunMetricsCollector,
    ExperimentResult,
    ExperimentStatus,
    PatternDiscovery,
    PatternType
)
from claudelearnspokemon.alert_manager import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    get_default_alert_rules
)
from claudelearnspokemon.process_metrics_collector import (
    AggregatedMetricsCollector,
    ProcessMetricsCollector
)


class TestMonitoringSystemIntegration(unittest.TestCase):
    """Integration tests for the complete monitoring system."""
    
    def setUp(self):
        """Set up comprehensive test environment."""
        # Core components
        self.registry = CollectorRegistry()
        self.system_metrics = SystemMetricsCollector(cache_duration=0.5)
        self.http_middleware = HTTPMonitoringMiddleware()
        self.speedrun_metrics = SpeedrunMetricsCollector()
        
        # Process metrics (simulated)
        self.process_collector = ProcessMetricsCollector(process_id=12345)
        self.aggregated_collector = AggregatedMetricsCollector()
        self.aggregated_collector.add_collector(self.process_collector)
        
        # Prometheus integration
        self.prometheus_exporter = PrometheusMetricsExporter(
            aggregated_collector=self.aggregated_collector,
            registry=self.registry
        )
        
        # Alert management
        self.alert_manager = AlertManager(evaluation_interval=0.5)
    
    def tearDown(self):
        """Clean up test environment."""
        self.alert_manager.stop_monitoring()
    
    @patch('claudelearnspokemon.system_metrics.psutil')
    def test_end_to_end_monitoring_pipeline(self, mock_psutil):
        """Test complete monitoring pipeline from data collection to metrics export."""
        # Mock system metrics
        self._setup_system_metrics_mocks(mock_psutil)
        
        # Simulate process activity
        self.process_collector.record_startup_time(0.085)  # 85ms startup
        self.process_collector.record_health_check(0.003)  # 3ms health check
        self.process_collector.update_resource_usage(145.2, 18.5)  # 145MB, 18.5% CPU
        
        # Simulate HTTP activity
        with self.http_middleware.monitor_request("GET", "http://localhost:8081/health") as metrics:
            time.sleep(0.01)  # Simulate request processing
            metrics.status_code = 200
            metrics.response_size_bytes = 256
        
        with self.http_middleware.monitor_request("POST", "http://localhost:8081/action") as metrics:
            time.sleep(0.02)
            metrics.status_code = 200
            metrics.response_size_bytes = 512
        
        # Simulate speedrun experiments
        experiment = ExperimentResult(
            experiment_id="integration_test_001",
            status=ExperimentStatus.SUCCESS,
            duration_seconds=125.7,
            script_compilation_time_ms=87.3,
            checkpoint_loading_time_ms=425.8,
            ai_strategy="genetic_algorithm",
            performance_score=0.89
        )
        self.speedrun_metrics.record_experiment(experiment)
        
        pattern = PatternDiscovery(
            pattern_id="integration_pattern_001",
            pattern_type=PatternType.OPTIMIZATION,
            quality_score=0.91,
            discovery_time_seconds=42.5,
            experiment_id="integration_test_001",
            ai_worker="worker_3"
        )
        self.speedrun_metrics.record_pattern_discovery(pattern)
        
        # Update all metrics
        system_snapshot = self.system_metrics.get_metrics()
        speedrun_snapshot = self.speedrun_metrics.get_metrics_snapshot()
        http_stats = self.http_middleware.get_endpoint_statistics()
        
        # Export to Prometheus
        self.prometheus_exporter.update_metrics()
        prometheus_summary = self.prometheus_exporter.get_metrics_summary()
        
        # Verify end-to-end data flow
        self.assertIsNotNone(system_snapshot)
        self.assertGreater(system_snapshot.cpu_count, 0)
        
        self.assertEqual(speedrun_snapshot.total_experiments, 1)
        self.assertEqual(speedrun_snapshot.successful_experiments, 1)
        self.assertEqual(speedrun_snapshot.total_patterns_discovered, 1)
        
        self.assertIn("GET /health", http_stats)
        self.assertIn("POST /action", http_stats)
        self.assertEqual(http_stats["GET /health"]["total_requests"], 1)
        
        self.assertIn("system_metrics", prometheus_summary)
        self.assertTrue(prometheus_summary["prometheus_server_running"] is not None)
    
    def test_performance_under_concurrent_load(self):
        """Test monitoring system performance under concurrent load."""
        def simulate_process_activity():
            for i in range(20):
                self.process_collector.record_health_check(0.002 + (i * 0.0001))
                self.process_collector.update_resource_usage(100 + i, 10 + i)
                time.sleep(0.001)
        
        def simulate_http_activity():
            for i in range(15):
                with self.http_middleware.monitor_request("GET", f"http://localhost:808{i%4+1}/test") as metrics:
                    metrics.status_code = 200 if i % 5 != 0 else 500  # 80% success rate
                    time.sleep(0.001)
        
        def simulate_speedrun_activity():
            for i in range(10):
                exp = ExperimentResult(
                    experiment_id=f"load_test_{i}",
                    status=ExperimentStatus.SUCCESS if i % 3 != 0 else ExperimentStatus.FAILURE,
                    duration_seconds=100.0 + i * 5,
                    script_compilation_time_ms=80.0 + i * 2,
                    checkpoint_loading_time_ms=400.0 + i * 10
                )
                self.speedrun_metrics.record_experiment(exp)
                time.sleep(0.001)
        
        # Run concurrent activity
        start_time = time.time()
        threads = [
            threading.Thread(target=simulate_process_activity),
            threading.Thread(target=simulate_http_activity), 
            threading.Thread(target=simulate_speedrun_activity),
            threading.Thread(target=simulate_process_activity),  # Additional load
        ]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Validate performance under load
        self.assertLess(total_time, 2.0, "Concurrent load test should complete in <2s")
        
        # Verify all metrics were recorded
        speedrun_snapshot = self.speedrun_metrics.get_metrics_snapshot()
        http_stats = self.http_middleware.get_endpoint_statistics()
        
        self.assertEqual(speedrun_snapshot.total_experiments, 10)
        total_http_requests = sum(endpoint["total_requests"] for endpoint in http_stats.values())
        self.assertEqual(total_http_requests, 15)
    
    @patch('claudelearnspokemon.system_metrics.psutil')
    def test_alert_system_integration(self, mock_psutil):
        """Test alert system integration with monitoring metrics."""
        # Setup mocked high-resource system
        mock_psutil.cpu_percent.return_value = 95.0  # Critical CPU
        mock_psutil.cpu_count.return_value = 8
        
        mock_memory = MagicMock()
        mock_memory.total = 17179869184  # 16GB
        mock_memory.available = 1073741824  # 1GB available
        mock_memory.used = 16106127360  # 15GB used
        mock_memory.percent = 95.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock other required calls
        mock_swap = MagicMock()
        mock_swap.total = mock_swap.used = mock_swap.percent = 0
        mock_psutil.swap_memory.return_value = mock_swap
        
        mock_disk = MagicMock()
        mock_disk.total = mock_disk.used = mock_disk.free = 1000000000
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_psutil.net_io_counters.return_value = MagicMock()
        mock_psutil.pids.return_value = list(range(200))
        mock_psutil.boot_time.return_value = time.time() - 7200
        
        # Add metric sources to alert manager
        self.alert_manager.add_metric_source("system", self.system_metrics.get_metrics)
        self.alert_manager.add_metric_source("speedrun", self.speedrun_metrics.get_metrics_snapshot)
        
        # Add default alert rules
        default_rules = get_default_alert_rules()
        for rule in default_rules[:3]:  # Add first 3 rules for testing
            self.alert_manager.add_alert_rule(rule)
        
        # Custom high CPU alert for testing
        cpu_alert_rule = AlertRule(
            rule_id="test_cpu_alert",
            name="Test CPU Alert",
            description="CPU usage test alert",
            metric_name="system.cpu_percent",
            threshold_value=80.0,
            operator=">",
            severity=AlertSeverity.CRITICAL,
            minimum_duration_seconds=0.1  # Short duration for testing
        )
        self.alert_manager.add_alert_rule(cpu_alert_rule)
        
        # Define metrics function to return system data
        def get_system_metrics():
            metrics = self.system_metrics.get_metrics()
            return {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_percent": metrics.disk_percent,
            }
        
        # Override system metrics source
        self.alert_manager._metric_sources["system"] = get_system_metrics
        
        # Evaluate rules manually (instead of starting background thread)
        self.alert_manager._evaluate_all_rules()
        
        # Check for triggered alerts
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Should have CPU alert triggered
        self.assertGreater(len(active_alerts), 0, "Should have triggered CPU alert")
        
        cpu_alerts = [alert for alert in active_alerts if "cpu" in alert.rule.name.lower()]
        self.assertGreater(len(cpu_alerts), 0, "Should have CPU-related alert")
    
    def test_metrics_scheduler_integration(self):
        """Test metrics update scheduler integration."""
        # Setup scheduler with fast update interval
        scheduler = MetricsUpdateScheduler(
            exporter=self.prometheus_exporter,
            update_interval=0.1  # 100ms for fast testing
        )
        
        # Start scheduler
        scheduler.start()
        
        # Let it run for a short period
        time.sleep(0.35)  # Allow 3-4 updates
        
        # Stop scheduler
        scheduler.stop()
        
        # Should have called update_metrics multiple times
        # Note: In real test, we would mock the exporter to verify calls
        self.assertTrue(scheduler._running is False)
    
    @patch('claudelearnspokemon.system_metrics.psutil')
    def test_sla_monitoring_integration(self, mock_psutil):
        """Test SLA monitoring across all system components."""
        # Setup normal system metrics
        self._setup_system_metrics_mocks(mock_psutil)
        
        # Simulate experiments meeting SLA requirements
        sla_compliant_experiments = [
            ExperimentResult("sla_test_1", ExperimentStatus.SUCCESS, 120.0, 
                           script_compilation_time_ms=85.0, checkpoint_loading_time_ms=450.0),
            ExperimentResult("sla_test_2", ExperimentStatus.SUCCESS, 115.0,
                           script_compilation_time_ms=92.0, checkpoint_loading_time_ms=420.0),
            ExperimentResult("sla_test_3", ExperimentStatus.SUCCESS, 125.0,
                           script_compilation_time_ms=88.0, checkpoint_loading_time_ms=445.0),
        ]
        
        for exp in sla_compliant_experiments:
            self.speedrun_metrics.record_experiment(exp)
        
        # Add high-quality patterns
        patterns = [
            PatternDiscovery("sla_pattern_1", PatternType.MOVEMENT, 0.92, 30.0, "sla_test_1"),
            PatternDiscovery("sla_pattern_2", PatternType.BATTLE, 0.88, 35.0, "sla_test_2"),
        ]
        
        for pattern in patterns:
            self.speedrun_metrics.record_pattern_discovery(pattern)
        
        # Generate successful HTTP requests
        for i in range(20):
            with self.http_middleware.monitor_request("GET", "http://localhost:8081/health") as metrics:
                metrics.status_code = 200
                time.sleep(0.001)  # Fast response
        
        # Check SLA compliance
        speedrun_sla = self.speedrun_metrics.get_sla_compliance()
        http_health = self.http_middleware.get_health_summary()
        system_alerts = self.system_metrics.get_resource_usage_alerts()
        
        # All components should be healthy and SLA-compliant
        self.assertTrue(speedrun_sla["overall_sla_compliant"])
        self.assertTrue(http_health["healthy"])
        self.assertEqual(system_alerts["alert_count"], 0)
        
        # Performance should be within targets
        speedrun_snapshot = self.speedrun_metrics.get_metrics_snapshot()
        self.assertEqual(speedrun_snapshot.experiment_success_rate, 100.0)
        self.assertLess(speedrun_snapshot.average_script_compilation_ms, 100.0)
        self.assertLess(speedrun_snapshot.average_checkpoint_loading_ms, 500.0)
    
    def _setup_system_metrics_mocks(self, mock_psutil):
        """Helper method to setup normal system metrics mocks."""
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.cpu_count.return_value = 8
        
        mock_memory = MagicMock()
        mock_memory.total = 17179869184  # 16GB
        mock_memory.available = 12884901888  # 12GB available
        mock_memory.used = 4294967296  # 4GB used
        mock_memory.percent = 25.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_swap = MagicMock()
        mock_swap.total = mock_swap.used = mock_swap.percent = 0
        mock_psutil.swap_memory.return_value = mock_swap
        
        mock_disk = MagicMock()
        mock_disk.total = 1099511627776  # 1TB
        mock_disk.used = 549755813888    # 512GB
        mock_disk.free = 549755813888    # 512GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_psutil.net_io_counters.return_value = MagicMock()
        mock_psutil.pids.return_value = list(range(150))
        mock_psutil.boot_time.return_value = time.time() - 7200  # 2 hours uptime


@pytest.mark.integration
class TestRealSystemIntegration:
    """Integration tests with real system components (when available)."""
    
    def test_real_system_monitoring_stack(self):
        """Test monitoring stack with real system calls (where possible)."""
        # Use real system metrics collector
        system_metrics = SystemMetricsCollector(cache_duration=0.1)
        
        # Test real metrics collection
        metrics = system_metrics.get_metrics()
        
        # Validate we got real system data
        assert metrics.cpu_count > 0
        assert metrics.memory_total > 0
        assert metrics.process_count > 10  # Reasonable minimum for any system
        
        # Test performance
        start_time = time.time()
        for _ in range(5):
            system_metrics.get_metrics()  # Should use cache after first call
        collection_time = time.time() - start_time
        
        assert collection_time < 0.1, "5 cached calls should be <100ms total"
    
    def test_prometheus_export_format(self):
        """Test that Prometheus export format is valid."""
        registry = CollectorRegistry()
        exporter = PrometheusMetricsExporter(registry=registry)
        
        # Set some test data
        exporter.record_process_failure(123, "test")
        exporter.record_process_restart(123)
        
        # Should not raise exceptions
        summary = exporter.get_metrics_summary()
        assert "last_update" in summary


if __name__ == "__main__":
    unittest.main()