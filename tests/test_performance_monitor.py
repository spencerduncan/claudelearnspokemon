"""
Comprehensive test suite for ClaudeCodeManager Performance Monitoring System.

Test Categories:
- TestPerformanceMonitorBasics: Core performance monitoring functionality
- TestConversationPerformanceTracking: Conversation metrics and efficiency tracking
- TestCompressionEffectivenessMonitoring: Context compression monitoring
- TestAlertingSystem: Performance threshold alerts and notifications
- TestPerformanceAnalytics: Analytics, trends, and reporting
- TestClaudeCodeManagerIntegration: Integration with ClaudeCodeManager
- TestExternalMetricsExport: External system integration
"""

import json
import tempfile
import time
import unittest
from typing import Any

import pytest

from claudelearnspokemon.claude_code_manager import ClaudeCodeManager
from claudelearnspokemon.performance_monitor import (
    AlertingSystem,
    AlertSeverity,
    CompressionEffectivenessMetrics,
    CompressionEffectivenessMonitor,
    ConversationEfficiencyMetrics,
    ConversationMetrics,
    ConversationPerformanceTracker,
    ExternalMetricsExporter,
    MonitoringThresholds,
    PerformanceAlert,
    PerformanceMonitor,
)
from claudelearnspokemon.process_metrics_collector import AggregatedMetricsCollector


@pytest.mark.fast
class TestPerformanceMonitorBasics(unittest.TestCase):
    """Test core PerformanceMonitor functionality and initialization."""

    def setUp(self):
        """Set up test environment."""
        self.aggregated_collector = AggregatedMetricsCollector()
        self.performance_monitor = PerformanceMonitor(
            aggregated_collector=self.aggregated_collector,
            enable_real_time_monitoring=False  # Disable for testing
        )

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.stop_monitoring()

    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initializes correctly."""
        self.assertIsInstance(self.performance_monitor, PerformanceMonitor)
        self.assertIsInstance(self.performance_monitor.conversation_tracker, ConversationPerformanceTracker)
        self.assertIsInstance(self.performance_monitor.compression_monitor, CompressionEffectivenessMonitor)
        self.assertIsInstance(self.performance_monitor.alerting_system, AlertingSystem)
        self.assertIsInstance(self.performance_monitor.metrics_exporter, ExternalMetricsExporter)

    def test_monitoring_thresholds_initialization(self):
        """Test monitoring thresholds are properly initialized."""
        thresholds = MonitoringThresholds()

        # Response time thresholds
        self.assertEqual(thresholds.tactical_response_time_warning, 2000.0)
        self.assertEqual(thresholds.tactical_response_time_critical, 5000.0)
        self.assertEqual(thresholds.strategic_response_time_warning, 5000.0)
        self.assertEqual(thresholds.strategic_response_time_critical, 10000.0)

        # Efficiency thresholds
        self.assertEqual(thresholds.turn_efficiency_warning, 0.7)
        self.assertEqual(thresholds.turn_efficiency_critical, 0.5)

        # Compression thresholds
        self.assertEqual(thresholds.compression_ratio_warning, 0.6)
        self.assertEqual(thresholds.compression_ratio_critical, 0.4)

    def test_custom_monitoring_thresholds(self):
        """Test PerformanceMonitor with custom thresholds."""
        custom_thresholds = MonitoringThresholds(
            tactical_response_time_warning=1500.0,
            tactical_response_time_critical=3000.0,
            turn_efficiency_warning=0.8,
            turn_efficiency_critical=0.6
        )

        monitor = PerformanceMonitor(
            aggregated_collector=self.aggregated_collector,
            thresholds=custom_thresholds,
            enable_real_time_monitoring=False
        )

        self.assertEqual(monitor.alerting_system.thresholds.tactical_response_time_warning, 1500.0)
        self.assertEqual(monitor.alerting_system.thresholds.tactical_response_time_critical, 3000.0)
        self.assertEqual(monitor.alerting_system.thresholds.turn_efficiency_warning, 0.8)
        self.assertEqual(monitor.alerting_system.thresholds.turn_efficiency_critical, 0.6)

    def test_context_manager_functionality(self):
        """Test PerformanceMonitor context manager functionality."""
        with PerformanceMonitor(
            aggregated_collector=self.aggregated_collector,
            enable_real_time_monitoring=True
        ) as monitor:
            self.assertIsInstance(monitor, PerformanceMonitor)
            # Context manager should start monitoring automatically


@pytest.mark.fast
class TestConversationPerformanceTracking(unittest.TestCase):
    """Test conversation performance tracking functionality."""

    def setUp(self):
        """Set up test environment."""
        self.tracker = ConversationPerformanceTracker(max_history=100)

    def test_conversation_tracker_initialization(self):
        """Test ConversationPerformanceTracker initializes correctly."""
        self.assertIsInstance(self.tracker, ConversationPerformanceTracker)
        self.assertEqual(len(self.tracker._conversation_metrics), 0)
        self.assertEqual(len(self.tracker._active_conversations), 0)

    def test_start_conversation_tracking(self):
        """Test starting conversation tracking."""
        conversation_id = "test_conv_1"
        self.tracker.start_conversation_tracking(conversation_id)

        self.assertIn(conversation_id, self.tracker._active_conversations)
        self.assertIsInstance(self.tracker._active_conversations[conversation_id], float)

    def test_record_conversation_exchange(self):
        """Test recording conversation exchange with minimal overhead."""
        conversation_id = "test_conv_1"

        # Record conversation exchange
        start_time = time.perf_counter()
        self.tracker.record_conversation_exchange(
            conversation_id=conversation_id,
            response_time_ms=1500.0,
            turn_number=1,
            tokens_used=100,
            context_size=5000,
            process_type="sonnet_tactical",
            success=True
        )
        recording_time = (time.perf_counter() - start_time) * 1000

        # Verify recording overhead is minimal (<5ms for testing)
        self.assertLess(recording_time, 5.0, "Recording overhead should be minimal")

        # Verify metrics were recorded
        self.assertEqual(len(self.tracker._conversation_metrics), 1)

        metrics = self.tracker._conversation_metrics[0]
        self.assertEqual(metrics.conversation_id, conversation_id)
        self.assertEqual(metrics.response_time_ms, 1500.0)
        self.assertEqual(metrics.turn_number, 1)
        self.assertEqual(metrics.tokens_used, 100)
        self.assertEqual(metrics.context_size, 5000)
        self.assertEqual(metrics.process_type, "sonnet_tactical")
        self.assertTrue(metrics.success)

    def test_record_multiple_exchanges(self):
        """Test recording multiple conversation exchanges."""
        conversation_id = "test_conv_multi"

        # Record multiple exchanges
        for i in range(5):
            self.tracker.record_conversation_exchange(
                conversation_id=conversation_id,
                response_time_ms=1000.0 + (i * 100),
                turn_number=i + 1,
                tokens_used=100 + i * 10,
                context_size=5000 + i * 500,
                process_type="sonnet_tactical",
                success=i < 4  # Last one fails for testing
            )

        self.assertEqual(len(self.tracker._conversation_metrics), 5)

        # Verify last exchange was marked as failed
        last_metrics = self.tracker._conversation_metrics[-1]
        self.assertFalse(last_metrics.success)

    def test_conversation_efficiency_metrics_calculation(self):
        """Test conversation efficiency metrics calculation."""
        # Record multiple exchanges for different process types
        tactical_conversations = ["tactical_1", "tactical_2", "tactical_3"]
        strategic_conversations = ["strategic_1", "strategic_2"]

        # Record tactical conversations
        for conv_id in tactical_conversations:
            for turn in range(1, 4):
                self.tracker.record_conversation_exchange(
                    conversation_id=conv_id,
                    response_time_ms=1500.0 + (turn * 100),
                    turn_number=turn,
                    tokens_used=100,
                    context_size=5000,
                    process_type="sonnet_tactical",
                    success=True
                )

        # Record strategic conversations
        for conv_id in strategic_conversations:
            for turn in range(1, 6):
                self.tracker.record_conversation_exchange(
                    conversation_id=conv_id,
                    response_time_ms=3000.0 + (turn * 200),
                    turn_number=turn,
                    tokens_used=200,
                    context_size=10000,
                    process_type="opus_strategic",
                    success=True
                )

        # Calculate efficiency metrics
        efficiency_metrics = self.tracker.get_conversation_efficiency_metrics(time_window_minutes=60)

        self.assertGreater(len(efficiency_metrics), 0)

        # Find tactical and strategic metrics
        tactical_metrics = [em for em in efficiency_metrics if "tactical" in em.conversation_type]
        strategic_metrics = [em for em in efficiency_metrics if "strategic" in em.conversation_type]

        self.assertGreater(len(tactical_metrics), 0)
        self.assertGreater(len(strategic_metrics), 0)

        # Verify tactical metrics
        tactical_em = tactical_metrics[0]
        self.assertEqual(tactical_em.completion_rate, 1.0)  # All successful
        self.assertGreater(tactical_em.avg_response_time_ms, 1000.0)
        self.assertGreater(tactical_em.efficiency_score, 0.0)

        # Verify strategic metrics
        strategic_em = strategic_metrics[0]
        self.assertEqual(strategic_em.completion_rate, 1.0)  # All successful
        self.assertGreater(strategic_em.avg_response_time_ms, 3000.0)
        self.assertGreater(strategic_em.efficiency_score, 0.0)

    def test_get_recent_metrics(self):
        """Test getting recent conversation metrics."""
        # Record some metrics
        for i in range(10):
            self.tracker.record_conversation_exchange(
                conversation_id=f"conv_{i}",
                response_time_ms=1000.0 + i * 100,
                turn_number=1,
                tokens_used=100,
                context_size=5000,
                process_type="sonnet_tactical",
                success=True
            )

        # Get recent metrics
        recent_metrics = self.tracker.get_recent_metrics(5)
        self.assertEqual(len(recent_metrics), 5)

        # Verify we get the most recent ones
        self.assertEqual(recent_metrics[-1].conversation_id, "conv_9")
        self.assertEqual(recent_metrics[0].conversation_id, "conv_5")


@pytest.mark.fast
class TestCompressionEffectivenessMonitoring(unittest.TestCase):
    """Test context compression effectiveness monitoring."""

    def setUp(self):
        """Set up test environment."""
        self.monitor = CompressionEffectivenessMonitor(max_history=50)

    def test_compression_monitor_initialization(self):
        """Test CompressionEffectivenessMonitor initializes correctly."""
        self.assertIsInstance(self.monitor, CompressionEffectivenessMonitor)
        self.assertEqual(len(self.monitor._compression_metrics), 0)

    def test_record_compression_event(self):
        """Test recording compression events."""
        compression_id = "test_compression_1"

        self.monitor.record_compression_event(
            compression_id=compression_id,
            original_size=10000,
            compressed_size=3000,
            compression_time_ms=800.0,
            critical_info_preserved=True
        )

        self.assertEqual(len(self.monitor._compression_metrics), 1)

        metrics = self.monitor._compression_metrics[0]
        self.assertEqual(metrics.compression_id, compression_id)
        self.assertEqual(metrics.original_size, 10000)
        self.assertEqual(metrics.compressed_size, 3000)
        self.assertEqual(metrics.compression_ratio, 0.3)  # 3000/10000 = 0.3
        self.assertEqual(metrics.compression_time_ms, 800.0)
        self.assertTrue(metrics.critical_info_preserved)
        self.assertGreater(metrics.effectiveness_score, 0.0)

    def test_compression_effectiveness_score_calculation(self):
        """Test compression effectiveness score calculation."""
        # Good compression: fast, high ratio, preserved info
        self.monitor.record_compression_event(
            compression_id="good_compression",
            original_size=10000,
            compressed_size=2000,  # 80% compression
            compression_time_ms=500.0,  # Fast
            critical_info_preserved=True
        )

        # Poor compression: slow, low ratio, lost info
        self.monitor.record_compression_event(
            compression_id="poor_compression",
            original_size=10000,
            compressed_size=8000,  # 20% compression
            compression_time_ms=2500.0,  # Slow
            critical_info_preserved=False
        )

        good_metrics = self.monitor._compression_metrics[0]
        poor_metrics = self.monitor._compression_metrics[1]

        # Good compression should have higher effectiveness score
        self.assertGreater(good_metrics.effectiveness_score, poor_metrics.effectiveness_score)
        self.assertGreater(good_metrics.effectiveness_score, 0.7)  # Should be high
        self.assertLess(poor_metrics.effectiveness_score, 0.5)  # Should be low

    def test_compression_analytics(self):
        """Test compression analytics generation."""
        # Record multiple compression events
        compression_data = [
            (10000, 3000, 600.0, True),
            (15000, 4500, 800.0, True),
            (8000, 3200, 1200.0, False),
            (12000, 2400, 700.0, True),
            (20000, 8000, 1800.0, False),
        ]

        for i, (orig, comp, time_ms, preserved) in enumerate(compression_data):
            self.monitor.record_compression_event(
                compression_id=f"compression_{i}",
                original_size=orig,
                compressed_size=comp,
                compression_time_ms=time_ms,
                critical_info_preserved=preserved
            )

        # Get analytics
        analytics = self.monitor.get_compression_analytics(time_window_minutes=60)

        self.assertEqual(analytics["total_compressions"], 5)
        self.assertGreater(analytics["avg_compression_ratio"], 0.0)
        self.assertGreater(analytics["avg_compression_time_ms"], 0.0)
        self.assertGreater(analytics["avg_effectiveness_score"], 0.0)
        self.assertEqual(analytics["critical_info_preservation_rate"], 0.6)  # 3/5 = 0.6
        self.assertGreater(analytics["compression_frequency_per_hour"], 0.0)
        self.assertEqual(len(analytics["recent_metrics"]), 5)

    def test_empty_analytics(self):
        """Test analytics with no compression events."""
        analytics = self.monitor.get_compression_analytics(time_window_minutes=60)

        self.assertEqual(analytics["total_compressions"], 0)
        self.assertEqual(analytics["avg_compression_ratio"], 0.0)
        self.assertEqual(analytics["avg_compression_time_ms"], 0.0)
        self.assertEqual(analytics["avg_effectiveness_score"], 0.0)
        self.assertEqual(analytics["critical_info_preservation_rate"], 0.0)
        self.assertEqual(analytics["compression_frequency_per_hour"], 0.0)


@pytest.mark.fast
class TestAlertingSystem(unittest.TestCase):
    """Test performance alerting system."""

    def setUp(self):
        """Set up test environment."""
        self.thresholds = MonitoringThresholds(
            tactical_response_time_warning=1000.0,
            tactical_response_time_critical=2000.0,
            strategic_response_time_warning=2000.0,
            strategic_response_time_critical=4000.0,
            turn_efficiency_warning=0.8,
            turn_efficiency_critical=0.6,
            compression_ratio_warning=0.7,
            compression_ratio_critical=0.5
        )
        self.alerting_system = AlertingSystem(self.thresholds)

    def test_alerting_system_initialization(self):
        """Test AlertingSystem initializes correctly."""
        self.assertIsInstance(self.alerting_system, AlertingSystem)
        self.assertEqual(len(self.alerting_system._active_alerts), 0)
        self.assertEqual(len(self.alerting_system._alert_history), 0)

    def test_conversation_metrics_alert_generation(self):
        """Test alert generation from conversation metrics."""
        # Create metrics that should trigger warning
        warning_metrics = ConversationMetrics(
            conversation_id="warning_conv",
            response_time_ms=1500.0,  # Above warning threshold (1000ms)
            turn_number=1,
            tokens_used=100,
            context_size=5000,
            timestamp=time.time(),
            process_type="sonnet_tactical",
            success=True
        )

        alert = self.alerting_system.check_conversation_metrics(warning_metrics)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, AlertSeverity.WARNING)
        self.assertIn("Slow response time", alert.message)

        # Create metrics that should trigger critical alert
        critical_metrics = ConversationMetrics(
            conversation_id="critical_conv",
            response_time_ms=2500.0,  # Above critical threshold (2000ms)
            turn_number=1,
            tokens_used=100,
            context_size=5000,
            timestamp=time.time(),
            process_type="sonnet_tactical",
            success=True
        )

        critical_alert = self.alerting_system.check_conversation_metrics(critical_metrics)
        self.assertIsNotNone(critical_alert)
        self.assertEqual(critical_alert.severity, AlertSeverity.CRITICAL)
        self.assertIn("Critical response time", critical_alert.message)

    def test_efficiency_metrics_alert_generation(self):
        """Test alert generation from efficiency metrics."""
        # Create efficiency metrics that should trigger critical alert
        critical_efficiency = ConversationEfficiencyMetrics(
            conversation_type="sonnet_tactical",
            avg_response_time_ms=1200.0,
            avg_turns_per_conversation=5.0,
            completion_rate=0.9,
            efficiency_score=0.5,  # Below critical threshold (0.6)
            timestamp=time.time()
        )

        alert = self.alerting_system.check_efficiency_metrics(critical_efficiency)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, AlertSeverity.CRITICAL)
        self.assertIn("Critical turn efficiency", alert.message)

    def test_compression_metrics_alert_generation(self):
        """Test alert generation from compression metrics."""
        # Create compression metrics that should trigger warning
        poor_compression = CompressionEffectivenessMetrics(
            compression_id="poor_comp",
            original_size=10000,
            compressed_size=4000,  # 40% compression (poor)
            compression_ratio=0.4,
            compression_time_ms=500.0,
            effectiveness_score=0.6,
            critical_info_preserved=True,
            timestamp=time.time()
        )

        alert = self.alerting_system.check_compression_metrics(poor_compression)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, AlertSeverity.WARNING)
        self.assertIn("compression ratio", alert.message)

    def test_alert_callback_functionality(self):
        """Test alert callback functionality."""
        callback_called = []

        def test_callback(alert: PerformanceAlert):
            callback_called.append(alert)

        self.alerting_system.add_alert_callback(test_callback)

        # Trigger an alert
        metrics = ConversationMetrics(
            conversation_id="callback_test",
            response_time_ms=2500.0,  # Critical
            turn_number=1,
            tokens_used=100,
            context_size=5000,
            timestamp=time.time(),
            process_type="sonnet_tactical",
            success=True
        )

        alert = self.alerting_system.check_conversation_metrics(metrics)

        # Verify callback was called
        self.assertEqual(len(callback_called), 1)
        self.assertEqual(callback_called[0].alert_id, alert.alert_id)

    def test_active_alerts_management(self):
        """Test active alerts management."""
        # Generate multiple alerts
        for i in range(3):
            metrics = ConversationMetrics(
                conversation_id=f"alert_conv_{i}",
                response_time_ms=2500.0,
                turn_number=1,
                tokens_used=100,
                context_size=5000,
                timestamp=time.time(),
                process_type="sonnet_tactical",
                success=True
            )
            self.alerting_system.check_conversation_metrics(metrics)

        active_alerts = self.alerting_system.get_active_alerts()
        self.assertEqual(len(active_alerts), 3)

        # All alerts should be unresolved
        for alert in active_alerts:
            self.assertFalse(alert.resolved)

    def test_alert_summary_generation(self):
        """Test alert summary generation."""
        # Generate various types of alerts
        critical_metrics = ConversationMetrics(
            conversation_id="critical_test",
            response_time_ms=2500.0,
            turn_number=1,
            tokens_used=100,
            context_size=5000,
            timestamp=time.time(),
            process_type="sonnet_tactical",
            success=True
        )
        self.alerting_system.check_conversation_metrics(critical_metrics)

        warning_metrics = ConversationMetrics(
            conversation_id="warning_test",
            response_time_ms=1200.0,
            turn_number=1,
            tokens_used=100,
            context_size=5000,
            timestamp=time.time(),
            process_type="sonnet_tactical",
            success=True
        )
        self.alerting_system.check_conversation_metrics(warning_metrics)

        summary = self.alerting_system.get_alert_summary(time_window_minutes=60)

        self.assertEqual(summary["total_alerts"], 2)
        self.assertEqual(summary["critical_alerts"], 1)
        self.assertEqual(summary["warning_alerts"], 1)
        self.assertEqual(summary["info_alerts"], 0)
        self.assertEqual(summary["active_alerts"], 2)
        self.assertEqual(summary["resolved_alerts"], 0)


@pytest.mark.fast
class TestExternalMetricsExporter(unittest.TestCase):
    """Test external metrics export functionality."""

    def setUp(self):
        """Set up test environment."""
        self.exporter = ExternalMetricsExporter()

    def test_exporter_initialization(self):
        """Test ExternalMetricsExporter initializes correctly."""
        self.assertIsInstance(self.exporter, ExternalMetricsExporter)
        self.assertEqual(len(self.exporter._export_callbacks), 0)

    def test_register_exporter_callback(self):
        """Test registering exporter callbacks."""
        callback_called = []

        def test_exporter(metrics: dict[str, Any]):
            callback_called.append(metrics)

        self.exporter.register_exporter("test_exporter", test_exporter)

        self.assertEqual(len(self.exporter._export_callbacks), 1)
        self.assertIn("test_exporter", self.exporter._export_callbacks)

    def test_export_metrics_to_callbacks(self):
        """Test exporting metrics to registered callbacks."""
        callback_results = []

        def test_exporter_1(metrics: dict[str, Any]):
            callback_results.append(("exporter_1", metrics))

        def test_exporter_2(metrics: dict[str, Any]):
            callback_results.append(("exporter_2", metrics))

        self.exporter.register_exporter("exporter_1", test_exporter_1)
        self.exporter.register_exporter("exporter_2", test_exporter_2)

        test_metrics = {"test_metric": 123, "another_metric": "value"}
        self.exporter.export_metrics(test_metrics)

        self.assertEqual(len(callback_results), 2)
        self.assertEqual(callback_results[0][0], "exporter_1")
        self.assertEqual(callback_results[0][1], test_metrics)
        self.assertEqual(callback_results[1][0], "exporter_2")
        self.assertEqual(callback_results[1][1], test_metrics)

    def test_export_to_json_file(self):
        """Test exporting metrics to JSON file."""
        test_metrics = {
            "system_metrics": {"total_processes": 5},
            "conversation_performance": {"avg_response_time": 1200.0},
            "timestamp": time.time()
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            self.exporter.export_to_json_file(test_metrics, temp_path)

            # Verify file was created and contains expected data
            with open(temp_path) as f:
                exported_data = json.load(f)

            self.assertEqual(exported_data["system_metrics"]["total_processes"], 5)
            self.assertEqual(exported_data["conversation_performance"]["avg_response_time"], 1200.0)
            self.assertIn("timestamp", exported_data)

        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(temp_path)
            except:
                pass

    def test_export_callback_error_handling(self):
        """Test error handling in export callbacks."""
        def failing_exporter(metrics: dict[str, Any]):
            raise Exception("Export failed")

        def working_exporter(metrics: dict[str, Any]):
            # This should still work despite the failing exporter
            pass

        self.exporter.register_exporter("failing", failing_exporter)
        self.exporter.register_exporter("working", working_exporter)

        # Should not raise exception despite failing exporter
        test_metrics = {"test": "data"}
        self.exporter.export_metrics(test_metrics)  # Should complete without error


@pytest.mark.medium
class TestClaudeCodeManagerIntegration(unittest.TestCase):
    """Test integration of performance monitoring with ClaudeCodeManager."""

    def setUp(self):
        """Set up test environment."""
        # Create ClaudeCodeManager with performance monitoring enabled
        self.manager = ClaudeCodeManager(
            max_workers=2,
            enable_performance_monitoring=True
        )

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'manager'):
            self.manager.shutdown()

    def test_claude_code_manager_with_performance_monitoring(self):
        """Test ClaudeCodeManager initialization with performance monitoring."""
        self.assertIsNotNone(self.manager.performance_monitor)
        self.assertIsInstance(self.manager.performance_monitor, PerformanceMonitor)

    def test_claude_code_manager_without_performance_monitoring(self):
        """Test ClaudeCodeManager initialization without performance monitoring."""
        manager_no_monitoring = ClaudeCodeManager(
            max_workers=2,
            enable_performance_monitoring=False
        )

        try:
            self.assertIsNone(manager_no_monitoring.performance_monitor)
        finally:
            manager_no_monitoring.shutdown()

    def test_record_conversation_exchange_integration(self):
        """Test recording conversation exchanges through ClaudeCodeManager."""
        self.manager.record_conversation_exchange(
            conversation_id="integration_test",
            response_time_ms=1500.0,
            turn_number=1,
            tokens_used=100,
            context_size=5000,
            process_type="sonnet_tactical",
            success=True
        )

        # Verify metrics were recorded
        recent_metrics = self.manager.performance_monitor.conversation_tracker.get_recent_metrics(1)
        self.assertEqual(len(recent_metrics), 1)
        self.assertEqual(recent_metrics[0].conversation_id, "integration_test")

    def test_record_compression_event_integration(self):
        """Test recording compression events through ClaudeCodeManager."""
        self.manager.record_compression_event(
            compression_id="integration_compression",
            original_size=10000,
            compressed_size=3000,
            compression_time_ms=800.0,
            critical_info_preserved=True
        )

        # Verify compression metrics were recorded
        analytics = self.manager.performance_monitor.compression_monitor.get_compression_analytics(60)
        self.assertEqual(analytics["total_compressions"], 1)

    def test_get_comprehensive_performance_metrics(self):
        """Test getting comprehensive performance metrics."""
        # Record some test data
        self.manager.record_conversation_exchange(
            conversation_id="metrics_test",
            response_time_ms=1200.0,
            turn_number=1,
            tokens_used=150,
            context_size=6000,
            process_type="opus_strategic",
            success=True
        )

        # Get comprehensive metrics
        metrics = self.manager.get_performance_metrics(time_window_minutes=60)

        self.assertIn("report_timestamp", metrics)
        self.assertIn("system_metrics", metrics)
        self.assertIn("conversation_performance", metrics)
        self.assertIn("compression_effectiveness", metrics)
        self.assertIn("system_reliability", metrics)
        self.assertIn("alerts", metrics)
        self.assertIn("analytics", metrics)

    def test_get_active_performance_alerts(self):
        """Test getting active performance alerts."""
        # Initially no alerts
        alerts = self.manager.get_active_performance_alerts()
        self.assertEqual(len(alerts), 0)

        # Record metrics that should trigger alerts
        self.manager.record_conversation_exchange(
            conversation_id="alert_test",
            response_time_ms=6000.0,  # Should trigger critical alert
            turn_number=1,
            tokens_used=100,
            context_size=5000,
            process_type="sonnet_tactical",
            success=True
        )

        alerts = self.manager.get_active_performance_alerts()
        self.assertGreater(len(alerts), 0)

    def test_configure_alert_callback_integration(self):
        """Test configuring alert callbacks through ClaudeCodeManager."""
        callback_calls = []

        def test_alert_callback(alert):
            callback_calls.append(alert)

        self.manager.configure_alert_callback(test_alert_callback)

        # Trigger an alert
        self.manager.record_conversation_exchange(
            conversation_id="callback_integration_test",
            response_time_ms=8000.0,  # Critical alert
            turn_number=1,
            tokens_used=100,
            context_size=5000,
            process_type="sonnet_tactical",
            success=True
        )

        # Verify callback was called
        self.assertGreater(len(callback_calls), 0)

    def test_export_metrics_to_file_integration(self):
        """Test exporting metrics to file through ClaudeCodeManager."""
        # Record some test metrics
        self.manager.record_conversation_exchange(
            conversation_id="export_test",
            response_time_ms=1800.0,
            turn_number=2,
            tokens_used=200,
            context_size=7000,
            process_type="opus_strategic",
            success=True
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            self.manager.export_metrics_to_file(temp_path)

            # Verify file was created and contains metrics
            with open(temp_path) as f:
                exported_metrics = json.load(f)

            self.assertIn("report_timestamp", exported_metrics)
            self.assertIn("system_metrics", exported_metrics)
            self.assertIn("conversation_performance", exported_metrics)

        finally:
            import os
            try:
                os.unlink(temp_path)
            except:
                pass


@pytest.mark.slow
class TestPerformanceMonitoringBenchmarks(unittest.TestCase):
    """Performance benchmarks for the monitoring system itself."""

    def setUp(self):
        """Set up test environment."""
        self.aggregated_collector = AggregatedMetricsCollector()
        self.performance_monitor = PerformanceMonitor(
            aggregated_collector=self.aggregated_collector,
            enable_real_time_monitoring=False
        )

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.stop_monitoring()

    def test_conversation_tracking_performance(self):
        """Test conversation tracking performance overhead."""
        conversation_id = "perf_test"
        num_exchanges = 100

        # Benchmark conversation exchange recording
        start_time = time.perf_counter()
        for i in range(num_exchanges):
            self.performance_monitor.record_conversation_exchange(
                conversation_id=f"{conversation_id}_{i}",
                response_time_ms=1200.0,
                turn_number=1,
                tokens_used=100,
                context_size=5000,
                process_type="sonnet_tactical",
                success=True
            )
        total_time = (time.perf_counter() - start_time) * 1000

        avg_time_per_recording = total_time / num_exchanges

        print("\n=== Conversation Tracking Performance ===")
        print(f"Total time for {num_exchanges} recordings: {total_time:.2f}ms")
        print(f"Average time per recording: {avg_time_per_recording:.3f}ms")
        print("Target: <1ms per recording")

        # Performance assertion - should be well under 1ms per recording
        self.assertLess(avg_time_per_recording, 1.0,
                       f"Conversation tracking overhead too high: {avg_time_per_recording:.3f}ms")

    def test_compression_monitoring_performance(self):
        """Test compression monitoring performance overhead."""
        num_compressions = 50

        start_time = time.perf_counter()
        for i in range(num_compressions):
            self.performance_monitor.record_compression_event(
                compression_id=f"perf_compression_{i}",
                original_size=10000,
                compressed_size=3000,
                compression_time_ms=800.0,
                critical_info_preserved=True
            )
        total_time = (time.perf_counter() - start_time) * 1000

        avg_time_per_recording = total_time / num_compressions

        print("\n=== Compression Monitoring Performance ===")
        print(f"Total time for {num_compressions} recordings: {total_time:.2f}ms")
        print(f"Average time per recording: {avg_time_per_recording:.3f}ms")
        print("Target: <2ms per recording")

        # Performance assertion
        self.assertLess(avg_time_per_recording, 2.0,
                       f"Compression monitoring overhead too high: {avg_time_per_recording:.3f}ms")

    def test_comprehensive_report_generation_performance(self):
        """Test performance of comprehensive report generation."""
        # Create test data
        for i in range(50):
            self.performance_monitor.record_conversation_exchange(
                conversation_id=f"report_test_{i}",
                response_time_ms=1200.0 + i * 10,
                turn_number=i % 10 + 1,
                tokens_used=100,
                context_size=5000,
                process_type="sonnet_tactical" if i % 2 == 0 else "opus_strategic",
                success=True
            )

        for i in range(20):
            self.performance_monitor.record_compression_event(
                compression_id=f"report_compression_{i}",
                original_size=10000,
                compressed_size=3000 + i * 100,
                compression_time_ms=800.0 + i * 50,
                critical_info_preserved=True
            )

        # Benchmark report generation
        start_time = time.perf_counter()
        report = self.performance_monitor.get_comprehensive_performance_report(60)
        report_time = (time.perf_counter() - start_time) * 1000

        print("\n=== Comprehensive Report Generation Performance ===")
        print(f"Report generation time: {report_time:.2f}ms")
        print(f"Report size: {len(str(report))} characters")
        print("Target: <100ms for report generation")

        # Performance assertion
        self.assertLess(report_time, 100.0,
                       f"Report generation too slow: {report_time:.2f}ms")

        # Verify report completeness
        self.assertIn("conversation_performance", report)
        self.assertIn("compression_effectiveness", report)
        self.assertIn("system_reliability", report)


if __name__ == "__main__":
    unittest.main()
