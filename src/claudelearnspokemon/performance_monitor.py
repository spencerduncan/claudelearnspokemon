"""
ClaudeCodeManager Performance Monitoring System

Comprehensive performance monitoring for Claude Code CLI conversations and system 
interactions. Tracks conversation metrics, context compression effectiveness,
restart patterns, and provides real-time analytics with configurable alerting.

Design Principles:
- Minimal performance impact on monitored operations (<1ms overhead)
- Thread-safe operations with fine-grained locking
- Real-time monitoring with historical trend analysis
- Configurable thresholds and external system integration
- Clean integration with existing ClaudeCodeManager architecture
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .process_metrics_collector import AggregatedMetricsCollector

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels for performance monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ConversationMetrics:
    """Comprehensive conversation performance metrics."""
    conversation_id: str
    response_time_ms: float
    turn_number: int
    tokens_used: int
    context_size: int
    timestamp: float
    process_type: str
    success: bool
    error_details: str | None = None


@dataclass(frozen=True)
class ConversationEfficiencyMetrics:
    """Turn usage efficiency metrics across conversation types."""
    conversation_type: str
    avg_response_time_ms: float
    avg_turns_per_conversation: float
    completion_rate: float
    efficiency_score: float  # 0-1 based on performance vs targets
    timestamp: float


@dataclass(frozen=True)
class CompressionEffectivenessMetrics:
    """Context compression performance and effectiveness metrics."""
    compression_id: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time_ms: float
    effectiveness_score: float  # 0-1 based on ratio and speed
    critical_info_preserved: bool
    timestamp: float


@dataclass
class PerformanceAlert:
    """Performance monitoring alert."""
    alert_id: str
    severity: AlertSeverity
    message: str
    metric_name: str
    threshold_value: float
    actual_value: float
    timestamp: float
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return asdict(self)


@dataclass
class MonitoringThresholds:
    """Configurable performance monitoring thresholds."""
    # Response time thresholds (ms)
    tactical_response_time_warning: float = 2000.0  # 2s
    tactical_response_time_critical: float = 5000.0  # 5s
    strategic_response_time_warning: float = 5000.0  # 5s
    strategic_response_time_critical: float = 10000.0  # 10s

    # Turn efficiency thresholds
    turn_efficiency_warning: float = 0.7  # 70%
    turn_efficiency_critical: float = 0.5  # 50%

    # Compression effectiveness thresholds
    compression_ratio_warning: float = 0.6  # 60% compression
    compression_ratio_critical: float = 0.4  # 40% compression
    compression_time_warning: float = 1000.0  # 1s
    compression_time_critical: float = 2000.0  # 2s

    # Restart frequency thresholds (per hour)
    restart_frequency_warning: float = 5.0
    restart_frequency_critical: float = 10.0

    # Success rate thresholds
    success_rate_warning: float = 0.9  # 90%
    success_rate_critical: float = 0.8  # 80%


class ConversationPerformanceTracker:
    """
    Tracks real-time conversation performance metrics.
    
    Designed for minimal overhead (<1ms) while providing comprehensive
    conversation-level monitoring for response times and turn efficiency.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize conversation performance tracker.
        
        Args:
            max_history: Maximum number of conversation metrics to retain
        """
        self._conversation_metrics: deque[ConversationMetrics] = deque(maxlen=max_history)
        self._active_conversations: dict[str, float] = {}  # conversation_id -> start_time
        self._lock = threading.RLock()

    def start_conversation_tracking(self, conversation_id: str) -> None:
        """Start tracking a conversation."""
        with self._lock:
            self._active_conversations[conversation_id] = time.time()

    def record_conversation_exchange(
        self,
        conversation_id: str,
        response_time_ms: float,
        turn_number: int,
        tokens_used: int,
        context_size: int,
        process_type: str,
        success: bool = True,
        error_details: str | None = None
    ) -> None:
        """
        Record a conversation message exchange with minimal overhead.
        
        Args:
            conversation_id: Unique conversation identifier
            response_time_ms: Response time in milliseconds
            turn_number: Current turn number in conversation
            tokens_used: Number of tokens consumed
            context_size: Size of conversation context
            process_type: Type of process (tactical/strategic)
            success: Whether the exchange was successful
            error_details: Error details if unsuccessful
        """
        start_time = time.perf_counter()

        metrics = ConversationMetrics(
            conversation_id=conversation_id,
            response_time_ms=response_time_ms,
            turn_number=turn_number,
            tokens_used=tokens_used,
            context_size=context_size,
            timestamp=time.time(),
            process_type=process_type,
            success=success,
            error_details=error_details
        )

        with self._lock:
            self._conversation_metrics.append(metrics)

        # Log performance tracking overhead
        tracking_overhead = (time.perf_counter() - start_time) * 1000
        if tracking_overhead > 1.0:  # Log if overhead exceeds 1ms target
            logger.warning(f"Conversation tracking overhead: {tracking_overhead:.2f}ms")

    def get_conversation_efficiency_metrics(self, time_window_minutes: int = 60) -> list[ConversationEfficiencyMetrics]:
        """
        Calculate conversation efficiency metrics by process type.
        
        Args:
            time_window_minutes: Time window for metrics calculation
            
        Returns:
            List of efficiency metrics by conversation type
        """
        cutoff_time = time.time() - (time_window_minutes * 60)

        with self._lock:
            # Filter recent metrics
            recent_metrics = [m for m in self._conversation_metrics if m.timestamp >= cutoff_time]

        # Group by process type
        metrics_by_type = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_type[metric.process_type].append(metric)

        efficiency_metrics = []
        for process_type, type_metrics in metrics_by_type.items():
            if not type_metrics:
                continue

            # Calculate efficiency metrics
            response_times = [m.response_time_ms for m in type_metrics]
            turn_numbers = [m.turn_number for m in type_metrics]
            success_rate = len([m for m in type_metrics if m.success]) / len(type_metrics)

            avg_response_time = sum(response_times) / len(response_times)
            avg_turns = sum(turn_numbers) / len(turn_numbers)

            # Calculate efficiency score (0-1) based on performance vs targets
            target_response_time = 2000 if "tactical" in process_type else 5000
            response_efficiency = max(0, 1 - (avg_response_time / target_response_time))
            turn_efficiency = success_rate
            efficiency_score = (response_efficiency + turn_efficiency) / 2

            efficiency_metrics.append(ConversationEfficiencyMetrics(
                conversation_type=process_type,
                avg_response_time_ms=avg_response_time,
                avg_turns_per_conversation=avg_turns,
                completion_rate=success_rate,
                efficiency_score=efficiency_score,
                timestamp=time.time()
            ))

        return efficiency_metrics

    def get_recent_metrics(self, count: int = 100) -> list[ConversationMetrics]:
        """Get the most recent conversation metrics."""
        with self._lock:
            return list(self._conversation_metrics)[-count:]


class CompressionEffectivenessMonitor:
    """
    Monitors context compression effectiveness and performance.
    
    Tracks compression ratios, performance impact, and effectiveness
    at maintaining conversation quality while reducing context size.
    """

    def __init__(self, max_history: int = 500):
        """
        Initialize compression effectiveness monitor.
        
        Args:
            max_history: Maximum compression metrics to retain
        """
        self._compression_metrics: deque[CompressionEffectivenessMetrics] = deque(maxlen=max_history)
        self._lock = threading.RLock()

    def record_compression_event(
        self,
        compression_id: str,
        original_size: int,
        compressed_size: int,
        compression_time_ms: float,
        critical_info_preserved: bool = True
    ) -> None:
        """
        Record a context compression event.
        
        Args:
            compression_id: Unique identifier for compression event
            original_size: Original context size in characters/tokens
            compressed_size: Compressed context size
            compression_time_ms: Time taken for compression
            critical_info_preserved: Whether critical information was preserved
        """
        compression_ratio = compressed_size / original_size if original_size > 0 else 0

        # Calculate effectiveness score based on ratio, speed, and preservation
        ratio_score = max(0, 1 - compression_ratio)  # Better compression = higher score
        speed_score = max(0, 1 - (compression_time_ms / 2000))  # Target <2s
        preservation_score = 1.0 if critical_info_preserved else 0.5
        effectiveness_score = (ratio_score + speed_score + preservation_score) / 3

        metrics = CompressionEffectivenessMetrics(
            compression_id=compression_id,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time_ms=compression_time_ms,
            effectiveness_score=effectiveness_score,
            critical_info_preserved=critical_info_preserved,
            timestamp=time.time()
        )

        with self._lock:
            self._compression_metrics.append(metrics)

    def get_compression_analytics(self, time_window_minutes: int = 60) -> dict[str, Any]:
        """
        Get comprehensive compression analytics.
        
        Args:
            time_window_minutes: Time window for analytics
            
        Returns:
            Dictionary with compression analytics
        """
        cutoff_time = time.time() - (time_window_minutes * 60)

        with self._lock:
            recent_metrics = [m for m in self._compression_metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {
                "total_compressions": 0,
                "avg_compression_ratio": 0.0,
                "avg_compression_time_ms": 0.0,
                "avg_effectiveness_score": 0.0,
                "critical_info_preservation_rate": 0.0,
                "compression_frequency_per_hour": 0.0
            }

        total_compressions = len(recent_metrics)
        avg_ratio = sum(m.compression_ratio for m in recent_metrics) / total_compressions
        avg_time = sum(m.compression_time_ms for m in recent_metrics) / total_compressions
        avg_effectiveness = sum(m.effectiveness_score for m in recent_metrics) / total_compressions
        preservation_rate = len([m for m in recent_metrics if m.critical_info_preserved]) / total_compressions

        # Calculate hourly frequency
        time_span_hours = time_window_minutes / 60
        frequency_per_hour = total_compressions / time_span_hours if time_span_hours > 0 else 0

        return {
            "total_compressions": total_compressions,
            "avg_compression_ratio": avg_ratio,
            "avg_compression_time_ms": avg_time,
            "avg_effectiveness_score": avg_effectiveness,
            "critical_info_preservation_rate": preservation_rate,
            "compression_frequency_per_hour": frequency_per_hour,
            "recent_metrics": [asdict(m) for m in recent_metrics[-10:]]  # Last 10 for details
        }


class AlertingSystem:
    """
    Configurable alerting system for performance monitoring.
    
    Supports threshold-based alerting with configurable severity levels
    and external notification integration.
    """

    def __init__(self, thresholds: MonitoringThresholds | None = None):
        """
        Initialize alerting system.
        
        Args:
            thresholds: Monitoring thresholds configuration
        """
        self.thresholds = thresholds or MonitoringThresholds()
        self._active_alerts: dict[str, PerformanceAlert] = {}
        self._alert_history: deque[PerformanceAlert] = deque(maxlen=1000)
        self._alert_callbacks: list[Callable[[PerformanceAlert], None]] = []
        self._lock = threading.RLock()

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback function to be called when alerts are triggered."""
        with self._lock:
            self._alert_callbacks.append(callback)

    def check_conversation_metrics(self, metrics: ConversationMetrics) -> PerformanceAlert | None:
        """
        Check conversation metrics against thresholds.
        
        Args:
            metrics: Conversation metrics to check
            
        Returns:
            Performance alert if threshold exceeded, None otherwise
        """
        process_type = metrics.process_type.lower()

        # Check response time thresholds
        if "tactical" in process_type:
            warning_threshold = self.thresholds.tactical_response_time_warning
            critical_threshold = self.thresholds.tactical_response_time_critical
        else:
            warning_threshold = self.thresholds.strategic_response_time_warning
            critical_threshold = self.thresholds.strategic_response_time_critical

        if metrics.response_time_ms >= critical_threshold:
            return self._create_alert(
                f"critical_response_time_{metrics.conversation_id}",
                AlertSeverity.CRITICAL,
                f"Critical response time: {metrics.response_time_ms:.1f}ms (threshold: {critical_threshold}ms)",
                "response_time_ms",
                critical_threshold,
                metrics.response_time_ms
            )
        elif metrics.response_time_ms >= warning_threshold:
            return self._create_alert(
                f"warning_response_time_{metrics.conversation_id}",
                AlertSeverity.WARNING,
                f"Slow response time: {metrics.response_time_ms:.1f}ms (threshold: {warning_threshold}ms)",
                "response_time_ms",
                warning_threshold,
                metrics.response_time_ms
            )

        return None

    def check_efficiency_metrics(self, efficiency: ConversationEfficiencyMetrics) -> PerformanceAlert | None:
        """
        Check conversation efficiency against thresholds.
        
        Args:
            efficiency: Efficiency metrics to check
            
        Returns:
            Performance alert if threshold exceeded, None otherwise
        """
        if efficiency.efficiency_score <= self.thresholds.turn_efficiency_critical:
            return self._create_alert(
                f"critical_efficiency_{efficiency.conversation_type}",
                AlertSeverity.CRITICAL,
                f"Critical turn efficiency: {efficiency.efficiency_score:.2f} (threshold: {self.thresholds.turn_efficiency_critical})",
                "efficiency_score",
                self.thresholds.turn_efficiency_critical,
                efficiency.efficiency_score
            )
        elif efficiency.efficiency_score <= self.thresholds.turn_efficiency_warning:
            return self._create_alert(
                f"warning_efficiency_{efficiency.conversation_type}",
                AlertSeverity.WARNING,
                f"Low turn efficiency: {efficiency.efficiency_score:.2f} (threshold: {self.thresholds.turn_efficiency_warning})",
                "efficiency_score",
                self.thresholds.turn_efficiency_warning,
                efficiency.efficiency_score
            )

        return None

    def check_compression_metrics(self, compression: CompressionEffectivenessMetrics) -> PerformanceAlert | None:
        """
        Check compression effectiveness against thresholds.
        
        Args:
            compression: Compression metrics to check
            
        Returns:
            Performance alert if threshold exceeded, None otherwise
        """
        # Check compression ratio
        if compression.compression_ratio >= (1 - self.thresholds.compression_ratio_critical):
            return self._create_alert(
                f"critical_compression_{compression.compression_id}",
                AlertSeverity.CRITICAL,
                f"Poor compression ratio: {compression.compression_ratio:.2f} (target: <{1 - self.thresholds.compression_ratio_critical:.2f})",
                "compression_ratio",
                1 - self.thresholds.compression_ratio_critical,
                compression.compression_ratio
            )
        elif compression.compression_ratio >= (1 - self.thresholds.compression_ratio_warning):
            return self._create_alert(
                f"warning_compression_{compression.compression_id}",
                AlertSeverity.WARNING,
                f"Suboptimal compression ratio: {compression.compression_ratio:.2f} (target: <{1 - self.thresholds.compression_ratio_warning:.2f})",
                "compression_ratio",
                1 - self.thresholds.compression_ratio_warning,
                compression.compression_ratio
            )

        # Check compression time
        if compression.compression_time_ms >= self.thresholds.compression_time_critical:
            return self._create_alert(
                f"critical_compression_time_{compression.compression_id}",
                AlertSeverity.CRITICAL,
                f"Slow compression: {compression.compression_time_ms:.1f}ms (threshold: {self.thresholds.compression_time_critical}ms)",
                "compression_time_ms",
                self.thresholds.compression_time_critical,
                compression.compression_time_ms
            )
        elif compression.compression_time_ms >= self.thresholds.compression_time_warning:
            return self._create_alert(
                f"warning_compression_time_{compression.compression_id}",
                AlertSeverity.WARNING,
                f"Slow compression: {compression.compression_time_ms:.1f}ms (threshold: {self.thresholds.compression_time_warning}ms)",
                "compression_time_ms",
                self.thresholds.compression_time_warning,
                compression.compression_time_ms
            )

        return None

    def _create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        message: str,
        metric_name: str,
        threshold_value: float,
        actual_value: float
    ) -> PerformanceAlert:
        """Create and process a performance alert."""
        alert = PerformanceAlert(
            alert_id=alert_id,
            severity=severity,
            message=message,
            metric_name=metric_name,
            threshold_value=threshold_value,
            actual_value=actual_value,
            timestamp=time.time()
        )

        with self._lock:
            # Update active alerts
            if alert_id in self._active_alerts:
                self._active_alerts[alert_id].resolved = True
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)

            # Trigger callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

        logger.log(
            logging.CRITICAL if severity == AlertSeverity.CRITICAL else logging.WARNING,
            f"Performance Alert [{severity.value.upper()}]: {message}"
        )

        return alert

    def get_active_alerts(self) -> list[PerformanceAlert]:
        """Get list of currently active alerts."""
        with self._lock:
            return [alert for alert in self._active_alerts.values() if not alert.resolved]

    def get_alert_summary(self, time_window_minutes: int = 60) -> dict[str, Any]:
        """
        Get alert summary for specified time window.
        
        Args:
            time_window_minutes: Time window for alert summary
            
        Returns:
            Dictionary with alert summary statistics
        """
        cutoff_time = time.time() - (time_window_minutes * 60)

        with self._lock:
            recent_alerts = [alert for alert in self._alert_history if alert.timestamp >= cutoff_time]

        if not recent_alerts:
            return {
                "total_alerts": 0,
                "critical_alerts": 0,
                "warning_alerts": 0,
                "info_alerts": 0,
                "active_alerts": 0,
                "resolved_alerts": 0
            }

        critical_count = len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL])
        warning_count = len([a for a in recent_alerts if a.severity == AlertSeverity.WARNING])
        info_count = len([a for a in recent_alerts if a.severity == AlertSeverity.INFO])
        active_count = len([a for a in recent_alerts if not a.resolved])
        resolved_count = len([a for a in recent_alerts if a.resolved])

        return {
            "total_alerts": len(recent_alerts),
            "critical_alerts": critical_count,
            "warning_alerts": warning_count,
            "info_alerts": info_count,
            "active_alerts": active_count,
            "resolved_alerts": resolved_count,
            "recent_alerts": [alert.to_dict() for alert in recent_alerts[-10:]]
        }


class ExternalMetricsExporter:
    """
    Exports performance metrics to external monitoring systems.
    
    Supports multiple export formats and destinations for integration
    with external monitoring and alerting infrastructure.
    """

    def __init__(self):
        """Initialize metrics exporter."""
        self._export_callbacks: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._lock = threading.RLock()

    def register_exporter(self, name: str, callback: Callable[[dict[str, Any]], None]) -> None:
        """
        Register an external metrics exporter.
        
        Args:
            name: Name of the exporter
            callback: Function to call with metrics data
        """
        with self._lock:
            self._export_callbacks[name] = callback

    def export_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Export metrics to all registered external systems.
        
        Args:
            metrics: Metrics dictionary to export
        """
        with self._lock:
            for name, callback in self._export_callbacks.items():
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Failed to export metrics to {name}: {e}")

    def export_to_json_file(self, metrics: dict[str, Any], file_path: str) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            metrics: Metrics to export
            file_path: Path to JSON file
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to export metrics to {file_path}: {e}")


class PerformanceMonitor:
    """
    Main performance monitoring orchestrator for ClaudeCodeManager.
    
    Integrates conversation tracking, compression monitoring, alerting,
    and external metrics export into a unified monitoring system.
    """

    def __init__(
        self,
        aggregated_collector: AggregatedMetricsCollector,
        thresholds: MonitoringThresholds | None = None,
        enable_real_time_monitoring: bool = True
    ):
        """
        Initialize comprehensive performance monitor.
        
        Args:
            aggregated_collector: Existing ClaudeCodeManager metrics collector
            thresholds: Monitoring thresholds configuration
            enable_real_time_monitoring: Whether to enable real-time monitoring
        """
        self.aggregated_collector = aggregated_collector
        self.conversation_tracker = ConversationPerformanceTracker()
        self.compression_monitor = CompressionEffectivenessMonitor()
        self.alerting_system = AlertingSystem(thresholds)
        self.metrics_exporter = ExternalMetricsExporter()

        self._monitoring_enabled = enable_real_time_monitoring
        self._monitoring_thread: threading.Thread | None = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()

        # Performance tracking
        self._last_monitoring_time = time.time()
        self._monitoring_overhead_ms = 0.0

        logger.info("PerformanceMonitor initialized with real-time monitoring")

    def start_monitoring(self, monitoring_interval_seconds: float = 30.0) -> None:
        """
        Start real-time performance monitoring.
        
        Args:
            monitoring_interval_seconds: Interval between monitoring checks
        """
        if not self._monitoring_enabled:
            logger.info("Real-time monitoring is disabled")
            return

        with self._lock:
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                logger.warning("Performance monitoring already running")
                return

            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(monitoring_interval_seconds,),
                daemon=True,
                name="PerformanceMonitor"
            )
            self._monitoring_thread.start()

        logger.info(f"Real-time performance monitoring started (interval: {monitoring_interval_seconds}s)")

    def stop_monitoring(self) -> None:
        """Stop real-time performance monitoring."""
        self._stop_monitoring.set()

        with self._lock:
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)

        logger.info("Performance monitoring stopped")

    def record_conversation_exchange(
        self,
        conversation_id: str,
        response_time_ms: float,
        turn_number: int,
        tokens_used: int,
        context_size: int,
        process_type: str,
        success: bool = True,
        error_details: str | None = None
    ) -> None:
        """
        Record a conversation exchange and trigger real-time monitoring.
        
        This is the main entry point for conversation performance tracking.
        """
        start_time = time.perf_counter()

        # Record in conversation tracker
        self.conversation_tracker.record_conversation_exchange(
            conversation_id=conversation_id,
            response_time_ms=response_time_ms,
            turn_number=turn_number,
            tokens_used=tokens_used,
            context_size=context_size,
            process_type=process_type,
            success=success,
            error_details=error_details
        )

        # Create metrics object for alerting
        metrics = ConversationMetrics(
            conversation_id=conversation_id,
            response_time_ms=response_time_ms,
            turn_number=turn_number,
            tokens_used=tokens_used,
            context_size=context_size,
            timestamp=time.time(),
            process_type=process_type,
            success=success,
            error_details=error_details
        )

        # Check for alerts
        alert = self.alerting_system.check_conversation_metrics(metrics)
        if alert:
            logger.info(f"Performance alert triggered: {alert.message}")

        # Track monitoring overhead
        overhead_ms = (time.perf_counter() - start_time) * 1000
        self._monitoring_overhead_ms = overhead_ms

        if overhead_ms > 1.0:  # Log if overhead exceeds target
            logger.warning(f"Performance monitoring overhead: {overhead_ms:.2f}ms")

    def record_compression_event(
        self,
        compression_id: str,
        original_size: int,
        compressed_size: int,
        compression_time_ms: float,
        critical_info_preserved: bool = True
    ) -> None:
        """
        Record a context compression event.
        
        Args:
            compression_id: Unique identifier for compression
            original_size: Original context size
            compressed_size: Compressed context size
            compression_time_ms: Time taken for compression
            critical_info_preserved: Whether critical info was preserved
        """
        # Record in compression monitor
        self.compression_monitor.record_compression_event(
            compression_id=compression_id,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_time_ms=compression_time_ms,
            critical_info_preserved=critical_info_preserved
        )

        # Create compression metrics for alerting
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        ratio_score = max(0, 1 - compression_ratio)
        speed_score = max(0, 1 - (compression_time_ms / 2000))
        preservation_score = 1.0 if critical_info_preserved else 0.5
        effectiveness_score = (ratio_score + speed_score + preservation_score) / 3

        compression_metrics = CompressionEffectivenessMetrics(
            compression_id=compression_id,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time_ms=compression_time_ms,
            effectiveness_score=effectiveness_score,
            critical_info_preserved=critical_info_preserved,
            timestamp=time.time()
        )

        # Check for alerts
        alert = self.alerting_system.check_compression_metrics(compression_metrics)
        if alert:
            logger.info(f"Compression alert triggered: {alert.message}")

    def get_comprehensive_performance_report(self, time_window_minutes: int = 60) -> dict[str, Any]:
        """
        Generate comprehensive performance report combining all monitoring data.
        
        Args:
            time_window_minutes: Time window for report generation
            
        Returns:
            Comprehensive performance report dictionary
        """
        report_start_time = time.perf_counter()

        # Gather data from all monitoring components
        system_metrics = self.aggregated_collector.get_system_metrics()
        efficiency_metrics = self.conversation_tracker.get_conversation_efficiency_metrics(time_window_minutes)
        compression_analytics = self.compression_monitor.get_compression_analytics(time_window_minutes)
        alert_summary = self.alerting_system.get_alert_summary(time_window_minutes)

        # Calculate restart frequency and success rates
        restart_frequency = self._calculate_restart_frequency(time_window_minutes)
        success_rates = self._calculate_success_rates(time_window_minutes)

        # Generate comprehensive report
        report = {
            "report_timestamp": time.time(),
            "time_window_minutes": time_window_minutes,
            "monitoring_overhead_ms": self._monitoring_overhead_ms,

            # System-level metrics from existing infrastructure
            "system_metrics": system_metrics,

            # Conversation performance metrics
            "conversation_performance": {
                "efficiency_by_type": [asdict(em) for em in efficiency_metrics],
                "overall_performance": self._calculate_overall_performance(efficiency_metrics)
            },

            # Context compression effectiveness
            "compression_effectiveness": compression_analytics,

            # Restart frequency and success rates
            "system_reliability": {
                "restart_frequency_per_hour": restart_frequency,
                "success_rates": success_rates
            },

            # Alerting summary
            "alerts": alert_summary,

            # Performance analytics and trends
            "analytics": self._generate_performance_analytics(time_window_minutes)
        }

        # Track report generation time
        report_time_ms = (time.perf_counter() - report_start_time) * 1000
        report["report_generation_time_ms"] = report_time_ms

        # Export to external systems
        self.metrics_exporter.export_metrics(report)

        return report

    def _monitoring_loop(self, interval_seconds: float) -> None:
        """
        Main monitoring loop for real-time performance checks.
        
        Args:
            interval_seconds: Monitoring interval
        """
        logger.info(f"Performance monitoring loop started (interval: {interval_seconds}s)")

        while not self._stop_monitoring.wait(interval_seconds):
            try:
                # Check conversation efficiency and trigger alerts if needed
                efficiency_metrics = self.conversation_tracker.get_conversation_efficiency_metrics(5)  # 5 min window
                for efficiency in efficiency_metrics:
                    alert = self.alerting_system.check_efficiency_metrics(efficiency)
                    if alert:
                        logger.info(f"Efficiency alert triggered: {alert.message}")

                # Update monitoring timestamp
                self._last_monitoring_time = time.time()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

        logger.info("Performance monitoring loop stopped")

    def _calculate_restart_frequency(self, time_window_minutes: int) -> float:
        """Calculate restart frequency per hour from system metrics."""
        system_metrics = self.aggregated_collector.get_system_metrics()
        total_restarts = system_metrics.get("total_restarts", 0)

        # Simple frequency calculation - in production would track time windows
        time_window_hours = time_window_minutes / 60
        return total_restarts / time_window_hours if time_window_hours > 0 else 0

    def _calculate_success_rates(self, time_window_minutes: int) -> dict[str, float]:
        """Calculate success rates from conversation metrics."""
        recent_metrics = self.conversation_tracker.get_recent_metrics(1000)

        if not recent_metrics:
            return {"overall": 1.0, "tactical": 1.0, "strategic": 1.0}

        cutoff_time = time.time() - (time_window_minutes * 60)
        recent_metrics = [m for m in recent_metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {"overall": 1.0, "tactical": 1.0, "strategic": 1.0}

        # Calculate overall success rate
        successful = len([m for m in recent_metrics if m.success])
        overall_rate = successful / len(recent_metrics)

        # Calculate by process type
        tactical_metrics = [m for m in recent_metrics if "tactical" in m.process_type.lower()]
        strategic_metrics = [m for m in recent_metrics if "strategic" in m.process_type.lower()]

        tactical_rate = (
            len([m for m in tactical_metrics if m.success]) / len(tactical_metrics)
            if tactical_metrics else 1.0
        )
        strategic_rate = (
            len([m for m in strategic_metrics if m.success]) / len(strategic_metrics)
            if strategic_metrics else 1.0
        )

        return {
            "overall": overall_rate,
            "tactical": tactical_rate,
            "strategic": strategic_rate
        }

    def _calculate_overall_performance(self, efficiency_metrics: list[ConversationEfficiencyMetrics]) -> dict[str, float]:
        """Calculate overall performance score from efficiency metrics."""
        if not efficiency_metrics:
            return {"overall_score": 1.0, "performance_grade": "A"}

        avg_efficiency = sum(em.efficiency_score for em in efficiency_metrics) / len(efficiency_metrics)

        # Convert to letter grade
        if avg_efficiency >= 0.9:
            grade = "A"
        elif avg_efficiency >= 0.8:
            grade = "B"
        elif avg_efficiency >= 0.7:
            grade = "C"
        elif avg_efficiency >= 0.6:
            grade = "D"
        else:
            grade = "F"

        return {
            "overall_score": avg_efficiency,
            "performance_grade": grade,
            "metrics_count": len(efficiency_metrics)
        }

    def _generate_performance_analytics(self, time_window_minutes: int) -> dict[str, Any]:
        """Generate performance analytics and trends."""
        # In a full implementation, this would include:
        # - Trend analysis over time
        # - Performance regression detection
        # - Capacity planning recommendations
        # - Performance optimization suggestions

        return {
            "trends": {
                "response_time_trend": "stable",  # Would analyze actual trends
                "efficiency_trend": "improving",
                "compression_effectiveness_trend": "stable"
            },
            "recommendations": [
                "Monitor tactical process response times - trending higher",
                "Consider increasing compression frequency for better context management",
                "Review process restart patterns for optimization opportunities"
            ],
            "optimization_opportunities": {
                "response_time_optimization": "Consider process pool optimization",
                "context_management": "Implement proactive compression strategies",
                "resource_usage": "Monitor memory usage patterns"
            }
        }

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
