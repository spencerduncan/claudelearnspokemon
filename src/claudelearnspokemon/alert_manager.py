"""
Alert management system for Pokemon speedrun learning agent monitoring.

Provides configurable alerting based on system metrics, performance thresholds,
and SLA violations with multiple notification channels and escalation policies.

Performance requirements:
- Alert evaluation: <10ms per check cycle
- Real-time threshold monitoring with minimal latency
- Configurable alert suppression and escalation
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert lifecycle status."""
    
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


@dataclass
class AlertRule:
    """Configuration for a monitoring alert rule."""
    
    rule_id: str
    name: str
    description: str
    metric_name: str
    threshold_value: float
    operator: str  # ">", "<", ">=", "<=", "==", "!="
    severity: AlertSeverity
    evaluation_window_seconds: float = 300.0  # 5 minutes default
    minimum_duration_seconds: float = 60.0    # 1 minute sustained
    suppression_window_seconds: float = 3600.0  # 1 hour cooldown
    enabled: bool = True


@dataclass
class Alert:
    """Active or historical alert instance."""
    
    alert_id: str
    rule: AlertRule
    current_value: float
    threshold_value: float
    status: AlertStatus
    triggered_at: float
    resolved_at: Optional[float] = None
    suppressed_until: Optional[float] = None
    escalation_count: int = 0
    message: str = ""
    
    @property
    def duration_seconds(self) -> float:
        """Calculate alert duration."""
        end_time = self.resolved_at or time.time()
        return end_time - self.triggered_at
    
    @property
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.status == AlertStatus.ACTIVE


class AlertManager:
    """
    Comprehensive alert management system.
    
    Evaluates monitoring rules, manages alert lifecycle,
    and coordinates notification delivery with performance optimization.
    """
    
    def __init__(self, evaluation_interval: float = 30.0):
        """
        Initialize alert manager.

        Args:
            evaluation_interval: Seconds between alert rule evaluations
        """
        self.evaluation_interval = evaluation_interval
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._alert_rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._max_history_size = 1000
        
        # Metrics sources (injected)
        self._metric_sources: Dict[str, Callable[[], Dict[str, Any]]] = {}
        
        # Notification handlers
        self._notification_handlers: List[Callable[[Alert], None]] = []
        
        # Evaluation loop
        self._running = False
        self._evaluation_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self._evaluation_times: List[float] = []
        self._max_timing_samples = 50
        
        logger.info(f"AlertManager initialized (evaluation_interval={evaluation_interval}s)")

    def add_alert_rule(self, rule: AlertRule):
        """
        Add or update an alert rule.
        
        Args:
            rule: AlertRule configuration to add
        """
        with self._lock:
            self._alert_rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")

    def remove_alert_rule(self, rule_id: str):
        """
        Remove an alert rule.
        
        Args:
            rule_id: ID of rule to remove
        """
        with self._lock:
            if rule_id in self._alert_rules:
                del self._alert_rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")
                
                # Resolve any active alerts for this rule
                alerts_to_resolve = [
                    alert for alert in self._active_alerts.values()
                    if alert.rule.rule_id == rule_id
                ]
                for alert in alerts_to_resolve:
                    self._resolve_alert(alert, "Rule removed")

    def add_metric_source(self, source_name: str, source_func: Callable[[], Dict[str, Any]]):
        """
        Add a metrics source function.
        
        Args:
            source_name: Name identifier for the source
            source_func: Function returning metrics dictionary
        """
        with self._lock:
            self._metric_sources[source_name] = source_func
            logger.info(f"Added metric source: {source_name}")

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """
        Add a notification handler for alerts.
        
        Args:
            handler: Function to call when alerts are triggered/resolved
        """
        with self._lock:
            self._notification_handlers.append(handler)
            logger.info(f"Added notification handler: {handler.__name__}")

    def start_monitoring(self):
        """Start the alert evaluation loop."""
        with self._lock:
            if self._running:
                logger.warning("Alert monitoring already running")
                return
                
            self._running = True
            self._evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
            self._evaluation_thread.start()
            logger.info("Alert monitoring started")

    def stop_monitoring(self):
        """Stop the alert evaluation loop."""
        with self._lock:
            self._running = False
            
        if self._evaluation_thread and self._evaluation_thread.is_alive():
            self._evaluation_thread.join(timeout=5.0)
            
        logger.info("Alert monitoring stopped")

    def _evaluation_loop(self):
        """Main alert evaluation loop."""
        while self._running:
            try:
                start_time = time.time()
                self._evaluate_all_rules()
                evaluation_time = time.time() - start_time
                
                # Track performance
                self._record_evaluation_time(evaluation_time)
                
                # Sleep until next evaluation
                time.sleep(self.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                time.sleep(5.0)  # Brief pause on error

    def _evaluate_all_rules(self):
        """Evaluate all enabled alert rules against current metrics."""
        with self._lock:
            enabled_rules = [rule for rule in self._alert_rules.values() if rule.enabled]
            
        # Collect all metrics once for efficiency
        current_metrics = self._collect_all_metrics()
        
        for rule in enabled_rules:
            try:
                self._evaluate_rule(rule, current_metrics)
            except Exception as e:
                logger.error(f"Failed to evaluate rule {rule.rule_id}: {e}")

    def _collect_all_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all registered sources."""
        all_metrics = {}
        
        for source_name, source_func in self._metric_sources.items():
            try:
                metrics = source_func()
                all_metrics[source_name] = metrics
            except Exception as e:
                logger.error(f"Failed to collect metrics from {source_name}: {e}")
                all_metrics[source_name] = {}
                
        return all_metrics

    def _evaluate_rule(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Evaluate a single alert rule against current metrics."""
        # Extract metric value
        metric_value = self._extract_metric_value(rule.metric_name, metrics)
        if metric_value is None:
            logger.debug(f"Metric {rule.metric_name} not found for rule {rule.rule_id}")
            return
            
        # Check threshold condition
        threshold_violated = self._check_threshold(
            metric_value, rule.threshold_value, rule.operator
        )
        
        current_time = time.time()
        alert_key = f"{rule.rule_id}_{rule.metric_name}"
        
        if threshold_violated:
            # Check if alert already exists
            if alert_key in self._active_alerts:
                existing_alert = self._active_alerts[alert_key]
                # Update current value
                existing_alert.current_value = metric_value
                
                # Check for escalation
                if (current_time - existing_alert.triggered_at > 
                    rule.evaluation_window_seconds * 2):  # 2x window for escalation
                    self._escalate_alert(existing_alert)
            else:
                # Create new alert
                self._create_alert(rule, metric_value, current_time, alert_key)
        else:
            # Resolve existing alert if present
            if alert_key in self._active_alerts:
                self._resolve_alert(self._active_alerts[alert_key], "Threshold no longer violated")

    def _extract_metric_value(self, metric_name: str, metrics: Dict[str, Any]) -> Optional[float]:
        """Extract specific metric value from metrics dictionary."""
        # Support dot notation for nested metrics (e.g., "system.cpu_percent")
        keys = metric_name.split('.')
        value = metrics
        
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else:
                    return None
                    
            return float(value) if value is not None else None
        except (KeyError, TypeError, ValueError):
            return None

    def _check_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Check if metric value violates threshold."""
        operators = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: abs(v - t) < 1e-9,  # Float equality
            "!=": lambda v, t: abs(v - t) >= 1e-9,
        }
        
        if operator not in operators:
            logger.error(f"Unknown operator: {operator}")
            return False
            
        return operators[operator](value, threshold)

    def _create_alert(self, rule: AlertRule, value: float, timestamp: float, alert_key: str):
        """Create and register a new alert."""
        # Check suppression window
        if self._is_suppressed(rule, timestamp):
            logger.debug(f"Alert {rule.rule_id} suppressed until cooldown expires")
            return
            
        alert = Alert(
            alert_id=f"{alert_key}_{int(timestamp)}",
            rule=rule,
            current_value=value,
            threshold_value=rule.threshold_value,
            status=AlertStatus.ACTIVE,
            triggered_at=timestamp,
            message=f"{rule.description} - Current: {value}, Threshold: {rule.threshold_value}"
        )
        
        with self._lock:
            self._active_alerts[alert_key] = alert
            self._alert_history.append(alert)
            
            # Trim history if needed
            if len(self._alert_history) > self._max_history_size:
                self._alert_history = self._alert_history[-self._max_history_size:]
        
        logger.warning(f"ALERT TRIGGERED: {rule.name} - {alert.message}")
        self._send_notifications(alert)

    def _resolve_alert(self, alert: Alert, reason: str):
        """Resolve an active alert."""
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = time.time()
        
        # Remove from active alerts
        with self._lock:
            alert_key = f"{alert.rule.rule_id}_{alert.rule.metric_name}"
            self._active_alerts.pop(alert_key, None)
        
        logger.info(f"ALERT RESOLVED: {alert.rule.name} - {reason}")
        self._send_notifications(alert)

    def _escalate_alert(self, alert: Alert):
        """Escalate an alert that has been active too long."""
        alert.escalation_count += 1
        alert.status = AlertStatus.ESCALATED
        
        logger.error(
            f"ALERT ESCALATED: {alert.rule.name} - "
            f"Active for {alert.duration_seconds:.0f}s (escalation #{alert.escalation_count})"
        )
        self._send_notifications(alert)

    def _is_suppressed(self, rule: AlertRule, current_time: float) -> bool:
        """Check if alert rule is currently suppressed."""
        # Look for recent resolved alerts for this rule
        for alert in reversed(self._alert_history[-50:]):  # Check recent history
            if (alert.rule.rule_id == rule.rule_id and 
                alert.resolved_at and
                current_time - alert.resolved_at < rule.suppression_window_seconds):
                return True
        return False

    def _send_notifications(self, alert: Alert):
        """Send notifications for alert via all registered handlers."""
        for handler in self._notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler {handler.__name__} failed: {e}")

    def _record_evaluation_time(self, evaluation_time: float):
        """Record evaluation timing for performance monitoring."""
        with self._lock:
            self._evaluation_times.append(evaluation_time)
            if len(self._evaluation_times) > self._max_timing_samples:
                self._evaluation_times.pop(0)
                
            # Alert if evaluation is slow
            if evaluation_time > 0.010:  # 10ms threshold
                logger.warning(
                    f"Alert evaluation took {evaluation_time:.4f}s (>0.010s target)"
                )

    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts."""
        with self._lock:
            return list(self._active_alerts.values())

    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history, optionally limited."""
        with self._lock:
            history = self._alert_history[:]
            return history[-limit:] if limit else history

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics."""
        with self._lock:
            active_alerts = list(self._active_alerts.values())
            all_history = self._alert_history[:]
            
        # Calculate statistics
        total_alerts = len(all_history)
        active_count = len(active_alerts)
        
        # Severity breakdown
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                alert for alert in all_history if alert.rule.severity == severity
            ])
        
        # Recent alert rate (last hour)
        current_time = time.time()
        recent_alerts = [
            alert for alert in all_history
            if current_time - alert.triggered_at < 3600.0  # 1 hour
        ]
        
        return {
            "active_alerts": active_count,
            "total_historical_alerts": total_alerts,
            "recent_alerts_last_hour": len(recent_alerts),
            "severity_breakdown": severity_counts,
            "enabled_rules": len([r for r in self._alert_rules.values() if r.enabled]),
            "total_rules": len(self._alert_rules),
            "metric_sources": len(self._metric_sources),
            "notification_handlers": len(self._notification_handlers),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get alert manager performance metrics."""
        with self._lock:
            if not self._evaluation_times:
                return {"error": "No evaluation timing available"}
                
            timing_values = self._evaluation_times[:]
            
        return {
            "evaluation_performance": {
                "average_ms": round(sum(timing_values) / len(timing_values) * 1000, 3),
                "max_ms": round(max(timing_values) * 1000, 3),
                "min_ms": round(min(timing_values) * 1000, 3),
                "target_ms": 10.0,
            },
            "monitoring_status": {
                "running": self._running,
                "evaluation_interval_seconds": self.evaluation_interval,
                "last_evaluation": max(timing_values) if timing_values else 0,
            }
        }


# Predefined alert rules for Pokemon speedrun system
def get_default_alert_rules() -> List[AlertRule]:
    """Get default alert rules for Pokemon speedrun monitoring."""
    return [
        # Performance regression alerts
        AlertRule(
            rule_id="compilation_performance",
            name="Script Compilation Performance",
            description="Script compilation time exceeds target",
            metric_name="speedrun.average_script_compilation_ms",
            threshold_value=150.0,  # 150ms threshold (target is 100ms)
            operator=">",
            severity=AlertSeverity.WARNING,
            minimum_duration_seconds=300.0,  # 5 minutes sustained
        ),
        
        AlertRule(
            rule_id="checkpoint_performance",
            name="Checkpoint Loading Performance", 
            description="Checkpoint loading time exceeds target",
            metric_name="speedrun.average_checkpoint_loading_ms",
            threshold_value=750.0,  # 750ms threshold (target is 500ms)
            operator=">",
            severity=AlertSeverity.WARNING,
            minimum_duration_seconds=300.0,
        ),
        
        # System resource alerts
        AlertRule(
            rule_id="cpu_high",
            name="High CPU Usage",
            description="System CPU usage is critically high",
            metric_name="system.cpu_percent",
            threshold_value=85.0,
            operator=">",
            severity=AlertSeverity.CRITICAL,
            minimum_duration_seconds=120.0,  # 2 minutes sustained
        ),
        
        AlertRule(
            rule_id="memory_high",
            name="High Memory Usage",
            description="System memory usage is critically high",
            metric_name="system.memory_percent",
            threshold_value=90.0,
            operator=">",
            severity=AlertSeverity.CRITICAL,
            minimum_duration_seconds=120.0,
        ),
        
        # SLA violation alerts
        AlertRule(
            rule_id="experiment_success_rate",
            name="Low Experiment Success Rate",
            description="Experiment success rate below SLA",
            metric_name="speedrun.experiment_success_rate",
            threshold_value=95.0,
            operator="<",
            severity=AlertSeverity.CRITICAL,
            minimum_duration_seconds=900.0,  # 15 minutes sustained
        ),
        
        # Health monitoring alerts
        AlertRule(
            rule_id="healthy_processes",
            name="Insufficient Healthy Processes",
            description="Number of healthy processes below minimum",
            metric_name="prometheus.healthy_processes",
            threshold_value=3.0,  # At least 3 healthy processes
            operator="<",
            severity=AlertSeverity.CRITICAL,
            minimum_duration_seconds=60.0,  # 1 minute
        ),
    ]