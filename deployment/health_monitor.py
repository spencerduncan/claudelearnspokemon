#!/usr/bin/env python3
"""
Production Health Monitor for Message Routing Engine Deployment
Continuously monitors SLA compliance and triggers automatic rollbacks

Monitors:
- P95 latency thresholds
- Success rate compliance  
- Circuit breaker status
- Memory usage and resource health
- Queue depth and backlog

Triggers rollback on:
- SLA violations exceeding threshold duration
- Circuit breaker activation
- Critical system failures
- Resource exhaustion
"""

import asyncio
import json
import logging
import os
import time
import subprocess
import signal
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import aiohttp
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class HealthThresholds:
    """SLA thresholds for each deployment phase."""
    
    # Latency thresholds (ms)
    p95_latency_ms: float
    p99_latency_ms: float
    avg_latency_ms: float
    
    # Success rate thresholds (%)
    min_success_rate: float
    
    # Resource thresholds
    max_memory_mb: float
    max_cpu_percent: float
    max_queue_depth: int
    
    # Timing thresholds
    max_violation_duration_seconds: int = 300  # 5 minutes
    check_interval_seconds: int = 15
    
    @classmethod
    def for_phase(cls, phase: str) -> 'HealthThresholds':
        """Get thresholds for deployment phase."""
        thresholds = {
            "shadow": cls(
                p95_latency_ms=45.0,
                p99_latency_ms=75.0,
                avg_latency_ms=25.0,
                min_success_rate=98.5,
                max_memory_mb=128.0,
                max_cpu_percent=70.0,
                max_queue_depth=50,
                max_violation_duration_seconds=300
            ),
            "partial": cls(
                p95_latency_ms=48.0,
                p99_latency_ms=80.0,
                avg_latency_ms=30.0,
                min_success_rate=99.0,
                max_memory_mb=256.0,
                max_cpu_percent=80.0,
                max_queue_depth=100,
                max_violation_duration_seconds=180  # Stricter for partial
            ),
            "full": cls(
                p95_latency_ms=50.0,
                p99_latency_ms=85.0,
                avg_latency_ms=35.0,
                min_success_rate=99.13,
                max_memory_mb=512.0,
                max_cpu_percent=85.0,
                max_queue_depth=200,
                max_violation_duration_seconds=120  # Strictest for production
            )
        }
        
        return thresholds.get(phase, thresholds["full"])


@dataclass
class HealthMetrics:
    """Current system health metrics."""
    
    timestamp: float
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    success_rate: float = 100.0
    error_rate: float = 0.0
    requests_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    queue_depth: int = 0
    circuit_breaker_open: bool = False
    active_connections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthViolation:
    """Record of an SLA violation."""
    
    timestamp: float
    metric_name: str
    actual_value: float
    threshold_value: float
    severity: str  # warning, critical
    message: str
    
    def __post_init__(self):
        self.age_seconds = time.time() - self.timestamp


class HealthMonitor:
    """Production health monitor with automatic rollback capabilities."""
    
    def __init__(self, 
                 phase: str = "full",
                 prometheus_url: str = "http://localhost:9090",
                 deployment_script: str = "/workspace/repo/deployment/deploy.sh"):
        self.phase = phase
        self.prometheus_url = prometheus_url
        self.deployment_script = deployment_script
        self.thresholds = HealthThresholds.for_phase(phase)
        
        # State tracking
        self.violations: List[HealthViolation] = []
        self.is_healthy = True
        self.rollback_triggered = False
        self.monitoring_active = True
        
        # Results directory
        self.results_dir = Path("/workspace/repo/deployment/results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"Health monitor initialized for phase: {phase}")
        logger.info(f"Thresholds: P95<{self.thresholds.p95_latency_ms}ms, "
                   f"Success>{self.thresholds.min_success_rate}%")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.monitoring_active = False
    
    async def start_monitoring(self):
        """Main monitoring loop."""
        logger.info("Starting health monitoring...")
        
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = await self.collect_metrics()
                
                # Evaluate health status
                violations = self.evaluate_health(metrics)
                
                # Update violation tracking
                self.update_violations(violations)
                
                # Check if rollback should be triggered
                should_rollback = self.should_trigger_rollback()
                
                if should_rollback and not self.rollback_triggered:
                    await self.trigger_rollback(violations)
                    break
                
                # Log status
                await self.log_health_status(metrics, violations)
                
                # Save metrics to file
                await self.save_metrics(metrics)
                
                # Wait for next check
                await asyncio.sleep(self.thresholds.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.thresholds.check_interval_seconds)
        
        logger.info("Health monitoring stopped")
    
    async def collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics from Prometheus."""
        metrics = HealthMetrics(timestamp=time.time())
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                
                # P95 latency
                p95_query = 'histogram_quantile(0.95, routing_duration_seconds_bucket{system="new_routing_engine"})'
                metrics.p95_latency_ms = await self._query_prometheus(session, p95_query) * 1000
                
                # P99 latency  
                p99_query = 'histogram_quantile(0.99, routing_duration_seconds_bucket{system="new_routing_engine"})'
                metrics.p99_latency_ms = await self._query_prometheus(session, p99_query) * 1000
                
                # Average latency
                avg_query = 'avg(routing_duration_seconds{system="new_routing_engine"})'
                metrics.avg_latency_ms = await self._query_prometheus(session, avg_query) * 1000
                
                # Success rate
                success_query = 'rate(routing_requests_total{status="success",system="new_routing_engine"}[5m]) / rate(routing_requests_total{system="new_routing_engine"}[5m]) * 100'
                metrics.success_rate = await self._query_prometheus(session, success_query)
                
                # Error rate
                error_query = 'rate(routing_requests_total{status!="success",system="new_routing_engine"}[5m]) / rate(routing_requests_total{system="new_routing_engine"}[5m]) * 100'
                metrics.error_rate = await self._query_prometheus(session, error_query)
                
                # Requests per second
                rps_query = 'rate(routing_requests_total{system="new_routing_engine"}[5m])'
                metrics.requests_per_second = await self._query_prometheus(session, rps_query)
                
                # Circuit breaker status
                cb_query = 'circuit_breaker_open{system="new_routing_engine"}'
                metrics.circuit_breaker_open = await self._query_prometheus(session, cb_query) > 0
                
                # Memory usage (if available)
                memory_query = 'container_memory_usage_bytes{name=~".*routing-engine.*"} / 1024 / 1024'
                metrics.memory_usage_mb = await self._query_prometheus(session, memory_query)
                
                # CPU usage (if available)
                cpu_query = 'rate(container_cpu_usage_seconds_total{name=~".*routing-engine.*"}[5m]) * 100'
                metrics.cpu_usage_percent = await self._query_prometheus(session, cpu_query)
                
                # Queue depth (if available)
                queue_query = 'queue_depth_total{system="new_routing_engine"}'
                metrics.queue_depth = int(await self._query_prometheus(session, queue_query))
                
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            # Return metrics with current timestamp but default values
        
        return metrics
    
    async def _query_prometheus(self, session: aiohttp.ClientSession, query: str) -> float:
        """Query Prometheus and return single value."""
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            params = {"query": query}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("data", {}).get("result", [])
                    if results:
                        return float(results[0]["value"][1])
                return 0.0
        except Exception as e:
            logger.debug(f"Prometheus query failed: {query} - {e}")
            return 0.0
    
    def evaluate_health(self, metrics: HealthMetrics) -> List[HealthViolation]:
        """Evaluate current metrics against thresholds."""
        violations = []
        
        # P95 latency check
        if metrics.p95_latency_ms > self.thresholds.p95_latency_ms:
            violations.append(HealthViolation(
                timestamp=metrics.timestamp,
                metric_name="p95_latency_ms",
                actual_value=metrics.p95_latency_ms,
                threshold_value=self.thresholds.p95_latency_ms,
                severity="critical" if metrics.p95_latency_ms > self.thresholds.p95_latency_ms * 1.5 else "warning",
                message=f"P95 latency {metrics.p95_latency_ms:.1f}ms exceeds threshold {self.thresholds.p95_latency_ms}ms"
            ))
        
        # Success rate check
        if metrics.success_rate < self.thresholds.min_success_rate:
            violations.append(HealthViolation(
                timestamp=metrics.timestamp,
                metric_name="success_rate",
                actual_value=metrics.success_rate,
                threshold_value=self.thresholds.min_success_rate,
                severity="critical",
                message=f"Success rate {metrics.success_rate:.2f}% below threshold {self.thresholds.min_success_rate}%"
            ))
        
        # Circuit breaker check
        if metrics.circuit_breaker_open:
            violations.append(HealthViolation(
                timestamp=metrics.timestamp,
                metric_name="circuit_breaker_open",
                actual_value=1.0,
                threshold_value=0.0,
                severity="critical",
                message="Circuit breaker is open - routing failures detected"
            ))
        
        # Memory usage check
        if metrics.memory_usage_mb > self.thresholds.max_memory_mb:
            violations.append(HealthViolation(
                timestamp=metrics.timestamp,
                metric_name="memory_usage_mb",
                actual_value=metrics.memory_usage_mb,
                threshold_value=self.thresholds.max_memory_mb,
                severity="warning" if metrics.memory_usage_mb < self.thresholds.max_memory_mb * 1.2 else "critical",
                message=f"Memory usage {metrics.memory_usage_mb:.1f}MB exceeds threshold {self.thresholds.max_memory_mb}MB"
            ))
        
        # Queue depth check
        if metrics.queue_depth > self.thresholds.max_queue_depth:
            violations.append(HealthViolation(
                timestamp=metrics.timestamp,
                metric_name="queue_depth",
                actual_value=float(metrics.queue_depth),
                threshold_value=float(self.thresholds.max_queue_depth),
                severity="warning" if metrics.queue_depth < self.thresholds.max_queue_depth * 1.5 else "critical",
                message=f"Queue depth {metrics.queue_depth} exceeds threshold {self.thresholds.max_queue_depth}"
            ))
        
        return violations
    
    def update_violations(self, new_violations: List[HealthViolation]):
        """Update violation tracking with new violations."""
        # Add new violations
        self.violations.extend(new_violations)
        
        # Remove violations older than violation duration threshold
        current_time = time.time()
        self.violations = [
            v for v in self.violations 
            if current_time - v.timestamp < self.thresholds.max_violation_duration_seconds * 2
        ]
        
        # Update health status
        current_critical_violations = [
            v for v in self.violations
            if v.severity == "critical" and current_time - v.timestamp < self.thresholds.max_violation_duration_seconds
        ]
        
        self.is_healthy = len(current_critical_violations) == 0
    
    def should_trigger_rollback(self) -> bool:
        """Determine if rollback should be triggered."""
        if self.rollback_triggered:
            return False
        
        current_time = time.time()
        
        # Check for sustained critical violations
        recent_critical_violations = [
            v for v in self.violations
            if v.severity == "critical" and current_time - v.timestamp < self.thresholds.max_violation_duration_seconds
        ]
        
        if len(recent_critical_violations) > 0:
            # Group by metric name and check duration
            metric_violations = {}
            for violation in recent_critical_violations:
                if violation.metric_name not in metric_violations:
                    metric_violations[violation.metric_name] = []
                metric_violations[violation.metric_name].append(violation)
            
            # Check if any metric has sustained violations
            for metric_name, violations in metric_violations.items():
                if len(violations) >= 3:  # At least 3 violations
                    oldest_violation = min(violations, key=lambda v: v.timestamp)
                    duration = current_time - oldest_violation.timestamp
                    
                    if duration >= self.thresholds.max_violation_duration_seconds:
                        logger.error(f"Sustained violations detected for {metric_name} over {duration}s")
                        return True
        
        # Check for immediate rollback conditions
        immediate_rollback_conditions = [
            "circuit_breaker_open",
            "success_rate"
        ]
        
        for condition in immediate_rollback_conditions:
            recent_condition_violations = [
                v for v in recent_critical_violations
                if v.metric_name == condition
            ]
            if recent_condition_violations:
                logger.error(f"Immediate rollback condition triggered: {condition}")
                return True
        
        return False
    
    async def trigger_rollback(self, violations: List[HealthViolation]):
        """Trigger automatic rollback to previous stable version."""
        self.rollback_triggered = True
        
        logger.error("=== AUTOMATIC ROLLBACK TRIGGERED ===")
        logger.error("Critical SLA violations detected:")
        
        for violation in violations:
            if violation.severity == "critical":
                logger.error(f"  - {violation.message}")
        
        # Save rollback decision
        rollback_info = {
            "timestamp": datetime.now().isoformat(),
            "phase": self.phase,
            "trigger_reason": "SLA_violations",
            "violations": [asdict(v) for v in violations if v.severity == "critical"],
            "rollback_initiated": True
        }
        
        rollback_file = self.results_dir / "rollback_decision.json"
        with open(rollback_file, 'w') as f:
            json.dump(rollback_info, f, indent=2)
        
        # Execute rollback
        try:
            logger.info("Executing deployment rollback...")
            process = await asyncio.create_subprocess_exec(
                self.deployment_script, "rollback",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("Rollback completed successfully")
                logger.info(stdout.decode())
            else:
                logger.error(f"Rollback failed with code {process.returncode}")
                logger.error(stderr.decode())
                
        except Exception as e:
            logger.error(f"Failed to execute rollback: {e}")
        
        # Stop monitoring after rollback
        self.monitoring_active = False
    
    async def log_health_status(self, metrics: HealthMetrics, violations: List[HealthViolation]):
        """Log current health status."""
        status = "HEALTHY" if self.is_healthy else "UNHEALTHY"
        
        if len(violations) == 0:
            logger.info(f"Health check: {status} - "
                       f"P95: {metrics.p95_latency_ms:.1f}ms, "
                       f"Success: {metrics.success_rate:.2f}%, "
                       f"RPS: {metrics.requests_per_second:.1f}")
        else:
            logger.warning(f"Health check: {status} - {len(violations)} violations:")
            for violation in violations:
                logger.warning(f"  - {violation.message}")
    
    async def save_metrics(self, metrics: HealthMetrics):
        """Save metrics to file for analysis."""
        timestamp = datetime.fromtimestamp(metrics.timestamp)
        metrics_file = self.results_dir / f"health_metrics_{timestamp.strftime('%Y%m%d')}.jsonl"
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')


async def main():
    """Main entry point for health monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health Monitor for Message Routing Engine")
    parser.add_argument("--phase", choices=["shadow", "partial", "full"], default="full",
                       help="Deployment phase to monitor")
    parser.add_argument("--prometheus-url", default="http://localhost:9090",
                       help="Prometheus server URL")
    parser.add_argument("--deployment-script", default="/workspace/repo/deployment/deploy.sh",
                       help="Path to deployment script")
    
    args = parser.parse_args()
    
    # Initialize health monitor
    monitor = HealthMonitor(
        phase=args.phase,
        prometheus_url=args.prometheus_url,
        deployment_script=args.deployment_script
    )
    
    # Start monitoring
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Health monitoring stopped by user")
    except Exception as e:
        logger.error(f"Health monitoring failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))