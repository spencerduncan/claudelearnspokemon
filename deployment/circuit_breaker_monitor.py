#!/usr/bin/env python3
"""
Circuit Breaker Monitor and Integration for Message Routing Engine
Provides enhanced circuit breaker monitoring and automatic recovery management

Features:
- Real-time circuit breaker state monitoring
- Automatic fallback management
- Circuit breaker tuning based on deployment phase
- Integration with health monitoring system
- Recovery coordination and validation
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration for different deployment phases."""
    
    failure_threshold: int
    recovery_timeout: float
    success_threshold: int
    monitoring_interval: float = 10.0
    
    @classmethod
    def for_phase(cls, phase: str) -> 'CircuitBreakerConfig':
        """Get circuit breaker config for deployment phase."""
        configs = {
            "shadow": cls(
                failure_threshold=10,
                recovery_timeout=60.0,
                success_threshold=3,
                monitoring_interval=15.0
            ),
            "partial": cls(
                failure_threshold=5,
                recovery_timeout=30.0,
                success_threshold=2,
                monitoring_interval=10.0
            ),
            "full": cls(
                failure_threshold=5,
                recovery_timeout=30.0,
                success_threshold=3,
                monitoring_interval=5.0
            )
        }
        
        return configs.get(phase, configs["full"])


@dataclass
class CircuitBreakerStatus:
    """Current circuit breaker status."""
    
    timestamp: float
    state: str  # CLOSED, OPEN, HALF_OPEN
    failure_count: int
    success_count: int
    last_failure_time: Optional[float]
    last_success_time: Optional[float]
    total_requests: int
    successful_requests: int
    failed_requests: int
    rejected_requests: int
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CircuitBreakerMonitor:
    """Monitor and manage circuit breaker behavior."""
    
    def __init__(self, 
                 phase: str = "full",
                 routing_engine_url: str = "http://localhost:8081",
                 results_dir: str = "/workspace/repo/deployment/results"):
        
        self.phase = phase
        self.routing_engine_url = routing_engine_url
        self.config = CircuitBreakerConfig.for_phase(phase)
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # State tracking
        self.current_status: Optional[CircuitBreakerStatus] = None
        self.status_history: List[CircuitBreakerStatus] = []
        self.monitoring_active = True
        
        # Recovery management
        self.recovery_in_progress = False
        self.recovery_attempts = 0
        self.last_recovery_time = 0.0
        
        logger.info(f"Circuit breaker monitor initialized for phase: {phase}")
        logger.info(f"Config: failures={self.config.failure_threshold}, "
                   f"recovery_timeout={self.config.recovery_timeout}s")
    
    async def start_monitoring(self):
        """Start circuit breaker monitoring loop."""
        logger.info("Starting circuit breaker monitoring...")
        
        while self.monitoring_active:
            try:
                # Get current circuit breaker status
                status = await self.get_circuit_breaker_status()
                
                if status:
                    # Update status tracking
                    self.update_status(status)
                    
                    # Check if recovery management is needed
                    await self.manage_recovery(status)
                    
                    # Log status changes
                    self.log_status_changes(status)
                    
                    # Save status to file
                    await self.save_status(status)
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Circuit breaker monitoring error: {e}")
                await asyncio.sleep(self.config.monitoring_interval)
    
    async def get_circuit_breaker_status(self) -> Optional[CircuitBreakerStatus]:
        """Get current circuit breaker status from routing engine."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                
                # Get circuit breaker health status
                health_url = f"{self.routing_engine_url}/health/routing"
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Extract circuit breaker info from health status
                        routing_health = health_data.get("routing_engine", {})
                        router_health = routing_health.get("router_health", {})
                        
                        if router_health:
                            return self.parse_circuit_breaker_status(router_health)
                
                # Fallback: try admin endpoint for detailed status
                admin_url = f"{self.routing_engine_url}/admin/status"
                async with session.get(admin_url) as response:
                    if response.status == 200:
                        admin_data = await response.json()
                        return self.parse_circuit_breaker_status(admin_data)
        
        except Exception as e:
            logger.debug(f"Failed to get circuit breaker status: {e}")
        
        return None
    
    def parse_circuit_breaker_status(self, data: Dict[str, Any]) -> CircuitBreakerStatus:
        """Parse circuit breaker status from API response."""
        
        # Extract metrics (structure may vary based on implementation)
        metrics = data.get("metrics", {})
        
        return CircuitBreakerStatus(
            timestamp=time.time(),
            state=data.get("routing_mode", "UNKNOWN").upper(),
            failure_count=data.get("failure_count", 0),
            success_count=metrics.get("successful_routes", 0),
            last_failure_time=None,  # Would need to be added to router health
            last_success_time=None,  # Would need to be added to router health
            total_requests=metrics.get("total_routing_requests", 0),
            successful_requests=metrics.get("successful_routes", 0),
            failed_requests=metrics.get("failed_routes", 0),
            rejected_requests=metrics.get("rejected_requests", 0)
        )
    
    def update_status(self, status: CircuitBreakerStatus):
        """Update status tracking with new status."""
        if self.current_status:
            # Store previous status in history
            self.status_history.append(self.current_status)
            
            # Keep only recent history (last hour)
            cutoff_time = time.time() - 3600
            self.status_history = [
                s for s in self.status_history 
                if s.timestamp > cutoff_time
            ]
        
        self.current_status = status
    
    async def manage_recovery(self, status: CircuitBreakerStatus):
        """Manage circuit breaker recovery process."""
        
        # Check if circuit breaker is open
        if status.state == "OPEN" and not self.recovery_in_progress:
            await self.initiate_recovery(status)
        
        # Monitor recovery progress
        elif self.recovery_in_progress:
            await self.monitor_recovery_progress(status)
    
    async def initiate_recovery(self, status: CircuitBreakerStatus):
        """Initiate circuit breaker recovery process."""
        
        self.recovery_in_progress = True
        self.recovery_attempts += 1
        self.last_recovery_time = time.time()
        
        logger.warning(f"Circuit breaker OPEN detected - initiating recovery (attempt {self.recovery_attempts})")
        
        # Log recovery initiation
        recovery_info = {
            "timestamp": datetime.now().isoformat(),
            "phase": self.phase,
            "recovery_attempt": self.recovery_attempts,
            "failure_count": status.failure_count,
            "success_rate": status.success_rate,
            "trigger": "circuit_breaker_open"
        }
        
        recovery_file = self.results_dir / "circuit_breaker_recoveries.jsonl"
        with open(recovery_file, 'a') as f:
            f.write(json.dumps(recovery_info) + '\n')
        
        # Wait for recovery timeout before checking
        await asyncio.sleep(min(self.config.recovery_timeout, 60))
    
    async def monitor_recovery_progress(self, status: CircuitBreakerStatus):
        """Monitor progress of circuit breaker recovery."""
        
        recovery_duration = time.time() - self.last_recovery_time
        
        if status.state == "CLOSED":
            # Recovery successful
            logger.info(f"Circuit breaker recovery successful after {recovery_duration:.1f}s")
            self.recovery_in_progress = False
            
        elif status.state == "HALF_OPEN":
            # Recovery in progress
            logger.info(f"Circuit breaker in HALF_OPEN state - testing recovery ({recovery_duration:.1f}s)")
            
        elif recovery_duration > self.config.recovery_timeout * 3:
            # Recovery taking too long - escalate
            logger.error(f"Circuit breaker recovery failed after {recovery_duration:.1f}s - escalating")
            await self.escalate_recovery_failure(status)
    
    async def escalate_recovery_failure(self, status: CircuitBreakerStatus):
        """Escalate when circuit breaker recovery fails."""
        
        self.recovery_in_progress = False
        
        # Create escalation record
        escalation = {
            "timestamp": datetime.now().isoformat(),
            "phase": self.phase,
            "recovery_attempts": self.recovery_attempts,
            "failure_details": status.to_dict(),
            "action": "escalation_to_rollback"
        }
        
        escalation_file = self.results_dir / "circuit_breaker_escalations.jsonl"
        with open(escalation_file, 'a') as f:
            f.write(json.dumps(escalation) + '\n')
        
        logger.critical("Circuit breaker recovery failed - creating rollback trigger")
        
        # Create rollback trigger file for health monitor
        rollback_trigger = {
            "timestamp": datetime.now().isoformat(),
            "source": "circuit_breaker_monitor",
            "reason": "circuit_breaker_recovery_failed",
            "phase": self.phase,
            "recovery_attempts": self.recovery_attempts
        }
        
        trigger_file = self.results_dir / "rollback_trigger.json"
        with open(trigger_file, 'w') as f:
            json.dump(rollback_trigger, f, indent=2)
    
    def log_status_changes(self, status: CircuitBreakerStatus):
        """Log circuit breaker status changes."""
        
        if not self.current_status:
            logger.info(f"Circuit breaker initial state: {status.state}")
            return
        
        # Check for state changes
        if self.current_status.state != status.state:
            logger.warning(f"Circuit breaker state changed: {self.current_status.state} → {status.state}")
            
            # Log additional context for state changes
            if status.state == "OPEN":
                logger.error(f"Circuit breaker OPENED - "
                           f"failure_count={status.failure_count}, "
                           f"success_rate={status.success_rate:.1f}%")
            
            elif status.state == "CLOSED":
                logger.info(f"Circuit breaker CLOSED - "
                          f"success_count={status.success_count}, "
                          f"success_rate={status.success_rate:.1f}%")
    
    async def save_status(self, status: CircuitBreakerStatus):
        """Save circuit breaker status to file."""
        
        timestamp = datetime.fromtimestamp(status.timestamp)
        status_file = self.results_dir / f"circuit_breaker_status_{timestamp.strftime('%Y%m%d')}.jsonl"
        
        with open(status_file, 'a') as f:
            f.write(json.dumps(status.to_dict()) + '\n')
    
    async def update_circuit_breaker_config(self, new_config: Dict[str, Any]):
        """Update circuit breaker configuration dynamically."""
        
        try:
            async with aiohttp.ClientSession() as session:
                config_url = f"{self.routing_engine_url}/admin/config"
                
                async with session.post(config_url, json=new_config) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Circuit breaker config updated: {result}")
                        return True
                    else:
                        logger.error(f"Failed to update circuit breaker config: {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Error updating circuit breaker config: {e}")
            return False
    
    async def force_circuit_breaker_reset(self):
        """Force reset circuit breaker (emergency use only)."""
        
        logger.warning("FORCING circuit breaker reset - emergency override")
        
        try:
            # This would require an admin endpoint to reset the circuit breaker
            async with aiohttp.ClientSession() as session:
                reset_url = f"{self.routing_engine_url}/admin/circuit-breaker/reset"
                
                async with session.post(reset_url) as response:
                    if response.status == 200:
                        logger.info("Circuit breaker force reset successful")
                        self.recovery_attempts = 0
                        self.recovery_in_progress = False
                        return True
                    else:
                        logger.error(f"Failed to force reset circuit breaker: {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Error forcing circuit breaker reset: {e}")
            return False
    
    def get_recovery_summary(self) -> Dict[str, Any]:
        """Get summary of recovery activities."""
        
        return {
            "phase": self.phase,
            "recovery_attempts": self.recovery_attempts,
            "recovery_in_progress": self.recovery_in_progress,
            "last_recovery_time": self.last_recovery_time,
            "current_state": self.current_status.state if self.current_status else "UNKNOWN",
            "success_rate": self.current_status.success_rate if self.current_status else 0.0,
            "monitoring_active": self.monitoring_active
        }


async def main():
    """Main entry point for circuit breaker monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Circuit Breaker Monitor for Message Routing Engine")
    parser.add_argument("--phase", choices=["shadow", "partial", "full"], default="full",
                       help="Deployment phase")
    parser.add_argument("--routing-url", default="http://localhost:8081",
                       help="Routing engine URL")
    parser.add_argument("--results-dir", default="/workspace/repo/deployment/results",
                       help="Results directory")
    
    args = parser.parse_args()
    
    # Initialize circuit breaker monitor
    monitor = CircuitBreakerMonitor(
        phase=args.phase,
        routing_engine_url=args.routing_url,
        results_dir=args.results_dir
    )
    
    # Start monitoring
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Circuit breaker monitoring stopped by user")
    except Exception as e:
        logger.error(f"Circuit breaker monitoring failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))