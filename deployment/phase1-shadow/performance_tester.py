#!/usr/bin/env python3
"""
Performance testing script for Message Routing Engine shadow deployment.
Validates routing performance, latency, and correctness during shadow testing phase.
"""

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
import aiohttp
import click

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Configuration for performance testing."""
    target_host: str = "traffic-splitter"
    target_port: int = 80
    test_duration_hours: float = 48.0
    rps_target: int = 50
    shadow_validation: bool = True
    results_dir: str = "/app/results"
    
    # SLA requirements from rollout config
    max_routing_time_ms: float = 45.0
    min_success_rate: float = 98.5
    
    # Test scenarios
    strategic_percentage: float = 20.0  # 20% strategic, 80% tactical
    priority_distribution: Dict[str, float] = None
    
    def __post_init__(self):
        if self.priority_distribution is None:
            self.priority_distribution = {
                "HIGH": 0.1,
                "NORMAL": 0.8,
                "LOW": 0.1
            }


@dataclass
class RequestResult:
    """Result of a single routing request."""
    timestamp: float
    request_type: str  # strategic, tactical, auto
    priority: str
    success: bool
    response_time_ms: float
    worker_id: str = None
    worker_type: str = None
    error_message: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceTester:
    """Performance testing harness for shadow deployment validation."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Test state
        self.start_time = None
        self.end_time = None
        self.total_requests = 0
        self.results: List[RequestResult] = []
        
        # Create health check file
        (self.results_dir / "health").touch()
        
        logger.info(f"Performance tester initialized: {config.rps_target} RPS for {config.test_duration_hours}h")
    
    async def run_performance_test(self):
        """Main performance testing loop."""
        self.start_time = time.time()
        self.end_time = self.start_time + (self.config.test_duration_hours * 3600)
        
        logger.info("Starting performance test...")
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                
                # Pre-test health checks
                await self.validate_endpoints(session)
                
                # Start continuous testing
                await self.execute_load_test(session)
                
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            raise
        finally:
            await self.generate_final_report()
    
    async def validate_endpoints(self, session: aiohttp.ClientSession):
        """Validate all endpoints are responding before starting load test."""
        base_url = f"http://{self.config.target_host}:{self.config.target_port}"
        
        endpoints = [
            "/health",
            "/health/routing",
            "/health/classifier",
            "/health/queues"
        ]
        
        logger.info("Validating endpoints...")
        
        for endpoint in endpoints:
            try:
                async with session.get(f"{base_url}{endpoint}") as response:
                    if response.status == 200:
                        logger.info(f"✓ {endpoint} is healthy")
                    else:
                        logger.warning(f"⚠ {endpoint} returned {response.status}")
            except Exception as e:
                logger.error(f"✗ {endpoint} failed: {e}")
                
        logger.info("Endpoint validation complete")
    
    async def execute_load_test(self, session: aiohttp.ClientSession):
        """Execute the main load testing loop."""
        request_interval = 1.0 / self.config.rps_target
        
        while time.time() < self.end_time:
            loop_start = time.time()
            
            # Generate batch of requests for this interval
            batch_size = max(1, int(self.config.rps_target / 10))  # 10 batches per second
            
            tasks = []
            for _ in range(batch_size):
                task = asyncio.create_task(self.make_routing_request(session))
                tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, RequestResult):
                    self.results.append(result)
                    self.total_requests += 1
                elif isinstance(result, Exception):
                    logger.error(f"Request failed: {result}")
            
            # Periodic reporting
            if self.total_requests % 1000 == 0:
                await self.log_interim_metrics()
            
            # Rate limiting - maintain target RPS
            loop_duration = time.time() - loop_start
            sleep_time = max(0, request_interval * batch_size - loop_duration)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        logger.info("Load testing phase complete")
    
    async def make_routing_request(self, session: aiohttp.ClientSession) -> RequestResult:
        """Make a single routing request and measure performance."""
        start_time = time.time()
        
        # Generate test request
        request_type, endpoint, payload = self.generate_test_request()
        
        try:
            url = f"http://{self.config.target_host}:{self.config.target_port}{endpoint}"
            
            async with session.post(url, json=payload) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    return RequestResult(
                        timestamp=start_time,
                        request_type=request_type,
                        priority=payload["priority"],
                        success=data.get("success", False),
                        response_time_ms=response_time,
                        worker_id=data.get("worker_id"),
                        worker_type=data.get("worker_type")
                    )
                else:
                    error_text = await response.text()
                    return RequestResult(
                        timestamp=start_time,
                        request_type=request_type,
                        priority=payload["priority"],
                        success=False,
                        response_time_ms=response_time,
                        error_message=f"HTTP {response.status}: {error_text}"
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return RequestResult(
                timestamp=start_time,
                request_type=request_type,
                priority=payload["priority"],
                success=False,
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    def generate_test_request(self) -> tuple[str, str, Dict[str, Any]]:
        """Generate a realistic test request."""
        # Determine request type
        if random.random() < self.config.strategic_percentage / 100:
            request_type = "strategic"
            endpoint = "/route/strategic"
            content = self.generate_strategic_content()
        else:
            request_type = "tactical"
            endpoint = "/route/tactical"  
            content = self.generate_tactical_content()
        
        # Select priority based on distribution
        priority = self.select_priority()
        
        # Generate context
        context = self.generate_request_context()
        
        payload = {
            "content": content,
            "context": context,
            "priority": priority,
            "force_strategic": request_type == "strategic",
            "force_tactical": request_type == "tactical"
        }
        
        return request_type, endpoint, payload
    
    def generate_strategic_content(self) -> str:
        """Generate realistic strategic planning content."""
        strategic_tasks = [
            "Analyze optimal route through Victory Road",
            "Plan team composition for Elite Four battles",
            "Determine optimal grinding strategy for level requirements",
            "Evaluate trade-offs between different speedrun categories",
            "Create backup strategies for RNG-dependent sections",
            "Plan resource management for long speedrun segments"
        ]
        return random.choice(strategic_tasks)
    
    def generate_tactical_content(self) -> str:
        """Generate realistic tactical development content."""
        tactical_tasks = [
            "Generate movement script for Cerulean Cave",
            "Optimize battle script for gym leader fights",
            "Create item management automation",
            "Develop encounter manipulation code",
            "Generate menu navigation sequences",
            "Create save state management scripts"
        ]
        return random.choice(tactical_tasks)
    
    def select_priority(self) -> str:
        """Select priority based on configured distribution."""
        rand = random.random()
        cumulative = 0
        
        for priority, probability in self.config.priority_distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return priority
        
        return "NORMAL"  # Fallback
    
    def generate_request_context(self) -> Dict[str, Any]:
        """Generate realistic request context."""
        contexts = [
            {"game_state": "start", "route": "pewter_city"},
            {"game_state": "mid_game", "route": "cycling_road"},
            {"game_state": "end_game", "route": "victory_road"},
            {"session_id": f"session_{random.randint(1000, 9999)}"},
            {"retry_count": random.randint(0, 3)},
        ]
        return random.choice(contexts)
    
    async def log_interim_metrics(self):
        """Log interim performance metrics."""
        if not self.results:
            return
            
        recent_results = self.results[-1000:]  # Last 1000 requests
        
        success_count = sum(1 for r in recent_results if r.success)
        success_rate = (success_count / len(recent_results)) * 100
        
        response_times = [r.response_time_ms for r in recent_results if r.success]
        if response_times:
            p95_latency = sorted(response_times)[int(0.95 * len(response_times))]
            avg_latency = sum(response_times) / len(response_times)
        else:
            p95_latency = avg_latency = 0
        
        elapsed_minutes = (time.time() - self.start_time) / 60
        current_rps = len(recent_results) / min(60, elapsed_minutes * 60)  # Last minute
        
        logger.info(
            f"Metrics - Total: {self.total_requests}, "
            f"Success Rate: {success_rate:.1f}%, "
            f"P95 Latency: {p95_latency:.1f}ms, "
            f"Avg Latency: {avg_latency:.1f}ms, "
            f"Current RPS: {current_rps:.1f}"
        )
        
        # Check SLA compliance
        if success_rate < self.config.min_success_rate:
            logger.warning(f"⚠ Success rate {success_rate:.1f}% below SLA {self.config.min_success_rate}%")
        
        if p95_latency > self.config.max_routing_time_ms:
            logger.warning(f"⚠ P95 latency {p95_latency:.1f}ms exceeds SLA {self.config.max_routing_time_ms}ms")
    
    async def generate_final_report(self):
        """Generate comprehensive final performance report."""
        logger.info("Generating final performance report...")
        
        if not self.results:
            logger.error("No results to analyze")
            return
        
        # Calculate overall metrics
        total_requests = len(self.results)
        successful_requests = [r for r in self.results if r.success]
        failed_requests = [r for r in self.results if not r.success]
        
        success_rate = (len(successful_requests) / total_requests) * 100
        
        # Response time analysis
        if successful_requests:
            response_times = [r.response_time_ms for r in successful_requests]
            response_times.sort()
            
            metrics = {
                "p50": response_times[int(0.50 * len(response_times))],
                "p90": response_times[int(0.90 * len(response_times))],
                "p95": response_times[int(0.95 * len(response_times))],
                "p99": response_times[int(0.99 * len(response_times))],
                "avg": sum(response_times) / len(response_times),
                "min": min(response_times),
                "max": max(response_times)
            }
        else:
            metrics = {k: 0 for k in ["p50", "p90", "p95", "p99", "avg", "min", "max"]}
        
        # Request type analysis
        strategic_requests = [r for r in self.results if r.request_type == "strategic"]
        tactical_requests = [r for r in self.results if r.request_type == "tactical"]
        
        # SLA compliance analysis
        sla_compliant_requests = [
            r for r in successful_requests 
            if r.response_time_ms <= self.config.max_routing_time_ms
        ]
        sla_compliance_rate = (len(sla_compliant_requests) / len(successful_requests)) * 100 if successful_requests else 0
        
        # Test duration
        actual_duration = time.time() - self.start_time
        target_requests = int(self.config.rps_target * actual_duration)
        throughput_achievement = (total_requests / target_requests) * 100 if target_requests > 0 else 0
        
        # Final report
        report = {
            "test_summary": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(time.time()).isoformat(),
                "duration_hours": actual_duration / 3600,
                "target_rps": self.config.rps_target,
                "actual_rps": total_requests / actual_duration
            },
            "request_metrics": {
                "total_requests": total_requests,
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate_percent": success_rate,
                "throughput_achievement_percent": throughput_achievement
            },
            "latency_metrics": {
                "response_times_ms": metrics,
                "sla_compliance_rate_percent": sla_compliance_rate
            },
            "request_type_breakdown": {
                "strategic": {
                    "count": len(strategic_requests),
                    "success_rate": (len([r for r in strategic_requests if r.success]) / len(strategic_requests)) * 100 if strategic_requests else 0
                },
                "tactical": {
                    "count": len(tactical_requests),
                    "success_rate": (len([r for r in tactical_requests if r.success]) / len(tactical_requests)) * 100 if tactical_requests else 0
                }
            },
            "sla_assessment": {
                "success_rate_sla": {
                    "target": self.config.min_success_rate,
                    "actual": success_rate,
                    "passed": success_rate >= self.config.min_success_rate
                },
                "latency_sla": {
                    "target_p95_ms": self.config.max_routing_time_ms,
                    "actual_p95_ms": metrics["p95"],
                    "passed": metrics["p95"] <= self.config.max_routing_time_ms
                }
            },
            "error_analysis": {
                "error_messages": list(set(r.error_message for r in failed_requests if r.error_message)),
                "error_counts": {}
            }
        }
        
        # Error count analysis
        for result in failed_requests:
            if result.error_message:
                report["error_analysis"]["error_counts"][result.error_message] = \
                    report["error_analysis"]["error_counts"].get(result.error_message, 0) + 1
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full report
        report_file = self.results_dir / f"performance_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save raw results (sample)
        results_sample = self.results[::max(1, len(self.results) // 10000)]  # Max 10k samples
        results_file = self.results_dir / f"raw_results_sample_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in results_sample], f, indent=2)
        
        # Log summary
        logger.info("═" * 80)
        logger.info("PERFORMANCE TEST SUMMARY")
        logger.info("═" * 80)
        logger.info(f"Duration: {actual_duration/3600:.1f}h")
        logger.info(f"Total Requests: {total_requests:,}")
        logger.info(f"Success Rate: {success_rate:.2f}%")
        logger.info(f"P95 Latency: {metrics['p95']:.1f}ms")
        logger.info(f"Average Latency: {metrics['avg']:.1f}ms")
        logger.info(f"Actual RPS: {total_requests/actual_duration:.1f}")
        
        # SLA Assessment
        success_sla_passed = success_rate >= self.config.min_success_rate
        latency_sla_passed = metrics['p95'] <= self.config.max_routing_time_ms
        
        logger.info("─" * 40)
        logger.info("SLA ASSESSMENT")
        logger.info("─" * 40)
        logger.info(f"Success Rate SLA: {'✓ PASS' if success_sla_passed else '✗ FAIL'} "
                   f"({success_rate:.2f}% vs {self.config.min_success_rate}%)")
        logger.info(f"Latency SLA: {'✓ PASS' if latency_sla_passed else '✗ FAIL'} "
                   f"({metrics['p95']:.1f}ms vs {self.config.max_routing_time_ms}ms)")
        
        overall_pass = success_sla_passed and latency_sla_passed
        logger.info(f"Overall: {'✓ SHADOW TESTING PASSED' if overall_pass else '✗ SHADOW TESTING FAILED'}")
        logger.info("═" * 80)
        
        # Create exit status file
        exit_status = {
            "passed": overall_pass,
            "report_file": str(report_file),
            "timestamp": timestamp
        }
        
        with open(self.results_dir / "test_status.json", 'w') as f:
            json.dump(exit_status, f, indent=2)
        
        logger.info(f"Full report saved to: {report_file}")


@click.command()
@click.option("--target-host", default=lambda: os.getenv("TARGET_HOST", "traffic-splitter"))
@click.option("--target-port", type=int, default=lambda: int(os.getenv("TARGET_PORT", "80")))
@click.option("--test-duration-hours", type=float, default=lambda: float(os.getenv("TEST_DURATION_HOURS", "48")))
@click.option("--rps-target", type=int, default=lambda: int(os.getenv("RPS_TARGET", "50")))
@click.option("--shadow-validation", type=bool, default=lambda: os.getenv("SHADOW_VALIDATION", "true").lower() == "true")
@click.option("--results-dir", default=lambda: os.getenv("RESULTS_DIR", "/app/results"))
def main(target_host, target_port, test_duration_hours, rps_target, shadow_validation, results_dir):
    """Run performance testing for message routing engine shadow deployment."""
    
    config = TestConfig(
        target_host=target_host,
        target_port=target_port,
        test_duration_hours=test_duration_hours,
        rps_target=rps_target,
        shadow_validation=shadow_validation,
        results_dir=results_dir
    )
    
    tester = PerformanceTester(config)
    
    try:
        asyncio.run(tester.run_performance_test())
    except KeyboardInterrupt:
        logger.info("Performance test interrupted by user")
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()