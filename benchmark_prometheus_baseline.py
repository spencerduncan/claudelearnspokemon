#!/usr/bin/env python3
"""
Performance baseline measurements for Prometheus metrics integration.

This benchmark establishes current performance characteristics of the monitoring
infrastructure before adding Prometheus export capabilities. Results will be used
to validate that Prometheus integration introduces <10MB memory overhead and
maintains <100ms scrape latency targets.

Author: Claude Code - Scientist Worker - Statistical Validation Focus
"""

import gc
import psutil
import statistics
import threading
import time
from typing import Dict, List, Any
from unittest.mock import Mock

# Import existing monitoring components
import sys
sys.path.insert(0, 'src')

from claudelearnspokemon.process_metrics_collector import ProcessMetricsCollector, AggregatedMetricsCollector
from claudelearnspokemon.health_monitor import HealthMonitor
from claudelearnspokemon.circuit_breaker import CircuitBreaker, CircuitConfig


class BaselineMetrics:
    """Statistical container for baseline performance measurements."""
    
    def __init__(self):
        self.memory_usage_mb: List[float] = []
        self.cpu_usage_percent: List[float] = []
        self.health_check_latency_ms: List[float] = []
        self.metrics_collection_latency_ms: List[float] = []
        self.circuit_breaker_latency_ms: List[float] = []
        
    def add_measurement(self, memory_mb: float, cpu_percent: float, 
                       health_ms: float, metrics_ms: float, circuit_ms: float):
        """Add a complete measurement sample."""
        self.memory_usage_mb.append(memory_mb)
        self.cpu_usage_percent.append(cpu_percent)
        self.health_check_latency_ms.append(health_ms)
        self.metrics_collection_latency_ms.append(metrics_ms)
        self.circuit_breaker_latency_ms.append(circuit_ms)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics for all metrics."""
        def calc_stats(data: List[float]) -> Dict[str, float]:
            if not data:
                return {"mean": 0.0, "std_dev": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": statistics.mean(data),
                "std_dev": statistics.stdev(data) if len(data) > 1 else 0.0,
                "p95": self._percentile(data, 95),
                "p99": self._percentile(data, 99),
                "min": min(data),
                "max": max(data),
                "coefficient_of_variation": (statistics.stdev(data) / statistics.mean(data)) if len(data) > 1 and statistics.mean(data) > 0 else 0.0
            }
        
        return {
            "sample_count": len(self.memory_usage_mb),
            "memory_usage_mb": calc_stats(self.memory_usage_mb),
            "cpu_usage_percent": calc_stats(self.cpu_usage_percent),
            "health_check_latency_ms": calc_stats(self.health_check_latency_ms),
            "metrics_collection_latency_ms": calc_stats(self.metrics_collection_latency_ms),
            "circuit_breaker_latency_ms": calc_stats(self.circuit_breaker_latency_ms)
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


def create_mock_emulator_pool():
    """Create mock EmulatorPool for testing."""
    mock_pool = Mock()
    mock_pool.clients_by_port = {
        8080: Mock(port=8080, container_id="container_001_test"),
        8081: Mock(port=8081, container_id="container_002_test"), 
        8082: Mock(port=8082, container_id="container_003_test"),
        8083: Mock(port=8083, container_id="container_004_test")
    }
    
    mock_pool.get_status.return_value = {
        "active_emulators": 4,
        "healthy_emulators": 4,
        "total_requests": 150,
        "average_response_time": 0.045
    }
    
    # Mock health endpoints to return 200
    def mock_health_check():
        time.sleep(0.002)  # Simulate 2ms network latency
        return True
    
    for client in mock_pool.clients_by_port.values():
        client.health_check = mock_health_check
    
    return mock_pool


def benchmark_existing_monitoring(iterations: int = 100) -> BaselineMetrics:
    """
    Benchmark the existing monitoring infrastructure performance.
    
    Args:
        iterations: Number of measurement iterations
        
    Returns:
        BaselineMetrics with performance statistics
    """
    baseline = BaselineMetrics()
    
    # Initialize monitoring components
    collectors = [ProcessMetricsCollector(i) for i in range(4)]
    aggregated_collector = AggregatedMetricsCollector()
    for collector in collectors:
        aggregated_collector.add_collector(collector)
    
    # Setup health monitor with mock emulator pool
    mock_pool = create_mock_emulator_pool()
    health_monitor = HealthMonitor(mock_pool, check_interval=1.0, health_timeout=0.5)
    
    # Setup circuit breakers
    circuit_breakers = [
        CircuitBreaker(f"test_service_{i}", CircuitConfig(failure_threshold=3))
        for i in range(4)
    ]
    
    # Warm up - ensure JIT compilation and caching is complete
    print("Performing warmup iterations...")
    for _ in range(10):
        # Exercise all monitoring components
        for i, collector in enumerate(collectors):
            collector.record_startup_time(0.1 + i * 0.01)
            collector.record_health_check(0.005 + i * 0.001)
            collector.update_resource_usage(25.0 + i * 5.0, 10.0 + i * 2.0)
        
        system_metrics = aggregated_collector.get_system_metrics()
        health_report = health_monitor.force_check()
        
        for cb in circuit_breakers:
            cb.call(lambda: "success", "test_operation")
    
    print(f"Starting {iterations} baseline measurement iterations...")
    
    # Perform actual benchmark
    process = psutil.Process()
    
    for iteration in range(iterations):
        gc.collect()  # Ensure consistent memory measurement
        
        # Measure memory before operations
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = process.cpu_percent()
        
        # Time health check operations
        health_start = time.perf_counter()
        health_report = health_monitor.force_check()
        health_duration = (time.perf_counter() - health_start) * 1000  # ms
        
        # Time metrics collection operations
        metrics_start = time.perf_counter()
        for collector in collectors:
            collector.record_startup_time(0.1)
            collector.record_health_check(0.005)
            collector.update_resource_usage(25.0, 10.0)
            _ = collector.get_performance_summary()
        
        system_metrics = aggregated_collector.get_system_metrics()
        metrics_duration = (time.perf_counter() - metrics_start) * 1000  # ms
        
        # Time circuit breaker operations
        circuit_start = time.perf_counter()
        for cb in circuit_breakers:
            _ = cb.get_health_status()
            _ = cb.get_metrics()
        circuit_duration = (time.perf_counter() - circuit_start) * 1000  # ms
        
        # Measure memory after operations
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        cpu_after = process.cpu_percent()
        
        # Calculate resource usage for this iteration
        memory_usage = max(memory_after, memory_before)  # Peak memory
        cpu_usage = max(cpu_after, cpu_before)  # Peak CPU
        
        baseline.add_measurement(
            memory_usage, cpu_usage,
            health_duration, metrics_duration, circuit_duration
        )
        
        # Progress indicator
        if (iteration + 1) % 20 == 0:
            print(f"  Completed {iteration + 1}/{iterations} iterations")
        
        # Small delay to allow CPU measurement accuracy
        time.sleep(0.001)
    
    return baseline


def analyze_performance_requirements(baseline: BaselineMetrics) -> Dict[str, bool]:
    """
    Analyze baseline performance against Prometheus integration requirements.
    
    Args:
        baseline: Measured baseline performance
        
    Returns:
        Dictionary of requirement compliance results
    """
    stats = baseline.get_statistics()
    
    # Define performance requirements from strategic plan
    requirements = {
        "health_check_latency_sla": stats["health_check_latency_ms"]["p95"] < 50.0,  # <50ms SLA
        "metrics_collection_efficiency": stats["metrics_collection_latency_ms"]["p95"] < 10.0,  # <10ms 
        "circuit_breaker_overhead": stats["circuit_breaker_latency_ms"]["p95"] < 5.0,  # <5ms
        "memory_baseline": stats["memory_usage_mb"]["max"] < 100.0,  # Current memory usage
        "performance_stability": all([
            stats[metric]["coefficient_of_variation"] < 0.5  # CV < 50%
            for metric in ["health_check_latency_ms", "metrics_collection_latency_ms", "circuit_breaker_latency_ms"]
        ])
    }
    
    return requirements


def generate_performance_report(baseline: BaselineMetrics) -> str:
    """Generate comprehensive performance report."""
    stats = baseline.get_statistics()
    requirements = analyze_performance_requirements(baseline)
    
    report = []
    report.append("🔬 PROMETHEUS INTEGRATION - PERFORMANCE BASELINE REPORT")
    report.append("=" * 65)
    report.append(f"Sample size: {stats['sample_count']} iterations")
    report.append(f"Measurement timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Memory usage analysis
    mem_stats = stats["memory_usage_mb"]
    report.append("📊 MEMORY USAGE ANALYSIS")
    report.append("-" * 30)
    report.append(f"Mean memory usage: {mem_stats['mean']:.2f} MB ± {mem_stats['std_dev']:.2f} MB")
    report.append(f"Peak memory usage: {mem_stats['max']:.2f} MB")
    report.append(f"Memory stability (CV): {mem_stats['coefficient_of_variation']:.3f}")
    report.append(f"Memory budget available: {100.0 - mem_stats['max']:.2f} MB (for Prometheus overhead)")
    report.append("")
    
    # Latency analysis
    report.append("⏱️  LATENCY PERFORMANCE ANALYSIS")
    report.append("-" * 35)
    
    for metric_name, display_name in [
        ("health_check_latency_ms", "Health Check"),
        ("metrics_collection_latency_ms", "Metrics Collection"),
        ("circuit_breaker_latency_ms", "Circuit Breaker")
    ]:
        metric_stats = stats[metric_name]
        report.append(f"{display_name}:")
        report.append(f"  Mean: {metric_stats['mean']:.3f} ms ± {metric_stats['std_dev']:.3f} ms")
        report.append(f"  P95:  {metric_stats['p95']:.3f} ms")
        report.append(f"  P99:  {metric_stats['p99']:.3f} ms")
        report.append(f"  CV:   {metric_stats['coefficient_of_variation']:.3f}")
        report.append("")
    
    # Requirements compliance
    report.append("✅ REQUIREMENTS COMPLIANCE ANALYSIS")
    report.append("-" * 40)
    
    compliance_items = [
        ("health_check_latency_sla", "Health check P95 < 50ms"),
        ("metrics_collection_efficiency", "Metrics collection P95 < 10ms"),
        ("circuit_breaker_overhead", "Circuit breaker P95 < 5ms"),
        ("memory_baseline", "Memory usage < 100MB baseline"),
        ("performance_stability", "Performance stability (CV < 0.5)")
    ]
    
    all_passed = True
    for req_key, req_desc in compliance_items:
        status = "✅ PASS" if requirements[req_key] else "❌ FAIL"
        report.append(f"{status}: {req_desc}")
        if not requirements[req_key]:
            all_passed = False
    
    report.append("")
    
    # Prometheus integration projections
    report.append("📈 PROMETHEUS INTEGRATION PROJECTIONS")
    report.append("-" * 40)
    
    prometheus_overhead_memory = 8.0  # Estimated 8MB for prometheus-client
    prometheus_scrape_latency = 25.0  # Estimated 25ms for metrics export
    
    projected_memory = mem_stats["max"] + prometheus_overhead_memory
    projected_scrape_time = stats["metrics_collection_latency_ms"]["p95"] + prometheus_scrape_latency
    
    report.append(f"Projected memory usage: {projected_memory:.2f} MB")
    report.append(f"Projected scrape latency: {projected_scrape_time:.2f} ms")
    report.append(f"Memory target compliance: {'✅' if projected_memory < 110 else '❌'} (<110MB target)")
    report.append(f"Scrape latency compliance: {'✅' if projected_scrape_time < 100 else '❌'} (<100ms target)")
    report.append("")
    
    # Summary and recommendations
    report.append("🎯 BASELINE SUMMARY & RECOMMENDATIONS")
    report.append("-" * 40)
    
    if all_passed and projected_memory < 110 and projected_scrape_time < 100:
        report.append("✅ All baseline requirements met - Proceed with Prometheus integration")
        report.append("✅ Projected overhead within acceptable limits")
        report.append("📋 Recommendations:")
        report.append("   • Implement PrometheusMetricsExporter with adapter pattern")
        report.append("   • Add histogram metrics for latency tracking")
        report.append("   • Include A/B testing capability for monitoring impact")
    else:
        report.append("⚠️  Baseline requirements not fully met - Review before integration")
        report.append("📋 Required actions:")
        if not all_passed:
            report.append("   • Investigate performance bottlenecks in existing monitoring")
        if projected_memory >= 110:
            report.append("   • Optimize memory usage before adding Prometheus overhead")
        if projected_scrape_time >= 100:
            report.append("   • Optimize metrics collection latency")
    
    return "\n".join(report)


def main():
    """Execute baseline performance benchmark."""
    print("🚀 Starting Prometheus Integration Baseline Benchmark")
    print("=" * 60)
    print()
    
    # Run baseline measurement
    baseline_metrics = benchmark_existing_monitoring(iterations=100)
    
    # Generate and display report
    report = generate_performance_report(baseline_metrics)
    print(report)
    
    # Save baseline data for future comparison
    import json
    
    baseline_data = {
        "timestamp": time.time(),
        "statistics": baseline_metrics.get_statistics(),
        "requirements_compliance": analyze_performance_requirements(baseline_metrics)
    }
    
    with open("prometheus_baseline_measurements.json", "w") as f:
        json.dump(baseline_data, f, indent=2)
    
    print(f"\n💾 Baseline data saved to: prometheus_baseline_measurements.json")
    
    # Return exit code based on compliance
    requirements = analyze_performance_requirements(baseline_metrics)
    if all(requirements.values()):
        print("🎉 Baseline benchmark completed successfully!")
        return 0
    else:
        print("⚠️  Baseline benchmark identified performance concerns")
        return 1


if __name__ == "__main__":
    exit(main())