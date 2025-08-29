#!/usr/bin/env python3
"""
Prometheus Integration Benchmark Validation Suite.

Comprehensive performance validation with statistical analysis comparing
baseline performance vs. Prometheus-enabled monitoring. Implements A/B testing
methodology with hypothesis testing for regression detection.

Features:
- Statistical significance testing (t-tests, Mann-Whitney U)
- Performance regression detection with confidence intervals
- Memory usage analysis with coefficient of variation
- Latency distribution analysis with percentiles
- SLA compliance validation with error budgets
- A/B testing framework for monitoring impact assessment

Author: Claude Code - Scientist Worker - Statistical Performance Validation
"""

import gc
import json
import numpy as np
import psutil
import scipy.stats as stats
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import Mock

# Import monitoring components
import sys
sys.path.insert(0, 'src')

from claudelearnspokemon.prometheus_exporter import PrometheusMetricsExporter
from claudelearnspokemon.metrics_endpoint import MetricsEndpoint
from claudelearnspokemon.process_metrics_collector import ProcessMetricsCollector, AggregatedMetricsCollector
from claudelearnspokemon.health_monitor import HealthMonitor
from claudelearnspokemon.circuit_breaker import CircuitBreaker, CircuitConfig


@dataclass
class BenchmarkResult:
    """Statistical container for benchmark results."""
    
    scenario_name: str
    sample_count: int
    mean_latency_ms: float
    std_dev_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    coefficient_of_variation: float
    mean_memory_mb: float
    peak_memory_mb: float
    memory_stability_cv: float
    throughput_ops_per_sec: float
    error_rate_percent: float
    
    @property
    def performance_grade(self) -> str:
        """Calculate overall performance grade."""
        if (self.p95_latency_ms < 50 and 
            self.coefficient_of_variation < 0.3 and 
            self.error_rate_percent < 1.0):
            return "A"
        elif (self.p95_latency_ms < 100 and 
              self.coefficient_of_variation < 0.5 and 
              self.error_rate_percent < 5.0):
            return "B"
        else:
            return "C"


class StatisticalAnalyzer:
    """Statistical analysis toolkit for performance benchmarks."""
    
    @staticmethod
    def calculate_descriptive_stats(data: List[float]) -> Dict[str, float]:
        """Calculate comprehensive descriptive statistics."""
        if not data:
            return {}
        
        data_array = np.array(data)
        
        return {
            "count": len(data),
            "mean": np.mean(data_array),
            "std_dev": np.std(data_array, ddof=1) if len(data) > 1 else 0.0,
            "median": np.median(data_array),
            "min": np.min(data_array),
            "max": np.max(data_array),
            "p25": np.percentile(data_array, 25),
            "p75": np.percentile(data_array, 75),
            "p90": np.percentile(data_array, 90),
            "p95": np.percentile(data_array, 95),
            "p99": np.percentile(data_array, 99),
            "iqr": np.percentile(data_array, 75) - np.percentile(data_array, 25),
            "cv": np.std(data_array, ddof=1) / np.mean(data_array) if np.mean(data_array) > 0 and len(data) > 1 else 0.0
        }
    
    @staticmethod
    def perform_two_sample_test(control_data: List[float], treatment_data: List[float], 
                               alpha: float = 0.05) -> Dict[str, Any]:
        """Perform two-sample statistical test (t-test and Mann-Whitney U)."""
        if len(control_data) < 2 or len(treatment_data) < 2:
            return {"error": "Insufficient data for statistical testing"}
        
        # Test for normality
        control_normal = stats.shapiro(control_data).pvalue > alpha if len(control_data) <= 5000 else True
        treatment_normal = stats.shapiro(treatment_data).pvalue > alpha if len(treatment_data) <= 5000 else True
        
        results = {
            "control_normal": control_normal,
            "treatment_normal": treatment_normal,
            "alpha": alpha
        }
        
        # Parametric test (t-test)
        if control_normal and treatment_normal:
            t_stat, t_pvalue = stats.ttest_ind(control_data, treatment_data, equal_var=False)
            results["t_test"] = {
                "statistic": t_stat,
                "p_value": t_pvalue,
                "significant": t_pvalue < alpha,
                "effect_direction": "treatment_faster" if t_stat > 0 else "control_faster"
            }
        
        # Non-parametric test (Mann-Whitney U)
        u_stat, u_pvalue = stats.mannwhitneyu(control_data, treatment_data, alternative='two-sided')
        results["mann_whitney"] = {
            "statistic": u_stat,
            "p_value": u_pvalue,
            "significant": u_pvalue < alpha,
            "effect_direction": "groups_differ" if u_pvalue < alpha else "no_difference"
        }
        
        # Effect size (Cohen's d)
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        pooled_std = np.sqrt(((len(control_data) - 1) * np.var(control_data, ddof=1) + 
                             (len(treatment_data) - 1) * np.var(treatment_data, ddof=1)) /
                            (len(control_data) + len(treatment_data) - 2))
        
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0.0
        
        results["effect_size"] = {
            "cohens_d": cohens_d,
            "magnitude": (
                "negligible" if abs(cohens_d) < 0.2 else
                "small" if abs(cohens_d) < 0.5 else
                "medium" if abs(cohens_d) < 0.8 else
                "large"
            )
        }
        
        # Confidence interval for difference in means
        diff_mean = treatment_mean - control_mean
        diff_se = np.sqrt(np.var(control_data, ddof=1) / len(control_data) + 
                         np.var(treatment_data, ddof=1) / len(treatment_data))
        df = len(control_data) + len(treatment_data) - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        ci_lower = diff_mean - t_critical * diff_se
        ci_upper = diff_mean + t_critical * diff_se
        
        results["confidence_interval"] = {
            "difference_in_means": diff_mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": 1 - alpha,
            "contains_zero": ci_lower <= 0 <= ci_upper
        }
        
        return results


class PrometheusIntegrationBenchmark:
    """Comprehensive benchmark suite for Prometheus integration validation."""
    
    def __init__(self, iterations_per_scenario: int = 100, warmup_iterations: int = 10):
        """
        Initialize benchmark suite.
        
        Args:
            iterations_per_scenario: Number of measurements per benchmark scenario
            warmup_iterations: Number of warmup iterations to exclude from results
        """
        self.iterations_per_scenario = iterations_per_scenario
        self.warmup_iterations = warmup_iterations
        self.analyzer = StatisticalAnalyzer()
        
        # Benchmark results storage
        self.results: Dict[str, BenchmarkResult] = {}
        self.raw_measurements: Dict[str, Dict[str, List[float]]] = {}
        
        # Load baseline if available
        self.baseline_data = self._load_baseline_data()
        
    def _load_baseline_data(self) -> Optional[Dict]:
        """Load baseline performance data if available."""
        try:
            with open("prometheus_baseline_measurements.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def _create_mock_emulator_pool(self, num_emulators: int = 4):
        """Create mock emulator pool for testing."""
        mock_pool = Mock()
        mock_pool.clients_by_port = {}
        
        for i in range(num_emulators):
            port = 8080 + i
            client = Mock()
            client.port = port
            client.container_id = f"test_container_{i:03d}"
            client.health_check = lambda: (time.sleep(0.002), True)[1]  # 2ms mock latency
            mock_pool.clients_by_port[port] = client
        
        mock_pool.get_status.return_value = {
            "active_emulators": num_emulators,
            "healthy_emulators": num_emulators,
            "total_requests": 100 + num_emulators * 10,
            "average_response_time": 0.035
        }
        
        return mock_pool
    
    def benchmark_baseline_performance(self) -> BenchmarkResult:
        """Benchmark baseline monitoring performance (without Prometheus)."""
        print("📊 Benchmarking baseline monitoring performance...")
        
        latencies = []
        memory_usage = []
        error_count = 0
        
        # Initialize monitoring components (no Prometheus)
        process_collectors = [ProcessMetricsCollector(i) for i in range(4)]
        aggregated_collector = AggregatedMetricsCollector()
        for collector in process_collectors:
            aggregated_collector.add_collector(collector)
        
        health_monitor = HealthMonitor(self._create_mock_emulator_pool())
        circuit_breakers = [CircuitBreaker(f"service_{i}", CircuitConfig()) for i in range(3)]
        
        process = psutil.Process()
        
        # Warmup
        for _ in range(self.warmup_iterations):
            self._perform_monitoring_operations(process_collectors, aggregated_collector, 
                                              health_monitor, circuit_breakers)
        
        # Actual measurements
        for i in range(self.iterations_per_scenario):
            gc.collect()
            
            start_memory = process.memory_info().rss / 1024 / 1024
            start_time = time.perf_counter()
            
            try:
                self._perform_monitoring_operations(process_collectors, aggregated_collector,
                                                  health_monitor, circuit_breakers)
            except Exception:
                error_count += 1
            
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            memory_usage.append(max(start_memory, end_memory))
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{self.iterations_per_scenario} baseline measurements")
        
        # Calculate statistics
        latency_stats = self.analyzer.calculate_descriptive_stats(latencies)
        memory_stats = self.analyzer.calculate_descriptive_stats(memory_usage)
        
        result = BenchmarkResult(
            scenario_name="baseline_monitoring",
            sample_count=len(latencies),
            mean_latency_ms=latency_stats["mean"],
            std_dev_latency_ms=latency_stats["std_dev"],
            p95_latency_ms=latency_stats["p95"],
            p99_latency_ms=latency_stats["p99"],
            min_latency_ms=latency_stats["min"],
            max_latency_ms=latency_stats["max"],
            coefficient_of_variation=latency_stats["cv"],
            mean_memory_mb=memory_stats["mean"],
            peak_memory_mb=memory_stats["max"],
            memory_stability_cv=memory_stats["cv"],
            throughput_ops_per_sec=1000.0 / latency_stats["mean"] if latency_stats["mean"] > 0 else 0.0,
            error_rate_percent=(error_count / self.iterations_per_scenario) * 100
        )
        
        self.results["baseline"] = result
        self.raw_measurements["baseline"] = {"latencies": latencies, "memory": memory_usage}
        
        return result
    
    def benchmark_prometheus_performance(self) -> BenchmarkResult:
        """Benchmark monitoring performance with Prometheus integration."""
        print("🔬 Benchmarking Prometheus-enabled monitoring performance...")
        
        latencies = []
        memory_usage = []
        error_count = 0
        
        # Initialize monitoring components with Prometheus
        process_collectors = [ProcessMetricsCollector(i) for i in range(4)]
        aggregated_collector = AggregatedMetricsCollector()
        for collector in process_collectors:
            aggregated_collector.add_collector(collector)
        
        health_monitor = HealthMonitor(self._create_mock_emulator_pool())
        circuit_breakers = [CircuitBreaker(f"service_{i}", CircuitConfig()) for i in range(3)]
        
        # Add Prometheus integration
        prometheus_exporter = PrometheusMetricsExporter()
        for collector in process_collectors:
            prometheus_exporter.register_process_collector(collector)
        prometheus_exporter.register_aggregated_collector(aggregated_collector)
        prometheus_exporter.register_health_monitor(health_monitor, "test_monitor")
        for i, breaker in enumerate(circuit_breakers):
            prometheus_exporter.register_circuit_breaker(breaker, f"service_{i}")
        
        process = psutil.Process()
        
        # Warmup
        for _ in range(self.warmup_iterations):
            self._perform_monitoring_operations_with_prometheus(
                process_collectors, aggregated_collector, health_monitor, circuit_breakers,
                prometheus_exporter
            )
        
        # Actual measurements
        for i in range(self.iterations_per_scenario):
            gc.collect()
            
            start_memory = process.memory_info().rss / 1024 / 1024
            start_time = time.perf_counter()
            
            try:
                self._perform_monitoring_operations_with_prometheus(
                    process_collectors, aggregated_collector, health_monitor, circuit_breakers,
                    prometheus_exporter
                )
            except Exception:
                error_count += 1
            
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            memory_usage.append(max(start_memory, end_memory))
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{self.iterations_per_scenario} Prometheus measurements")
        
        # Calculate statistics
        latency_stats = self.analyzer.calculate_descriptive_stats(latencies)
        memory_stats = self.analyzer.calculate_descriptive_stats(memory_usage)
        
        result = BenchmarkResult(
            scenario_name="prometheus_monitoring",
            sample_count=len(latencies),
            mean_latency_ms=latency_stats["mean"],
            std_dev_latency_ms=latency_stats["std_dev"],
            p95_latency_ms=latency_stats["p95"],
            p99_latency_ms=latency_stats["p99"],
            min_latency_ms=latency_stats["min"],
            max_latency_ms=latency_stats["max"],
            coefficient_of_variation=latency_stats["cv"],
            mean_memory_mb=memory_stats["mean"],
            peak_memory_mb=memory_stats["max"],
            memory_stability_cv=memory_stats["cv"],
            throughput_ops_per_sec=1000.0 / latency_stats["mean"] if latency_stats["mean"] > 0 else 0.0,
            error_rate_percent=(error_count / self.iterations_per_scenario) * 100
        )
        
        self.results["prometheus"] = result
        self.raw_measurements["prometheus"] = {"latencies": latencies, "memory": memory_usage}
        
        return result
    
    def benchmark_http_endpoint_performance(self) -> BenchmarkResult:
        """Benchmark HTTP /metrics endpoint performance."""
        print("🌐 Benchmarking HTTP metrics endpoint performance...")
        
        latencies = []
        memory_usage = []
        error_count = 0
        
        # Set up HTTP metrics endpoint
        prometheus_exporter = PrometheusMetricsExporter()
        metrics_endpoint = MetricsEndpoint(host="localhost", port=18005, 
                                         metrics_exporter=prometheus_exporter)
        
        process = psutil.Process()
        
        try:
            # Start endpoint
            if not metrics_endpoint.start():
                raise RuntimeError("Failed to start metrics endpoint")
            
            time.sleep(0.1)  # Allow server to start
            
            # Warmup
            for _ in range(self.warmup_iterations):
                try:
                    import requests
                    requests.get("http://localhost:18005/metrics", timeout=2)
                except Exception:
                    pass
            
            # Actual measurements
            for i in range(self.iterations_per_scenario):
                gc.collect()
                
                start_memory = process.memory_info().rss / 1024 / 1024
                start_time = time.perf_counter()
                
                try:
                    import requests
                    response = requests.get("http://localhost:18005/metrics", timeout=5)
                    if response.status_code != 200:
                        error_count += 1
                except Exception:
                    error_count += 1
                
                end_time = time.perf_counter()
                end_memory = process.memory_info().rss / 1024 / 1024
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                memory_usage.append(max(start_memory, end_memory))
                
                if (i + 1) % 20 == 0:
                    print(f"  Completed {i + 1}/{self.iterations_per_scenario} HTTP measurements")
        
        finally:
            # Cleanup
            metrics_endpoint.stop()
        
        # Calculate statistics
        latency_stats = self.analyzer.calculate_descriptive_stats(latencies)
        memory_stats = self.analyzer.calculate_descriptive_stats(memory_usage)
        
        result = BenchmarkResult(
            scenario_name="http_endpoint",
            sample_count=len(latencies),
            mean_latency_ms=latency_stats["mean"],
            std_dev_latency_ms=latency_stats["std_dev"],
            p95_latency_ms=latency_stats["p95"],
            p99_latency_ms=latency_stats["p99"],
            min_latency_ms=latency_stats["min"],
            max_latency_ms=latency_stats["max"],
            coefficient_of_variation=latency_stats["cv"],
            mean_memory_mb=memory_stats["mean"],
            peak_memory_mb=memory_stats["max"],
            memory_stability_cv=memory_stats["cv"],
            throughput_ops_per_sec=1000.0 / latency_stats["mean"] if latency_stats["mean"] > 0 else 0.0,
            error_rate_percent=(error_count / self.iterations_per_scenario) * 100
        )
        
        self.results["http_endpoint"] = result
        self.raw_measurements["http_endpoint"] = {"latencies": latencies, "memory": memory_usage}
        
        return result
    
    def _perform_monitoring_operations(self, process_collectors, aggregated_collector, 
                                     health_monitor, circuit_breakers):
        """Perform baseline monitoring operations."""
        # Update process metrics
        for i, collector in enumerate(process_collectors):
            collector.record_startup_time(0.1 + i * 0.01)
            collector.record_health_check(0.005 + i * 0.001)
            collector.update_resource_usage(40.0 + i * 5, 10.0 + i * 2)
            collector.get_performance_summary()
        
        # Get system metrics
        aggregated_collector.get_system_metrics()
        
        # Health monitor check
        health_monitor.get_stats()
        
        # Circuit breaker operations
        for breaker in circuit_breakers:
            breaker.get_health_status()
            breaker.get_metrics()
    
    def _perform_monitoring_operations_with_prometheus(self, process_collectors, aggregated_collector,
                                                     health_monitor, circuit_breakers, prometheus_exporter):
        """Perform monitoring operations with Prometheus export."""
        # Baseline operations
        self._perform_monitoring_operations(process_collectors, aggregated_collector,
                                          health_monitor, circuit_breakers)
        
        # Prometheus export
        prometheus_exporter.export_metrics()
    
    def perform_statistical_comparison(self) -> Dict[str, Any]:
        """Perform statistical comparison between baseline and Prometheus performance."""
        if "baseline" not in self.results or "prometheus" not in self.results:
            return {"error": "Missing baseline or prometheus results for comparison"}
        
        baseline_latencies = self.raw_measurements["baseline"]["latencies"]
        prometheus_latencies = self.raw_measurements["prometheus"]["latencies"]
        
        # Latency comparison
        latency_comparison = self.analyzer.perform_two_sample_test(
            baseline_latencies, prometheus_latencies
        )
        
        # Memory comparison
        baseline_memory = self.raw_measurements["baseline"]["memory"]
        prometheus_memory = self.raw_measurements["prometheus"]["memory"]
        
        memory_comparison = self.analyzer.perform_two_sample_test(
            baseline_memory, prometheus_memory
        )
        
        return {
            "latency_comparison": latency_comparison,
            "memory_comparison": memory_comparison,
            "performance_regression_detected": (
                latency_comparison.get("mann_whitney", {}).get("significant", False) and
                np.mean(prometheus_latencies) > np.mean(baseline_latencies)
            ),
            "memory_regression_detected": (
                memory_comparison.get("mann_whitney", {}).get("significant", False) and
                np.mean(prometheus_memory) > np.mean(baseline_memory)
            )
        }
    
    def validate_sla_compliance(self) -> Dict[str, Any]:
        """Validate SLA compliance across all scenarios."""
        sla_results = {}
        
        for scenario_name, result in self.results.items():
            sla_results[scenario_name] = {
                "latency_sla_50ms": result.p95_latency_ms < 50.0,
                "latency_sla_100ms": result.p95_latency_ms < 100.0,
                "memory_sla_100mb": result.peak_memory_mb < 100.0,
                "stability_sla": result.coefficient_of_variation < 0.5,
                "error_rate_sla": result.error_rate_percent < 5.0,
                "overall_grade": result.performance_grade
            }
        
        return sla_results
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive benchmark report with statistical analysis."""
        report = []
        report.append("🔬 PROMETHEUS INTEGRATION - COMPREHENSIVE BENCHMARK REPORT")
        report.append("=" * 70)
        report.append(f"Benchmark Configuration:")
        report.append(f"  • Iterations per scenario: {self.iterations_per_scenario}")
        report.append(f"  • Warmup iterations: {self.warmup_iterations}")
        report.append(f"  • Statistical significance level: α = 0.05")
        report.append("")
        
        # Individual scenario results
        report.append("📊 INDIVIDUAL SCENARIO PERFORMANCE")
        report.append("-" * 45)
        
        for scenario_name, result in self.results.items():
            report.append(f"\n{scenario_name.upper()} SCENARIO:")
            report.append(f"  Latency Statistics:")
            report.append(f"    Mean: {result.mean_latency_ms:.3f} ms ± {result.std_dev_latency_ms:.3f} ms")
            report.append(f"    P95:  {result.p95_latency_ms:.3f} ms")
            report.append(f"    P99:  {result.p99_latency_ms:.3f} ms")
            report.append(f"    CV:   {result.coefficient_of_variation:.3f}")
            report.append(f"  Memory Statistics:")
            report.append(f"    Mean: {result.mean_memory_mb:.2f} MB")
            report.append(f"    Peak: {result.peak_memory_mb:.2f} MB")
            report.append(f"    CV:   {result.memory_stability_cv:.3f}")
            report.append(f"  Performance:")
            report.append(f"    Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
            report.append(f"    Error Rate: {result.error_rate_percent:.2f}%")
            report.append(f"    Grade: {result.performance_grade}")
        
        # Statistical comparison
        if len(self.results) >= 2:
            report.append(f"\n🧮 STATISTICAL COMPARISON ANALYSIS")
            report.append("-" * 40)
            
            comparison = self.perform_statistical_comparison()
            if "error" not in comparison:
                latency_comp = comparison["latency_comparison"]
                memory_comp = comparison["memory_comparison"]
                
                report.append(f"Latency Comparison (Baseline vs Prometheus):")
                if "mann_whitney" in latency_comp:
                    mw = latency_comp["mann_whitney"]
                    report.append(f"  Mann-Whitney U: p={mw['p_value']:.4f} ({'significant' if mw['significant'] else 'not significant'})")
                
                if "effect_size" in latency_comp:
                    effect = latency_comp["effect_size"]
                    report.append(f"  Effect Size (Cohen's d): {effect['cohens_d']:.3f} ({effect['magnitude']})")
                
                if "confidence_interval" in latency_comp:
                    ci = latency_comp["confidence_interval"]
                    report.append(f"  95% CI for difference: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}] ms")
                
                report.append(f"Memory Comparison (Baseline vs Prometheus):")
                if "mann_whitney" in memory_comp:
                    mw = memory_comp["mann_whitney"]
                    report.append(f"  Mann-Whitney U: p={mw['p_value']:.4f} ({'significant' if mw['significant'] else 'not significant'})")
                
                # Regression detection
                regression_status = "🚨 REGRESSION DETECTED" if comparison.get("performance_regression_detected") else "✅ NO REGRESSION"
                memory_regression_status = "🚨 MEMORY REGRESSION" if comparison.get("memory_regression_detected") else "✅ NO MEMORY REGRESSION"
                
                report.append(f"Regression Analysis:")
                report.append(f"  Performance: {regression_status}")
                report.append(f"  Memory: {memory_regression_status}")
        
        # SLA Compliance
        report.append(f"\n✅ SLA COMPLIANCE VALIDATION")
        report.append("-" * 35)
        
        sla_results = self.validate_sla_compliance()
        for scenario_name, sla_data in sla_results.items():
            report.append(f"\n{scenario_name.upper()}:")
            for sla_name, passed in sla_data.items():
                if sla_name != "overall_grade":
                    status = "✅ PASS" if passed else "❌ FAIL"
                    report.append(f"  {sla_name}: {status}")
            report.append(f"  Overall Grade: {sla_data['overall_grade']}")
        
        # Summary and recommendations
        report.append(f"\n🎯 BENCHMARK SUMMARY & RECOMMENDATIONS")
        report.append("-" * 45)
        
        all_grades = [result.performance_grade for result in self.results.values()]
        overall_grade = min(all_grades) if all_grades else "C"
        
        if overall_grade == "A":
            report.append("✅ EXCELLENT: All performance targets met with statistical confidence")
            report.append("📋 Recommendations:")
            report.append("   • Deploy Prometheus integration to production")
            report.append("   • Enable comprehensive metrics collection")
            report.append("   • Set up monitoring dashboards and alerting")
        elif overall_grade == "B":
            report.append("⚠️  ACCEPTABLE: Performance targets mostly met, minor concerns")
            report.append("📋 Recommendations:")
            report.append("   • Deploy with monitoring for performance regression")
            report.append("   • Consider optimization for high-load scenarios")
            report.append("   • Implement gradual rollout strategy")
        else:
            report.append("❌ CONCERNING: Performance targets not consistently met")
            report.append("📋 Required Actions:")
            report.append("   • Investigate performance bottlenecks")
            report.append("   • Optimize Prometheus integration overhead")
            report.append("   • Consider alternative monitoring approaches")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "prometheus_benchmark_results.json"):
        """Save benchmark results to JSON file."""
        results_data = {
            "timestamp": time.time(),
            "configuration": {
                "iterations_per_scenario": self.iterations_per_scenario,
                "warmup_iterations": self.warmup_iterations
            },
            "results": {
                name: {
                    "scenario_name": result.scenario_name,
                    "sample_count": result.sample_count,
                    "mean_latency_ms": result.mean_latency_ms,
                    "std_dev_latency_ms": result.std_dev_latency_ms,
                    "p95_latency_ms": result.p95_latency_ms,
                    "p99_latency_ms": result.p99_latency_ms,
                    "coefficient_of_variation": result.coefficient_of_variation,
                    "mean_memory_mb": result.mean_memory_mb,
                    "peak_memory_mb": result.peak_memory_mb,
                    "throughput_ops_per_sec": result.throughput_ops_per_sec,
                    "error_rate_percent": result.error_rate_percent,
                    "performance_grade": result.performance_grade
                }
                for name, result in self.results.items()
            },
            "statistical_comparison": self.perform_statistical_comparison(),
            "sla_compliance": self.validate_sla_compliance()
        }
        
        with open(filename, "w") as f:
            json.dump(results_data, f, indent=2)


def main():
    """Execute comprehensive Prometheus integration benchmark."""
    print("🚀 Starting Comprehensive Prometheus Integration Benchmark")
    print("=" * 65)
    print()
    
    # Initialize benchmark suite
    benchmark = PrometheusIntegrationBenchmark(iterations_per_scenario=50, warmup_iterations=5)
    
    try:
        # Run all benchmark scenarios
        print("Phase 1: Baseline Performance Measurement")
        baseline_result = benchmark.benchmark_baseline_performance()
        print(f"  ✅ Baseline completed: {baseline_result.performance_grade} grade")
        print()
        
        print("Phase 2: Prometheus Integration Performance")
        prometheus_result = benchmark.benchmark_prometheus_performance()
        print(f"  ✅ Prometheus completed: {prometheus_result.performance_grade} grade")
        print()
        
        print("Phase 3: HTTP Endpoint Performance")
        http_result = benchmark.benchmark_http_endpoint_performance()
        print(f"  ✅ HTTP endpoint completed: {http_result.performance_grade} grade")
        print()
        
        # Generate comprehensive report
        print("Phase 4: Statistical Analysis & Report Generation")
        report = benchmark.generate_comprehensive_report()
        print(report)
        
        # Save results
        benchmark.save_results()
        print(f"\n💾 Detailed results saved to: prometheus_benchmark_results.json")
        
        # Determine exit code
        comparison = benchmark.perform_statistical_comparison()
        regression_detected = (comparison.get("performance_regression_detected", False) or
                             comparison.get("memory_regression_detected", False))
        
        overall_grades = [result.performance_grade for result in benchmark.results.values()]
        acceptable_performance = all(grade in ["A", "B"] for grade in overall_grades)
        
        if acceptable_performance and not regression_detected:
            print("🎉 Benchmark validation completed successfully!")
            return 0
        else:
            print("⚠️  Benchmark validation identified performance concerns")
            return 1
            
    except Exception as e:
        print(f"❌ Benchmark failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())