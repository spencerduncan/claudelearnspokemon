"""
Performance integration tests with real Docker Pokemon-gym servers.

Comprehensive performance validation using actual containers to establish
baselines, detect regressions, and validate John Botmack performance standards.

Author: John Botmack - Performance Engineering
"""

import concurrent.futures
import json
import statistics
import time
from contextlib import contextmanager

import pytest

from claudelearnspokemon.pokemon_gym_adapter import PokemonGymAdapter
from claudelearnspokemon.pokemon_gym_factory import create_pokemon_client

# Performance thresholds - John Botmack standards
PERFORMANCE_REQUIREMENTS = {
    "client_creation_ms": 100,
    "action_execution_ms": 100,
    "status_check_ms": 50,
    "session_init_ms": 1000,
    "concurrent_overhead_pct": 20,  # Max 20% overhead for concurrent operations
}

BENCHMARK_ITERATIONS = 20
CONCURRENT_CLIENT_COUNT = 4


@contextmanager
def performance_timer(operation_name: str):
    """Context manager for high-precision performance timing."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    print(f"{operation_name}: {duration_ms:.2f}ms")


class PerformanceCollector:
    """Collects and analyzes performance metrics."""

    def __init__(self):
        self.metrics: dict[str, list[float]] = {}

    def record(self, metric_name: str, value_ms: float):
        """Record a performance metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value_ms)

    def analyze(self, metric_name: str) -> dict[str, float]:
        """Analyze recorded metrics."""
        values = self.metrics.get(metric_name, [])
        if not values:
            return {}

        return {
            "count": len(values),
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "p95": sorted(values)[int(0.95 * len(values))] if len(values) > 1 else values[0],
            "p99": sorted(values)[int(0.99 * len(values))] if len(values) > 1 else values[0],
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    def validate_requirement(self, metric_name: str, requirement_ms: float) -> bool:
        """Validate metric against performance requirement."""
        analysis = self.analyze(metric_name)
        if not analysis:
            return False

        # Use 95th percentile for validation
        return analysis["p95"] <= requirement_ms

    def export_baseline(self, filename: str = "performance_baseline.json"):
        """Export performance metrics as baseline for regression testing."""
        baseline_data = {}
        for metric_name in self.metrics:
            baseline_data[metric_name] = self.analyze(metric_name)

        with open(filename, "w") as f:
            json.dump(baseline_data, f, indent=2)

        return baseline_data


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance integration tests with real Docker containers."""

    def setup_method(self):
        """Set up performance collector for each test."""
        self.perf = PerformanceCollector()

    def test_client_creation_performance_benchmark(self, pokemon_gym_container):
        """Benchmark client creation performance with real Docker server."""
        container, server_url, port = pokemon_gym_container
        container_id = container.get_wrapped_container().id

        # Benchmark different adapter types
        adapter_types = ["auto", "benchflow", "direct"]

        for adapter_type in adapter_types:
            for _i in range(BENCHMARK_ITERATIONS):
                start_time = time.perf_counter()

                client = create_pokemon_client(
                    port=port,
                    container_id=container_id,
                    adapter_type=adapter_type,
                    server_url=server_url,
                )

                creation_time = (time.perf_counter() - start_time) * 1000
                self.perf.record(f"client_creation_{adapter_type}", creation_time)

                # Cleanup
                if hasattr(client, "close"):
                    client.close()

        # Validate performance requirements
        for adapter_type in adapter_types:
            metric_name = f"client_creation_{adapter_type}"
            analysis = self.perf.analyze(metric_name)

            assert analysis["p95"] <= PERFORMANCE_REQUIREMENTS["client_creation_ms"], (
                f"Client creation for {adapter_type} P95: {analysis['p95']:.1f}ms "
                f"exceeds {PERFORMANCE_REQUIREMENTS['client_creation_ms']}ms requirement"
            )

    def test_action_execution_performance_benchmark(self, pokemon_gym_container):
        """Benchmark action execution performance with real server."""
        container, server_url, port = pokemon_gym_container

        adapter = PokemonGymAdapter(
            port=port, container_id=container.get_wrapped_container().id, server_url=server_url
        )

        try:
            adapter.initialize_session()

            # Warm up
            for _ in range(5):
                adapter.execute_action("A")

            # Benchmark action execution
            action_types = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT"]

            for action in action_types:
                for _i in range(BENCHMARK_ITERATIONS):
                    start_time = time.perf_counter()
                    adapter.execute_action(action)
                    execution_time = (time.perf_counter() - start_time) * 1000
                    self.perf.record(f"action_execution_{action}", execution_time)

            # Validate performance requirements
            for action in action_types:
                metric_name = f"action_execution_{action}"
                analysis = self.perf.analyze(metric_name)

                assert analysis["p95"] <= PERFORMANCE_REQUIREMENTS["action_execution_ms"], (
                    f"Action {action} P95: {analysis['p95']:.1f}ms "
                    f"exceeds {PERFORMANCE_REQUIREMENTS['action_execution_ms']}ms requirement"
                )

        finally:
            adapter.close()

    def test_status_check_performance_benchmark(self, pokemon_gym_container):
        """Benchmark status check performance with real server."""
        container, server_url, port = pokemon_gym_container

        adapter = PokemonGymAdapter(
            port=port, container_id=container.get_wrapped_container().id, server_url=server_url
        )

        try:
            adapter.initialize_session()

            # Benchmark status checks
            for _i in range(BENCHMARK_ITERATIONS * 2):  # More iterations for frequent operation
                start_time = time.perf_counter()
                adapter.get_session_status()
                status_time = (time.perf_counter() - start_time) * 1000
                self.perf.record("status_check", status_time)

            # Validate performance requirements
            analysis = self.perf.analyze("status_check")

            assert analysis["p95"] <= PERFORMANCE_REQUIREMENTS["status_check_ms"], (
                f"Status check P95: {analysis['p95']:.1f}ms "
                f"exceeds {PERFORMANCE_REQUIREMENTS['status_check_ms']}ms requirement"
            )

            # Status checks should be very fast
            assert (
                analysis["avg"] <= PERFORMANCE_REQUIREMENTS["status_check_ms"] * 0.5
            ), f"Status check average: {analysis['avg']:.1f}ms too high for frequent operation"

        finally:
            adapter.close()

    def test_concurrent_performance_overhead(self, multiple_pokemon_containers):
        """Test performance overhead of concurrent operations."""
        server_data = multiple_pokemon_containers

        if len(server_data) < CONCURRENT_CLIENT_COUNT:
            pytest.skip(f"Need {CONCURRENT_CLIENT_COUNT} containers for concurrent testing")

        # First, measure single-client baseline
        container, server_url, port = server_data[0]
        adapter = PokemonGymAdapter(
            port=port, container_id=container.get_wrapped_container().id, server_url=server_url
        )

        try:
            adapter.initialize_session()

            # Single-client baseline
            single_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                adapter.execute_action("A")
                single_times.append((time.perf_counter() - start_time) * 1000)

            baseline_avg = statistics.mean(single_times)

        finally:
            adapter.close()

        # Now test concurrent performance
        def concurrent_worker(worker_data):
            """Worker function for concurrent testing."""
            worker_id, (container, server_url, port) = worker_data
            adapter = PokemonGymAdapter(
                port=port, container_id=container.get_wrapped_container().id, server_url=server_url
            )

            times = []
            try:
                adapter.initialize_session()

                for _ in range(10):
                    start_time = time.perf_counter()
                    adapter.execute_action(f"A_{worker_id}")
                    times.append((time.perf_counter() - start_time) * 1000)

                return times

            finally:
                adapter.close()

        # Execute concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_CLIENT_COUNT) as executor:
            futures = []
            for i, server_info in enumerate(server_data[:CONCURRENT_CLIENT_COUNT]):
                future = executor.submit(concurrent_worker, (i, server_info))
                futures.append(future)

            # Collect results
            concurrent_results = []
            for future in concurrent.futures.as_completed(futures, timeout=60):
                concurrent_results.extend(future.result())

        concurrent_avg = statistics.mean(concurrent_results)
        overhead_pct = ((concurrent_avg - baseline_avg) / baseline_avg) * 100

        # Validate concurrent overhead
        assert overhead_pct <= PERFORMANCE_REQUIREMENTS["concurrent_overhead_pct"], (
            f"Concurrent overhead: {overhead_pct:.1f}% "
            f"exceeds {PERFORMANCE_REQUIREMENTS['concurrent_overhead_pct']}% limit"
        )

        # Record metrics
        self.perf.record("single_client_action", baseline_avg)
        for time_ms in concurrent_results:
            self.perf.record("concurrent_client_action", time_ms)

    def test_memory_usage_performance(self, pokemon_gym_container):
        """Test memory usage and performance correlation."""
        import os

        import psutil

        container, server_url, port = pokemon_gym_container

        # Get current process for memory monitoring
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        adapters = []
        try:
            # Create multiple adapters to test memory scaling
            for i in range(10):
                adapter = PokemonGymAdapter(
                    port=port,
                    container_id=container.get_wrapped_container().id,
                    server_url=server_url,
                )
                adapter.initialize_session()
                adapters.append(adapter)

                # Measure memory after each adapter
                current_memory = process.memory_info().rss
                memory_increase = (current_memory - initial_memory) / (1024 * 1024)  # MB

                # Test performance doesn't degrade with memory usage
                start_time = time.perf_counter()
                adapter.execute_action("A")
                action_time = (time.perf_counter() - start_time) * 1000

                self.perf.record("memory_scaling_action", action_time)

                # Memory should not grow excessively
                assert (
                    memory_increase < 100
                ), f"Memory usage grew {memory_increase:.1f}MB with {i+1} adapters"

            # Performance should remain consistent across memory scaling
            analysis = self.perf.analyze("memory_scaling_action")
            assert (
                analysis["max"] - analysis["min"] < 50
            ), "Performance variance too high across memory scaling"

        finally:
            for adapter in adapters:
                try:
                    adapter.close()
                except Exception:
                    pass

    def test_network_latency_impact(self, pokemon_gym_container):
        """Test performance under different network latency conditions."""
        container, server_url, port = pokemon_gym_container

        adapter = PokemonGymAdapter(
            port=port, container_id=container.get_wrapped_container().id, server_url=server_url
        )

        try:
            adapter.initialize_session()

            # Test burst operations to simulate network queueing
            burst_sizes = [1, 5, 10, 20]

            for burst_size in burst_sizes:
                burst_times = []

                for _burst in range(3):  # Multiple bursts per size
                    start_time = time.perf_counter()

                    # Execute burst of operations
                    for i in range(burst_size):
                        adapter.execute_action(f"A_{i}")

                    total_time = (time.perf_counter() - start_time) * 1000
                    avg_time_per_op = total_time / burst_size
                    burst_times.append(avg_time_per_op)

                burst_avg = statistics.mean(burst_times)
                self.perf.record(f"burst_{burst_size}_avg", burst_avg)

                # Larger bursts should not significantly increase per-operation time
                if burst_size > 1:
                    single_op_avg = statistics.mean(self.perf.metrics["burst_1_avg"])
                    overhead_pct = ((burst_avg - single_op_avg) / single_op_avg) * 100

                    assert (
                        overhead_pct <= 30
                    ), f"Burst size {burst_size} overhead: {overhead_pct:.1f}% too high"

        finally:
            adapter.close()

    def test_performance_regression_detection(self, pokemon_gym_container):
        """Establish performance baseline and test regression detection."""
        container, server_url, port = pokemon_gym_container

        adapter = PokemonGymAdapter(
            port=port, container_id=container.get_wrapped_container().id, server_url=server_url
        )

        try:
            adapter.initialize_session()

            # Comprehensive performance baseline
            operations = [
                ("initialize", lambda: adapter.initialize_session()),
                ("action_A", lambda: adapter.execute_action("A")),
                ("action_B", lambda: adapter.execute_action("B")),
                ("status", lambda: adapter.get_session_status()),
                ("stop", lambda: adapter.stop_session()),
            ]

            for op_name, operation in operations:
                if op_name == "initialize":
                    continue  # Already initialized

                for _i in range(BENCHMARK_ITERATIONS):
                    start_time = time.perf_counter()

                    try:
                        operation()
                    except Exception:
                        if op_name == "stop":
                            # Re-initialize after stop
                            adapter.initialize_session()
                        continue

                    op_time = (time.perf_counter() - start_time) * 1000
                    self.perf.record(f"baseline_{op_name}", op_time)

            # Export baseline for future regression testing
            baseline_data = self.perf.export_baseline(
                f"tests/integration/performance_baseline_{int(time.time())}.json"
            )

            # Validate all operations meet requirements
            requirements_map = {
                "action_A": PERFORMANCE_REQUIREMENTS["action_execution_ms"],
                "action_B": PERFORMANCE_REQUIREMENTS["action_execution_ms"],
                "status": PERFORMANCE_REQUIREMENTS["status_check_ms"],
            }

            for op_name, requirement in requirements_map.items():
                metric_name = f"baseline_{op_name}"
                if metric_name in baseline_data:
                    p95_time = baseline_data[metric_name]["p95"]
                    assert p95_time <= requirement, (
                        f"Baseline {op_name} P95: {p95_time:.1f}ms "
                        f"exceeds {requirement}ms requirement"
                    )

        finally:
            adapter.close()

    def test_throughput_measurement(self, pokemon_gym_container):
        """Measure maximum throughput with real Docker server."""
        container, server_url, port = pokemon_gym_container

        adapter = PokemonGymAdapter(
            port=port, container_id=container.get_wrapped_container().id, server_url=server_url
        )

        try:
            adapter.initialize_session()

            # Measure throughput over time period
            test_duration = 5.0  # seconds
            start_time = time.perf_counter()
            end_time = start_time + test_duration

            operation_count = 0
            operation_times = []

            while time.perf_counter() < end_time:
                op_start = time.perf_counter()
                adapter.execute_action("A")
                op_time = (time.perf_counter() - op_start) * 1000

                operation_times.append(op_time)
                operation_count += 1

            actual_duration = time.perf_counter() - start_time
            throughput_ops_per_sec = operation_count / actual_duration

            # Record throughput metrics
            self.perf.record("throughput_ops_per_sec", throughput_ops_per_sec)

            # Validate minimum throughput
            min_throughput = 10  # operations per second
            assert throughput_ops_per_sec >= min_throughput, (
                f"Throughput {throughput_ops_per_sec:.1f} ops/sec "
                f"below minimum {min_throughput} ops/sec"
            )

            # Validate operation times remain consistent under load
            avg_op_time = statistics.mean(operation_times)
            max_op_time = max(operation_times)

            assert max_op_time <= PERFORMANCE_REQUIREMENTS["action_execution_ms"], (
                f"Max operation time under load: {max_op_time:.1f}ms "
                f"exceeds {PERFORMANCE_REQUIREMENTS['action_execution_ms']}ms"
            )

            print(f"Sustained throughput: {throughput_ops_per_sec:.1f} ops/sec")
            print(f"Average operation time: {avg_op_time:.1f}ms")

        finally:
            adapter.close()
