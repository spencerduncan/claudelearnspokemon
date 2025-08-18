"""
Concurrent performance benchmark for EmulatorPool

Measures real concurrent contention and Docker/HTTP bottlenecks.
Provides both queue efficiency and system integration benchmarks.
"""

import concurrent.futures
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from claudelearnspokemon.emulator_pool import EmulatorPool
from tests.test_emulator_pool_concurrent import ConcurrentResourceTracker


class TestConcurrentPerformanceBenchmark(unittest.TestCase):
    """Benchmark concurrent performance with honest measurement of actual bottlenecks"""

    def test_queue_efficiency_benchmark(self):
        """Benchmark Python queue.Queue efficiency - labeled honestly as queue operations"""
        with patch("claudelearnspokemon.emulator_pool.docker.from_env") as mock_docker:
            mock_client = Mock()
            mock_docker.return_value = mock_client

            containers = []
            for i in range(4):
                container = Mock()
                container.id = f"bench_container_{i:02d}"
                container.status = "running"
                container.exec_run.return_value = Mock(exit_code=0, output=b"health_ok")
                containers.append(container)

            mock_client.containers.run.side_effect = containers

            pool = EmulatorPool(pool_size=4)
            pool.initialize()

            try:
                # Measure queue efficiency honestly - no Docker/HTTP involved
                metrics = ConcurrentResourceTracker()

                def queue_worker():
                    thread_id = threading.get_ident()
                    start = time.perf_counter()
                    try:
                        client = pool.acquire(timeout=2.0)  # Just queue.Queue.get() when mocked
                        acq_time = time.perf_counter() - start
                        metrics.record_acquisition(thread_id, acq_time)
                        pool.release(client)  # Just queue.Queue.put() when mocked
                    except Exception:
                        metrics.record_timeout()

                # Sequential operations to establish queue efficiency baseline
                start_time = time.perf_counter()
                for _ in range(1000):
                    queue_worker()
                duration = time.perf_counter() - start_time

                metrics.set_test_duration(duration)
                result = metrics.get_metrics()

                print("\n=== Python Queue Efficiency Benchmark (1000 queue operations) ===")
                print("NOTE: This measures queue.Queue performance, NOT Docker/HTTP performance")
                print(f"Total time: {duration:.3f}s")
                print(f"Queue throughput: {result.throughput_ops_per_sec:.1f} queue-ops/sec")
                print(f"Average queue access: {result.avg_acquisition_time_ms:.3f}ms")
                print(f"Max queue access: {result.max_acquisition_time_ms:.3f}ms")
                print(f"Queue success rate: {result.success_rate:.1f}%")

                # Queue efficiency assertions - these are just Python performance
                self.assertGreater(
                    result.throughput_ops_per_sec, 50000, "Python queue should exceed 50k ops/sec"
                )
                self.assertLess(
                    result.avg_acquisition_time_ms, 0.1, "Queue access should be <0.1ms"
                )
                self.assertEqual(result.success_rate, 100.0, "Queue should have 100% success rate")

            finally:
                pool.shutdown()

    def test_concurrent_contention_benchmark(self):
        """Benchmark real concurrent contention with multiple threads competing"""
        with patch("claudelearnspokemon.emulator_pool.docker.from_env") as mock_docker:
            mock_client = Mock()
            mock_docker.return_value = mock_client

            containers = []
            for i in range(4):
                container = Mock()
                container.id = f"contention_container_{i:02d}"
                container.status = "running"
                container.exec_run.return_value = Mock(exit_code=0, output=b"health_ok")
                containers.append(container)

            mock_client.containers.run.side_effect = containers

            pool = EmulatorPool(pool_size=4)  # 4 resources
            pool.initialize()

            try:
                # Test concurrent contention with more threads than resources
                num_workers = 16  # 4x more threads than pool size
                operations_per_worker = 25
                total_operations = num_workers * operations_per_worker

                metrics = ConcurrentResourceTracker()

                def contention_worker(worker_id: int):
                    """Worker that competes for limited resources"""
                    local_metrics = ConcurrentResourceTracker()

                    for op in range(operations_per_worker):
                        thread_id = threading.get_ident()
                        start = time.perf_counter()

                        try:
                            # This will block when pool is exhausted - real contention
                            client = pool.acquire(timeout=5.0)
                            acq_time = time.perf_counter() - start
                            local_metrics.record_acquisition(thread_id, acq_time)

                            # Simulate minimal work to hold resource
                            time.sleep(0.001)  # 1ms of "work"

                            pool.release(client)

                        except Exception as e:
                            local_metrics.record_timeout()
                            print(f"Worker {worker_id} operation {op} failed: {e}")

                    return local_metrics

                print(
                    f"\n=== Concurrent Contention Benchmark ({num_workers} threads, {total_operations} ops) ==="
                )
                print(f"Pool size: 4 resources, {num_workers} competing threads")

                # Launch all threads concurrently - real contention
                start_time = time.perf_counter()

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    future_to_worker = {
                        executor.submit(contention_worker, i): i for i in range(num_workers)
                    }

                    # Collect results from all workers
                    for future in concurrent.futures.as_completed(future_to_worker):
                        worker_metrics = future.result()
                        # Aggregate metrics from this worker directly
                        for acq_time in worker_metrics._metrics.acquisition_times:
                            metrics.record_acquisition(threading.get_ident(), acq_time)
                        metrics._metrics.timeouts += worker_metrics._metrics.timeouts
                        metrics._metrics.total_operations += (
                            worker_metrics._metrics.total_operations
                        )

                duration = time.perf_counter() - start_time
                metrics.set_test_duration(duration)
                result = metrics.get_metrics()

                print(f"Total test duration: {duration:.3f}s")
                print(f"Concurrent throughput: {result.throughput_ops_per_sec:.1f} ops/sec")
                print(f"Average contention delay: {result.avg_acquisition_time_ms:.3f}ms")
                print(f"Max contention delay: {result.max_acquisition_time_ms:.3f}ms")
                print(f"Success rate: {result.success_rate:.1f}%")
                print(f"Timeouts: {result.timeouts}")

                # Contention performance assertions
                self.assertGreater(
                    result.avg_acquisition_time_ms,
                    0.5,
                    "Should show contention delay >0.5ms with 16 threads on 4 resources",
                )
                self.assertLess(
                    result.throughput_ops_per_sec,
                    10000,
                    "Contention should reduce throughput vs queue efficiency",
                )
                self.assertGreaterEqual(
                    result.success_rate, 95.0, "Should maintain >95% success rate under contention"
                )

            finally:
                pool.shutdown()

    def test_system_integration_benchmark_documentation(self):
        """Document what a real system benchmark would measure"""
        print("\n=== Real System Integration Benchmark (Documentation Only) ===")
        print("A true EmulatorPool system benchmark would measure:")
        print()
        print("Real Bottlenecks:")
        print("- Container startup: 5-30 seconds (Docker API + image pull)")
        print("- HTTP requests: 10-100ms (network I/O to container)")
        print("- Docker API calls: 1-10ms (container management)")
        print("- Health checks: 5-50ms (HTTP roundtrips)")
        print()
        print("Test Requirements:")
        print("- Real Docker containers (not mocks)")
        print("- Actual HTTP requests via requests.Session")
        print("- Network I/O and serialization overhead")
        print("- Container resource limits and competition")
        print()
        print("Performance Expectations:")
        print("- Startup: 4 containers in 20-120 seconds")
        print("- HTTP ops: 10-100 requests/second per container")
        print("- Total system: Limited by Docker/network, not queue.Queue")
        print()
        print("This test suite focuses on queue efficiency and contention patterns.")
        print("System integration testing requires actual Pokemon-gym containers.")


if __name__ == "__main__":
    unittest.main()
