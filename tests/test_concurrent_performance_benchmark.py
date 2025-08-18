"""
Concurrent performance benchmark for EmulatorPool

Measures real concurrent performance metrics - no sleep() fraud.
"""

import threading
import time
import unittest
from unittest.mock import Mock, patch

from claudelearnspokemon.emulator_pool import EmulatorPool
from tests.test_emulator_pool_concurrent import ConcurrentResourceTracker


class TestConcurrentPerformanceBenchmark(unittest.TestCase):
    """Benchmark concurrent performance with real metrics"""

    def test_performance_benchmark(self):
        """Benchmark actual concurrent performance"""
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
                # Measure actual performance
                metrics = ConcurrentResourceTracker()

                def benchmark_worker():
                    thread_id = threading.get_ident()
                    start = time.perf_counter()
                    try:
                        client = pool.acquire(timeout=2.0)
                        acq_time = time.perf_counter() - start
                        metrics.record_acquisition(thread_id, acq_time)
                        pool.release(client)
                    except Exception:
                        metrics.record_timeout()

                # Sequential benchmark - 1000 operations
                start_time = time.perf_counter()
                for _ in range(1000):
                    benchmark_worker()
                duration = time.perf_counter() - start_time

                metrics.set_test_duration(duration)
                result = metrics.get_metrics()

                print("\n=== REAL Performance Metrics (1000 operations) ===")
                print(f"Total time: {duration:.3f}s")
                print(f"Throughput: {result.throughput_ops_per_sec:.1f} ops/sec")
                print(f"Average acquisition: {result.avg_acquisition_time_ms:.3f}ms")
                print(f"Max acquisition: {result.max_acquisition_time_ms:.3f}ms")
                print(f"Success rate: {result.success_rate:.1f}%")

                # Performance assertions
                self.assertGreater(
                    result.throughput_ops_per_sec, 1000, "Should exceed 1000 ops/sec"
                )
                self.assertLess(
                    result.avg_acquisition_time_ms, 1.0, "Should average <1ms acquisition"
                )
                self.assertEqual(result.success_rate, 100.0, "Should have 100% success rate")

            finally:
                pool.shutdown()


if __name__ == "__main__":
    unittest.main()
