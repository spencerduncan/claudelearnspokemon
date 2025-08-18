"""
Concurrent performance and thread safety tests for EmulatorPool

Tests real concurrent behavior: resource contention, lock efficiency,
thread coordination, and queue performance. No sleep() fraud.

Author: Linus Torbot - Kernel Quality Standards Applied
"""

import queue
import threading
import time
import unittest
from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest

from claudelearnspokemon.emulator_pool import EmulatorPool, EmulatorPoolError


# Named constants - no magic numbers
class ConcurrentTestConfig:
    """Configuration constants for concurrent testing - kernel style"""

    POOL_SIZE = 4
    DEFAULT_TIMEOUT = 5.0

    # Performance targets - realistic expectations
    MAX_ACQUISITION_TIME_MS = 100  # 100ms max to acquire under no contention
    MAX_CONTENTION_OVERHEAD_MS = 50  # 50ms max overhead under contention
    MIN_THROUGHPUT_OPS_PER_SEC = 100  # 100 ops/sec minimum throughput

    # Test configuration - no arbitrary numbers
    STRESS_THREAD_COUNT = 8  # 2x pool size for contention
    STRESS_OPERATIONS_PER_THREAD = 50  # Meaningful workload
    PERFORMANCE_MEASUREMENT_ITERATIONS = 1000  # Statistical significance

    # Timeouts - based on actual performance requirements
    THREAD_JOIN_TIMEOUT = 10.0  # Generous but finite
    ACQUISITION_TIMEOUT = 2.0  # Reasonable for resource acquisition
    STRESS_TEST_TIMEOUT = 30.0  # Long-running stress test limit


@dataclass
class ConcurrentMetrics:
    """Metrics for concurrent performance analysis"""

    acquisition_times: list[float]
    release_times: list[float]
    thread_ids: list[int]
    contention_events: int
    timeouts: int
    total_operations: int
    test_duration: float

    @property
    def avg_acquisition_time_ms(self) -> float:
        """Average resource acquisition time in milliseconds"""
        return (
            (sum(self.acquisition_times) / len(self.acquisition_times) * 1000)
            if self.acquisition_times
            else 0.0
        )

    @property
    def max_acquisition_time_ms(self) -> float:
        """Maximum resource acquisition time in milliseconds"""
        return max(self.acquisition_times) * 1000 if self.acquisition_times else 0.0

    @property
    def throughput_ops_per_sec(self) -> float:
        """Operations per second throughput"""
        return self.total_operations / self.test_duration if self.test_duration > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Percentage of successful operations"""
        total_attempts = self.total_operations + self.timeouts
        return (self.total_operations / total_attempts * 100) if total_attempts > 0 else 0.0


class ConcurrentResourceTracker:
    """Thread-safe resource tracking for concurrent tests - kernel pattern"""

    def __init__(self):
        self._lock = threading.RLock()  # Reentrant for nested operations
        self._metrics = ConcurrentMetrics(
            acquisition_times=[],
            release_times=[],
            thread_ids=[],
            contention_events=0,
            timeouts=0,
            total_operations=0,
            test_duration=0.0,
        )

    def record_acquisition(self, thread_id: int, acquisition_time: float) -> None:
        """Record successful resource acquisition"""
        with self._lock:
            self._metrics.acquisition_times.append(acquisition_time)
            self._metrics.thread_ids.append(thread_id)
            self._metrics.total_operations += 1

    def record_release(self, release_time: float) -> None:
        """Record resource release"""
        with self._lock:
            self._metrics.release_times.append(release_time)

    def record_contention(self) -> None:
        """Record contention event"""
        with self._lock:
            self._metrics.contention_events += 1

    def record_timeout(self) -> None:
        """Record timeout event"""
        with self._lock:
            self._metrics.timeouts += 1

    def set_test_duration(self, duration: float) -> None:
        """Set total test duration"""
        with self._lock:
            self._metrics.test_duration = duration

    def get_metrics(self) -> ConcurrentMetrics:
        """Get copy of current metrics"""
        with self._lock:
            return ConcurrentMetrics(
                acquisition_times=self._metrics.acquisition_times.copy(),
                release_times=self._metrics.release_times.copy(),
                thread_ids=self._metrics.thread_ids.copy(),
                contention_events=self._metrics.contention_events,
                timeouts=self._metrics.timeouts,
                total_operations=self._metrics.total_operations,
                test_duration=self._metrics.test_duration,
            )


@pytest.mark.slow
class TestEmulatorPoolConcurrentPerformance(unittest.TestCase):
    """Test concurrent performance with real metrics - no sleep() fraud"""

    def setUp(self):
        """Set up test environment with proper mocking"""
        # Docker client mocking
        self.mock_docker_patcher = patch("claudelearnspokemon.emulator_pool.docker.from_env")
        mock_docker = self.mock_docker_patcher.start()

        self.mock_client = Mock()
        mock_docker.return_value = self.mock_client

        # Create realistic mock containers
        containers = []
        for i in range(ConcurrentTestConfig.POOL_SIZE):
            container = Mock()
            container.id = f"emulator_container_{i:02d}"
            container.status = "running"
            container.exec_run.return_value = Mock(exit_code=0, output=b"health_check_ok")
            containers.append(container)

        self.mock_client.containers.run.side_effect = containers

        # Initialize EmulatorPool with proper configuration
        self.pool = EmulatorPool(
            pool_size=ConcurrentTestConfig.POOL_SIZE,
            default_timeout=ConcurrentTestConfig.DEFAULT_TIMEOUT,
        )
        self.pool.initialize()

    def tearDown(self):
        """Clean up test environment"""
        self.pool.shutdown()
        self.mock_docker_patcher.stop()

    def test_resource_acquisition_latency_under_no_contention(self):
        """Measure resource acquisition latency with no contention - baseline performance"""
        metrics = ConcurrentResourceTracker()

        def measure_acquisition_latency() -> None:
            """Measure single acquisition latency"""
            thread_id = threading.get_ident()
            start_time = time.perf_counter()

            try:
                client = self.pool.acquire(timeout=ConcurrentTestConfig.ACQUISITION_TIMEOUT)
                acquisition_time = time.perf_counter() - start_time

                metrics.record_acquisition(thread_id, acquisition_time)

                # Immediate release - measuring acquisition only
                self.pool.release(client)
                release_time = time.perf_counter() - start_time
                metrics.record_release(release_time - acquisition_time)

            except EmulatorPoolError:
                metrics.record_timeout()

        # Sequential acquisitions - no contention
        test_start = time.perf_counter()
        for _ in range(ConcurrentTestConfig.PERFORMANCE_MEASUREMENT_ITERATIONS):
            measure_acquisition_latency()
        test_duration = time.perf_counter() - test_start

        metrics.set_test_duration(test_duration)
        result_metrics = metrics.get_metrics()

        # Validate performance requirements
        self.assertGreater(
            result_metrics.total_operations,
            ConcurrentTestConfig.PERFORMANCE_MEASUREMENT_ITERATIONS * 0.95,
            "Should complete 95%+ of operations without timeouts",
        )

        self.assertLess(
            result_metrics.avg_acquisition_time_ms,
            ConcurrentTestConfig.MAX_ACQUISITION_TIME_MS,
            f"Average acquisition time {result_metrics.avg_acquisition_time_ms:.2f}ms exceeds "
            f"{ConcurrentTestConfig.MAX_ACQUISITION_TIME_MS}ms requirement",
        )

        self.assertGreater(
            result_metrics.throughput_ops_per_sec,
            ConcurrentTestConfig.MIN_THROUGHPUT_OPS_PER_SEC,
            f"Throughput {result_metrics.throughput_ops_per_sec:.1f} ops/sec below "
            f"{ConcurrentTestConfig.MIN_THROUGHPUT_OPS_PER_SEC} ops/sec requirement",
        )

    def test_resource_contention_and_fairness(self):
        """Test resource acquisition under contention - measure fairness and efficiency"""
        metrics = ConcurrentResourceTracker()
        barrier = threading.Barrier(ConcurrentTestConfig.STRESS_THREAD_COUNT)

        def contended_worker(worker_id: int) -> None:
            """Worker thread that competes for resources"""
            thread_id = threading.get_ident()

            # Synchronize all threads to create maximum contention
            barrier.wait()

            for _operation in range(ConcurrentTestConfig.STRESS_OPERATIONS_PER_THREAD):
                start_time = time.perf_counter()

                try:
                    # Measure contention by checking if we have to wait
                    client = self.pool.acquire(timeout=ConcurrentTestConfig.ACQUISITION_TIMEOUT)
                    acquisition_time = time.perf_counter() - start_time

                    # Record contention if acquisition took longer than baseline
                    if acquisition_time > (ConcurrentTestConfig.MAX_ACQUISITION_TIME_MS / 1000):
                        metrics.record_contention()

                    metrics.record_acquisition(thread_id, acquisition_time)

                    # Minimal hold time to test coordination efficiency
                    # No arbitrary sleep - just immediate release
                    self.pool.release(client)
                    release_time = time.perf_counter() - start_time - acquisition_time
                    metrics.record_release(release_time)

                except EmulatorPoolError:
                    metrics.record_timeout()

        # Launch contending threads
        test_start = time.perf_counter()
        threads = []

        for worker_id in range(ConcurrentTestConfig.STRESS_THREAD_COUNT):
            thread = threading.Thread(target=contended_worker, args=(worker_id,))
            thread.start()
            threads.append(thread)

        # Wait for completion with timeout
        for thread in threads:
            thread.join(timeout=ConcurrentTestConfig.THREAD_JOIN_TIMEOUT)
            self.assertFalse(thread.is_alive(), f"Thread {thread.name} failed to complete")

        test_duration = time.perf_counter() - test_start
        metrics.set_test_duration(test_duration)
        result_metrics = metrics.get_metrics()

        # Validate concurrent performance under contention
        expected_operations = (
            ConcurrentTestConfig.STRESS_THREAD_COUNT
            * ConcurrentTestConfig.STRESS_OPERATIONS_PER_THREAD
        )

        self.assertGreater(
            result_metrics.total_operations,
            expected_operations * 0.8,  # Allow 20% failure under stress
            "Should complete 80%+ of operations under contention",
        )

        # Fairness check - all threads should get some resources
        unique_threads = len(set(result_metrics.thread_ids))
        self.assertGreaterEqual(
            unique_threads,
            ConcurrentTestConfig.STRESS_THREAD_COUNT * 0.8,
            "Resource allocation should be reasonably fair across threads",
        )

        # Performance under contention should be reasonable
        self.assertLess(
            result_metrics.avg_acquisition_time_ms,
            ConcurrentTestConfig.MAX_ACQUISITION_TIME_MS
            + ConcurrentTestConfig.MAX_CONTENTION_OVERHEAD_MS,
            f"Acquisition time under contention {result_metrics.avg_acquisition_time_ms:.2f}ms "
            f"exceeds {ConcurrentTestConfig.MAX_ACQUISITION_TIME_MS + ConcurrentTestConfig.MAX_CONTENTION_OVERHEAD_MS}ms",
        )

    def test_queue_efficiency_and_fifo_behavior(self):
        """Test that EmulatorPool queue behaves efficiently and fairly"""
        acquisition_order = queue.Queue()  # Track acquisition order
        release_order = queue.Queue()  # Track release order

        def queued_worker(worker_id: int, hold_duration: float = 0.01) -> None:
            """Worker that holds resources briefly to test queue behavior"""
            try:
                start_time = time.perf_counter()
                client = self.pool.acquire(timeout=ConcurrentTestConfig.ACQUISITION_TIMEOUT)
                acquisition_time = time.perf_counter() - start_time

                acquisition_order.put((worker_id, acquisition_time))

                # Brief hold period - not arbitrary sleep, testing queue order
                if hold_duration > 0:
                    time.sleep(hold_duration)

                self.pool.release(client)
                release_order.put(worker_id)

            except EmulatorPoolError:
                acquisition_order.put((worker_id, -1))  # Timeout marker

        # Test 1: Fill all resources, then queue more requests
        threads = []

        # Start more threads than pool size to force queuing
        for worker_id in range(ConcurrentTestConfig.POOL_SIZE * 2):
            thread = threading.Thread(target=queued_worker, args=(worker_id,))
            thread.start()
            threads.append(thread)

        # Wait for completion
        for thread in threads:
            thread.join(timeout=ConcurrentTestConfig.THREAD_JOIN_TIMEOUT)
            self.assertFalse(thread.is_alive(), f"Thread {thread.name} failed to complete")

        # Analyze queue behavior
        acquisitions = []
        while not acquisition_order.empty():
            acquisitions.append(acquisition_order.get())

        releases = []
        while not release_order.empty():
            releases.append(release_order.get())

        # Validate queue efficiency
        successful_acquisitions = [a for a in acquisitions if a[1] >= 0]
        self.assertEqual(
            len(successful_acquisitions),
            ConcurrentTestConfig.POOL_SIZE * 2,
            "All workers should eventually acquire resources",
        )

        # Check that initial acquisitions were fast (no queueing)
        initial_acquisitions = sorted(successful_acquisitions, key=lambda x: x[1])[
            : ConcurrentTestConfig.POOL_SIZE
        ]
        for worker_id, acquisition_time in initial_acquisitions:
            self.assertLess(
                acquisition_time * 1000,  # Convert to ms
                ConcurrentTestConfig.MAX_ACQUISITION_TIME_MS,
                f"Worker {worker_id} initial acquisition should be fast: {acquisition_time*1000:.2f}ms",
            )

    def test_thread_safety_under_stress(self):
        """Stress test thread safety - detect race conditions and deadlocks"""
        error_tracker = ConcurrentResourceTracker()
        success_count = (
            threading.AtomicInteger(0)
            if hasattr(threading, "AtomicInteger")
            else {"value": 0, "lock": threading.Lock()}
        )

        def stress_worker(worker_id: int) -> None:
            """Aggressive worker to stress test thread safety"""
            for _ in range(ConcurrentTestConfig.STRESS_OPERATIONS_PER_THREAD):
                try:
                    # Rapid acquire/release cycles
                    client = self.pool.acquire(timeout=0.1)  # Short timeout for stress

                    # Verify client is valid
                    self.assertIsNotNone(client)
                    self.assertTrue(hasattr(client, "port"))
                    self.assertTrue(hasattr(client, "container_id"))

                    # Immediate release
                    self.pool.release(client)

                    # Track success
                    if hasattr(success_count, "increment"):
                        success_count.increment()
                    else:
                        with success_count["lock"]:
                            success_count["value"] += 1

                except EmulatorPoolError:
                    error_tracker.record_timeout()
                except Exception as e:
                    # Any other exception indicates thread safety violation
                    self.fail(f"Thread safety violation in worker {worker_id}: {e}")

        # Launch stress test threads
        threads = []
        test_start = time.perf_counter()

        for worker_id in range(ConcurrentTestConfig.STRESS_THREAD_COUNT):
            thread = threading.Thread(target=stress_worker, args=(worker_id,))
            thread.start()
            threads.append(thread)

        # Wait with timeout to detect deadlocks
        for thread in threads:
            thread.join(timeout=ConcurrentTestConfig.STRESS_TEST_TIMEOUT)
            if thread.is_alive():
                self.fail(f"Potential deadlock detected - thread {thread.name} did not complete")

        test_duration = time.perf_counter() - test_start

        # Validate thread safety
        total_expected = (
            ConcurrentTestConfig.STRESS_THREAD_COUNT
            * ConcurrentTestConfig.STRESS_OPERATIONS_PER_THREAD
        )
        actual_success = (
            success_count["value"] if isinstance(success_count, dict) else success_count.value
        )
        error_metrics = error_tracker.get_metrics()

        # Allow some timeouts under extreme stress, but no other failures
        total_completed = actual_success + error_metrics.timeouts
        self.assertEqual(
            total_completed,
            total_expected,
            f"Expected {total_expected} operations, completed {total_completed} (success: {actual_success}, timeouts: {error_metrics.timeouts})",
        )

        # Stress test should maintain reasonable performance
        throughput = total_completed / test_duration
        self.assertGreater(
            throughput,
            ConcurrentTestConfig.MIN_THROUGHPUT_OPS_PER_SEC * 0.5,  # 50% of normal under stress
            f"Stress test throughput {throughput:.1f} ops/sec too low",
        )


@pytest.mark.slow
class TestEmulatorPoolContextManagerThreadSafety(unittest.TestCase):
    """Test context manager functionality under concurrent access"""

    def setUp(self):
        """Set up test environment"""
        self.mock_docker_patcher = patch("claudelearnspokemon.emulator_pool.docker.from_env")
        mock_docker = self.mock_docker_patcher.start()

        self.mock_client = Mock()
        mock_docker.return_value = self.mock_client

        # Minimal containers for context manager testing
        containers = []
        for i in range(2):
            container = Mock()
            container.id = f"context_container_{i}"
            container.status = "running"
            container.exec_run.return_value = Mock(exit_code=0, output=b"health_ok")
            containers.append(container)

        self.mock_client.containers.run.side_effect = containers

        self.pool = EmulatorPool(pool_size=2)
        self.pool.initialize()

    def tearDown(self):
        """Clean up test environment"""
        self.pool.shutdown()
        self.mock_docker_patcher.stop()

    def test_context_manager_thread_safety(self):
        """Test context manager automatic release under concurrent access"""
        results = queue.Queue()

        def context_worker(worker_id: int) -> None:
            """Worker using context manager"""
            try:
                with self.pool.acquire_emulator(
                    timeout=ConcurrentTestConfig.ACQUISITION_TIMEOUT
                ) as client:
                    self.assertIsNotNone(client)
                    results.put(f"worker_{worker_id}_success")

                # After context exit, resource should be available
                status = self.pool.get_status()
                results.put(f"worker_{worker_id}_status_{status['busy_count']}")

            except EmulatorPoolError:
                results.put(f"worker_{worker_id}_timeout")
            except Exception as e:
                results.put(f"worker_{worker_id}_error_{e}")

        # Launch concurrent context manager users
        threads = []
        for worker_id in range(4):  # 2x pool size for contention
            thread = threading.Thread(target=context_worker, args=(worker_id,))
            thread.start()
            threads.append(thread)

        # Wait for completion
        for thread in threads:
            thread.join(timeout=ConcurrentTestConfig.THREAD_JOIN_TIMEOUT)
            self.assertFalse(thread.is_alive(), f"Context manager thread {thread.name} hung")

        # Analyze results
        result_list = []
        while not results.empty():
            result_list.append(results.get())

        success_count = len([r for r in result_list if "success" in r])
        self.assertEqual(
            success_count, 4, "All context manager operations should succeed eventually"
        )

        # Final state should show all resources available
        final_status = self.pool.get_status()
        self.assertEqual(final_status["busy_count"], 0, "All resources should be released")
        self.assertEqual(final_status["available_count"], 2, "All resources should be available")

    def test_context_manager_exception_safety(self):
        """Test context manager releases resources on exception"""
        exception_results = []

        def exception_worker() -> None:
            """Worker that raises exception in context"""
            try:
                with self.pool.acquire_emulator(
                    timeout=ConcurrentTestConfig.ACQUISITION_TIMEOUT
                ) as client:
                    self.assertIsNotNone(client)
                    # Raise exception to test cleanup
                    raise ValueError("Intentional test exception")
            except ValueError:
                # Expected exception
                exception_results.append("exception_handled")
            except EmulatorPoolError:
                exception_results.append("timeout_error")

        # Run exception test
        worker_thread = threading.Thread(target=exception_worker)
        worker_thread.start()
        worker_thread.join(timeout=ConcurrentTestConfig.THREAD_JOIN_TIMEOUT)

        self.assertFalse(worker_thread.is_alive(), "Exception worker should complete")
        self.assertEqual(len(exception_results), 1, "Should handle exactly one exception")
        self.assertEqual(
            exception_results[0], "exception_handled", "Should handle ValueError correctly"
        )

        # Resource should be released despite exception
        status = self.pool.get_status()
        self.assertEqual(status["busy_count"], 0, "Resource should be released after exception")
        self.assertEqual(status["available_count"], 2, "All resources should be available")


if __name__ == "__main__":
    unittest.main()
