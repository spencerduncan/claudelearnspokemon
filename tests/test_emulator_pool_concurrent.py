"""
Test suite for EmulatorPool concurrent access handling

Tests thread safety, deadlock prevention, timeout handling, and queue management.
"""

import random
import threading
import time
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

from claudelearnspokemon.emulator_pool import EmulatorPool, EmulatorPoolError


class FastTimeoutContext:
    """High-performance timeout context that eliminates background threads"""

    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if elapsed < self.timeout_seconds:
            # Sleep for remaining time to simulate timeout without threads
            remaining = self.timeout_seconds - elapsed
            time.sleep(remaining)
        return False

    def check_timeout(self):
        """Check if timeout has been reached"""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) >= self.timeout_seconds


@contextmanager
def managed_thread(target, args=(), timeout=10.0):
    """Context manager for thread lifecycle with timeout and cleanup"""
    thread = threading.Thread(target=target, args=args)
    thread.start()
    try:
        yield thread
    finally:
        thread.join(timeout=timeout)
        if thread.is_alive():
            # Thread didn't complete in time - this is a test failure
            raise AssertionError(f"Thread {thread.name} failed to complete within {timeout}s")


@pytest.mark.slow
class TestEmulatorPoolConcurrentAccess(unittest.TestCase):
    """Test concurrent access patterns and thread safety"""

    def setUp(self):
        """Set up test environment"""
        # Set up Docker mocking
        self.mock_docker_patcher = patch("claudelearnspokemon.emulator_pool.docker.from_env")
        mock_docker = self.mock_docker_patcher.start()

        self.mock_client = Mock()
        mock_docker.return_value = self.mock_client

        # Create mock containers for the pool
        containers = []
        for i in range(4):
            container = Mock()
            container.id = f"container_{i}"
            container.status = "running"
            container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
            containers.append(container)

        self.mock_client.containers.run.side_effect = containers

        # Initialize the pool with mocked Docker
        self.pool = EmulatorPool(pool_size=4, default_timeout=5.0)
        self.pool.initialize()

    def tearDown(self):
        """Clean up test environment"""
        self.pool.shutdown()
        self.mock_docker_patcher.stop()

    def test_handles_concurrent_acquisition_requests(self):
        """Test multiple threads can acquire different emulators simultaneously"""
        acquisition_results = []
        acquire_times = []
        release_times = []

        def acquire_and_release(thread_id):
            """Acquire emulator, hold briefly, then release"""
            start_time = time.time()

            client = self.pool.acquire(timeout=10.0)
            acquire_time = time.time()

            if client is not None:
                # Hold the emulator briefly - optimized timing
                time.sleep(0.05 + random.uniform(0, 0.02))

                self.pool.release(client)
                release_time = time.time()

                acquisition_results.append(
                    {
                        "thread_id": thread_id,
                        "success": True,
                        "client": client,
                        "acquire_time": acquire_time - start_time,
                        "hold_time": release_time - acquire_time,
                    }
                )
                acquire_times.append(acquire_time)
                release_times.append(release_time)
            else:
                acquisition_results.append(
                    {
                        "thread_id": thread_id,
                        "success": False,
                        "client": None,
                        "acquire_time": None,
                        "hold_time": None,
                    }
                )

        # Launch 4 threads (same as pool size)
        threads = []
        for i in range(4):
            thread = threading.Thread(target=acquire_and_release, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=15.0)
            self.assertFalse(thread.is_alive(), "Thread should have completed")

        # Verify results
        successful_acquisitions = [r for r in acquisition_results if r["success"]]
        self.assertEqual(
            len(successful_acquisitions), 4, "All threads should successfully acquire emulators"
        )

        # Verify unique clients were assigned
        clients = [r["client"] for r in successful_acquisitions]
        self.assertEqual(len(set(clients)), 4, "Each thread should get a unique emulator")

        # Verify timing constraints
        for result in successful_acquisitions:
            self.assertLess(
                result["acquire_time"], 2.0, "Acquisition should be fast with available emulators"
            )
            self.assertGreater(result["hold_time"], 0.05, "Should hold emulator for expected time")

    def test_blocks_acquisition_when_all_busy(self):
        """Test acquisition blocks when all emulators are in use"""
        block_test_results = []

        def hold_emulator_long(thread_id):
            """Acquire and hold emulator for extended time"""
            client = self.pool.acquire(timeout=2.0)
            if client:
                block_test_results.append(f"holder_{thread_id}_acquired")
                time.sleep(0.3)  # Optimized hold time
                self.pool.release(client)
                block_test_results.append(f"holder_{thread_id}_released")

        def wait_for_emulator(thread_id):
            """Try to acquire when all should be busy"""
            start_time = time.time()
            client = self.pool.acquire(timeout=3.0)
            acquire_time = time.time() - start_time

            if client:
                block_test_results.append(f"waiter_{thread_id}_acquired_after_{acquire_time:.2f}s")
                self.pool.release(client)
            else:
                block_test_results.append(f"waiter_{thread_id}_timeout_after_{acquire_time:.2f}s")

        # Start 4 holder threads first
        holder_threads = []
        for i in range(4):
            thread = threading.Thread(target=hold_emulator_long, args=(i,))
            holder_threads.append(thread)
            thread.start()

        # Give holders time to acquire
        time.sleep(0.1)

        # Start waiter threads
        waiter_threads = []
        for i in range(2):
            thread = threading.Thread(target=wait_for_emulator, args=(i,))
            waiter_threads.append(thread)
            thread.start()

        # Wait for all threads with timeout management
        all_threads = holder_threads + waiter_threads
        for thread in all_threads:
            thread.join(timeout=3.0)
            if thread.is_alive():
                self.fail(f"Thread {thread.name} did not complete within timeout")

        # Verify blocking behavior
        acquired_messages = [msg for msg in block_test_results if "acquired" in msg]
        self.assertGreaterEqual(
            len(acquired_messages), 4, "At least holders should acquire successfully"
        )

        # Verify waiters had to wait
        waiter_messages = [msg for msg in block_test_results if "waiter_" in msg]
        waited_messages = [
            msg
            for msg in waiter_messages
            if "after_" in msg and (float(msg.split("after_")[1].split("s")[0]) > 0.1)
        ]
        self.assertGreater(len(waited_messages), 0, "Some waiters should have been blocked")

    def test_timeout_handling_for_acquisitions(self):
        """Test acquisition timeout when resources unavailable"""
        timeout_results = []

        def occupy_all_emulators():
            """Acquire all emulators and hold them"""
            clients = []
            for _i in range(4):
                client = self.pool.acquire(timeout=1.0)
                if client:
                    clients.append(client)

            # Hold for optimized duration
            time.sleep(1.5)

            # Release all
            for client in clients:
                self.pool.release(client)

            return len(clients)

        def test_timeout(thread_id, timeout_seconds):
            """Test acquisition with specific timeout - fixed exception handling"""
            start_time = time.time()
            client = None
            try:
                client = self.pool.acquire(timeout=timeout_seconds)
                success = True
            except EmulatorPoolError:
                success = False

            elapsed = time.time() - start_time
            timeout_results.append(
                {
                    "thread_id": thread_id,
                    "timeout_requested": timeout_seconds,
                    "elapsed_time": elapsed,
                    "success": success,
                }
            )

            if client:
                self.pool.release(client)

        # Start occupier thread
        occupier = threading.Thread(target=occupy_all_emulators)
        occupier.start()

        # Give time for occupation
        time.sleep(0.2)

        # Start timeout test threads with faster timeouts
        test_threads = []
        timeout_values = [0.2, 0.4, 0.6, 0.8]

        for i, timeout in enumerate(timeout_values):
            thread = threading.Thread(target=test_timeout, args=(i, timeout))
            test_threads.append(thread)
            thread.start()

        # Wait for all threads with proper timeout management
        occupier.join(timeout=5.0)
        if occupier.is_alive():
            self.fail("Occupier thread did not complete")

        for thread in test_threads:
            thread.join(timeout=3.0)
            if thread.is_alive():
                self.fail("Test thread did not complete")

        # Verify timeout behavior with optimized thresholds
        for result in timeout_results:
            if result["timeout_requested"] < 1.2:  # Should timeout before release
                self.assertFalse(
                    result["success"], f"Thread {result['thread_id']} should have timed out"
                )
                self.assertAlmostEqual(
                    result["elapsed_time"],
                    result["timeout_requested"],
                    delta=0.3,
                    msg="Timeout should be respected",
                )

    def test_sequential_acquisition_after_release(self):
        """Test that threads can sequentially acquire emulators when they become available"""
        acquisition_results = []

        def occupy_all_emulators():
            """Occupy all emulators"""
            clients = []
            for _i in range(4):
                client = self.pool.acquire(timeout=1.0)
                if client:
                    clients.append(client)

            # Hold for optimized time to let waiters queue up
            time.sleep(0.4)

            # Release one at a time with optimized delays
            for client in clients:
                self.pool.release(client)
                time.sleep(0.05)  # Optimized delay between releases

        def waiter_thread(thread_id):
            """Wait for emulator to become available"""
            start_time = time.time()
            try:
                client = self.pool.acquire(timeout=3.0)  # Reduced timeout
                acquire_time = time.time()
                if client:
                    acquisition_results.append(
                        {
                            "thread_id": thread_id,
                            "acquire_time": acquire_time,
                            "wait_time": acquire_time - start_time,
                        }
                    )
                    # Hold briefly
                    time.sleep(0.02)  # Minimized hold time
                    self.pool.release(client)
            except EmulatorPoolError:
                # Expected timeout - no action needed
                pass

        # Start occupier
        occupier = threading.Thread(target=occupy_all_emulators)
        occupier.start()

        time.sleep(0.2)  # Let occupation happen

        # Start waiter threads
        waiter_threads = []
        for i in range(2):  # Only 2 waiters to make test more predictable
            thread = threading.Thread(target=waiter_thread, args=(i,))
            waiter_threads.append(thread)
            thread.start()
            time.sleep(0.02)  # Optimized delay between starts

        # Wait for completion with optimized timeouts
        occupier.join(timeout=5.0)
        if occupier.is_alive():
            self.fail("Occupier thread did not complete")

        for thread in waiter_threads:
            thread.join(timeout=3.0)
            if thread.is_alive():
                self.fail("Waiter thread did not complete")

        # Verify at least one waiter succeeded
        self.assertGreaterEqual(len(acquisition_results), 1, "At least one waiter should succeed")

        # Verify reasonable wait times with optimized threshold
        for result in acquisition_results:
            self.assertLess(result["wait_time"], 2.0, "Wait time should be reasonable")

    def test_graceful_cleanup_on_timeout(self):
        """Test cleanup when acquisition times out"""
        cleanup_results = []

        def occupy_and_check_cleanup():
            """Occupy all emulators and verify clean state after timeouts"""
            # Acquire all emulators
            clients = []
            for _i in range(4):
                client = self.pool.acquire(timeout=1.0)
                if client:
                    clients.append(client)

            # Check status during occupation
            status = self.pool.get_status()
            cleanup_results.append(f"occupied_count:{status['busy_count']}")

            # Hold for optimized duration longer than timeout tests
            time.sleep(1.2)

            # Check queue status (should have been cleaned up)
            status = self.pool.get_status()
            cleanup_results.append(f"queue_after_timeout:{status['queue_size']}")

            # Release all
            for client in clients:
                self.pool.release(client)

            # Final status check
            status = self.pool.get_status()
            cleanup_results.append(f"final_available:{status['available_count']}")

        def timeout_requester(thread_id):
            """Make request that will timeout - fixed exception handling"""
            try:
                client = self.pool.acquire(timeout=0.5)  # Will timeout faster
                cleanup_results.append(f"requester_{thread_id}_result:{client is not None}")
                if client:
                    self.pool.release(client)
            except EmulatorPoolError:
                cleanup_results.append(f"requester_{thread_id}_result:False")

        # Start occupier
        occupier = threading.Thread(target=occupy_and_check_cleanup)
        occupier.start()

        time.sleep(0.2)

        # Start multiple timeout requesters
        requesters = []
        for i in range(3):
            thread = threading.Thread(target=timeout_requester, args=(i,))
            requesters.append(thread)
            thread.start()

        # Wait for completion with proper thread management
        for thread in requesters:
            thread.join(timeout=3.0)
            if thread.is_alive():
                self.fail("Requester thread did not complete")

        occupier.join(timeout=5.0)
        if occupier.is_alive():
            self.fail("Occupier thread did not complete")

        # Verify cleanup
        occupied_msg = [msg for msg in cleanup_results if "occupied_count:" in msg][0]
        self.assertEqual(occupied_msg, "occupied_count:4", "All should be occupied")

        queue_msg = [msg for msg in cleanup_results if "queue_after_timeout:" in msg][0]
        self.assertEqual(queue_msg, "queue_after_timeout:0", "Queue should be cleaned up")

        final_msg = [msg for msg in cleanup_results if "final_available:" in msg][0]
        self.assertEqual(final_msg, "final_available:4", "All should be available finally")

        # Verify all requests timed out
        timeout_msgs = [msg for msg in cleanup_results if "requester_" in msg]
        for msg in timeout_msgs:
            self.assertIn(":False", msg, "Timeout requests should fail")


@pytest.mark.slow
class TestEmulatorPoolContextManager(unittest.TestCase):
    """Test context manager functionality"""

    def setUp(self):
        # Set up Docker mocking
        self.mock_docker_patcher = patch("claudelearnspokemon.emulator_pool.docker.from_env")
        mock_docker = self.mock_docker_patcher.start()

        self.mock_client = Mock()
        mock_docker.return_value = self.mock_client

        # Create mock containers for the pool
        containers = []
        for i in range(2):
            container = Mock()
            container.id = f"container_{i}"
            container.status = "running"
            container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
            containers.append(container)

        self.mock_client.containers.run.side_effect = containers

        self.pool = EmulatorPool(pool_size=2)
        self.pool.initialize()

    def tearDown(self):
        self.pool.shutdown()
        self.mock_docker_patcher.stop()

    def test_context_manager_automatic_release(self):
        """Test context manager automatically releases emulator"""
        with self.pool.acquire_emulator(timeout=2.0) as client:
            self.assertIsNotNone(client)
            status = self.pool.get_status()
            self.assertEqual(status["busy_count"], 1)

        # After context exit
        status = self.pool.get_status()
        self.assertEqual(status["busy_count"], 0)
        self.assertEqual(status["available_count"], 2)

    def test_context_manager_exception_handling(self):
        """Test context manager releases on exception"""
        try:
            with self.pool.acquire_emulator(timeout=2.0) as client:
                self.assertIsNotNone(client)
                status = self.pool.get_status()
                self.assertEqual(status["busy_count"], 1)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be released
        status = self.pool.get_status()
        self.assertEqual(status["busy_count"], 0)
        self.assertEqual(status["available_count"], 2)


if __name__ == "__main__":
    unittest.main()
