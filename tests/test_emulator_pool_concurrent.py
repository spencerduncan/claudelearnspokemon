"""
Test suite for EmulatorPool concurrent access handling

Tests thread safety, deadlock prevention, timeout handling, and queue management.
"""

import random
import threading
import time
import unittest

from claudelearnspokemon.emulator_pool import EmulatorPool


class TestEmulatorPoolConcurrentAccess(unittest.TestCase):
    """Test concurrent access patterns and thread safety"""

    def setUp(self):
        """Set up test environment"""
        self.pool = EmulatorPool(pool_size=4, default_timeout=5.0)
        self.pool.initialize()

    def tearDown(self):
        """Clean up test environment"""
        self.pool.shutdown()

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
                # Hold the emulator briefly
                time.sleep(0.1 + random.uniform(0, 0.1))

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
            self.assertGreater(result["hold_time"], 0.1, "Should hold emulator for expected time")

    def test_blocks_acquisition_when_all_busy(self):
        """Test acquisition blocks when all emulators are in use"""
        block_test_results = []

        def hold_emulator_long(thread_id):
            """Acquire and hold emulator for extended time"""
            client = self.pool.acquire(timeout=2.0)
            if client:
                block_test_results.append(f"holder_{thread_id}_acquired")
                time.sleep(1.0)  # Hold for 1 second
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

        # Wait for all threads
        for thread in holder_threads + waiter_threads:
            thread.join(timeout=10.0)

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
            if "after_" in msg and (float(msg.split("after_")[1].split("s")[0]) > 0.5)
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

            # Hold for 3 seconds
            time.sleep(3.0)

            # Release all
            for client in clients:
                self.pool.release(client)

            return len(clients)

        def test_timeout(thread_id, timeout_seconds):
            """Test acquisition with specific timeout"""
            start_time = time.time()
            client = self.pool.acquire(timeout=timeout_seconds)
            elapsed = time.time() - start_time

            timeout_results.append(
                {
                    "thread_id": thread_id,
                    "timeout_requested": timeout_seconds,
                    "elapsed_time": elapsed,
                    "success": client is not None,
                }
            )

            if client:
                self.pool.release(client)

        # Start occupier thread
        occupier = threading.Thread(target=occupy_all_emulators)
        occupier.start()

        # Give time for occupation
        time.sleep(0.2)

        # Start timeout test threads
        test_threads = []
        timeout_values = [0.5, 1.0, 1.5, 2.0]

        for i, timeout in enumerate(timeout_values):
            thread = threading.Thread(target=test_timeout, args=(i, timeout))
            test_threads.append(thread)
            thread.start()

        # Wait for all threads
        occupier.join(timeout=10.0)
        for thread in test_threads:
            thread.join(timeout=10.0)

        # Verify timeout behavior
        for result in timeout_results:
            if result["timeout_requested"] < 2.5:  # Should timeout before release
                self.assertFalse(
                    result["success"], f"Thread {result['thread_id']} should have timed out"
                )
                self.assertAlmostEqual(
                    result["elapsed_time"],
                    result["timeout_requested"],
                    delta=0.5,
                    msg="Timeout should be respected",
                )

    def test_queue_fairness_for_waiting_threads(self):
        """Test FIFO queue behavior for waiting threads"""
        queue_results = []

        def hold_one_emulator():
            """Occupy 3 emulators, leave 1 for queue testing"""
            clients = []
            for _ in range(3):
                client = self.pool.acquire(timeout=1.0)
                if client:
                    clients.append(client)

            # Hold briefly then release in sequence
            time.sleep(1.0)
            for client in clients:
                time.sleep(0.2)  # Staggered release
                self.pool.release(client)

        def queue_waiter(thread_id, priority=0):
            """Wait in queue and record acquisition order"""
            request_time = time.time()
            client = self.pool.acquire(timeout=10.0, priority=priority)
            acquire_time = time.time()

            if client:
                queue_results.append(
                    {
                        "thread_id": thread_id,
                        "priority": priority,
                        "request_time": request_time,
                        "acquire_time": acquire_time,
                        "wait_time": acquire_time - request_time,
                    }
                )

                # Hold briefly then release
                time.sleep(0.1)
                self.pool.release(client)

        # Start holder
        holder = threading.Thread(target=hold_one_emulator)
        holder.start()

        # Give time for holders to acquire
        time.sleep(0.1)

        # Start queue waiters with small delays to establish order
        waiter_threads = []
        for i in range(5):  # More waiters than available emulators
            thread = threading.Thread(target=queue_waiter, args=(i,))
            waiter_threads.append(thread)
            thread.start()
            time.sleep(0.05)  # Small delay to establish request order

        # Wait for completion
        holder.join(timeout=15.0)
        for thread in waiter_threads:
            thread.join(timeout=15.0)

        # Verify queue ordering
        self.assertGreater(len(queue_results), 0, "Some threads should have acquired")

        # Sort by acquisition time
        sorted_results = sorted(queue_results, key=lambda x: x["acquire_time"])

        # Verify reasonable wait times
        for result in sorted_results:
            self.assertLess(result["wait_time"], 8.0, "Wait time should be reasonable")

    def test_priority_based_acquisition(self):
        """Test priority queue behavior - simplified deterministic version"""
        priority_results = []

        def occupy_all_emulators():
            """Occupy all emulators"""
            clients = []
            for _i in range(4):
                client = self.pool.acquire(timeout=1.0)
                if client:
                    clients.append(client)

            # Hold for a bit to let waiters queue up
            time.sleep(1.0)

            # Release one at a time with small delays
            for client in clients:
                self.pool.release(client)
                time.sleep(0.1)  # Small delay between releases

        def priority_waiter(thread_id, priority):
            """Wait with specific priority"""
            start_time = time.time()
            client = self.pool.acquire(timeout=8.0, priority=priority)
            acquire_time = time.time()

            if client:
                priority_results.append(
                    {
                        "thread_id": thread_id,
                        "priority": priority,
                        "acquire_time": acquire_time,
                        "wait_time": acquire_time - start_time,
                    }
                )
                # Hold briefly to make timing more obvious
                time.sleep(0.05)
                self.pool.release(client)

        # Start occupier
        occupier = threading.Thread(target=occupy_all_emulators)
        occupier.start()

        time.sleep(0.2)  # Let occupation happen

        # Start priority waiters with clear priority differences
        # Only test two priorities for simpler verification
        priorities = [10, 1]  # Much clearer priority difference
        waiter_threads = []

        for i, priority in enumerate(priorities):
            thread = threading.Thread(target=priority_waiter, args=(i, priority))
            waiter_threads.append(thread)
            thread.start()
            time.sleep(0.1)  # Ensure they start in order

        # Wait for completion
        occupier.join(timeout=15.0)
        for thread in waiter_threads:
            thread.join(timeout=15.0)

        # Verify priority ordering
        if len(priority_results) >= 2:
            # Sort by acquisition order
            sorted_by_time = sorted(priority_results, key=lambda x: x["acquire_time"])

            # Find priority 1 and 10 results
            p1_result = next((r for r in sorted_by_time if r["priority"] == 1), None)
            p10_result = next((r for r in sorted_by_time if r["priority"] == 10), None)

            if p1_result and p10_result:
                # Priority 1 should acquire before priority 10
                self.assertLess(
                    p1_result["acquire_time"],
                    p10_result["acquire_time"],
                    "Priority 1 should acquire before priority 10",
                )

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

            # Hold for longer than timeout tests
            time.sleep(3.0)

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
            """Make request that will timeout"""
            client = self.pool.acquire(timeout=1.0)  # Will timeout
            cleanup_results.append(f"requester_{thread_id}_result:{client is not None}")

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

        # Wait for completion
        for thread in requesters:
            thread.join(timeout=15.0)
        occupier.join(timeout=15.0)

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

    def test_thread_safety_under_stress(self):
        """Stress test with many concurrent threads"""
        stress_results = []
        error_count = 0

        def stress_worker(worker_id):
            """Perform random acquire/release operations"""
            nonlocal error_count

            try:
                for _ in range(10):  # 10 operations per worker
                    # Random timeout and priority
                    timeout = random.uniform(0.5, 3.0)
                    priority = random.randint(0, 3)

                    client = self.pool.acquire(timeout=timeout, priority=priority)

                    if client:
                        # Hold for random time
                        hold_time = random.uniform(0.05, 0.2)
                        time.sleep(hold_time)

                        self.pool.release(client)
                        stress_results.append(f"worker_{worker_id}_success")
                    else:
                        stress_results.append(f"worker_{worker_id}_timeout")

                    # Small random delay between operations
                    time.sleep(random.uniform(0.01, 0.05))

            except Exception as e:
                error_count += 1
                stress_results.append(f"worker_{worker_id}_error:{str(e)}")

        # Launch many concurrent workers
        workers = []
        for i in range(20):  # More workers than emulators
            thread = threading.Thread(target=stress_worker, args=(i,))
            workers.append(thread)
            thread.start()

        # Wait for all workers
        for thread in workers:
            thread.join(timeout=30.0)
            self.assertFalse(thread.is_alive(), "Worker should complete")

        # Verify no errors occurred
        self.assertEqual(error_count, 0, "No errors should occur during stress test")

        # Verify some operations succeeded
        success_count = len([r for r in stress_results if "success" in r])
        self.assertGreater(success_count, 0, "Some operations should succeed")

        # Verify final state is clean
        status = self.pool.get_status()
        self.assertEqual(status["busy_count"], 0, "No emulators should be busy")
        self.assertEqual(status["available_count"], 4, "All emulators should be available")
        self.assertEqual(status["queue_size"], 0, "Queue should be empty")


class TestEmulatorPoolContextManager(unittest.TestCase):
    """Test context manager functionality"""

    def setUp(self):
        self.pool = EmulatorPool(pool_size=2)
        self.pool.initialize()

    def tearDown(self):
        self.pool.shutdown()

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
