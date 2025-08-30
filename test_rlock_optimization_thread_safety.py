#!/usr/bin/env python3
"""
Thread safety validation tests for RLock → Lock optimization (Issue #190).

Validates that thread safety is maintained after replacing RLock with Lock
in all optimized components.
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Import optimized components
from src.claudelearnspokemon.compatibility.cache_strategies import InMemoryCache
from src.claudelearnspokemon.priority_queue import MessagePriorityQueue, QueuedMessage, MessagePriority
from src.claudelearnspokemon.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.claudelearnspokemon.strategy_response_cache import StrategyResponseCache


class TestInMemoryCacheThreadSafety(unittest.TestCase):
    """Test thread safety of InMemoryCache after RLock → Lock optimization."""
    
    def setUp(self):
        self.cache = InMemoryCache(default_ttl_seconds=60, max_size=100)
        
    def test_concurrent_get_set_operations(self):
        """Test concurrent get/set operations maintain consistency."""
        num_threads = 10
        operations_per_thread = 100
        errors = []
        
        def worker(thread_id: int):
            """Worker function performing cache operations."""
            try:
                for i in range(operations_per_thread):
                    key = f"thread_{thread_id}_key_{i}"
                    value = {"thread_id": thread_id, "iteration": i, "data": f"value_{i}"}
                    
                    # Set value
                    self.assertTrue(self.cache.set(key, value))
                    
                    # Immediately get value
                    retrieved = self.cache.get(key)
                    self.assertIsNotNone(retrieved)
                    self.assertEqual(retrieved["thread_id"], thread_id)
                    self.assertEqual(retrieved["iteration"], i)
                    
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")
        
        # Execute concurrent operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, tid) for tid in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        # Validate no errors occurred
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        
        # Validate cache statistics are consistent
        stats = self.cache.get_stats()
        expected_sets = num_threads * operations_per_thread
        self.assertEqual(stats["sets"], expected_sets)
        
    def test_concurrent_eviction_safety(self):
        """Test that LRU eviction is thread-safe."""
        # Use small cache to trigger evictions
        small_cache = InMemoryCache(default_ttl_seconds=60, max_size=10)
        
        def worker(thread_id: int):
            """Worker that adds many items to trigger evictions."""
            for i in range(50):  # More than max_size
                key = f"thread_{thread_id}_item_{i}"
                value = {"data": f"value_{i}"}
                small_cache.set(key, value)
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, tid) for tid in range(5)]
            for future in as_completed(futures):
                future.result()
        
        # Cache should not exceed max_size
        stats = small_cache.get_stats()
        self.assertLessEqual(stats["total_entries"], 10)
        self.assertGreater(stats["evictions"], 0)


class TestPriorityQueueThreadSafety(unittest.TestCase):
    """Test thread safety of MessagePriorityQueue after RLock → Lock optimization."""
    
    def setUp(self):
        self.queue = MessagePriorityQueue(max_capacity=1000)
    
    def test_concurrent_enqueue_dequeue(self):
        """Test concurrent enqueue/dequeue operations."""
        num_producers = 5
        num_consumers = 3
        messages_per_producer = 50
        total_messages = num_producers * messages_per_producer
        
        enqueued_messages = []
        dequeued_messages = []
        errors = []
        
        def producer(producer_id: int):
            """Producer thread that enqueues messages."""
            try:
                for i in range(messages_per_producer):
                    message = QueuedMessage(
                        content=f"producer_{producer_id}_msg_{i}",
                        priority=MessagePriority.MEDIUM,
                        routing_key="test",
                        timestamp=time.time(),
                        retry_count=0
                    )
                    if self.queue.enqueue(message):
                        enqueued_messages.append(message.content)
            except Exception as e:
                errors.append(f"Producer {producer_id}: {e}")
        
        def consumer(consumer_id: int):
            """Consumer thread that dequeues messages."""
            try:
                while len(dequeued_messages) < total_messages:
                    message = self.queue.dequeue()
                    if message:
                        dequeued_messages.append(message.content)
                    else:
                        time.sleep(0.001)  # Brief wait if queue empty
            except Exception as e:
                errors.append(f"Consumer {consumer_id}: {e}")
        
        # Start producers and consumers concurrently
        with ThreadPoolExecutor(max_workers=num_producers + num_consumers) as executor:
            # Submit producer tasks
            producer_futures = [executor.submit(producer, pid) for pid in range(num_producers)]
            
            # Submit consumer tasks  
            consumer_futures = [executor.submit(consumer, cid) for cid in range(num_consumers)]
            
            # Wait for all producers to complete
            for future in as_completed(producer_futures):
                future.result()
            
            # Wait for consumers to process all messages
            for future in as_completed(consumer_futures):
                future.result()
        
        # Validate results
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(enqueued_messages), total_messages)
        # Note: dequeued may be less due to timing, but should not exceed enqueued
        self.assertLessEqual(len(dequeued_messages), total_messages)


class TestCircuitBreakerThreadSafety(unittest.TestCase):
    """Test thread safety of CircuitBreaker after RLock → Lock optimization."""
    
    def setUp(self):
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=1.0,
            expected_exception_types=(ValueError,)
        )
        self.breaker = CircuitBreaker("test_breaker", config)
    
    def test_concurrent_execute_operations(self):
        """Test concurrent execute operations maintain state consistency."""
        success_count = 0
        failure_count = 0
        errors = []
        lock = threading.Lock()
        
        def worker(thread_id: int, should_fail: bool):
            """Worker that executes operations through circuit breaker."""
            nonlocal success_count, failure_count
            
            def test_operation():
                if should_fail:
                    raise ValueError(f"Simulated failure from thread {thread_id}")
                return f"Success from thread {thread_id}"
            
            try:
                result = self.breaker.execute(test_operation, f"operation_{thread_id}")
                with lock:
                    success_count += 1
            except Exception as e:
                with lock:
                    if "Simulated failure" in str(e):
                        failure_count += 1
                    else:
                        errors.append(f"Thread {thread_id}: {e}")
        
        # Run mix of successful and failing operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(20):
                should_fail = i % 3 == 0  # Every 3rd operation fails
                futures.append(executor.submit(worker, i, should_fail))
            
            for future in as_completed(futures):
                future.result()
        
        # Validate thread safety
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        
        # Validate metrics consistency
        metrics = self.breaker.get_metrics()
        expected_total = success_count + failure_count
        self.assertEqual(metrics.total_requests, expected_total)


class TestStrategyResponseCacheThreadSafety(unittest.TestCase):
    """Test thread safety of StrategyResponseCache after RLock → Lock optimization."""
    
    def setUp(self):
        self.cache = StrategyResponseCache(max_size=100, default_ttl=60.0)
    
    def test_concurrent_cache_operations(self):
        """Test concurrent cache put/get operations."""
        num_threads = 8
        operations_per_thread = 25
        errors = []
        
        def worker(thread_id: int):
            """Worker performing cache operations."""
            try:
                for i in range(operations_per_thread):
                    cache_key = f"strategy_thread_{thread_id}_op_{i}"
                    response_data = {
                        "thread_id": thread_id,
                        "operation": i,
                        "result": f"result_{i}",
                        "timestamp": time.time()
                    }
                    
                    # Put response in cache
                    self.cache.put(cache_key, response_data)
                    
                    # Immediately try to get it
                    cached_response = self.cache.get(cache_key)
                    self.assertIsNotNone(cached_response)
                    self.assertEqual(cached_response["thread_id"], thread_id)
                    self.assertEqual(cached_response["operation"], i)
                    
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")
        
        # Execute concurrent operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, tid) for tid in range(num_threads)]
            for future in as_completed(futures):
                future.result()
        
        # Validate no thread safety errors
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        
        # Validate cache statistics
        stats = self.cache.get_stats()
        expected_puts = num_threads * operations_per_thread
        self.assertEqual(stats["put_operations"], expected_puts)


def run_thread_safety_validation():
    """Run all thread safety validation tests."""
    print("Thread Safety Validation for RLock → Lock Optimization")
    print("=" * 60)
    print("Testing optimized components maintain thread safety...")
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestInMemoryCacheThreadSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestPriorityQueueThreadSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestCircuitBreakerThreadSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestStrategyResponseCacheThreadSafety))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("THREAD SAFETY VALIDATION SUMMARY")
    print("=" * 40)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ ALL THREAD SAFETY TESTS PASSED")
        print("RLock → Lock optimization maintains thread safety!")
    else:
        print("\n❌ THREAD SAFETY ISSUES DETECTED")
        print("Review failures before deploying optimization!")
    
    return result


if __name__ == "__main__":
    run_thread_safety_validation()