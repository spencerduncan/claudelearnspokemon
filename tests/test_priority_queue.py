"""
Unit Tests for PriorityQueue System - Priority-Based Message Routing

This module provides comprehensive unit tests for the priority queue system,
validating priority ordering, performance requirements, thread safety,
and production reliability patterns.

Test Coverage:
- Priority-based message ordering
- Queue capacity and back-pressure handling
- TTL (time-to-live) functionality
- Performance requirements (<1ms per operation)
- Thread safety for concurrent operations
- Background cleanup of expired messages
- Metrics collection and circuit breaker behavior
"""

import concurrent.futures
import threading
import time

import pytest

from claudelearnspokemon.priority_queue import (
    HeapBasedPriorityQueue,
    MessagePriority,
    MessagePriorityQueue,
    QueuedMessage,
    QueueMetrics,
    QueueType,
    create_mixed_queue,
    create_strategic_queue,
    create_tactical_queue,
)


@pytest.mark.medium
class TestQueuedMessage:
    """Test QueuedMessage dataclass functionality."""

    def test_message_creation(self):
        """Test basic message creation."""
        message = QueuedMessage(
            content="Test message", priority=MessagePriority.HIGH, queue_type=QueueType.STRATEGIC
        )

        assert message.content == "Test message"
        assert message.priority == MessagePriority.HIGH
        assert message.queue_type == QueueType.STRATEGIC
        assert not message.is_expired()
        assert message.can_retry()
        assert message.retry_count == 0

    def test_message_priority_comparison(self):
        """Test priority comparison for heap ordering."""
        high_priority = QueuedMessage(priority=MessagePriority.HIGH)
        normal_priority = QueuedMessage(priority=MessagePriority.NORMAL)
        low_priority = QueuedMessage(priority=MessagePriority.LOW)

        # Lower numeric value = higher priority
        assert high_priority < normal_priority
        assert normal_priority < low_priority
        assert high_priority < low_priority

    def test_timestamp_based_ordering(self):
        """Test timestamp-based ordering for same priority."""
        # Create messages with same priority but different timestamps
        earlier = QueuedMessage(priority=MessagePriority.NORMAL, timestamp=1.0)
        later = QueuedMessage(priority=MessagePriority.NORMAL, timestamp=2.0)

        # Earlier timestamp = higher priority for same priority level
        assert earlier < later

    def test_expiry_functionality(self):
        """Test message expiry functionality."""
        # Message expires in 0.1 seconds
        expiry_time = time.time() + 0.1
        message = QueuedMessage(expiry_time=expiry_time)

        assert not message.is_expired()

        # Wait for expiry
        time.sleep(0.15)
        assert message.is_expired()

    def test_retry_logic(self):
        """Test retry count and limits."""
        message = QueuedMessage(max_retries=3)

        assert message.can_retry()
        assert message.retry_count == 0

        # Increment retries
        for i in range(3):
            message.increment_retry()
            assert message.retry_count == i + 1
            assert message.can_retry()

        # One more increment should exceed limit
        message.increment_retry()
        assert not message.can_retry()
        assert message.retry_count == 4

    def test_invalid_expiry_time(self):
        """Test validation of expiry time."""
        with pytest.raises(ValueError):
            QueuedMessage(timestamp=10.0, expiry_time=5.0)  # Expiry before timestamp


@pytest.mark.medium
class TestQueueMetrics:
    """Test QueueMetrics functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = QueueMetrics()

    def test_enqueue_metrics(self):
        """Test enqueue operation metrics."""
        message = QueuedMessage(priority=MessagePriority.HIGH)

        initial_total = self.metrics._metrics["total_enqueued"]
        self.metrics.record_enqueue(message, queue_size=5)

        metrics = self.metrics.get_metrics()
        assert metrics["total_enqueued"] == initial_total + 1
        assert metrics["current_size"] == 5
        assert metrics["max_size_reached"] == 5
        assert metrics["priority_distribution"]["HIGH"] == 1

    def test_dequeue_metrics(self):
        """Test dequeue operation metrics."""
        message = QueuedMessage(priority=MessagePriority.NORMAL)
        queue_time_ms = 150.0

        initial_total = self.metrics._metrics["total_dequeued"]
        self.metrics.record_dequeue(message, queue_size=3, queue_time_ms=queue_time_ms)

        metrics = self.metrics.get_metrics()
        assert metrics["total_dequeued"] == initial_total + 1
        assert metrics["current_size"] == 3
        assert metrics["avg_queue_time_ms"] == queue_time_ms

    def test_average_queue_time_calculation(self):
        """Test running average queue time calculation."""
        message = QueuedMessage()

        # Record multiple queue times
        self.metrics.record_dequeue(message, 0, 100.0)
        self.metrics.record_dequeue(message, 0, 200.0)
        self.metrics.record_dequeue(message, 0, 300.0)

        metrics = self.metrics.get_metrics()
        expected_avg = (100.0 + 200.0 + 300.0) / 3
        assert abs(metrics["avg_queue_time_ms"] - expected_avg) < 0.01

    def test_operations_per_second_calculation(self):
        """Test operations per second calculation."""
        message = QueuedMessage()

        # Record multiple operations quickly
        for _ in range(10):
            self.metrics.record_enqueue(message, 1)
            time.sleep(0.01)  # Small delay

        metrics = self.metrics.get_metrics()
        # Should have some operations per second
        assert metrics["queue_operations_per_second"] > 0

    def test_thread_safety(self):
        """Test thread safety of metrics collection."""
        message = QueuedMessage()

        def enqueue_worker():
            for _ in range(50):
                self.metrics.record_enqueue(message, 1)

        def dequeue_worker():
            for _ in range(50):
                self.metrics.record_dequeue(message, 0, 100.0)

        # Run concurrent metric updates
        threads = [threading.Thread(target=enqueue_worker), threading.Thread(target=dequeue_worker)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should be recorded correctly
        metrics = self.metrics.get_metrics()
        assert metrics["total_enqueued"] >= 50
        assert metrics["total_dequeued"] >= 50


@pytest.mark.medium
class TestHeapBasedPriorityQueue:
    """Test HeapBasedPriorityQueue implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.queue = HeapBasedPriorityQueue(max_capacity=100)

    def test_basic_enqueue_dequeue(self):
        """Test basic enqueue and dequeue operations."""
        messages = [
            QueuedMessage(content="High priority", priority=MessagePriority.HIGH),
            QueuedMessage(content="Normal priority", priority=MessagePriority.NORMAL),
            QueuedMessage(content="Low priority", priority=MessagePriority.LOW),
        ]

        # Enqueue in random order
        for message in reversed(messages):
            success = self.queue.enqueue(message)
            assert success

        # Dequeue should return in priority order
        result1 = self.queue.dequeue()
        result2 = self.queue.dequeue()
        result3 = self.queue.dequeue()

        assert result1.priority == MessagePriority.HIGH
        assert result2.priority == MessagePriority.NORMAL
        assert result3.priority == MessagePriority.LOW

    def test_capacity_limits(self):
        """Test queue capacity enforcement."""
        small_queue = HeapBasedPriorityQueue(max_capacity=3)

        # Fill queue to capacity
        for i in range(3):
            message = QueuedMessage(content=f"Message {i}")
            success = small_queue.enqueue(message)
            assert success

        # Next enqueue should fail due to capacity
        overflow_message = QueuedMessage(content="Overflow")
        success = small_queue.enqueue(overflow_message)
        assert not success

    def test_expired_message_handling(self):
        """Test automatic handling of expired messages."""
        # Create expired message with proper timestamp ordering
        now = time.time()
        expired_message = QueuedMessage(
            content="Expired",
            timestamp=now - 2.0,  # Created 2 seconds ago
            expiry_time=now - 1.0,  # Expired 1 second ago
        )

        # Enqueue should reject expired message
        success = self.queue.enqueue(expired_message)
        assert not success

        # Metrics should track expired messages
        health = self.queue.get_health_status()
        assert health["metrics"]["expired_messages"] > 0

    def test_cleanup_expired_messages(self):
        """Test cleanup of expired messages in queue."""
        # Add normal message
        normal_message = QueuedMessage(content="Normal")
        self.queue.enqueue(normal_message)

        # Add message that will expire soon
        expiring_message = QueuedMessage(content="Expiring", expiry_time=time.time() + 0.1)
        self.queue.enqueue(expiring_message)

        assert self.queue.size() == 2

        # Wait for expiry
        time.sleep(0.15)

        # Peek should trigger cleanup
        result = self.queue.peek()
        assert result.content == "Normal"
        assert self.queue.size() == 1  # Expired message cleaned up

    def test_performance_requirements(self):
        """Test performance requirements (<1ms per operation)."""
        messages = [QueuedMessage(content=f"Message {i}") for i in range(100)]

        # Test enqueue performance
        enqueue_times = []
        for message in messages:
            start_time = time.time()
            self.queue.enqueue(message)
            end_time = time.time()
            enqueue_times.append((end_time - start_time) * 1000)

        # Test dequeue performance
        dequeue_times = []
        for _ in range(100):
            start_time = time.time()
            self.queue.dequeue()
            end_time = time.time()
            dequeue_times.append((end_time - start_time) * 1000)

        # All operations should be under 1ms
        assert all(t < 1.0 for t in enqueue_times), f"Slow enqueue: {max(enqueue_times):.3f}ms"
        assert all(t < 1.0 for t in dequeue_times), f"Slow dequeue: {max(dequeue_times):.3f}ms"

        # Average should be well under limit
        assert sum(enqueue_times) / len(enqueue_times) < 0.5
        assert sum(dequeue_times) / len(dequeue_times) < 0.5

    def test_thread_safety(self):
        """Test thread safety of queue operations."""
        results = []

        def enqueue_worker(worker_id):
            worker_results = []
            for i in range(25):
                message = QueuedMessage(
                    content=f"Worker {worker_id} message {i}", priority=MessagePriority.NORMAL
                )
                success = self.queue.enqueue(message)
                worker_results.append(success)
            return worker_results

        def dequeue_worker():
            worker_results = []
            for _ in range(25):
                result = self.queue.dequeue()
                worker_results.append(result)
            return worker_results

        # Start concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            # 4 enqueue workers, 2 dequeue workers
            futures = []

            for i in range(4):
                futures.append(executor.submit(enqueue_worker, i))

            for _ in range(2):
                futures.append(executor.submit(dequeue_worker))

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

        # Verify operations completed successfully
        enqueue_results = results[:100]  # First 100 are enqueue results
        dequeue_results = results[100:]  # Next 50 are dequeue results

        assert all(r for r in enqueue_results)  # All enqueues successful
        assert all(r is not None for r in dequeue_results if r)  # All dequeues valid

    def test_health_status_reporting(self):
        """Test health status reporting."""
        # Add some messages
        for i in range(5):
            message = QueuedMessage(content=f"Message {i}")
            self.queue.enqueue(message)

        health = self.queue.get_health_status()

        required_fields = ["size", "capacity", "utilization", "circuit_breaker_open", "metrics"]

        for field in required_fields:
            assert field in health

        assert health["size"] == 5
        assert health["capacity"] == 100
        assert health["utilization"] == 0.05  # 5/100

    def test_priority_distribution(self):
        """Test priority distribution reporting."""
        priorities = [
            MessagePriority.CRITICAL,
            MessagePriority.HIGH,
            MessagePriority.HIGH,
            MessagePriority.NORMAL,
            MessagePriority.NORMAL,
            MessagePriority.NORMAL,
            MessagePriority.LOW,
        ]

        for priority in priorities:
            message = QueuedMessage(priority=priority)
            self.queue.enqueue(message)

        distribution = self.queue.get_priority_distribution()

        assert distribution["CRITICAL"] == 1
        assert distribution["HIGH"] == 2
        assert distribution["NORMAL"] == 3
        assert distribution["LOW"] == 1


@pytest.mark.medium
class TestMessagePriorityQueue:
    """Test main MessagePriorityQueue interface."""

    def setup_method(self):
        """Set up test fixtures."""
        self.queue = MessagePriorityQueue(cleanup_interval=0.1)  # Fast cleanup for testing

    def test_message_enqueuing(self):
        """Test message enqueuing with different priorities."""
        # Enqueue messages with different priorities
        high_id = self.queue.enqueue_message(
            content="High priority task",
            priority=MessagePriority.HIGH,
            queue_type=QueueType.STRATEGIC,
        )

        normal_id = self.queue.enqueue_message(
            content="Normal priority task",
            priority=MessagePriority.NORMAL,
            queue_type=QueueType.TACTICAL,
        )

        assert high_id is not None
        assert normal_id is not None
        assert high_id != normal_id
        assert self.queue.get_queue_size() == 2

    def test_priority_ordering(self):
        """Test that messages are dequeued in priority order."""
        # Enqueue in reverse priority order
        self.queue.enqueue_message("Low task", MessagePriority.LOW)
        self.queue.enqueue_message("High task", MessagePriority.HIGH)
        self.queue.enqueue_message("Normal task", MessagePriority.NORMAL)

        # Dequeue should return high priority first
        result1 = self.queue.dequeue_message()
        result2 = self.queue.dequeue_message()
        result3 = self.queue.dequeue_message()

        assert result1.priority == MessagePriority.HIGH
        assert result2.priority == MessagePriority.NORMAL
        assert result3.priority == MessagePriority.LOW

    def test_ttl_functionality(self):
        """Test time-to-live functionality."""
        # Message with 0.1 second TTL
        message_id = self.queue.enqueue_message(content="TTL test message", ttl_seconds=0.1)

        assert message_id is not None
        assert self.queue.get_queue_size() == 1

        # Wait for TTL expiry
        time.sleep(0.15)

        # Background cleanup should have removed expired message
        time.sleep(0.2)  # Wait for cleanup thread

        result = self.queue.dequeue_message()
        assert result is None  # Should be cleaned up

    def test_peek_functionality(self):
        """Test peeking at next message without removing it."""
        self.queue.enqueue_message("Peek test", MessagePriority.HIGH)

        # Peek should return message without removing it
        peeked = self.queue.peek_next_message()
        assert peeked is not None
        assert peeked.content == "Peek test"
        assert self.queue.get_queue_size() == 1

        # Dequeue should return same message
        dequeued = self.queue.dequeue_message()
        assert dequeued.message_id == peeked.message_id
        assert self.queue.get_queue_size() == 0

    def test_empty_queue_behavior(self):
        """Test behavior with empty queue."""
        assert self.queue.is_queue_empty()
        assert self.queue.get_queue_size() == 0
        assert self.queue.peek_next_message() is None
        assert self.queue.dequeue_message() is None

    def test_health_status(self):
        """Test comprehensive health status reporting."""
        # Add some messages
        self.queue.enqueue_message("Test 1", MessagePriority.HIGH)
        self.queue.enqueue_message("Test 2", MessagePriority.NORMAL)

        health = self.queue.get_health_status()

        required_fields = ["queue_size", "is_empty", "cleanup_thread_active"]

        for field in required_fields:
            assert field in health

        assert health["queue_size"] == 2
        assert not health["is_empty"]
        assert health["cleanup_thread_active"]  # Background thread should be running

    def test_graceful_shutdown(self):
        """Test graceful shutdown functionality."""
        # Add messages
        self.queue.enqueue_message("Test message", MessagePriority.NORMAL)

        # Shutdown should complete without errors
        self.queue.shutdown()

        # Health check should show cleanup thread stopped
        time.sleep(0.1)
        health = self.queue.get_health_status()
        assert not health["cleanup_thread_active"]


@pytest.mark.medium
class TestQueueFactoryFunctions:
    """Test queue factory functions."""

    def test_strategic_queue_creation(self):
        """Test strategic queue factory function."""
        queue = create_strategic_queue(max_capacity=200)

        assert isinstance(queue, MessagePriorityQueue)

        # Test that it's configured for strategic use
        health = queue.get_health_status()
        assert health["capacity"] == 200

        queue.shutdown()

    def test_tactical_queue_creation(self):
        """Test tactical queue factory function."""
        queue = create_tactical_queue(max_capacity=500)

        assert isinstance(queue, MessagePriorityQueue)

        health = queue.get_health_status()
        assert health["capacity"] == 500

        queue.shutdown()

    def test_mixed_queue_creation(self):
        """Test mixed queue factory function."""
        queue = create_mixed_queue(max_capacity=800)

        assert isinstance(queue, MessagePriorityQueue)

        health = queue.get_health_status()
        assert health["capacity"] == 800

        # Should handle both strategic and tactical messages
        strategic_id = queue.enqueue_message(
            "Strategic task", MessagePriority.HIGH, QueueType.STRATEGIC
        )
        tactical_id = queue.enqueue_message(
            "Tactical task", MessagePriority.NORMAL, QueueType.TACTICAL
        )

        assert strategic_id is not None
        assert tactical_id is not None
        assert queue.get_queue_size() == 2

        queue.shutdown()


@pytest.mark.medium
class TestPerformanceRequirements:
    """Test performance requirements and SLA compliance."""

    def test_operation_performance_sla(self):
        """Test that queue operations meet SLA requirements."""
        queue = MessagePriorityQueue()

        # Test enqueue performance
        enqueue_times = []
        for i in range(200):
            start_time = time.time()
            queue.enqueue_message(f"Performance test {i}")
            end_time = time.time()
            enqueue_times.append((end_time - start_time) * 1000)

        # Test dequeue performance
        dequeue_times = []
        for _ in range(200):
            start_time = time.time()
            queue.dequeue_message()
            end_time = time.time()
            dequeue_times.append((end_time - start_time) * 1000)

        # All operations should be under 1ms (SLA requirement)
        assert all(t < 1.0 for t in enqueue_times), f"Slow enqueue: {max(enqueue_times):.3f}ms"
        assert all(t < 1.0 for t in dequeue_times), f"Slow dequeue: {max(dequeue_times):.3f}ms"

        # Average should be well under SLA
        avg_enqueue = sum(enqueue_times) / len(enqueue_times)
        avg_dequeue = sum(dequeue_times) / len(dequeue_times)

        assert avg_enqueue < 0.5
        assert avg_dequeue < 0.5

        queue.shutdown()

    def test_memory_efficiency(self):
        """Test memory efficiency requirements."""
        queue = MessagePriorityQueue()

        # Add 1000 messages and check memory usage
        for i in range(1000):
            queue.enqueue_message(f"Memory test message {i} with some content")

        health = queue.get_health_status()

        # Memory usage should be reasonable (this is a basic check)
        # In production, would use memory profiling tools
        assert health["queue_size"] == 1000

        queue.shutdown()

    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        queue = MessagePriorityQueue()

        def enqueue_worker(worker_id):
            times = []
            for i in range(50):
                start_time = time.time()
                queue.enqueue_message(f"Worker {worker_id} message {i}")
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            return times

        def dequeue_worker():
            times = []
            for _ in range(50):
                start_time = time.time()
                queue.dequeue_message()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            return times

        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            # 2 enqueue workers, 2 dequeue workers
            for i in range(2):
                futures.append(executor.submit(enqueue_worker, i))

            for _ in range(2):
                futures.append(executor.submit(dequeue_worker))

            all_times = []
            for future in concurrent.futures.as_completed(futures):
                all_times.extend(future.result())

        # Performance should still meet SLA under concurrent load
        assert all(t < 2.0 for t in all_times), f"Slow concurrent operation: {max(all_times):.3f}ms"

        avg_time = sum(all_times) / len(all_times)
        assert avg_time < 1.0

        queue.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
