#!/usr/bin/env python3
"""
Quick validation demo for the message routing engine.
Tests core functionality without complex integration dependencies.
"""

# Set up path
import sys
import time
from unittest.mock import Mock

sys.path.insert(0, "/home/sd/worktrees/issue-59/src")

from claudelearnspokemon.message_classifier import MessageClassifier, MessageType
from claudelearnspokemon.priority_queue import MessagePriority, MessagePriorityQueue
from claudelearnspokemon.routing_strategy import (
    WorkerInfo,
    WorkerState,
    create_least_loaded_strategy,
)


def test_message_classification():
    """Test message classification functionality."""
    print("Testing Message Classification...")

    classifier = MessageClassifier()

    # Test strategic message
    strategic_result = classifier.classify_message("Plan the Pokemon Red speedrun strategy")
    print(
        f"Strategic classification: {strategic_result.message_type.value} (confidence: {strategic_result.confidence:.2f})"
    )

    # Test tactical message
    tactical_result = classifier.classify_message("Implement the debug function for Pokemon gym")
    print(
        f"Tactical classification: {tactical_result.message_type.value} (confidence: {tactical_result.confidence:.2f})"
    )

    # Test performance
    start_time = time.time()
    for _ in range(100):
        classifier.classify_message("Performance test message")
    end_time = time.time()

    avg_time = (end_time - start_time) / 100 * 1000
    print(f"Average classification time: {avg_time:.2f}ms")

    assert strategic_result.message_type == MessageType.STRATEGIC
    assert tactical_result.message_type == MessageType.TACTICAL
    assert avg_time < 5.0  # <5ms requirement

    print("✅ Message classification tests passed\n")


def test_priority_queue():
    """Test priority queue functionality."""
    print("Testing Priority Queue...")

    queue = MessagePriorityQueue()

    # Enqueue messages with different priorities
    critical_id = queue.enqueue_message("Critical task", MessagePriority.CRITICAL)
    normal_id = queue.enqueue_message("Normal task", MessagePriority.NORMAL)
    high_id = queue.enqueue_message("High task", MessagePriority.HIGH)

    print(f"Enqueued 3 messages: {critical_id}, {normal_id}, {high_id}")

    # Dequeue should return in priority order
    first = queue.dequeue_message()
    second = queue.dequeue_message()
    third = queue.dequeue_message()

    print(f"Dequeue order: {first.priority.name}, {second.priority.name}, {third.priority.name}")

    assert first.priority == MessagePriority.CRITICAL
    assert second.priority == MessagePriority.HIGH
    assert third.priority == MessagePriority.NORMAL

    # Test performance
    start_time = time.time()
    for i in range(1000):
        queue.enqueue_message(f"Perf test {i}", MessagePriority.NORMAL)

    for _ in range(1000):
        queue.dequeue_message()

    end_time = time.time()
    avg_time = (end_time - start_time) / 2000 * 1000
    print(f"Average queue operation time: {avg_time:.3f}ms")

    assert avg_time < 1.0  # <1ms requirement

    queue.shutdown()
    print("✅ Priority queue tests passed\n")


def test_routing_strategy():
    """Test routing strategy functionality."""
    print("Testing Routing Strategy...")

    strategy = create_least_loaded_strategy()

    # Create mock workers with different loads
    mock_workers = []
    for i in range(4):
        mock_process = Mock()
        worker = WorkerInfo(
            worker_id=f"worker_{i}",
            process=mock_process,
            state=WorkerState.HEALTHY,
            current_load=i * 2,  # 0, 2, 4, 6
            max_capacity=10,
        )
        mock_workers.append(worker)

    # Test worker selection
    from claudelearnspokemon.priority_queue import QueuedMessage

    message = QueuedMessage(content="Test routing")

    selected_worker = strategy.select_worker(message, mock_workers)
    print(f"Selected worker: {selected_worker}")

    # Should select worker_0 (lowest load)
    assert selected_worker == "worker_0"

    # Test performance
    start_time = time.time()
    for _ in range(1000):
        strategy.select_worker(message, mock_workers)
    end_time = time.time()

    avg_time = (end_time - start_time) / 1000 * 1000
    print(f"Average routing decision time: {avg_time:.3f}ms")

    assert avg_time < 5.0  # <5ms requirement

    print("✅ Routing strategy tests passed\n")


def test_end_to_end_performance():
    """Test end-to-end routing performance."""
    print("Testing End-to-End Performance...")

    classifier = MessageClassifier()
    queue = MessagePriorityQueue()
    strategy = create_least_loaded_strategy()

    # Create mock workers
    mock_workers = []
    for i in range(4):
        mock_process = Mock()
        worker = WorkerInfo(
            worker_id=f"worker_{i}",
            process=mock_process,
            state=WorkerState.HEALTHY,
            current_load=0,
            max_capacity=10,
        )
        mock_workers.append(worker)

    # Test complete pipeline
    test_messages = [
        "Strategic Pokemon Red route optimization planning",
        "Implement Pokemon gym script debug function",
        "Analyze the overall speedrun approach strategy",
        "Execute the tile observer sequence implementation",
    ]

    total_times = []

    for message in test_messages:
        start_time = time.time()

        # 1. Classify message
        classification = classifier.classify_message(message)

        # 2. Queue message
        queue.enqueue_message(
            message,
            MessagePriority.NORMAL,
            context={"classification": classification.message_type.value},
        )

        # 3. Dequeue message
        queued_message = queue.dequeue_message()

        # 4. Route to worker
        selected_worker = strategy.select_worker(queued_message, mock_workers)

        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        total_times.append(total_time)

        print(
            f"Message: '{message[:40]}...' -> {classification.message_type.value} -> {selected_worker} ({total_time:.2f}ms)"
        )

    avg_total_time = sum(total_times) / len(total_times)
    max_time = max(total_times)

    print(f"Average end-to-end time: {avg_total_time:.2f}ms")
    print(f"Max end-to-end time: {max_time:.2f}ms")

    # Should meet <50ms requirement
    assert avg_total_time < 50.0
    assert max_time < 50.0

    queue.shutdown()
    print("✅ End-to-end performance tests passed\n")


def main():
    """Run all validation tests."""
    print("🚀 Message Routing Engine Validation Demo")
    print("=" * 50)

    try:
        test_message_classification()
        test_priority_queue()
        test_routing_strategy()
        test_end_to_end_performance()

        print("🎉 All tests passed! Message routing engine is production-ready.")
        print("Performance targets achieved:")
        print("  ✅ Message classification: <5ms per message")
        print("  ✅ Queue operations: <1ms per operation")
        print("  ✅ Routing decisions: <5ms per decision")
        print("  ✅ End-to-end routing: <50ms total")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
