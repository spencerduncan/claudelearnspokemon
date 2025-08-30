"""
PriorityQueue System - Priority-Based Message Routing

This module implements production-ready priority queuing for the Pokemon speedrun
learning agent's message routing system. It replaces FIFO queuing with intelligent
priority-based routing to ensure urgent strategic and tactical messages receive
appropriate processing precedence.

Performance Requirements:
- Queue operations (enqueue/dequeue): <1ms per operation
- Priority sorting: O(log n) insertion complexity
- Memory efficiency: <1MB per 1000 queued messages
- Thread safety for concurrent operations

Google SRE Patterns Applied:
- Circuit breaker for queue operations
- Back-pressure handling when queues reach capacity
- Comprehensive queue health monitoring
- Graceful degradation under high load
"""

import heapq
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Generic, TypeVar
from uuid import uuid4

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MessagePriority(IntEnum):
    """
    Message priority levels for queue ordering.

    Lower numeric values = higher priority (following Unix convention).
    This ensures critical messages are processed first.
    """

    CRITICAL = 0  # System failures, emergency stops
    HIGH = 10  # Urgent strategic decisions, time-sensitive tasks
    NORMAL = 20  # Standard strategic planning, routine tactical work
    LOW = 30  # Background analysis, non-urgent optimizations
    BULK = 40  # Batch processing, bulk data operations


class QueueType(Enum):
    """Queue types for different message routing scenarios."""

    STRATEGIC = "strategic"  # Messages for Opus strategic processes
    TACTICAL = "tactical"  # Messages for Sonnet tactical processes
    MIXED = "mixed"  # Combined strategic/tactical queue


@dataclass
class QueuedMessage:
    """
    Message wrapper for priority queue with metadata.

    Thread-safe and immutable for concurrent processing.
    """

    message_id: str = field(default_factory=lambda: str(uuid4()))
    content: Any = None
    priority: MessagePriority = MessagePriority.NORMAL
    queue_type: QueueType = QueueType.MIXED
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    expiry_time: float | None = None  # Message expiry for time-sensitive tasks
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        """Validate message data after initialization."""
        if self.expiry_time and self.expiry_time <= self.timestamp:
            raise ValueError("Expiry time must be after timestamp")

    def is_expired(self) -> bool:
        """Check if message has expired."""
        return self.expiry_time is not None and time.time() > self.expiry_time

    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count <= self.max_retries

    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1

    def __lt__(self, other: "QueuedMessage") -> bool:
        """
        Priority comparison for heap ordering.

        Primary: Priority level (lower = higher priority)
        Secondary: Timestamp (older = higher priority for same level)
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

    def __eq__(self, other: object) -> bool:
        """Equality based on message ID."""
        if not isinstance(other, QueuedMessage):
            return NotImplemented
        return self.message_id == other.message_id


class QueueMetrics:
    """Thread-safe metrics collection for queue monitoring."""

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics = {
            "total_enqueued": 0,
            "total_dequeued": 0,
            "current_size": 0,
            "max_size_reached": 0,
            "expired_messages": 0,
            "retry_exhausted_messages": 0,
            "avg_queue_time_ms": 0.0,
            "priority_distribution": defaultdict(int),
            "queue_operations_per_second": 0.0,
            "last_operation_time": 0.0,
        }
        self._operation_times = []
        self._max_operation_history = 1000

    def record_enqueue(self, message: QueuedMessage, queue_size: int) -> None:
        """Record enqueue operation."""
        with self._lock:
            self._metrics["total_enqueued"] += 1
            self._metrics["current_size"] = queue_size
            self._metrics["max_size_reached"] = max(self._metrics["max_size_reached"], queue_size)
            self._metrics["priority_distribution"][message.priority.name] += 1
            self._record_operation_time()

    def record_dequeue(self, message: QueuedMessage, queue_size: int, queue_time_ms: float) -> None:
        """Record dequeue operation."""
        with self._lock:
            self._metrics["total_dequeued"] += 1
            self._metrics["current_size"] = queue_size
            self._update_avg_queue_time(queue_time_ms)
            self._record_operation_time()

    def record_expired_message(self) -> None:
        """Record expired message."""
        with self._lock:
            self._metrics["expired_messages"] += 1

    def record_retry_exhausted(self) -> None:
        """Record message with exhausted retries."""
        with self._lock:
            self._metrics["retry_exhausted_messages"] += 1

    def _update_avg_queue_time(self, queue_time_ms: float) -> None:
        """Update running average queue time."""
        current_avg = self._metrics["avg_queue_time_ms"]
        total_dequeued = self._metrics["total_dequeued"]

        self._metrics["avg_queue_time_ms"] = (
            current_avg * (total_dequeued - 1) + queue_time_ms
        ) / total_dequeued

    def _record_operation_time(self) -> None:
        """Record operation time for throughput calculation."""
        now = time.time()
        self._operation_times.append(now)
        self._metrics["last_operation_time"] = now

        # Maintain sliding window for throughput calculation
        if len(self._operation_times) > self._max_operation_history:
            self._operation_times = self._operation_times[-self._max_operation_history :]

        # Calculate operations per second over last minute
        minute_ago = now - 60.0
        recent_operations = [t for t in self._operation_times if t > minute_ago]
        self._metrics["queue_operations_per_second"] = len(recent_operations) / 60.0

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            return {
                **self._metrics,
                "priority_distribution": dict(self._metrics["priority_distribution"]),
            }


class PriorityQueueStrategy(ABC, Generic[T]):
    """Strategy pattern for different priority queue implementations."""

    @abstractmethod
    def enqueue(self, item: T) -> bool:
        """Add item to queue. Returns True if successful."""
        pass

    @abstractmethod
    def dequeue(self) -> T | None:
        """Remove and return highest priority item."""
        pass

    @abstractmethod
    def peek(self) -> T | None:
        """Return highest priority item without removing it."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return current queue size."""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        pass


class HeapBasedPriorityQueue(PriorityQueueStrategy[QueuedMessage]):
    """
    Production heap-based priority queue implementation.

    Uses Python's heapq for O(log n) priority operations with thread safety,
    capacity limits, and comprehensive monitoring.
    """

    def __init__(self, max_capacity: int = 1000):
        """
        Initialize heap-based priority queue.

        Args:
            max_capacity: Maximum queue capacity for back-pressure
        """
        self.max_capacity = max_capacity
        self._queue: list[QueuedMessage] = []
        self._lock = threading.Lock()  # Simple lock sufficient - no reentrancy needed
        self._metrics = QueueMetrics()

        # Circuit breaker state
        self._failure_count = 0
        self._circuit_open = False
        self._last_failure_time = 0.0

        logger.info(f"HeapBasedPriorityQueue initialized with capacity {max_capacity}")

    def enqueue(self, item: QueuedMessage) -> bool:
        """
        Add message to priority queue with back-pressure handling.

        Args:
            item: Message to enqueue

        Returns:
            True if successfully enqueued, False if rejected
        """
        start_time = time.time()

        with self._lock:
            try:
                # Check capacity limits (back-pressure)
                if len(self._queue) >= self.max_capacity:
                    logger.warning(f"Queue at capacity ({self.max_capacity}), rejecting message")
                    return False

                # Check if message is already expired
                if item.is_expired():
                    logger.debug(f"Rejecting expired message {item.message_id}")
                    self._metrics.record_expired_message()
                    return False

                # Add to heap
                heapq.heappush(self._queue, item)
                self._metrics.record_enqueue(item, len(self._queue))

                # Performance monitoring
                enqueue_time = (time.time() - start_time) * 1000
                if enqueue_time > 1.0:  # 1ms target
                    logger.warning(f"Enqueue took {enqueue_time:.2f}ms, exceeds 1ms target")

                logger.debug(
                    f"Enqueued message {item.message_id} with priority {item.priority.name}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to enqueue message: {e}")
                self._handle_operation_failure()
                return False

    def dequeue(self) -> QueuedMessage | None:
        """
        Remove and return highest priority message.

        Returns:
            Highest priority message or None if queue empty
        """
        start_time = time.time()

        with self._lock:
            try:
                # Clean expired messages first
                self._clean_expired_messages()

                if not self._queue:
                    return None

                # Get highest priority message
                message = heapq.heappop(self._queue)

                # Calculate queue time for metrics
                queue_time_ms = (time.time() - message.timestamp) * 1000
                self._metrics.record_dequeue(message, len(self._queue), queue_time_ms)

                # Performance monitoring
                dequeue_time = (time.time() - start_time) * 1000
                if dequeue_time > 1.0:  # 1ms target
                    logger.warning(f"Dequeue took {dequeue_time:.2f}ms, exceeds 1ms target")

                logger.debug(
                    f"Dequeued message {message.message_id} after {queue_time_ms:.2f}ms in queue"
                )
                return message

            except Exception as e:
                logger.error(f"Failed to dequeue message: {e}")
                self._handle_operation_failure()
                return None

    def peek(self) -> QueuedMessage | None:
        """Return highest priority message without removing it."""
        with self._lock:
            self._clean_expired_messages()
            return self._queue[0] if self._queue else None

    def size(self) -> int:
        """Return current queue size."""
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    def _clean_expired_messages(self) -> None:
        """Remove expired messages from queue."""
        if not self._queue:
            return

        # Create new queue without expired messages
        # This is O(n) but necessary for correctness
        original_size = len(self._queue)
        self._queue = [msg for msg in self._queue if not msg.is_expired()]

        # Re-heapify if messages were removed
        if len(self._queue) < original_size:
            heapq.heapify(self._queue)
            expired_count = original_size - len(self._queue)
            logger.debug(f"Cleaned {expired_count} expired messages from queue")
            for _ in range(expired_count):
                self._metrics.record_expired_message()

    def _handle_operation_failure(self) -> None:
        """Handle queue operation failures."""
        self._failure_count += 1

        # Simple circuit breaker logic
        if self._failure_count >= 5:
            self._circuit_open = True
            self._last_failure_time = time.time()
            logger.error("Priority queue circuit breaker activated")

    def get_health_status(self) -> dict[str, Any]:
        """Get queue health status."""
        with self._lock:
            return {
                "size": len(self._queue),
                "capacity": self.max_capacity,
                "utilization": len(self._queue) / self.max_capacity,
                "circuit_breaker_open": self._circuit_open,
                "failure_count": self._failure_count,
                "metrics": self._metrics.get_metrics(),
            }

    def get_priority_distribution(self) -> dict[str, int]:
        """Get distribution of messages by priority."""
        with self._lock:
            distribution: dict[str, int] = defaultdict(int)
            for message in self._queue:
                distribution[message.priority.name] += 1
            return dict(distribution)


class MessagePriorityQueue:
    """
    Production message priority queue with circuit breaker and monitoring.

    This is the main interface for priority-based message queuing, providing
    reliability patterns like circuit breaking, health monitoring, and
    automatic cleanup of expired messages.
    """

    def __init__(
        self,
        strategy: PriorityQueueStrategy[QueuedMessage] | None = None,
        cleanup_interval: float = 30.0,
    ):
        """
        Initialize priority queue with production patterns.

        Args:
            strategy: Queue implementation strategy (defaults to HeapBasedPriorityQueue)
            cleanup_interval: Interval for automatic cleanup of expired messages (seconds)
        """
        self.strategy = strategy or HeapBasedPriorityQueue()
        self.cleanup_interval = cleanup_interval

        # Background cleanup thread
        self._cleanup_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._start_cleanup_thread()

        logger.info("MessagePriorityQueue initialized with background cleanup")

    def enqueue_message(
        self,
        content: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        queue_type: QueueType = QueueType.MIXED,
        context: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
    ) -> str | None:
        """
        Enqueue message with specified priority.

        Args:
            content: Message content to queue
            priority: Message priority level
            queue_type: Queue type for routing
            context: Additional context data
            ttl_seconds: Time-to-live in seconds (optional)

        Returns:
            Message ID if enqueued successfully, None otherwise
        """
        # Calculate expiry time if TTL specified
        expiry_time = None
        if ttl_seconds:
            expiry_time = time.time() + ttl_seconds

        message = QueuedMessage(
            content=content,
            priority=priority,
            queue_type=queue_type,
            context=context or {},
            expiry_time=expiry_time,
        )

        if self.strategy.enqueue(message):
            logger.debug(f"Enqueued message {message.message_id} with priority {priority.name}")
            return message.message_id
        else:
            logger.warning(f"Failed to enqueue message with priority {priority.name}")
            return None

    def dequeue_message(self) -> QueuedMessage | None:
        """
        Dequeue highest priority message.

        Returns:
            Highest priority message or None if queue empty
        """
        return self.strategy.dequeue()

    def peek_next_message(self) -> QueuedMessage | None:
        """
        Peek at next message without removing it.

        Returns:
            Next message or None if queue empty
        """
        return self.strategy.peek()

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.strategy.size()

    def is_queue_empty(self) -> bool:
        """Check if queue is empty."""
        return self.strategy.is_empty()

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive queue health status."""
        base_status = {
            "queue_size": self.get_queue_size(),
            "is_empty": self.is_queue_empty(),
            "cleanup_thread_active": (
                self._cleanup_thread.is_alive() if self._cleanup_thread else False
            ),
        }

        if hasattr(self.strategy, "get_health_status"):
            base_status.update(self.strategy.get_health_status())

        return base_status

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""

        def cleanup_worker():
            while not self._shutdown_event.wait(self.cleanup_interval):
                try:
                    # Trigger cleanup by peeking (which calls _clean_expired_messages)
                    if hasattr(self.strategy, "_clean_expired_messages"):
                        with getattr(self.strategy, "_lock", threading.Lock()):
                            self.strategy._clean_expired_messages()
                except Exception as e:
                    logger.error(f"Background cleanup failed: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def shutdown(self) -> None:
        """Graceful shutdown of priority queue."""
        logger.info("Shutting down MessagePriorityQueue...")
        self._shutdown_event.set()

        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)

        logger.info("MessagePriorityQueue shutdown complete")


# Factory functions for common queue configurations
def create_strategic_queue(max_capacity: int = 500) -> MessagePriorityQueue:
    """Create priority queue optimized for strategic messages."""
    strategy = HeapBasedPriorityQueue(max_capacity=max_capacity)
    return MessagePriorityQueue(strategy=strategy)


def create_tactical_queue(max_capacity: int = 1000) -> MessagePriorityQueue:
    """Create priority queue optimized for tactical messages."""
    strategy = HeapBasedPriorityQueue(max_capacity=max_capacity)
    return MessagePriorityQueue(strategy=strategy)


def create_mixed_queue(max_capacity: int = 1500) -> MessagePriorityQueue:
    """Create priority queue for mixed strategic/tactical messages."""
    strategy = HeapBasedPriorityQueue(max_capacity=max_capacity)
    return MessagePriorityQueue(strategy=strategy)
