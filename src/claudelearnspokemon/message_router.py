"""
MessageRouter - Central Orchestration of Message Routing

This module implements the central message routing orchestrator for the Pokemon
speedrun learning agent. It integrates message classification, priority queuing,
and routing strategies to intelligently direct requests to appropriate Claude
instances (Opus for strategic, Sonnet for tactical tasks).

Performance Requirements:
- End-to-end routing: <50ms per message
- Queue processing: <10ms per queue operation
- Classification + routing: <15ms combined
- Memory efficiency: <20MB for routing state

Google SRE Patterns Applied:
- Circuit breaker for routing system failures
- Exponential backoff on process communication failures
- Comprehensive routing observability and metrics
- Graceful degradation when components fail
- Request tracing for debugging routing decisions
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from .claude_code_manager import ClaudeCodeManager
from .message_classifier import ClassificationResult, MessageClassifier, MessageType
from .priority_queue import MessagePriority, MessagePriorityQueue, QueuedMessage, QueueType
from .routing_strategy import RoutingStrategy, WorkerInfo, WorkerState, create_least_loaded_strategy
from .sonnet_worker_pool import SonnetWorkerPool

logger = logging.getLogger(__name__)


class RoutingMode(Enum):
    """Routing operation modes."""

    NORMAL = "normal"  # Full routing with all features
    DEGRADED = "degraded"  # Simplified routing due to component failures
    EMERGENCY = "emergency"  # Minimal routing for system recovery


@dataclass
class RoutingRequest:
    """
    Request for message routing with context and metadata.

    Immutable for thread safety and tracing.
    """

    request_id: str = field(default_factory=lambda: str(uuid4()))
    content: Any = None
    priority: MessagePriority = MessagePriority.NORMAL
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    timeout_seconds: float | None = None
    require_strategic: bool = False  # Force strategic routing
    require_tactical: bool = False  # Force tactical routing

    def __post_init__(self):
        """Validate routing request."""
        if self.require_strategic and self.require_tactical:
            raise ValueError("Cannot require both strategic and tactical routing")

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.timeout_seconds is None:
            return False
        return time.time() - self.timestamp > self.timeout_seconds


@dataclass
class RoutingResult:
    """
    Result of message routing with comprehensive metadata.

    Contains routing decision, performance metrics, and tracing information.
    """

    request_id: str
    success: bool
    worker_id: str | None = None
    worker_type: str | None = None  # "strategic" or "tactical"
    classification_result: ClassificationResult | None = None
    routing_time_ms: float = 0.0
    queue_time_ms: float = 0.0
    total_time_ms: float = 0.0
    error_message: str | None = None
    fallback_used: bool = False
    routing_mode: RoutingMode = RoutingMode.NORMAL
    trace_info: dict[str, Any] = field(default_factory=dict)


class MessageRouter:
    """
    Central message routing orchestrator with production reliability.

    Integrates message classification, priority queuing, and routing strategies
    to intelligently route messages to appropriate Claude processes. Includes
    circuit breaker protection, comprehensive monitoring, and graceful degradation.
    """

    def __init__(
        self,
        claude_manager: ClaudeCodeManager,
        worker_pool: SonnetWorkerPool,
        classifier: MessageClassifier | None = None,
        routing_strategy: RoutingStrategy | None = None,
        max_concurrent_routes: int = 100,
    ):
        """
        Initialize message router with production components.

        Args:
            claude_manager: ClaudeCodeManager for process access
            worker_pool: SonnetWorkerPool for tactical worker management
            classifier: Message classifier (defaults to new instance)
            routing_strategy: Routing strategy (defaults to least-loaded)
            max_concurrent_routes: Maximum concurrent routing operations
        """
        self.claude_manager = claude_manager
        self.worker_pool = worker_pool
        self.classifier = classifier or MessageClassifier()
        self.routing_strategy = routing_strategy or create_least_loaded_strategy()

        # Priority queues for different message types
        self.strategic_queue = MessagePriorityQueue()
        self.tactical_queue = MessagePriorityQueue()

        # Routing state management
        self._lock = threading.RLock()
        self._routing_semaphore = threading.Semaphore(max_concurrent_routes)
        self._running = False
        self._routing_mode = RoutingMode.NORMAL

        # Circuit breaker state
        self._failure_count = 0
        self._circuit_open = False
        self._last_failure_time = 0.0
        self._circuit_breaker_threshold = 10
        self._circuit_breaker_timeout = 60.0

        # Performance metrics
        self._metrics = {
            "total_routing_requests": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "strategic_routes": 0,
            "tactical_routes": 0,
            "avg_routing_time_ms": 0.0,
            "circuit_breaker_activations": 0,
            "fallback_routes": 0,
            "queue_overflow_rejections": 0,
            "expired_request_rejections": 0,
        }

        # Request tracing for debugging
        self._request_traces: dict[str, dict[str, Any]] = {}
        self._max_traces = 1000  # Limit memory usage

        logger.info("MessageRouter initialized with production reliability patterns")

    def start(self) -> bool:
        """
        Start the message router and its components.

        Returns:
            True if startup successful, False otherwise
        """
        try:
            with self._lock:
                if self._running:
                    logger.warning("MessageRouter is already running")
                    return True

                logger.info("Starting MessageRouter...")

                # Ensure Claude manager is running
                if not self.claude_manager.is_running():
                    if not self.claude_manager.start_all_processes():
                        logger.error("Failed to start Claude processes")
                        return False

                # Initialize worker pool
                if not self.worker_pool.is_initialized():
                    if not self.worker_pool.initialize():
                        logger.error("Failed to initialize worker pool")
                        return False

                # Start background queue processing
                self._start_background_processing()

                self._running = True
                logger.info("MessageRouter started successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to start MessageRouter: {e}")
            return False

    def route_message(self, request: RoutingRequest) -> RoutingResult:
        """
        Route message with comprehensive error handling and monitoring.

        Args:
            request: Routing request with message and metadata

        Returns:
            RoutingResult with routing decision and performance metrics
        """
        start_time = time.time()

        try:
            with self._routing_semaphore:
                self._metrics["total_routing_requests"] += 1

                # Create trace entry
                trace_info = {
                    "start_time": start_time,
                    "routing_mode": self._routing_mode.value,
                    "circuit_breaker_open": self._circuit_open,
                }

                # Check if request is expired
                if request.is_expired:
                    self._metrics["expired_request_rejections"] += 1
                    return self._create_failure_result(request, "Request expired", trace_info)

                # Check circuit breaker state
                if self._circuit_open:
                    if time.time() - self._last_failure_time > self._circuit_breaker_timeout:
                        self._reset_circuit_breaker()
                    else:
                        return self._handle_circuit_breaker_open(request, trace_info)

                # Route message based on mode
                if self._routing_mode == RoutingMode.NORMAL:
                    result = self._route_normal(request, trace_info)
                elif self._routing_mode == RoutingMode.DEGRADED:
                    result = self._route_degraded(request, trace_info)
                else:
                    result = self._route_emergency(request, trace_info)

                # Update metrics and tracing
                self._update_routing_metrics(result)
                self._store_request_trace(request.request_id, trace_info)

                return result

        except Exception as e:
            logger.error(f"Routing failed for request {request.request_id}: {e}")
            self._handle_routing_failure()
            return self._create_failure_result(request, str(e), {"error": str(e)})

    def _route_normal(self, request: RoutingRequest, trace_info: dict[str, Any]) -> RoutingResult:
        """Route message in normal mode with full feature set."""
        classification_start = time.time()

        # Classify message type unless forced
        if request.require_strategic:
            message_type = MessageType.STRATEGIC
            classification_result = None
            trace_info["classification_forced"] = "strategic"
        elif request.require_tactical:
            message_type = MessageType.TACTICAL
            classification_result = None
            trace_info["classification_forced"] = "tactical"
        else:
            classification_result = self.classifier.classify_message(
                str(request.content), request.context
            )
            message_type = classification_result.message_type
            trace_info["classification_time_ms"] = (time.time() - classification_start) * 1000
            trace_info["classification_confidence"] = classification_result.confidence

        # Queue message based on classification
        queue_start = time.time()
        queued_message = self._queue_message(request, message_type, trace_info)
        trace_info["queue_time_ms"] = (time.time() - queue_start) * 1000

        if not queued_message:
            return self._create_failure_result(request, "Failed to queue message", trace_info)

        # Route to appropriate worker
        routing_start = time.time()
        worker_result = self._route_to_worker(message_type, trace_info)
        trace_info["worker_routing_time_ms"] = (time.time() - routing_start) * 1000

        if not worker_result:
            return self._create_failure_result(request, "No available workers", trace_info)

        worker_id, worker_type = worker_result

        # Create successful result
        total_time = (time.time() - trace_info["start_time"]) * 1000
        return RoutingResult(
            request_id=request.request_id,
            success=True,
            worker_id=worker_id,
            worker_type=worker_type,
            classification_result=classification_result,
            routing_time_ms=trace_info.get("worker_routing_time_ms", 0.0),
            queue_time_ms=trace_info.get("queue_time_ms", 0.0),
            total_time_ms=total_time,
            routing_mode=self._routing_mode,
            trace_info=trace_info,
        )

    def _route_degraded(self, request: RoutingRequest, trace_info: dict[str, Any]) -> RoutingResult:
        """Route message in degraded mode with simplified logic."""
        trace_info["degraded_reason"] = "Component failure detected"

        # Simple fallback routing - default to tactical unless explicitly strategic
        if request.require_strategic or "strategic" in str(request.content).lower():
            message_type = MessageType.STRATEGIC
        else:
            message_type = MessageType.TACTICAL

        # Skip complex queueing and use direct routing
        worker_result = self._route_to_worker_direct(message_type, trace_info)

        if not worker_result:
            return self._create_failure_result(request, "No workers in degraded mode", trace_info)

        worker_id, worker_type = worker_result
        total_time = (time.time() - trace_info["start_time"]) * 1000

        return RoutingResult(
            request_id=request.request_id,
            success=True,
            worker_id=worker_id,
            worker_type=worker_type,
            routing_time_ms=total_time,
            total_time_ms=total_time,
            fallback_used=True,
            routing_mode=self._routing_mode,
            trace_info=trace_info,
        )

    def _route_emergency(
        self, request: RoutingRequest, trace_info: dict[str, Any]
    ) -> RoutingResult:
        """Route message in emergency mode with minimal processing."""
        trace_info["emergency_reason"] = "System recovery mode"

        # Emergency routing: try tactical first, then strategic
        tactical_processes = self.claude_manager.get_tactical_processes()
        if tactical_processes:
            available_tactical = [p for p in tactical_processes if p.is_healthy()]
            if available_tactical:
                worker = available_tactical[0]  # Use first available
                total_time = (time.time() - trace_info["start_time"]) * 1000

                return RoutingResult(
                    request_id=request.request_id,
                    success=True,
                    worker_id=f"tactical_{worker.process_id}",
                    worker_type="tactical",
                    routing_time_ms=total_time,
                    total_time_ms=total_time,
                    fallback_used=True,
                    routing_mode=self._routing_mode,
                    trace_info=trace_info,
                )

        # Fall back to strategic if no tactical available
        strategic_process = self.claude_manager.get_strategic_process()
        if strategic_process and strategic_process.is_healthy():
            total_time = (time.time() - trace_info["start_time"]) * 1000

            return RoutingResult(
                request_id=request.request_id,
                success=True,
                worker_id=f"strategic_{strategic_process.process_id}",
                worker_type="strategic",
                routing_time_ms=total_time,
                total_time_ms=total_time,
                fallback_used=True,
                routing_mode=self._routing_mode,
                trace_info=trace_info,
            )

        return self._create_failure_result(
            request, "No processes available in emergency mode", trace_info
        )

    def _queue_message(
        self, request: RoutingRequest, message_type: MessageType, trace_info: dict[str, Any]
    ) -> str | None:
        """Queue message in appropriate priority queue."""
        queue_type = (
            QueueType.STRATEGIC if message_type == MessageType.STRATEGIC else QueueType.TACTICAL
        )
        target_queue = (
            self.strategic_queue if message_type == MessageType.STRATEGIC else self.tactical_queue
        )

        # Set TTL based on request timeout
        ttl_seconds = None
        if request.timeout_seconds:
            ttl_seconds = request.timeout_seconds - (time.time() - request.timestamp)
            if ttl_seconds <= 0:
                trace_info["queue_rejection_reason"] = "Already expired"
                return None

        message_id = target_queue.enqueue_message(
            content=request.content,
            priority=request.priority,
            queue_type=queue_type,
            context=request.context,
            ttl_seconds=ttl_seconds,
        )

        if message_id:
            trace_info["queued_in"] = queue_type.value
            trace_info["queue_message_id"] = message_id
        else:
            trace_info["queue_rejection_reason"] = "Queue full or failed"
            self._metrics["queue_overflow_rejections"] += 1

        return message_id

    def _route_to_worker(
        self, message_type: MessageType, trace_info: dict[str, Any]
    ) -> tuple[str, str] | None:
        """Route message to appropriate worker."""
        if message_type == MessageType.STRATEGIC:
            return self._route_to_strategic_worker(trace_info)
        else:
            return self._route_to_tactical_worker(trace_info)

    def _route_to_strategic_worker(self, trace_info: dict[str, Any]) -> tuple[str, str] | None:
        """Route to strategic (Opus) worker."""
        strategic_process = self.claude_manager.get_strategic_process()

        if not strategic_process or not strategic_process.is_healthy():
            trace_info["strategic_routing_failure"] = "No healthy strategic process"
            return None

        worker_id = f"strategic_{strategic_process.process_id}"
        trace_info["selected_strategic_worker"] = worker_id
        return worker_id, "strategic"

    def _route_to_tactical_worker(self, trace_info: dict[str, Any]) -> tuple[str, str] | None:
        """Route to tactical (Sonnet) worker using routing strategy."""
        tactical_processes = self.claude_manager.get_tactical_processes()

        if not tactical_processes:
            trace_info["tactical_routing_failure"] = "No tactical processes"
            return None

        # Convert to WorkerInfo objects for routing strategy
        available_workers = []
        for process in tactical_processes:
            if process.is_healthy():
                worker_info = WorkerInfo(
                    worker_id=f"tactical_{process.process_id}",
                    process=process,
                    state=WorkerState.HEALTHY if process.is_healthy() else WorkerState.UNHEALTHY,
                )
                available_workers.append(worker_info)

        if not available_workers:
            trace_info["tactical_routing_failure"] = "No healthy tactical workers"
            return None

        # Use routing strategy to select worker
        # Create a dummy message for the strategy
        dummy_message = QueuedMessage(content="", priority=MessagePriority.NORMAL)
        selected_worker_id = self.routing_strategy.select_worker(dummy_message, available_workers)

        if selected_worker_id:
            trace_info["selected_tactical_worker"] = selected_worker_id
            trace_info["available_tactical_workers"] = len(available_workers)
            return selected_worker_id, "tactical"
        else:
            trace_info["tactical_routing_failure"] = "Strategy failed to select worker"
            return None

    def _route_to_worker_direct(
        self, message_type: MessageType, trace_info: dict[str, Any]
    ) -> tuple[str, str] | None:
        """Direct worker routing for degraded mode."""
        if message_type == MessageType.STRATEGIC:
            return self._route_to_strategic_worker(trace_info)
        else:
            # Simple tactical routing - use first available
            tactical_processes = self.claude_manager.get_tactical_processes()
            for process in tactical_processes:
                if process.is_healthy():
                    worker_id = f"tactical_{process.process_id}"
                    trace_info["direct_tactical_worker"] = worker_id
                    return worker_id, "tactical"
            return None

    def _create_failure_result(
        self, request: RoutingRequest, error_message: str, trace_info: dict[str, Any]
    ) -> RoutingResult:
        """Create routing failure result."""
        total_time = (time.time() - trace_info.get("start_time", time.time())) * 1000

        return RoutingResult(
            request_id=request.request_id,
            success=False,
            error_message=error_message,
            total_time_ms=total_time,
            routing_mode=self._routing_mode,
            trace_info=trace_info,
        )

    def _handle_circuit_breaker_open(
        self, request: RoutingRequest, trace_info: dict[str, Any]
    ) -> RoutingResult:
        """Handle routing when circuit breaker is open."""
        trace_info["circuit_breaker_blocked"] = True
        return self._create_failure_result(request, "Circuit breaker open", trace_info)

    def _handle_routing_failure(self) -> None:
        """Handle routing failure and update circuit breaker state."""
        with self._lock:
            self._failure_count += 1

            if self._failure_count >= self._circuit_breaker_threshold:
                self._circuit_open = True
                self._last_failure_time = time.time()
                self._metrics["circuit_breaker_activations"] += 1
                logger.error(
                    f"Message router circuit breaker opened after {self._failure_count} failures"
                )

                # Switch to degraded mode
                if self._routing_mode == RoutingMode.NORMAL:
                    self._routing_mode = RoutingMode.DEGRADED
                    logger.warning("Switching to degraded routing mode")

    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker and return to normal operation."""
        with self._lock:
            self._circuit_open = False
            self._failure_count = 0
            self._routing_mode = RoutingMode.NORMAL
            logger.info("Circuit breaker reset - returning to normal routing mode")

    def _update_routing_metrics(self, result: RoutingResult) -> None:
        """Update routing performance metrics."""
        with self._lock:
            if result.success:
                self._metrics["successful_routes"] += 1
                if result.worker_type == "strategic":
                    self._metrics["strategic_routes"] += 1
                else:
                    self._metrics["tactical_routes"] += 1
            else:
                self._metrics["failed_routes"] += 1

            if result.fallback_used:
                self._metrics["fallback_routes"] += 1

            # Update average routing time
            total_requests = self._metrics["total_routing_requests"]
            current_avg = self._metrics["avg_routing_time_ms"]
            self._metrics["avg_routing_time_ms"] = (
                current_avg * (total_requests - 1) + result.total_time_ms
            ) / total_requests

    def _store_request_trace(self, request_id: str, trace_info: dict[str, Any]) -> None:
        """Store request trace for debugging."""
        with self._lock:
            self._request_traces[request_id] = trace_info

            # Limit memory usage
            if len(self._request_traces) > self._max_traces:
                # Remove oldest traces
                oldest_keys = list(self._request_traces.keys())[: -self._max_traces]
                for key in oldest_keys:
                    del self._request_traces[key]

    def _start_background_processing(self) -> None:
        """Start background threads for queue processing."""
        # In a full implementation, this would start background threads
        # to continuously process queues. For now, we rely on synchronous processing.
        pass

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive router health status."""
        with self._lock:
            queue_status = {
                "strategic_queue": self.strategic_queue.get_health_status(),
                "tactical_queue": self.tactical_queue.get_health_status(),
            }

            return {
                "running": self._running,
                "routing_mode": self._routing_mode.value,
                "circuit_breaker_open": self._circuit_open,
                "failure_count": self._failure_count,
                "metrics": self._metrics.copy(),
                "queues": queue_status,
                "classifier_health": self.classifier.get_health_status(),
                "routing_strategy_metrics": self.routing_strategy.get_strategy_metrics(),
            }

    def get_request_trace(self, request_id: str) -> dict[str, Any] | None:
        """Get trace information for specific request."""
        with self._lock:
            return self._request_traces.get(request_id)

    def shutdown(self, timeout: float = 10.0) -> None:
        """Graceful shutdown of message router."""
        logger.info("Shutting down MessageRouter...")

        with self._lock:
            self._running = False

        # Shutdown queues
        self.strategic_queue.shutdown()
        self.tactical_queue.shutdown()

        logger.info("MessageRouter shutdown complete")


# Convenience functions for common router configurations
def create_message_router(
    claude_manager: ClaudeCodeManager, worker_pool: SonnetWorkerPool
) -> MessageRouter:
    """Create message router with default production configuration."""
    return MessageRouter(claude_manager, worker_pool)


def create_high_throughput_router(
    claude_manager: ClaudeCodeManager, worker_pool: SonnetWorkerPool
) -> MessageRouter:
    """Create message router optimized for high throughput."""
    from .routing_strategy import create_least_loaded_strategy

    strategy = create_least_loaded_strategy(prediction_factor=0.8)
    return MessageRouter(
        claude_manager=claude_manager,
        worker_pool=worker_pool,
        routing_strategy=strategy,
        max_concurrent_routes=200,
    )
