"""
Conversation state management for Claude CLI processes with turn tracking.

This module implements conversation state tracking following Clean Code principles,
providing turn limits, message history, and context preservation for optimal
Claude CLI communication performance.

Performance targets maintained:
- Turn tracking: <1ms per operation
- State persistence: <10ms
- Memory efficient history management
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .prompts import ProcessType

logger = logging.getLogger(__name__)


class ConversationStatus(Enum):
    """Status of a Claude conversation."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    APPROACHING_LIMIT = "approaching_limit"
    LIMIT_REACHED = "limit_reached"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class MessageExchange:
    """Single message exchange with timing and metadata."""

    message: str
    response: str | None = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float | None = None
    turn_number: int = 0
    error: str | None = None

    def __post_init__(self):
        """Calculate message hash for deduplication."""
        self.message_hash = hash(self.message)


@dataclass
class ConversationMetrics:
    """Performance metrics for conversation tracking."""

    total_turns: int = 0
    total_duration_ms: float = 0.0
    average_response_time_ms: float = 0.0
    failed_messages: int = 0
    context_preservation_events: int = 0

    def update_response_time(self, duration_ms: float):
        """Update average response time with new measurement."""
        if self.total_turns > 0:
            total_time = self.average_response_time_ms * self.total_turns
            self.average_response_time_ms = (total_time + duration_ms) / (self.total_turns + 1)
        else:
            self.average_response_time_ms = duration_ms


class ConversationState:
    """
    High-performance conversation state manager for Claude CLI processes.

    Implements turn limiting, context preservation, and performance tracking
    optimized for Pokemon speedrun learning agent requirements.

    Performance optimizations:
    - Atomic turn counter operations for thread safety
    - Circular buffer for memory-efficient history management
    - Lazy context preservation to minimize latency
    """

    # Turn limits by process type - like frame budgets in real-time rendering
    TURN_LIMITS = {
        ProcessType.OPUS_STRATEGIC: 100,  # Strategic planning needs more turns
        ProcessType.SONNET_TACTICAL: 20,  # Tactical execution is more focused
    }

    # Warning thresholds (% of limit) - like GPU memory warnings
    WARNING_THRESHOLDS = {
        ProcessType.OPUS_STRATEGIC: 0.85,  # Warn at 85 turns
        ProcessType.SONNET_TACTICAL: 0.80,  # Warn at 16 turns
    }

    def __init__(self, process_type: ProcessType, process_id: int):
        """
        Initialize conversation state for optimal performance.

        Args:
            process_type: Type of Claude process (strategic/tactical)
            process_id: Unique identifier for the process
        """
        self.process_type = process_type
        self.process_id = process_id
        self._lock = threading.Lock()  # Thread-safe like GPU context locks

        # Performance-critical state variables
        self._turn_count = 0
        self._max_turns = self.TURN_LIMITS[process_type]
        self._warning_threshold = int(self._max_turns * self.WARNING_THRESHOLDS[process_type])
        self._status = ConversationStatus.INITIALIZING

        # Circular buffer for memory efficiency - like render command buffers
        self._history_size = min(50, self._max_turns)  # Limit memory usage
        self._message_history: list[MessageExchange] = []
        self._history_index = 0

        # Performance metrics
        self.metrics = ConversationMetrics()

        # System prompt tracking
        self._system_prompt_sent = False
        self._initialization_time: float | None = None

        logger.info(
            f"ConversationState initialized for process {process_id} "
            f"({process_type.value}): {self._max_turns} turn limit"
        )

    def initialize_conversation(self, system_prompt: str) -> bool:
        """
        Initialize conversation with system prompt.

        Args:
            system_prompt: Initial system prompt to establish context

        Returns:
            True if initialization successful
        """
        start_time = time.time()

        with self._lock:
            if self._status != ConversationStatus.INITIALIZING:
                logger.warning(f"Process {self.process_id} already initialized")
                return False

            # Record system prompt (turn 0)
            system_exchange = MessageExchange(
                message=system_prompt, turn_number=0, timestamp=start_time
            )
            self._message_history.append(system_exchange)

            self._system_prompt_sent = True
            self._status = ConversationStatus.ACTIVE
            self._initialization_time = time.time() - start_time

            logger.info(
                f"Conversation initialized for process {self.process_id} "
                f"in {self._initialization_time*1000:.1f}ms"
            )

            return True

    def can_send_message(self) -> bool:
        """
        Check if message can be sent without exceeding turn limits.

        Returns:
            True if within turn limits, False if limit reached
        """
        with self._lock:
            return self._status == ConversationStatus.ACTIVE and self._turn_count < self._max_turns

    def record_message_exchange(
        self,
        message: str,
        response: str | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> int:
        """
        Record message exchange with performance tracking.

        Args:
            message: Message sent to Claude
            response: Response received (if any)
            duration_ms: Response time in milliseconds
            error: Error message if exchange failed

        Returns:
            Current turn number after recording
        """
        with self._lock:
            # Increment turn counter (atomic operation)
            self._turn_count += 1
            current_turn = self._turn_count

            # Create exchange record
            exchange = MessageExchange(
                message=message,
                response=response,
                turn_number=current_turn,
                duration_ms=duration_ms,
                error=error,
            )

            # Add to circular history buffer
            if len(self._message_history) < self._history_size:
                self._message_history.append(exchange)
            else:
                # Circular buffer replacement
                self._message_history[self._history_index % self._history_size] = exchange
                self._history_index += 1

            # Update metrics
            self.metrics.total_turns = current_turn
            if duration_ms:
                self.metrics.update_response_time(duration_ms)
                self.metrics.total_duration_ms += duration_ms

            if error:
                self.metrics.failed_messages += 1

            # Update conversation status based on turn count
            self._update_status()

            logger.debug(
                f"Process {self.process_id} turn {current_turn}/{self._max_turns}: "
                f"{len(message)} chars, {duration_ms:.1f}ms response"
                if duration_ms
                else f"{len(message)} chars"
            )

            return current_turn

    def _update_status(self):
        """Update conversation status based on current turn count."""
        if self._turn_count >= self._max_turns:
            self._status = ConversationStatus.LIMIT_REACHED
        elif self._turn_count >= self._warning_threshold:
            self._status = ConversationStatus.APPROACHING_LIMIT
        else:
            self._status = ConversationStatus.ACTIVE

    def get_conversation_context(self, recent_turns: int = 5) -> str:
        """
        Get recent conversation context for context preservation.

        Args:
            recent_turns: Number of recent turns to include

        Returns:
            JSON-formatted context string for Claude CLI
        """
        with self._lock:
            # Get most recent exchanges
            recent_exchanges = (
                self._message_history[-recent_turns:] if self._message_history else []
            )

            context = {
                "conversation_id": f"process_{self.process_id}",
                "turn_count": self._turn_count,
                "max_turns": self._max_turns,
                "process_type": self.process_type.value,
                "recent_exchanges": [
                    {
                        "turn": ex.turn_number,
                        "message": (
                            ex.message[:200] + "..." if len(ex.message) > 200 else ex.message
                        ),
                        "response_preview": (
                            ex.response[:100] + "..."
                            if ex.response and len(ex.response) > 100
                            else ex.response
                        ),
                        "timestamp": ex.timestamp,
                    }
                    for ex in recent_exchanges
                    if ex.response  # Only include completed exchanges
                ],
            }

            return json.dumps(context, indent=2)

    def get_status_summary(self) -> dict[str, Any]:
        """
        Get comprehensive conversation status for monitoring.

        Returns:
            Dictionary with status, metrics, and performance data
        """
        with self._lock:
            return {
                "process_id": self.process_id,
                "process_type": self.process_type.value,
                "status": self._status.value,
                "turn_count": self._turn_count,
                "max_turns": self._max_turns,
                "turns_remaining": self._max_turns - self._turn_count,
                "warning_threshold": self._warning_threshold,
                "approaching_limit": self._turn_count >= self._warning_threshold,
                "limit_reached": self._status == ConversationStatus.LIMIT_REACHED,
                "system_prompt_sent": self._system_prompt_sent,
                "initialization_time_ms": (
                    self._initialization_time * 1000 if self._initialization_time else None
                ),
                "metrics": {
                    "total_turns": self.metrics.total_turns,
                    "total_duration_ms": self.metrics.total_duration_ms,
                    "average_response_time_ms": self.metrics.average_response_time_ms,
                    "failed_messages": self.metrics.failed_messages,
                    "success_rate": (
                        (self.metrics.total_turns - self.metrics.failed_messages)
                        / max(1, self.metrics.total_turns)
                    ),
                },
                "memory_usage": {
                    "history_entries": len(self._message_history),
                    "history_buffer_size": self._history_size,
                    "circular_buffer_index": self._history_index,
                },
            }

    def needs_context_compression(self) -> bool:
        """
        Check if conversation needs context compression for turn limit management.

        Returns:
            True if approaching limit and should compress context
        """
        with self._lock:
            return self._status == ConversationStatus.APPROACHING_LIMIT

    def reset_conversation(self, preserve_metrics: bool = True):
        """
        Reset conversation state for fresh start while optionally preserving metrics.

        Args:
            preserve_metrics: Whether to keep performance metrics across reset
        """
        with self._lock:
            old_metrics = self.metrics if preserve_metrics else ConversationMetrics()

            self._turn_count = 0
            self._status = ConversationStatus.INITIALIZING
            self._system_prompt_sent = False
            self._initialization_time = None
            self._message_history.clear()
            self._history_index = 0

            if preserve_metrics:
                # Increment context preservation event counter
                old_metrics.context_preservation_events += 1
                self.metrics = old_metrics
            else:
                self.metrics = ConversationMetrics()

            logger.info(
                f"Conversation reset for process {self.process_id} "
                f"(metrics {'preserved' if preserve_metrics else 'cleared'})"
            )

    def terminate_conversation(self, reason: str = "Normal termination"):
        """
        Terminate conversation and mark as completed.

        Args:
            reason: Reason for termination
        """
        with self._lock:
            self._status = ConversationStatus.TERMINATED

            logger.info(
                f"Conversation terminated for process {self.process_id}: {reason}. "
                f"Final stats: {self._turn_count}/{self._max_turns} turns, "
                f"{self.metrics.average_response_time_ms:.1f}ms avg response"
            )

    @property
    def turn_count(self) -> int:
        """Get current turn count (thread-safe)."""
        with self._lock:
            return self._turn_count

    @property
    def status(self) -> ConversationStatus:
        """Get current conversation status (thread-safe)."""
        with self._lock:
            return self._status

    @property
    def turns_remaining(self) -> int:
        """Get remaining turns before limit (thread-safe)."""
        with self._lock:
            return self._max_turns - self._turn_count


class ConversationStateManager:
    """
    Centralized manager for conversation states across multiple processes.

    Provides system-wide conversation monitoring and coordination while
    maintaining individual process state isolation for optimal performance.
    """

    def __init__(self):
        """Initialize the conversation state manager."""
        self.states: dict[int, ConversationState] = {}
        self._lock = threading.Lock()

        logger.info("ConversationStateManager initialized")

    def add_conversation(self, conversation_state: ConversationState):
        """
        Add conversation state to management.

        Args:
            conversation_state: ConversationState to manage
        """
        with self._lock:
            self.states[conversation_state.process_id] = conversation_state

        logger.debug(f"Added conversation state for process {conversation_state.process_id}")

    def remove_conversation(self, process_id: int):
        """
        Remove conversation state from management.

        Args:
            process_id: ID of process to remove
        """
        with self._lock:
            removed_state = self.states.pop(process_id, None)

        if removed_state:
            logger.debug(f"Removed conversation state for process {process_id}")

    def get_conversation(self, process_id: int) -> ConversationState | None:
        """
        Get conversation state for specific process.

        Args:
            process_id: ID of the process

        Returns:
            ConversationState if found, None otherwise
        """
        with self._lock:
            return self.states.get(process_id)

    def get_system_summary(self) -> dict[str, Any]:
        """
        Get system-wide conversation status summary.

        Returns:
            Dictionary with aggregated conversation statistics
        """
        with self._lock:
            total_conversations = len(self.states)
            if total_conversations == 0:
                return {"total_conversations": 0, "conversations": []}

            # Aggregate metrics across all conversations
            total_turns = sum(state.turn_count for state in self.states.values())
            active_conversations = sum(
                1 for state in self.states.values() if state.status == ConversationStatus.ACTIVE
            )
            approaching_limit = sum(
                1
                for state in self.states.values()
                if state.status == ConversationStatus.APPROACHING_LIMIT
            )

            conversations_summary = [state.get_status_summary() for state in self.states.values()]

            return {
                "total_conversations": total_conversations,
                "active_conversations": active_conversations,
                "approaching_limit": approaching_limit,
                "total_turns_across_all": total_turns,
                "average_turns_per_conversation": total_turns / total_conversations,
                "conversations": conversations_summary,
            }
