"""
Input buffering and batching strategies for high-frequency input optimization.

Implements intelligent buffering to reduce HTTP overhead when multiple inputs
are sent in rapid succession, while maintaining input order correctness.

Following Uncle Bob's Clean Code principles:
- Single Responsibility: InputBuffer only handles buffering logic
- Open/Closed: Extensible buffering strategies
- Clean abstractions with clear interfaces

Author: Uncle Bot - Performance Optimization Specialist
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class BufferedInput:
    """Represents a single buffered input with metadata."""

    sequence: str
    timestamp: float
    future: asyncio.Future[dict[str, Any]]


@dataclass
class BufferConfig:
    """Configuration for input buffering behavior."""

    max_wait_ms: float = 5.0  # Maximum time to wait before flushing buffer
    max_batch_size: int = 10  # Maximum inputs per batch
    min_batch_size: int = 2  # Minimum inputs to trigger batching
    high_frequency_threshold: int = 5  # Inputs per second to trigger buffering


class InputBuffer:
    """
    Intelligent input buffering for high-frequency scenarios.

    Collects rapid successive inputs and batches them for optimal performance
    while maintaining correctness and reasonable latency bounds.
    """

    def __init__(
        self,
        batch_sender: Callable[[list[str]], Awaitable[list[dict[str, Any]]]],
        config: BufferConfig | None = None,
    ):
        """
        Initialize input buffer with batch sending capability.

        Args:
            batch_sender: Async function to send batched inputs
            config: Buffer configuration (uses defaults if None)
        """
        self.batch_sender = batch_sender
        self.config = config or BufferConfig()

        # Buffer state
        self.buffer: deque[BufferedInput] = deque()
        self.flush_task: asyncio.Task | None = None
        self.input_timestamps: deque[float] = deque(maxlen=20)  # Track frequency

        # Statistics for monitoring
        self.stats = {
            "total_inputs": 0,
            "batched_inputs": 0,
            "buffer_flushes": 0,
            "avg_batch_size": 0.0,
        }

    async def add_input(self, input_sequence: str) -> dict[str, Any]:
        """
        Add input to buffer with intelligent batching strategy.

        Args:
            input_sequence: Input sequence to buffer

        Returns:
            Response data from processing the input
        """
        self.stats["total_inputs"] += 1
        current_time = time.time()
        self.input_timestamps.append(current_time)

        # Create future for this input's result
        future: asyncio.Future[dict[str, Any]] = asyncio.Future()
        buffered_input = BufferedInput(input_sequence, current_time, future)

        # Check if we should use buffering based on input frequency
        if self._should_buffer():
            # Add to buffer and potentially trigger flush
            self.buffer.append(buffered_input)
            await self._maybe_flush_buffer()
        else:
            # Send immediately for low-frequency inputs
            try:
                results = await self.batch_sender([input_sequence])
                future.set_result(results[0] if results else {"status": "error"})
            except Exception as e:
                future.set_exception(e)

        return await future

    def _should_buffer(self) -> bool:
        """
        Determine if input should be buffered based on frequency analysis.

        Returns:
            True if input should be buffered, False for immediate processing
        """
        if len(self.input_timestamps) < 2:
            return False

        # Calculate recent input frequency (inputs per second)
        recent_inputs = [ts for ts in self.input_timestamps if time.time() - ts <= 1.0]

        frequency = len(recent_inputs)
        return frequency >= self.config.high_frequency_threshold

    async def _maybe_flush_buffer(self) -> None:
        """
        Conditionally flush buffer based on size and timing constraints.
        """
        if not self.buffer:
            return

        should_flush_size = len(self.buffer) >= self.config.max_batch_size
        should_flush_time = self._should_flush_by_time()

        if should_flush_size:
            # Immediate flush for full buffer
            await self._flush_buffer()
        elif should_flush_time and len(self.buffer) >= self.config.min_batch_size:
            # Time-based flush with minimum batch size
            await self._flush_buffer()
        elif not self.flush_task and self.buffer:
            # Schedule delayed flush
            self.flush_task = asyncio.create_task(self._delayed_flush())

    def _should_flush_by_time(self) -> bool:
        """
        Check if buffer should be flushed based on time constraints.

        Returns:
            True if oldest input has exceeded max wait time
        """
        if not self.buffer:
            return False

        oldest_time = self.buffer[0].timestamp
        wait_time_ms = (time.time() - oldest_time) * 1000
        return wait_time_ms >= self.config.max_wait_ms

    async def _delayed_flush(self) -> None:
        """
        Delayed flush after max wait time to ensure reasonable latency.
        """
        try:
            # Wait for max wait time
            await asyncio.sleep(self.config.max_wait_ms / 1000.0)

            # Flush if buffer still has inputs
            if self.buffer:
                await self._flush_buffer()
        finally:
            self.flush_task = None

    async def _flush_buffer(self) -> None:
        """
        Flush current buffer contents as a batch.
        """
        if not self.buffer:
            return

        # Extract inputs and futures from buffer
        inputs_to_send = []
        futures_to_resolve = []

        while self.buffer:
            buffered_input = self.buffer.popleft()
            inputs_to_send.append(buffered_input.sequence)
            futures_to_resolve.append(buffered_input.future)

        # Cancel delayed flush task if running
        if self.flush_task:
            self.flush_task.cancel()
            self.flush_task = None

        # Send batch and resolve futures
        try:
            results = await self.batch_sender(inputs_to_send)

            # Resolve each future with its corresponding result
            for i, future in enumerate(futures_to_resolve):
                if not future.done():
                    result = results[i] if i < len(results) else {"status": "error"}
                    future.set_result(result)

            # Update statistics
            self._update_stats(len(inputs_to_send))

        except Exception as e:
            # Resolve all futures with the exception
            for future in futures_to_resolve:
                if not future.done():
                    future.set_exception(e)

    def _update_stats(self, batch_size: int) -> None:
        """Update buffer statistics for monitoring."""
        self.stats["buffer_flushes"] += 1
        self.stats["batched_inputs"] += batch_size

        # Calculate running average batch size
        total_flushes = self.stats["buffer_flushes"]
        if total_flushes > 0:
            self.stats["avg_batch_size"] = self.stats["batched_inputs"] / total_flushes

    def get_stats(self) -> dict[str, Any]:
        """
        Get buffer performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        return {
            **self.stats,
            "buffer_size": len(self.buffer),
            "buffering_ratio": (self.stats["batched_inputs"] / max(1, self.stats["total_inputs"])),
        }

    async def flush_all(self) -> None:
        """
        Force flush all buffered inputs immediately.

        Useful for cleanup or ensuring all inputs are processed.
        """
        if self.buffer:
            await self._flush_buffer()


class OptimizedInputManager:
    """
    Manager that combines InputBuffer with PokemonGymClient for optimized input processing.

    Provides a clean interface that automatically handles buffering decisions
    and delegates to appropriate processing methods.
    """

    def __init__(self, pokemon_client: Any, buffer_config: BufferConfig | None = None):
        """
        Initialize optimized input manager.

        Args:
            pokemon_client: PokemonGymClient instance
            buffer_config: Optional buffer configuration
        """
        self.client = pokemon_client
        self.buffer = InputBuffer(batch_sender=self._batch_sender_adapter, config=buffer_config)

    async def _batch_sender_adapter(self, input_sequences: list[str]) -> list[dict[str, Any]]:
        """Adapter to connect InputBuffer with PokemonGymClient batch method."""
        return await self.client.send_input_batch_async(input_sequences)

    async def send_optimized(self, input_sequence: str) -> dict[str, Any]:
        """
        Send input with automatic optimization based on usage patterns.

        Args:
            input_sequence: Input sequence to send

        Returns:
            Response from emulator
        """
        return await self.buffer.add_input(input_sequence)

    def get_performance_stats(self) -> dict[str, Any]:
        """Get optimization performance statistics."""
        return self.buffer.get_stats()

    async def shutdown(self) -> None:
        """Graceful shutdown - flush all pending inputs."""
        await self.buffer.flush_all()
