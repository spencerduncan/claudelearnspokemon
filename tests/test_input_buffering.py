"""
Tests for intelligent input buffering and batching strategies.

Validates the InputBuffer and OptimizedInputManager implementations
following TDD principles and Uncle Bob's testing philosophy.

Author: Uncle Bot - Clean Code Testing Specialist
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from claudelearnspokemon.input_buffer import (
    BufferConfig,
    InputBuffer,
    OptimizedInputManager,
)


class TestBufferConfig:
    """Test buffer configuration dataclass."""

    def test_default_configuration(self) -> None:
        """Test default buffer configuration values."""
        config = BufferConfig()

        assert config.max_wait_ms == 5.0
        assert config.max_batch_size == 10
        assert config.min_batch_size == 2
        assert config.high_frequency_threshold == 5

    def test_custom_configuration(self) -> None:
        """Test custom buffer configuration."""
        config = BufferConfig(
            max_wait_ms=10.0, max_batch_size=20, min_batch_size=3, high_frequency_threshold=8
        )

        assert config.max_wait_ms == 10.0
        assert config.max_batch_size == 20
        assert config.min_batch_size == 3
        assert config.high_frequency_threshold == 8


class TestInputBuffer:
    """Test intelligent input buffering logic."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.mock_batch_sender = AsyncMock()
        self.config = BufferConfig(
            max_wait_ms=10.0, max_batch_size=3, min_batch_size=2, high_frequency_threshold=3
        )
        self.buffer = InputBuffer(self.mock_batch_sender, self.config)

    @pytest.mark.asyncio
    async def test_low_frequency_immediate_processing(self) -> None:
        """Test that low-frequency inputs are processed immediately."""
        self.mock_batch_sender.return_value = [{"status": "success", "input": "A"}]

        # Send single input (low frequency)
        result = await self.buffer.add_input("A")

        # Should be processed immediately
        assert result["status"] == "success"
        assert result["input"] == "A"
        self.mock_batch_sender.assert_called_once_with(["A"])

    @pytest.mark.asyncio
    async def test_high_frequency_buffering(self) -> None:
        """Test that high-frequency inputs trigger buffering."""
        self.mock_batch_sender.return_value = [
            {"status": "success", "input": "A"},
            {"status": "success", "input": "B"},
            {"status": "success", "input": "C"},
        ]

        # Simulate high frequency by adding timestamps
        current_time = time.time()
        for _i in range(5):  # Exceed threshold
            self.buffer.input_timestamps.append(current_time - 0.1 * _i)

        # Send inputs rapidly
        tasks = [self.buffer.add_input("A"), self.buffer.add_input("B"), self.buffer.add_input("C")]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)

        # Should be batched (only one call to batch_sender)
        self.mock_batch_sender.assert_called_once_with(["A", "B", "C"])

    @pytest.mark.asyncio
    async def test_buffer_size_limit_flush(self) -> None:
        """Test that buffer flushes when size limit is reached."""
        self.mock_batch_sender.return_value = [
            {"status": "success", "input": "A"},
            {"status": "success", "input": "B"},
            {"status": "success", "input": "C"},
        ]

        # Force high frequency mode
        current_time = time.time()
        for _i in range(5):
            self.buffer.input_timestamps.append(current_time)

        # Add exactly max_batch_size inputs
        tasks = []
        for input_char in ["A", "B", "C"]:  # max_batch_size = 3
            tasks.append(self.buffer.add_input(input_char))

        results = await asyncio.gather(*tasks)

        # Should flush immediately due to size limit
        assert len(results) == 3
        self.mock_batch_sender.assert_called_once()

    @pytest.mark.asyncio
    async def test_time_based_flush(self) -> None:
        """Test that buffer flushes based on time limits."""
        self.mock_batch_sender.return_value = [
            {"status": "success", "input": "A"},
            {"status": "success", "input": "B"},
        ]

        # Force high frequency mode
        current_time = time.time()
        for _i in range(5):
            self.buffer.input_timestamps.append(current_time)

        # Add inputs that won't trigger size-based flush
        task1 = self.buffer.add_input("A")
        await asyncio.sleep(0.001)  # Small delay
        task2 = self.buffer.add_input("B")

        # Wait for time-based flush (slightly longer than max_wait_ms)
        results = await asyncio.gather(task1, task2)

        assert len(results) == 2
        assert all(r["status"] == "success" for r in results)

    @pytest.mark.asyncio
    async def test_buffer_statistics(self) -> None:
        """Test buffer performance statistics tracking."""
        self.mock_batch_sender.return_value = [{"status": "success"}]

        # Process some inputs
        await self.buffer.add_input("A")  # Immediate (low frequency)

        # Force high frequency for batching
        current_time = time.time()
        for _i in range(5):
            self.buffer.input_timestamps.append(current_time)

        await self.buffer.add_input("B")
        await self.buffer.add_input("C")

        # Wait for potential buffering
        await asyncio.sleep(0.02)

        stats = self.buffer.get_stats()

        assert stats["total_inputs"] == 3
        assert "buffer_flushes" in stats
        assert "avg_batch_size" in stats
        assert "buffering_ratio" in stats

    @pytest.mark.asyncio
    async def test_error_handling_in_batch_sender(self) -> None:
        """Test error handling when batch sender fails."""
        self.mock_batch_sender.side_effect = Exception("Network error")

        # Send input that should fail
        with pytest.raises(Exception, match="Network error"):
            await self.buffer.add_input("A")

    @pytest.mark.asyncio
    async def test_flush_all_functionality(self) -> None:
        """Test manual flush of all buffered inputs."""
        self.mock_batch_sender.return_value = [{"status": "success"}]

        # Force buffering mode
        current_time = time.time()
        for _i in range(5):
            self.buffer.input_timestamps.append(current_time)

        # Add input to buffer without waiting
        task = self.buffer.add_input("A")

        # Force flush
        await self.buffer.flush_all()

        # Task should complete
        result = await task
        assert result["status"] == "success"


class TestOptimizedInputManager:
    """Test the high-level optimized input manager."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.mock_client = Mock()
        self.mock_client.send_input_batch_async = AsyncMock()
        self.manager = OptimizedInputManager(
            self.mock_client, BufferConfig(max_wait_ms=5.0, max_batch_size=3)
        )

    @pytest.mark.asyncio
    async def test_manager_delegates_to_buffer(self) -> None:
        """Test that manager properly delegates to input buffer."""
        self.mock_client.send_input_batch_async.return_value = [
            {"status": "success", "input": "UP"}
        ]

        result = await self.manager.send_optimized("UP")

        assert result["status"] == "success"
        assert result["input"] == "UP"

    @pytest.mark.asyncio
    async def test_batch_sender_adapter(self) -> None:
        """Test the batch sender adapter functionality."""
        expected_results = [
            {"status": "success", "input": "A"},
            {"status": "success", "input": "B"},
        ]
        self.mock_client.send_input_batch_async.return_value = expected_results

        # Call adapter directly
        results = await self.manager._batch_sender_adapter(["A", "B"])

        assert results == expected_results
        self.mock_client.send_input_batch_async.assert_called_once_with(["A", "B"])

    def test_performance_stats_access(self) -> None:
        """Test access to performance statistics."""
        stats = self.manager.get_performance_stats()

        assert isinstance(stats, dict)
        assert "total_inputs" in stats
        assert "buffering_ratio" in stats

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self) -> None:
        """Test graceful shutdown functionality."""
        # Should not raise any exceptions
        await self.manager.shutdown()


class TestBufferingIntegration:
    """Integration tests for buffering with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_mixed_frequency_scenario(self) -> None:
        """Test buffering behavior with mixed input frequencies."""
        mock_batch_sender = AsyncMock()
        mock_batch_sender.return_value = [{"status": "success"}]

        buffer = InputBuffer(mock_batch_sender, BufferConfig())

        # Low frequency inputs (should be immediate)
        await buffer.add_input("INITIAL")
        await asyncio.sleep(0.5)  # Wait to reset frequency

        # High frequency burst (should be buffered)
        tasks = []
        for i in range(4):  # Rapid succession
            tasks.append(buffer.add_input(f"RAPID_{i}"))
            await asyncio.sleep(0.001)  # Very small delay

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 4
        assert all(r["status"] == "success" for r in results)

        # Check batching behavior
        stats = buffer.get_stats()
        assert stats["total_inputs"] == 5  # 1 initial + 4 rapid

    @pytest.mark.asyncio
    async def test_pokemon_gameplay_simulation(self) -> None:
        """Test buffering with Pokemon-like input patterns."""
        mock_batch_sender = AsyncMock()

        # Mock to return appropriate number of results based on input count
        def mock_sender(inputs):
            return [{"status": "success", "input": inp} for inp in inputs]

        mock_batch_sender.side_effect = mock_sender

        config = BufferConfig(
            max_wait_ms=3.0,  # Gaming-appropriate latency
            max_batch_size=8,
            min_batch_size=2,
            high_frequency_threshold=4,
        )

        buffer = InputBuffer(mock_batch_sender, config)

        # Simulate Pokemon combat - rapid button presses
        combat_inputs = ["A", "A", "B", "UP", "A", "START"]

        # Simulate high frequency by manipulating timestamps
        current_time = time.time()
        for i in range(6):
            buffer.input_timestamps.append(current_time - 0.1 * i)

        # Send inputs rapidly
        tasks = [buffer.add_input(inp) for inp in combat_inputs]
        results = await asyncio.gather(*tasks)

        # All should process successfully
        assert len(results) == 6

        # Debug: Print results to understand the failure
        print(f"Results: {results}")

        # Check each result individually
        for i, result in enumerate(results):
            print(f"Result {i}: {result}")
            assert result is not None, f"Result {i} is None"

        assert all(r["status"] == "success" for r in results)

        # Should achieve batching efficiency
        stats = buffer.get_stats()
        assert stats["total_inputs"] == 6
        assert stats["buffering_ratio"] > 0  # Some inputs were batched
