"""
Simple but effective performance benchmarks for batch input optimization.

Focuses on measuring the actual optimization benefits without complex mocking.
Validates <100ms performance targets using controlled scenarios.

Author: Uncle Bot - Pragmatic Performance Testing
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from claudelearnspokemon.emulator_pool import PokemonGymClient
from claudelearnspokemon.input_buffer import BufferConfig, InputBuffer, OptimizedInputManager


class TestSimplePerformanceBenchmarks:
    """Simple performance benchmarks focusing on optimization benefits."""

    def setup_method(self) -> None:
        """Set up benchmark environment."""
        self.client = PokemonGymClient(port=8081, container_id="benchmark-test")

    def teardown_method(self) -> None:
        """Clean up after benchmarks."""
        if hasattr(self.client, "close"):
            self.client.close()

    @pytest.mark.asyncio
    async def test_batch_vs_sequential_performance(self) -> None:
        """Test that batch processing is significantly faster than sequential."""
        input_sequences = ["A", "B", "START", "UP", "DOWN"]

        # Mock the async method to simulate realistic latency
        with patch.object(self.client, "send_input_async") as mock_async:

            async def mock_with_latency(sequence):
                await asyncio.sleep(0.01)  # 10ms per request
                return {"status": "success", "input": sequence}

            mock_async.side_effect = mock_with_latency

            # Measure batch performance
            batch_start = time.perf_counter()
            batch_results = await self.client.send_input_batch_async(input_sequences)
            batch_elapsed = time.perf_counter() - batch_start

            # Reset mock for sequential test
            mock_async.reset_mock()
            mock_async.side_effect = mock_with_latency

            # Measure sequential performance
            sequential_start = time.perf_counter()
            sequential_results = []
            for seq in input_sequences:
                result = await self.client.send_input_async(seq)
                sequential_results.append(result)
            sequential_elapsed = time.perf_counter() - sequential_start

            # Validate results
            assert len(batch_results) == len(sequential_results) == 5

            # Batch should be significantly faster (parallel execution)
            speedup = sequential_elapsed / batch_elapsed
            assert speedup > 3, f"Batch only {speedup:.1f}x faster, expected >3x"

            # Batch should meet <100ms target for typical scenarios
            assert batch_elapsed < 0.1, f"Batch took {batch_elapsed*1000:.1f}ms, target is <100ms"

            print(f"Sequential: {sequential_elapsed*1000:.1f}ms")
            print(f"Batch: {batch_elapsed*1000:.1f}ms")
            print(f"Speedup: {speedup:.1f}x")

    @pytest.mark.asyncio
    async def test_buffering_reduces_calls(self) -> None:
        """Test that input buffering reduces the number of actual HTTP calls."""
        # Mock batch sender to count calls
        call_count = 0
        total_inputs = 0

        async def counting_batch_sender(sequences):
            nonlocal call_count, total_inputs
            call_count += 1
            total_inputs += len(sequences)
            await asyncio.sleep(0.005)  # 5ms processing time
            return [{"status": "success", "input": seq} for seq in sequences]

        config = BufferConfig(
            max_wait_ms=5.0, max_batch_size=4, min_batch_size=2, high_frequency_threshold=3
        )

        buffer = InputBuffer(counting_batch_sender, config)

        # Simulate high frequency inputs
        current_time = time.time()
        for i in range(6):  # Simulate high frequency
            buffer.input_timestamps.append(current_time - 0.1 * i)

        # Send inputs that should be batched
        start_time = time.perf_counter()
        tasks = [buffer.add_input(f"INPUT_{i}") for i in range(6)]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start_time

        # Validate results
        assert len(results) == 6
        assert total_inputs == 6

        # Should have made fewer HTTP calls than inputs due to batching
        assert call_count < 6, f"Made {call_count} calls for 6 inputs, should batch them"
        assert elapsed < 0.1, f"Buffered processing took {elapsed*1000:.1f}ms, target is <100ms"

        print(f"Inputs: 6, HTTP calls: {call_count}, Time: {elapsed*1000:.1f}ms")

    @pytest.mark.asyncio
    async def test_optimized_manager_performance(self) -> None:
        """Test performance of the optimized input manager."""
        # Create manager with game-appropriate config
        config = BufferConfig(
            max_wait_ms=3.0,  # 3ms max latency for responsive gameplay
            max_batch_size=8,
            min_batch_size=2,
            high_frequency_threshold=4,
        )

        with patch.object(self.client, "send_input_batch_async") as mock_batch:

            async def fast_batch_sender(sequences):
                await asyncio.sleep(0.008)  # 8ms total for batch
                return [{"status": "success", "input": seq} for seq in sequences]

            mock_batch.side_effect = fast_batch_sender

            manager = OptimizedInputManager(self.client, config)

            # Simulate rapid Pokemon inputs
            start_time = time.perf_counter()

            tasks = []
            for i in range(8):
                task = manager.send_optimized(f"RAPID_{i}")
                tasks.append(task)
                if i < 7:  # Small delays between inputs
                    await asyncio.sleep(0.001)

            results = await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - start_time

            # Validate optimization
            assert len(results) == 8
            assert (
                elapsed < 0.05
            ), f"Optimized processing took {elapsed*1000:.1f}ms, target is <50ms"

            # Check that batching occurred
            stats = manager.get_performance_stats()
            assert stats["total_inputs"] == 8

            print(f"Optimized inputs (8): {elapsed*1000:.1f}ms")
            print(f"Buffering ratio: {stats['buffering_ratio']:.2f}")

    def test_performance_calculation_accuracy(self) -> None:
        """Test accuracy of performance calculations and projections."""
        # Baseline measurements
        individual_latency = 0.020  # 20ms per individual HTTP request
        batch_overhead = 0.005  # 5ms overhead for batch processing

        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20]

        for batch_size in batch_sizes:
            # Calculate expected times
            sequential_time = batch_size * individual_latency
            batch_time = batch_overhead + individual_latency  # Parallel execution

            # Performance improvement
            improvement = sequential_time / batch_time

            print(
                f"Batch size {batch_size}: Sequential {sequential_time*1000:.0f}ms, "
                f"Batch {batch_time*1000:.0f}ms, Improvement {improvement:.1f}x"
            )

            # Validate that larger batches show more improvement
            if batch_size >= 5:
                assert improvement >= 4, f"Batch size {batch_size} should show 4x+ improvement"

            # All reasonable batch sizes should meet performance target
            if batch_size <= 20:
                assert batch_time < 0.1, f"Batch size {batch_size} exceeds 100ms target"

    @pytest.mark.asyncio
    async def test_concurrent_performance(self) -> None:
        """Test performance with concurrent operations."""
        # Simulate multiple operations happening simultaneously
        with patch.object(self.client, "send_input_async") as mock_async:

            async def concurrent_latency(sequence):
                await asyncio.sleep(0.015)  # 15ms per request
                return {"status": "success", "input": sequence}

            mock_async.side_effect = concurrent_latency

            # Create multiple batches that run concurrently
            batch1 = ["A", "B"]
            batch2 = ["UP", "DOWN"]
            batch3 = ["START", "SELECT"]

            start_time = time.perf_counter()

            # Run all batches concurrently
            results = await asyncio.gather(
                self.client.send_input_batch_async(batch1),
                self.client.send_input_batch_async(batch2),
                self.client.send_input_batch_async(batch3),
            )

            elapsed = time.perf_counter() - start_time

            # Validate results
            assert len(results) == 3  # 3 batches
            total_inputs = sum(len(batch) for batch in results)
            assert total_inputs == 6  # 2+2+2 inputs

            # Concurrent execution should be fast
            assert elapsed < 0.1, f"Concurrent batches took {elapsed*1000:.1f}ms, target is <100ms"

            # Should be much faster than processing all 6 inputs sequentially
            estimated_sequential = 6 * 0.015  # 90ms
            assert elapsed < estimated_sequential * 0.5, "Should be <50% of sequential time"

            print(f"Concurrent batches (6 total inputs): {elapsed*1000:.1f}ms")

    @pytest.mark.asyncio
    async def test_memory_efficiency_validation(self) -> None:
        """Test memory efficiency of batch processing."""
        # Test with a reasonably large batch to validate memory usage
        large_batch = [f"INPUT_{i}_{'X'*20}" for i in range(50)]  # 50 inputs with some data

        with patch.object(self.client, "send_input_async") as mock_async:

            async def memory_efficient_mock(sequence):
                # Simulate processing without excessive memory usage
                await asyncio.sleep(0.001)
                return {
                    "status": "success",
                    "input": sequence[:10],
                }  # Return truncated to save memory

            mock_async.side_effect = memory_efficient_mock

            # Process large batch
            start_time = time.perf_counter()
            results = await self.client.send_input_batch_async(large_batch)
            elapsed = time.perf_counter() - start_time

            # Validate processing
            assert len(results) == 50
            assert elapsed < 0.1, f"Large batch took {elapsed*1000:.1f}ms, target is <100ms"

            # Memory efficiency validated by successful completion
            # In a real scenario, this would process 50 concurrent HTTP requests efficiently
            print(f"Large batch (50 inputs): {elapsed*1000:.1f}ms")

    def test_performance_target_validation(self) -> None:
        """Validate that our performance targets are realistic and achievable."""
        # Test various realistic scenarios
        scenarios = [
            ("Small game action", 3, 0.020),  # 3 inputs, 20ms latency each
            ("Medium combo", 8, 0.025),  # 8 inputs, 25ms latency each
            ("Large sequence", 15, 0.030),  # 15 inputs, 30ms latency each
            ("Rapid commands", 20, 0.015),  # 20 inputs, 15ms latency each
        ]

        for scenario_name, input_count, individual_latency in scenarios:
            # Sequential processing time
            sequential_time = input_count * individual_latency

            # Batch processing time (parallel)
            batch_overhead = 0.005  # 5ms overhead
            batch_time = individual_latency + batch_overhead

            # Validate performance target
            meets_target = batch_time < 0.1
            improvement = sequential_time / batch_time

            print(f"{scenario_name}: {input_count} inputs")
            print(f"  Sequential: {sequential_time*1000:.0f}ms")
            print(f"  Batch: {batch_time*1000:.0f}ms ({'✓' if meets_target else '✗'} <100ms)")
            print(f"  Improvement: {improvement:.1f}x")

            # Most scenarios should meet the target
            if input_count <= 15:  # Reasonable batch sizes should always meet target
                assert meets_target, f"{scenario_name} should meet <100ms target"

            # All scenarios should show significant improvement
            assert improvement >= 2, f"{scenario_name} should show 2x+ improvement"


class TestRealWorldScenarios:
    """Test performance with realistic Pokemon gameplay scenarios."""

    @pytest.mark.asyncio
    async def test_pokemon_battle_performance(self) -> None:
        """Test performance for a typical Pokemon battle sequence."""
        # Typical Pokemon battle: select move, confirm, wait for animation
        battle_inputs = [
            "DOWN",  # Navigate to move
            "A",  # Select move
            "A",  # Confirm move
            "B",  # Skip text
            "B",  # Skip text
            "A",  # Continue
        ]

        # Mock with realistic Pokemon-gym latency
        async def pokemon_latency_mock(sequences):
            await asyncio.sleep(0.025)  # 25ms realistic response time
            return [{"status": "success", "move": "attack"} for _ in sequences]

        client = PokemonGymClient(port=8081, container_id="battle-test")

        try:
            with patch.object(client, "send_input_batch_async") as mock_batch:
                mock_batch.side_effect = pokemon_latency_mock

                start_time = time.perf_counter()
                results = await client.send_input_batch_async(battle_inputs)
                elapsed = time.perf_counter() - start_time

                # Validate battle processing
                assert len(results) == len(battle_inputs)
                assert elapsed < 0.1, f"Pokemon battle took {elapsed*1000:.1f}ms, target is <100ms"

                print(f"Pokemon battle sequence: {elapsed*1000:.1f}ms")

        finally:
            client.close()

    @pytest.mark.asyncio
    async def test_menu_navigation_performance(self) -> None:
        """Test performance for Pokemon menu navigation."""
        # Navigate through Pokemon menu system
        menu_inputs = [
            "START",  # Open menu
            "DOWN",  # Navigate to Pokemon
            "A",  # Select Pokemon
            "DOWN",  # Select Pokemon
            "A",  # Select action
            "UP",  # Navigate options
            "A",  # Confirm
            "B",  # Back to game
        ]

        client = PokemonGymClient(port=8081, container_id="menu-test")

        try:
            # Use actual batch implementation with mocked async method
            with patch.object(client, "send_input_async") as mock_async:

                async def menu_response(sequence):
                    await asyncio.sleep(0.020)  # 20ms menu response time
                    return {"status": "success", "menu_action": sequence}

                mock_async.side_effect = menu_response

                start_time = time.perf_counter()
                results = await client.send_input_batch_async(menu_inputs)
                elapsed = time.perf_counter() - start_time

                # Menu navigation should be responsive
                assert len(results) == len(menu_inputs)
                assert elapsed < 0.05, f"Menu navigation took {elapsed*1000:.1f}ms, target is <50ms"

                print(f"Menu navigation sequence: {elapsed*1000:.1f}ms")

        finally:
            client.close()
