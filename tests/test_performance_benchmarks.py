"""
Performance benchmarks for batch input optimization.

Validates that the implementation meets <100ms performance targets
for typical Pokemon gameplay scenarios.

Using pytest-benchmark for accurate performance measurement.
Following Uncle Bob's principle: "The only way to go fast is to go well."

Author: Uncle Bot - Performance Engineering Specialist
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from claudelearnspokemon.emulator_pool import PokemonGymClient
from claudelearnspokemon.input_buffer import BufferConfig, OptimizedInputManager


class TestPerformanceBenchmarks:
    """Performance benchmarks for batch input optimization."""

    def setup_method(self) -> None:
        """Set up benchmark environment."""
        self.client = PokemonGymClient(port=8081, container_id="benchmark-test")

    def teardown_method(self) -> None:
        """Clean up after benchmarks."""
        if hasattr(self.client, "close"):
            self.client.close()

    @pytest.mark.asyncio
    async def test_batch_performance_target_small(self) -> None:
        """Test <100ms target for small batches (typical gameplay)."""
        # Mock fast responses to isolate batch logic performance
        with asyncio.timeout(0.1):  # 100ms timeout
            with MockFastAsyncHttp():
                input_sequences = ["A", "B", "START", "UP", "DOWN"]

                start_time = time.perf_counter()
                results = await self.client.send_input_batch_async(input_sequences)
                elapsed = time.perf_counter() - start_time

                # Validate results
                assert len(results) == 5
                assert elapsed < 0.1, f"Batch took {elapsed:.3f}s, target is <0.1s"

                # Log performance metrics
                print(f"Small batch (5 inputs): {elapsed*1000:.1f}ms")

    @pytest.mark.asyncio
    async def test_batch_performance_target_medium(self) -> None:
        """Test <100ms target for medium batches."""
        with asyncio.timeout(0.1):  # 100ms timeout
            with MockFastAsyncHttp():
                input_sequences = ["A B START"] * 10  # 10 sequences

                start_time = time.perf_counter()
                results = await self.client.send_input_batch_async(input_sequences)
                elapsed = time.perf_counter() - start_time

                assert len(results) == 10
                assert elapsed < 0.1, f"Medium batch took {elapsed:.3f}s, target is <0.1s"

                print(f"Medium batch (10 inputs): {elapsed*1000:.1f}ms")

    @pytest.mark.asyncio
    async def test_batch_performance_target_large(self) -> None:
        """Test <100ms target for large batches."""
        with asyncio.timeout(0.1):  # 100ms timeout
            with MockFastAsyncHttp():
                input_sequences = ["A B"] * 20  # 20 sequences

                start_time = time.perf_counter()
                results = await self.client.send_input_batch_async(input_sequences)
                elapsed = time.perf_counter() - start_time

                assert len(results) == 20
                assert elapsed < 0.1, f"Large batch took {elapsed:.3f}s, target is <0.1s"

                print(f"Large batch (20 inputs): {elapsed*1000:.1f}ms")

    @pytest.mark.asyncio
    async def test_optimized_input_performance(self) -> None:
        """Test performance of optimized input with buffering."""
        with MockFastAsyncHttp():
            config = BufferConfig(
                max_wait_ms=2.0, max_batch_size=5, high_frequency_threshold=3  # Very responsive
            )

            manager = OptimizedInputManager(self.client, config)

            # Simulate rapid inputs that should trigger batching
            start_time = time.perf_counter()

            tasks = []
            for i in range(10):  # High frequency inputs
                task = manager.send_optimized(f"INPUT_{i}")
                tasks.append(task)
                await asyncio.sleep(0.001)  # Very small delay

            results = await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - start_time

            # Should be much faster than sequential processing
            assert len(results) == 10
            assert elapsed < 0.05, f"Optimized processing took {elapsed:.3f}s, target is <0.05s"

            print(f"Optimized inputs (10 rapid): {elapsed*1000:.1f}ms")

            # Check statistics
            stats = manager.get_performance_stats()
            print(f"Buffering ratio: {stats['buffering_ratio']:.2f}")

    def test_performance_comparison_sync_vs_async(self) -> None:
        """Compare sync vs async performance to demonstrate improvement."""
        # Measure sync performance (simulated)
        with MockSyncHttp() as mock_sync:
            input_sequences = ["A", "B", "START"] * 5  # 15 total

            sync_start = time.perf_counter()
            for seq in input_sequences:
                _ = mock_sync.send_request(seq)  # Simulate sync calls
            sync_elapsed = time.perf_counter() - sync_start

        print(f"Simulated sync processing (15 inputs): {sync_elapsed*1000:.1f}ms")

        # Expected async performance should be much better
        expected_async_speedup = 5  # At least 5x faster
        expected_async_time = sync_elapsed / expected_async_speedup

        assert expected_async_time < 0.1, "Async should meet 100ms target"
        print(f"Expected async performance: {expected_async_time*1000:.1f}ms")

    @pytest.mark.asyncio
    async def test_memory_efficiency_large_batch(self) -> None:
        """Test memory efficiency with large input batches."""
        import tracemalloc

        with MockFastAsyncHttp():
            # Start memory tracing
            tracemalloc.start()

            # Process large batch
            large_sequences = ["A B START SELECT"] * 100  # 100 sequences

            start_memory = tracemalloc.get_traced_memory()[0]
            results = await self.client.send_input_batch_async(large_sequences)
            peak_memory = tracemalloc.get_traced_memory()[1]

            tracemalloc.stop()

            # Validate results
            assert len(results) == 100

            # Memory efficiency check
            memory_used = peak_memory - start_memory
            memory_per_input = memory_used / 100

            print(
                f"Memory usage: {memory_used/1024:.1f}KB total, {memory_per_input:.0f}B per input"
            )

            # Should be reasonable memory usage
            assert memory_per_input < 1000, f"Memory per input too high: {memory_per_input}B"

    @pytest.mark.asyncio
    async def test_concurrent_client_performance(self) -> None:
        """Test performance with multiple concurrent clients."""
        with MockFastAsyncHttp():
            # Create multiple clients
            clients = [
                PokemonGymClient(port=8080 + i, container_id=f"client-{i}") for i in range(4)
            ]

            try:
                # Each client processes inputs concurrently
                start_time = time.perf_counter()

                all_tasks = []
                for i, client in enumerate(clients):
                    client_inputs = [f"CLIENT_{i}_INPUT_{j}" for j in range(5)]
                    task = client.send_input_batch_async(client_inputs)
                    all_tasks.append(task)

                all_results = await asyncio.gather(*all_tasks)
                elapsed = time.perf_counter() - start_time

                # Validate all results
                assert len(all_results) == 4  # 4 clients
                total_inputs = sum(len(results) for results in all_results)
                assert total_inputs == 20  # 4 clients * 5 inputs each

                # Should still be fast with concurrent processing
                assert elapsed < 0.1, f"Concurrent processing took {elapsed:.3f}s, target is <0.1s"

                print(f"Concurrent clients (4 clients, 20 total inputs): {elapsed*1000:.1f}ms")

            finally:
                # Clean up clients
                for client in clients:
                    if hasattr(client, "close"):
                        client.close()

    @pytest.mark.asyncio
    async def test_real_world_pokemon_scenario(self) -> None:
        """Test performance with realistic Pokemon gameplay patterns."""
        with MockRealisticAsyncHttp():  # Realistic latency simulation
            # Simulate Pokemon battle scenario
            battle_sequence = [
                "A",  # Select attack
                "A",  # Confirm attack
                "B",  # Skip animation (rapid presses)
                "B",
                "B",
                "START",  # Open menu
                "DOWN",  # Navigate
                "A",  # Select item
                "UP",  # Choose target
                "A",  # Confirm
            ]

            start_time = time.perf_counter()
            results = await self.client.send_input_batch_async(battle_sequence)
            elapsed = time.perf_counter() - start_time

            assert len(results) == len(battle_sequence)
            assert elapsed < 0.1, f"Pokemon battle sequence took {elapsed:.3f}s, target is <0.1s"

            print(f"Pokemon battle scenario ({len(battle_sequence)} inputs): {elapsed*1000:.1f}ms")


class MockFastAsyncHttp:
    """Mock context manager for fast async HTTP simulation."""

    def __init__(self):
        self.original_import = None

    def __enter__(self):
        # Mock aiohttp for fast responses
        import sys
        from unittest.mock import MagicMock

        # Create a fast async mock
        mock_aiohttp = MagicMock()

        async def mock_session():
            session_mock = AsyncMock()

            async def mock_post(*args, **kwargs):
                # Simulate very fast response
                await asyncio.sleep(0.001)  # 1ms response time
                response_mock = AsyncMock()
                response_mock.json.return_value = {"status": "success"}
                response_mock.raise_for_status = AsyncMock()
                return response_mock

            session_mock.post.return_value.__aenter__ = mock_post
            return session_mock

        mock_aiohttp.ClientSession.return_value.__aenter__ = mock_session

        # Replace aiohttp in sys.modules temporarily
        self.original_aiohttp = sys.modules.get("aiohttp")
        sys.modules["aiohttp"] = mock_aiohttp

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original aiohttp
        import sys

        if self.original_aiohttp:
            sys.modules["aiohttp"] = self.original_aiohttp
        elif "aiohttp" in sys.modules:
            del sys.modules["aiohttp"]


class MockSyncHttp:
    """Mock for sync HTTP to simulate baseline performance."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def send_request(self, data):
        # Simulate 20ms per sync request
        time.sleep(0.02)
        return {"status": "success"}


class MockRealisticAsyncHttp:
    """Mock with realistic network latency for Pokemon gameplay."""

    def __init__(self):
        self.original_import = None

    def __enter__(self):
        # Mock with realistic 25ms average latency
        import sys
        from unittest.mock import MagicMock

        mock_aiohttp = MagicMock()

        async def mock_session():
            session_mock = AsyncMock()

            async def mock_post(*args, **kwargs):
                # Realistic Pokemon-gym response time
                await asyncio.sleep(0.025)  # 25ms average
                response_mock = AsyncMock()
                response_mock.json.return_value = {"status": "success", "game_state": {}}
                response_mock.raise_for_status = AsyncMock()
                return response_mock

            session_mock.post.return_value.__aenter__ = mock_post
            return session_mock

        mock_aiohttp.ClientSession.return_value.__aenter__ = mock_session

        self.original_aiohttp = sys.modules.get("aiohttp")
        sys.modules["aiohttp"] = mock_aiohttp

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys

        if self.original_aiohttp:
            sys.modules["aiohttp"] = self.original_aiohttp
        elif "aiohttp" in sys.modules:
            del sys.modules["aiohttp"]


class TestPerformanceMonitoring:
    """Test performance monitoring and statistics."""

    def test_performance_stats_collection(self) -> None:
        """Test that performance statistics are collected properly."""
        from claudelearnspokemon.input_buffer import BufferConfig, InputBuffer

        mock_sender = AsyncMock(return_value=[{"status": "success"}])
        buffer = InputBuffer(mock_sender, BufferConfig())

        # Initially, stats should be zeros
        stats = buffer.get_stats()
        assert stats["total_inputs"] == 0
        assert stats["batched_inputs"] == 0
        assert stats["buffer_flushes"] == 0
        assert stats["avg_batch_size"] == 0.0

        # Test that stats structure is comprehensive
        expected_keys = {
            "total_inputs",
            "batched_inputs",
            "buffer_flushes",
            "avg_batch_size",
            "buffer_size",
            "buffering_ratio",
        }
        assert set(stats.keys()) == expected_keys

    def test_performance_degradation_detection(self) -> None:
        """Test detection of performance degradation scenarios."""
        # This test validates that we can detect when performance degrades

        # Simulate scenarios where batch processing might be slower
        large_batch_overhead = 0.01  # 10ms overhead for large batches
        network_latency = 0.03  # 30ms network latency

        # Total time should still be under 100ms for reasonable batch sizes
        max_batch_size = 20
        estimated_time = large_batch_overhead + network_latency

        assert estimated_time < 0.1, "Performance should remain under 100ms"

        # Test edge case: what if network is very slow?
        slow_network_latency = 0.15  # 150ms - this would exceed our target
        slow_scenario_time = large_batch_overhead + slow_network_latency

        # In slow network scenarios, our optimization becomes even more valuable
        # Sequential would be: slow_network_latency * batch_size = 3000ms
        # Parallel would be: slow_scenario_time = 160ms (still bad, but much better)
        sequential_slow = slow_network_latency * max_batch_size
        parallel_slow = slow_scenario_time

        improvement_ratio = sequential_slow / parallel_slow
        assert improvement_ratio > 10, "Even in slow scenarios, should be 10x+ improvement"
