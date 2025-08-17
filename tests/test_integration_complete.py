"""
Complete integration tests for batch input performance optimization.

Tests the full integration of all optimization components working together:
- PokemonGymClient async methods
- InputBuffer intelligence
- OptimizedInputManager coordination
- EmulatorPool integration

Following Uncle Bob's testing philosophy: "Tests are the specs."

Author: Uncle Bot - Integration Testing Specialist
"""

import asyncio
from unittest.mock import patch

import pytest

from claudelearnspokemon.emulator_pool import EmulatorPool, PokemonGymClient
from claudelearnspokemon.input_buffer import BufferConfig, OptimizedInputManager


class TestCompleteIntegration:
    """Complete integration tests for the batch input optimization system."""

    @pytest.mark.asyncio
    async def test_full_optimization_pipeline(self) -> None:
        """Test the complete optimization pipeline from client to buffer to execution."""
        client = PokemonGymClient(port=8081, container_id="integration-test")

        try:
            # Mock the HTTP layer to focus on optimization logic
            with patch.object(client, "send_input_async") as mock_async:

                async def realistic_response(sequence):
                    await asyncio.sleep(0.01)  # 10ms realistic latency
                    return {"status": "success", "input": sequence, "timestamp": "2025-08-17"}

                mock_async.side_effect = realistic_response

                # Test 1: Single optimized input (low frequency - should be immediate)
                result1 = await client.send_input_optimized("A")
                assert result1["status"] == "success"
                assert result1["input"] == "A"

                # Test 2: Batch processing (should be concurrent)
                batch_inputs = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]
                results2 = await client.send_input_batch_async(batch_inputs)

                assert len(results2) == 6
                assert all(r["status"] == "success" for r in results2)

                # Test 3: Rapid optimized inputs (should trigger buffering)
                rapid_tasks = []
                for i in range(5):
                    task = client.send_input_optimized(f"RAPID_{i}")
                    rapid_tasks.append(task)
                    await asyncio.sleep(0.001)  # Very small delay

                results3 = await asyncio.gather(*rapid_tasks)
                assert len(results3) == 5
                assert all(r["status"] == "success" for r in results3)

                # Verify optimization was applied
                if hasattr(client, "_optimized_manager"):
                    stats = client._optimized_manager.get_performance_stats()
                    assert stats["total_inputs"] > 0

        finally:
            client.close()

    @pytest.mark.asyncio
    async def test_emulator_pool_batch_integration(self) -> None:
        """Test batch optimization integration with EmulatorPool."""
        # Create mock emulator pool
        pool = EmulatorPool(pool_size=2, base_port=9001)

        try:
            # Mock the Docker container management
            with (
                patch("docker.from_env"),
                patch.object(pool, "_start_single_container"),
                patch.object(pool, "_wait_for_container_ready"),
                patch.object(pool, "_verify_container_health"),
            ):

                # Create mock clients for the pool
                mock_client1 = PokemonGymClient(port=9001, container_id="mock-1")
                mock_client2 = PokemonGymClient(port=9002, container_id="mock-2")

                # Mock their async methods
                for client in [mock_client1, mock_client2]:

                    async def pool_response(sequence, client=client):
                        await asyncio.sleep(0.008)  # 8ms response
                        return {"status": "success", "client": client.port, "input": sequence}

                    with patch.object(client, "send_input_async", side_effect=pool_response):
                        pass

                # Manually set up the pool clients
                pool.clients_by_port = {9001: mock_client1, 9002: mock_client2}
                pool.available_clients.put(mock_client1)
                pool.available_clients.put(mock_client2)

                # Test batch processing with pool
                with (
                    patch.object(mock_client1, "send_input_async") as mock1,
                    patch.object(mock_client2, "send_input_async") as mock2,
                ):

                    async def mock_response(sequence):
                        await asyncio.sleep(0.005)
                        return {"status": "success", "input": sequence}

                    mock1.side_effect = mock_response
                    mock2.side_effect = mock_response

                    # Acquire clients and test batch processing
                    client1 = pool.acquire(timeout=1.0)
                    client2 = pool.acquire(timeout=1.0)

                    try:
                        # Both clients process batches concurrently
                        batch1 = ["A", "B", "C"]
                        batch2 = ["X", "Y", "Z"]

                        results = await asyncio.gather(
                            client1.send_input_batch_async(batch1),
                            client2.send_input_batch_async(batch2),
                        )

                        # Verify both batches processed successfully
                        assert len(results) == 2
                        assert len(results[0]) == 3  # First batch
                        assert len(results[1]) == 3  # Second batch

                    finally:
                        pool.release(client1)
                        pool.release(client2)
        finally:
            # Clean up pool
            pool.shutdown()

    @pytest.mark.asyncio
    async def test_error_recovery_integration(self) -> None:
        """Test error recovery and fallback behavior in the integrated system."""
        client = PokemonGymClient(port=8081, container_id="error-test")

        try:
            with patch.object(client, "send_input_async") as mock_async:

                # Mock some failures and some successes
                call_count = 0

                async def mixed_response(sequence):
                    nonlocal call_count
                    call_count += 1

                    if call_count <= 2:  # First two calls succeed
                        await asyncio.sleep(0.01)
                        return {"status": "success", "input": sequence}
                    elif call_count == 3:  # Third call fails
                        raise Exception("Network error")
                    else:  # Subsequent calls succeed
                        await asyncio.sleep(0.01)
                        return {"status": "success", "input": sequence}

                mock_async.side_effect = mixed_response

                # Test 1: Successful batch processing
                results1 = await client.send_input_batch_async(["A", "B"])
                assert len(results1) == 2
                assert all(r["status"] == "success" for r in results1)

                # Test 2: Batch with failure should propagate error
                with pytest.raises(Exception, match="Failed to process input batch"):
                    await client.send_input_batch_async(["C", "D"])

                # Test 3: Recovery - subsequent operations should work
                results3 = await client.send_input_batch_async(["E"])
                assert len(results3) == 1
                assert results3[0]["status"] == "success"

        finally:
            client.close()

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self) -> None:
        """Test performance monitoring throughout the integrated system."""
        client = PokemonGymClient(port=8081, container_id="monitoring-test")

        try:
            with patch.object(client, "send_input_async") as mock_async:

                async def timed_response(sequence):
                    await asyncio.sleep(0.005)  # 5ms response
                    return {"status": "success", "input": sequence, "processing_time": "5ms"}

                mock_async.side_effect = timed_response

                # Configure optimized manager with monitoring
                config = BufferConfig(
                    max_wait_ms=2.0, max_batch_size=6, min_batch_size=2, high_frequency_threshold=3
                )

                manager = OptimizedInputManager(client, config)

                # Process various input patterns
                import time

                start_time = time.perf_counter()

                # Low frequency (immediate processing)
                await manager.send_optimized("SINGLE")
                await asyncio.sleep(0.1)  # Reset frequency

                # High frequency (buffering)
                tasks = []
                for i in range(4):  # Should trigger buffering
                    task = manager.send_optimized(f"FREQ_{i}")
                    tasks.append(task)
                    await asyncio.sleep(0.002)  # High frequency

                results = await asyncio.gather(*tasks)
                total_time = time.perf_counter() - start_time

                # Validate results
                assert len(results) == 4
                assert all(r["status"] == "success" for r in results)

                # Check performance statistics
                stats = manager.get_performance_stats()

                assert stats["total_inputs"] == 5  # 1 single + 4 batched
                assert "buffering_ratio" in stats
                assert "avg_batch_size" in stats
                assert stats["buffer_flushes"] >= 1

                # Should be fast due to optimization
                assert total_time < 0.5, f"Integration took {total_time*1000:.1f}ms, should be fast"

                print(f"Integration performance: {total_time*1000:.1f}ms")
                print(f"Stats: {stats}")

        finally:
            client.close()

    @pytest.mark.asyncio
    async def test_realistic_pokemon_workflow(self) -> None:
        """Test a realistic Pokemon speedrun workflow with optimization."""
        client = PokemonGymClient(port=8081, container_id="pokemon-workflow")

        try:
            with patch.object(client, "send_input_async") as mock_async:

                # Mock Pokemon-gym responses
                async def pokemon_response(sequence):
                    await asyncio.sleep(0.015)  # 15ms Pokemon-gym latency
                    return {
                        "status": "success",
                        "input": sequence,
                        "game_state": {"player_x": 10, "player_y": 15, "hp": 85, "level": 12},
                    }

                mock_async.side_effect = pokemon_response

                # Simulate realistic Pokemon speedrun sequence

                # 1. Move through Pallet Town (batch movement)
                movement_sequence = ["DOWN", "DOWN", "RIGHT", "RIGHT", "UP"]
                move_results = await client.send_input_batch_async(movement_sequence)

                assert len(move_results) == 5
                assert all("game_state" in r for r in move_results)

                # 2. Battle sequence (rapid inputs with optimization)
                battle_inputs = [
                    "A",  # Select Fight
                    "DOWN",  # Choose move
                    "A",  # Confirm move
                    "B",
                    "B",
                    "B",  # Skip text rapidly
                    "A",  # Continue
                ]

                import time

                battle_start = time.perf_counter()

                # Use optimized input for battle (should handle rapid B presses efficiently)
                battle_tasks = []
                for battle_input in battle_inputs:
                    task = client.send_input_optimized(battle_input)
                    battle_tasks.append(task)
                    await asyncio.sleep(0.001)  # Rapid succession

                battle_results = await asyncio.gather(*battle_tasks)
                battle_time = time.perf_counter() - battle_start

                assert len(battle_results) == 7
                assert all(r["status"] == "success" for r in battle_results)
                assert battle_time < 0.1, f"Battle took {battle_time*1000:.1f}ms, should be <100ms"

                # 3. Menu navigation (medium batch)
                menu_sequence = [
                    "START",  # Open menu
                    "DOWN",  # Navigate to Pokemon
                    "A",  # Select
                    "UP",
                    "UP",  # Navigate Pokemon list
                    "A",  # Select Pokemon
                    "B",
                    "B",  # Close menus
                ]

                menu_results = await client.send_input_batch_async(menu_sequence)
                assert len(menu_results) == 7

                print("Pokemon workflow completed:")
                print(f"  Movement: {len(move_results)} inputs")
                print(f"  Battle: {len(battle_results)} inputs ({battle_time*1000:.1f}ms)")
                print(f"  Menu: {len(menu_results)} inputs")

        finally:
            client.close()

    def test_system_integration_validation(self) -> None:
        """Validate that all system components integrate properly."""
        # Test that all required classes can be imported and instantiated
        from claudelearnspokemon.emulator_pool import EmulatorPool, PokemonGymClient
        from claudelearnspokemon.input_buffer import (
            BufferConfig,
            OptimizedInputManager,
        )

        # Test client creation
        client = PokemonGymClient(port=8081, container_id="validation")
        assert hasattr(client, "send_input_async")
        assert hasattr(client, "send_input_batch_async")
        assert hasattr(client, "send_input_optimized")

        # Test buffer configuration
        config = BufferConfig()
        assert config.max_wait_ms > 0
        assert config.max_batch_size > 0

        # Test manager creation
        manager = OptimizedInputManager(client, config)
        assert hasattr(manager, "send_optimized")
        assert hasattr(manager, "get_performance_stats")

        # Test pool creation
        pool = EmulatorPool()
        assert hasattr(pool, "initialize")
        assert hasattr(pool, "acquire")

        client.close()
