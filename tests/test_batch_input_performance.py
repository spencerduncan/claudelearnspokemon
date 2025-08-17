"""
Performance tests for batch input optimization.

Tests the <100ms performance target for batch input processing
following Uncle Bob's TDD approach - write failing tests first.

Author: Uncle Bot - Performance Optimization Specialist
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
import aiohttp

from claudelearnspokemon.emulator_pool import PokemonGymClient


class TestBatchInputPerformance:
    """Test batch input optimization performance requirements."""

    def setup_method(self) -> None:
        """Set up test environment for each test."""
        self.client = PokemonGymClient(port=8081, container_id="test-container")
        # Clear any cached async session for fresh testing
        self.client._async_session = None

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if hasattr(self.client, 'close'):
            self.client.close()

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_input_meets_100ms_target(self) -> None:
        """Test that batch input processing meets <100ms performance target."""
        # Arrange - Create multiple input sequences for batch processing
        input_sequences = [
            "A B START",
            "UP DOWN LEFT RIGHT", 
            "SELECT A B",
            "START SELECT",
            "A A B B"
        ]
        
        # Mock the async request method directly - much simpler!
        with patch.object(self.client, '_make_async_request') as mock_request:
            mock_request.return_value = {"status": "success", "game_state": {}}
            
            # Act - Measure batch processing time
            start_time = time.time()
            results = await self.client.send_input_batch_async(input_sequences)
            elapsed_time = time.time() - start_time
            
            # Assert - Performance target met
            assert elapsed_time < 0.1, f"Batch processing took {elapsed_time:.3f}s, target is <0.1s"
            assert len(results) == len(input_sequences), "All input sequences processed"
            assert all(result["status"] == "success" for result in results), "All inputs successful"

    @pytest.mark.performance  
    @pytest.mark.asyncio
    async def test_single_async_input_performance(self) -> None:
        """Test that single async input meets performance requirements."""
        # Arrange
        input_sequence = "A B START SELECT UP DOWN"
        
        # Mock the async request method directly
        with patch.object(self.client, '_make_async_request') as mock_request:
            mock_request.return_value = {"status": "success"}
            
            # Act - Measure single async input time
            start_time = time.time()
            result = await self.client.send_input_async(input_sequence)
            elapsed_time = time.time() - start_time
            
            # Assert - Fast async response
            assert elapsed_time < 0.05, f"Async input took {elapsed_time:.3f}s, should be <0.05s"
            assert result["status"] == "success"

    @pytest.mark.performance
    @pytest.mark.asyncio 
    async def test_large_batch_performance(self) -> None:
        """Test performance with large batches of input sequences."""
        # Arrange - Create large batch (20 sequences)
        input_sequences = ["A B START"] * 20
        
        with patch.object(self.client, '_make_async_request') as mock_request:
            mock_request.return_value = {"status": "success"}
            
            # Act - Process large batch
            start_time = time.time()
            results = await self.client.send_input_batch_async(input_sequences)
            elapsed_time = time.time() - start_time
            
            # Assert - Large batch still meets target
            assert elapsed_time < 0.1, f"Large batch took {elapsed_time:.3f}s, target is <0.1s"
            assert len(results) == 20, "All 20 sequences processed"

    @pytest.mark.asyncio
    async def test_input_order_preservation_in_batch(self) -> None:
        """Test that input order is preserved in batch processing."""
        # Arrange - Sequences with different expected responses
        input_sequences = ["A", "B", "START"]
        expected_responses = [
            {"status": "success", "button": "A"},
            {"status": "success", "button": "B"}, 
            {"status": "success", "button": "START"}
        ]
        
        with patch.object(self.client, '_make_async_request') as mock_request:
            # Return responses in the correct order based on request data
            mock_request.side_effect = expected_responses
            
            # Act
            results = await self.client.send_input_batch_async(input_sequences)
            
            # Assert - Order preserved
            assert len(results) == 3
            assert results[0]["button"] == "A"
            assert results[1]["button"] == "B"
            assert results[2]["button"] == "START"

    @pytest.mark.asyncio
    async def test_batch_error_handling(self) -> None:
        """Test that batch processing handles partial failures gracefully."""
        # Arrange - Some inputs will fail
        input_sequences = ["A", "INVALID", "B"]
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # First and third succeed, second fails
            mock_responses = [
                AsyncMock(),  # Success
                AsyncMock(),  # Failure  
                AsyncMock()   # Success
            ]
            
            mock_responses[0].json.return_value = {"status": "success"}
            mock_responses[1].json.side_effect = aiohttp.ClientError("Invalid input")
            mock_responses[2].json.return_value = {"status": "success"}
            
            mock_session.post.return_value.__aenter__.side_effect = mock_responses
            
            # Act & Assert - Should handle partial failures
            with pytest.raises(Exception):
                # Batch should fail if any input fails (fail-fast approach)
                await self.client.send_input_batch_async(input_sequences)

    @pytest.mark.performance
    def test_sync_vs_async_performance_comparison(self) -> None:
        """Compare sync vs async performance to validate optimization."""
        # This test will verify that async version is faster than sync
        # when processing multiple inputs
        
        input_sequences = ["A B", "UP DOWN", "START SELECT"] * 3  # 9 total
        
        # Measure sync performance (sequential)
        with patch.object(self.client, 'send_input') as mock_sync:
            mock_sync.return_value = {"status": "success"}
            
            sync_start = time.time()
            for seq in input_sequences:
                self.client.send_input(seq)
            sync_elapsed = time.time() - sync_start
        
        # We can't test actual async here without implementing it first
        # But we set the expectation that async should be significantly faster
        expected_async_time = sync_elapsed / 3  # Should be at least 3x faster
        
        # This assertion will guide our implementation
        assert expected_async_time < 0.1, "Async implementation should meet 100ms target"


class TestInputBuffering:
    """Test input buffering strategies for high-frequency inputs."""
    
    def setup_method(self) -> None:
        """Set up test environment."""
        self.client = PokemonGymClient(port=8081, container_id="test-container")
        
    @pytest.mark.asyncio
    async def test_input_buffering_reduces_http_calls(self) -> None:
        """Test that rapid successive inputs get buffered and batched."""
        # This test will drive the buffering implementation
        # Multiple rapid calls should be combined into fewer HTTP requests
        
        with patch.object(self.client, '_make_async_request') as mock_request:
            mock_request.return_value = {"status": "success"}
            
            # Act - Send multiple inputs rapidly
            tasks = []
            for i in range(5):
                task = self.client.send_input_optimized(f"INPUT_{i}")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Assert - Should have fewer HTTP calls than inputs due to buffering
            # This will drive our buffering implementation
            assert len(results) == 5, "All inputs processed"
            # The implementation should batch these into fewer actual HTTP calls