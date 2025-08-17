"""
Basic tests for async input functionality.
Simple validation without performance markers.
"""

from unittest.mock import patch

import aiohttp
import pytest

from claudelearnspokemon.emulator_pool import PokemonGymClient


class TestAsyncInputBasic:
    """Basic tests for async input methods."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.client = PokemonGymClient(port=8081, container_id="test-container")

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if hasattr(self.client, "close"):
            self.client.close()

    @pytest.mark.asyncio
    async def test_send_input_async_basic(self) -> None:
        """Test basic async input functionality."""
        # Mock the _make_async_request method directly to avoid aiohttp complexity
        with patch.object(self.client, "_make_async_request") as mock_request:
            mock_request.return_value = {"status": "success", "result": "A pressed"}

            # Test async input
            result = await self.client.send_input_async("A")

            # Verify result
            assert result["status"] == "success"
            assert result["result"] == "A pressed"

            # Verify the internal method was called correctly
            mock_request.assert_called_once_with("/input", {"inputs": "A"})

    @pytest.mark.asyncio
    async def test_send_input_batch_async_basic(self) -> None:
        """Test basic batch async functionality."""
        input_sequences = ["A", "B", "START"]

        # Mock the _make_async_request method directly
        expected_results = []
        for i, seq in enumerate(input_sequences):
            expected_results.append({"status": "success", "input": seq, "id": i})

        # Mock each individual request
        with patch.object(self.client, "_make_async_request") as mock_request:
            mock_request.side_effect = expected_results

            # Test batch processing
            results = await self.client.send_input_batch_async(input_sequences)

            # Verify results
            assert len(results) == 3
            assert results[0]["input"] == "A"
            assert results[1]["input"] == "B"
            assert results[2]["input"] == "START"

            # Verify all HTTP calls were made
            assert mock_request.call_count == 3

    @pytest.mark.asyncio
    async def test_send_input_optimized_basic(self) -> None:
        """Test optimized input method."""
        # Mock the internal method directly
        with patch.object(self.client, "_make_async_request") as mock_request:
            mock_request.return_value = {"status": "success"}

            # Test optimized input (currently delegates to async)
            result = await self.client.send_input_optimized("UP DOWN")

            assert result["status"] == "success"
            mock_request.assert_called_once_with("/input", {"inputs": "UP DOWN"})

    @pytest.mark.asyncio
    async def test_empty_batch(self) -> None:
        """Test batch processing with empty input list."""
        result = await self.client.send_input_batch_async([])
        assert result == []

    @pytest.mark.asyncio
    async def test_async_error_handling(self) -> None:
        """Test error handling in async methods."""
        # Mock the internal method to raise an exception
        from claudelearnspokemon.emulator_pool import EmulatorPoolError

        with patch.object(self.client, "_make_async_request") as mock_request:
            # Make the internal method raise an aiohttp error
            mock_request.side_effect = aiohttp.ClientError("Connection failed")

            # Should raise EmulatorPoolError
            with pytest.raises(EmulatorPoolError, match="Failed to send async input"):
                await self.client.send_input_async("A")
