"""
Debug test for async mocking issues.
"""

from unittest.mock import AsyncMock, patch

import pytest

from claudelearnspokemon.emulator_pool import PokemonGymClient


class TestMockDebug:
    """Debug async context manager mocking."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.client = PokemonGymClient(port=8081, container_id="test-container")

    @pytest.mark.asyncio
    async def test_debug_async_session_mock(self) -> None:
        """Debug the async session mocking."""
        print(f"Client has _async_session: {hasattr(self.client, '_async_session')}")
        print(f"_async_session value: {getattr(self.client, '_async_session', 'NOT_SET')}")

        # Test calling _get_async_session directly
        try:
            session = await self.client._get_async_session()
            print(f"Got session: {session}")
            print(f"Session type: {type(session)}")
        except Exception as e:
            print(f"Error getting session: {e}")

        # Now test with mock
        with patch.object(self.client, "_get_async_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value = mock_session

            # Create async context manager mock
            class AsyncContextManagerMock:
                def __init__(self, response):
                    self.response = response

                async def __aenter__(self):
                    return self.response

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return None

            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={"status": "success"})
            mock_response.raise_for_status = AsyncMock()

            mock_session.post.return_value = AsyncContextManagerMock(mock_response)

            try:
                result = await self.client.send_input_async("A B")
                print(f"Success! Result: {result}")
            except Exception as e:
                print(f"Error with proper mock: {e}")
