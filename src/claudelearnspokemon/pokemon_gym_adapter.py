"""
Pokemon Gym Adapter - Bridges EmulatorPool with benchflow-ai/pokemon-gym API.

Implements the Adapter Pattern to translate between our expected interface
and the actual benchflow-ai API, following SOLID principles and clean code practices.

Key Components:
- PokemonGymAdapter: Main adapter implementing PokemonGymClient interface
- SessionManager: Handles benchflow-ai session lifecycle
- Error handling with proper exception hierarchy
- Performance optimization for <100ms batch operations

Author: Uncle Bot - Clean Code Implementation
"""

import json
import logging
from typing import Any

import requests

# Configure logging for production observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PokemonGymAdapterError(Exception):
    """Custom exception for Pokemon Gym Adapter operations."""

    pass


class SessionManager:
    """
    Manages benchflow-ai session lifecycle with automatic recovery.

    Implements Single Responsibility Principle - only handles session state.
    """

    def __init__(self, base_url: str, config: dict[str, Any] | None = None):
        self.base_url = base_url
        self.config = config or {}
        self.session_id: str | None = None
        self.is_initialized: bool = False
        self._session = requests.Session()

    def initialize_session(self) -> dict[str, Any]:
        """Initialize new session with benchflow-ai server."""
        try:
            response = self._session.post(
                f"{self.base_url}/initialize", json={"config": self.config}, timeout=10
            )
            response.raise_for_status()

            data = response.json()
            self.session_id = data.get("session_id")
            self.is_initialized = True

            logger.info(f"Session initialized: {self.session_id}")
            return dict(data)

        except requests.RequestException as e:
            raise PokemonGymAdapterError(f"Session initialization failed: {e}") from e

    def stop_session(self) -> dict[str, Any]:
        """Stop current session."""
        if not self.is_initialized:
            return {"status": "not_initialized"}

        try:
            response = self._session.post(
                f"{self.base_url}/stop", json={"session_id": self.session_id}, timeout=5
            )
            response.raise_for_status()

            data = response.json()
            self.is_initialized = False
            self.session_id = None

            logger.info("Session stopped successfully")
            return dict(data)

        except requests.RequestException as e:
            logger.warning(f"Session stop failed: {e}")
            # Reset state even if stop failed
            self.is_initialized = False
            self.session_id = None
            return {"status": "error", "error": str(e)}

    def reset_session(self) -> dict[str, Any]:
        """Reset session by stopping and reinitializing."""
        self.stop_session()
        return self.initialize_session()

    def close(self) -> None:
        """Close HTTP session resources."""
        if self.is_initialized:
            self.stop_session()
        self._session.close()


class PokemonGymAdapter:
    """
    Adapter implementing PokemonGymClient interface for benchflow-ai/pokemon-gym API.

    Translates batch input sequences to sequential API calls while maintaining
    the same interface as the original PokemonGymClient.

    Follows Clean Code principles:
    - Single Responsibility: Only handles API translation
    - Open/Closed: Extensible for new input types
    - Interface Segregation: Clean, focused interface
    - Dependency Inversion: Depends on abstractions (HTTP client)
    """

    def __init__(self, port: int, container_id: str, config: dict[str, Any] | None = None):
        """
        Initialize adapter for specific emulator instance.

        Args:
            port: HTTP port for emulator communication
            container_id: Docker container ID for this emulator
            config: Optional configuration for benchflow-ai session
        """
        self.port = port
        self.container_id = container_id
        self.base_url = f"http://localhost:{port}"
        self.config = config or {}

        # Performance configuration
        self.input_timeout = 10.0
        self.state_timeout = 5.0

        # Session management (Dependency Injection)
        self.session_manager = SessionManager(self.base_url, self.config)

        # HTTP client for direct requests
        self.session = requests.Session()

        logger.info(f"PokemonGymAdapter initialized for port {port}, container {container_id[:12]}")

    def send_input(self, input_sequence: str) -> dict[str, Any]:
        """
        Send input sequence to the emulator.

        Translates batch input string to sequential benchflow-ai action calls.

        Args:
            input_sequence: Button inputs (e.g., "A B START")

        Returns:
            Response data from emulator

        Raises:
            PokemonGymAdapterError: On communication failure
        """
        if not input_sequence.strip():
            return self._create_empty_input_response()

        try:
            self._ensure_session_initialized()
            buttons = self._parse_input_sequence(input_sequence)
            results = self._execute_button_sequence(buttons)
            return self._create_success_response(results)

        except Exception as e:
            raise PokemonGymAdapterError(
                f"Failed to send input to emulator on port {self.port}: {e}"
            ) from e

    def get_state(self) -> dict[str, Any]:
        """
        Get current game state from emulator.

        Maps benchflow-ai status response to our expected format.

        Returns:
            Current game state data

        Raises:
            PokemonGymAdapterError: On communication failure
        """
        try:
            # Ensure session is initialized for auto-initialization test
            self._ensure_session_initialized()

            response = self.session.get(f"{self.base_url}/status", timeout=self.state_timeout)
            response.raise_for_status()

            benchflow_data = response.json()
            return self._map_state_response(benchflow_data)

        except json.JSONDecodeError as e:
            raise PokemonGymAdapterError(
                f"Invalid response from emulator on port {self.port}: {e}"
            ) from e
        except Exception as e:
            # Handle timeout and other exceptions
            if "timeout" in str(e).lower():
                raise PokemonGymAdapterError(
                    f"Timeout accessing emulator on port {self.port}: {e}"
                ) from e
            raise PokemonGymAdapterError(
                f"Failed to get state from emulator on port {self.port}: {e}"
            ) from e

    def reset_game(self) -> dict[str, Any]:
        """
        Reset the game to initial state.

        Implements reset via stop/initialize sequence as required by benchflow-ai.

        Returns:
            Reset confirmation from emulator

        Raises:
            PokemonGymAdapterError: On communication failure
        """
        try:
            self.session_manager.reset_session()
            return {
                "status": "initialized",
                "session_id": self.session_manager.session_id,
                "message": "Game reset successfully",
            }
        except Exception as e:
            raise PokemonGymAdapterError(
                f"Failed to reset emulator on port {self.port}: {e}"
            ) from e

    def is_healthy(self) -> bool:
        """
        Check if emulator is responding and session is valid.

        Returns:
            True if emulator is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/status", timeout=3)

            if response.status_code != 200:
                return False

            data = response.json()
            # Check for session errors in response
            if "error" in data and "session" in data["error"].lower():
                return False

            return True

        except Exception:
            return False

    def close(self) -> None:
        """Close HTTP session and cleanup resources."""
        self.session_manager.close()
        self.session.close()

    def _ensure_session_initialized(self) -> None:
        """Ensure session is initialized, initialize if needed."""
        if not self.session_manager.is_initialized:
            self.session_manager.initialize_session()

    def _parse_input_sequence(self, input_sequence: str) -> list[str]:
        """
        Parse input sequence string into list of individual buttons.

        Args:
            input_sequence: Space-separated button sequence

        Returns:
            List of individual button names

        Raises:
            PokemonGymAdapterError: On invalid button names
        """
        # Normalize whitespace and split
        buttons = input_sequence.strip().upper().split()

        # Validate button names
        valid_buttons = {
            "A",
            "B",
            "START",
            "SELECT",
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "L",
            "R",  # Shoulder buttons for later Pokemon games
        }

        for button in buttons:
            if button not in valid_buttons:
                raise PokemonGymAdapterError(f"Invalid button name: {button}")

        return buttons

    def _send_single_action(self, button: str) -> dict[str, Any]:
        """
        Send single button press action to benchflow-ai API.

        Args:
            button: Single button name

        Returns:
            Action response data
        """
        response = self.session.post(
            f"{self.base_url}/action",
            json={"action_type": "press_key", "keys": [button]},
            timeout=self.input_timeout,
        )
        response.raise_for_status()
        return dict(response.json())

    def _map_state_response(self, benchflow_data: dict[str, Any]) -> dict[str, Any]:
        """
        Map benchflow-ai status response to our expected format.

        Implements defensive programming - handles missing/malformed data gracefully.

        Args:
            benchflow_data: Raw response from benchflow-ai /status

        Returns:
            Mapped state data in our expected format
        """
        # Defensive parsing with sensible defaults
        game_status = benchflow_data.get("game_status", "unknown")

        # Map player position data - return empty dict if no player data
        player_data = benchflow_data.get("player")
        if isinstance(player_data, dict) and player_data:
            player_position = {
                "x": player_data.get("x", 0),
                "y": player_data.get("y", 0),
                "map_id": player_data.get("map_id", "unknown"),
            }
        else:
            player_position = {}

        # Map screen data - return empty dict if no screen data
        screen_data = benchflow_data.get("screen")
        if isinstance(screen_data, dict) and screen_data:
            screen_info = {
                "tiles": screen_data.get("tiles", []),
                "width": screen_data.get("width", 20),
                "height": screen_data.get("height", 18),
            }
        else:
            screen_info = {}

        # Map frame count
        frame_count = benchflow_data.get("frame_count", 0)
        if not isinstance(frame_count, int):
            frame_count = 0

        return {
            "game_status": game_status,
            "player_position": player_position,
            "screen_data": screen_info,
            "frame_count": frame_count,
        }

    def _is_session_error(self, exception: Exception) -> bool:
        """
        Check if exception indicates session expiration.

        Args:
            exception: Exception to analyze

        Returns:
            True if this is a session-related error
        """
        error_str = str(exception).lower()
        session_indicators = ["session_expired", "session", "unauthorized", "401"]

        # Also check if it's a 400 error with session_expired in the response
        if hasattr(exception, "response") and exception.response is not None:
            try:
                response_data = exception.response.json()
                if isinstance(response_data, dict) and "session_expired" in response_data.get(
                    "error", ""
                ):
                    return True
            except (json.JSONDecodeError, AttributeError):
                pass

        return any(indicator in error_str for indicator in session_indicators)

    def _calculate_retry_delays(self, max_retries: int = 3) -> list[float]:
        """
        Calculate exponential backoff delays for retries.

        Args:
            max_retries: Maximum number of retries

        Returns:
            List of delay times in seconds
        """
        delays = []
        for i in range(max_retries):
            # Exponential backoff: 0.1, 0.2, 0.4 seconds
            delay = 0.1 * (2**i)
            delays.append(delay)
        return delays

    def _create_empty_input_response(self) -> dict[str, Any]:
        """Create response for empty input sequence."""
        return {"status": "no_input"}

    def _execute_button_sequence(self, buttons: list[str]) -> list[dict[str, Any]]:
        """
        Execute sequence of button actions with error recovery.

        Implements automatic session recovery when session expires.

        Args:
            buttons: List of button names to execute

        Returns:
            List of action results
        """
        results = []
        for button in buttons:
            result = self._execute_single_button_with_recovery(button)
            results.append(result)
        return results

    def _execute_single_button_with_recovery(self, button: str) -> dict[str, Any]:
        """
        Execute single button action with automatic session recovery.

        Args:
            button: Button name to execute

        Returns:
            Action result
        """
        try:
            return self._send_single_action(button)
        except requests.RequestException as e:
            if self._is_session_error(e):
                logger.info("Session expired, reinitializing...")
                self.session_manager.reset_session()
                return self._send_single_action(button)
            else:
                raise

    def _create_success_response(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Create formatted success response for input sequence."""
        return {"status": "success", "actions_completed": len(results), "results": results}

    def __str__(self) -> str:
        """String representation for logging and debugging."""
        return f"PokemonGymAdapter(port={self.port}, container={self.container_id})"


# Factory function for creating appropriate client types
def create_pokemon_gym_client(
    port: int,
    container_id: str,
    api_type: str = "benchflow",
    config: dict[str, Any] | None = None,
) -> Any:
    """
    Factory function to create appropriate Pokemon Gym client.

    Implements Factory Pattern to choose between adapter and direct client.

    Args:
        port: HTTP port for communication
        container_id: Docker container ID
        api_type: "benchflow" for adapter, "legacy" for direct client
        config: Optional configuration

    Returns:
        PokemonGymAdapter or PokemonGymClient instance
    """
    if api_type == "benchflow":
        return PokemonGymAdapter(port, container_id, config)
    elif api_type == "legacy":
        # Would return PokemonGymClient in real implementation
        # For now, return adapter as fallback
        return PokemonGymAdapter(port, container_id, config)
    else:
        raise ValueError(f"Unknown api_type: {api_type}")
