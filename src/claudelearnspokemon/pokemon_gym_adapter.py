"""
PokemonGymAdapter: Adapter pattern implementation for benchflow-ai/pokemon-gym server integration.

Bridges the gap between our EmulatorPool API expectations and the benchflow-ai API format.
Implements production patterns for session management, error recovery, and observability.

Author: Bot Dean - Production-First Engineering
"""

import logging
import time
from typing import Any

import requests
from requests.exceptions import RequestException, Timeout

# Avoid circular import - AdapterError inherits from Exception directly

# Configure logging for production observability
logger = logging.getLogger(__name__)


class AdapterError(Exception):
    """
    Custom exception for adapter operations.

    Provides actionable error messages for production debugging.
    """

    pass


class PokemonGymAdapter:
    """
    Adapter for benchflow-ai/pokemon-gym server integration.

    Implements the same interface as PokemonGymClient but translates calls
    to the benchflow-ai API format. Provides production features:
    - Session lifecycle management with auto-initialization
    - Input translation from batch strings to sequential API calls
    - State mapping from benchflow-ai format to expected structure
    - Comprehensive error handling and recovery
    - Circuit breaker pattern for failed operations
    """

    def __init__(self, port: int, container_id: str, input_delay: float = 0.05):
        """
        Initialize adapter for benchflow-ai server communication.

        Args:
            port: HTTP port for server communication
            container_id: Docker container ID for this emulator
            input_delay: Delay between sequential key presses (default: 50ms)
        """
        self.port = port
        self.container_id = container_id
        self.base_url = f"http://localhost:{port}"
        self.session = requests.Session()
        self.input_delay = input_delay

        # Session management state
        self.session_id: str | None = None
        self.initialized = False
        self.session_start: float | None = None

        # Circuit breaker state for production resilience
        self.consecutive_failures = 0
        self.circuit_open = False
        self.last_failure_time: float | None = None
        self.circuit_timeout = 30.0  # 30s circuit breaker timeout

        # Session configuration for benchflow-ai
        self.session_config = {
            "headless": True,
            "sound": False,
            "save_video": False,
            "fast_video": True,
        }

        logger.info(
            f"PokemonGymAdapter initialized for port {port}, container {container_id[:12]}, "
            f"input_delay={input_delay}s"
        )

    def send_input(self, input_sequence: str) -> dict[str, Any]:
        """
        Send input sequence to the emulator via sequential API calls.

        Translates batch input strings like "A B START" into sequential
        POST /action calls with proper timing between inputs.

        Args:
            input_sequence: Button inputs (A, B, START, etc.) separated by spaces

        Returns:
            Response data from the last action call

        Raises:
            AdapterError: On communication failure or invalid input
        """
        self._ensure_session()

        if not input_sequence or not input_sequence.strip():
            logger.warning("Empty input sequence provided")
            return {"status": "success", "message": "No input to process"}

        buttons = input_sequence.strip().split()
        logger.info(f"Translating input sequence: {buttons} on {self}")

        responses = []

        try:
            for i, button in enumerate(buttons):
                if not self._is_valid_button(button):
                    raise AdapterError(f"Invalid button '{button}' in sequence: {input_sequence}")

                # Send individual key press
                response = self.session.post(
                    f"{self.base_url}/action",
                    json={
                        "action_type": "press_key",
                        "keys": [button.upper()],  # Normalize to uppercase
                    },
                    timeout=10,
                )
                response.raise_for_status()

                response_data = response.json()
                responses.append(response_data)

                # Add delay between inputs (except after last input)
                if i < len(buttons) - 1:
                    time.sleep(self.input_delay)

            # Reset circuit breaker on success
            self.consecutive_failures = 0
            self.circuit_open = False

            # Return the last response as the overall result
            return responses[-1] if responses else {"status": "success"}

        except (RequestException, Timeout) as e:
            self._handle_request_failure(e)
            raise AdapterError(
                f"Failed to send input sequence '{input_sequence}' to adapter on port {self.port}: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during input translation: {e}")
            raise AdapterError(f"Input translation failed: {e}") from e

    def get_state(self) -> dict[str, Any]:
        """
        Get current game state from emulator.

        Maps the benchflow-ai /status response to our expected state format
        for backward compatibility with existing EmulatorPool code.

        Returns:
            Current game state data in compatible format

        Raises:
            AdapterError: On communication failure
        """
        self._ensure_session()

        try:
            response = self.session.get(f"{self.base_url}/status", timeout=5)
            response.raise_for_status()

            benchflow_state = response.json()

            # Map benchflow-ai response format to our expected structure
            mapped_state = self._map_state_format(benchflow_state)

            # Reset circuit breaker on success
            self.consecutive_failures = 0
            self.circuit_open = False

            return mapped_state

        except (RequestException, Timeout) as e:
            self._handle_request_failure(e)
            raise AdapterError(f"Failed to get state from adapter on port {self.port}: {e}") from e

    def reset_game(self) -> dict[str, Any]:
        """
        Reset the game to initial state.

        Implements reset via stop/initialize sequence since benchflow-ai
        doesn't have a direct reset endpoint.

        Returns:
            Reset confirmation from emulator

        Raises:
            AdapterError: On communication failure
        """
        logger.info(f"Resetting game via stop/initialize sequence on {self}")

        try:
            # Stop current session if active
            if self.initialized:
                try:
                    stop_response = self.session.post(f"{self.base_url}/stop", timeout=10)
                    stop_response.raise_for_status()
                    logger.info("Successfully stopped current session for reset")
                except Exception as e:
                    logger.warning(f"Failed to stop session during reset: {e}")
                    # Continue with reset anyway

            # Clear session state
            self.session_id = None
            self.initialized = False
            self.session_start = None

            # Initialize new session
            self._initialize_session()

            # Reset circuit breaker on success
            self.consecutive_failures = 0
            self.circuit_open = False

            return {"status": "success", "message": "Game reset completed"}

        except Exception as e:
            self._handle_request_failure(e)
            raise AdapterError(f"Failed to reset emulator on port {self.port}: {e}") from e

    def is_healthy(self) -> bool:
        """
        Check if emulator is responding to health checks.

        Uses the /status endpoint to verify both server connectivity
        and session validity.

        Returns:
            True if emulator is healthy, False otherwise
        """
        try:
            # Check circuit breaker state
            if self.circuit_open:
                if time.time() - (self.last_failure_time or 0) < self.circuit_timeout:
                    return False
                else:
                    # Try to close circuit breaker
                    self.circuit_open = False
                    logger.info("Circuit breaker timeout expired, attempting health check")

            response = self.session.get(f"{self.base_url}/status", timeout=3)

            if response.status_code != 200:
                return False

            status_data = response.json()

            # Check if session is active (basic validation)
            session_active = status_data.get(
                "session_active", True
            )  # Default to True if not specified

            # Reset circuit breaker on successful health check
            self.consecutive_failures = 0
            self.circuit_open = False

            return session_active

        except Exception as e:
            logger.debug(f"Health check failed for adapter on port {self.port}: {e}")
            return False

    def close(self) -> None:
        """Close the adapter and clean up resources."""
        try:
            # Stop benchflow-ai session if active
            if self.initialized:
                try:
                    response = self.session.post(f"{self.base_url}/stop", timeout=5)
                    response.raise_for_status()
                    logger.info(f"Successfully stopped session for {self}")
                except Exception as e:
                    logger.warning(f"Failed to stop session during close: {e}")

            # Close HTTP session
            self.session.close()

            # Clear state
            self.session_id = None
            self.initialized = False
            self.session_start = None

        except Exception as e:
            logger.error(f"Error during adapter close: {e}")

    def __str__(self) -> str:
        return f"PokemonGymAdapter(port={self.port}, container={self.container_id[:12]}, session={self.session_id or 'none'})"

    # Private helper methods for production robustness

    def _ensure_session(self) -> None:
        """
        Ensure a valid session is active, initializing if necessary.

        Implements circuit breaker pattern to avoid thundering herd
        on repeated session failures.
        """
        # Check circuit breaker
        if self.circuit_open:
            if time.time() - (self.last_failure_time or 0) < self.circuit_timeout:
                raise AdapterError(
                    f"Circuit breaker open for adapter on port {self.port}. "
                    f"Try again after {self.circuit_timeout}s timeout."
                )
            else:
                self.circuit_open = False
                logger.info("Circuit breaker timeout expired, retrying session initialization")

        if not self.initialized or self._session_expired():
            self._initialize_session()

    def _initialize_session(self) -> None:
        """
        Initialize a new benchflow-ai session.

        Raises:
            AdapterError: On initialization failure
        """
        try:
            logger.info(f"Initializing benchflow-ai session on port {self.port}")

            response = self.session.post(
                f"{self.base_url}/initialize",
                json=self.session_config,
                timeout=15,  # Longer timeout for initialization
            )
            response.raise_for_status()

            init_data = response.json()
            self.session_id = init_data.get("session_id", f"session_{int(time.time())}")
            self.initialized = True
            self.session_start = time.time()

            logger.info(f"Session initialized successfully: {self.session_id} on port {self.port}")

        except (RequestException, Timeout) as e:
            self.initialized = False
            self.session_id = None
            self._handle_request_failure(e)
            raise AdapterError(f"Failed to initialize session on port {self.port}: {e}") from e

    def _session_expired(self) -> bool:
        """
        Check if the current session has expired.

        Uses simple timeout-based expiration (benchflow-ai sessions typically
        last 30 minutes but we'll check after 25 minutes for safety).
        """
        if not self.session_start:
            return True

        session_age = time.time() - self.session_start
        max_session_age = 1500  # 25 minutes in seconds

        return session_age > max_session_age

    def _map_state_format(self, benchflow_state: dict[str, Any]) -> dict[str, Any]:
        """
        Map benchflow-ai state format to our expected structure.

        Args:
            benchflow_state: Raw response from benchflow-ai /status endpoint

        Returns:
            State data in format compatible with existing EmulatorPool code
        """
        # Provide defensive mapping with fallbacks
        return {
            "location": benchflow_state.get("location", {}),
            "coordinates": benchflow_state.get("coordinates", {"x": 0, "y": 0}),
            "player": {
                "name": benchflow_state.get("player_name", "Red"),
                "money": benchflow_state.get("money", 0),
                "badges": benchflow_state.get("badges", []),
            },
            "inventory": benchflow_state.get("inventory", []),
            "pokemon": benchflow_state.get("party", []),
            "game_state": benchflow_state.get("game_state", {}),
            # Include original data for debugging
            "_original": benchflow_state,
            "_adapter_info": {
                "session_id": self.session_id,
                "session_age": time.time() - (self.session_start or time.time()),
                "adapter_type": "benchflow-ai",
            },
        }

    def _is_valid_button(self, button: str) -> bool:
        """
        Validate if button name is supported by benchflow-ai.

        Args:
            button: Button name to validate

        Returns:
            True if button is valid, False otherwise
        """
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
            "R",  # Shoulder buttons if supported
        }

        return button.upper() in valid_buttons

    def _handle_request_failure(self, error: Exception) -> None:
        """
        Handle request failures with circuit breaker pattern.

        Args:
            error: The exception that occurred
        """
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

        # Open circuit breaker after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.circuit_open = True
            logger.warning(
                f"Circuit breaker opened for adapter on port {self.port} "
                f"after {self.consecutive_failures} consecutive failures"
            )

        # Mark session as potentially invalid
        if isinstance(error, RequestException | Timeout):
            # Network errors might indicate session problems
            if self.consecutive_failures >= 2:
                self.initialized = False
                self.session_id = None
                logger.info("Marked session as invalid due to repeated failures")
