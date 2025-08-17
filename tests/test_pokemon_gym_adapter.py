"""
Test suite for PokemonGymAdapter compatibility layer.

Tests the adapter implementation with production patterns including
error handling, session management, and performance validation.

Author: Bot Dean - Production-First Engineering
"""

import time
from unittest.mock import Mock, patch

import pytest
from requests.exceptions import RequestException

from src.claudelearnspokemon.pokemon_gym_adapter import AdapterError, PokemonGymAdapter


class TestPokemonGymAdapterBasics:
    """Test basic adapter functionality and initialization."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.adapter = PokemonGymAdapter(port=8081, container_id="test_container_123")

    def test_initialization(self):
        """Test adapter initializes correctly."""
        assert self.adapter.port == 8081
        assert self.adapter.container_id == "test_container_123"
        assert self.adapter.base_url == "http://localhost:8081"
        assert self.adapter.input_delay == 0.05
        assert not self.adapter.initialized
        assert self.adapter.session_id is None

    def test_string_representation(self):
        """Test string representation provides useful debugging info."""
        str_repr = str(self.adapter)
        assert "PokemonGymAdapter" in str_repr
        assert "8081" in str_repr
        assert "test_container_123"[:12] in str_repr

    def test_custom_input_delay(self):
        """Test adapter accepts custom input delay configuration."""
        custom_adapter = PokemonGymAdapter(
            port=8082, container_id="test_container_456", input_delay=0.1
        )
        assert custom_adapter.input_delay == 0.1


class TestPokemonGymAdapterSessionManagement:
    """Test session lifecycle management with production patterns."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.adapter = PokemonGymAdapter(port=8081, container_id="test_container_123")
        self.mock_session = Mock()
        self.adapter.session = self.mock_session

    def test_session_initialization_success(self):
        """Test successful session initialization."""
        # Mock successful initialization response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"session_id": "test_session_123"}
        mock_response.raise_for_status.return_value = None
        self.mock_session.post.return_value = mock_response

        # Initialize session
        self.adapter._initialize_session()

        # Verify session state
        assert self.adapter.initialized
        assert self.adapter.session_id == "test_session_123"
        assert self.adapter.session_start is not None

        # Verify API call
        self.mock_session.post.assert_called_once_with(
            "http://localhost:8081/initialize",
            json={
                "headless": True,
                "sound": False,
                "save_video": False,
                "fast_video": True,
            },
            timeout=15,
        )

    def test_session_initialization_failure(self):
        """Test session initialization failure handling."""
        # Mock failed initialization response
        self.mock_session.post.side_effect = RequestException("Connection failed")

        # Verify initialization raises AdapterError
        with pytest.raises(AdapterError) as exc_info:
            self.adapter._initialize_session()

        assert "Failed to initialize session" in str(exc_info.value)
        assert not self.adapter.initialized
        assert self.adapter.session_id is None

    def test_session_expiration_detection(self):
        """Test session expiration detection logic."""
        # Set up expired session
        self.adapter.session_start = time.time() - 2000  # 33+ minutes ago

        assert self.adapter._session_expired()

        # Set up fresh session
        self.adapter.session_start = time.time() - 300  # 5 minutes ago

        assert not self.adapter._session_expired()

    def test_ensure_session_with_valid_session(self):
        """Test _ensure_session with valid active session."""
        # Set up valid session
        self.adapter.initialized = True
        self.adapter.session_start = time.time() - 300  # Fresh session

        # Should not try to initialize
        self.adapter._ensure_session()

        self.mock_session.post.assert_not_called()

    def test_ensure_session_with_expired_session(self):
        """Test _ensure_session reinitializes expired session."""
        # Set up expired session
        self.adapter.initialized = True
        self.adapter.session_start = time.time() - 2000  # Expired

        # Mock successful reinitialization
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"session_id": "new_session_456"}
        mock_response.raise_for_status.return_value = None
        self.mock_session.post.return_value = mock_response

        # Should reinitialize
        self.adapter._ensure_session()

        # Verify reinitialization occurred
        self.mock_session.post.assert_called_once()
        assert self.adapter.session_id == "new_session_456"


class TestPokemonGymAdapterInputTranslation:
    """Test input translation from batch strings to sequential API calls."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.adapter = PokemonGymAdapter(port=8081, container_id="test_container_123")
        self.mock_session = Mock()
        self.adapter.session = self.mock_session

        # Mock session as initialized to skip initialization
        self.adapter.initialized = True
        self.adapter.session_start = time.time()

    def test_single_button_input(self):
        """Test single button input translation."""
        # Mock successful action response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        self.mock_session.post.return_value = mock_response

        # Send single button input
        result = self.adapter.send_input("A")

        # Verify API call
        self.mock_session.post.assert_called_once_with(
            "http://localhost:8081/action",
            json={"action_type": "press_key", "keys": ["A"]},
            timeout=10,
        )

        # Verify result
        assert result == {"status": "success"}

    def test_multiple_button_input_sequence(self):
        """Test multiple button input sequence with timing."""
        # Mock successful action responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        self.mock_session.post.return_value = mock_response

        # Mock time.sleep to track timing
        with patch("time.sleep") as mock_sleep:
            _ = self.adapter.send_input("A B START")

        # Verify three API calls made
        assert self.mock_session.post.call_count == 3

        # Verify calls were made with correct data
        calls = self.mock_session.post.call_args_list
        assert calls[0][1]["json"] == {"action_type": "press_key", "keys": ["A"]}
        assert calls[1][1]["json"] == {"action_type": "press_key", "keys": ["B"]}
        assert calls[2][1]["json"] == {"action_type": "press_key", "keys": ["START"]}

        # Verify timing delays (should be called 2 times for 3 inputs)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(0.05)

    def test_empty_input_handling(self):
        """Test handling of empty input sequences."""
        result = self.adapter.send_input("")

        # Should not make any API calls
        self.mock_session.post.assert_not_called()

        # Should return success message
        assert result["status"] == "success"

    def test_invalid_button_handling(self):
        """Test handling of invalid button names."""
        with pytest.raises(AdapterError) as exc_info:
            self.adapter.send_input("A INVALID_BUTTON B")

        assert "Invalid button 'INVALID_BUTTON'" in str(exc_info.value)

    def test_button_validation(self):
        """Test button name validation."""
        valid_buttons = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R"]
        for button in valid_buttons:
            assert self.adapter._is_valid_button(button)

        invalid_buttons = ["X", "Y", "INVALID", "123", ""]
        for button in invalid_buttons:
            assert not self.adapter._is_valid_button(button)

    def test_input_case_normalization(self):
        """Test input button names are normalized to uppercase."""
        # Mock successful action response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        self.mock_session.post.return_value = mock_response

        # Send lowercase input
        self.adapter.send_input("a b start")

        # Verify calls were made with uppercase
        calls = self.mock_session.post.call_args_list
        assert calls[0][1]["json"]["keys"] == ["A"]
        assert calls[1][1]["json"]["keys"] == ["B"]
        assert calls[2][1]["json"]["keys"] == ["START"]


class TestPokemonGymAdapterStateMapping:
    """Test state mapping from benchflow-ai format to expected format."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.adapter = PokemonGymAdapter(port=8081, container_id="test_container_123")
        self.mock_session = Mock()
        self.adapter.session = self.mock_session

        # Mock session as initialized
        self.adapter.initialized = True
        self.adapter.session_start = time.time()
        self.adapter.session_id = "test_session_789"

    def test_state_retrieval_success(self):
        """Test successful state retrieval and mapping."""
        # Mock benchflow-ai status response
        benchflow_state = {
            "location": {"map": "pallet_town"},
            "coordinates": {"x": 10, "y": 15},
            "player_name": "Red",
            "money": 5000,
            "badges": ["boulder", "cascade"],
            "inventory": [{"item": "pokeball", "count": 5}],
            "party": [{"name": "pikachu", "level": 25}],
            "game_state": {"stage": "exploring"},
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = benchflow_state
        mock_response.raise_for_status.return_value = None
        self.mock_session.get.return_value = mock_response

        # Get state
        result = self.adapter.get_state()

        # Verify API call
        self.mock_session.get.assert_called_once_with("http://localhost:8081/status", timeout=5)

        # Verify mapped structure
        assert result["location"] == {"map": "pallet_town"}
        assert result["coordinates"] == {"x": 10, "y": 15}
        assert result["player"]["name"] == "Red"
        assert result["player"]["money"] == 5000
        assert result["player"]["badges"] == ["boulder", "cascade"]
        assert result["inventory"] == [{"item": "pokeball", "count": 5}]
        assert result["pokemon"] == [{"name": "pikachu", "level": 25}]

        # Verify metadata included
        assert result["_adapter_info"]["session_id"] == "test_session_789"
        assert result["_adapter_info"]["adapter_type"] == "benchflow-ai"
        assert result["_original"] == benchflow_state

    def test_state_mapping_with_missing_fields(self):
        """Test state mapping handles missing fields gracefully."""
        # Minimal benchflow-ai response
        minimal_state = {"location": {"map": "unknown"}}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = minimal_state
        mock_response.raise_for_status.return_value = None
        self.mock_session.get.return_value = mock_response

        # Get state
        result = self.adapter.get_state()

        # Verify defaults are provided
        assert result["player"]["name"] == "Red"  # Default
        assert result["player"]["money"] == 0  # Default
        assert result["inventory"] == []  # Default
        assert result["pokemon"] == []  # Default

    def test_state_retrieval_failure(self):
        """Test state retrieval failure handling."""
        # Mock failed response
        self.mock_session.get.side_effect = RequestException("Connection failed")

        # Should raise AdapterError
        with pytest.raises(AdapterError) as exc_info:
            self.adapter.get_state()

        assert "Failed to get state" in str(exc_info.value)


class TestPokemonGymAdapterResetFunctionality:
    """Test game reset functionality via stop/initialize sequence."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.adapter = PokemonGymAdapter(port=8081, container_id="test_container_123")
        self.mock_session = Mock()
        self.adapter.session = self.mock_session

    def test_reset_with_active_session(self):
        """Test reset with active session performs stop/initialize."""
        # Set up active session
        self.adapter.initialized = True
        self.adapter.session_id = "old_session"
        self.adapter.session_start = time.time() - 300

        # Mock stop response
        mock_stop_response = Mock()
        mock_stop_response.status_code = 200
        mock_stop_response.raise_for_status.return_value = None

        # Mock initialize response
        mock_init_response = Mock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {"session_id": "new_session"}
        mock_init_response.raise_for_status.return_value = None

        # Configure mock to return different responses
        self.mock_session.post.side_effect = [mock_stop_response, mock_init_response]

        # Reset game
        result = self.adapter.reset_game()

        # Verify stop and initialize calls
        assert self.mock_session.post.call_count == 2
        calls = self.mock_session.post.call_args_list

        # First call should be stop
        assert calls[0][0][0] == "http://localhost:8081/stop"

        # Second call should be initialize
        assert calls[1][0][0] == "http://localhost:8081/initialize"

        # Verify session was reset and reinitialized
        assert self.adapter.session_id == "new_session"
        assert self.adapter.initialized
        assert result["status"] == "success"

    def test_reset_without_active_session(self):
        """Test reset without active session just initializes."""
        # No active session
        self.adapter.initialized = False

        # Mock initialize response
        mock_init_response = Mock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {"session_id": "new_session"}
        mock_init_response.raise_for_status.return_value = None
        self.mock_session.post.return_value = mock_init_response

        # Reset game
        _ = self.adapter.reset_game()

        # Should only call initialize
        assert self.mock_session.post.call_count == 1
        call = self.mock_session.post.call_args_list[0]
        assert call[0][0] == "http://localhost:8081/initialize"

        # Verify session initialized
        assert self.adapter.session_id == "new_session"
        assert self.adapter.initialized


class TestPokemonGymAdapterHealthChecking:
    """Test health checking functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.adapter = PokemonGymAdapter(port=8081, container_id="test_container_123")
        self.mock_session = Mock()
        self.adapter.session = self.mock_session

    def test_health_check_success(self):
        """Test successful health check."""
        # Mock healthy status response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"session_active": True}
        self.mock_session.get.return_value = mock_response

        # Check health
        assert self.adapter.is_healthy()

        # Verify API call
        self.mock_session.get.assert_called_once_with("http://localhost:8081/status", timeout=3)

    def test_health_check_inactive_session(self):
        """Test health check with inactive session."""
        # Mock inactive session response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"session_active": False}
        self.mock_session.get.return_value = mock_response

        # Should be unhealthy
        assert not self.adapter.is_healthy()

    def test_health_check_network_failure(self):
        """Test health check with network failure."""
        # Mock network failure
        self.mock_session.get.side_effect = RequestException("Connection failed")

        # Should be unhealthy
        assert not self.adapter.is_healthy()


class TestPokemonGymAdapterCircuitBreaker:
    """Test circuit breaker pattern for production resilience."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.adapter = PokemonGymAdapter(port=8081, container_id="test_container_123")
        self.mock_session = Mock()
        self.adapter.session = self.mock_session

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after consecutive failures."""
        # Mock failures
        self.mock_session.post.side_effect = RequestException("Connection failed")

        # Trigger multiple failures to open circuit breaker
        for _ in range(3):
            try:
                self.adapter._initialize_session()
            except AdapterError:
                pass

        # Circuit should be open
        assert self.adapter.circuit_open
        assert self.adapter.consecutive_failures >= 3

    def test_circuit_breaker_timeout_functionality(self):
        """Test circuit breaker respects timeout."""
        # Force circuit open
        self.adapter.circuit_open = True
        self.adapter.last_failure_time = time.time() - 10  # 10 seconds ago
        self.adapter.circuit_timeout = 5  # 5 second timeout

        # Mock successful response for when circuit tries to close
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        self.mock_session.get.return_value = mock_response

        # Health check should try to close circuit after timeout
        result = self.adapter.is_healthy()

        # Circuit should be closed and health check should succeed
        assert not self.adapter.circuit_open
        assert result


class TestPokemonGymAdapterResourceCleanup:
    """Test proper resource cleanup and session management."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.adapter = PokemonGymAdapter(port=8081, container_id="test_container_123")
        self.mock_session = Mock()
        self.adapter.session = self.mock_session

    def test_close_with_active_session(self):
        """Test closing adapter with active session."""
        # Set up active session
        self.adapter.initialized = True
        self.adapter.session_id = "test_session"

        # Mock successful stop response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        self.mock_session.post.return_value = mock_response

        # Close adapter
        self.adapter.close()

        # Verify stop was called
        self.mock_session.post.assert_called_once_with("http://localhost:8081/stop", timeout=5)

        # Verify session was closed
        self.mock_session.close.assert_called_once()

        # Verify state was cleared
        assert not self.adapter.initialized
        assert self.adapter.session_id is None

    def test_close_handles_stop_failure_gracefully(self):
        """Test close handles stop failure gracefully."""
        # Set up active session
        self.adapter.initialized = True

        # Mock stop failure
        self.mock_session.post.side_effect = RequestException("Stop failed")

        # Should not raise exception
        self.adapter.close()

        # Session should still be closed
        self.mock_session.close.assert_called_once()
        assert not self.adapter.initialized
