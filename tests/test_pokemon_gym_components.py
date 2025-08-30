"""
Tests for the refactored Pokemon Gym Adapter components.

Tests the extracted components for better separation of concerns:
- PokemonGymClient: HTTP client management
- ErrorRecoveryHandler: Error recovery operations  
- PerformanceMonitor: Performance tracking
- InputValidator: Input validation and parsing

Author: Claude Code - Refactoring Implementation
"""

import json
import pytest
import requests
import responses
from unittest.mock import Mock, patch

from claudelearnspokemon.pokemon_gym_adapter import (
    PokemonGymClient,
    ErrorRecoveryHandler,
    PerformanceMonitor,
    InputValidator,
    PokemonGymAdapterError,
    SessionManager
)


@pytest.mark.fast
class TestPokemonGymClient:
    """Test the extracted HTTP client component."""

    def test_client_initialization(self):
        """Test PokemonGymClient initializes correctly."""
        client = PokemonGymClient("http://localhost:8080")
        
        assert client.base_url == "http://localhost:8080"
        assert client.timeout_config["action"] == 1.0
        assert client.timeout_config["initialization"] == 3.0
        assert client.timeout_config["status"] == 1.0
        
    def test_client_custom_config(self):
        """Test client accepts custom timeout configuration."""
        timeout_config = {"action": 0.5, "status": 0.2}
        client = PokemonGymClient("http://localhost:8080", timeout_config=timeout_config)
        
        assert client.timeout_config["action"] == 0.5
        assert client.timeout_config["status"] == 0.2
        
    @responses.activate
    def test_post_action(self):
        """Test post_action method makes correct HTTP call."""
        responses.add(
            responses.POST,
            "http://localhost:8080/action",
            json={"status": "success", "frame_count": 1},
            status=200
        )
        
        client = PokemonGymClient("http://localhost:8080")
        result = client.post_action("A")
        
        assert result["status"] == "success"
        assert len(responses.calls) == 1
        call = responses.calls[0]
        import json
        request_data = json.loads(call.request.body)
        assert request_data == {"action_type": "press_key", "keys": ["A"]}
        
    @responses.activate
    def test_get_status(self):
        """Test get_status method retrieves game state."""
        status_data = {"game_status": "running", "frame_count": 100}
        responses.add(
            responses.GET,
            "http://localhost:8080/status", 
            json=status_data,
            status=200
        )
        
        client = PokemonGymClient("http://localhost:8080")
        result = client.get_status()
        
        assert result == status_data
        
    @responses.activate  
    def test_post_initialize(self):
        """Test post_initialize method starts session."""
        config = {"headless": True}
        init_response = {"session_id": "test-123", "status": "initialized"}
        responses.add(
            responses.POST,
            "http://localhost:8080/initialize",
            json=init_response,
            status=200
        )
        
        client = PokemonGymClient("http://localhost:8080")
        result = client.post_initialize(config)
        
        assert result == init_response
        call = responses.calls[0]
        import json
        request_data = json.loads(call.request.body)
        assert request_data == {"config": config}


@pytest.mark.fast  
class TestErrorRecoveryHandler:
    """Test the extracted error recovery component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_gym_client = Mock(spec=PokemonGymClient)
        self.mock_session_manager = Mock(spec=SessionManager)
        self.handler = ErrorRecoveryHandler(self.mock_gym_client, self.mock_session_manager)
        
    def test_is_session_error_detects_session_expired(self):
        """Test session error detection for expired sessions."""
        error = requests.HTTPError("400 Client Error")
        error.response = Mock()
        error.response.json.return_value = {"error": "session_expired"}
        
        assert self.handler.is_session_error(error) is True
        
    def test_is_session_error_detects_unauthorized(self):
        """Test session error detection for unauthorized errors."""
        error = requests.HTTPError("401 Unauthorized")
        
        assert self.handler.is_session_error(error) is True
        
    def test_is_session_error_ignores_other_errors(self):
        """Test session error detection ignores non-session errors."""
        error = requests.HTTPError("500 Internal Server Error")
        
        assert self.handler.is_session_error(error) is False
        
    def test_calculate_retry_delays_exponential_backoff(self):
        """Test retry delay calculation uses exponential backoff."""
        delays = self.handler.calculate_retry_delays(max_retries=3)
        
        assert len(delays) == 3
        assert delays[0] == 0.1  # 100ms
        assert delays[1] == 0.2  # 200ms  
        assert delays[2] == 0.4  # 400ms
        
    def test_emergency_session_recovery(self):
        """Test emergency session recovery flow."""
        self.mock_gym_client.post_initialize.return_value = {"session_id": "recovery-123"}
        
        # Set initial state
        self.mock_session_manager.is_initialized = True
        self.mock_session_manager.session_id = "old-session"
        
        self.handler.emergency_session_recovery()
        
        # Should initialize with minimal config
        self.mock_gym_client.post_initialize.assert_called_once()
        call_args = self.mock_gym_client.post_initialize.call_args
        config = call_args[0][0]  # First positional argument
        assert config == {"headless": True, "sound": False}
        
        # Should set new session info
        assert self.mock_session_manager.session_id == "recovery-123"
        assert self.mock_session_manager.is_initialized is True
        
    def test_force_clean_state(self):
        """Test force clean state resets session tracking."""
        self.mock_session_manager.is_initialized = True
        self.mock_session_manager.session_id = "test-session"
        self.mock_session_manager._reset_in_progress = True
        
        self.handler.force_clean_state()
        
        assert self.mock_session_manager.is_initialized is False
        assert self.mock_session_manager.session_id is None
        assert self.mock_session_manager._reset_in_progress is False


@pytest.mark.fast
class TestPerformanceMonitor:
    """Test the extracted performance monitoring component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
        
    def test_track_operation_time(self):
        """Test operation time tracking."""
        self.monitor.track_operation_time("reset_game", 250.5)
        self.monitor.track_operation_time("reset_game", 180.0)
        
        stats = self.monitor.get_performance_stats("reset_game")
        assert stats["count"] == 2
        assert stats["avg_ms"] == 215.25
        assert stats["min_ms"] == 180.0
        assert stats["max_ms"] == 250.5
        
    def test_validate_performance_sla_within_limit(self):
        """Test SLA validation when performance is within limit."""
        result = self.monitor.validate_performance_sla("reset_game", 400.0, 500.0)
        
        assert result["operation"] == "reset_game"
        assert result["duration_ms"] == 400.0
        assert result["sla_ms"] == 500.0
        assert result["sla_exceeded"] is False
        assert result["performance_warning"] is False
        
    def test_validate_performance_sla_exceeds_limit(self):
        """Test SLA validation when performance exceeds limit."""  
        result = self.monitor.validate_performance_sla("reset_game", 600.0, 500.0)
        
        assert result["operation"] == "reset_game"
        assert result["duration_ms"] == 600.0
        assert result["sla_ms"] == 500.0
        assert result["sla_exceeded"] is True
        assert result["performance_warning"] is True
        
    def test_get_performance_stats_empty(self):
        """Test performance stats for non-existent operation."""
        stats = self.monitor.get_performance_stats("non_existent")
        assert stats == {"count": 0}


@pytest.mark.fast
class TestInputValidator:
    """Test the extracted input validation component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()
        
    def test_parse_input_sequence_valid_buttons(self):
        """Test parsing valid button sequence."""
        result = self.validator.parse_input_sequence("A B START UP DOWN")
        assert result == ["A", "B", "START", "UP", "DOWN"]
        
    def test_parse_input_sequence_case_insensitive(self):
        """Test parsing is case insensitive."""
        result = self.validator.parse_input_sequence("a b start up down")
        assert result == ["A", "B", "START", "UP", "DOWN"]
        
    def test_parse_input_sequence_whitespace_normalization(self):
        """Test input parsing handles extra whitespace."""
        result = self.validator.parse_input_sequence("  A   B  START  ")
        assert result == ["A", "B", "START"]
        
    def test_parse_input_sequence_test_compatibility(self):
        """Test parsing handles test action patterns."""
        result = self.validator.parse_input_sequence("THREAD_0_ACTION_0 A THREAD_1_ACTION_1")
        assert result == ["A", "A", "A"]  # Test actions mapped to A button
        
    def test_parse_input_sequence_invalid_button(self):
        """Test parsing rejects invalid button names."""
        with pytest.raises(PokemonGymAdapterError) as exc_info:
            self.validator.parse_input_sequence("A INVALID_BUTTON B")
        assert "invalid button" in str(exc_info.value).lower()
        
    def test_create_empty_input_response(self):
        """Test empty input response creation."""
        result = self.validator.create_empty_input_response()
        assert result == {"status": "no_input"}
        
    def test_create_success_response(self):
        """Test success response creation."""
        action_results = [{"status": "success"}, {"status": "success"}]
        result = self.validator.create_success_response(action_results)
        
        assert result["status"] == "success"
        assert result["actions_completed"] == 2
        assert result["results"] == action_results


@pytest.mark.medium
class TestComponentIntegration:
    """Test integration between the extracted components."""
    
    def test_error_recovery_with_gym_client_integration(self):
        """Test error recovery handler works with gym client integration scenarios."""
        # Set up real integration between ErrorRecoveryHandler and PokemonGymClient
        gym_client = PokemonGymClient("http://localhost:8080")
        session_manager = Mock(spec=SessionManager)
        session_manager.is_initialized = True
        session_manager.session_id = "test-session-123"
        
        handler = ErrorRecoveryHandler(gym_client, session_manager)
        
        # Test scenario 1: Session error detection and recovery with actual gym client
        with patch.object(gym_client, 'post_action') as mock_post_action, \
             patch.object(gym_client, 'post_initialize') as mock_post_init:
            
            # Simulate session expired error from gym client
            session_error = requests.HTTPError("401 Unauthorized") 
            session_error.response = Mock()
            session_error.response.json.return_value = {"error": "session_expired", "code": 401}
            
            mock_post_action.side_effect = session_error
            mock_post_init.return_value = {"session_id": "recovery-session-456", "status": "initialized"}
            
            # Test that handler correctly identifies session error
            assert handler.is_session_error(session_error) is True
            
            # Test emergency recovery with actual gym client interaction
            handler.emergency_session_recovery()
            
            # Verify gym client was called for recovery initialization
            mock_post_init.assert_called_once()
            recovery_config = mock_post_init.call_args[0][0]
            assert recovery_config == {"headless": True, "sound": False}
            
            # Verify session manager state was updated
            assert session_manager.session_id == "recovery-session-456"
            assert session_manager.is_initialized is True
            
        # Test scenario 2: Retry delay calculation integration
        delays = handler.calculate_retry_delays(max_retries=3)
        assert len(delays) == 3
        assert delays == [0.1, 0.2, 0.4]  # Exponential backoff
        
        # Test scenario 3: Force clean state integration with session manager
        session_manager.is_initialized = True
        session_manager.session_id = "active-session"
        session_manager._reset_in_progress = True
        
        handler.force_clean_state()
        
        assert session_manager.is_initialized is False
        assert session_manager.session_id is None
        assert session_manager._reset_in_progress is False
        
        # Test scenario 4: Integration with non-session errors (should not trigger recovery)
        non_session_error = requests.HTTPError("500 Internal Server Error")
        assert handler.is_session_error(non_session_error) is False
        
        # Test scenario 5: Emergency recovery failure handling
        with patch.object(gym_client, 'post_initialize') as mock_post_init:
            mock_post_init.side_effect = requests.ConnectionError("Network unreachable")
            
            with pytest.raises(PokemonGymAdapterError) as exc_info:
                handler.emergency_session_recovery()
            
            assert "Emergency recovery failed" in str(exc_info.value)
            assert "Network unreachable" in str(exc_info.value)
            
        # Test scenario 6: Session error detection with malformed response
        malformed_error = requests.HTTPError("400 Bad Request")
        malformed_error.response = Mock()
        malformed_error.response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        # Should handle malformed response gracefully and fall back to string analysis
        result = handler.is_session_error(malformed_error)
        assert result is False  # "400 Bad Request" doesn't contain session indicators
        
    def test_performance_monitor_with_real_operations(self):
        """Test performance monitor tracking real operations."""
        monitor = PerformanceMonitor()
        
        # Simulate multiple operations
        for i in range(5):
            duration = 100.0 + (i * 50)  # 100, 150, 200, 250, 300
            monitor.track_operation_time("send_input", duration)
            
        stats = monitor.get_performance_stats("send_input")
        assert stats["count"] == 5
        assert stats["avg_ms"] == 200.0
        assert stats["min_ms"] == 100.0
        assert stats["max_ms"] == 300.0


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v"])