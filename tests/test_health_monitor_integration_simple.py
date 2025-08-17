"""
Simple integration tests for HealthMonitor with EmulatorPool interface.

Focuses on core integration points without complex dependencies.

Author: John Botmack - Performance-First Integration Testing
"""

import unittest
from unittest.mock import Mock, patch

from src.claudelearnspokemon.emulator_pool import PokemonGymClient
from src.claudelearnspokemon.health_monitor import HealthMonitor


class TestHealthMonitorIntegration(unittest.TestCase):
    """Test HealthMonitor integration with EmulatorPool interface."""

    def setUp(self):
        """Set up integration test fixtures."""
        # Create mock EmulatorPool that mimics the real one's interface
        self.mock_emulator_pool = Mock()

        # Create real PokemonGymClient instances for realistic testing
        client1 = PokemonGymClient(8081, "container1-abc123")
        client2 = PokemonGymClient(8082, "container2-def456")

        # Set up EmulatorPool interface that HealthMonitor expects
        self.mock_emulator_pool.clients_by_port = {8081: client1, 8082: client2}
        self.mock_emulator_pool.get_status.return_value = {
            "available_count": 2,
            "busy_count": 0,
            "total_count": 2,
            "status": "healthy",
        }

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "health_monitor"):
            self.health_monitor.stop()

    def test_health_monitor_interface_compatibility(self):
        """Test HealthMonitor works with EmulatorPool interface."""
        # Create HealthMonitor with mock EmulatorPool
        self.health_monitor = HealthMonitor(
            emulator_pool=self.mock_emulator_pool, check_interval=30.0, health_timeout=3.0
        )

        # Verify integration attributes
        assert self.health_monitor.emulator_pool == self.mock_emulator_pool
        assert self.health_monitor.check_interval == 30.0
        assert self.health_monitor.health_timeout == 3.0
        assert not self.health_monitor._running

    @patch("requests.Session")
    def test_force_check_integration(self, mock_session_class):
        """Test force_check method integrates correctly."""
        # Mock successful HTTP responses
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        self.health_monitor = HealthMonitor(
            emulator_pool=self.mock_emulator_pool, check_interval=30.0
        )

        # Perform health check
        result = self.health_monitor.force_check()

        # Verify integration with EmulatorPool
        self.mock_emulator_pool.get_status.assert_called_once()

        # Verify result structure
        assert result["overall_status"] == "healthy"
        assert result["healthy_count"] == 2
        assert result["total_count"] == 2
        assert "pool_status" in result
        assert "emulators" in result

        # Verify pool_status integration
        pool_status = result["pool_status"]
        assert pool_status["available_count"] == 2
        assert pool_status["total_count"] == 2
        assert pool_status["status"] == "healthy"

        # Verify HTTP calls were made to correct ports
        expected_calls = 2  # One for each emulator
        assert mock_session.get.call_count == expected_calls

    def test_clients_by_port_structure(self):
        """Test that EmulatorPool interface provides expected structure."""
        # Verify the structure that HealthMonitor expects
        assert hasattr(self.mock_emulator_pool, "clients_by_port")
        assert isinstance(self.mock_emulator_pool.clients_by_port, dict)

        for port, client in self.mock_emulator_pool.clients_by_port.items():
            assert isinstance(port, int)
            assert hasattr(client, "port")
            assert hasattr(client, "container_id")
            assert client.port == port

    @patch("requests.Session")
    def test_health_status_change_detection(self, mock_session_class):
        """Test health status change detection works with integration."""
        health_changes = []

        def track_changes(port, old_healthy, new_healthy):
            health_changes.append((port, old_healthy, new_healthy))

        self.health_monitor = HealthMonitor(
            emulator_pool=self.mock_emulator_pool,
            check_interval=30.0,
            on_health_change=track_changes,
        )

        # Mock session for changing responses
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # First check - all healthy
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        result1 = self.health_monitor.force_check()
        assert result1["healthy_count"] == 2
        assert len(health_changes) == 0  # No changes on first check

        # Second check - one emulator fails
        def failing_get(url, timeout):
            if "8081" in url:
                response = Mock()
                response.status_code = 200
                return response
            else:
                import requests

                raise requests.RequestException("Connection failed")

        mock_session.get.side_effect = failing_get

        result2 = self.health_monitor.force_check()
        assert result2["healthy_count"] == 1
        assert result2["overall_status"] == "degraded"
        assert len(health_changes) == 1
        assert health_changes[0] == ("8082", True, False)

    def test_emulator_pool_get_status_method(self):
        """Test get_status method returns expected structure."""
        # Call get_status like HealthMonitor would
        status = self.mock_emulator_pool.get_status()

        # Verify required keys are present
        required_keys = {"available_count", "busy_count", "total_count", "status"}
        assert set(status.keys()).issuperset(required_keys)

        # Verify values
        assert status["available_count"] == 2
        assert status["busy_count"] == 0
        assert status["total_count"] == 2
        assert status["status"] == "healthy"


if __name__ == "__main__":
    unittest.main()
