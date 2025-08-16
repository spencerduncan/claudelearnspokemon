"""
Unit tests for EmulatorPool Docker container lifecycle management.

Tests cover production failure scenarios, timeouts, and resource management.
Following Bot Dean's philosophy: test all failure modes first.
"""

import logging
from unittest.mock import Mock, patch

import docker
import pytest
from claudelearnspokemon.emulator_pool import EmulatorPool, EmulatorPoolError
from docker.errors import APIError, ImageNotFound


class TestEmulatorPool:
    """Test EmulatorPool with production-grade failure scenarios."""

    def setup_method(self) -> None:
        """Set up test environment for each test."""
        self.pool = EmulatorPool()

    def teardown_method(self) -> None:
        """Clean up after each test - production hygiene."""
        # Always attempt cleanup to prevent test pollution
        try:
            self.pool.shutdown()
        except Exception:
            pass  # Pool might not be initialized

    @pytest.mark.unit
    def test_initialization_defaults(self) -> None:
        """Test EmulatorPool initializes with production defaults."""
        pool = EmulatorPool()
        assert pool.pool_size == 4  # Default pool size
        assert pool.base_port == 8081  # Default starting port
        assert pool.image_name == "pokemon-gym:latest"
        assert pool.startup_timeout == 30  # Production timeout
        assert pool.containers == []
        assert pool.client is None

    @pytest.mark.unit
    def test_initialization_custom_parameters(self) -> None:
        """Test EmulatorPool accepts custom configuration."""
        pool = EmulatorPool(
            pool_size=8, base_port=9000, image_name="custom-pokemon:v1", startup_timeout=60
        )
        assert pool.pool_size == 8
        assert pool.base_port == 9000
        assert pool.image_name == "custom-pokemon:v1"
        assert pool.startup_timeout == 60

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_initialize_success(self, mock_docker) -> None:
        """Test successful container pool initialization."""
        # Mock Docker client and containers
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_containers = []
        for i in range(4):
            container = Mock()
            container.id = f"container_{i}"
            container.status = "running"
            # Mock health check to return success
            container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
            mock_containers.append(container)

        mock_client.containers.run.side_effect = mock_containers

        # Initialize pool
        self.pool.initialize(4)

        # Verify Docker client creation
        mock_docker.assert_called_once()

        # Verify containers were started with correct parameters
        assert mock_client.containers.run.call_count == 4

        expected_calls = []
        for i in range(4):
            port = 8081 + i
            expected_calls.append(
                {
                    "image": "pokemon-gym:latest",
                    "ports": {"8080/tcp": port},
                    "detach": True,
                    "remove": True,
                    "name": f"pokemon-emulator-{port}",
                }
            )

        # Verify each container was started with correct parameters
        for i, call in enumerate(mock_client.containers.run.call_args_list):
            args, kwargs = call
            assert kwargs["image"] == "pokemon-gym:latest"
            assert kwargs["ports"] == {"8080/tcp": 8081 + i}
            assert kwargs["detach"] is True
            assert kwargs["remove"] is True
            assert kwargs["name"] == f"pokemon-emulator-{8081 + i}"

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_initialize_docker_daemon_unavailable(self, mock_docker) -> None:
        """Test failure when Docker daemon is unavailable - critical production scenario."""
        mock_docker.side_effect = docker.errors.DockerException("Docker daemon not running")

        with pytest.raises(EmulatorPoolError) as exc_info:
            self.pool.initialize(4)

        assert "Docker daemon unavailable" in str(exc_info.value)
        assert "Ensure Docker daemon is running" in str(exc_info.value)

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_initialize_image_not_found(self, mock_docker) -> None:
        """Test failure when Pokemon-gym image doesn't exist."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.side_effect = ImageNotFound("Image not found")

        with pytest.raises(EmulatorPoolError) as exc_info:
            self.pool.initialize(4)

        assert "Pokemon-gym image not found" in str(exc_info.value)

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_initialize_port_conflict(self, mock_docker) -> None:
        """Test handling of port conflicts - common production issue."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        # First container succeeds, second fails due to port conflict
        successful_container = Mock(id="container_1", status="running")
        successful_container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        mock_client.containers.run.side_effect = [
            successful_container,
            APIError("Port 8082 already in use"),
        ]

        with pytest.raises(EmulatorPoolError) as exc_info:
            self.pool.initialize(4)

        assert "Failed to start container on port 8082" in str(exc_info.value)

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_initialize_partial_failure_cleanup(self, mock_docker) -> None:
        """Test cleanup of successfully started containers when initialization fails."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        # Two containers succeed, third fails
        successful_containers = [
            Mock(id="container_1", status="running"),
            Mock(id="container_2", status="running"),
        ]

        # Mock health checks for successful containers
        for container in successful_containers:
            container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        mock_client.containers.run.side_effect = [
            successful_containers[0],
            successful_containers[1],
            APIError("Resource exhausted"),
        ]

        with pytest.raises(EmulatorPoolError):
            self.pool.initialize(4)

        # Verify cleanup was attempted on successful containers
        successful_containers[0].stop.assert_called_once()
        successful_containers[1].stop.assert_called_once()

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_initialize_container_startup_timeout(self, mock_docker) -> None:
        """Test timeout handling during container startup."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        # Mock container that doesn't become ready within timeout
        mock_container = Mock()
        mock_container.id = "slow_container"
        mock_container.status = "starting"  # Never becomes "running"
        mock_client.containers.run.return_value = mock_container

        # Set short timeout for test
        pool = EmulatorPool(startup_timeout=1)

        with pytest.raises(EmulatorPoolError) as exc_info:
            pool.initialize(1)

        assert "Container startup timeout" in str(exc_info.value)

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_shutdown_success(self, mock_docker) -> None:
        """Test graceful shutdown of all containers."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        # Create mock containers
        mock_containers = []
        for i in range(4):
            container = Mock()
            container.id = f"container_{i}"
            container.status = "running"
            # Mock health check to return success
            container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
            mock_containers.append(container)

        mock_client.containers.run.side_effect = mock_containers

        # Initialize and then shutdown
        self.pool.initialize(4)
        self.pool.shutdown()

        # Verify all containers were stopped
        for container in mock_containers:
            container.stop.assert_called_once()

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_shutdown_with_stop_failures(self, mock_docker) -> None:
        """Test shutdown continues even if some containers fail to stop."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        # Create mock containers, one will fail to stop
        mock_containers = []
        for i in range(3):
            container = Mock()
            container.id = f"container_{i}"
            container.status = "running"
            # Mock health check to return success
            container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
            if i == 1:
                # Second container fails to stop
                container.stop.side_effect = APIError("Container stop failed")
            mock_containers.append(container)

        mock_client.containers.run.side_effect = mock_containers

        # Initialize and shutdown
        self.pool.initialize(3)
        self.pool.shutdown()  # Should not raise exception

        # Verify all stop attempts were made
        for container in mock_containers:
            container.stop.assert_called_once()

    @pytest.mark.unit
    def test_shutdown_before_initialization(self) -> None:
        """Test shutdown is safe when called before initialization."""
        # Should not raise exception
        self.pool.shutdown()

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_multiple_shutdown_calls(self, mock_docker) -> None:
        """Test multiple shutdown calls are safe - idempotent operation."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        container = Mock(id="container_1", status="running")
        container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
        mock_client.containers.run.return_value = container

        # Initialize, then shutdown multiple times
        self.pool.initialize(1)
        self.pool.shutdown()
        self.pool.shutdown()  # Second call should be safe

        # Container should only be stopped once
        container.stop.assert_called_once()

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_logging_configuration(self, mock_docker, caplog) -> None:
        """Test that proper logging is configured for operations."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        container = Mock(id="test_container", status="running")
        container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
        mock_client.containers.run.return_value = container

        with caplog.at_level(logging.INFO):
            self.pool.initialize(1)

        # Verify logging messages
        assert "Initializing EmulatorPool with 1 containers" in caplog.text
        assert "Starting container on port 8081" in caplog.text
        assert "Container test_container started successfully" in caplog.text

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_container_health_check(self, mock_docker) -> None:
        """Test container health verification after startup."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        # Container starts but health check fails
        mock_container = Mock()
        mock_container.id = "unhealthy_container"
        mock_container.status = "running"
        mock_container.exec_run.return_value = Mock(exit_code=1)  # Health check fails

        mock_client.containers.run.return_value = mock_container

        with pytest.raises(EmulatorPoolError) as exc_info:
            self.pool.initialize(1)

        assert "Container health check failed" in str(exc_info.value)

    @pytest.mark.unit
    def test_get_container_ports(self) -> None:
        """Test port calculation for different pool sizes."""
        pool = EmulatorPool(base_port=9000)

        # Test port calculation
        ports = pool._get_container_ports(5)
        expected_ports = [9000, 9001, 9002, 9003, 9004]
        assert ports == expected_ports

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_resource_cleanup_on_exception(self, mock_docker) -> None:
        """Test that resources are properly cleaned up when exceptions occur."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        # First container succeeds, second fails
        successful_container = Mock(id="success", status="running")
        successful_container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
        mock_client.containers.run.side_effect = [
            successful_container,
            RuntimeError("Unexpected error"),
        ]

        with pytest.raises(EmulatorPoolError):
            self.pool.initialize(2)

        # Verify successful container was cleaned up
        successful_container.stop.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.slow
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_full_lifecycle_integration(self, mock_docker) -> None:
        """Integration test for full container lifecycle."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        # Create realistic container mocks
        containers = []
        for i in range(4):
            container = Mock()
            container.id = f"pokemon-emulator-{8081+i}"
            container.status = "running"
            container.exec_run.return_value = Mock(exit_code=0)  # Health check passes
            containers.append(container)

        mock_client.containers.run.side_effect = containers

        # Full lifecycle test
        self.pool.initialize(4)
        assert len(self.pool.containers) == 4

        self.pool.shutdown()
        assert len(self.pool.containers) == 0

        # Verify all containers were managed correctly
        for container in containers:
            container.stop.assert_called_once()


class TestEmulatorPoolError:
    """Test custom exception class."""

    @pytest.mark.unit
    def test_emulator_pool_error_inheritance(self) -> None:
        """Test EmulatorPoolError is proper Exception subclass."""
        error = EmulatorPoolError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
