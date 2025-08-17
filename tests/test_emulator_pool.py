"""
Unit tests for EmulatorPool Docker container lifecycle management.

Tests cover production failure scenarios, timeouts, and resource management.
Following Bot Dean's philosophy: test all failure modes first.
"""

import logging
from unittest.mock import Mock, patch

import docker
import pytest
from docker.errors import APIError, ImageNotFound

from claudelearnspokemon.emulator_pool import (
    EmulatorPool,
    EmulatorPoolError,
    ExecutionResult,
    PokemonGymClient,
)


@pytest.mark.medium
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


@pytest.mark.medium
class TestEmulatorPoolResourcePool:
    """Test resource pool functionality for EmulatorPool."""

    def setup_method(self) -> None:
        """Set up test environment for resource pool tests."""
        self.pool = EmulatorPool(pool_size=2, base_port=9000)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        try:
            self.pool.shutdown()
        except Exception:
            pass

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_acquire_release_basic_functionality(self, mock_client_class, mock_docker) -> None:
        """Test basic acquire and release operations."""
        # Setup mocks
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        mock_containers = []
        mock_clients = []
        for i in range(2):
            container = Mock()
            container.id = f"container_{i}"
            container.status = "running"
            container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
            mock_containers.append(container)

            client = Mock()
            client.port = 9000 + i
            client.container_id = container.id
            mock_clients.append(client)

        mock_docker_client.containers.run.side_effect = mock_containers
        mock_client_class.side_effect = mock_clients

        # Initialize pool
        self.pool.initialize()

        # Test acquire
        acquired_client = self.pool.acquire()
        assert acquired_client in mock_clients
        # Simplified architecture: queue handles busy/available state automatically

        # Test release
        self.pool.release(acquired_client)
        # Client should be back in the available pool (simplified tracking)

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_emulator_pool_blocks_acquisition_when_all_busy(
        self, mock_client_class, mock_docker
    ) -> None:
        """Test that acquire blocks when all emulators are busy."""
        # Setup mocks
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        mock_containers = []
        mock_clients = []
        for i in range(2):
            container = Mock()
            container.id = f"container_{i}"
            container.status = "running"
            container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
            mock_containers.append(container)

            client = Mock()
            client.port = 9000 + i
            client.container_id = container.id
            mock_clients.append(client)

        mock_docker_client.containers.run.side_effect = mock_containers
        mock_client_class.side_effect = mock_clients

        # Initialize pool
        self.pool.initialize()

        # Acquire all clients
        self.pool.acquire()  # First client acquired
        self.pool.acquire()  # Second client acquired, pool now full

        # Try to acquire when all busy - should timeout
        with pytest.raises(EmulatorPoolError) as exc_info:
            self.pool.acquire(timeout=0.1)

        assert "No emulators available within 0.1s timeout" in str(exc_info.value)
        assert "All 2 emulators are currently busy" in str(exc_info.value)

    @pytest.mark.unit
    def test_acquire_before_initialization_fails(self) -> None:
        """Test that acquire fails before pool initialization."""
        with pytest.raises(EmulatorPoolError) as exc_info:
            self.pool.acquire()

        assert "EmulatorPool not initialized" in str(exc_info.value)

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_release_invalid_client_type(self, mock_client_class, mock_docker) -> None:
        """Test that release rejects invalid client types."""
        # Setup mocks for initialization
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        container = Mock()
        container.id = "container_1"
        container.status = "running"
        container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        client = Mock()
        client.port = 9000
        client.container_id = container.id

        mock_docker_client.containers.run.return_value = container
        mock_client_class.return_value = client

        self.pool.initialize()

        # Try to release invalid client type (string doesn't have port/container_id)
        with pytest.raises(EmulatorPoolError) as exc_info:
            self.pool.release("not_a_client")

        assert "Invalid client type - must be PokemonGymClient" in str(exc_info.value)

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    @patch("claudelearnspokemon.emulator_pool.requests.Session")
    def test_execute_script_success(self, mock_session, mock_client_class, mock_docker) -> None:
        """Test successful script execution."""
        # Setup mocks
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        container = Mock()
        container.id = "container_1"
        container.status = "running"
        container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        mock_client = Mock()
        mock_client.port = 9000
        mock_client.container_id = container.id
        mock_client.send_input.return_value = {"status": "success"}
        mock_client.get_state.return_value = {"game_state": "running"}

        mock_docker_client.containers.run.return_value = container
        mock_client_class.return_value = mock_client

        self.pool.initialize()

        # Execute script
        result = self.pool.execute_script("PRESS A")

        assert result.success is True
        assert result.output["response"]["status"] == "success"
        assert result.output["final_state"]["game_state"] == "running"
        assert result.execution_time is not None

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_execute_script_client_failure(self, mock_client_class, mock_docker) -> None:
        """Test script execution when client fails."""
        # Setup mocks
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        container = Mock()
        container.id = "container_1"
        container.status = "running"
        container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        mock_client = Mock()
        mock_client.port = 9000
        mock_client.container_id = container.id
        mock_client.send_input.side_effect = Exception("Connection failed")

        mock_docker_client.containers.run.return_value = container
        mock_client_class.return_value = mock_client

        self.pool.initialize()

        # Execute script with failure
        result = self.pool.execute_script("PRESS A")

        assert result.success is False
        assert "Connection failed" in result.error
        assert result.execution_time is not None

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_health_check_all_healthy(self, mock_client_class, mock_docker) -> None:
        """Test health check when all emulators are healthy."""
        # Setup mocks
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        mock_containers = []
        mock_clients = []
        for i in range(2):
            container = Mock()
            container.id = f"container_{i}"
            container.status = "running"
            container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
            mock_containers.append(container)

            client = Mock()
            client.port = 9000 + i
            client.container_id = container.id
            client.is_healthy.return_value = True
            mock_clients.append(client)

        mock_docker_client.containers.run.side_effect = mock_containers
        mock_client_class.side_effect = mock_clients

        self.pool.initialize()

        # Check health
        health_status = self.pool.health_check()

        assert health_status["status"] == "healthy"
        assert health_status["healthy_count"] == 2
        assert health_status["total_count"] == 2
        assert len(health_status["emulators"]) == 2

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_health_check_degraded(self, mock_client_class, mock_docker) -> None:
        """Test health check when some emulators are unhealthy."""
        # Setup mocks
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        mock_containers = []
        mock_clients = []
        for i in range(2):
            container = Mock()
            container.id = f"container_{i}"
            container.status = "running"
            container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
            mock_containers.append(container)

            client = Mock()
            client.port = 9000 + i
            client.container_id = container.id
            client.is_healthy.return_value = i == 0  # First healthy, second unhealthy
            mock_clients.append(client)

        mock_docker_client.containers.run.side_effect = mock_containers
        mock_client_class.side_effect = mock_clients

        self.pool.initialize()

        # Check health
        health_status = self.pool.health_check()

        assert health_status["status"] == "degraded"
        assert health_status["healthy_count"] == 1
        assert health_status["total_count"] == 2

    @pytest.mark.unit
    def test_health_check_before_initialization(self) -> None:
        """Test health check before initialization."""
        health_status = self.pool.health_check()

        assert health_status["status"] == "not_initialized"
        assert health_status["healthy_count"] == 0
        assert health_status["total_count"] == 0

    # Removed restart_emulator tests - functionality too complex for workstation use
    # For workstation development, if an emulator fails, shutdown and reinitialize the entire pool

    @pytest.mark.unit
    def test_compile_script_basic_dsl(self) -> None:
        """Test basic DSL compilation."""
        script = "PRESS A PRESS B MOVE UP WAIT"
        result = self.pool._compile_script(script)
        assert result == "A B UP WAIT"

    @pytest.mark.unit
    def test_compile_script_case_insensitive(self) -> None:
        """Test DSL compilation is case insensitive."""
        script = "press a move down press start"
        result = self.pool._compile_script(script)
        assert result == "A DOWN START"


@pytest.mark.medium
class TestPokemonGymClient:
    """Test PokemonGymClient functionality."""

    @pytest.mark.unit
    def test_initialization(self) -> None:
        """Test PokemonGymClient initialization."""
        client = PokemonGymClient(8081, "container123")
        assert client.port == 8081
        assert client.container_id == "container123"
        assert client.base_url == "http://localhost:8081"

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.requests.Session")
    def test_send_input_success(self, mock_session_class) -> None:
        """Test successful input sending."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = PokemonGymClient(8081, "container123")
        result = client.send_input("A B START")

        assert result["status"] == "ok"
        mock_session.post.assert_called_once()

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.requests.Session")
    def test_get_state_success(self, mock_session_class) -> None:
        """Test successful state retrieval."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"state": "running"}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = PokemonGymClient(8081, "container123")
        result = client.get_state()

        assert result["state"] == "running"

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.requests.Session")
    def test_is_healthy_true(self, mock_session_class) -> None:
        """Test health check returns True."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = PokemonGymClient(8081, "container123")
        assert client.is_healthy() is True

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.requests.Session")
    def test_is_healthy_false_on_exception(self, mock_session_class) -> None:
        """Test health check returns False on exception."""
        import requests

        mock_session = Mock()
        mock_session.get.side_effect = requests.RequestException("Connection failed")
        mock_session_class.return_value = mock_session

        # Create client after mocking the session
        client = PokemonGymClient(8081, "container123")
        assert client.is_healthy() is False


@pytest.mark.medium
class TestExecutionResult:
    """Test ExecutionResult functionality."""

    @pytest.mark.unit
    def test_successful_result(self) -> None:
        """Test successful execution result."""
        result = ExecutionResult(success=True, output={"data": "test"}, execution_time=1.5)
        assert result.success is True
        assert result.output["data"] == "test"
        assert result.execution_time == 1.5
        assert "SUCCESS" in str(result)

    @pytest.mark.unit
    def test_failed_result(self) -> None:
        """Test failed execution result."""
        result = ExecutionResult(success=False, output=None, error="Test error", execution_time=0.5)
        assert result.success is False
        assert result.error == "Test error"
        assert result.execution_time == 0.5
        assert "FAILURE" in str(result)


@pytest.mark.medium
class TestEmulatorPoolError:
    """Test custom exception class."""

    @pytest.mark.unit
    def test_emulator_pool_error_inheritance(self) -> None:
        """Test EmulatorPoolError is proper Exception subclass."""
        error = EmulatorPoolError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
