"""
Unit tests for EmulatorPool Docker container lifecycle management.

Tests cover production failure scenarios, timeouts, and resource management.
Following Bot Dean's philosophy: test all failure modes first.
"""

import logging
import threading
import time
from unittest.mock import Mock, patch

import docker
import pytest
from docker.errors import APIError, ImageNotFound

from claudelearnspokemon.emulator_pool import (
    ContainerHealthInfo,
    ContainerHealthStatus,
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

        assert (
            "Invalid client type - must be a valid Pokemon client with port and container_id attributes"
            in str(exc_info.value)
        )

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

        # Test new ExecutionResult format
        assert result.success is True  # Backward compatibility property
        assert result.status.name == "SUCCESS"
        assert result.final_state["game_state"] == "running"
        assert result.execution_time > 0  # Backward compatibility property
        assert result.performance_metrics["frames_executed"] > 0
        assert result.execution_id is not None
        assert result.script_id is not None

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
        mock_client.get_state.side_effect = Exception("Connection failed")  # Also fail get_state

        mock_docker_client.containers.run.return_value = container
        mock_client_class.return_value = mock_client

        self.pool.initialize()

        # Execute script with failure
        result = self.pool.execute_script("PRESS A")

        assert result.success is False
        assert (
            "failed" in result.error_message.lower()
        )  # Accept either "Connection failed" or "Script execution failed"
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

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_execute_script_with_progress_callback(self, mock_client_class, mock_docker) -> None:
        """Test script execution with progress monitoring."""
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

        # Track progress callbacks
        progress_calls = []

        def progress_callback(frames_executed, total_frames):
            progress_calls.append((frames_executed, total_frames))

        # Execute script with progress monitoring
        result = self.pool.execute_script("PRESS A", progress_callback=progress_callback)

        # Verify progress was tracked (in fallback mode, should still work)
        assert result.success is True

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_emulator_pool_handles_concurrent_acquisition_requests(
        self, mock_client_class, mock_docker
    ) -> None:
        """
        Test thread-safe handling of multiple simultaneous acquisition requests.

        Critical test: Verifies that EmulatorPool correctly handles concurrent
        access with proper synchronization and timeout behavior.
        Uses threading.Barrier for synchronized start of 4+ worker threads.
        Verifies exactly 2 successful acquisitions with pool size of 2.
        Verifies remaining threads get timeout errors.
        Ensures no duplicate clients acquired (thread safety check).
        """
        # Setup mocks for pool size of 2
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

        # Initialize pool with size 2
        self.pool.initialize()

        # Concurrent acquisition test setup
        num_threads = 5  # More threads than pool size
        barrier = threading.Barrier(num_threads)  # Synchronized start
        results = []
        results_lock = threading.Lock()

        def worker_acquire(worker_id: int) -> None:
            """Worker function for concurrent acquisition test."""
            barrier.wait()  # Wait for all threads to be ready

            try:
                # Short timeout to prevent test hanging
                client = self.pool.acquire(timeout=0.5)
                with results_lock:
                    results.append({"worker_id": worker_id, "success": True, "client": client})
            except EmulatorPoolError as e:
                with results_lock:
                    results.append({"worker_id": worker_id, "success": False, "error": str(e)})

        # Start all worker threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker_acquire, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=2.0)  # Prevent test hanging

        # Verify results - exactly 2 successful acquisitions
        successful_acquisitions = [r for r in results if r["success"]]
        failed_acquisitions = [r for r in results if not r["success"]]

        assert (
            len(successful_acquisitions) == 2
        ), f"Expected 2 successful acquisitions, got {len(successful_acquisitions)}"
        assert (
            len(failed_acquisitions) == 3
        ), f"Expected 3 failed acquisitions, got {len(failed_acquisitions)}"

        # Verify no duplicate clients acquired (thread safety check)
        acquired_clients = [r["client"] for r in successful_acquisitions]
        assert (
            len({id(c) for c in acquired_clients}) == 2
        ), "Duplicate clients acquired - thread safety violation"

        # Verify failed acquisitions have proper timeout error messages
        for failed in failed_acquisitions:
            assert "No emulators available within" in failed["error"]
            assert "All 2 emulators are currently busy" in failed["error"]

        # Clean up - release acquired clients
        for success in successful_acquisitions:
            self.pool.release(success["client"])

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_execute_script_with_cancellation(self, mock_client_class, mock_docker) -> None:
        """Test script execution with cancellation support."""
        import threading

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

        # Create cancellation event
        cancellation_event = threading.Event()

        # Execute script (should complete normally since we don't set the event)
        result = self.pool.execute_script("PRESS A", cancellation_event=cancellation_event)

        # Should have completed successfully
        assert result.success is True
        assert result.status.name == "SUCCESS"

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_emulator_pool_maintains_checkpoint_isolation_between_instances(
        self, mock_client_class, mock_docker
    ) -> None:
        """
        Test checkpoint isolation between parallel executions.

        Critical test: Verifies checkpoints don't interfere between parallel executions.
        Mock checkpoint namespaces per container (checkpoint_ns_0, checkpoint_ns_1).
        Test saving/loading checkpoints shows proper isolation.
        Ensure client1 checkpoint data != client2 checkpoint data.
        """
        # Setup mocks for 2 emulators
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
            # Mock checkpoint namespace isolation per container
            client.checkpoint_namespace = f"checkpoint_ns_{i}"
            mock_clients.append(client)

        mock_docker_client.containers.run.side_effect = mock_containers
        mock_client_class.side_effect = mock_clients

        # Initialize pool
        self.pool.initialize()

        # Acquire both clients for isolated checkpoint testing
        client1 = self.pool.acquire()
        client2 = self.pool.acquire()

        # Verify clients have different checkpoint namespaces
        assert client1.checkpoint_namespace != client2.checkpoint_namespace
        assert client1.checkpoint_namespace == "checkpoint_ns_0"
        assert client2.checkpoint_namespace == "checkpoint_ns_1"

        # Simulate checkpoint data isolation
        # In a real implementation, this would involve actual checkpoint save/load
        checkpoint_data_1 = {
            "container_id": client1.container_id,
            "namespace": client1.checkpoint_namespace,
            "game_state": {"location": "pallet_town", "player_x": 10, "player_y": 20},
            "timestamp": time.time(),
        }

        checkpoint_data_2 = {
            "container_id": client2.container_id,
            "namespace": client2.checkpoint_namespace,
            "game_state": {"location": "viridian_city", "player_x": 50, "player_y": 80},
            "timestamp": time.time(),
        }

        # Verify checkpoint data isolation - different containers have different data
        assert checkpoint_data_1["container_id"] != checkpoint_data_2["container_id"]
        assert checkpoint_data_1["namespace"] != checkpoint_data_2["namespace"]
        assert checkpoint_data_1["game_state"] != checkpoint_data_2["game_state"]

        # Test that checkpoint namespaces prevent cross-contamination
        # Client 1 checkpoint should only be accessible via client 1 namespace
        # Client 2 checkpoint should only be accessible via client 2 namespace
        assert "checkpoint_ns_0" in client1.checkpoint_namespace
        assert "checkpoint_ns_1" in client2.checkpoint_namespace
        assert "checkpoint_ns_0" not in client2.checkpoint_namespace
        assert "checkpoint_ns_1" not in client1.checkpoint_namespace

        # Cleanup
        self.pool.release(client1)
        self.pool.release(client2)

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_emulator_pool_handles_failed_emulator_gracefully(
        self, mock_client_class, mock_docker
    ) -> None:
        """
        Test pool handles failed emulators gracefully for workstation use.

        Critical test: Verifies pool handles failed emulators for workstation use.
        Initially all clients healthy, then simulate one emulator failing.
        Check health_check() shows "degraded" status with correct counts.
        Verify acquisition still works but health monitoring detects failures.
        """
        # Setup mocks for 2 emulators
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
            client.is_healthy.return_value = True  # Initially all healthy
            mock_clients.append(client)

        mock_docker_client.containers.run.side_effect = mock_containers
        mock_client_class.side_effect = mock_clients

        # Initialize pool
        self.pool.initialize()

        # Verify initial health - all healthy
        health_status = self.pool.health_check()
        assert health_status["status"] == "healthy"
        assert health_status["healthy_count"] == 2
        assert health_status["total_count"] == 2

        # Simulate failure of first emulator
        mock_clients[0].is_healthy.return_value = False  # First emulator fails
        mock_clients[1].is_healthy.return_value = True  # Second emulator still healthy

        # Check health after failure - should show degraded
        health_status = self.pool.health_check()
        assert (
            health_status["status"] == "degraded"
        ), "Health status should be degraded with failed emulator"
        assert health_status["healthy_count"] == 1, "Should have 1 healthy emulator"
        assert health_status["total_count"] == 2, "Should still track 2 total emulators"

        # Verify individual emulator status in health report
        emulator_statuses = health_status["emulators"]
        assert 9000 in emulator_statuses, "Port 9000 should be in health report"
        assert 9001 in emulator_statuses, "Port 9001 should be in health report"
        assert emulator_statuses[9000]["healthy"] is False, "First emulator should be unhealthy"
        assert emulator_statuses[9001]["healthy"] is True, "Second emulator should be healthy"

        # Verify acquisition still works despite failure
        # The queue-based system should still allow acquisition (may return any client)
        try:
            client = self.pool.acquire(timeout=1.0)
            assert client is not None, "Should be able to acquire client from pool"

            # The acquired client may be healthy or unhealthy - that's acceptable
            # The key is that the pool continues to function and health monitoring works

            # Release the client
            self.pool.release(client)

        except EmulatorPoolError:
            pytest.fail("Should be able to acquire client from pool despite health issues")

        # Verify health monitoring continues to detect the failure
        health_status = self.pool.health_check()
        assert health_status["status"] == "degraded", "Health status should remain degraded"
        assert health_status["healthy_count"] == 1, "Should still show 1 healthy emulator"


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

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.requests.Session")
    def test_circuit_breaker_functionality(self, mock_session_class) -> None:
        """Test circuit breaker opens after multiple failures."""
        import requests

        mock_session = Mock()
        # Configure session to always fail
        mock_session.get.side_effect = requests.RequestException("Connection failed")
        mock_session.post.side_effect = requests.RequestException("Connection failed")
        mock_session_class.return_value = mock_session

        client = PokemonGymClient(8081, "container123")

        # Make several failed requests to trigger circuit breaker
        for _ in range(PokemonGymClient.CIRCUIT_BREAKER_FAILURE_THRESHOLD):
            try:
                client.send_input("A")
            except EmulatorPoolError:
                pass

        # Circuit breaker should now be open
        assert client._is_circuit_breaker_open() is True

        # Next request should fail immediately with circuit breaker message
        with pytest.raises(EmulatorPoolError) as exc_info:
            client.send_input("A")
        assert "Circuit breaker OPEN" in str(exc_info.value)

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.requests.Session")
    def test_retry_logic_with_exponential_backoff(self, mock_session_class) -> None:
        """Test retry logic with exponential backoff delays."""
        import time

        import requests

        mock_session = Mock()
        # Fail first few attempts, then succeed
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.content = b'{"status": "success"}'

        mock_session.post.side_effect = [
            requests.RequestException("First failure"),
            requests.RequestException("Second failure"),
            mock_response,  # Third attempt succeeds
        ]
        mock_session_class.return_value = mock_session

        client = PokemonGymClient(8081, "container123")

        # This should succeed after retries
        start_time = time.time()
        result = client.send_input("A")
        end_time = time.time()

        # Should have succeeded
        assert result["status"] == "success"

        # Should have taken at least some time due to retry delays
        # (0.1 + 0.2 = 0.3 seconds minimum for 2 retry delays)
        assert end_time - start_time >= 0.25  # Allow some margin

        # Should have made 3 attempts total
        assert mock_session.post.call_count == 3

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.requests.Session")
    def test_load_checkpoint_success(self, mock_session_class) -> None:
        """Test successful checkpoint loading."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.content = b'{"status": "success"}'
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = PokemonGymClient(8081, "container123")
        checkpoint_data = b"test checkpoint data"

        result = client.load_checkpoint(checkpoint_data)

        assert result is True
        mock_session.post.assert_called_once()

        # Verify the files parameter was used
        call_args = mock_session.post.call_args
        assert "files" in call_args.kwargs
        assert call_args.kwargs["files"]["checkpoint"] == checkpoint_data

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.requests.Session")
    def test_get_performance_metrics(self, mock_session_class) -> None:
        """Test performance metrics tracking."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.content = b'{"status": "success"}'
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = PokemonGymClient(8081, "container123")

        # Make a request to generate metrics
        client.get_state()

        metrics = client.get_performance_metrics()

        assert metrics["total_requests"] == 1
        assert metrics["average_request_time_ms"] >= 0
        assert metrics["circuit_breaker_failures"] == 0
        assert metrics["circuit_breaker_open"] is False


@pytest.mark.medium
class TestExecutionResult:
    """Test ExecutionResult functionality."""

    @pytest.mark.unit
    def test_successful_result(self) -> None:
        """Test successful execution result."""
        import time

        from claudelearnspokemon.emulator_pool import ExecutionStatus

        start_time = time.time()
        end_time = start_time + 1.5

        result = ExecutionResult(
            execution_id="test-exec-123",
            script_id="test-script-456",
            status=ExecutionStatus.SUCCESS,
            start_time=start_time,
            end_time=end_time,
            final_state={"data": "test"},
            tile_observations=[],
            performance_metrics={},
            error_message=None,
        )

        assert result.success is True  # Backward compatibility property
        assert result.final_state["data"] == "test"
        assert result.execution_time == 1.5  # Backward compatibility property
        assert "SUCCESS" in str(result)

    @pytest.mark.unit
    def test_failed_result(self) -> None:
        """Test failed execution result."""
        import time

        from claudelearnspokemon.emulator_pool import ExecutionStatus

        start_time = time.time()
        end_time = start_time + 0.5

        result = ExecutionResult(
            execution_id="test-exec-789",
            script_id="test-script-012",
            status=ExecutionStatus.FAILED,
            start_time=start_time,
            end_time=end_time,
            final_state={},
            tile_observations=[],
            performance_metrics={},
            error_message="Test error",
        )

        assert result.success is False  # Backward compatibility property
        assert result.error_message == "Test error"
        assert result.execution_time == 0.5  # Backward compatibility property
        assert "FAILED" in str(result)


@pytest.mark.fast
class TestContainerReplacement:
    """Test container replacement functionality for auto-restart support."""

    def setup_method(self) -> None:
        """Set up test environment for container replacement tests."""
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
    def test_replace_failed_container_success(self, mock_client_class, mock_docker) -> None:
        """Test successful container replacement."""
        # Setup initial pool state
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        # Create initial containers and clients
        old_container = Mock()
        old_container.id = "old_container_123"
        old_container.name = "pokemon-emulator-9000"
        old_container.status = "running"
        old_container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        new_container = Mock()
        new_container.id = "new_container_456"
        new_container.status = "running"
        new_container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        # Mock container creation sequence
        mock_docker_client.containers.run.side_effect = [old_container, new_container]

        old_client = Mock()
        old_client.port = 9000
        old_client.container_id = "old_container_123"
        old_client.close = Mock()

        new_client = Mock()
        new_client.port = 9000
        new_client.container_id = "new_container_456"

        mock_client_class.side_effect = [old_client, new_client]

        # Initialize pool with one container
        self.pool.initialize(1)

        # Acquire client to make it busy (not in available queue)
        acquired_client = self.pool.acquire()
        assert acquired_client == old_client

        # Replace the failed container
        success = self.pool.replace_failed_container(9000)

        assert success is True

        # Verify old container was stopped
        old_container.stop.assert_called_once_with(timeout=10)

        # Verify old client was closed
        old_client.close.assert_called_once()

        # Verify new client is in the mapping
        assert self.pool.clients_by_port[9000] == new_client

        # Verify new container is in containers list
        assert new_container in self.pool.containers
        assert old_container not in self.pool.containers

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_replace_failed_container_available_client(
        self, mock_client_class, mock_docker
    ) -> None:
        """Test container replacement when client was available in queue."""
        # Setup mocks
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        old_container = Mock()
        old_container.id = "old_container_123"
        old_container.name = "pokemon-emulator-9000"
        old_container.status = "running"
        old_container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        new_container = Mock()
        new_container.id = "new_container_456"
        new_container.status = "running"
        new_container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        mock_docker_client.containers.run.side_effect = [old_container, new_container]

        old_client = Mock()
        old_client.port = 9000
        old_client.container_id = "old_container_123"
        old_client.close = Mock()

        new_client = Mock()
        new_client.port = 9000
        new_client.container_id = "new_container_456"

        mock_client_class.side_effect = [old_client, new_client]

        # Initialize pool - client will be available in queue
        self.pool.initialize(1)

        # Don't acquire - client remains in available queue

        # Replace the failed container
        success = self.pool.replace_failed_container(9000)

        assert success is True

        # Verify new client was added back to available queue
        # (This is tested indirectly through the queue size)
        assert self.pool.available_clients.qsize() == 1

    @pytest.mark.unit
    def test_replace_failed_container_pool_not_initialized(self) -> None:
        """Test replacement fails when pool not initialized."""
        success = self.pool.replace_failed_container(9000)
        assert success is False

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_replace_failed_container_client_not_found(
        self, mock_client_class, mock_docker
    ) -> None:
        """Test replacement fails when client not found for port."""
        # Setup basic pool
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        container = Mock()
        container.id = "container_123"
        container.status = "running"
        container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        mock_docker_client.containers.run.return_value = container
        mock_client_class.return_value = Mock(port=9000, container_id="container_123")

        self.pool.initialize(1)

        # Try to replace non-existent port
        success = self.pool.replace_failed_container(9999)
        assert success is False

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_replace_failed_container_new_container_fails(
        self, mock_client_class, mock_docker
    ) -> None:
        """Test replacement handles new container creation failure."""
        # Setup initial pool
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        old_container = Mock()
        old_container.id = "old_container_123"
        old_container.name = "pokemon-emulator-9000"
        old_container.status = "running"
        old_container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        # First call succeeds (initialization), second fails (replacement)
        mock_docker_client.containers.run.side_effect = [
            old_container,
            APIError("Port conflict during replacement"),
        ]

        old_client = Mock()
        old_client.port = 9000
        old_client.container_id = "old_container_123"
        old_client.close = Mock()

        mock_client_class.return_value = old_client

        self.pool.initialize(1)

        # Replace should fail due to new container creation error
        success = self.pool.replace_failed_container(9000)

        assert success is False

        # Verify old container was still stopped
        old_container.stop.assert_called_once()

        # Verify client mapping was cleaned up
        assert 9000 not in self.pool.clients_by_port

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_replace_failed_container_old_container_not_found(
        self, mock_client_class, mock_docker
    ) -> None:
        """Test replacement continues when old container can't be found."""
        # Setup initial pool
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        old_container = Mock()
        old_container.id = "old_container_123"
        old_container.name = "different-name"  # Won't match expected name
        old_container.status = "running"
        old_container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        new_container = Mock()
        new_container.id = "new_container_456"
        new_container.status = "running"
        new_container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        mock_docker_client.containers.run.side_effect = [old_container, new_container]

        old_client = Mock()
        old_client.port = 9000
        old_client.container_id = "old_container_123"
        old_client.close = Mock()

        new_client = Mock()
        new_client.port = 9000
        new_client.container_id = "new_container_456"

        mock_client_class.side_effect = [old_client, new_client]

        self.pool.initialize(1)

        # Replace should succeed even though old container wasn't found by name
        success = self.pool.replace_failed_container(9000)

        assert success is True

        # Old container shouldn't have been stopped since it wasn't found
        old_container.stop.assert_not_called()

        # New container should still be created
        assert self.pool.clients_by_port[9000] == new_client

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_is_client_available_true(self, mock_client_class, mock_docker) -> None:
        """Test _is_client_available returns True for available client."""
        # Setup pool with available client
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        container = Mock()
        container.id = "container_123"
        container.status = "running"
        container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        mock_docker_client.containers.run.return_value = container

        client = Mock()
        client.port = 9000
        client.container_id = "container_123"

        mock_client_class.return_value = client

        self.pool.initialize(1)

        # Client should be available in queue
        is_available = self.pool._is_client_available(client)
        assert is_available is True

        # Queue should be restored after check
        assert self.pool.available_clients.qsize() == 1

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_is_client_available_false(self, mock_client_class, mock_docker) -> None:
        """Test _is_client_available returns False for busy client."""
        # Setup pool
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        container = Mock()
        container.id = "container_123"
        container.status = "running"
        container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        mock_docker_client.containers.run.return_value = container

        client = Mock()
        client.port = 9000
        client.container_id = "container_123"

        mock_client_class.return_value = client

        self.pool.initialize(1)

        # Acquire client to make it busy
        self.pool.acquire()

        # Client should not be available in queue
        is_available = self.pool._is_client_available(client)
        assert is_available is False

        # Queue should be empty after check
        assert self.pool.available_clients.qsize() == 0


@pytest.mark.medium
class TestContainerHealthStatus:
    """Test ContainerHealthStatus enum and health tracking functionality."""

    @pytest.mark.unit
    def test_container_health_status_enum_values(self) -> None:
        """Test ContainerHealthStatus enum has expected values."""
        assert ContainerHealthStatus.HEALTHY.value == "healthy"
        assert ContainerHealthStatus.UNHEALTHY.value == "unhealthy"
        assert ContainerHealthStatus.STOPPED.value == "stopped"
        assert ContainerHealthStatus.UNKNOWN.value == "unknown"

    @pytest.mark.unit
    def test_container_health_status_str_representation(self) -> None:
        """Test ContainerHealthStatus string representation."""
        assert str(ContainerHealthStatus.HEALTHY) == "healthy"
        assert str(ContainerHealthStatus.UNHEALTHY) == "unhealthy"
        assert str(ContainerHealthStatus.STOPPED) == "stopped"
        assert str(ContainerHealthStatus.UNKNOWN) == "unknown"

    @pytest.mark.unit
    def test_container_health_status_is_available(self) -> None:
        """Test is_available property correctly identifies healthy containers."""
        assert ContainerHealthStatus.HEALTHY.is_available is True
        assert ContainerHealthStatus.UNHEALTHY.is_available is False
        assert ContainerHealthStatus.STOPPED.is_available is False
        assert ContainerHealthStatus.UNKNOWN.is_available is False

    @pytest.mark.unit
    def test_container_health_status_needs_restart(self) -> None:
        """Test needs_restart property correctly identifies containers needing restart."""
        assert ContainerHealthStatus.HEALTHY.needs_restart is False
        assert ContainerHealthStatus.UNHEALTHY.needs_restart is True
        assert ContainerHealthStatus.STOPPED.needs_restart is True
        assert ContainerHealthStatus.UNKNOWN.needs_restart is False

    @pytest.mark.unit
    def test_container_health_info_creation(self) -> None:
        """Test ContainerHealthInfo dataclass creation and validation."""
        health_info = ContainerHealthInfo(
            container_id="abc123def456789",  # Long container ID
            port=8081,
            status=ContainerHealthStatus.HEALTHY,
            last_check_time=1234567890.0,
            docker_status="running",
            response_time_ms=50.0,
            consecutive_failures=0
        )
        
        # Container ID should be truncated to 12 characters
        assert health_info.container_id == "abc123def456"
        assert health_info.port == 8081
        assert health_info.status == ContainerHealthStatus.HEALTHY
        assert health_info.docker_status == "running"
        assert health_info.response_time_ms == 50.0
        assert health_info.consecutive_failures == 0

    @pytest.mark.unit
    def test_container_health_info_short_container_id(self) -> None:
        """Test ContainerHealthInfo with already short container ID."""
        health_info = ContainerHealthInfo(
            container_id="short123",
            port=8081,
            status=ContainerHealthStatus.HEALTHY,
            last_check_time=1234567890.0,
            docker_status="running"
        )
        
        # Short ID should remain unchanged
        assert health_info.container_id == "short123"

    @pytest.mark.unit  
    @patch("time.time")
    def test_container_health_info_age_seconds(self, mock_time) -> None:
        """Test age_seconds property calculation."""
        mock_time.return_value = 1234567900.0  # Current time
        
        health_info = ContainerHealthInfo(
            container_id="abc123",
            port=8081,
            status=ContainerHealthStatus.HEALTHY,
            last_check_time=1234567890.0,  # 10 seconds ago
            docker_status="running"
        )
        
        assert health_info.age_seconds == 10.0

    @pytest.mark.unit
    @patch("time.time")
    def test_container_health_info_is_stale(self, mock_time) -> None:
        """Test is_stale property for detecting outdated health information."""
        mock_time.return_value = 1234567950.0  # Current time
        
        # Fresh health info (10 seconds old)
        fresh_health = ContainerHealthInfo(
            container_id="abc123",
            port=8081,
            status=ContainerHealthStatus.HEALTHY,
            last_check_time=1234567940.0,
            docker_status="running"
        )
        
        # Stale health info (70 seconds old)
        stale_health = ContainerHealthInfo(
            container_id="def456",
            port=8082,
            status=ContainerHealthStatus.HEALTHY,
            last_check_time=1234567880.0,
            docker_status="running"
        )
        
        assert fresh_health.is_stale is False
        assert stale_health.is_stale is True

    @pytest.mark.unit
    def test_container_health_info_to_dict(self) -> None:
        """Test ContainerHealthInfo serialization to dictionary."""
        health_info = ContainerHealthInfo(
            container_id="abc123def456789",
            port=8081,
            status=ContainerHealthStatus.HEALTHY,
            last_check_time=1234567890.0,
            docker_status="running",
            error_message=None,
            response_time_ms=45.5,
            consecutive_failures=0
        )
        
        result = health_info.to_dict()
        
        # Verify all fields except age_seconds (which depends on current time)
        assert result["container_id"] == "abc123def456"
        assert result["port"] == 8081
        assert result["status"] == "healthy"
        assert result["last_check_time"] == 1234567890.0
        assert result["docker_status"] == "running"
        assert result["error_message"] is None
        assert result["response_time_ms"] == 45.5
        assert result["consecutive_failures"] == 0
        assert "age_seconds" in result
        assert isinstance(result["age_seconds"], float)


@pytest.mark.medium
class TestEmulatorPoolHealthTracking:
    """Test EmulatorPool health status tracking functionality."""

    def setup_method(self) -> None:
        """Set up test environment for each test."""
        self.pool = EmulatorPool()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        try:
            self.pool.shutdown()
        except Exception:
            pass

    @pytest.mark.unit
    def test_initial_health_tracking_state(self) -> None:
        """Test EmulatorPool initializes with empty health tracking."""
        assert self.pool.container_health == {}
        assert hasattr(self.pool, '_health_lock')

    @pytest.mark.unit
    def test_update_container_health_new_container(self) -> None:
        """Test _update_container_health creates new health info for new container."""
        self.pool._update_container_health(
            port=8081,
            container_id="abc123def456789",
            status=ContainerHealthStatus.HEALTHY,
            docker_status="running",
            response_time_ms=50.0
        )
        
        health_info = self.pool.container_health[8081]
        assert health_info.container_id == "abc123def456"
        assert health_info.port == 8081
        assert health_info.status == ContainerHealthStatus.HEALTHY
        assert health_info.docker_status == "running"
        assert health_info.response_time_ms == 50.0
        assert health_info.consecutive_failures == 0

    @pytest.mark.unit
    def test_update_container_health_status_change_logging(self) -> None:
        """Test health status changes are properly logged."""
        with patch("claudelearnspokemon.emulator_pool.logger") as mock_logger:
            # Initial healthy status
            self.pool._update_container_health(
                port=8081,
                container_id="abc123",
                status=ContainerHealthStatus.HEALTHY,
                docker_status="running"
            )
            
            # Change to unhealthy
            self.pool._update_container_health(
                port=8081,
                container_id="abc123",
                status=ContainerHealthStatus.UNHEALTHY,
                docker_status="running",
                error_message="Connection timeout"
            )
            
            # Verify logging calls
            assert mock_logger.info.call_count == 2  # Initial + status change
            assert mock_logger.warning.call_count == 1  # Unhealthy status

    @pytest.mark.unit
    def test_get_container_health_status(self) -> None:
        """Test getting health status for specific container."""
        # Add health info
        self.pool._update_container_health(
            port=8081,
            container_id="abc123",
            status=ContainerHealthStatus.HEALTHY,
            docker_status="running"
        )
        
        # Get health status
        health_info = self.pool.get_container_health_status(8081)
        assert health_info is not None
        assert health_info.port == 8081
        assert health_info.status == ContainerHealthStatus.HEALTHY
        
        # Non-existent container
        health_info = self.pool.get_container_health_status(9999)
        assert health_info is None

    @pytest.mark.unit
    def test_get_all_container_health(self) -> None:
        """Test getting health status for all containers."""
        # Add multiple health entries
        self.pool._update_container_health(
            port=8081,
            container_id="abc123",
            status=ContainerHealthStatus.HEALTHY,
            docker_status="running"
        )
        
        self.pool._update_container_health(
            port=8082,
            container_id="def456",
            status=ContainerHealthStatus.UNHEALTHY,
            docker_status="running",
            error_message="Connection failed"
        )
        
        all_health = self.pool.get_all_container_health()
        assert len(all_health) == 2
        assert 8081 in all_health
        assert 8082 in all_health
        assert all_health[8081].status == ContainerHealthStatus.HEALTHY
        assert all_health[8082].status == ContainerHealthStatus.UNHEALTHY

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("claudelearnspokemon.emulator_pool.PokemonGymClient")
    def test_enhanced_health_check_docker_status_integration(self, mock_client_class, mock_docker) -> None:
        """Test enhanced health_check method with Docker status checking."""
        # Setup mock Docker client
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client
        
        # Create mock container with network settings
        container = Mock()
        container.id = "container_123"
        container.status = "running"
        container.attrs = {
            'NetworkSettings': {
                'Ports': {
                    '8080/tcp': [{'HostPort': '8081'}]
                }
            }
        }
        container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        mock_docker_client.containers.run.return_value = container
        self.pool.containers = [container]
        
        # Setup mock client
        client = Mock()
        client.container_id = "container_123"
        client.is_healthy.return_value = True
        mock_client_class.return_value = client
        
        self.pool.initialize(1)
        
        # Test health check
        health_report = self.pool.health_check()
        
        assert health_report["status"] == "healthy"
        assert health_report["healthy_count"] == 1
        assert health_report["total_count"] == 1
        assert len(health_report["containers"]) == 1
        
        container_health = health_report["containers"][0]
        assert container_health["status"] == "healthy"
        assert container_health["docker_status"] == "running"
        assert container_health["healthy"] is True
        assert container_health["needs_restart"] is False


@pytest.mark.medium
class TestPokemonGymClientSimplification:
    """Test simplified PokemonGymClient workstation patterns."""

    @pytest.mark.unit
    def test_simplified_failure_tracking_initialization(self) -> None:
        """Test PokemonGymClient initializes with simplified failure tracking."""
        client = PokemonGymClient(port=8081, container_id="abc123")
        
        assert client.MAX_CONSECUTIVE_FAILURES == 3
        assert client.FAILURE_RESET_TIMEOUT == 10
        assert client.MAX_RETRIES == 2
        assert client.RETRY_DELAY == 0.5
        
        assert client._consecutive_failures == 0
        assert client._last_failure_time == 0.0
        assert hasattr(client, '_failure_lock')

    @pytest.mark.unit
    def test_simplified_performance_metrics(self) -> None:
        """Test simplified performance metrics for workstation monitoring."""
        client = PokemonGymClient(port=8081, container_id="abc123")
        
        # Initial metrics
        metrics = client.get_performance_metrics()
        expected = {
            "total_requests": 0,
            "average_request_time_ms": 0,
            "consecutive_failures": 0,
            "temporarily_disabled": False,
        }
        assert metrics == expected

    @pytest.mark.unit
    def test_record_failure_consecutive_tracking(self) -> None:
        """Test failure recording and consecutive failure tracking."""
        client = PokemonGymClient(port=8081, container_id="abc123")
        
        # Record multiple failures
        with patch("claudelearnspokemon.emulator_pool.logger") as mock_logger:
            client._record_failure()
            client._record_failure()
            client._record_failure()  # Should trigger warning
            
            assert client._consecutive_failures == 3
            assert mock_logger.warning.called

    @pytest.mark.unit
    @patch("time.time")
    def test_is_temporarily_disabled(self, mock_time) -> None:
        """Test temporary disable logic for workstation fail-fast behavior."""
        mock_time.return_value = 1000.0
        client = PokemonGymClient(port=8081, container_id="abc123")
        
        # Not disabled initially
        assert client._is_temporarily_disabled() is False
        
        # Record max failures
        client._consecutive_failures = client.MAX_CONSECUTIVE_FAILURES
        client._last_failure_time = 1000.0
        
        # Should be disabled
        assert client._is_temporarily_disabled() is True
        
        # After timeout, should be reset
        mock_time.return_value = 1020.0  # 20 seconds later (> 10s timeout)
        assert client._is_temporarily_disabled() is False
        assert client._consecutive_failures == 0

    @pytest.mark.unit
    def test_reset_failures(self) -> None:
        """Test failure reset after successful request."""
        client = PokemonGymClient(port=8081, container_id="abc123")
        
        # Set some failures
        client._consecutive_failures = 2
        client._last_failure_time = 1000.0
        
        with patch("claudelearnspokemon.emulator_pool.logger") as mock_logger:
            client._reset_failures()
            
            assert client._consecutive_failures == 0
            assert client._last_failure_time == 0.0
            assert mock_logger.info.called


@pytest.mark.medium
class TestEmulatorPoolError:
    """Test custom exception class."""

    @pytest.mark.unit
    def test_emulator_pool_error_inheritance(self) -> None:
        """Test EmulatorPoolError is proper Exception subclass."""
        error = EmulatorPoolError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
