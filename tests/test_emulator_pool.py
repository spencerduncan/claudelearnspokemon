"""
Test suite for EmulatorPool Docker container management and concurrent access

Combines comprehensive Docker container testing with concurrent access validation.
Tests both basic pool operations and advanced thread safety scenarios.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest
from docker.errors import APIError, ImageNotFound

from claudelearnspokemon.emulator_pool import (
    EmulatorInstance,
    EmulatorPool,
    EmulatorPoolError,
    EmulatorState,
)


class TestEmulatorPoolBasic:
    """Test basic EmulatorPool functionality with Docker integration"""

    @pytest.fixture
    def pool(self) -> EmulatorPool:
        """Create EmulatorPool instance for testing"""
        pool_instance = EmulatorPool(pool_size=4, default_timeout=2.0)
        yield pool_instance
        # Cleanup after test
        if hasattr(pool_instance, "_initialized") and pool_instance._initialized:
            pool_instance.shutdown()

    @pytest.fixture
    def mock_docker_client(self) -> Mock:
        """Mock Docker client for testing"""
        with patch("claudelearnspokemon.emulator_pool.docker.from_env") as mock_docker:
            mock_client = Mock()
            mock_docker.return_value = mock_client

            # Create mock containers
            containers = []
            for i in range(4):
                container = Mock()
                container.id = f"container_{i}"
                container.status = "running"
                container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")
                containers.append(container)

            mock_client.containers.run.side_effect = containers
            yield mock_client

    def test_emulator_pool_starts_containers_on_sequential_ports(
        self, pool: EmulatorPool, mock_docker_client: Mock
    ) -> None:
        """Test pool initializes Docker containers on sequential ports"""
        pool.initialize(pool_size=4)

        # Verify initialization state
        assert pool._initialized is True
        assert len(pool._emulators) == 4

        # Verify sequential ports starting from 8081
        expected_ports = [8081, 8082, 8083, 8084]
        actual_ports = sorted(pool._emulators.keys())
        assert actual_ports == expected_ports

        # Verify all emulators are available
        for _port, emulator in pool._emulators.items():
            assert emulator.state == EmulatorState.AVAILABLE
            assert emulator.owner_thread_id is None
            assert emulator.container is not None

    def test_emulator_pool_tracks_emulator_availability(
        self, pool: EmulatorPool, mock_docker_client: Mock
    ) -> None:
        """Test pool tracks emulator state correctly"""
        pool.initialize()

        # Initially all available
        status = pool.get_status()
        assert status["available_count"] == 4
        assert status["busy_count"] == 0

        # Acquire one emulator
        client = pool.acquire()
        assert client is not None

        # Verify state change
        status = pool.get_status()
        assert status["available_count"] == 3
        assert status["busy_count"] == 1

        # Release emulator
        pool.release(client)

        # Verify state restored
        status = pool.get_status()
        assert status["available_count"] == 4
        assert status["busy_count"] == 0

    def test_emulator_pool_blocks_acquisition_when_all_busy(
        self, pool: EmulatorPool, mock_docker_client: Mock
    ) -> None:
        """Test acquisition blocks when all emulators are busy"""
        pool.initialize()

        # Acquire all emulators
        clients = []
        for _ in range(4):
            client = pool.acquire()
            assert client is not None
            clients.append(client)

        # Next acquisition should block/timeout
        start_time = time.time()
        blocked_client = pool.acquire(timeout=0.5)
        duration = time.time() - start_time

        assert blocked_client is None
        assert 0.4 < duration < 0.7  # Should timeout around 0.5s

        # Release one and verify acquisition works
        pool.release(clients[0])
        client = pool.acquire(timeout=0.1)
        assert client is not None

    def test_emulator_pool_execute_script_ownership_validation(
        self, pool: EmulatorPool, mock_docker_client: Mock
    ) -> None:
        """Test script execution validates client ownership"""
        pool.initialize()

        # Acquire emulator
        client = pool.acquire()
        assert client is not None

        # Mock script
        mock_script = MagicMock()
        mock_script.id = "test_script"

        # Should succeed with correct owner
        result = pool.execute_script(client, mock_script, "checkpoint_1")
        assert result["success"] is True

        # Clean up
        pool.release(client)

    def test_emulator_pool_maintains_checkpoint_isolation_between_instances(
        self, pool: EmulatorPool, mock_docker_client: Mock
    ) -> None:
        """Test emulators maintain checkpoint isolation"""
        pool.initialize()

        # Acquire two emulators
        client1 = pool.acquire()
        client2 = pool.acquire()

        assert client1 is not None
        assert client2 is not None

        # Mock script objects
        mock_script1 = MagicMock()
        mock_script1.id = "script_1"

        mock_script2 = MagicMock()
        mock_script2.id = "script_2"

        # Execute scripts on different emulators with different checkpoints
        result1 = pool.execute_script(client1, mock_script1, "checkpoint_A")
        result2 = pool.execute_script(client2, mock_script2, "checkpoint_B")

        # Verify isolation - different ports and checkpoints
        assert result1["port"] != result2["port"]
        assert result1["checkpoint_id"] == "checkpoint_A"
        assert result2["checkpoint_id"] == "checkpoint_B"

        # Clean up
        pool.release(client1)
        pool.release(client2)

    def test_emulator_pool_gracefully_shuts_down_all_containers(
        self, pool: EmulatorPool, mock_docker_client: Mock
    ) -> None:
        """Test pool gracefully shuts down all Docker containers"""
        pool.initialize()

        # Acquire two emulators
        client1 = pool.acquire()
        client2 = pool.acquire()

        assert client1 is not None
        assert client2 is not None

        # Get status before shutdown
        status = pool.get_status()
        assert status["busy_count"] == 2
        assert status["available_count"] == 2

        # Shutdown
        pool.shutdown()

        # Verify shutdown state
        assert pool._shutdown is True
        assert len(pool._emulators) == 0
        assert len(pool._available_ports) == 0
        assert len(pool._busy_ports) == 0

        # Verify new acquisitions fail
        client = pool.acquire(timeout=0.5)
        assert client is None

    def test_emulator_pool_health_check(self, pool: EmulatorPool, mock_docker_client: Mock) -> None:
        """Test health check functionality with Docker containers"""
        pool.initialize()

        # All should be healthy initially
        health = pool.health_check()
        expected_ports = [8081, 8082, 8083, 8084]

        assert set(health.keys()) == set(expected_ports)
        for port, is_healthy in health.items():
            assert is_healthy is True, f"Port {port} should be healthy"

        # Simulate container failure
        pool._emulators[8081].state = EmulatorState.FAILED

        health = pool.health_check()
        assert health[8081] is False, "Failed emulator should be unhealthy"
        assert health[8082] is True, "Other emulators should be healthy"

    def test_emulator_pool_restart_emulator_conditions(
        self, pool: EmulatorPool, mock_docker_client: Mock
    ) -> None:
        """Test conditions for Docker container restart"""
        pool.initialize()

        # Should not restart non-existent emulator
        success = pool.restart_emulator(9999)
        assert success is False

        # Should not restart busy emulator
        client = pool.acquire()
        port = None
        for p, emulator in pool._emulators.items():
            if emulator.client == client:
                port = p
                break

        assert port is not None
        success = pool.restart_emulator(port)
        assert success is False

        # Should restart available emulator
        pool.release(client)

        # Mock the container creation for restart
        new_container = Mock()
        new_container.id = "restarted-container"
        new_container.status = "running"
        new_container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        with patch.object(pool, "_start_single_container", return_value=new_container):
            success = pool.restart_emulator(port)
            assert success is True

    def test_emulator_pool_restarts_failed_emulator_automatically(
        self, pool: EmulatorPool, mock_docker_client: Mock
    ) -> None:
        """Test automatic restart of failed Docker containers"""
        pool.initialize()

        # Get a test port
        test_port = 8081
        emulator = pool._emulators[test_port]

        # Simulate emulator failure
        emulator.state = EmulatorState.FAILED
        pool._available_ports.discard(test_port)

        # Verify failed state
        status = pool.get_status()
        assert status["available_count"] == 3

        # Mock the container creation for restart
        new_container = Mock()
        new_container.id = "restarted-container"
        new_container.status = "running"
        new_container.exec_run.return_value = Mock(exit_code=0, output=b"health_check")

        # Restart the failed emulator
        with patch.object(pool, "_start_single_container", return_value=new_container):
            restart_success = pool.restart_emulator(test_port)
            assert restart_success is True

        # Verify it's available again
        status = pool.get_status()
        assert status["available_count"] == 4

        emulator = pool._emulators[test_port]
        assert emulator.state == EmulatorState.AVAILABLE
        assert test_port in pool._available_ports

    # Docker-specific error handling tests

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_initialize_image_not_found(self, mock_docker: Mock, pool: EmulatorPool) -> None:
        """Test failure when Pokemon-gym image doesn't exist."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.side_effect = ImageNotFound("Image not found")

        with pytest.raises(EmulatorPoolError) as exc_info:
            pool.initialize(1)

        assert "not found" in str(exc_info.value)

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_initialize_port_conflict(self, mock_docker: Mock, pool: EmulatorPool) -> None:
        """Test handling of port conflicts - common production issue."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        # Simulate port conflict
        api_error = APIError("port is already allocated")
        mock_client.containers.run.side_effect = api_error

        with pytest.raises(EmulatorPoolError) as exc_info:
            pool.initialize(1)

        assert "port" in str(exc_info.value).lower()

    @pytest.mark.unit
    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_initialize_container_startup_timeout(self, mock_docker: Mock) -> None:
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
    def test_container_health_check(self, mock_docker: Mock, pool: EmulatorPool) -> None:
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
            pool.initialize(1)

        assert "Container health check failed" in str(exc_info.value)


class TestEmulatorInstance:
    """Test EmulatorInstance data class"""

    def test_emulator_instance_creation(self) -> None:
        """Test EmulatorInstance initialization"""
        emulator = EmulatorInstance(port=8081)

        assert emulator.port == 8081
        assert emulator.client is None
        assert emulator.state == EmulatorState.AVAILABLE
        assert emulator.owner_thread_id is None
        assert emulator.acquired_at is None

    def test_emulator_instance_availability_check(self) -> None:
        """Test EmulatorInstance availability logic"""
        emulator = EmulatorInstance(port=8081)

        # Should be available initially
        assert emulator.is_available() is True

        # Should not be available when busy
        emulator.state = EmulatorState.BUSY
        assert emulator.is_available() is False

        # Should not be available when owned by thread
        emulator.state = EmulatorState.AVAILABLE
        emulator.owner_thread_id = 12345
        assert emulator.is_available() is False

        # Should not be available when failed
        emulator.owner_thread_id = None
        emulator.state = EmulatorState.FAILED
        assert emulator.is_available() is False


class TestEmulatorPoolConfiguration:
    """Test EmulatorPool configuration and lifecycle"""

    def test_custom_pool_size_and_timeout(self) -> None:
        """Test EmulatorPool with custom configuration"""
        pool = EmulatorPool(pool_size=6, default_timeout=15.0, base_port=9000)

        assert pool.pool_size == 6
        assert pool.default_timeout == 15.0
        assert pool.base_port == 9000

    def test_acquisition_before_initialization(self) -> None:
        """Test acquisition behavior before pool initialization"""
        pool = EmulatorPool()

        # Should timeout waiting for initialization
        start_time = time.time()
        client = pool.acquire(timeout=0.5)
        duration = time.time() - start_time

        assert client is None
        assert 0.4 < duration < 0.7

    @patch("claudelearnspokemon.emulator_pool.docker.from_env")
    def test_double_initialization_handling(self, mock_docker: Mock) -> None:
        """Test multiple initialization calls are handled gracefully"""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.return_value = Mock(
            id="test", status="running", exec_run=Mock(return_value=Mock(exit_code=0))
        )

        pool = EmulatorPool()
        pool.initialize(1)

        # Second initialization should be ignored
        pool.initialize(2)  # Different size should be ignored

        assert pool.pool_size == 1  # Should remain at original size

    def test_double_shutdown_handling(self) -> None:
        """Test multiple shutdown calls are safe - idempotent operation"""
        pool = EmulatorPool()

        # Multiple shutdowns should not raise exception
        pool.shutdown()
        pool.shutdown()  # Should be safe


# Import concurrent access tests from the dedicated test file
# Note: These tests are maintained in test_emulator_pool_concurrent.py
# to keep them focused on concurrent access scenarios while this file
# focuses on basic Docker container operations.
