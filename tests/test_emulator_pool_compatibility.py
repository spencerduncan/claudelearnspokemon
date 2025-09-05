"""
Test suite for EmulatorPool compatibility layer integration.

Tests that the EmulatorPool works correctly with both direct clients
and adapters, maintaining backward compatibility while enabling
transparent server type selection.

Author: Bot Dean - Production-First Engineering
"""

from unittest.mock import Mock, call, patch

import pytest

from src.claudelearnspokemon.emulator_pool import EmulatorPool, EmulatorPoolError, PokemonGymClient
from src.claudelearnspokemon.pokemon_gym_adapter import PokemonGymAdapter


@pytest.mark.fast
@pytest.mark.medium
class TestEmulatorPoolCompatibilityLayerIntegration:
    """Test EmulatorPool integration with compatibility layer."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear any existing cache
        from src.claudelearnspokemon.pokemon_gym_factory import clear_detection_cache

        clear_detection_cache()

    def test_emulator_pool_initialization_with_default_adapter_type(self):
        """Test EmulatorPool initializes with default adapter_type='auto'."""
        pool = EmulatorPool(pool_size=2, base_port=8081)

        assert pool.adapter_type == "auto"
        assert pool.input_delay == 0.05
        assert pool.detection_timeout == 3.0

    def test_emulator_pool_initialization_with_custom_adapter_config(self):
        """Test EmulatorPool accepts custom adapter configuration."""
        pool = EmulatorPool(
            pool_size=2,
            base_port=8081,
            adapter_type="benchflow",
            input_delay=0.1,
            detection_timeout=5.0,
        )

        assert pool.adapter_type == "benchflow"
        assert pool.input_delay == 0.1
        assert pool.detection_timeout == 5.0

    @patch("src.claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("src.claudelearnspokemon.emulator_pool.create_pokemon_client")
    def test_emulator_pool_uses_factory_for_client_creation(self, mock_create_client, mock_docker):
        """Test EmulatorPool uses factory to create clients."""
        # Mock Docker client and container
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        mock_container = Mock()
        mock_container.id = "test_container_123"
        mock_container.status = "running"
        mock_docker_client.containers.run.return_value = mock_container

        # Mock container execution for health check
        mock_exec_result = Mock()
        mock_exec_result.exit_code = 0
        mock_container.exec_run.return_value = mock_exec_result

        # Mock factory client creation
        mock_client = Mock(spec=PokemonGymClient)
        mock_client.port = 8081
        mock_client.container_id = "test_container_123"
        mock_create_client.return_value = mock_client

        # Create and initialize pool
        pool = EmulatorPool(
            pool_size=1,
            base_port=8081,
            adapter_type="benchflow",
            input_delay=0.08,
            detection_timeout=4.0,
        )
        pool.initialize()

        # Verify factory was called with correct parameters
        mock_create_client.assert_called_once_with(
            port=8081,
            container_id="test_container_123",
            adapter_type="benchflow",
            input_delay=0.08,
            detection_timeout=4.0,
        )

        # Verify client was added to pool
        assert len(pool.clients_by_port) == 1
        assert pool.clients_by_port[8081] == mock_client

    @patch("src.claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("src.claudelearnspokemon.emulator_pool.COMPATIBILITY_LAYER_AVAILABLE", False)
    def test_emulator_pool_fallback_without_compatibility_layer(self, mock_docker):
        """Test EmulatorPool falls back to direct client when compatibility layer unavailable."""
        # Mock Docker client and container
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        mock_container = Mock()
        mock_container.id = "test_container_123"
        mock_container.status = "running"
        mock_docker_client.containers.run.return_value = mock_container

        # Mock container execution for health check
        mock_exec_result = Mock()
        mock_exec_result.exit_code = 0
        mock_container.exec_run.return_value = mock_exec_result

        # Create and initialize pool
        pool = EmulatorPool(pool_size=1, base_port=8081, adapter_type="benchflow")
        pool.initialize()

        # Should create direct client as fallback
        assert len(pool.clients_by_port) == 1
        client = pool.clients_by_port[8081]
        assert isinstance(client, PokemonGymClient)

    @patch("src.claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("src.claudelearnspokemon.emulator_pool.create_pokemon_client")
    def test_emulator_pool_handles_factory_errors(self, mock_create_client, mock_docker):
        """Test EmulatorPool handles factory creation errors properly."""
        # Mock Docker client and container
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        mock_container = Mock()
        mock_container.id = "test_container_123"
        mock_container.status = "running"
        mock_docker_client.containers.run.return_value = mock_container

        # Mock container execution for health check
        mock_exec_result = Mock()
        mock_exec_result.exit_code = 0
        mock_container.exec_run.return_value = mock_exec_result

        # Mock factory failure
        mock_create_client.side_effect = Exception("Factory creation failed")

        # Pool initialization should fail with proper error
        pool = EmulatorPool(pool_size=1, base_port=8081)

        with pytest.raises(EmulatorPoolError) as exc_info:
            pool.initialize()

        assert "Failed to start container on port 8081" in str(exc_info.value)


@pytest.mark.fast
@pytest.mark.medium
class TestEmulatorPoolBackwardCompatibility:
    """Test that EmulatorPool maintains backward compatibility."""

    @patch("src.claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("src.claudelearnspokemon.emulator_pool.create_pokemon_client")
    def test_existing_emulator_pool_code_still_works(self, mock_create_client, mock_docker):
        """Test that existing EmulatorPool usage patterns still work."""
        # Mock Docker and container setup
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        mock_container = Mock()
        mock_container.id = "test_container_123"
        mock_container.status = "running"
        mock_docker_client.containers.run.return_value = mock_container

        mock_exec_result = Mock()
        mock_exec_result.exit_code = 0
        mock_container.exec_run.return_value = mock_exec_result

        # Mock client creation
        mock_client = Mock(spec=PokemonGymClient)
        mock_client.port = 8081
        mock_client.container_id = "test_container_123"
        mock_create_client.return_value = mock_client

        # Test old-style EmulatorPool creation (without new parameters)
        pool = EmulatorPool(pool_size=2, base_port=8081)

        # Should initialize successfully
        pool.initialize()

        # Should have default adapter configuration
        assert pool.adapter_type == "auto"
        assert pool.input_delay == 0.05
        assert pool.detection_timeout == 3.0

        # Should create clients successfully
        assert len(pool.clients_by_port) == 2

    def test_emulator_pool_constructor_backward_compatibility(self):
        """Test EmulatorPool constructor maintains backward compatibility."""
        # Old-style constructor call should work
        pool = EmulatorPool(
            pool_size=4, base_port=8081, image_name="pokemon-gym:latest", startup_timeout=30
        )

        # Should have expected old parameters
        assert pool.pool_size == 4
        assert pool.base_port == 8081
        assert pool.image_name == "pokemon-gym:latest"
        assert pool.startup_timeout == 30

        # Should have default new parameters
        assert pool.adapter_type == "auto"
        assert pool.input_delay == 0.05
        assert pool.detection_timeout == 3.0

    @patch("src.claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("src.claudelearnspokemon.emulator_pool.create_pokemon_client")
    def test_emulator_pool_methods_work_with_any_client_type(self, mock_create_client, mock_docker):
        """Test EmulatorPool methods work with both client types."""
        # Mock Docker setup
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        mock_container = Mock()
        mock_container.id = "test_container_123"
        mock_container.status = "running"
        mock_container.attrs = {"NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "8081"}]}}}
        mock_docker_client.containers.run.return_value = mock_container

        mock_exec_result = Mock()
        mock_exec_result.exit_code = 0
        mock_container.exec_run.return_value = mock_exec_result

        # Test with adapter
        mock_adapter = Mock(spec=PokemonGymAdapter)
        mock_adapter.port = 8081
        mock_adapter.container_id = "test_container_123"
        mock_adapter.is_healthy.return_value = True
        mock_create_client.return_value = mock_adapter

        pool = EmulatorPool(pool_size=1, base_port=8081)
        pool.initialize()

        # Test acquire/release still works
        client = pool.acquire()
        assert client == mock_adapter

        pool.release(client)

        # Test health check works
        health = pool.health_check()
        assert health["healthy_count"] == 1

        pool.shutdown()


@pytest.mark.fast
@pytest.mark.medium
class TestEmulatorPoolWithDifferentAdapterTypes:
    """Test EmulatorPool behavior with different adapter types."""

    @patch("src.claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("src.claudelearnspokemon.emulator_pool.create_pokemon_client")
    def test_emulator_pool_with_direct_adapter_type(self, mock_create_client, mock_docker):
        """Test EmulatorPool with explicit direct adapter type."""
        # Setup mocks
        self._setup_docker_mocks(mock_docker, mock_create_client, PokemonGymClient)

        pool = EmulatorPool(pool_size=1, base_port=8081, adapter_type="direct")
        pool.initialize()

        # Verify factory called with direct type
        mock_create_client.assert_called_once_with(
            port=8081,
            container_id="test_container_123",
            adapter_type="direct",
            input_delay=0.05,
            detection_timeout=3.0,
        )

    @patch("src.claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("src.claudelearnspokemon.emulator_pool.create_pokemon_client")
    def test_emulator_pool_with_benchflow_adapter_type(self, mock_create_client, mock_docker):
        """Test EmulatorPool with explicit benchflow adapter type."""
        # Setup mocks
        self._setup_docker_mocks(mock_docker, mock_create_client, PokemonGymAdapter)

        pool = EmulatorPool(pool_size=1, base_port=8081, adapter_type="benchflow")
        pool.initialize()

        # Verify factory called with benchflow type
        mock_create_client.assert_called_once_with(
            port=8081,
            container_id="test_container_123",
            adapter_type="benchflow",
            input_delay=0.05,
            detection_timeout=3.0,
        )

    @patch("src.claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("src.claudelearnspokemon.emulator_pool.create_pokemon_client")
    def test_emulator_pool_with_auto_adapter_type(self, mock_create_client, mock_docker):
        """Test EmulatorPool with auto adapter type (detection)."""
        # Setup mocks
        self._setup_docker_mocks(mock_docker, mock_create_client, PokemonGymAdapter)

        pool = EmulatorPool(pool_size=1, base_port=8081, adapter_type="auto")
        pool.initialize()

        # Verify factory called with auto type
        mock_create_client.assert_called_once_with(
            port=8081,
            container_id="test_container_123",
            adapter_type="auto",
            input_delay=0.05,
            detection_timeout=3.0,
        )

    @patch("src.claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("src.claudelearnspokemon.emulator_pool.create_pokemon_client")
    def test_emulator_pool_with_fallback_adapter_type(self, mock_create_client, mock_docker):
        """Test EmulatorPool with fallback adapter type."""
        # Setup mocks
        self._setup_docker_mocks(mock_docker, mock_create_client, PokemonGymClient)

        pool = EmulatorPool(pool_size=1, base_port=8081, adapter_type="fallback")
        pool.initialize()

        # Verify factory called with fallback type
        mock_create_client.assert_called_once_with(
            port=8081,
            container_id="test_container_123",
            adapter_type="fallback",
            input_delay=0.05,
            detection_timeout=3.0,
        )

    def _setup_docker_mocks(self, mock_docker, mock_create_client, client_class):
        """Helper to set up Docker and client mocks."""
        # Mock Docker client and container
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        mock_container = Mock()
        mock_container.id = "test_container_123"
        mock_container.status = "running"
        mock_docker_client.containers.run.return_value = mock_container

        # Mock container execution for health check
        mock_exec_result = Mock()
        mock_exec_result.exit_code = 0
        mock_container.exec_run.return_value = mock_exec_result

        # Mock client creation
        mock_client = Mock(spec=client_class)
        mock_client.port = 8081
        mock_client.container_id = "test_container_123"
        mock_create_client.return_value = mock_client


@pytest.mark.fast
@pytest.mark.medium
class TestEmulatorPoolScriptExecution:
    """Test EmulatorPool script execution with compatibility layer."""

    @patch("src.claudelearnspokemon.emulator_pool.create_pokemon_client")
    def test_execute_script_works_with_adapter(self, mock_create_client):
        """Test execute_script works correctly with adapter clients."""
        # Mock adapter client
        mock_adapter = Mock(spec=PokemonGymAdapter)
        mock_adapter.port = 8081
        mock_adapter.container_id = "test_container_123"
        mock_adapter.send_input.return_value = {"status": "success"}
        mock_adapter.get_state.return_value = {"location": {"map": "town"}}
        mock_create_client.return_value = mock_adapter

        # Create pool with adapter
        pool = EmulatorPool(pool_size=1, adapter_type="benchflow")

        # Mock initialized state
        pool.containers = [Mock()]  # Pretend we have containers
        pool.available_clients.put(mock_adapter)

        # Execute script
        result = pool.execute_script("A B START")

        # Verify script was executed
        assert result.success
        # With new implementation, individual inputs are sent separately
        expected_calls = [call("A"), call("B"), call("START")]
        mock_adapter.send_input.assert_has_calls(expected_calls)
        # get_state is called twice: once for initial state, once for final state
        assert mock_adapter.get_state.call_count == 2

    @patch("src.claudelearnspokemon.emulator_pool.create_pokemon_client")
    def test_execute_script_works_with_direct_client(self, mock_create_client):
        """Test execute_script works correctly with direct clients."""
        # Mock direct client
        mock_client = Mock(spec=PokemonGymClient)
        mock_client.port = 8081
        mock_client.container_id = "test_container_123"
        mock_client.send_input.return_value = {"status": "success"}
        mock_client.get_state.return_value = {"location": {"map": "town"}}
        mock_create_client.return_value = mock_client

        # Create pool with direct client
        pool = EmulatorPool(pool_size=1, adapter_type="direct")

        # Mock initialized state
        pool.containers = [Mock()]  # Pretend we have containers
        pool.available_clients.put(mock_client)

        # Execute script
        result = pool.execute_script("A B START")

        # Verify script was executed
        assert result.success
        # With new implementation, individual inputs are sent separately
        expected_calls = [call("A"), call("B"), call("START")]
        mock_client.send_input.assert_has_calls(expected_calls)
        # get_state is called twice: once for initial state, once for final state
        assert mock_client.get_state.call_count == 2


@pytest.mark.fast
@pytest.mark.medium
class TestEmulatorPoolLogging:
    """Test EmulatorPool logging with compatibility layer."""

    @patch("src.claudelearnspokemon.emulator_pool.logger")
    def test_emulator_pool_logs_adapter_type(self, mock_logger):
        """Test EmulatorPool logs adapter type during initialization."""
        _ = EmulatorPool(pool_size=2, base_port=8081, adapter_type="benchflow")

        # Verify logging includes adapter_type
        mock_logger.info.assert_called_with(
            "EmulatorPool configured: size=2, base_port=8081, "
            "image=pokemon-gym:latest, timeout=30s, adapter_type=benchflow"
        )

    @patch("src.claudelearnspokemon.emulator_pool.docker.from_env")
    @patch("src.claudelearnspokemon.emulator_pool.create_pokemon_client")
    @patch("src.claudelearnspokemon.emulator_pool.logger")
    def test_emulator_pool_logs_client_creation_success(
        self, mock_logger, mock_create_client, mock_docker
    ):
        """Test EmulatorPool logs successful client creation via compatibility layer."""
        # Setup mocks
        mock_docker_client = Mock()
        mock_docker.return_value = mock_docker_client

        mock_container = Mock()
        mock_container.id = "test_container_123"
        mock_container.status = "running"
        mock_docker_client.containers.run.return_value = mock_container

        mock_exec_result = Mock()
        mock_exec_result.exit_code = 0
        mock_container.exec_run.return_value = mock_exec_result

        mock_client = Mock(spec=PokemonGymClient)
        mock_create_client.return_value = mock_client

        # Initialize pool
        pool = EmulatorPool(pool_size=1, base_port=8081)
        pool.initialize()

        # Verify compatibility layer usage is logged
        mock_logger.info.assert_any_call(
            "Created Pokemon client via compatibility layer for port 8081"
        )
