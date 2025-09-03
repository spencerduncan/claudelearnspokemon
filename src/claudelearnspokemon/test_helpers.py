"""
Common test helper utilities to reduce duplication across test files.

This module provides standardized test patterns, mocks, and assertions
used throughout the test suite, particularly for Docker/emulator testing.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock, patch

import pytest
import responses


@dataclass
class MockContainerConfig:
    """Configuration for mock container creation."""

    container_id: str
    port: int
    status: str = "running"
    health_check_response: bytes = b"health_check"
    health_check_exit_code: int = 0
    should_fail: bool = False
    failure_exception: Exception | None = None


@dataclass
class MockDockerSetup:
    """Configuration for Docker mock setup."""

    containers: list[MockContainerConfig] = field(default_factory=list)
    docker_available: bool = True
    docker_exception: Exception | None = None
    cleanup_failures: list[int] = field(default_factory=list)  # Container indices that fail cleanup


class DockerTestHelpers:
    """Helper utilities for Docker-related testing."""

    @staticmethod
    def create_mock_container(config: MockContainerConfig) -> Mock:
        """Create a mock container with standard behavior."""
        container = Mock()
        container.id = config.container_id
        container.status = config.status

        # Configure health check
        container.exec_run.return_value = Mock(
            exit_code=config.health_check_exit_code, output=config.health_check_response
        )

        # Configure stop behavior
        if config.container_id in []:  # Will be set by test setup
            container.stop.side_effect = Exception("Stop failed")
        else:
            container.stop.return_value = None

        return container

    @staticmethod
    def setup_docker_mock(docker_mock: Mock, setup: MockDockerSetup) -> list[Mock]:
        """Set up Docker mock with containers."""
        if not setup.docker_available:
            if setup.docker_exception:
                docker_mock.side_effect = setup.docker_exception
            return []

        # Create mock Docker client
        mock_client = Mock()
        docker_mock.return_value = mock_client

        # Create mock containers
        mock_containers = []
        container_side_effects = []

        for i, config in enumerate(setup.containers):
            container = DockerTestHelpers.create_mock_container(config)
            mock_containers.append(container)

            # Configure cleanup failures
            if i in setup.cleanup_failures:
                container.stop.side_effect = Exception("Container stop failed")

            if config.should_fail and config.failure_exception:
                container_side_effects.append(config.failure_exception)
            else:
                container_side_effects.append(container)

        # Set up container.run side effects
        if len(container_side_effects) == 1:
            mock_client.containers.run.return_value = container_side_effects[0]
        else:
            mock_client.containers.run.side_effect = container_side_effects

        return mock_containers

    @staticmethod
    @contextmanager
    def docker_test_environment(setup: MockDockerSetup):
        """Context manager for Docker test environment."""
        with patch("claudelearnspokemon.emulator_pool.docker.from_env") as mock_docker:
            mock_containers = DockerTestHelpers.setup_docker_mock(mock_docker, setup)
            yield mock_docker, mock_containers


class EmulatorTestHelpers:
    """Helper utilities for emulator pool testing."""

    @staticmethod
    def create_standard_container_configs(
        count: int = 4, base_port: int = 8081, all_healthy: bool = True
    ) -> list[MockContainerConfig]:
        """Create standard container configurations."""
        configs = []
        for i in range(count):
            config = MockContainerConfig(
                container_id=f"container_{i+1}",
                port=base_port + i,
                status="running" if all_healthy else ("starting" if i == 0 else "running"),
                health_check_exit_code=0 if all_healthy else (1 if i == 0 else 0),
            )
            configs.append(config)
        return configs

    @staticmethod
    def create_failure_scenario(
        success_count: int = 2, failure_type: str = "port_conflict", total_count: int = 4
    ) -> MockDockerSetup:
        """Create a failure scenario for testing."""
        configs = EmulatorTestHelpers.create_standard_container_configs(success_count)

        # Add failure configuration
        failure_exceptions = {
            "port_conflict": Exception("Port already in use"),
            "resource_exhausted": Exception("Resource exhausted"),
            "image_not_found": Exception("Image not found"),
        }

        if success_count < total_count:
            failure_config = MockContainerConfig(
                container_id="container_fail",
                port=8081 + success_count,
                should_fail=True,
                failure_exception=failure_exceptions.get(failure_type, Exception("Unknown error")),
            )
            configs.append(failure_config)

        return MockDockerSetup(containers=configs)


class ResponsesTestHelpers:
    """Helper utilities for HTTP responses testing."""

    @staticmethod
    def setup_emulator_responses(port: int, endpoints: dict[str, Any] | None = None) -> None:
        """Set up standard emulator HTTP responses."""
        base_url = f"http://localhost:{port}"

        # Default endpoints
        default_endpoints = {
            "/state": {"status": "success", "game_state": {"x": 0, "y": 0}},
            "/input": {"status": "success", "frames_advanced": 1},
            "/reset": {"status": "success", "message": "Game reset"},
            "/health": {"status": "healthy"},
        }

        endpoints = endpoints or default_endpoints

        for endpoint, response_data in endpoints.items():
            url = f"{base_url}{endpoint}"
            responses.add(
                responses.GET if endpoint in ["/state", "/health"] else responses.POST,
                url,
                json=response_data,
                status=200,
            )

    @staticmethod
    @contextmanager
    def emulator_responses_context(port: int, endpoints: dict[str, Any] | None = None):
        """Context manager for emulator responses."""
        with responses.RequestsMock() as rsps:
            ResponsesTestHelpers.setup_emulator_responses(port, endpoints)
            yield rsps


class AssertionHelpers:
    """Helper utilities for common test assertions."""

    @staticmethod
    def assert_container_started_correctly(
        mock_client: Mock, expected_ports: list[int], expected_image: str = "pokemon-gym:latest"
    ) -> None:
        """Assert containers were started with correct parameters."""
        assert mock_client.containers.run.call_count == len(expected_ports)

        for i, call in enumerate(mock_client.containers.run.call_args_list):
            args, kwargs = call
            assert kwargs["image"] == expected_image
            assert kwargs["ports"] == {"8080/tcp": expected_ports[i]}
            assert kwargs["detach"] is True
            assert kwargs["remove"] is True
            assert kwargs["name"] == f"pokemon-emulator-{expected_ports[i]}"

    @staticmethod
    def assert_containers_stopped(mock_containers: list[Mock]) -> None:
        """Assert all containers were stopped."""
        for container in mock_containers:
            container.stop.assert_called_once()

    @staticmethod
    def assert_health_checks_performed(mock_containers: list[Mock]) -> None:
        """Assert health checks were performed on containers."""
        for container in mock_containers:
            container.exec_run.assert_called()

    @staticmethod
    def assert_metrics_recorded(component: Any, expected_metrics: dict[str, int | float]) -> None:
        """Assert expected metrics were recorded."""
        if hasattr(component, "get_metrics"):
            metrics = component.get_metrics()
            for metric_name, expected_value in expected_metrics.items():
                if isinstance(expected_value, int):
                    assert metrics.get(metric_name, 0) >= expected_value
                else:
                    assert abs(metrics.get(metric_name, 0) - expected_value) < 0.01


class TestDataHelpers:
    """Helper utilities for creating test data."""

    @staticmethod
    def create_game_state(x: int = 0, y: int = 0, map_id: int = 1, **kwargs) -> dict[str, Any]:
        """Create standard game state for testing."""
        state = {
            "x": x,
            "y": y,
            "map_id": map_id,
            "timestamp": time.time(),
        }
        state.update(kwargs)
        return state

    @staticmethod
    def create_execution_result(
        success: bool = True, duration_ms: int = 100, **kwargs
    ) -> dict[str, Any]:
        """Create execution result for testing."""
        result = {
            "success": success,
            "execution_time": duration_ms / 1000.0,
            "duration_ms": duration_ms,
            "final_state": TestDataHelpers.create_game_state(),
            "timestamp": time.time(),
        }
        result.update(kwargs)
        return result

    @staticmethod
    def create_strategy_response(
        experiments: list[str] | None = None, confidence: float = 0.8, **kwargs
    ) -> dict[str, Any]:
        """Create strategy response for testing."""
        response = {
            "strategy_id": f"test_strategy_{int(time.time())}",
            "experiments": experiments or ["PRESS_A", "PRESS_START"],
            "directives": ["Test directive"],
            "confidence": confidence,
            "timestamp": time.time(),
        }
        response.update(kwargs)
        return response


class FixtureHelpers:
    """Helper utilities for creating pytest fixtures."""

    @staticmethod
    def create_emulator_pool_fixture():
        """Create standard emulator pool fixture."""

        def fixture_func():
            try:
                from claudelearnspokemon.emulator_pool import EmulatorPool  # type: ignore[import-not-found]
                return EmulatorPool(startup_timeout=5)
            except ImportError:
                # Fallback if emulator_pool is not available
                return None

        return fixture_func

    @staticmethod
    def create_mock_components_fixture():
        """Create fixture with commonly mocked components."""

        def fixture_func():
            return {
                "claude_manager": Mock(),
                "circuit_breaker": Mock(),
                "response_cache": Mock(),
                "logger": Mock(),
            }

        return fixture_func


class ParametrizedTestHelpers:
    """Helper utilities for parametrized tests."""

    @staticmethod
    def container_count_params():
        """Common container count parameters."""
        return pytest.mark.parametrize("container_count", [1, 2, 4, 8])

    @staticmethod
    def priority_params():
        """Common priority parameters."""
        return pytest.mark.parametrize("priority", ["LOW", "NORMAL", "HIGH", "CRITICAL"])

    @staticmethod
    def failure_scenario_params():
        """Common failure scenario parameters."""
        return pytest.mark.parametrize(
            "failure_type,expected_error",
            [
                ("port_conflict", "Port already in use"),
                ("resource_exhausted", "Resource exhausted"),
                ("image_not_found", "Image not found"),
            ],
        )


class BenchmarkHelpers:
    """Helper utilities for performance testing."""

    @staticmethod
    @contextmanager
    def performance_timer():
        """Context manager for timing operations."""
        start_time = time.time()
        yield
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        print(f"Operation took {duration_ms:.2f}ms")

    @staticmethod
    def assert_performance_requirement(
        duration_ms: float, max_duration_ms: float, operation_name: str = "operation"
    ):
        """Assert performance requirement is met."""
        assert duration_ms <= max_duration_ms, (
            f"{operation_name} took {duration_ms:.2f}ms, "
            f"exceeding requirement of {max_duration_ms}ms"
        )


# Convenience decorators for common test patterns
def docker_test(setup: MockDockerSetup | None = None):
    """Decorator for Docker tests."""

    def decorator(test_func):
        def wrapper(*args, **kwargs):
            test_setup = setup or MockDockerSetup(
                containers=EmulatorTestHelpers.create_standard_container_configs()
            )
            with DockerTestHelpers.docker_test_environment(test_setup) as (mock_docker, containers):
                return test_func(
                    *args, mock_docker=mock_docker, mock_containers=containers, **kwargs
                )

        return wrapper

    return decorator


def emulator_responses_test(port: int = 8081, endpoints: dict[str, Any] | None = None):
    """Decorator for emulator HTTP response tests."""

    def decorator(test_func):
        def wrapper(*args, **kwargs):
            with ResponsesTestHelpers.emulator_responses_context(port, endpoints):
                return test_func(*args, **kwargs)

        return wrapper

    return decorator
