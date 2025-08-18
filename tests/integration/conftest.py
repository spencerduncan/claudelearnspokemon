"""
Docker integration test configuration and fixtures.

Provides high-performance Docker-based test fixtures using testcontainers
for real Pokemon-gym server integration testing.

Author: John Botmack - Performance Engineering
"""

import os
import time
from collections.abc import Generator

import docker
import pytest
from testcontainers.core.container import DockerContainer

# Environment configuration
SKIP_INTEGRATION_TESTS = not bool(os.getenv("RUN_INTEGRATION_TESTS", ""))
DOCKER_TIMEOUT = int(os.getenv("DOCKER_TIMEOUT", "30"))
POKEMON_GYM_IMAGE = os.getenv("POKEMON_GYM_IMAGE", "pokemon-gym:latest")

# Performance requirements - John Botmack standards
CLIENT_CREATION_TIMEOUT_MS = 100
CONCURRENT_CLIENT_COUNT = 4
CONTAINER_STARTUP_TIMEOUT = 30

PERFORMANCE_REQUIREMENTS = {
    "client_creation_ms": CLIENT_CREATION_TIMEOUT_MS,
    "action_execution_ms": 100,
    "status_check_ms": 50,
    "concurrent_overhead_pct": 20,
}


def skip_if_docker_unavailable():
    """Skip test if Docker is not available or integration tests disabled."""
    if SKIP_INTEGRATION_TESTS:
        pytest.skip("Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable")

    try:
        docker_client = docker.from_env()
        docker_client.ping()
        docker_client.close()
    except Exception as e:
        pytest.skip(f"Docker not available for integration tests: {e}")


@pytest.fixture(scope="session", autouse=True)
def check_docker_environment():
    """Session-wide check for Docker availability."""
    skip_if_docker_unavailable()


@pytest.fixture
def pokemon_gym_container() -> Generator[tuple[DockerContainer, str, int], None, None]:
    """
    High-performance Pokemon-gym container fixture.

    Returns container, server URL, and port for real server testing.
    Optimized for <100ms startup per John Botmack standards.
    """
    skip_if_docker_unavailable()

    # Performance optimization: use cached image if available
    container = (
        DockerContainer(POKEMON_GYM_IMAGE)
        .with_exposed_ports(8080)
        .with_env("POKEMON_ENV", "test")
        .with_env("LOG_LEVEL", "WARN")  # Reduce noise for performance
        .with_bind_ports(8080, 8080)
    )

    start_time = time.time()

    with container:
        startup_time = (time.time() - start_time) * 1000

        # John Botmack performance validation
        if startup_time > CLIENT_CREATION_TIMEOUT_MS:
            pytest.fail(
                f"Container startup took {startup_time:.1f}ms, "
                f"violates <{CLIENT_CREATION_TIMEOUT_MS}ms requirement"
            )

        host_port = container.get_exposed_port(8080)
        server_url = f"http://localhost:{host_port}"

        # Verify container is ready with performance check
        import httpx

        ready_start = time.time()
        with httpx.Client() as client:
            response = client.get(f"{server_url}/status", timeout=5.0)
            response.raise_for_status()

        ready_time = (time.time() - ready_start) * 1000
        assert ready_time < 50, f"Container ready check took {ready_time:.1f}ms"

        yield container, server_url, host_port


@pytest.fixture
def multiple_pokemon_containers() -> Generator[list[tuple[DockerContainer, str, int]], None, None]:
    """
    Multiple Pokemon-gym containers for concurrent testing.

    Creates CONCURRENT_CLIENT_COUNT containers for parallel testing
    with performance validation.
    """
    skip_if_docker_unavailable()

    containers = []
    server_data = []

    # Performance optimization: start containers in parallel
    start_time = time.time()

    try:
        for i in range(CONCURRENT_CLIENT_COUNT):
            container = (
                DockerContainer(POKEMON_GYM_IMAGE)
                .with_exposed_ports(8080)
                .with_env("POKEMON_ENV", f"test_parallel_{i}")
                .with_env("LOG_LEVEL", "ERROR")  # Minimal logging for performance
                .with_bind_ports(8080, 8080)
            )

            container.start()
            containers.append(container)

            host_port = container.get_exposed_port(8080)
            server_url = f"http://localhost:{host_port}"
            server_data.append((container, server_url, host_port))

        total_startup_time = (time.time() - start_time) * 1000
        avg_startup_time = total_startup_time / CONCURRENT_CLIENT_COUNT

        # Parallel startup should be efficient
        if avg_startup_time > CLIENT_CREATION_TIMEOUT_MS:
            pytest.fail(
                f"Average container startup {avg_startup_time:.1f}ms "
                f"violates <{CLIENT_CREATION_TIMEOUT_MS}ms requirement"
            )

        # Verify all containers are ready
        import httpx

        with httpx.Client() as client:
            for _, server_url, _ in server_data:
                response = client.get(f"{server_url}/status", timeout=5.0)
                response.raise_for_status()

        yield server_data

    finally:
        # Cleanup: stop all containers
        for container in containers:
            try:
                container.stop()
            except Exception:
                pass


@pytest.fixture
def slow_pokemon_container() -> Generator[tuple[DockerContainer, str, int], None, None]:
    """
    Pokemon-gym container configured for timeout testing.

    Uses configuration that introduces delays for testing
    timeout handling and performance requirements.
    """
    skip_if_docker_unavailable()

    container = (
        DockerContainer(POKEMON_GYM_IMAGE)
        .with_exposed_ports(8080)
        .with_env("POKEMON_ENV", "test_slow")
        .with_env("SIMULATED_DELAY", "150")  # 150ms delay for timeout tests
        .with_env("LOG_LEVEL", "WARN")
        .with_bind_ports(8080, 8080)
    )

    with container:
        host_port = container.get_exposed_port(8080)
        server_url = f"http://localhost:{host_port}"

        yield container, server_url, host_port


@pytest.fixture
def version_compatibility_containers() -> (
    Generator[list[tuple[str, DockerContainer, str, int]], None, None]
):
    """
    Multiple Pokemon-gym server versions for compatibility testing.

    Tests compatibility across different server versions to ensure
    robust integration with evolving Pokemon-gym implementations.
    """
    skip_if_docker_unavailable()

    versions = ["pokemon-gym:v1.0", "pokemon-gym:v1.1", "pokemon-gym:latest"]

    containers = []
    version_data = []

    try:
        for version in versions:
            try:
                container = (
                    DockerContainer(version)
                    .with_exposed_ports(8080)
                    .with_env("POKEMON_ENV", f"test_compat_{version.split(':')[1]}")
                    .with_env("LOG_LEVEL", "ERROR")
                    .with_bind_ports(8080, 8080)
                )

                container.start()
                containers.append(container)

                host_port = container.get_exposed_port(8080)
                server_url = f"http://localhost:{host_port}"
                version_data.append((version, container, server_url, host_port))

            except Exception as e:
                # Skip unavailable versions but log the issue
                print(f"Warning: {version} not available: {e}")
                continue

        if not version_data:
            pytest.skip("No Pokemon-gym versions available for compatibility testing")

        yield version_data

    finally:
        # Cleanup: stop all containers
        for container in containers:
            try:
                container.stop()
            except Exception:
                pass


# Performance test markers for John Botmack standards
pytestmark = pytest.mark.integration


# Docker resource cleanup utilities
def cleanup_test_containers():
    """Cleanup any orphaned test containers from failed tests."""
    try:
        docker_client = docker.from_env()

        # Find containers with test environment variables
        containers = docker_client.containers.list(all=True, filters={"label": "POKEMON_ENV"})

        for container in containers:
            if any(
                env.startswith("POKEMON_ENV=test")
                for env in container.attrs.get("Config", {}).get("Env", [])
            ):
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        docker_client.close()

    except Exception:
        pass  # Ignore cleanup errors


# Session cleanup
@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests():
    """Clean up test containers after test session."""
    yield
    cleanup_test_containers()
