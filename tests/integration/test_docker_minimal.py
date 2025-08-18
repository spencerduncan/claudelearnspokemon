"""Minimal Docker integration test for Pokemon-gym containers."""

import time

import pytest
import requests
from testcontainers.compose import DockerCompose


@pytest.fixture(scope="session")
def pokemon_gym_server():
    """Start minimal Pokemon-gym server for integration testing."""
    with DockerCompose(".", compose_file_name="docker-compose.test.yml") as compose:
        # Wait for server to be ready
        port = compose.get_service_port("pokemon-gym", 8080)
        server_url = f"http://localhost:{port}"

        # Basic health check
        for _ in range(10):
            try:
                response = requests.get(f"{server_url}/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.5)

        yield server_url


@pytest.mark.medium
def test_docker_integration_basic(pokemon_gym_server):
    """Test basic Docker container connectivity."""
    response = requests.get(f"{pokemon_gym_server}/health")
    assert response.status_code == 200


@pytest.mark.medium
def test_docker_integration_performance(pokemon_gym_server):
    """Test response time meets requirements."""
    start = time.perf_counter()
    response = requests.get(f"{pokemon_gym_server}/status")
    duration_ms = (time.perf_counter() - start) * 1000

    assert response.status_code == 200
    assert duration_ms < 100  # <100ms requirement
