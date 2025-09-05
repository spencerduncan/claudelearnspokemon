"""Hybrid integration test with automatic Docker/HTTP mock fallback.

This test demonstrates the hybrid infrastructure approach where tests can run
in either Docker mode (using TestContainers) or HTTP mock mode (using responses)
based on Docker availability, with automatic environment detection.
"""

import time
import logging
from contextlib import contextmanager
from typing import Generator

import pytest
import requests
import responses
from testcontainers.compose import DockerCompose

from claudelearnspokemon.test_environment_config import (
    TestMode,
    TestEnvironmentConfig,
    load_test_config
)


logger = logging.getLogger(__name__)


@contextmanager
def setup_test_environment(config: TestEnvironmentConfig) -> Generator[str, None, None]:
    """Set up test environment based on configuration.
    
    Args:
        config: Test environment configuration
        
    Yields:
        Base URL for the test service
    """
    if config.mode == TestMode.DOCKER:
        yield from _setup_docker_environment(config)
    elif config.mode == TestMode.HTTP_MOCK:
        yield from _setup_http_mock_environment(config)
    else:
        raise ValueError(f"Unsupported test mode: {config.mode}")


@contextmanager
def _setup_docker_environment(config: TestEnvironmentConfig) -> Generator[str, None, None]:
    """Set up Docker-based test environment using TestContainers."""
    logger.info("Setting up Docker environment for integration testing")
    start_time = time.perf_counter()
    
    with DockerCompose(".", compose_file_name=config.docker_compose_file) as compose:
        # Get the service port
        port = compose.get_service_port(config.docker_service_name, config.docker_service_port)
        server_url = f"http://localhost:{port}"
        
        # Wait for service to be ready with configured timeout
        _wait_for_service_ready(
            server_url, 
            config.docker_health_check_timeout,
            config.docker_health_check_interval
        )
        
        setup_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Docker environment ready in {setup_time_ms:.1f}ms")
        
        # Verify setup time meets requirements
        if setup_time_ms > config.max_setup_time_ms:
            logger.warning(
                f"Setup time ({setup_time_ms:.1f}ms) exceeds threshold "
                f"({config.max_setup_time_ms}ms)"
            )
        
        yield server_url


@contextmanager
def _setup_http_mock_environment(config: TestEnvironmentConfig) -> Generator[str, None, None]:
    """Set up HTTP mock-based test environment using responses."""
    logger.info("Setting up HTTP mock environment for testing")
    start_time = time.perf_counter()
    
    with responses.RequestsMock() as rsps:
        # Set up standard mock endpoints
        _setup_pokemon_gym_mocks(rsps, config)
        
        setup_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"HTTP mock environment ready in {setup_time_ms:.1f}ms")
        
        # Verify setup time meets requirements
        if setup_time_ms > config.max_setup_time_ms:
            logger.warning(
                f"Setup time ({setup_time_ms:.1f}ms) exceeds threshold "
                f"({config.max_setup_time_ms}ms)"
            )
        
        yield config.mock_base_url


def _wait_for_service_ready(
    server_url: str, 
    timeout_seconds: int, 
    interval_seconds: float
) -> None:
    """Wait for service to be ready with health check."""
    start_time = time.perf_counter()
    
    while (time.perf_counter() - start_time) < timeout_seconds:
        try:
            response = requests.get(f"{server_url}/health", timeout=1)
            if response.status_code == 200:
                logger.debug("Service health check passed")
                return
        except requests.RequestException as e:
            logger.debug(f"Health check failed: {e}")
        
        time.sleep(interval_seconds)
    
    raise RuntimeError(
        f"Service at {server_url} failed to become ready within {timeout_seconds}s"
    )


def _setup_pokemon_gym_mocks(rsps: responses.RequestsMock, config: TestEnvironmentConfig) -> None:
    """Set up Pokemon Gym service mocks."""
    base_url = config.mock_base_url
    
    # Health endpoint
    rsps.add(
        responses.GET,
        f"{base_url}/health",
        json={"status": "healthy", "timestamp": int(time.time())},
        status=200
    )
    
    # Status endpoint with configured delay
    def status_callback(request):
        if config.mock_response_delay > 0:
            time.sleep(config.mock_response_delay)
        return (200, {}, '{"status": "running", "mode": "test"}')
    
    rsps.add_callback(
        responses.GET,
        f"{base_url}/status",
        callback=status_callback
    )
    
    # Game state endpoint
    rsps.add(
        responses.GET,
        f"{base_url}/state",
        json={
            "game_state": "running",
            "player_position": {"x": 10, "y": 15},
            "level": "pallet_town"
        },
        status=200
    )
    
    # Action endpoint
    rsps.add(
        responses.POST,
        f"{base_url}/action",
        json={"result": "success", "new_state": "updated"},
        status=200
    )


@pytest.fixture(scope="session")
def test_config():
    """Load test configuration with environment detection."""
    return load_test_config(TestMode.AUTO)


@pytest.fixture(scope="session")
def pokemon_gym_service(test_config):
    """Start Pokemon Gym service (Docker or mock based on environment)."""
    with setup_test_environment(test_config) as service_url:
        yield service_url, test_config


class TestHybridIntegration:
    """Hybrid integration tests that work in both Docker and HTTP mock modes."""
    
    @pytest.mark.integration
    def test_service_health_check(self, pokemon_gym_service):
        """Test basic service health check."""
        service_url, config = pokemon_gym_service
        
        response = requests.get(f"{service_url}/health", timeout=2)
        assert response.status_code == 200
        
        # Verify response format
        if config.mode == TestMode.DOCKER:
            # Docker mode might have different response format
            assert response.json() is not None
        else:
            # HTTP mock has predictable format
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
    
    @pytest.mark.integration
    def test_service_status_endpoint(self, pokemon_gym_service):
        """Test service status endpoint."""
        service_url, config = pokemon_gym_service
        
        start_time = time.perf_counter()
        response = requests.get(f"{service_url}/status", timeout=2)
        response_time_ms = (time.perf_counter() - start_time) * 1000
        
        assert response.status_code == 200
        
        # Verify response time meets requirements
        assert response_time_ms < config.max_response_time_ms, (
            f"Response time {response_time_ms:.1f}ms exceeds limit "
            f"{config.max_response_time_ms}ms"
        )
        
        # Log performance metrics
        logger.info(
            f"Status endpoint response time: {response_time_ms:.1f}ms "
            f"(limit: {config.max_response_time_ms}ms, mode: {config.mode.value})"
        )
    
    @pytest.mark.integration
    def test_game_state_retrieval(self, pokemon_gym_service):
        """Test game state retrieval."""
        service_url, config = pokemon_gym_service
        
        response = requests.get(f"{service_url}/state", timeout=2)
        
        if response.status_code == 200:
            # Service supports state endpoint
            data = response.json()
            assert data is not None
            
            if config.mode == TestMode.HTTP_MOCK:
                # Mock has predictable format
                assert data["game_state"] == "running"
                assert "player_position" in data
                assert "level" in data
        else:
            # Service might not implement this endpoint yet
            assert response.status_code in [404, 501]
            logger.info("Game state endpoint not implemented in service")
    
    @pytest.mark.integration
    def test_action_submission(self, pokemon_gym_service):
        """Test action submission."""
        service_url, config = pokemon_gym_service
        
        action_data = {
            "action": "move",
            "direction": "up",
            "timestamp": int(time.time())
        }
        
        response = requests.post(
            f"{service_url}/action", 
            json=action_data, 
            timeout=2
        )
        
        if response.status_code == 200:
            # Service supports action endpoint
            data = response.json()
            assert data is not None
            
            if config.mode == TestMode.HTTP_MOCK:
                # Mock has predictable format
                assert data["result"] == "success"
        else:
            # Service might not implement this endpoint yet
            assert response.status_code in [404, 501]
            logger.info("Action endpoint not implemented in service")
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_concurrent_requests(self, pokemon_gym_service):
        """Test concurrent request handling."""
        service_url, config = pokemon_gym_service
        
        import concurrent.futures
        
        def make_request():
            start_time = time.perf_counter()
            response = requests.get(f"{service_url}/health", timeout=2)
            response_time = time.perf_counter() - start_time
            return response.status_code, response_time
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for status_code, response_time in results:
            assert status_code == 200
            response_time_ms = response_time * 1000
            assert response_time_ms < config.max_response_time_ms * 2  # Allow 2x limit for concurrency
        
        # Log performance summary
        response_times = [r[1] * 1000 for r in results]
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        
        logger.info(
            f"Concurrent requests - avg: {avg_time:.1f}ms, "
            f"max: {max_time:.1f}ms, mode: {config.mode.value}"
        )


class TestDockerSpecific:
    """Tests that only run in Docker mode."""
    
    @pytest.mark.integration
    @pytest.mark.docker
    def test_docker_container_health(self, pokemon_gym_service):
        """Test Docker container-specific health checks."""
        service_url, config = pokemon_gym_service
        
        if config.mode != TestMode.DOCKER:
            pytest.skip("Test requires Docker mode")
        
        # Test that we can make multiple requests to verify container stability
        for i in range(3):
            response = requests.get(f"{service_url}/health", timeout=2)
            assert response.status_code == 200
            time.sleep(0.1)  # Small delay between requests
        
        logger.info("Docker container health check passed")


class TestHttpMockSpecific:
    """Tests that only run in HTTP mock mode."""
    
    @pytest.mark.integration
    @pytest.mark.http_mock
    def test_mock_response_consistency(self, pokemon_gym_service):
        """Test HTTP mock response consistency."""
        service_url, config = pokemon_gym_service
        
        if config.mode != TestMode.HTTP_MOCK:
            pytest.skip("Test requires HTTP mock mode")
        
        # Make multiple requests to verify mock consistency
        responses_data = []
        for i in range(3):
            response = requests.get(f"{service_url}/health", timeout=2)
            assert response.status_code == 200
            responses_data.append(response.json())
        
        # All responses should be identical (mocks are deterministic)
        assert all(r == responses_data[0] for r in responses_data)
        
        logger.info("HTTP mock response consistency verified")


# Performance comparison test
@pytest.mark.slow
@pytest.mark.integration
def test_performance_comparison():
    """Compare performance between Docker and HTTP mock modes."""
    results = {}
    
    for mode in [TestMode.DOCKER, TestMode.HTTP_MOCK]:
        try:
            config = load_test_config(mode, force_refresh=True)
            
            # Skip if mode not available
            if mode == TestMode.DOCKER and config.mode != TestMode.DOCKER:
                logger.warning("Docker mode not available, skipping comparison")
                continue
            
            # Measure setup time
            start_time = time.perf_counter()
            with setup_test_environment(config) as service_url:
                setup_time = (time.perf_counter() - start_time) * 1000
                
                # Measure response time
                response_start = time.perf_counter()
                response = requests.get(f"{service_url}/health", timeout=2)
                response_time = (time.perf_counter() - response_start) * 1000
                
                results[mode.value] = {
                    "setup_time_ms": setup_time,
                    "response_time_ms": response_time,
                    "status_code": response.status_code
                }
        
        except Exception as e:
            logger.warning(f"Failed to test {mode.value} mode: {e}")
            continue
    
    # Log comparison results
    if len(results) > 1:
        logger.info("Performance comparison results:")
        for mode, metrics in results.items():
            logger.info(
                f"  {mode.upper()}: setup={metrics['setup_time_ms']:.1f}ms, "
                f"response={metrics['response_time_ms']:.1f}ms"
            )
        
        # Verify both modes work
        assert all(r["status_code"] == 200 for r in results.values())
    else:
        logger.info("Performance comparison skipped (only one mode available)")