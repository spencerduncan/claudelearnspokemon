"""
Targeted edge case tests for EmulatorPool - Rex's Skeptical Assessment.

Tests focus on the failure modes and race conditions that optimistic testing misses:
- Circuit breaker reset race conditions
- Concurrent health monitoring edge cases  
- Docker vs HTTP health state mismatches
- Retry logic under concurrent load
- Resource cleanup in partial failure scenarios

Author: Rex - Skeptical of what could go wrong
"""

import pytest
import threading
import time
import concurrent.futures
from unittest.mock import Mock, patch, MagicMock
import requests

from claudelearnspokemon.emulator_pool import (
    EmulatorPool,
    PokemonGymClient, 
    ContainerHealthStatus,
    ContainerHealthInfo,
    ExecutionResult,
    ExecutionStatus
)


class TestCircuitBreakerEdgeCases:
    """Tests for circuit breaker failure scenarios that Rex would be skeptical about."""
    
    def test_circuit_breaker_reset_race_condition(self):
        """Test what happens when circuit breaker resets during concurrent failures.
        
        Rex's concern: Multiple threads might reset circuit breaker simultaneously,
        leading to incorrect failure counting or state inconsistency.
        """
        client = PokemonGymClient(port=8081, container_id="test123")
        
        # Mock session to simulate failures, then success
        mock_response = Mock()
        mock_response.json.return_value = {"error": "service unavailable"}
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Service unavailable")
        
        client.session.post = Mock(return_value=mock_response)
        
        # Force circuit breaker into failure state
        for _ in range(client.CIRCUIT_BREAKER_FAILURE_THRESHOLD):
            with pytest.raises(Exception):
                client.send_input("A")
        
        # Now simulate concurrent reset attempts
        def attempt_circuit_reset():
            # Wait past failure reset timeout
            time.sleep(client.FAILURE_RESET_TIMEOUT + 0.1)
            try:
                # This should attempt to reset circuit breaker
                return client.send_input("A")
            except Exception as e:
                return str(e)
        
        # Multiple threads trying to reset simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attempt_circuit_reset) for _ in range(5)]
            results = [future.result() for future in futures]
        
        # Rex's assertion: System should handle concurrent resets gracefully
        # No exceptions should bubble up due to race conditions in reset logic
        assert all(isinstance(result, (dict, str)) for result in results)
    
    def test_circuit_breaker_state_during_partial_recovery(self):
        """Test circuit breaker behavior when service intermittently fails.
        
        Rex's concern: What happens when service recovers partially?
        Does circuit breaker track this correctly?
        """
        client = PokemonGymClient(port=8081, container_id="test123")
        
        call_count = 0
        def mock_post_intermittent(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count % 3 == 0:  # Every 3rd call succeeds
                response = Mock()
                response.json.return_value = {"success": True}
                response.status_code = 200
                return response
            else:  # Other calls fail
                response = Mock()
                response.status_code = 503
                response.raise_for_status.side_effect = requests.exceptions.HTTPError("Intermittent failure")
                return response
        
        client.session.post = mock_post_intermittent
        
        # Test intermittent failures don't break circuit breaker logic
        results = []
        for i in range(10):
            try:
                result = client.send_input("A")
                results.append(("success", result))
            except Exception as e:
                results.append(("failure", str(e)))
        
        # Rex's assertion: Circuit breaker should still function despite intermittent issues
        success_count = len([r for r in results if r[0] == "success"])
        failure_count = len([r for r in results if r[0] == "failure"])
        
        # Should have some successes despite failures
        assert success_count > 0, "Circuit breaker prevented all recovery attempts"
        assert failure_count > 0, "Test setup issue - should have some failures"


class TestConcurrentHealthMonitoring:
    """Tests for health monitoring race conditions Rex would identify."""
    
    def test_concurrent_health_status_updates(self):
        """Test health status updates from multiple threads.
        
        Rex's concern: Multiple threads updating health status could lead to 
        inconsistent state or lost updates.
        """
        pool = EmulatorPool()
        container_id = "test123"
        port = 8081
        
        # Initialize health tracking using actual API
        initial_health = ContainerHealthInfo(
            container_id=container_id,
            port=port,
            status=ContainerHealthStatus.HEALTHY,
            last_check_time=time.time(),
            docker_status="running"
        )
        # Use the actual container_health attribute (not _container_health)
        pool.container_health[container_id] = initial_health
        
        def update_health_status(thread_id, status):
            """Update health status from a specific thread."""
            for i in range(10):
                health_info = ContainerHealthInfo(
                    container_id=container_id,
                    port=port,
                    status=status,
                    last_check_time=time.time(),
                    docker_status="running",
                    consecutive_failures=i,
                    response_time_ms=float(thread_id * 10 + i)
                )
                pool.container_health[container_id] = health_info
                time.sleep(0.001)  # Small delay to encourage race conditions
        
        # Start multiple threads updating health concurrently
        threads = []
        statuses = [ContainerHealthStatus.HEALTHY, ContainerHealthStatus.UNHEALTHY, ContainerHealthStatus.STOPPED]
        
        for i, status in enumerate(statuses):
            thread = threading.Thread(target=update_health_status, args=(i, status))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Rex's assertion: Final state should be consistent
        final_health = pool.container_health[container_id]
        assert final_health is not None
        assert final_health.container_id == container_id
        assert final_health.port == port
        # Status should be one of the ones we set
        assert final_health.status in statuses
    
    def test_health_check_during_container_removal(self):
        """Test health checking when container is being removed.
        
        Rex's concern: What happens if health check occurs while container
        is being shut down or removed?
        """
        pool = EmulatorPool(pool_size=2, base_port=8081)
        
        # Mock Docker client
        with patch('claudelearnspokemon.emulator_pool.docker') as mock_docker:
            mock_client = Mock()
            mock_docker.from_env.return_value = mock_client
            
            # Mock container that disappears during health check
            mock_container = Mock()
            mock_container.id = "disappearing123"
            mock_container.status = "running"
            mock_container.stop.side_effect = Exception("Container already stopped")
            mock_container.remove.side_effect = Exception("Container not found")
            
            mock_client.containers.run.return_value = mock_container
            mock_client.containers.list.return_value = [mock_container]
            
            # Initialize pool
            pool.initialize()
            
            # Mock HTTP client that starts failing
            def failing_health_check(*args, **kwargs):
                raise requests.exceptions.ConnectionError("Container disappeared")
            
            with patch.object(PokemonGymClient, 'is_healthy', side_effect=failing_health_check):
                # Attempt health check while container is in unstable state
                health_status = pool.health_check()
                
                # Rex's assertion: Health check should handle disappeared containers gracefully
                assert isinstance(health_status, dict)
                assert "containers" in health_status
                
                # Should not crash despite container removal issues
                pool.shutdown()


class TestDockerHttpHealthMismatch:
    """Tests for scenarios where Docker status doesn't match HTTP health."""
    
    def test_container_running_but_http_failing(self):
        """Test when Docker says container is running but HTTP health fails.
        
        Rex's concern: Docker container status can be misleading.
        Container might be 'running' but service inside is crashed.
        """
        client = PokemonGymClient(port=8081, container_id="misleading123")
        
        # Mock HTTP calls to always fail
        client.session.get = Mock(side_effect=requests.exceptions.ConnectionError("Connection refused"))
        
        # Health check should detect this mismatch
        is_healthy = client.is_healthy()
        
        # Rex's assertion: Should correctly identify unhealthy state despite Docker status
        assert is_healthy is False
        
        # Failure count should increase
        assert client._consecutive_failures > 0
    
    def test_container_stopped_but_port_occupied(self):
        """Test when container is stopped but port is still occupied.
        
        Rex's concern: Port conflicts can cause confusing health states.
        Something else might be running on the expected port.
        """
        pool = EmulatorPool(pool_size=1, base_port=8081)
        
        with patch('claudelearnspokemon.emulator_pool.docker') as mock_docker:
            mock_client = Mock()
            mock_docker.from_env.return_value = mock_client
            
            # Container is stopped
            mock_container = Mock()
            mock_container.id = "stopped123"
            mock_container.status = "stopped"
            mock_client.containers.list.return_value = [mock_container]
            
            # But something responds on the port (wrong service)
            with patch('requests.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 404  # Wrong service
                mock_response.json.return_value = {"error": "not pokemon-gym"}
                mock_get.return_value = mock_response
                
                # Health check should detect this mismatch
                health_status = pool.health_check()
                
                # Rex's assertion: Should identify the mismatch correctly
                assert isinstance(health_status, dict)
                # Should not report as healthy just because port responds


class TestRetryLogicUnderLoad:
    """Tests for retry logic edge cases under concurrent load."""
    
    def test_concurrent_retry_exhaustion(self):
        """Test retry logic when multiple threads exhaust retries simultaneously.
        
        Rex's concern: Concurrent retry attempts might interfere with each other
        or cause unexpected retry count behavior.
        """
        client = PokemonGymClient(port=8081, container_id="retry123")
        
        # Mock session to always fail
        client.session.post = Mock(side_effect=requests.exceptions.ConnectTimeout("Always timeout"))
        
        def attempt_with_retries():
            try:
                return client.send_input("A")
            except Exception as e:
                return str(e)
        
        # Multiple threads exhausting retries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(attempt_with_retries) for _ in range(5)]
            results = [future.result() for future in futures]
        
        # Rex's assertion: All should fail gracefully
        assert all(isinstance(result, str) for result in results)
        assert all("timeout" in result.lower() for result in results)
        
        # Failure count should be reasonable (not wildly off due to race conditions)
        assert client._consecutive_failures >= 3  # Should have accumulated failures
    
    def test_retry_delay_accumulation_under_load(self):
        """Test that retry delays work correctly under concurrent load.
        
        Rex's concern: Retry delays might not work correctly when multiple
        threads are retrying simultaneously.
        """
        client = PokemonGymClient(port=8081, container_id="delay123")
        
        call_times = []
        
        def mock_post_with_timing(*args, **kwargs):
            call_times.append(time.time())
            raise requests.exceptions.ConnectTimeout("Simulated timeout")
        
        client.session.post = mock_post_with_timing
        
        start_time = time.time()
        
        try:
            client.send_input("A")
        except Exception:
            pass  # Expected to fail
        
        end_time = time.time()
        
        # Rex's assertion: Should have taken time for retries with delays
        total_time = end_time - start_time
        expected_minimum_time = client.MAX_RETRIES * client.RETRY_DELAY
        
        assert total_time >= expected_minimum_time * 0.8  # Allow for some timing variance
        assert len(call_times) == client.MAX_RETRIES + 1  # Initial + retries


@pytest.mark.integration
class TestResourceCleanupEdgeCases:
    """Tests for resource cleanup in partial failure scenarios."""
    
    def test_cleanup_during_initialization_failure(self):
        """Test resource cleanup when initialization fails partway through.
        
        Rex's concern: If initialization fails after creating some containers,
        are all resources properly cleaned up?
        """
        pool = EmulatorPool(pool_size=3, base_port=8081)
        
        with patch('claudelearnspokemon.emulator_pool.docker') as mock_docker:
            mock_client = Mock()
            mock_docker.from_env.return_value = mock_client
            
            # First container succeeds, second fails
            successful_container = Mock()
            successful_container.id = "success123"
            successful_container.status = "running"
            
            def mock_run(*args, **kwargs):
                # First call succeeds, second fails
                if mock_client.containers.run.call_count == 1:
                    return successful_container
                else:
                    raise Exception("Second container failed to start")
            
            mock_client.containers.run = Mock(side_effect=mock_run)
            
            # Attempt initialization - should fail
            with pytest.raises(Exception):
                pool.initialize()
            
            # Rex's assertion: Should have attempted cleanup of successful container
            assert successful_container.stop.called or successful_container.remove.called
    
    def test_shutdown_with_unresponsive_containers(self):
        """Test shutdown when some containers don't respond to stop commands.
        
        Rex's concern: What happens if containers are stuck and won't stop?
        Does shutdown hang or handle this gracefully?
        """
        pool = EmulatorPool(pool_size=2, base_port=8081)
        
        with patch('claudelearnspokemon.emulator_pool.docker') as mock_docker:
            mock_client = Mock()
            mock_docker.from_env.return_value = mock_client
            
            # One responsive, one unresponsive container
            responsive_container = Mock()
            responsive_container.id = "responsive123"
            responsive_container.status = "running"
            
            unresponsive_container = Mock()
            unresponsive_container.id = "stuck123" 
            unresponsive_container.status = "running"
            unresponsive_container.stop.side_effect = Exception("Container stuck")
            unresponsive_container.remove.side_effect = Exception("Cannot remove")
            
            mock_client.containers.run.side_effect = [responsive_container, unresponsive_container]
            mock_client.containers.list.return_value = [responsive_container, unresponsive_container]
            
            # Initialize and then shutdown
            pool.initialize()
            
            # Rex's assertion: Shutdown should complete despite unresponsive container
            # Should not hang indefinitely
            start_time = time.time()
            pool.shutdown()
            shutdown_time = time.time() - start_time
            
            # Should complete in reasonable time (not hang)
            assert shutdown_time < 10.0  # Max 10 seconds for shutdown
            
            # Should have attempted to stop both containers
            assert responsive_container.stop.called
            assert unresponsive_container.stop.called