"""
Docker-based integration tests with real Pokemon-gym server instances.

Tests basic functionality using actual Docker containers with real
network communication, validating production-like scenarios.

Author: John Botmack - Performance Engineering
"""

import time

import httpx
import pytest

from claudelearnspokemon.pokemon_gym_adapter import PokemonGymAdapter, PokemonGymAdapterError
from claudelearnspokemon.pokemon_gym_factory import create_pokemon_client


@pytest.mark.integration
@pytest.mark.medium
class TestRealServerIntegration:
    """Integration tests with actual Pokemon-gym Docker containers."""

    def test_client_creation_with_real_docker_server(self, pokemon_gym_container):
        """Test client creation with real Docker Pokemon-gym server."""
        container, server_url, port = pokemon_gym_container

        # Performance measurement - John Botmack standards
        start_time = time.time()

        client = create_pokemon_client(
            port=port, container_id=container.get_wrapped_container().id, adapter_type="auto"
        )

        creation_time = (time.time() - start_time) * 1000

        # Verify client created within performance requirements
        assert (
            creation_time < 100
        ), f"Client creation took {creation_time:.1f}ms, violates <100ms requirement"

        assert client is not None
        assert hasattr(client, "port")
        assert client.port == port

        # Test real connection to Docker container
        if hasattr(client, "get_session_status"):
            # Adapter client
            status = client.get_session_status()
            assert "active" in status or "status" in status
        else:
            # Direct client - test basic connectivity
            assert hasattr(client, "send_input")

    def test_real_session_lifecycle_with_docker(self, pokemon_gym_container):
        """Test complete session lifecycle with real Docker server."""
        container, server_url, port = pokemon_gym_container

        adapter = PokemonGymAdapter(
            port=port, container_id=container.get_wrapped_container().id, server_url=server_url
        )

        try:
            # Initialize session with real Docker server
            init_start = time.time()
            _ = adapter.initialize_session({"game": "pokemon_red"})  # Result unused in test
            init_time = (time.time() - init_start) * 1000

            assert init_time < 1000, f"Session initialization took {init_time:.1f}ms"
            assert adapter._session_initialized

            # Execute action with real Docker server
            action_start = time.time()
            _ = adapter.execute_action("A B")  # Result unused in test
            action_time = (time.time() - action_start) * 1000

            # Critical performance requirement
            assert (
                action_time < 100
            ), f"Action execution took {action_time:.1f}ms, violates <100ms requirement"

            # Stop session
            _ = adapter.stop_session()  # Result unused in test
            assert not adapter._session_initialized

        finally:
            adapter.close()

    def test_real_network_error_handling_with_docker(self, pokemon_gym_container):
        """Test network error handling with real Docker container."""
        container, server_url, port = pokemon_gym_container

        # Test with valid container first
        adapter = PokemonGymAdapter(
            port=port, container_id=container.get_wrapped_container().id, server_url=server_url
        )

        try:
            adapter.initialize_session()

            # Test with stopped container (simulate network failure)
            container.stop()

            # Should handle network errors gracefully
            with pytest.raises(
                (PokemonGymAdapterError, httpx.ConnectError, httpx.TimeoutException)
            ):
                adapter.execute_action("A")

        finally:
            adapter.close()

    def test_concurrent_clients_with_real_docker_servers(self, multiple_pokemon_containers):
        """Test concurrent clients with multiple real Docker servers."""
        server_data = multiple_pokemon_containers

        if len(server_data) < 4:
            pytest.skip("Need at least 4 containers for concurrent testing")

        adapters = []

        try:
            # Create concurrent adapters
            for _i, (container, server_url, port) in enumerate(server_data):
                adapter = PokemonGymAdapter(
                    port=port,
                    container_id=container.get_wrapped_container().id,
                    server_url=server_url,
                )
                adapters.append(adapter)

            import queue
            import threading

            results_queue = queue.Queue()

            def worker(adapter_idx, adapter):
                """Worker thread for concurrent testing."""
                try:
                    # Initialize session
                    adapter.initialize_session()

                    # Perform multiple operations
                    for i in range(3):
                        start_time = time.time()
                        _ = adapter.execute_action(f"A_{adapter_idx}_{i}")  # Result unused in test
                        operation_time = (time.time() - start_time) * 1000

                        results_queue.put((adapter_idx, i, operation_time, None))

                        # Verify performance requirement
                        if operation_time >= 100:
                            results_queue.put(
                                (
                                    adapter_idx,
                                    i,
                                    operation_time,
                                    f"Performance violation: {operation_time:.1f}ms",
                                )
                            )

                except Exception as e:
                    results_queue.put((adapter_idx, -1, 0, str(e)))

            # Start concurrent threads
            threads = []
            for i, adapter in enumerate(adapters):
                thread = threading.Thread(target=worker, args=(i, adapter))
                thread.start()
                threads.append(thread)

            # Wait for completion
            for thread in threads:
                thread.join(timeout=30)

            # Verify results
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())

            # Check for errors
            errors = [r for r in results if r[3] is not None]
            if errors:
                pytest.fail(f"Concurrent operations failed: {errors}")

            # Verify expected number of successful operations
            successful_ops = len([r for r in results if r[3] is None])
            expected_ops = len(adapters) * 3
            assert (
                successful_ops == expected_ops
            ), f"Expected {expected_ops} operations, got {successful_ops}"

        finally:
            # Cleanup adapters
            for adapter in adapters:
                try:
                    adapter.close()
                except Exception:
                    pass

    def test_docker_container_health_monitoring(self, pokemon_gym_container):
        """Test health monitoring with real Docker container."""
        container, server_url, port = pokemon_gym_container

        # Verify container is healthy
        assert container.get_wrapped_container().status == "running"

        # Test health endpoint
        with httpx.Client() as client:
            response = client.get(f"{server_url}/status", timeout=5.0)
            assert response.status_code == 200

            health_data = response.json()
            # Verify basic health indicators
            assert "active" in health_data or "status" in health_data

    def test_docker_container_resource_usage(self, pokemon_gym_container):
        """Test Docker container resource usage monitoring."""
        container, server_url, port = pokemon_gym_container

        docker_container = container.get_wrapped_container()

        # Get container stats
        stats = docker_container.stats(stream=False)

        # Verify resource usage is reasonable
        memory_usage = stats["memory_stats"]["usage"]
        cpu_usage = stats["cpu_stats"]["cpu_usage"]["total_usage"]

        # Basic resource sanity checks
        assert (
            memory_usage < 1024 * 1024 * 1024
        ), f"Memory usage too high: {memory_usage / (1024**2):.1f}MB"
        assert cpu_usage > 0, "CPU usage should be positive"

    def test_factory_method_with_real_docker_servers(self, pokemon_gym_container):
        """Test factory method creates clients that work with real Docker servers."""
        container, server_url, port = pokemon_gym_container

        adapter_types = ["auto", "benchflow", "direct"]

        for adapter_type in adapter_types:
            client = create_pokemon_client(
                port=port,
                container_id=container.get_wrapped_container().id,
                adapter_type=adapter_type,
                server_url=server_url,
            )

            # Verify client works with real Docker server
            if hasattr(client, "initialize_session"):
                # Adapter client
                try:
                    client.initialize_session()

                    # Test performance with real server
                    start_time = time.time()
                    _ = client.execute_action("A")  # Result unused in test
                    action_time = (time.time() - start_time) * 1000

                    assert action_time < 100, f"Action with {adapter_type} took {action_time:.1f}ms"

                finally:
                    client.close()
            else:
                # Direct client - verify basic functionality
                assert hasattr(client, "send_input")

    def test_real_server_performance_baseline(self, pokemon_gym_container):
        """Establish performance baseline with real Docker server."""
        container, server_url, port = pokemon_gym_container

        adapter = PokemonGymAdapter(
            port=port, container_id=container.get_wrapped_container().id, server_url=server_url
        )

        try:
            adapter.initialize_session()

            # Measure performance baseline
            action_times = []
            status_times = []

            for i in range(10):
                # Action performance
                start = time.time()
                adapter.execute_action(f"A_{i}")
                action_times.append((time.time() - start) * 1000)

                # Status performance
                start = time.time()
                adapter.get_session_status()
                status_times.append((time.time() - start) * 1000)

            # Calculate statistics
            avg_action = sum(action_times) / len(action_times)
            max_action = max(action_times)
            p95_action = sorted(action_times)[int(0.95 * len(action_times))]

            avg_status = sum(status_times) / len(status_times)
            max_status = max(status_times)

            # Performance validation - John Botmack standards
            assert avg_action < 50, f"Average action time {avg_action:.1f}ms exceeds 50ms target"
            assert (
                max_action < 100
            ), f"Max action time {max_action:.1f}ms violates 100ms requirement"
            assert p95_action < 80, f"95th percentile action time {p95_action:.1f}ms exceeds 80ms"

            assert avg_status < 25, f"Average status time {avg_status:.1f}ms exceeds 25ms target"
            assert max_status < 50, f"Max status time {max_status:.1f}ms violates 50ms requirement"

            # Store baseline for regression testing
            baseline_data = {
                "action_avg": avg_action,
                "action_max": max_action,
                "action_p95": p95_action,
                "status_avg": avg_status,
                "status_max": max_status,
            }

            print(f"Performance baseline established: {baseline_data}")

        finally:
            adapter.close()
