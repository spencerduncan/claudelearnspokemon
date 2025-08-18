"""
Validation tests for Docker integration test infrastructure setup.

Verifies that Docker integration test dependencies and infrastructure
are properly configured before running the full integration test suite.

Author: John Botmack - Performance Engineering
"""

import os

import pytest


@pytest.mark.fast
class TestIntegrationTestSetup:
    """Validate integration test infrastructure is ready."""

    def test_docker_client_availability(self):
        """Test Docker daemon is accessible."""
        try:
            import docker

            client = docker.from_env()
            client.ping()
            client.close()
            # Docker is available
        except Exception as e:
            if os.getenv("RUN_INTEGRATION_TESTS"):
                pytest.fail(f"Docker required for integration tests but not accessible: {e}")
            else:
                pytest.skip(f"Docker not available: {e}")

    def test_testcontainers_import(self):
        """Test testcontainers dependency is installed."""
        try:
            from testcontainers.core.container import DockerContainer  # noqa: F401

            # Imports successful - basic testcontainers functionality
        except ImportError as e:
            pytest.fail(f"testcontainers dependency missing: {e}")

    def test_integration_test_modules_importable(self):
        """Test integration test modules can be imported."""
        try:
            # Test conftest module
            import os
            import sys

            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "integration"))

            from conftest import (
                PERFORMANCE_REQUIREMENTS,  # noqa: F401
                pokemon_gym_container,  # noqa: F401
                skip_if_docker_unavailable,  # noqa: F401
            )

            # Imports successful
        except ImportError as e:
            pytest.fail(f"Integration test modules not importable: {e}")

    def test_environment_configuration(self):
        """Test environment variables are correctly configured."""
        run_integration = bool(os.getenv("RUN_INTEGRATION_TESTS", ""))

        if run_integration:
            # Validate required environment
            docker_timeout = os.getenv("DOCKER_TIMEOUT", "30")
            pokemon_image = os.getenv("POKEMON_GYM_IMAGE", "pokemon-gym:latest")

            assert docker_timeout.isdigit(), "DOCKER_TIMEOUT must be numeric"
            assert int(docker_timeout) > 0, "DOCKER_TIMEOUT must be positive"
            assert pokemon_image, "POKEMON_GYM_IMAGE cannot be empty"
        else:
            # Integration tests disabled - verify skip behavior
            pass

    def test_integration_directory_structure(self):
        """Test integration test directory structure exists."""
        integration_dir = os.path.join(os.path.dirname(__file__), "integration")

        # Check directory exists
        assert os.path.isdir(integration_dir), "Integration test directory missing"

        # Check required files exist
        required_files = [
            "__init__.py",
            "conftest.py",
            "test_server_integration.py",
            "test_performance_integration.py",
            "test_compatibility_integration.py",
            "README.md",
        ]

        for filename in required_files:
            filepath = os.path.join(integration_dir, filename)
            assert os.path.isfile(filepath), f"Required integration test file missing: {filename}"

    def test_performance_requirements_defined(self):
        """Test performance requirements are properly defined."""
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "integration"))

        from conftest import PERFORMANCE_REQUIREMENTS

        # Check required performance thresholds exist
        required_requirements = [
            "client_creation_ms",
            "action_execution_ms",
            "status_check_ms",
            "concurrent_overhead_pct",
        ]

        for requirement in required_requirements:
            assert (
                requirement in PERFORMANCE_REQUIREMENTS
            ), f"Performance requirement missing: {requirement}"
            value = PERFORMANCE_REQUIREMENTS[requirement]
            assert isinstance(
                value, int | float
            ), f"Performance requirement must be numeric: {requirement}"
            assert value > 0, f"Performance requirement must be positive: {requirement}"

    def test_github_actions_workflow_exists(self):
        """Test CI/CD workflow configuration exists."""
        workflow_path = os.path.join(
            os.path.dirname(__file__), "..", ".github", "workflows", "integration-tests.yml"
        )

        assert os.path.isfile(workflow_path), "GitHub Actions integration test workflow missing"

        # Read workflow content and validate key elements
        with open(workflow_path) as f:
            workflow_content = f.read()

        required_elements = [
            "RUN_INTEGRATION_TESTS: 1",
            "testcontainers",
            "pokemon-gym:latest",
            "pytest tests/integration/",
            "matrix:",
        ]

        for element in required_elements:
            assert element in workflow_content, f"Workflow missing required element: {element}"

    def test_performance_collector_functionality(self):
        """Test performance collector can be instantiated."""
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "integration"))

        from test_performance_integration import PerformanceCollector

        # Test collector creation and basic functionality
        collector = PerformanceCollector()

        # Test metric recording
        collector.record("test_metric", 50.5)
        collector.record("test_metric", 45.2)

        # Test analysis
        analysis = collector.analyze("test_metric")

        assert "avg" in analysis, "Performance analysis missing average"
        assert "p95" in analysis, "Performance analysis missing P95"
        assert analysis["count"] == 2, "Performance analysis count incorrect"
        assert 45 < analysis["avg"] < 50, "Performance analysis average incorrect"

        # Test requirement validation
        assert collector.validate_requirement(
            "test_metric", 60
        ), "Performance validation should pass"
        assert not collector.validate_requirement(
            "test_metric", 30
        ), "Performance validation should fail"

    @pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason="Integration tests disabled")
    def test_docker_integration_test_dry_run(self):
        """Test Docker integration test can be initialized (dry run)."""
        try:
            import docker
            from testcontainers.core.container import DockerContainer

            # Verify Docker daemon is working
            docker_client = docker.from_env()
            docker_client.ping()
            docker_client.close()

            # Test container configuration (without actually starting)
            container = DockerContainer("hello-world:latest")  # Use lightweight image
            container.with_exposed_ports(8080)

            # Container configuration successful
            assert container._docker_image == "hello-world:latest"

        except Exception as e:
            pytest.fail(f"Docker integration test dry run failed: {e}")
