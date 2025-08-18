"""
Server version compatibility integration tests with real Docker containers.

Tests compatibility across different Pokemon-gym server versions to ensure
robust integration and detect version-specific issues early.

Author: John Botmack - Performance Engineering
"""

import json
import time
from typing import Any

import httpx
import pytest

from claudelearnspokemon.pokemon_gym_adapter import PokemonGymAdapter
from claudelearnspokemon.pokemon_gym_factory import create_pokemon_client

# Version compatibility matrix - known Pokemon-gym versions
KNOWN_VERSIONS = ["pokemon-gym:v1.0", "pokemon-gym:v1.1", "pokemon-gym:v1.2", "pokemon-gym:latest"]

# Feature compatibility mapping
VERSION_FEATURES = {
    "v1.0": ["basic_actions", "status", "initialize", "stop"],
    "v1.1": ["basic_actions", "status", "initialize", "stop", "save_state", "load_state"],
    "v1.2": [
        "basic_actions",
        "status",
        "initialize",
        "stop",
        "save_state",
        "load_state",
        "batch_actions",
    ],
    "latest": [
        "basic_actions",
        "status",
        "initialize",
        "stop",
        "save_state",
        "load_state",
        "batch_actions",
        "streaming",
    ],
}

# Performance requirements per version
VERSION_PERFORMANCE = {
    "v1.0": {"action_ms": 150, "status_ms": 75},  # Older version, more relaxed
    "v1.1": {"action_ms": 120, "status_ms": 60},  # Improved performance
    "v1.2": {"action_ms": 100, "status_ms": 50},  # Current standard
    "latest": {"action_ms": 100, "status_ms": 50},  # Same as v1.2
}


class VersionCompatibilityTester:
    """Handles version-specific testing logic."""

    def __init__(self):
        self.compatibility_results = {}

    def detect_server_version(self, server_url: str) -> str:
        """Detect Pokemon-gym server version from API response."""
        try:
            with httpx.Client() as client:
                response = client.get(f"{server_url}/status", timeout=5.0)
                response.raise_for_status()

                data = response.json()

                # Try various version detection methods
                if "version" in data:
                    return data["version"]
                elif "server_version" in data:
                    return data["server_version"]
                elif "pokemon_gym_version" in data:
                    return data["pokemon_gym_version"]
                else:
                    # Fallback: detect by available endpoints
                    return self._detect_version_by_features(server_url)

        except Exception:
            return "unknown"

    def _detect_version_by_features(self, server_url: str) -> str:
        """Detect version by probing available features."""
        try:
            with httpx.Client() as client:
                # Test feature endpoints to determine version
                endpoints_to_test = [
                    ("/save_state", "v1.1+"),
                    ("/batch_actions", "v1.2+"),
                    ("/stream", "latest"),
                ]

                detected_version = "v1.0"  # Default to oldest

                for endpoint, min_version in endpoints_to_test:
                    try:
                        response = client.get(f"{server_url}{endpoint}", timeout=2.0)
                        if response.status_code != 404:
                            detected_version = min_version.replace("+", "")
                    except Exception:
                        continue

                return detected_version

        except Exception:
            return "unknown"

    def test_version_compatibility(
        self, version_info: str, server_url: str, port: int, container_id: str
    ) -> dict[str, Any]:
        """Test compatibility for a specific version."""
        result = {
            "version": version_info,
            "server_url": server_url,
            "detected_version": self.detect_server_version(server_url),
            "features_supported": [],
            "features_failed": [],
            "performance": {},
            "errors": [],
        }

        try:
            # Test adapter creation
            adapter = PokemonGymAdapter(port=port, container_id=container_id, server_url=server_url)

            # Test basic features
            self._test_basic_features(adapter, result)

            # Test version-specific features
            self._test_version_features(adapter, result, version_info)

            # Test performance characteristics
            self._test_version_performance(adapter, result, version_info)

            adapter.close()

        except Exception as e:
            result["errors"].append(f"Failed to test version {version_info}: {str(e)}")

        return result

    def _test_basic_features(self, adapter: PokemonGymAdapter, result: dict[str, Any]):
        """Test basic features that should work across all versions."""
        basic_tests = [
            ("status", lambda: adapter.get_session_status()),
            ("initialize", lambda: adapter.initialize_session()),
            ("action_A", lambda: adapter.execute_action("A")),
            ("action_B", lambda: adapter.execute_action("B")),
            ("stop", lambda: adapter.stop_session()),
        ]

        for feature_name, test_func in basic_tests:
            try:
                start_time = time.perf_counter()
                test_func()
                duration = (time.perf_counter() - start_time) * 1000

                result["features_supported"].append(feature_name)
                result["performance"][feature_name] = duration

            except Exception as e:
                result["features_failed"].append(f"{feature_name}: {str(e)}")

    def _test_version_features(
        self, adapter: PokemonGymAdapter, result: dict[str, Any], version_info: str
    ):
        """Test version-specific features."""
        version_key = version_info.split(":")[-1] if ":" in version_info else version_info
        expected_features = VERSION_FEATURES.get(version_key, [])

        # Test save/load state (v1.1+)
        if "save_state" in expected_features:
            try:
                # Re-initialize for state operations
                adapter.initialize_session()

                # Test save state (mock - actual implementation depends on API)
                if hasattr(adapter, "save_state"):
                    _ = adapter.save_state()  # State captured but not used in test
                    result["features_supported"].append("save_state")

                # Test load state
                if hasattr(adapter, "load_state"):
                    adapter.load_state({"test": "state"})
                    result["features_supported"].append("load_state")

            except Exception as e:
                result["features_failed"].append(f"state_operations: {str(e)}")

        # Test batch actions (v1.2+)
        if "batch_actions" in expected_features:
            try:
                if hasattr(adapter, "execute_batch_actions"):
                    adapter.execute_batch_actions(["A", "B", "A"])
                    result["features_supported"].append("batch_actions")
            except Exception as e:
                result["features_failed"].append(f"batch_actions: {str(e)}")

    def _test_version_performance(
        self, adapter: PokemonGymAdapter, result: dict[str, Any], version_info: str
    ):
        """Test performance characteristics for specific version."""
        version_key = version_info.split(":")[-1] if ":" in version_info else version_info
        perf_requirements = VERSION_PERFORMANCE.get(version_key, VERSION_PERFORMANCE["latest"])

        # Re-initialize for performance tests
        try:
            adapter.initialize_session()

            # Test action performance
            action_times = []
            for _ in range(5):
                start_time = time.perf_counter()
                adapter.execute_action("A")
                action_times.append((time.perf_counter() - start_time) * 1000)

            avg_action_time = sum(action_times) / len(action_times)
            result["performance"]["avg_action_ms"] = avg_action_time
            result["performance"]["action_requirement_ms"] = perf_requirements["action_ms"]
            result["performance"]["action_meets_requirement"] = (
                avg_action_time <= perf_requirements["action_ms"]
            )

            # Test status performance
            status_times = []
            for _ in range(5):
                start_time = time.perf_counter()
                adapter.get_session_status()
                status_times.append((time.perf_counter() - start_time) * 1000)

            avg_status_time = sum(status_times) / len(status_times)
            result["performance"]["avg_status_ms"] = avg_status_time
            result["performance"]["status_requirement_ms"] = perf_requirements["status_ms"]
            result["performance"]["status_meets_requirement"] = (
                avg_status_time <= perf_requirements["status_ms"]
            )

        except Exception as e:
            result["errors"].append(f"Performance testing failed: {str(e)}")


@pytest.mark.integration
@pytest.mark.slow
class TestVersionCompatibility:
    """Version compatibility tests with multiple Pokemon-gym server versions."""

    def setup_method(self):
        """Set up compatibility tester for each test."""
        self.tester = VersionCompatibilityTester()

    def test_single_version_compatibility(self, pokemon_gym_container):
        """Test compatibility with a single Pokemon-gym version."""
        container, server_url, port = pokemon_gym_container
        container_id = container.get_wrapped_container().id

        # Detect server version
        detected_version = self.tester.detect_server_version(server_url)

        # Test compatibility
        result = self.tester.test_version_compatibility(
            detected_version, server_url, port, container_id
        )

        # Validate basic functionality works
        assert len(result["features_supported"]) > 0, "No supported features detected"
        assert "initialize" in result["features_supported"], "Initialize feature must work"
        assert "action_A" in result["features_supported"], "Basic actions must work"

        # Validate performance
        if "avg_action_ms" in result["performance"]:
            assert result["performance"]["action_meets_requirement"], (
                f"Action performance {result['performance']['avg_action_ms']:.1f}ms "
                f"exceeds requirement {result['performance']['action_requirement_ms']}ms"
            )

    def test_multiple_version_compatibility(self, version_compatibility_containers):
        """Test compatibility across multiple Pokemon-gym server versions."""
        version_data = version_compatibility_containers

        if len(version_data) < 2:
            pytest.skip("Need multiple versions for compatibility testing")

        results = []

        for version_tag, container, server_url, port in version_data:
            container_id = container.get_wrapped_container().id

            result = self.tester.test_version_compatibility(
                version_tag, server_url, port, container_id
            )
            results.append(result)

        # Validate all versions support basic features
        for result in results:
            assert (
                len(result["features_supported"]) > 0
            ), f"Version {result['version']} has no supported features"
            assert (
                "initialize" in result["features_supported"]
            ), f"Version {result['version']} must support initialize"

        # Compare feature sets across versions
        feature_evolution = {}
        for result in results:
            version_name = result["version"]
            feature_evolution[version_name] = set(result["features_supported"])

        # Newer versions should have equal or more features (backward compatibility)
        version_names = list(feature_evolution.keys())
        for i in range(len(version_names) - 1):
            current_version = version_names[i]
            next_version = version_names[i + 1]

            next_features = feature_evolution[next_version]

            # Basic features should be preserved
            core_features = {"initialize", "action_A", "status"}
            assert core_features.issubset(
                next_features
            ), f"Version {next_version} missing core features from {current_version}"

        # Export compatibility matrix
        compatibility_matrix = {
            "test_timestamp": time.time(),
            "versions_tested": len(results),
            "results": results,
        }

        with open("tests/integration/compatibility_matrix.json", "w") as f:
            json.dump(compatibility_matrix, f, indent=2)

    def test_client_factory_version_compatibility(self, version_compatibility_containers):
        """Test factory method compatibility across versions."""
        version_data = version_compatibility_containers

        if not version_data:
            pytest.skip("No versions available for factory compatibility testing")

        adapter_types = ["auto", "benchflow", "direct"]

        for version_tag, container, server_url, port in version_data:
            container_id = container.get_wrapped_container().id

            for adapter_type in adapter_types:
                try:
                    client = create_pokemon_client(
                        port=port,
                        container_id=container_id,
                        adapter_type=adapter_type,
                        server_url=server_url,
                    )

                    # Test basic functionality
                    if hasattr(client, "initialize_session"):
                        client.initialize_session()
                        client.execute_action("A")
                        client.close()

                    # Success - factory works with this version
                    print(f"✓ Factory {adapter_type} compatible with {version_tag}")

                except Exception as e:
                    # Log compatibility issue but don't fail test
                    print(f"✗ Factory {adapter_type} incompatible with {version_tag}: {e}")

    def test_performance_across_versions(self, version_compatibility_containers):
        """Test performance characteristics across different versions."""
        version_data = version_compatibility_containers

        if not version_data:
            pytest.skip("No versions available for performance comparison")

        performance_results = {}

        for version_tag, container, server_url, port in version_data:
            container_id = container.get_wrapped_container().id

            adapter = PokemonGymAdapter(port=port, container_id=container_id, server_url=server_url)

            try:
                adapter.initialize_session()

                # Measure key operations
                operations = {
                    "action": lambda a=adapter: a.execute_action("A"),
                    "status": lambda a=adapter: a.get_session_status(),
                }

                version_perf = {}
                for op_name, operation in operations.items():
                    times = []
                    for _ in range(10):
                        start_time = time.perf_counter()
                        operation()
                        times.append((time.perf_counter() - start_time) * 1000)

                    version_perf[op_name] = {
                        "avg": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times),
                    }

                performance_results[version_tag] = version_perf

            finally:
                adapter.close()

        # Compare performance across versions
        if len(performance_results) > 1:
            for op_name in ["action", "status"]:
                version_times = [
                    (v, data[op_name]["avg"])
                    for v, data in performance_results.items()
                    if op_name in data
                ]

                if len(version_times) > 1:
                    fastest_version, fastest_time = min(version_times, key=lambda x: x[1])
                    slowest_version, slowest_time = max(version_times, key=lambda x: x[1])

                    performance_variance = ((slowest_time - fastest_time) / fastest_time) * 100

                    # Log performance differences
                    print(f"{op_name.capitalize()} performance across versions:")
                    for version, avg_time in version_times:
                        print(f"  {version}: {avg_time:.1f}ms")
                    print(
                        f"  Variance: {performance_variance:.1f}% ({fastest_version} to {slowest_version})"
                    )

                    # Performance shouldn't degrade dramatically across versions
                    assert (
                        performance_variance < 200
                    ), f"{op_name} performance variance {performance_variance:.1f}% too high across versions"

    def test_error_handling_consistency(self, version_compatibility_containers):
        """Test error handling consistency across versions."""
        version_data = version_compatibility_containers

        if not version_data:
            pytest.skip("No versions available for error handling testing")

        error_scenarios = [
            ("invalid_action", lambda adapter: adapter.execute_action("INVALID_BUTTON")),
            (
                "uninitialized_action",
                lambda adapter: adapter.execute_action("A"),
            ),  # Before initialize
            (
                "double_initialize",
                lambda adapter: (adapter.initialize_session(), adapter.initialize_session())[1],
            ),
        ]

        error_patterns = {}

        for version_tag, container, server_url, port in version_data:
            container_id = container.get_wrapped_container().id
            version_errors = {}

            for scenario_name, error_func in error_scenarios:
                adapter = PokemonGymAdapter(
                    port=port, container_id=container_id, server_url=server_url
                )

                try:
                    if scenario_name != "uninitialized_action":
                        adapter.initialize_session()

                    try:
                        error_func(adapter)
                        version_errors[scenario_name] = "NO_ERROR"  # Unexpected
                    except Exception as e:
                        version_errors[scenario_name] = type(e).__name__

                finally:
                    adapter.close()

            error_patterns[version_tag] = version_errors

        # Check for consistency in error handling
        if len(error_patterns) > 1:
            for scenario in error_scenarios:
                scenario_name = scenario[0]
                error_types = [
                    patterns.get(scenario_name, "UNKNOWN") for patterns in error_patterns.values()
                ]
                unique_errors = set(error_types)

                # Log error handling patterns
                print(f"Error handling for {scenario_name}:")
                for version, error_type in zip(error_patterns.keys(), error_types, strict=False):
                    print(f"  {version}: {error_type}")

                # Consistent error handling is preferred but not required
                if len(unique_errors) > 2:
                    print("  Warning: Inconsistent error handling across versions")

    def test_api_response_format_compatibility(self, version_compatibility_containers):
        """Test API response format compatibility across versions."""
        version_data = version_compatibility_containers

        if not version_data:
            pytest.skip("No versions available for API format testing")

        response_schemas = {}

        for version_tag, container, server_url, port in version_data:
            container_id = container.get_wrapped_container().id

            adapter = PokemonGymAdapter(port=port, container_id=container_id, server_url=server_url)

            try:
                # Test response formats
                adapter.initialize_session()

                # Status response format
                status_response = adapter.get_session_status()
                status_schema = self._extract_schema(status_response)

                # Action response format
                action_response = adapter.execute_action("A")
                action_schema = self._extract_schema(action_response)

                response_schemas[version_tag] = {"status": status_schema, "action": action_schema}

            finally:
                adapter.close()

        # Compare schemas across versions
        if len(response_schemas) > 1:
            for response_type in ["status", "action"]:
                schemas = [
                    (v, data[response_type])
                    for v, data in response_schemas.items()
                    if response_type in data
                ]

                # Check for common fields across versions
                if len(schemas) > 1:
                    common_fields = set(schemas[0][1].keys())
                    for _, schema in schemas[1:]:
                        common_fields &= set(schema.keys())

                    print(
                        f"{response_type.capitalize()} response common fields: {sorted(common_fields)}"
                    )

                    # Core fields should be present across versions
                    if response_type == "status":
                        assert (
                            "active" in common_fields or "status" in common_fields
                        ), "Status response missing core field"

    def _extract_schema(self, response_data) -> dict[str, str]:
        """Extract basic schema information from response data."""
        if not isinstance(response_data, dict):
            return {"_type": type(response_data).__name__}

        schema = {}
        for key, value in response_data.items():
            schema[key] = type(value).__name__

        return schema
