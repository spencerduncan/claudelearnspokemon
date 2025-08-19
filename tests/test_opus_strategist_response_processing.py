"""
OpusStrategist Strategy Response Processing Tests

Test suite for JSON parsing, validation, and strategic directive extraction
from Claude Opus responses. Emphasizes production failure scenarios and
resilience patterns.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from src.claudelearnspokemon.opus_strategist import (
    MalformedResponseError,
    OpusStrategist,
    ResponseCache,
    ResponseTimeoutError,
    StrategyResponse,
    StrategyResponseParser,
    StrategyValidationError,
)


@pytest.mark.slow
class TestStrategyResponseParser:
    """Test JSON parsing and validation for strategy responses."""

    def setup_method(self) -> None:
        """Setup test fixtures with production-like configurations."""
        self.parser = StrategyResponseParser(
            validation_timeout=5.0, max_response_size=10485760  # 10MB - realistic production limit
        )

    def test_parses_valid_strategy_response(self) -> None:
        """Test parsing well-formed strategy response."""
        valid_response = {
            "strategy_id": "strategy_001",
            "timestamp": "2025-08-18T15:00:00Z",
            "experiments": [
                {
                    "id": "exp_001",
                    "name": "Viridian Forest Speed Route",
                    "checkpoint": "checkpoint_viridian_entry",
                    "script_dsl": "GO_NORTH; AVOID_TRAINERS; CATCH_POKEMON(pikachu)",
                    "expected_outcome": "sub_60s_forest_clear",
                    "priority": 1,
                }
            ],
            "strategic_insights": [
                "Early Pikachu reduces Brock difficulty significantly",
                "Trainer avoidance saves 15-20 seconds average",
            ],
            "next_checkpoints": ["route_2", "pewter_gym"],
        }

        result = self.parser.parse_response(json.dumps(valid_response))

        assert isinstance(result, StrategyResponse)
        assert result.strategy_id == "strategy_001"
        assert len(result.experiments) == 1
        assert result.experiments[0].name == "Viridian Forest Speed Route"
        assert len(result.strategic_insights) == 2
        assert "route_2" in result.next_checkpoints

    def test_handles_malformed_json_gracefully(self) -> None:
        """Test graceful handling of invalid JSON."""
        malformed_json = '{"strategy_id": "test", "experiments": [missing_bracket'

        with pytest.raises(MalformedResponseError) as exc_info:
            self.parser.parse_response(malformed_json)

        assert "Invalid JSON format" in str(exc_info.value)
        assert exc_info.value.raw_response == malformed_json

    def test_validates_required_fields_presence(self) -> None:
        """Test validation of required response fields."""
        incomplete_response = {
            "strategy_id": "strategy_002",
            # Missing experiments and strategic_insights
        }

        with pytest.raises(StrategyValidationError) as exc_info:
            self.parser.parse_response(json.dumps(incomplete_response))

        assert "Missing required field: experiments" in str(exc_info.value)

    def test_handles_oversized_responses(self) -> None:
        """Test handling of responses exceeding size limits."""
        # Create oversized response
        oversized_response = {
            "strategy_id": "large_strategy",
            "experiments": [],
            "strategic_insights": ["x" * 20_000_000],  # 20MB insight
            "next_checkpoints": [],
        }

        with pytest.raises(MalformedResponseError) as exc_info:
            self.parser.parse_response(json.dumps(oversized_response))

        assert "Response exceeds maximum size" in str(exc_info.value)

    def test_validates_experiment_schema(self) -> None:
        """Test validation of experiment structure."""
        invalid_experiment_response = {
            "strategy_id": "exp_validation_test",
            "experiments": [
                {
                    "id": "exp_001",
                    # Missing required fields: name, checkpoint, script_dsl
                    "priority": 1,
                }
            ],
            "strategic_insights": [],
            "next_checkpoints": [],
        }

        with pytest.raises(StrategyValidationError) as exc_info:
            self.parser.parse_response(json.dumps(invalid_experiment_response))

        assert "experiment 0 missing required field" in str(exc_info.value).lower()

    def test_enforces_parsing_timeout(self) -> None:
        """Test timeout enforcement for slow parsing operations."""
        # Mock slow JSON parsing
        with patch("json.loads") as mock_json:
            mock_json.side_effect = lambda x: time.sleep(10)  # Simulate slow parsing

            parser = StrategyResponseParser(validation_timeout=0.1)

            with pytest.raises(ResponseTimeoutError) as exc_info:
                parser.parse_response('{"test": "data"}')

            assert "parsing timeout" in str(exc_info.value).lower()

    def test_handles_unicode_and_special_characters(self) -> None:
        """Test parsing responses with Unicode and special characters."""
        unicode_response = {
            "strategy_id": "unicode_test_🎮",
            "experiments": [
                {
                    "id": "exp_unicode",
                    "name": "Pokémon Center Route 中文",
                    "checkpoint": "checkpoint_unicode",
                    "script_dsl": 'TALK_TO_NPC("Hello, 世界!"); USE_ITEM(potion)',
                    "expected_outcome": "heal_pokemon",
                    "priority": 1,
                }
            ],
            "strategic_insights": ["Unicode handling is critical for international users"],
            "next_checkpoints": ["route_日本"],
        }

        result = self.parser.parse_response(json.dumps(unicode_response, ensure_ascii=False))

        assert result.strategy_id == "unicode_test_🎮"
        assert "世界" in result.experiments[0].script_dsl
        assert "route_日本" in result.next_checkpoints


@pytest.mark.slow
class TestResponseCache:
    """Test response caching with TTL and performance requirements."""

    def setup_method(self) -> None:
        """Setup cache with production-like configuration."""
        self.cache = ResponseCache(
            max_size=100, default_ttl=300, cleanup_interval=60  # 5 minutes  # 1 minute
        )

        # Sample response for caching
        self.sample_response = StrategyResponse(
            strategy_id="cached_strategy",
            experiments=[],
            strategic_insights=["Test insight"],
            next_checkpoints=["test_checkpoint"],
        )

    def test_caches_and_retrieves_responses(self) -> None:
        """Test basic cache functionality."""
        cache_key = "strategy_cache_test"

        # Cache miss initially
        assert self.cache.get(cache_key) is None

        # Store response
        self.cache.put(cache_key, self.sample_response)

        # Cache hit
        cached_response = self.cache.get(cache_key)
        assert cached_response is not None
        assert cached_response.strategy_id == "cached_strategy"

    def test_respects_ttl_expiration(self) -> None:
        """Test TTL-based cache expiration."""
        cache = ResponseCache(max_size=10, default_ttl=0.1)  # 100ms TTL
        cache_key = "ttl_test"

        cache.put(cache_key, self.sample_response)

        # Immediate retrieval should work
        assert cache.get(cache_key) is not None

        # Wait for TTL expiration
        time.sleep(0.2)

        # Should be expired
        assert cache.get(cache_key) is None

    def test_enforces_size_limits(self) -> None:
        """Test LRU eviction when cache size limit reached."""
        small_cache = ResponseCache(max_size=2, default_ttl=300)

        # Fill cache to capacity
        small_cache.put("key1", self.sample_response)
        small_cache.put("key2", self.sample_response)

        # Both should be present
        assert small_cache.get("key1") is not None
        assert small_cache.get("key2") is not None

        # Add third item, should evict LRU
        small_cache.put("key3", self.sample_response)

        # key1 should be evicted (was least recently used)
        assert small_cache.get("key1") is None
        assert small_cache.get("key2") is not None
        assert small_cache.get("key3") is not None

    def test_thread_safety_under_concurrent_access(self) -> None:
        """Test cache thread safety with concurrent operations."""
        results = []

        def cache_worker(worker_id: int):
            """Worker function for concurrent cache operations."""
            for i in range(10):
                key = f"worker_{worker_id}_item_{i}"
                response = StrategyResponse(
                    strategy_id=f"strategy_{worker_id}_{i}",
                    experiments=[],
                    strategic_insights=[f"Insight from worker {worker_id}"],
                    next_checkpoints=[],
                )

                self.cache.put(key, response)
                retrieved = self.cache.get(key)

                if retrieved and retrieved.strategy_id == response.strategy_id:
                    results.append(True)
                else:
                    results.append(False)

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cache_worker, i) for i in range(4)]
            for future in futures:
                future.result()

        # All operations should succeed without race conditions
        assert all(results), f"Thread safety test failed: {sum(results)}/{len(results)} succeeded"

    def test_cache_performance_requirements(self) -> None:
        """Test cache meets <100ms performance targets."""
        # Pre-populate cache
        for i in range(50):
            key = f"perf_test_{i}"
            response = StrategyResponse(
                strategy_id=f"strategy_{i}",
                experiments=[],
                strategic_insights=[f"Performance test insight {i}"],
                next_checkpoints=[],
            )
            self.cache.put(key, response)

        # Test retrieval performance
        start_time = time.time()

        for i in range(100):  # 100 cache operations
            key = f"perf_test_{i % 50}"  # Mix hits and misses
            self.cache.get(key)

        total_time = time.time() - start_time
        avg_time_per_op = (total_time / 100) * 1000  # Convert to ms

        assert avg_time_per_op < 1.0, f"Cache too slow: {avg_time_per_op:.2f}ms per operation"


@pytest.mark.slow
class TestOpusStrategistResponseProcessing:
    """Integration tests for OpusStrategist response processing."""

    def setup_method(self) -> None:
        """Setup OpusStrategist with mocked dependencies."""
        self.mock_claude_manager = Mock()
        self.mock_strategic_process = Mock()
        self.mock_claude_manager.get_strategic_process.return_value = self.mock_strategic_process
        self.strategist = OpusStrategist(claude_manager=self.mock_claude_manager)

    def test_processes_opus_response_end_to_end(self) -> None:
        """Test complete response processing pipeline."""
        # Mock Opus response
        opus_response = {
            "strategy_id": "integration_test_001",
            "experiments": [
                {
                    "id": "exp_integration",
                    "name": "Integration Test Route",
                    "checkpoint": "test_checkpoint",
                    "script_dsl": "TEST_COMMAND; ASSERT_SUCCESS",
                    "expected_outcome": "successful_test",
                    "priority": 1,
                }
            ],
            "strategic_insights": ["Integration test demonstrates end-to-end functionality"],
            "next_checkpoints": ["next_test_point"],
        }

        self.mock_strategic_process.send_message.return_value = json.dumps(opus_response)

        # Process strategy request
        game_state = {"location": "test_area", "health": 100}
        result = self.strategist.get_strategy(game_state)

        assert result.strategy_id == "integration_test_001"
        assert len(result.experiments) == 1
        assert result.experiments[0].name == "Integration Test Route"

    def test_caches_responses_for_performance(self) -> None:
        """Test response caching reduces Claude API calls."""
        game_state = {"location": "repeated_area", "health": 100}

        # First call should hit Claude
        opus_response = {
            "strategy_id": "cached_response_test",
            "experiments": [],
            "strategic_insights": ["First call to Claude"],
            "next_checkpoints": [],
        }
        self.mock_strategic_process.send_message.return_value = json.dumps(opus_response)

        result1 = self.strategist.get_strategy(game_state)
        assert self.mock_strategic_process.send_message.call_count == 1

        # Second identical call should use cache
        result2 = self.strategist.get_strategy(game_state)
        assert self.mock_strategic_process.send_message.call_count == 1  # No additional calls

        # Results should be identical
        assert result1.strategy_id == result2.strategy_id

    def test_handles_claude_failure_gracefully(self) -> None:
        """Test graceful degradation when Claude communication fails."""
        self.mock_claude_manager.send_to_opus.side_effect = Exception("Claude connection failed")

        game_state = {"location": "failure_test", "health": 50}

        # Should not raise exception, should return fallback strategy
        result = self.strategist.get_strategy(game_state)

        assert result is not None
        assert result.strategy_id.startswith("fallback_")
        assert len(result.experiments) > 0  # Should have fallback experiments

    def test_validates_response_within_timeout(self) -> None:
        """Test response processing meets performance requirements."""
        large_but_valid_response = {
            "strategy_id": "performance_test",
            "experiments": [
                {
                    "id": f"exp_{i}",
                    "name": f"Performance Test Route {i}",
                    "checkpoint": f"checkpoint_{i}",
                    "script_dsl": f"PERFORMANCE_TEST_{i}; MEASURE_TIME",
                    "expected_outcome": f"sub_optimal_time_{i}",
                    "priority": i % 5 + 1,
                }
                for i in range(50)  # 50 experiments
            ],
            "strategic_insights": [f"Insight {i}: Performance matters" for i in range(20)],
            "next_checkpoints": [f"checkpoint_{i}" for i in range(100, 150)],
        }

        self.mock_strategic_process.send_message.return_value = json.dumps(large_but_valid_response)

        start_time = time.time()
        result = self.strategist.get_strategy({"location": "performance_test"})
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Should meet performance requirement
        assert processing_time < 100, f"Response processing too slow: {processing_time:.2f}ms"
        assert result.strategy_id == "performance_test"
        assert len(result.experiments) == 50

    def test_extracts_strategic_directives_correctly(self) -> None:
        """Test extraction of actionable strategic directives."""
        complex_response = {
            "strategy_id": "directive_extraction_test",
            "experiments": [
                {
                    "id": "directive_exp_1",
                    "name": "Multi-objective Route",
                    "checkpoint": "complex_checkpoint",
                    "script_dsl": "OPTIMIZE_FOR(speed, xp); BALANCE_RESOURCES; CHECK_CONDITIONS",
                    "expected_outcome": "balanced_progression",
                    "priority": 1,
                    "directives": [
                        "speed_optimization",
                        "resource_management",
                        "condition_checking",
                    ],
                }
            ],
            "strategic_insights": [
                "DIRECTIVE: Focus on early-game speed optimization",
                "INSIGHT: Resource management reduces mid-game risk",
                "DIRECTIVE: Implement condition checking for route adaptation",
            ],
            "next_checkpoints": ["optimization_checkpoint", "management_checkpoint"],
        }

        self.mock_strategic_process.send_message.return_value = json.dumps(complex_response)

        result = self.strategist.get_strategy({"location": "directive_test"})
        directives = self.strategist.extract_directives(result)

        # Should extract both explicit directives and directive-marked insights
        assert "speed_optimization" in directives
        assert "resource_management" in directives
        assert "condition_checking" in directives
        assert "focus on early-game speed optimization" in [d.lower() for d in directives]


@pytest.mark.slow
class TestResponseProcessingPerformance:
    """Performance-focused tests for production readiness."""

    def test_concurrent_response_processing(self) -> None:
        """Test system handles concurrent response processing."""
        mock_claude_manager = Mock()
        mock_strategic_process = Mock()
        mock_claude_manager.get_strategic_process.return_value = mock_strategic_process

        # Thread-safe side_effect that generates appropriate response for each call
        def generate_response(prompt: str) -> str:
            # Extract worker_id from the prompt or use a unique identifier
            import threading
            import time

            thread_id = threading.current_thread().ident
            unique_id = f"{thread_id}_{int(time.time() * 1000000) % 1000000}"

            response = {
                "strategy_id": f"concurrent_test_{unique_id}",
                "experiments": [],
                "strategic_insights": [f"Concurrent insight {unique_id}"],
                "next_checkpoints": [],
            }
            return json.dumps(response)

        mock_strategic_process.send_message.side_effect = generate_response
        strategist = OpusStrategist(claude_manager=mock_claude_manager)

        def concurrent_processing(worker_id: int) -> bool:
            """Worker function for concurrent processing."""
            result = strategist.get_strategy({"worker_id": worker_id})
            # Just check that we got a valid strategy_id that starts with concurrent_test_
            return result.strategy_id.startswith("concurrent_test_")

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(concurrent_processing, i) for i in range(20)]
            results = [future.result() for future in futures]

        # All concurrent operations should succeed
        assert all(
            results
        ), f"Concurrent processing failed: {sum(results)}/{len(results)} succeeded"

    def test_memory_usage_remains_bounded(self) -> None:
        """Test memory usage doesn't grow unbounded with cache."""
        mock_claude_manager = Mock()
        mock_strategic_process = Mock()
        mock_claude_manager.get_strategic_process.return_value = mock_strategic_process
        strategist = OpusStrategist(claude_manager=mock_claude_manager)

        # Process many different requests to test cache bounds
        for i in range(200):  # More than cache size
            mock_response = {
                "strategy_id": f"memory_test_{i}",
                "experiments": [],
                "strategic_insights": [f"Memory test insight {i}"],
                "next_checkpoints": [],
            }

            mock_strategic_process.send_message.return_value = json.dumps(mock_response)
            strategist.get_strategy({"test_id": i})

        # Cache should have bounded size despite 200 unique requests
        cache_size = len(strategist.response_cache._cache)
        assert cache_size <= 100, f"Cache not properly bounded: {cache_size} items"
