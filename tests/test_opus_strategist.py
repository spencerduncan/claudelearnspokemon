"""
Test suite for OpusStrategist parallel results analysis.

Performance-focused test design following John Botmack principles:
- Pattern identification algorithms use O(n) complexity
- Result compression must complete in <100ms for strategic processing
- Statistical correlation analysis must handle parallel execution results
- Memory-efficient result caching with FIFO eviction policy

Test Categories:
- TestOpusStrategistBasics: Initialization and process integration
- TestParallelResultsAnalysis: Core pattern identification functionality
- TestResultCompression: Efficient strategic processing optimization
- TestStatisticalCorrelation: Parallel execution result correlation
- TestPerformanceTargets: Algorithmic performance validation
"""

import time
import unittest
from collections import namedtuple
from unittest.mock import Mock

import pytest

# Mock data structures for test cases
ExecutionResult = namedtuple(
    "ExecutionResult",
    [
        "worker_id",
        "success",
        "execution_time",
        "actions_taken",
        "final_state",
        "performance_metrics",
        "discovered_patterns",
    ],
)

ParallelResults = namedtuple(
    "ParallelResults",
    [
        "results",
        "correlation_matrix",
        "execution_timestamp",
        "worker_performance",
        "pattern_frequency",
    ],
)


@pytest.mark.fast
class TestOpusStrategistBasics(unittest.TestCase):
    """Test basic OpusStrategist functionality and integration."""

    def setUp(self):
        """Set up test environment with mocked ClaudeCodeManager."""
        # Mock ClaudeCodeManager with strategic process
        self.mock_manager = Mock()
        self.mock_strategic_process = Mock()
        self.mock_manager.get_strategic_process.return_value = self.mock_strategic_process

        # Import after mocking to avoid initialization issues
        from claudelearnspokemon.opus_strategist import OpusStrategist

        self.strategist = OpusStrategist(self.mock_manager)

    def test_initialization_with_claude_manager(self):
        """Test OpusStrategist initializes with ClaudeCodeManager integration."""
        self.assertIsNotNone(self.strategist)
        self.assertEqual(self.strategist.claude_manager, self.mock_manager)

        # Strategic process validation is now done at runtime, not initialization
        # self.mock_manager.get_strategic_process.assert_called_once()

    def test_request_strategy_method_exists(self):
        """Test that request_strategy method exists with correct signature."""
        # Check method exists
        self.assertTrue(hasattr(self.strategist, "request_strategy"))

        # Check method is callable
        self.assertTrue(callable(self.strategist.request_strategy))


@pytest.mark.fast
class TestRequestStrategyInterface(unittest.TestCase):
    """Test the strategic planning interface request_strategy method."""

    def setUp(self):
        """Set up test environment with mocked ClaudeCodeManager."""
        # Mock ClaudeCodeManager with strategic process
        self.mock_manager = Mock()
        self.mock_strategic_process = Mock()
        self.mock_manager.get_strategic_process.return_value = self.mock_strategic_process

        # Import after mocking to avoid initialization issues
        from claudelearnspokemon.opus_strategist import OpusStrategist

        self.strategist = OpusStrategist(self.mock_manager)

        # Test data
        self.game_state = {
            "location": "pallet_town",
            "health": 100,
            "level": 5,
            "pokemon_count": 1,
            "badges": 0,
            "x": 10,
            "y": 15,
        }

        self.recent_results = [
            {
                "worker_id": "worker_1",
                "success": True,
                "execution_time": 1.23,
                "actions_taken": ["A", "B", "START", "RIGHT"],
                "final_state": {"x": 12, "y": 15, "level": 5},
                "patterns_discovered": ["menu_optimization"],
            },
            {
                "worker_id": "worker_2",
                "success": False,
                "execution_time": 2.1,
                "actions_taken": ["A", "LEFT", "B"],
                "final_state": {"x": 9, "y": 15, "level": 5},
                "patterns_discovered": ["failed_movement"],
            },
        ]

    def test_opus_strategist_formats_game_state_for_context(self):
        """
        Test that request_strategy properly formats game state for strategic context.

        This test validates Issue #99 requirement: format strategic requests
        for Claude Opus consumption with clear game state context.
        """
        # Mock strategic process response
        mock_response = """{
            "strategy_id": "strategic_001",
            "experiments": [
                {
                    "id": "exp_001",
                    "name": "Optimize menu navigation",
                    "checkpoint": "pallet_town_start",
                    "script_dsl": "MENU_OPEN; SELECT_POKEMON; MENU_CLOSE",
                    "expected_outcome": "Faster menu operations",
                    "priority": "high"
                }
            ],
            "strategic_insights": [
                "Menu optimization shows high success rate",
                "Left movement patterns correlate with failures"
            ],
            "next_checkpoints": ["oak_lab_entrance", "route1_start"]
        }"""

        self.mock_strategic_process.send_message.return_value = mock_response

        # Execute request_strategy
        result = self.strategist.request_strategy(self.game_state, self.recent_results)

        # Verify strategic process was called
        self.mock_strategic_process.send_message.assert_called_once()

        # Get the formatted request sent to Opus
        call_args = self.mock_strategic_process.send_message.call_args[0][0]

        # Verify game state formatting
        self.assertIn("STRATEGIC PLANNING REQUEST", call_args)
        self.assertIn("Current Game State:", call_args)
        self.assertIn("location: pallet_town", call_args)
        self.assertIn("health: 100", call_args)
        self.assertIn("level: 5", call_args)
        self.assertIn("badges: 0", call_args)

        # Verify result is properly formatted
        self.assertIsInstance(result, dict)
        self.assertIn("strategy_id", result)
        self.assertIn("experiments", result)

    def test_opus_strategist_parses_json_strategy_response(self):
        """
        Test that request_strategy parses JSON strategy responses correctly.

        This test validates Issue #99 requirement: parse JSON strategy responses
        into actionable plans with proper error handling.
        """
        # Mock valid JSON response from Opus
        valid_json_response = """{
            "strategy_id": "strategic_002",
            "experiments": [
                {
                    "id": "exp_002",
                    "name": "Route optimization test",
                    "checkpoint": "current_position",
                    "script_dsl": "MOVE_RIGHT; MOVE_UP; INTERACT",
                    "expected_outcome": "Faster route completion",
                    "priority": "medium"
                }
            ],
            "strategic_insights": [
                "DIRECTIVE: Focus on right movement patterns",
                "Previous left movements show 50% failure rate"
            ],
            "next_checkpoints": ["route1_end"]
        }"""

        self.mock_strategic_process.send_message.return_value = valid_json_response

        # Execute request_strategy
        result = self.strategist.request_strategy(self.game_state, self.recent_results)

        # Verify JSON parsing worked correctly
        self.assertIsInstance(result, dict)
        self.assertEqual(result["strategy_id"], "strategic_002")
        self.assertIsInstance(result["experiments"], list)
        self.assertEqual(len(result["experiments"]), 1)

        # Verify experiment structure
        experiment = result["experiments"][0]
        self.assertEqual(experiment["id"], "exp_002")
        self.assertEqual(experiment["name"], "Route optimization test")
        self.assertEqual(experiment["priority"], "medium")

        # Verify insights parsing
        self.assertIn("strategic_insights", result)
        self.assertIsInstance(result["strategic_insights"], list)

    def test_request_strategy_formats_recent_results_context(self):
        """
        Test that recent_results are properly formatted for strategic context.

        This validates that recent execution results provide meaningful
        context for strategic planning decisions.
        """
        mock_response = """{
            "strategy_id": "results_based_003",
            "experiments": [],
            "strategic_insights": ["Based on recent results analysis"],
            "next_checkpoints": []
        }"""

        self.mock_strategic_process.send_message.return_value = mock_response

        # Execute with recent results
        self.strategist.request_strategy(self.game_state, self.recent_results)

        # Get the formatted request
        call_args = self.mock_strategic_process.send_message.call_args[0][0]

        # Verify recent results are included in strategic context
        self.assertIn("Recent Execution Results:", call_args)
        self.assertIn("worker_1", call_args)
        self.assertIn("worker_2", call_args)
        self.assertIn("success: True", call_args)
        self.assertIn("success: False", call_args)
        self.assertIn("menu_optimization", call_args)
        self.assertIn("failed_movement", call_args)

        # Verify performance metrics included
        self.assertIn("execution_time: 1.23", call_args)
        self.assertIn("execution_time: 2.1", call_args)

    def test_request_strategy_handles_malformed_json_response(self):
        """
        Test graceful handling of malformed JSON responses from Opus.

        This validates Issue #99 requirement: handle strategic planning
        request failures gracefully.
        """
        # Mock malformed JSON response
        malformed_response = """
        {
            "strategy_id": "malformed_001",
            "experiments": [
                "invalid_structure"  // This should be an object, not string
            ]
            "strategic_insights": "not_an_array"
            // Missing closing brace
        """

        self.mock_strategic_process.send_message.return_value = malformed_response

        # Execute request_strategy - should handle malformed response gracefully
        result = self.strategist.request_strategy(self.game_state, self.recent_results)

        # Should return fallback strategy, not crash
        self.assertIsInstance(result, dict)

        # Should contain fallback indicators
        self.assertTrue(
            "fallback" in str(result).lower()
            or "experiments" in result,  # Fallback should still have basic structure
            "Should return fallback strategy for malformed response",
        )

    def test_request_strategy_handles_empty_recent_results(self):
        """Test request_strategy handles empty recent_results gracefully."""
        mock_response = """{
            "strategy_id": "empty_results_001",
            "experiments": [],
            "strategic_insights": ["No recent results available"],
            "next_checkpoints": []
        }"""

        self.mock_strategic_process.send_message.return_value = mock_response

        # Execute with empty recent results
        result = self.strategist.request_strategy(self.game_state, [])

        # Should work without errors
        self.assertIsInstance(result, dict)

        # Get the formatted request
        call_args = self.mock_strategic_process.send_message.call_args[0][0]

        # Should handle empty results gracefully
        self.assertIn("Recent Execution Results:", call_args)
        self.assertIn("No recent results available", call_args)

    def test_request_strategy_performance_target(self):
        """
        Test that request_strategy meets <100ms performance target.

        This validates Issue #99 performance requirement for strategic operations.
        """
        import time

        mock_response = """{
            "strategy_id": "performance_001",
            "experiments": [],
            "strategic_insights": [],
            "next_checkpoints": []
        }"""

        self.mock_strategic_process.send_message.return_value = mock_response

        # Measure performance
        start_time = time.time()
        result = self.strategist.request_strategy(self.game_state, self.recent_results)
        end_time = time.time()

        execution_time_ms = (end_time - start_time) * 1000

        # Verify performance target
        self.assertLess(
            execution_time_ms,
            100,
            f"request_strategy took {execution_time_ms:.1f}ms, target is <100ms",
        )

        # Verify it still works correctly
        self.assertIsInstance(result, dict)

    @pytest.mark.skip(
        reason="API changed - _validate_strategic_process method removed, validation now at runtime"
    )
    def test_strategic_process_validation(self):
        """Test validation that strategic process is available."""
        # Test successful case
        # self.assertTrue(self.strategist._validate_strategic_process())

        # Test failure case - no strategic process available
        # self.mock_manager.get_strategic_process.return_value = None
        # self.assertFalse(self.strategist._validate_strategic_process())

    @pytest.mark.skip(
        reason="API changed - strategic process check now happens at runtime, not initialization"
    )
    def test_initialization_fails_with_no_strategic_process(self):
        """Test OpusStrategist fails gracefully when no strategic process available."""
        self.mock_manager.get_strategic_process.return_value = None

        from claudelearnspokemon.opus_strategist import OpusStrategist

        with self.assertRaises(RuntimeError) as context:
            OpusStrategist(self.mock_manager)

        self.assertIn("strategic process", str(context.exception).lower())


@pytest.mark.skip(
    reason="API changed - analyze_parallel_results method removed from new OpusStrategist API"
)
@pytest.mark.medium
class TestParallelResultsAnalysis(unittest.TestCase):
    """Test core parallel results analysis functionality."""

    def setUp(self):
        """Set up test environment with mock data."""
        self.mock_manager = Mock()
        self.mock_strategic_process = Mock()
        self.mock_manager.get_strategic_process.return_value = self.mock_strategic_process

        from claudelearnspokemon.opus_strategist import OpusStrategist

        self.strategist = OpusStrategist(self.mock_manager)

        # Create test execution results with patterns
        self.parallel_results = self._create_test_parallel_results()

    def _create_test_parallel_results(self) -> list[ExecutionResult]:
        """Create realistic test data for parallel execution results."""
        return [
            ExecutionResult(
                worker_id="worker_1",
                success=True,
                execution_time=1.23,
                actions_taken=["A", "B", "START", "A", "RIGHT"],
                final_state={"x": 10, "y": 5, "level": 3},
                performance_metrics={"frame_rate": 60, "input_lag": 16.7},
                discovered_patterns=["menu_optimization", "movement_sequence"],
            ),
            ExecutionResult(
                worker_id="worker_2",
                success=True,
                execution_time=1.45,
                actions_taken=["A", "B", "START", "B", "RIGHT"],
                final_state={"x": 12, "y": 5, "level": 3},
                performance_metrics={"frame_rate": 59, "input_lag": 18.2},
                discovered_patterns=["menu_optimization", "alternate_sequence"],
            ),
            ExecutionResult(
                worker_id="worker_3",
                success=False,
                execution_time=2.1,
                actions_taken=["A", "START", "LEFT", "A"],
                final_state={"x": 8, "y": 4, "level": 2},
                performance_metrics={"frame_rate": 45, "input_lag": 22.1},
                discovered_patterns=["failed_sequence"],
            ),
            ExecutionResult(
                worker_id="worker_4",
                success=True,
                execution_time=1.18,
                actions_taken=["B", "A", "START", "A", "RIGHT"],
                final_state={"x": 11, "y": 5, "level": 3},
                performance_metrics={"frame_rate": 61, "input_lag": 15.9},
                discovered_patterns=["menu_optimization", "speed_optimization"],
            ),
        ]

    def test_opus_strategist_identifies_patterns_across_parallel_results(self):
        """
        Test core requirement: identify patterns across multiple parallel execution results.

        This test validates the strategic intelligence layer can:
        1. Analyze execution results from multiple parallel streams
        2. Identify common patterns and correlations
        3. Extract strategic insights for learning acceleration
        4. Generate actionable pattern recommendations
        """
        # Mock strategic process response with pattern analysis
        self.mock_strategic_process.send_message.return_value = {
            "identified_patterns": [
                {
                    "pattern": "menu_optimization",
                    "frequency": 3,
                    "success_correlation": 1.0,
                    "performance_impact": "reduces execution time by 15%",
                },
                {
                    "pattern": "right_movement_sequence",
                    "frequency": 3,
                    "success_correlation": 1.0,
                    "performance_impact": "consistent position advancement",
                },
            ],
            "correlations": [
                {
                    "variables": ["execution_time", "success_rate"],
                    "correlation": -0.73,
                    "significance": "strong negative correlation",
                },
                {
                    "variables": ["frame_rate", "input_lag"],
                    "correlation": -0.85,
                    "significance": "strong performance relationship",
                },
            ],
            "strategic_insights": [
                "Menu optimization pattern shows consistent 100% success rate",
                "Performance correlation indicates frame_rate > 58 improves success",
                "Failed sequences correlate with LEFT movements - avoid in speedruns",
            ],
        }

        # Execute parallel results analysis
        analysis_result = self.strategist.analyze_parallel_results(self.parallel_results)

        # Verify pattern identification
        self.assertIsInstance(analysis_result, list)
        self.assertGreater(len(analysis_result), 0)

        # Verify strategic process was called with structured data
        self.mock_strategic_process.send_message.assert_called_once()
        call_args = self.mock_strategic_process.send_message.call_args[0][0]

        # Validate message structure contains parallel results data
        self.assertIn("parallel_results", call_args.lower())
        self.assertIn("pattern", call_args.lower())
        self.assertIn("correlation", call_args.lower())

        # Verify analysis results contain expected pattern insights
        patterns = analysis_result
        self.assertIn("menu_optimization", str(patterns))
        self.assertIn("correlation", str(patterns))

    def test_statistical_correlation_analysis(self):
        """Test statistical correlation analysis across parallel execution results."""
        # Configure mock to return correlation data
        self.mock_strategic_process.send_message.return_value = {
            "correlation_matrix": {
                ("execution_time", "success_rate"): -0.73,
                ("frame_rate", "input_lag"): -0.85,
                ("actions_count", "execution_time"): 0.67,
            },
            "significant_correlations": [
                {"variables": ["frame_rate", "input_lag"], "r": -0.85, "p_value": 0.001}
            ],
        }

        # Execute correlation analysis
        analysis_result = self.strategist.analyze_parallel_results(self.parallel_results)

        # Verify correlation analysis was performed
        self.assertIsNotNone(analysis_result)
        # Strategic process should receive structured correlation request
        call_message = self.mock_strategic_process.send_message.call_args[0][0]
        self.assertIn("correlation", call_message.lower())
        self.assertIn("statistical", call_message.lower())

    def test_pattern_frequency_analysis(self):
        """Test frequency analysis of discovered patterns across results."""
        _ = self.strategist.analyze_parallel_results(self.parallel_results)

        # Verify strategic process analyzes pattern frequencies
        call_message = self.mock_strategic_process.send_message.call_args[0][0]
        self.assertIn("frequency", call_message.lower())
        self.assertIn("pattern", call_message.lower())

    def test_handles_mixed_success_failure_results(self):
        """Test analysis handles mixed successful and failed execution results."""
        # Results contain both successes and failures
        mixed_results = self.parallel_results  # Already contains success/failure mix

        analysis_result = self.strategist.analyze_parallel_results(mixed_results)

        # Should handle mixed results without errors
        self.assertIsNotNone(analysis_result)

        # Strategic process should receive both successful and failed results
        call_message = self.mock_strategic_process.send_message.call_args[0][0]
        self.assertIn("success", call_message.lower())
        self.assertIn("failed", call_message.lower())


@pytest.mark.skip(
    reason="API changed - analyze_parallel_results method removed from new OpusStrategist API"
)
@pytest.mark.medium
class TestResultCompression(unittest.TestCase):
    """Test result compression for efficient strategic processing."""

    def setUp(self):
        """Set up test environment for compression testing."""
        self.mock_manager = Mock()
        self.mock_strategic_process = Mock()
        self.mock_manager.get_strategic_process.return_value = self.mock_strategic_process

        from claudelearnspokemon.opus_strategist import OpusStrategist

        self.strategist = OpusStrategist(self.mock_manager)

    def test_opus_strategist_compresses_results_summary_efficiently(self):
        """
        Test core requirement: compress result summaries for efficient strategic analysis.

        Performance target: <100ms for pattern queries through optimized compression.
        Validates that large result sets are compressed to essential patterns only.
        """
        # Create large result set to test compression efficiency
        large_results = []
        for i in range(100):  # 100 execution results
            large_results.append(
                ExecutionResult(
                    worker_id=f"worker_{i % 4}",
                    success=(i % 3 != 0),  # Mix of success/failure
                    execution_time=1.0 + (i * 0.01),
                    actions_taken=[f"action_{j}" for j in range(10 + i % 5)],
                    final_state={"x": i, "y": i % 10, "level": i % 5},
                    performance_metrics={"frame_rate": 60 - (i % 15), "input_lag": 16 + (i % 8)},
                    discovered_patterns=[f"pattern_{i % 7}", "common_pattern"],
                )
            )

        # Mock compressed response from strategic process
        self.mock_strategic_process.send_message.return_value = {
            "compressed_summary": {
                "total_results": 100,
                "success_rate": 0.67,
                "dominant_patterns": ["common_pattern", "pattern_1", "pattern_2"],
                "performance_summary": {"avg_execution_time": 1.5, "avg_frame_rate": 52.5},
                "key_correlations": [{"vars": ["frame_rate", "success"], "r": 0.23}],
            },
            "compression_ratio": 0.95,  # 95% size reduction
            "processing_insights": [
                "common_pattern appears in 100% of results - critical for speedrun",
                "Frame rate correlation weak but positive for success",
            ],
        }

        # Measure compression performance
        start_time = time.time()
        analysis_result = self.strategist.analyze_parallel_results(large_results)
        compression_time = time.time() - start_time

        # Validate performance target: <100ms for result compression
        self.assertLess(
            compression_time * 1000,
            100,
            f"Compression took {compression_time*1000:.1f}ms, target is <100ms",
        )

        # Verify compression was applied
        self.assertIsNotNone(analysis_result)

        # Strategic process should receive compression request
        call_message = self.mock_strategic_process.send_message.call_args[0][0]
        self.assertIn("compress", call_message.lower())
        self.assertIn("summary", call_message.lower())

    def test_compression_preserves_critical_patterns(self):
        """Test that compression preserves the most critical patterns for strategic analysis."""
        # Create results with clear critical patterns
        results_with_critical_patterns = [
            ExecutionResult(
                worker_id="worker_1",
                success=True,
                execution_time=1.0,
                actions_taken=["CRITICAL", "A", "B"],
                final_state={"level": 5},
                performance_metrics={"frame_rate": 60},
                discovered_patterns=["speed_critical", "success_pattern"],
            ),
            ExecutionResult(
                worker_id="worker_2",
                success=True,
                execution_time=1.1,
                actions_taken=["CRITICAL", "B", "A"],
                final_state={"level": 5},
                performance_metrics={"frame_rate": 59},
                discovered_patterns=["speed_critical", "success_pattern"],
            ),
            ExecutionResult(
                worker_id="worker_3",
                success=False,
                execution_time=2.0,
                actions_taken=["SLOW", "A", "B"],
                final_state={"level": 2},
                performance_metrics={"frame_rate": 30},
                discovered_patterns=["failure_pattern"],
            ),
        ]

        self.mock_strategic_process.send_message.return_value = {
            "critical_patterns_preserved": ["speed_critical", "success_pattern"],
            "compression_applied": True,
        }

        _ = self.strategist.analyze_parallel_results(results_with_critical_patterns)

        # Verify critical patterns are preserved in compression
        call_message = self.mock_strategic_process.send_message.call_args[0][0]
        self.assertIn("critical", call_message.lower())
        self.assertIn("preserve", call_message.lower())

    def test_large_result_set_memory_efficiency(self):
        """Test memory-efficient handling of large parallel result sets."""
        # Create very large result set (1000 results) to test memory handling
        large_result_set = []
        for i in range(1000):
            large_result_set.append(
                ExecutionResult(
                    worker_id=f"worker_{i % 4}",
                    success=True,
                    execution_time=1.0,
                    actions_taken=["A"] * (i % 10 + 1),  # Variable action lengths
                    final_state={"data": f"state_{i}"},
                    performance_metrics={"metric": i},
                    discovered_patterns=[f"pattern_{i % 20}"],
                )
            )

        self.mock_strategic_process.send_message.return_value = {
            "memory_efficient_summary": True,
            "pattern_count": 20,
            "compressed_size": "95% reduction",
        }

        # Should handle large result sets without memory issues
        analysis_result = self.strategist.analyze_parallel_results(large_result_set)
        self.assertIsNotNone(analysis_result)


@pytest.mark.skip(
    reason="API changed - analyze_parallel_results method removed from new OpusStrategist API"
)
@pytest.mark.fast
class TestPerformanceTargets(unittest.TestCase):
    """Test algorithmic performance targets for strategic processing."""

    def setUp(self):
        """Set up performance testing environment."""
        self.mock_manager = Mock()
        self.mock_strategic_process = Mock()
        self.mock_manager.get_strategic_process.return_value = self.mock_strategic_process

        from claudelearnspokemon.opus_strategist import OpusStrategist

        self.strategist = OpusStrategist(self.mock_manager)

    def test_pattern_query_performance_target(self):
        """Test that pattern queries complete within <100ms target."""
        # Create moderate result set for performance testing
        test_results = []
        for i in range(50):
            test_results.append(
                ExecutionResult(
                    worker_id=f"worker_{i % 4}",
                    success=True,
                    execution_time=1.0,
                    actions_taken=["A", "B"],
                    final_state={"x": i},
                    performance_metrics={"frame_rate": 60},
                    discovered_patterns=["pattern_a"],
                )
            )

        self.mock_strategic_process.send_message.return_value = {"patterns": ["pattern_a"]}

        # Measure pattern query performance
        start_time = time.time()
        self.strategist.analyze_parallel_results(test_results)
        query_time = time.time() - start_time

        # Validate <100ms performance target
        self.assertLess(
            query_time * 1000, 100, f"Pattern query took {query_time*1000:.1f}ms, target is <100ms"
        )

    def test_strategic_response_performance_target(self):
        """Test strategic response generation meets <500ms target."""
        test_results = [
            ExecutionResult(
                worker_id="worker_1",
                success=True,
                execution_time=1.0,
                actions_taken=["A"],
                final_state={},
                performance_metrics={},
                discovered_patterns=["test_pattern"],
            )
        ]

        # Mock strategic process with processing delay simulation
        def mock_slow_response(*args, **kwargs):
            time.sleep(0.1)  # Simulate 100ms processing
            return {"strategic_response": "test"}

        self.mock_strategic_process.send_message.side_effect = mock_slow_response

        # Measure strategic response time
        start_time = time.time()
        self.strategist.analyze_parallel_results(test_results)
        response_time = time.time() - start_time

        # Should meet <500ms target even with processing overhead
        self.assertLess(
            response_time * 1000,
            500,
            f"Strategic response took {response_time*1000:.1f}ms, target is <500ms",
        )


@pytest.mark.fast
class TestGameStateProcessing(unittest.TestCase):
    """Test game state processing functionality for strategic context formatting."""

    def setUp(self):
        """Set up test environment for game state processing."""
        self.mock_manager = Mock()
        self.mock_strategic_process = Mock()
        self.mock_manager.get_strategic_process.return_value = self.mock_strategic_process

        from claudelearnspokemon.opus_strategist import OpusStrategist

        self.strategist = OpusStrategist(self.mock_manager)

    def test_opus_strategist_formats_game_state_for_context(self):
        """
        Test core requirement: format game state data for strategic analysis by Opus.

        This test validates the strategic context formatting meets production requirements:
        1. Processes complex game state data efficiently (<50ms target)
        2. Extracts strategic information for Opus consumption
        3. Handles multiple game state representation formats
        4. Provides comprehensive strategic context
        """
        # Create comprehensive game state with all expected fields
        game_state = {
            "tiles": [[1, 2, 3] * 6 for _ in range(20)],  # 20x18 grid
            "position": {"x": 10, "y": 5},
            "player_position": (10, 5),
            "map_id": "route_1",
            "inventory": {"pokeball": 5, "potion": 3, "badge": 2},
            "health": 85,
            "level": 12,
            "pokemon_count": 3,
            "frame_count": 1500,
        }

        execution_results = [
            {
                "success": True,
                "execution_time": 1.2,
                "discovered_patterns": ["menu_optimization", "movement_sequence"],
                "final_state": {"x": 12, "y": 6},
            }
        ]

        # Test strategic context formatting
        start_time = time.time()
        strategic_context = self.strategist.format_game_state_for_context(
            game_state, execution_results, "strategic_analysis"
        )
        processing_time = (time.time() - start_time) * 1000

        # Validate performance target
        self.assertLess(
            processing_time, 50, f"Processing took {processing_time:.2f}ms (target: <50ms)"
        )

        # Verify comprehensive strategic context structure
        self.assertIsInstance(strategic_context, dict)
        self.assertIn("context_type", strategic_context)
        self.assertEqual(strategic_context["context_type"], "strategic_analysis")

        # Validate strategic analysis components
        required_sections = [
            "player_analysis",
            "environmental_analysis",
            "progress_analysis",
            "strategic_opportunities",
            "execution_analysis",
            "temporal_context",
            "data_quality",
        ]
        for section in required_sections:
            self.assertIn(section, strategic_context, f"Missing section: {section}")

        # Verify player analysis details
        player_analysis = strategic_context["player_analysis"]
        self.assertIn("position", player_analysis)
        self.assertIn("strategic_location_type", player_analysis)
        self.assertEqual(player_analysis["strategic_location_type"], "route")

        # Verify execution analysis integration
        execution_analysis = strategic_context["execution_analysis"]
        self.assertEqual(execution_analysis["result_count"], 1)
        self.assertEqual(execution_analysis["success_rate"], 1.0)
        self.assertIn("menu_optimization", execution_analysis["pattern_insights"])

    def test_game_state_formatting_handles_multiple_formats(self):
        """Test that game state formatting supports different representation formats."""

        # Test with GameState-like object
        class MockGameState:
            def __init__(self):
                self.tiles = [[1, 2] * 9 for _ in range(20)]
                self.position = type("Position", (), {"x": 5, "y": 3, "map_id": "viridian_city"})()
                self.inventory = {"pokeball": 2}

        gamestate_obj = MockGameState()

        strategic_context = self.strategist.format_game_state_for_context(
            {"tiles": gamestate_obj.tiles, "position": gamestate_obj.position}
        )

        self.assertIsInstance(strategic_context, dict)
        self.assertEqual(strategic_context["player_analysis"]["position"]["x"], 5)
        self.assertEqual(strategic_context["player_analysis"]["strategic_location_type"], "city")

    def test_game_state_formatting_handles_corrupted_data(self):
        """Test robust error handling for corrupted game state data."""
        # Test with corrupted tile data
        corrupted_game_state = {
            "tiles": "corrupted_data_not_array",
            "position": None,
            "inventory": {"corrupted": "data"},
        }

        strategic_context = self.strategist.format_game_state_for_context(corrupted_game_state)

        # Should provide fallback context instead of crashing
        self.assertIsInstance(strategic_context, dict)
        self.assertIn("data_quality", strategic_context)

        # Verify graceful degradation in analysis sections
        if strategic_context["context_type"] != "fallback":
            # Check for degraded status in analysis sections
            analysis_sections = ["player_analysis", "environmental_analysis", "progress_analysis"]
            for section in analysis_sections:
                if section in strategic_context:
                    # Either successful analysis or degraded with error info
                    analysis = strategic_context[section]
                    if "status" in analysis:
                        self.assertEqual(analysis["status"], "degraded")

    def test_game_state_formatting_handles_incomplete_data(self):
        """Test handling of incomplete game state data gracefully."""
        # Minimal game state with only basic information
        minimal_game_state = {
            "position": (8, 4),
            "map_id": "unknown_location",
        }

        strategic_context = self.strategist.format_game_state_for_context(minimal_game_state)

        self.assertIsInstance(strategic_context, dict)

        # Verify data quality assessment reflects incomplete data
        data_quality = strategic_context["data_quality"]
        self.assertLess(data_quality["completeness_score"], 1.0)
        self.assertGreater(len(data_quality["missing_fields"]), 0)

    def test_strategic_context_extraction_components(self):
        """Test individual strategic context extraction components."""
        game_state = {
            "tiles": [[50, 51, 200] * 6 for _ in range(20)],  # Include NPCs (tile 200)
            "position": {"x": 10, "y": 5},
            "map_id": "cerulean_gym",
            "inventory": {"pokeball": 2, "potion": 1},  # Low resources
            "health": 45,  # Low health
            "badges": 3,
        }

        strategic_context = self.strategist._extract_strategic_context(
            game_state, "pattern_discovery"
        )

        # Verify player analysis
        player_analysis = strategic_context["player_analysis"]
        self.assertEqual(player_analysis["strategic_location_type"], "gym")

        # Verify environmental analysis detects NPCs
        env_analysis = strategic_context["environmental_analysis"]
        self.assertGreater(len(env_analysis["npc_positions"]), 0)
        self.assertGreater(env_analysis["tile_diversity"], 1)

        # Verify strategic opportunities detection
        opportunities = strategic_context["strategic_opportunities"]
        opportunity_types = [opp["type"] for opp in opportunities]
        self.assertIn("gym_challenge", opportunity_types)
        self.assertIn("resource_acquisition", opportunity_types)  # Due to low pokeball count

    def test_player_position_analysis_multiple_formats(self):
        """Test player position analysis handles various data formats."""
        # Test with GamePosition-like object
        game_state_obj = {
            "position": type(
                "Position", (), {"x": 15, "y": 8, "map_id": "route_22", "facing_direction": "north"}
            )()
        }

        analysis = self.strategist._analyze_player_position(game_state_obj)

        self.assertEqual(analysis["position"]["x"], 15)
        self.assertEqual(analysis["position"]["y"], 8)
        self.assertEqual(analysis["map_context"], "route_22")
        self.assertEqual(analysis["facing_direction"], "north")
        self.assertEqual(analysis["strategic_location_type"], "route")

        # Test with tuple format
        game_state_tuple = {"player_position": (7, 3)}
        analysis_tuple = self.strategist._analyze_player_position(game_state_tuple)

        self.assertEqual(analysis_tuple["position"]["x"], 7)
        self.assertEqual(analysis_tuple["position"]["y"], 3)

    def test_tile_environment_analysis_with_numpy_arrays(self):
        """Test tile environment analysis with numpy arrays and various tile types."""
        import numpy as np

        # Create tile grid with strategic elements
        tiles = np.zeros((20, 18), dtype=np.uint8)
        tiles[5:7, 5:7] = 100  # Some terrain
        tiles[10, 9] = 255  # Player tile
        tiles[3, 3] = 205  # NPC tile
        tiles[15, 15] = 210  # Another NPC

        game_state = {"tiles": tiles}

        analysis = self.strategist._analyze_tile_environment(game_state)

        self.assertEqual(analysis["grid_dimensions"]["height"], 20)
        self.assertEqual(analysis["grid_dimensions"]["width"], 18)
        self.assertGreaterEqual(
            analysis["tile_diversity"], 4
        )  # At least background, terrain, player, NPCs
        self.assertEqual(len(analysis["npc_positions"]), 2)  # Two NPCs placed
        self.assertGreater(analysis["walkable_area_ratio"], 0.8)  # Most tiles should be walkable

    def test_game_progress_analysis_comprehensive(self):
        """Test comprehensive game progress analysis."""
        game_state = {
            "inventory": {
                "pokeball": 8,
                "potion": 5,
                "super_potion": 2,
                "badge": 4,
                "rare_candy": 1,
            },
            "badges": 4,
            "pokemon_count": 5,
            "level": 28,
            "health": {"current": 120, "max": 140},
        }

        analysis = self.strategist._analyze_game_progress(game_state)

        # Verify inventory analysis
        inventory_status = analysis["inventory_status"]
        self.assertEqual(inventory_status["total_items"], 5)
        self.assertEqual(inventory_status["item_diversity"], 5)

        # Verify strategic resources tracking
        strategic_resources = analysis["strategic_resources"]
        self.assertEqual(strategic_resources["pokeballs"], 8)
        self.assertEqual(strategic_resources["healing_items"], 5)

        # Verify progress indicators
        self.assertEqual(analysis["badges_earned"], 4)
        self.assertEqual(analysis["pokemon_count"], 5)
        self.assertEqual(analysis["level_progression"], 28)

        # Verify completion tracking
        completion = analysis["completion_indicators"]
        self.assertEqual(completion["badge_progress"], 0.5)  # 4/8 badges

    def test_execution_results_analysis(self):
        """Test analysis of execution results for strategic context."""
        execution_results = [
            {
                "success": True,
                "execution_time": 1.1,
                "discovered_patterns": ["fast_menu", "optimal_route"],
                "final_state": {"x": 10, "y": 5},
            },
            {
                "success": False,
                "execution_time": 2.5,
                "error": "timeout",
                "final_state": {"x": 8, "y": 4},
            },
            {
                "success": True,
                "execution_time": 0.9,
                "discovered_patterns": ["fast_menu", "speed_optimization"],
                "final_state": {"x": 12, "y": 6},
            },
        ]

        analysis = self.strategist._analyze_execution_results(execution_results)

        self.assertEqual(analysis["result_count"], 3)
        self.assertAlmostEqual(analysis["success_rate"], 2 / 3, places=2)

        # Verify performance metrics
        perf_metrics = analysis["performance_metrics"]
        self.assertAlmostEqual(perf_metrics["avg_execution_time"], (1.1 + 2.5 + 0.9) / 3, places=2)
        self.assertEqual(perf_metrics["max_execution_time"], 2.5)
        self.assertEqual(perf_metrics["min_execution_time"], 0.9)

        # Verify pattern insights extraction
        pattern_insights = analysis["pattern_insights"]
        self.assertIn("fast_menu", pattern_insights)
        self.assertIn("optimal_route", pattern_insights)
        self.assertIn("speed_optimization", pattern_insights)

        # Verify failure analysis
        failure_analysis = analysis["failure_analysis"]
        self.assertEqual(len(failure_analysis), 1)
        self.assertEqual(failure_analysis[0]["reason"], "timeout")

    def test_data_quality_assessment(self):
        """Test data quality assessment for reliability metrics."""
        # High quality game state
        complete_game_state = {
            "tiles": [[1, 2] * 9 for _ in range(20)],
            "position": (10, 5),
            "inventory": {"pokeball": 5},
            "map_id": "route_1",
            "health": 100,
            "level": 15,
            "pokemon_count": 3,
        }

        quality = self.strategist._assess_data_quality(complete_game_state)

        self.assertGreaterEqual(quality["completeness_score"], 0.8)
        self.assertEqual(quality["data_integrity"], "good")
        self.assertLessEqual(len(quality["missing_fields"]), 1)
        self.assertGreaterEqual(quality["reliability_score"], 0.9)

        # Poor quality game state
        incomplete_game_state = {"position": (5, 3)}

        quality_poor = self.strategist._assess_data_quality(incomplete_game_state)

        self.assertLess(quality_poor["completeness_score"], 0.5)
        self.assertEqual(quality_poor["data_integrity"], "poor")
        self.assertGreater(len(quality_poor["missing_fields"]), 5)

    def test_circuit_breaker_integration_with_game_state_processing(self):
        """Test circuit breaker integration protects against systematic failures."""
        # Import CircuitState from circuit_breaker module
        import time

        from claudelearnspokemon.circuit_breaker import CircuitState

        # Simulate circuit breaker in open state with recent trip time
        # This ensures the recovery timeout hasn't elapsed, so is_available() returns False
        with self.strategist.circuit_breaker._lock:
            self.strategist.circuit_breaker._state = CircuitState.OPEN
            self.strategist.circuit_breaker._last_trip_time = time.time()  # Recent trip time

        game_state = {"position": (10, 5), "map_id": "test"}

        strategic_context = self.strategist.format_game_state_for_context(game_state)

        # Should return fallback context when circuit breaker is open
        self.assertEqual(strategic_context["context_type"], "fallback")
        self.assertEqual(strategic_context["fallback_reason"], "circuit_breaker_open")
        self.assertIn("strategic_recommendations", strategic_context)

    def test_performance_under_load_game_state_processing(self):
        """Test game state processing performance under load conditions."""
        # Create large, complex game state
        large_game_state = {
            "tiles": [[i % 256 for i in range(18)] for _ in range(20)],
            "position": {"x": 10, "y": 5},
            "map_id": "complex_dungeon_level_5",
            "inventory": {f"item_{i}": i for i in range(50)},  # Large inventory
            "health": 150,
            "level": 45,
            "pokemon_count": 6,
        }

        # Large execution results set
        execution_results = [
            {
                "success": i % 3 != 0,
                "execution_time": 1.0 + (i * 0.1),
                "discovered_patterns": [f"pattern_{i}", f"pattern_{i+1}"],
                "final_state": {"x": 10 + i, "y": 5 + i},
            }
            for i in range(20)
        ]

        # Test performance under load
        start_time = time.time()
        strategic_context = self.strategist.format_game_state_for_context(
            large_game_state, execution_results
        )
        processing_time = (time.time() - start_time) * 1000

        # Should still meet performance target even with large data
        self.assertLess(processing_time, 100, f"Load test took {processing_time:.2f}ms")

        # Verify comprehensive analysis despite load
        self.assertIsInstance(strategic_context, dict)
        self.assertIn("execution_analysis", strategic_context)
        self.assertEqual(strategic_context["execution_analysis"]["result_count"], 20)

    def test_strategic_opportunities_identification(self):
        """Test identification of various strategic opportunities from game state."""
        # Gym scenario
        gym_state = {
            "map_id": "pewter_gym",
            "inventory": {"pokeball": 3, "potion": 1},  # Low resources
        }

        opportunities_gym = self.strategist._identify_strategic_opportunities(gym_state)
        opportunity_types = [opp["type"] for opp in opportunities_gym]

        self.assertIn("gym_challenge", opportunity_types)
        self.assertIn("resource_acquisition", opportunity_types)
        self.assertIn("healing_preparation", opportunity_types)

        # Verify priority and risk assessment
        gym_opportunity = next(opp for opp in opportunities_gym if opp["type"] == "gym_challenge")
        self.assertEqual(gym_opportunity["priority"], "critical")
        self.assertEqual(gym_opportunity["risk_level"], "high")


@pytest.mark.medium
class TestGameStateProcessingIntegration(unittest.TestCase):
    """Test integration of game state processing with existing OpusStrategist functionality."""

    def setUp(self):
        """Set up integration test environment."""
        self.mock_manager = Mock()
        self.mock_strategic_process = Mock()
        self.mock_manager.get_strategic_process.return_value = self.mock_strategic_process

        from claudelearnspokemon.opus_strategist import OpusStrategist

        self.strategist = OpusStrategist(self.mock_manager)

    def test_game_state_context_integrates_with_strategic_planning(self):
        """Test that formatted game state context integrates with strategic planning workflow."""
        # Set up game state and mock Opus response
        game_state = {
            "tiles": [[1, 2, 3] * 6 for _ in range(20)],
            "position": (10, 5),
            "map_id": "route_1",
            "inventory": {"pokeball": 5, "potion": 3},
            "level": 12,
        }

        # Mock Opus strategic response
        mock_opus_response = {
            "strategy_id": "route_1_optimization",
            "experiments": [
                {
                    "id": "exp_1",
                    "name": "Route optimization experiment",
                    "checkpoint": "route_1_start",
                    "script_dsl": "MOVE RIGHT; BATTLE; MOVE UP",
                    "expected_outcome": "level progression",
                    "priority": "high",
                }
            ],
            "strategic_insights": [
                "Current position optimal for training",
                "Resource levels sufficient for extended exploration",
            ],
            "next_checkpoints": ["route_1_center", "route_1_exit"],
        }

        self.mock_strategic_process.send_message.return_value = str(mock_opus_response)

        # Test integration: format game state then request strategy
        formatted_context = self.strategist.format_game_state_for_context(game_state)

        # Simulate using formatted context in strategy request
        enhanced_game_state = dict(game_state)
        enhanced_game_state.update({"strategic_context": formatted_context})

        strategy_response = self.strategist.get_strategy(
            enhanced_game_state, context={"analysis_type": "route_optimization"}
        )

        # Verify integration worked
        self.assertIsNotNone(strategy_response)
        self.mock_strategic_process.send_message.assert_called_once()

        # Verify the prompt included strategic context
        call_args = self.mock_strategic_process.send_message.call_args[0][0]
        self.assertIn("strategic", call_args.lower())

    def test_game_state_processing_metrics_integration(self):
        """Test that game state processing integrates with OpusStrategist metrics."""
        game_state = {"position": (5, 3), "map_id": "test_location"}

        # Process game state multiple times
        for _ in range(5):
            self.strategist.format_game_state_for_context(game_state)

        # Verify metrics are being tracked (circuit breaker should record successes)
        metrics = self.strategist.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("circuit_breaker_state", metrics)


if __name__ == "__main__":
    # Run with performance timing
    print("=== OpusStrategist Test Suite - Performance Focus ===")
    unittest.main(verbosity=2)
