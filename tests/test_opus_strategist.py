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
from typing import Any
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

    def _create_test_parallel_results(self) -> list[dict[str, Any]]:
        """Create realistic test data for parallel execution results."""
        return [
            {
                "worker_id": "worker_1",
                "success": True,
                "execution_time": 1.23,
                "actions_taken": ["A", "B", "START", "A", "RIGHT"],
                "final_state": {"x": 10, "y": 5, "level": 3},
                "performance_metrics": {"frame_rate": 60, "input_lag": 16.7},
                "discovered_patterns": ["menu_optimization", "movement_sequence"],
            },
            {
                "worker_id": "worker_2",
                "success": True,
                "execution_time": 1.45,
                "actions_taken": ["A", "B", "START", "B", "RIGHT"],
                "final_state": {"x": 12, "y": 5, "level": 3},
                "performance_metrics": {"frame_rate": 59, "input_lag": 18.2},
                "discovered_patterns": ["menu_optimization", "alternate_sequence"],
            },
            {
                "worker_id": "worker_3",
                "success": False,
                "execution_time": 2.1,
                "actions_taken": ["A", "START", "LEFT", "A"],
                "final_state": {"x": 8, "y": 4, "level": 2},
                "performance_metrics": {"frame_rate": 45, "input_lag": 22.1},
                "discovered_patterns": ["failed_sequence"],
            },
            {
                "worker_id": "worker_4",
                "success": True,
                "execution_time": 1.18,
                "actions_taken": ["B", "A", "START", "A", "RIGHT"],
                "final_state": {"x": 11, "y": 5, "level": 3},
                "performance_metrics": {"frame_rate": 61, "input_lag": 15.9},
                "discovered_patterns": ["menu_optimization", "speed_optimization"],
            },
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
        # Mock strategic process response with pattern analysis (JSON format)
        self.mock_strategic_process.send_message.return_value = """{
            "identified_patterns": [
                {
                    "pattern": "menu_optimization",
                    "frequency": 3,
                    "success_correlation": 1.0,
                    "performance_impact": "reduces execution time by 15%"
                },
                {
                    "pattern": "right_movement_sequence",
                    "frequency": 3,
                    "success_correlation": 1.0,
                    "performance_impact": "consistent position advancement"
                }
            ],
            "correlations": [
                {
                    "variables": ["execution_time", "success_rate"],
                    "correlation": -0.73,
                    "significance": "strong negative correlation"
                },
                {
                    "variables": ["frame_rate", "input_lag"],
                    "correlation": -0.85,
                    "significance": "strong performance relationship"
                }
            ],
            "strategic_insights": [
                "Menu optimization pattern shows consistent 100% success rate",
                "Performance correlation indicates frame_rate > 58 improves success",
                "Failed sequences correlate with LEFT movements - avoid in speedruns"
            ],
            "optimization_opportunities": [
                "Prioritize menu optimization patterns for consistent performance",
                "Use frame rate monitoring to predict success rates"
            ],
            "risk_factors": [
                "LEFT movement patterns show high failure correlation"
            ]
        }"""

        # Execute parallel results analysis
        analysis_result = self.strategist.analyze_parallel_results(self.parallel_results)

        # Verify analysis result structure
        self.assertIsInstance(analysis_result, list)
        self.assertEqual(
            len(analysis_result), 3
        )  # pattern_identification, statistical_correlation, strategic_insights

        # Verify analysis types are present
        analysis_types = [result["analysis_type"] for result in analysis_result]
        expected_types = ["pattern_identification", "statistical_correlation", "strategic_insights"]
        for expected_type in expected_types:
            self.assertIn(expected_type, analysis_types)

        # Verify strategic process was called with structured data
        self.mock_strategic_process.send_message.assert_called_once()
        call_args = self.mock_strategic_process.send_message.call_args[0][0]

        # Validate message structure contains parallel results data
        self.assertIn("PARALLEL RESULTS", call_args)
        self.assertIn("PATTERN ANALYSIS", call_args)
        self.assertIn("CORRELATION ANALYSIS", call_args)

        # Verify strategic insights contain expected pattern information
        strategic_insights_result = next(
            r for r in analysis_result if r["analysis_type"] == "strategic_insights"
        )
        insights_data = strategic_insights_result["results"]

        self.assertIn("identified_patterns", insights_data)
        self.assertIn("strategic_insights", insights_data)
        self.assertIn("menu_optimization", str(insights_data))

    def test_statistical_correlation_analysis(self):
        """Test statistical correlation analysis across parallel execution results."""
        # Configure mock to return correlation data (JSON format)
        self.mock_strategic_process.send_message.return_value = """{
            "identified_patterns": [],
            "correlations": [
                {"variables": ["frame_rate", "input_lag"], "correlation": -0.85, "significance": "strong"}
            ],
            "strategic_insights": [
                "Strong correlation found between frame_rate and input_lag"
            ],
            "optimization_opportunities": [
                "Monitor frame_rate to predict input_lag performance"
            ],
            "risk_factors": []
        }"""

        # Execute correlation analysis
        analysis_result = self.strategist.analyze_parallel_results(self.parallel_results)

        # Verify correlation analysis was performed
        self.assertIsNotNone(analysis_result)
        self.assertIsInstance(analysis_result, list)

        # Find statistical correlation result
        correlation_result = next(
            r for r in analysis_result if r["analysis_type"] == "statistical_correlation"
        )
        self.assertIsNotNone(correlation_result)

        # Strategic process should receive structured correlation request
        call_message = self.mock_strategic_process.send_message.call_args[0][0]
        self.assertIn("CORRELATION ANALYSIS", call_message)
        self.assertIn("STATISTICAL", call_message)

    def test_pattern_frequency_analysis(self):
        """Test frequency analysis of discovered patterns across results."""
        # Mock response for frequency analysis
        self.mock_strategic_process.send_message.return_value = """{
            "identified_patterns": [
                {"pattern": "menu_optimization", "frequency": 3, "strategic_value": "high"}
            ],
            "correlations": [],
            "strategic_insights": ["High-frequency patterns identified"],
            "optimization_opportunities": [],
            "risk_factors": []
        }"""

        _ = self.strategist.analyze_parallel_results(self.parallel_results)

        # Verify strategic process analyzes pattern frequencies
        call_message = self.mock_strategic_process.send_message.call_args[0][0]
        self.assertIn("frequency", call_message.lower())
        self.assertIn("pattern", call_message.lower())

    def test_handles_mixed_success_failure_results(self):
        """Test analysis handles mixed successful and failed execution results."""
        # Mock response for mixed results
        self.mock_strategic_process.send_message.return_value = """{
            "identified_patterns": [
                {"pattern": "menu_optimization", "success_rate": 0.75}
            ],
            "correlations": [],
            "strategic_insights": ["Mixed success/failure results analyzed"],
            "optimization_opportunities": ["Focus on successful patterns"],
            "risk_factors": [{"pattern": "failed_sequence", "failure_rate": 0.25}]
        }"""

        # Results contain both successes and failures
        mixed_results = self.parallel_results  # Already contains success/failure mix

        analysis_result = self.strategist.analyze_parallel_results(mixed_results)

        # Should handle mixed results without errors
        self.assertIsNotNone(analysis_result)

        # Strategic process should receive both successful and failed results
        call_message = self.mock_strategic_process.send_message.call_args[0][0]
        self.assertIn("success", call_message.lower())
        self.assertIn("failed", call_message.lower())


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


if __name__ == "__main__":
    # Run with performance timing
    print("=== OpusStrategist Test Suite - Performance Focus ===")
    unittest.main(verbosity=2)
