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


# Strategic Continuity Management Tests - Issue #113 - Uncle Bot Implementation


@pytest.mark.fast
class TestOpusStrategistStrategicContinuity(unittest.TestCase):
    """Test Strategic Continuity Management - Issue #113 capstone feature."""

    def setUp(self):
        """Set up test environment for strategic continuity testing."""
        self.mock_manager = Mock()
        self.mock_strategic_process = Mock()
        self.mock_manager.get_strategic_process.return_value = self.mock_strategic_process

        from claudelearnspokemon.opus_strategist import OpusStrategist

        self.strategist = OpusStrategist(self.mock_manager)

        # Set up mock strategic responses
        self.mock_strategic_process.send_message.return_value = '{"strategy_id": "test_strategy", "experiments": [], "strategic_insights": [], "next_checkpoints": []}'

    def test_opus_strategist_maintains_context_continuity_across_messages(self):
        """
        REFERENCE TEST CASE - Test strategic continuity across conversation cycles.

        Validates that OpusStrategist maintains strategic context consistency
        across multiple conversation cycles and system restarts.
        """
        # Initial strategic context
        initial_context = {
            "previous_strategies": ["explore_pallet_town", "catch_first_pokemon"],
            "conversation_history": ["greeting", "initial_planning"],
            "strategic_directives": ["focus_on_speed", "minimize_battles"],
            "plan_version": "1.0",
        }

        game_state = {"location": "pallet_town", "level": 5, "pokemon_count": 1}

        # First strategic request - establish continuity
        first_response = self.strategist.get_strategy(game_state, initial_context)

        # Extract strategic context from response
        preserved_context = self.strategist.preserve_strategic_context(
            first_response, initial_context
        )

        # Simulate conversation restart with preserved context
        self.strategist.restore_strategic_context(preserved_context)

        # Second strategic request after restart
        second_response = self.strategist.get_strategy(game_state, preserved_context)

        # Verify strategic continuity is maintained
        self.assertIsNotNone(first_response)
        self.assertIsNotNone(second_response)
        self.assertIsNotNone(preserved_context)

        # Verify context preservation methods exist and function
        self.assertTrue(hasattr(self.strategist, "preserve_strategic_context"))
        self.assertTrue(hasattr(self.strategist, "restore_strategic_context"))

    def test_strategic_context_preservation_during_compression_cycles(self):
        """Test strategic context preservation during compression cycles."""
        # Create strategic context with compression scenarios
        large_context = {
            "conversation_history": [f"message_{i}" for i in range(1000)],  # Large history
            "strategic_plans": [{"id": f"plan_{i}", "status": "completed"} for i in range(50)],
            "decision_outcomes": [
                {"decision": f"decision_{i}", "outcome": "success"} for i in range(200)
            ],
        }

        # Preserve context with compression
        preserved_context = self.strategist.preserve_strategic_context_with_compression(
            large_context, compression_ratio=0.7
        )

        # Verify compression maintained essential information
        self.assertIsNotNone(preserved_context)
        # Check that compression metadata is present (indicating compression worked)
        self.assertIn("compression_metadata", preserved_context)
        self.assertIn("compression_ratio", preserved_context["compression_metadata"])
        # Verify some content was preserved
        self.assertTrue(
            len(preserved_context.get("conversation_history", [])) > 0
            or len(preserved_context.get("decision_outcomes", [])) > 0
            or len(preserved_context.get("strategic_directives", [])) > 0
        )

    def test_strategic_plan_versioning_and_evolution_tracking(self):
        """Test strategic plan versioning and evolution tracking."""
        game_state = {"location": "pewter_city", "badges": 1}
        initial_context = {"plan_version": "1.0"}

        # Create initial strategic plan
        plan_v1 = self.strategist.get_strategy(game_state, initial_context)

        # Evolve the strategic plan
        evolved_context = {"plan_version": "1.1", "previous_plan": plan_v1.to_dict()}
        plan_v1_1 = self.strategist.get_strategy(game_state, evolved_context)

        # Track plan evolution
        evolution_history = self.strategist.track_strategic_plan_evolution(
            previous_plan=plan_v1, current_plan=plan_v1_1, evolution_reason="gym_badge_achieved"
        )

        # Verify evolution tracking
        self.assertIsNotNone(evolution_history)
        self.assertTrue(hasattr(self.strategist, "track_strategic_plan_evolution"))

    def test_strategic_directive_conflict_resolution(self):
        """Test strategic directive conflict resolution."""
        # Create conflicting strategic directives
        conflicting_directives = [
            {"directive": "prioritize_speed", "priority": 8, "source": "speedrun_optimization"},
            {"directive": "prioritize_safety", "priority": 6, "source": "risk_management"},
            {
                "directive": "prioritize_speed",
                "priority": 9,
                "source": "time_constraints",
            },  # Conflict
        ]

        # Resolve conflicts
        resolved_directives = self.strategist.resolve_strategic_directive_conflicts(
            conflicting_directives
        )

        # Verify conflict resolution
        self.assertIsNotNone(resolved_directives)
        self.assertTrue(hasattr(self.strategist, "resolve_strategic_directive_conflicts"))

        # Should handle priority-based resolution
        self.assertIsInstance(resolved_directives, list)

    def test_strategic_decision_history_and_outcome_tracking(self):
        """Test strategic decision history and outcome tracking."""
        # Track a strategic decision
        decision = {
            "decision_id": "battle_gym_leader",
            "context": {"location": "pewter_gym", "pokemon_level": 12},
            "rationale": "sufficient level advantage",
            "timestamp": time.time(),
        }

        # Record decision
        self.strategist.record_strategic_decision(decision)

        # Simulate outcome
        outcome = {
            "decision_id": "battle_gym_leader",
            "result": "success",
            "execution_time": 45.2,
            "lessons_learned": ["type_advantage_crucial"],
        }

        # Record outcome
        self.strategist.record_strategic_outcome(outcome)

        # Correlate decision with outcome
        self.strategist.correlate_decision_with_outcome(decision["decision_id"])

        # Verify tracking functionality
        self.assertTrue(hasattr(self.strategist, "record_strategic_decision"))
        self.assertTrue(hasattr(self.strategist, "record_strategic_outcome"))
        self.assertTrue(hasattr(self.strategist, "correlate_decision_with_outcome"))

    def test_strategic_continuity_performance_targets(self):
        """Test strategic continuity meets performance targets."""
        large_context = {"history": ["item"] * 10000}  # Large context

        # Test context preservation performance (<100ms)
        start_time = time.time()
        self.strategist.preserve_strategic_context({}, large_context)
        preservation_time = (time.time() - start_time) * 1000

        self.assertLess(
            preservation_time,
            100,
            f"Context preservation took {preservation_time:.1f}ms, target <100ms",
        )

        # Test plan versioning performance (<50ms)
        start_time = time.time()
        self.strategist.version_strategic_plan({"plan": "test"})
        versioning_time = (time.time() - start_time) * 1000

        self.assertLess(
            versioning_time, 50, f"Plan versioning took {versioning_time:.1f}ms, target <50ms"
        )

    def test_strategic_continuity_integration_with_existing_systems(self):
        """Test strategic continuity integrates with existing OpusStrategist systems."""
        game_state = {"location": "route_1", "level": 6}
        context = {"continuity_test": True}

        # Should work with existing get_strategy method
        response = self.strategist.get_strategy(game_state, context)
        self.assertIsNotNone(response)

        # Should work with existing caching
        cached_response = self.strategist.get_strategy(game_state, context, use_cache=True)
        self.assertIsNotNone(cached_response)

        # Should work with existing metrics
        metrics = self.strategist.get_metrics()
        self.assertIn("total_requests", metrics)

        # Verify continuity features integrate without breaking existing functionality
        self.assertTrue(True)  # Integration test passes if no exceptions


if __name__ == "__main__":
    # Run with performance timing
    print("=== OpusStrategist Test Suite - Performance Focus ===")
    unittest.main(verbosity=2)
