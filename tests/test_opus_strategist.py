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
class TestOpusStrategistContextSummarization(unittest.TestCase):
    """Test context summarization functionality for conversation continuity."""

    def setUp(self):
        """Set up test environment with mocked ClaudeCodeManager."""
        # Mock ClaudeCodeManager with strategic process
        self.mock_manager = Mock()
        self.mock_strategic_process = Mock()
        self.mock_manager.get_strategic_process.return_value = self.mock_strategic_process

        # Import and create strategist
        from claudelearnspokemon.opus_strategist import OpusStrategist, SummarizationStrategy

        self.strategist = OpusStrategist(self.mock_manager)
        self.SummarizationStrategy = SummarizationStrategy

        # Create test conversation history
        self.test_conversation = self._create_test_conversation_history()

    def _create_test_conversation_history(self) -> list[dict[str, Any]]:
        """Create realistic conversation history for testing."""
        return [
            {
                "role": "user",
                "content": "I need a strategic approach for the Pokemon Red speedrun. Current objective is to reach Pewter City efficiently.",
                "timestamp": time.time() - 3600,
            },
            {
                "role": "assistant",
                "content": "Strategic plan: Focus on route optimization and menu management. Key pattern discovered: A+START sequence reduces menu time by 15%. Critical objective: minimize wild encounters on Route 1.",
                "timestamp": time.time() - 3500,
            },
            {
                "role": "user",
                "content": "The menu optimization pattern works perfectly! Found another effective sequence: B+A+RIGHT for inventory navigation. Should we test parallel approaches?",
                "timestamp": time.time() - 3400,
            },
            {
                "role": "assistant",
                "content": "Excellent discovery! The B+A+RIGHT pattern shows strong tactical value. Strategic insight: parallel execution can test route variations simultaneously. Goal is to identify fastest Viridian Forest path.",
                "timestamp": time.time() - 3300,
            },
            {
                "role": "user",
                "content": "Implemented parallel testing and discovered critical speedrun optimization: diagonal movement saves 2 seconds per screen transition. This is essential for record attempts.",
                "timestamp": time.time() - 3200,
            },
        ]

    def test_opus_strategist_maintains_context_continuity_across_messages(self):
        """
        Test core requirement: maintain strategic context continuity across conversation restarts.

        This validates the key test case from the design document that ensures
        summarization preserves critical discoveries and strategic direction.
        """
        # Test different summarization strategies
        for strategy in [
            self.SummarizationStrategy.COMPREHENSIVE,
            self.SummarizationStrategy.STRATEGIC,
            self.SummarizationStrategy.TACTICAL,
        ]:
            with self.subTest(strategy=strategy):
                # Execute summarization
                summary = self.strategist.summarize_learnings(
                    self.test_conversation, max_summary_length=800, strategy=strategy
                )

                # Validate summary structure
                self.assertIsNotNone(summary)
                self.assertEqual(summary.strategy, strategy)
                self.assertEqual(summary.total_messages, len(self.test_conversation))

                # Validate context continuity preservation
                self.assertTrue(len(summary.compressed_content) > 0)
                self.assertTrue(summary.compression_ratio > 0)

                # Ensure critical information is preserved based on strategy
                if strategy == self.SummarizationStrategy.COMPREHENSIVE:
                    self.assertTrue(len(summary.preserved_insights) > 0)
                    self.assertTrue(len(summary.critical_discoveries) > 0)
                    self.assertTrue(len(summary.current_objectives) > 0)

                elif strategy == self.SummarizationStrategy.STRATEGIC:
                    self.assertTrue(len(summary.preserved_insights) > 0)
                    # Strategic strategy focuses on high-level insights

                elif strategy == self.SummarizationStrategy.TACTICAL:
                    self.assertTrue(len(summary.successful_patterns) > 0)
                    # Tactical strategy focuses on patterns and discoveries

    def test_summarization_preserves_critical_discoveries(self):
        """Test that summarization preserves critical strategic discoveries."""
        summary = self.strategist.summarize_learnings(
            self.test_conversation, strategy=self.SummarizationStrategy.COMPREHENSIVE
        )

        # Validate critical discoveries are captured
        self.assertGreater(len(summary.critical_discoveries), 0)

        # Check that specific discovered patterns are preserved
        discoveries_text = " ".join(summary.critical_discoveries)
        self.assertIn("menu", discoveries_text.lower())
        self.assertIn("optimization", discoveries_text.lower())

    def test_summarization_preserves_strategic_insights(self):
        """Test that summarization preserves strategic insights and planning."""
        summary = self.strategist.summarize_learnings(
            self.test_conversation, strategy=self.SummarizationStrategy.STRATEGIC
        )

        # Validate strategic insights are captured
        self.assertGreater(len(summary.preserved_insights), 0)

        # Check that strategic content is preserved
        insights_text = " ".join(summary.preserved_insights)
        self.assertIn("strategic", insights_text.lower())

    def test_summarization_preserves_current_objectives(self):
        """Test that summarization preserves current objectives and goals."""
        summary = self.strategist.summarize_learnings(
            self.test_conversation, strategy=self.SummarizationStrategy.COMPREHENSIVE
        )

        # Validate objectives are captured
        self.assertGreater(len(summary.current_objectives), 0)

        # Check that objectives are preserved
        objectives_text = " ".join(summary.current_objectives)
        self.assertTrue(
            any(
                keyword in objectives_text.lower()
                for keyword in ["objective", "goal", "reach", "pewter"]
            )
        )

    def test_summarization_performance_target(self):
        """Test that summarization meets <100ms performance target."""
        # Create larger conversation for performance testing
        large_conversation = []
        for i in range(50):
            large_conversation.extend(
                [
                    {
                        "role": "user",
                        "content": f"Message {i}: Strategic question about route optimization and pattern discovery in speedrun context.",
                        "timestamp": time.time() - i,
                    },
                    {
                        "role": "assistant",
                        "content": f"Response {i}: Strategic analysis with discovered patterns and tactical insights for performance optimization. Key finding: pattern_{i} shows promise for speedrun improvement.",
                        "timestamp": time.time() - i + 0.5,
                    },
                ]
            )

        # Measure summarization performance
        start_time = time.time()
        summary = self.strategist.summarize_learnings(
            large_conversation, strategy=self.SummarizationStrategy.COMPREHENSIVE
        )
        processing_time = time.time() - start_time

        # Validate performance target: <100ms
        self.assertLess(
            processing_time * 1000,
            100,
            f"Summarization took {processing_time*1000:.2f}ms, target is <100ms",
        )

        # Validate processing was successful
        self.assertIsNotNone(summary)
        self.assertGreater(summary.compression_ratio, 0.5)  # Significant compression

    def test_summarization_size_constraints(self):
        """Test that summarization respects configurable size constraints."""
        max_lengths = [200, 500, 1000, 1500]

        for max_length in max_lengths:
            with self.subTest(max_length=max_length):
                summary = self.strategist.summarize_learnings(
                    self.test_conversation,
                    max_summary_length=max_length,
                    strategy=self.SummarizationStrategy.COMPREHENSIVE,
                )

                # Validate size constraint is respected
                self.assertLessEqual(
                    len(summary.compressed_content),
                    max_length,
                    f"Summary length {len(summary.compressed_content)} exceeds max {max_length}",
                )

                # Validate content is still meaningful
                self.assertGreater(len(summary.compressed_content), 0)

    def test_different_summarization_strategies(self):
        """Test that different strategies produce appropriate focus."""
        strategies_results = {}

        for strategy in self.SummarizationStrategy:
            summary = self.strategist.summarize_learnings(self.test_conversation, strategy=strategy)
            strategies_results[strategy] = summary

        # Validate strategy-specific behavior
        comprehensive = strategies_results[self.SummarizationStrategy.COMPREHENSIVE]
        strategic = strategies_results[self.SummarizationStrategy.STRATEGIC]
        tactical = strategies_results[self.SummarizationStrategy.TACTICAL]
        minimal = strategies_results[self.SummarizationStrategy.MINIMAL]

        # Comprehensive should have the most preserved information
        self.assertGreaterEqual(
            len(comprehensive.preserved_insights) + len(comprehensive.critical_discoveries),
            len(strategic.preserved_insights) + len(strategic.critical_discoveries),
        )

        # Strategic should focus on strategic insights
        self.assertGreater(len(strategic.preserved_insights), 0)

        # Tactical should focus on patterns
        self.assertGreater(len(tactical.successful_patterns), 0)

        # Minimal should have the most aggressive compression
        self.assertLessEqual(len(minimal.compressed_content), len(comprehensive.compressed_content))

    def test_summarization_with_empty_conversation(self):
        """Test summarization handles edge case of empty conversation history."""
        empty_conversation = []

        summary = self.strategist.summarize_learnings(
            empty_conversation, strategy=self.SummarizationStrategy.COMPREHENSIVE
        )

        # Validate graceful handling of empty input
        self.assertEqual(summary.total_messages, 0)
        self.assertGreater(len(summary.compressed_content), 0)  # Should have header info
        self.assertEqual(len(summary.preserved_insights), 0)
        self.assertEqual(len(summary.critical_discoveries), 0)

    def test_summary_immutability(self):
        """Test that ConversationSummary objects are truly immutable."""
        summary = self.strategist.summarize_learnings(
            self.test_conversation, strategy=self.SummarizationStrategy.COMPREHENSIVE
        )

        # Attempt to modify frozen dataclass should raise AttributeError
        with self.assertRaises(AttributeError):
            summary.total_messages = 999  # type: ignore

        with self.assertRaises(AttributeError):
            summary.compressed_content = "modified"  # type: ignore

    def test_summary_serialization(self):
        """Test that ConversationSummary can be serialized and deserialized."""
        from claudelearnspokemon.opus_strategist import ConversationSummary

        original_summary = self.strategist.summarize_learnings(
            self.test_conversation, strategy=self.SummarizationStrategy.COMPREHENSIVE
        )

        # Test to_dict conversion
        summary_dict = original_summary.to_dict()
        self.assertIsInstance(summary_dict, dict)
        self.assertEqual(summary_dict["total_messages"], original_summary.total_messages)
        self.assertEqual(summary_dict["strategy"], original_summary.strategy.value)

        # Test from_dict reconstruction
        reconstructed_summary = ConversationSummary.from_dict(summary_dict)
        self.assertEqual(reconstructed_summary.summary_id, original_summary.summary_id)
        self.assertEqual(reconstructed_summary.strategy, original_summary.strategy)
        self.assertEqual(reconstructed_summary.total_messages, original_summary.total_messages)
        self.assertEqual(
            reconstructed_summary.compressed_content, original_summary.compressed_content
        )


if __name__ == "__main__":
    # Run with performance timing
    print("=== OpusStrategist Test Suite - Performance Focus ===")
    unittest.main(verbosity=2)
