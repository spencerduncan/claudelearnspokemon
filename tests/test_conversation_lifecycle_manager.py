"""
Unit tests for ConversationLifecycleManager usage statistics and optimization features.

This test suite validates the ConversationLifecycleManager functionality including
usage analytics, optimization recommendations, and performance requirements.
Following the established testing patterns from the existing codebase.
"""

import time
import unittest
from unittest.mock import Mock, patch

import pytest

from claudelearnspokemon.conversation_lifecycle_manager import (
    ConversationLifecycleManager,
    ConversationUsageMetrics,
    UsageOptimizationLevel,
    UsageOptimizationRecommendation,
)
from claudelearnspokemon.conversation_state import (
    ConversationState,
    ConversationStateManager,
    ConversationStatus,
)
from claudelearnspokemon.process_metrics_collector import AggregatedMetricsCollector
from claudelearnspokemon.prompts import ProcessType


@pytest.mark.fast
class TestConversationLifecycleManagerInitialization(unittest.TestCase):
    """Test ConversationLifecycleManager initialization and basic functionality."""

    def setUp(self):
        """Set up test dependencies."""
        self.conversation_manager = ConversationStateManager()
        self.metrics_aggregator = AggregatedMetricsCollector()
        self.lifecycle_manager = ConversationLifecycleManager(
            self.conversation_manager, self.metrics_aggregator
        )

    def test_lifecycle_manager_initialization(self):
        """Test ConversationLifecycleManager initializes correctly with dependencies."""
        self.assertIs(self.lifecycle_manager.conversation_manager, self.conversation_manager)
        self.assertIs(self.lifecycle_manager.metrics_aggregator, self.metrics_aggregator)
        self.assertEqual(self.lifecycle_manager._peak_concurrent, 0)
        self.assertEqual(self.lifecycle_manager._total_completed, 0)
        self.assertEqual(self.lifecycle_manager._total_failed, 0)
        self.assertIsInstance(self.lifecycle_manager._optimization_cache, dict)
        self.assertEqual(self.lifecycle_manager._cache_ttl, 30.0)

    def test_session_tracking_initialization(self):
        """Test session tracking variables are initialized correctly."""
        current_time = time.time()
        # Allow small time difference for test execution
        self.assertAlmostEqual(
            self.lifecycle_manager._session_start_time, current_time, delta=1.0
        )
        self.assertEqual(len(self.lifecycle_manager._optimization_cache), 0)
        self.assertEqual(self.lifecycle_manager._cache_timestamp, 0.0)


@pytest.mark.fast
class TestConversationUsageStatistics(unittest.TestCase):
    """Test usage statistics calculation and caching functionality."""

    def setUp(self):
        """Set up test environment with mock conversations."""
        self.conversation_manager = ConversationStateManager()
        self.metrics_aggregator = AggregatedMetricsCollector()
        self.lifecycle_manager = ConversationLifecycleManager(
            self.conversation_manager, self.metrics_aggregator
        )

        # Create mock conversation states for testing
        self.tactical_state1 = ConversationState(ProcessType.SONNET_TACTICAL, 1)
        self.tactical_state2 = ConversationState(ProcessType.SONNET_TACTICAL, 2)
        self.strategic_state = ConversationState(ProcessType.OPUS_STRATEGIC, 3)

        # Add to manager
        self.conversation_manager.add_conversation(self.tactical_state1)
        self.conversation_manager.add_conversation(self.tactical_state2)
        self.conversation_manager.add_conversation(self.strategic_state)

    def test_get_conversation_stats_basic_metrics(self):
        """Test basic conversation statistics calculation."""
        # Initialize conversations
        self.tactical_state1.initialize_conversation("Test prompt 1")
        self.tactical_state2.initialize_conversation("Test prompt 2")
        self.strategic_state.initialize_conversation("Test prompt 3")

        # Add some message exchanges
        self.tactical_state1.record_message_exchange("Message 1", "Response 1", 100.0)
        self.tactical_state1.record_message_exchange("Message 2", "Response 2", 150.0)
        self.tactical_state2.record_message_exchange("Message 1", "Response 1", 120.0)

        stats = self.lifecycle_manager.get_conversation_stats()

        self.assertIsInstance(stats, ConversationUsageMetrics)
        self.assertEqual(stats.total_conversations, 3)
        self.assertEqual(stats.active_conversations, 3)
        self.assertEqual(stats.total_turns_across_all, 3)
        self.assertAlmostEqual(stats.average_turns_per_conversation, 1.0, places=1)
        self.assertGreater(stats.average_response_time_ms, 0)
        self.assertGreater(stats.usage_efficiency_score, 0)

    def test_get_conversation_stats_performance_caching(self):
        """Test performance caching of conversation statistics (<10ms for cached results)."""
        # First call - should calculate fresh
        start_time = time.time()
        stats1 = self.lifecycle_manager.get_conversation_stats()
        first_call_time = (time.time() - start_time) * 1000

        # Second call - should use cache
        start_time = time.time()
        stats2 = self.lifecycle_manager.get_conversation_stats()
        cached_call_time = (time.time() - start_time) * 1000

        # Verify caching works
        self.assertIs(stats1, stats2)  # Should be same object from cache
        self.assertLess(cached_call_time, 10.0)  # <10ms target for cached results

        # Verify cache contains expected data
        self.assertIn('usage_metrics', self.lifecycle_manager._optimization_cache)
        self.assertGreater(self.lifecycle_manager._cache_timestamp, 0)

    def test_turn_distribution_analysis(self):
        """Test detailed turn distribution analysis functionality."""
        # Initialize ALL conversations to have predictable state
        self.tactical_state1.initialize_conversation("Test prompt 1")
        self.tactical_state2.initialize_conversation("Test prompt 2")
        self.strategic_state.initialize_conversation("Test prompt 3")

        # Create specific turn patterns
        # State 1: Low usage (2/20 = 10%)
        for i in range(2):
            self.tactical_state1.record_message_exchange(f"Message {i}", f"Response {i}")

        # State 2: Optimal usage (10/20 = 50%)
        for i in range(10):
            self.tactical_state2.record_message_exchange(f"Message {i}", f"Response {i}")

        # State 3: Low usage (0/100 = 0%) - strategic with no turns
        # (strategic_state has no additional turns, so 0% usage)

        stats = self.lifecycle_manager.get_conversation_stats()

        # Should have turn distribution data
        self.assertIsInstance(stats.turn_distribution, dict)
        expected_keys = ["low_usage_0_25", "optimal_25_75", "high_usage_75_100", "limit_reached"]
        for key in expected_keys:
            self.assertIn(key, stats.turn_distribution)

        # Verify distribution matches expected patterns
        # tactical_state1 (10% usage) and strategic_state (0% usage) both fall in low_usage
        self.assertEqual(stats.turn_distribution["low_usage_0_25"], 2)  # tactical_state1 + strategic_state
        self.assertEqual(stats.turn_distribution["optimal_25_75"], 1)   # tactical_state2


@pytest.mark.fast
class TestUsageOptimization(unittest.TestCase):
    """Test usage optimization analysis and recommendations."""

    def setUp(self):
        """Set up test environment for optimization testing."""
        self.conversation_manager = ConversationStateManager()
        self.metrics_aggregator = AggregatedMetricsCollector()
        self.lifecycle_manager = ConversationLifecycleManager(
            self.conversation_manager, self.metrics_aggregator
        )

    def test_optimization_opportunities_high_turns(self):
        """Test optimization opportunities for high average turns scenario."""
        # Create conversation with high turn usage
        high_turn_state = ConversationState(ProcessType.OPUS_STRATEGIC, 1)
        high_turn_state.initialize_conversation("Test prompt")
        
        # Add 80 turns (high usage)
        for i in range(80):
            high_turn_state.record_message_exchange(f"Message {i}", f"Response {i}", 100.0)
        
        self.conversation_manager.add_conversation(high_turn_state)

        stats = self.lifecycle_manager.get_conversation_stats()
        
        # Should identify high turn usage as optimization opportunity
        self.assertGreater(stats.average_turns_per_conversation, 50)
        self.assertGreater(len(stats.optimization_opportunities), 0)
        
        # Check for turn allocation optimization opportunity
        turn_optimization = any("turn" in opp.lower() for opp in stats.optimization_opportunities)
        self.assertTrue(turn_optimization)

    def test_optimization_opportunities_low_success_rate(self):
        """Test optimization opportunities for low success rate scenario."""
        # Create conversations with failures
        failed_state = ConversationState(ProcessType.SONNET_TACTICAL, 1)
        failed_state.initialize_conversation("Test prompt")
        
        # Add failed messages
        for i in range(3):
            failed_state.record_message_exchange(
                f"Message {i}", None, error=f"Error {i}"
            )
        
        self.conversation_manager.add_conversation(failed_state)

        stats = self.lifecycle_manager.get_conversation_stats()
        
        # Should identify low success rate
        self.assertLess(stats.success_rate, 0.9)
        self.assertGreater(len(stats.optimization_opportunities), 0)

    def test_get_usage_optimization_recommendations(self):
        """Test detailed optimization recommendations generation."""
        # Create suboptimal usage pattern
        suboptimal_state = ConversationState(ProcessType.OPUS_STRATEGIC, 1)
        suboptimal_state.initialize_conversation("Test prompt")
        
        # Add many turns to trigger turn allocation recommendation
        for i in range(60):
            suboptimal_state.record_message_exchange(f"Message {i}", f"Response {i}", 200.0)
        
        self.conversation_manager.add_conversation(suboptimal_state)

        recommendations = self.lifecycle_manager.get_usage_optimization_recommendations()
        
        self.assertIsInstance(recommendations, list)
        
        if recommendations:  # Only test if recommendations exist
            rec = recommendations[0]
            self.assertIsInstance(rec, UsageOptimizationRecommendation)
            self.assertIsInstance(rec.optimization_type, str)
            self.assertIsInstance(rec.description, str)
            self.assertIsInstance(rec.potential_savings_pct, float)
            self.assertIn(rec.implementation_effort, ["low", "medium", "high"])
            self.assertIsInstance(rec.impact_score, float)
            self.assertTrue(0.0 <= rec.impact_score <= 1.0)
            self.assertIsInstance(rec.specific_actions, list)


@pytest.mark.fast
class TestUsageEfficiencyCalculation(unittest.TestCase):
    """Test usage efficiency scoring and calculation logic."""

    def setUp(self):
        """Set up test environment for efficiency testing."""
        self.conversation_manager = ConversationStateManager()
        self.metrics_aggregator = AggregatedMetricsCollector()
        self.lifecycle_manager = ConversationLifecycleManager(
            self.conversation_manager, self.metrics_aggregator
        )

    def test_usage_efficiency_optimal_range(self):
        """Test usage efficiency calculation for conversations in optimal range."""
        # Create conversation in optimal turn utilization range (30-80%)
        optimal_state = ConversationState(ProcessType.SONNET_TACTICAL, 1)
        optimal_state.initialize_conversation("Test prompt")
        
        # Use 50% of turns (10/20) - should be optimal
        for i in range(10):
            optimal_state.record_message_exchange(f"Message {i}", f"Response {i}")
        
        self.conversation_manager.add_conversation(optimal_state)

        stats = self.lifecycle_manager.get_conversation_stats()
        
        # Optimal usage should have high efficiency score
        self.assertGreaterEqual(stats.usage_efficiency_score, 0.8)

    def test_usage_efficiency_suboptimal_patterns(self):
        """Test usage efficiency calculation for suboptimal patterns."""
        # Create under-utilized conversation (10% usage)
        under_state = ConversationState(ProcessType.SONNET_TACTICAL, 1)
        under_state.initialize_conversation("Test prompt")
        
        # Use only 10% of turns (2/20)
        for i in range(2):
            under_state.record_message_exchange(f"Message {i}", f"Response {i}")
        
        # Create over-utilized conversation (95% usage)
        over_state = ConversationState(ProcessType.SONNET_TACTICAL, 2)
        over_state.initialize_conversation("Test prompt")
        
        # Use 95% of turns (19/20)
        for i in range(19):
            over_state.record_message_exchange(f"Message {i}", f"Response {i}")
        
        self.conversation_manager.add_conversation(under_state)
        self.conversation_manager.add_conversation(over_state)

        stats = self.lifecycle_manager.get_conversation_stats()
        
        # Should have lower efficiency due to suboptimal patterns
        self.assertLess(stats.usage_efficiency_score, 0.8)


@pytest.mark.fast
class TestPerformanceRequirements(unittest.TestCase):
    """Test performance requirements and benchmarking."""

    def setUp(self):
        """Set up performance testing environment."""
        self.conversation_manager = ConversationStateManager()
        self.metrics_aggregator = AggregatedMetricsCollector()
        self.lifecycle_manager = ConversationLifecycleManager(
            self.conversation_manager, self.metrics_aggregator
        )

        # Create realistic conversation dataset
        for i in range(5):
            state = ConversationState(ProcessType.SONNET_TACTICAL, i)
            state.initialize_conversation(f"Test prompt {i}")
            # Add varying numbers of turns
            for j in range(i + 1):
                state.record_message_exchange(f"Message {j}", f"Response {j}", 100.0 + j * 10)
            self.conversation_manager.add_conversation(state)

    def test_conversation_stats_performance_requirement(self):
        """Test that conversation stats meet <100ms performance requirement."""
        # Clear cache to force fresh calculation
        self.lifecycle_manager._optimization_cache.clear()
        
        start_time = time.time()
        stats = self.lifecycle_manager.get_conversation_stats()
        calculation_time = (time.time() - start_time) * 1000
        
        # Should meet <100ms requirement for fresh calculation
        self.assertLess(calculation_time, 100.0)
        self.assertIsInstance(stats, ConversationUsageMetrics)

    def test_cached_stats_performance_requirement(self):
        """Test that cached stats meet <10ms performance requirement."""
        # First call to populate cache
        self.lifecycle_manager.get_conversation_stats()
        
        # Second call should be cached
        start_time = time.time()
        cached_stats = self.lifecycle_manager.get_conversation_stats()
        cached_time = (time.time() - start_time) * 1000
        
        # Should meet <10ms requirement for cached results
        self.assertLess(cached_time, 10.0)
        self.assertIsInstance(cached_stats, ConversationUsageMetrics)


@pytest.mark.fast
class TestSessionManagement(unittest.TestCase):
    """Test session summary and tracking functionality."""

    def setUp(self):
        """Set up session management testing."""
        self.conversation_manager = ConversationStateManager()
        self.metrics_aggregator = AggregatedMetricsCollector()
        self.lifecycle_manager = ConversationLifecycleManager(
            self.conversation_manager, self.metrics_aggregator
        )

    def test_get_session_summary(self):
        """Test comprehensive session summary generation."""
        # Create test conversation
        test_state = ConversationState(ProcessType.SONNET_TACTICAL, 1)
        test_state.initialize_conversation("Test prompt")
        test_state.record_message_exchange("Test message", "Test response", 100.0)
        self.conversation_manager.add_conversation(test_state)

        summary = self.lifecycle_manager.get_session_summary()

        # Verify summary structure
        expected_keys = [
            "session_duration_minutes",
            "usage_metrics", 
            "optimization_recommendations",
            "efficiency_assessment"
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)

        # Verify efficiency assessment structure
        efficiency = summary["efficiency_assessment"]
        self.assertIn("overall_score", efficiency)
        self.assertIn("turn_utilization", efficiency)
        self.assertIn("performance_targets_met", efficiency)

        # Verify performance targets structure
        targets = efficiency["performance_targets_met"]
        expected_targets = ["success_rate_90pct", "response_time_1s", "usage_efficiency_70pct"]
        for target in expected_targets:
            self.assertIn(target, targets)
            self.assertIsInstance(targets[target], bool)

    def test_reset_session_tracking(self):
        """Test session tracking reset functionality."""
        # Initialize some session data
        original_start_time = self.lifecycle_manager._session_start_time
        self.lifecycle_manager._peak_concurrent = 5
        self.lifecycle_manager._optimization_cache = {"test": "data"}
        self.lifecycle_manager._cache_timestamp = time.time()

        # Reset session
        time.sleep(0.01)  # Small delay to ensure time difference
        self.lifecycle_manager.reset_session_tracking()

        # Verify reset occurred
        self.assertGreater(self.lifecycle_manager._session_start_time, original_start_time)
        self.assertEqual(self.lifecycle_manager._peak_concurrent, 0)
        self.assertEqual(self.lifecycle_manager._total_completed, 0)
        self.assertEqual(self.lifecycle_manager._total_failed, 0)
        self.assertEqual(len(self.lifecycle_manager._optimization_cache), 0)
        self.assertEqual(self.lifecycle_manager._cache_timestamp, 0.0)


if __name__ == "__main__":
    # Run with verbose output to see individual test results
    unittest.main(verbosity=2)