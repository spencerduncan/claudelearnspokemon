"""
ConversationLifecycleManager - Usage Statistics and Lifecycle Orchestration

This module provides usage analytics and lifecycle coordination for Claude CLI
conversations, following Clean Code principles. It integrates conversation state
management with performance metrics to provide actionable insights for subscription
optimization and resource planning.

Performance Targets:
- Usage statistics queries: <10ms
- Turn analytics aggregation: <100ms
- Memory efficient usage history tracking
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .conversation_state import ConversationStateManager
from .process_metrics_collector import AggregatedMetricsCollector

logger = logging.getLogger(__name__)


class UsageOptimizationLevel(Enum):
    """Optimization recommendations based on usage patterns."""

    OPTIMAL = "optimal"
    MODERATE_OPTIMIZATION = "moderate_optimization"
    SIGNIFICANT_OPTIMIZATION = "significant_optimization"
    CRITICAL_OPTIMIZATION = "critical_optimization"


@dataclass
class ConversationUsageMetrics:
    """Usage metrics for conversation analytics and optimization."""

    total_conversations: int = 0
    total_turns_across_all: int = 0
    average_turns_per_conversation: float = 0.0
    active_conversations: int = 0
    completed_conversations: int = 0
    failed_conversations: int = 0
    conversations_approaching_limit: int = 0

    # Performance metrics
    average_response_time_ms: float = 0.0
    total_conversation_time_ms: float = 0.0
    success_rate: float = 0.0

    # Usage patterns
    peak_concurrent_conversations: int = 0
    turn_distribution: dict[str, int] = field(default_factory=dict)
    usage_efficiency_score: float = 0.0

    # Optimization insights
    estimated_cost_savings_pct: float = 0.0
    optimization_opportunities: list[str] = field(default_factory=list)
    recommended_turn_allocation: dict[str, int] = field(default_factory=dict)


@dataclass
class UsageOptimizationRecommendation:
    """Actionable recommendation for usage optimization."""

    optimization_type: str
    description: str
    potential_savings_pct: float
    implementation_effort: str  # "low", "medium", "high"
    impact_score: float  # 0.0 to 1.0
    specific_actions: list[str]


class ConversationLifecycleManager:
    """
    Orchestrates conversation lifecycle and provides usage analytics.

    This manager integrates ConversationStateManager with performance metrics
    to provide actionable insights for Claude subscription optimization,
    following the same clean architecture patterns as ClaudeCodeManager.

    Key Responsibilities:
    - Aggregate conversation usage statistics across all processes
    - Provide usage pattern analysis and optimization recommendations
    - Track conversation lifecycle efficiency metrics
    - Generate actionable insights for resource allocation
    """

    def __init__(
        self,
        conversation_state_manager: ConversationStateManager,
        metrics_aggregator: AggregatedMetricsCollector,
    ):
        """
        Initialize lifecycle manager with injected dependencies.

        Args:
            conversation_state_manager: Manager for conversation states
            metrics_aggregator: System-wide metrics aggregation
        """
        self.conversation_manager = conversation_state_manager
        self.metrics_aggregator = metrics_aggregator
        self._lock = threading.Lock()

        # Usage tracking
        self._session_start_time = time.time()
        self._peak_concurrent = 0
        self._total_completed = 0
        self._total_failed = 0
        self._optimization_cache: dict[str, Any] = {}
        self._cache_timestamp = 0.0
        self._cache_ttl = 30.0  # 30 seconds cache TTL for performance

        logger.info("ConversationLifecycleManager initialized")

    def get_conversation_stats(self) -> ConversationUsageMetrics:
        """
        Get comprehensive conversation usage statistics.

        Performance target: <10ms for cached results, <100ms for fresh calculation.

        Returns:
            ConversationUsageMetrics with current usage analytics
        """
        start_time = time.time()

        try:
            # Check cache for performance optimization
            if self._is_cache_valid():
                logger.debug("Returning cached conversation statistics")
                cached_metrics = self._optimization_cache.get("usage_metrics")
                if cached_metrics is not None:
                    return cached_metrics

            with self._lock:
                # Get conversation system summary
                conversation_summary = self.conversation_manager.get_system_summary()

                # Get process metrics for performance correlation
                system_metrics = self.metrics_aggregator.get_system_metrics()

                # Calculate comprehensive usage metrics
                metrics = self._calculate_usage_metrics(conversation_summary, system_metrics)

                # Add optimization recommendations
                metrics.optimization_opportunities = self._analyze_optimization_opportunities(
                    metrics
                )
                metrics.recommended_turn_allocation = self._calculate_optimal_turn_allocation(
                    metrics
                )

                # Cache results for performance
                self._cache_results(metrics)

                calculation_time = (time.time() - start_time) * 1000
                logger.debug(f"Conversation stats calculated in {calculation_time:.1f}ms")

                return metrics

        except Exception as e:
            logger.error(f"Error calculating conversation stats: {e}")
            # Return empty metrics on error
            return ConversationUsageMetrics()

    def _calculate_usage_metrics(
        self, conversation_summary: dict[str, Any], system_metrics: dict[str, Any]
    ) -> ConversationUsageMetrics:
        """Calculate detailed usage metrics from conversation and system data."""

        # Extract conversation data
        total_conversations = conversation_summary.get("total_conversations", 0)
        active_conversations = conversation_summary.get("active_conversations", 0)
        approaching_limit = conversation_summary.get("approaching_limit", 0)
        total_turns = conversation_summary.get("total_turns_across_all", 0)

        # Calculate averages safely
        avg_turns = total_turns / max(1, total_conversations)

        # Calculate success rate from individual conversation data
        conversations = conversation_summary.get("conversations", [])
        completed = sum(
            1 for conv in conversations if conv.get("status") in ["completed", "terminated"]
        )
        failed = sum(1 for conv in conversations if conv.get("status") == "failed")
        success_rate = completed / max(1, total_conversations)

        # Performance correlation
        avg_response_time = self._calculate_weighted_response_time(conversations)
        total_duration = sum(
            conv.get("metrics", {}).get("total_duration_ms", 0) for conv in conversations
        )

        # Usage efficiency calculation
        efficiency_score = self._calculate_usage_efficiency(conversations)

        # Turn distribution analysis
        turn_distribution = self._analyze_turn_distribution(conversations)

        # Track peak concurrent conversations
        current_concurrent = active_conversations + approaching_limit
        self._peak_concurrent = max(self._peak_concurrent, current_concurrent)

        return ConversationUsageMetrics(
            total_conversations=total_conversations,
            total_turns_across_all=total_turns,
            average_turns_per_conversation=avg_turns,
            active_conversations=active_conversations,
            completed_conversations=completed,
            failed_conversations=failed,
            conversations_approaching_limit=approaching_limit,
            average_response_time_ms=avg_response_time,
            total_conversation_time_ms=total_duration,
            success_rate=success_rate,
            peak_concurrent_conversations=self._peak_concurrent,
            turn_distribution=turn_distribution,
            usage_efficiency_score=efficiency_score,
        )

    def _analyze_optimization_opportunities(self, metrics: ConversationUsageMetrics) -> list[str]:
        """Analyze usage patterns to identify optimization opportunities."""
        opportunities = []

        # Turn allocation optimization
        if metrics.average_turns_per_conversation > 50:
            opportunities.append(
                f"High average turns ({metrics.average_turns_per_conversation:.1f}) - "
                "consider conversation context compression"
            )

        # Success rate optimization
        if metrics.success_rate < 0.9:
            opportunities.append(
                f"Success rate ({metrics.success_rate:.1%}) below optimal - "
                "review error handling and recovery patterns"
            )

        # Response time optimization
        if metrics.average_response_time_ms > 1000:
            opportunities.append(
                f"High response times ({metrics.average_response_time_ms:.0f}ms) - "
                "consider process optimization or resource scaling"
            )

        # Concurrent conversation optimization
        efficiency_threshold = 0.7
        if metrics.usage_efficiency_score < efficiency_threshold:
            opportunities.append(
                f"Usage efficiency ({metrics.usage_efficiency_score:.1%}) below target - "
                "optimize conversation lifecycle and turn allocation"
            )

        return opportunities

    def _calculate_optimal_turn_allocation(
        self, metrics: ConversationUsageMetrics
    ) -> dict[str, int]:
        """Calculate recommended turn allocation based on usage patterns."""

        # Base recommendations on observed patterns
        recommendations = {
            "opus_strategic": 80,  # Reduced from 100 if efficiency is low
            "sonnet_tactical": 15,  # Reduced from 20 if efficiency is low
        }

        # Adjust based on efficiency
        if metrics.usage_efficiency_score < 0.7:
            # More conservative allocation for low efficiency
            recommendations["opus_strategic"] = 60
            recommendations["sonnet_tactical"] = 12
        elif metrics.usage_efficiency_score > 0.9:
            # More generous allocation for high efficiency
            recommendations["opus_strategic"] = 100
            recommendations["sonnet_tactical"] = 20

        return recommendations

    def _calculate_weighted_response_time(self, conversations: list[dict[str, Any]]) -> float:
        """Calculate turn-weighted average response time across conversations."""
        total_weighted_time = 0.0
        total_turns = 0

        for conv in conversations:
            metrics = conv.get("metrics", {})
            turns = metrics.get("total_turns", 0)
            avg_response = metrics.get("average_response_time_ms", 0)

            if turns > 0:
                total_weighted_time += avg_response * turns
                total_turns += turns

        return total_weighted_time / max(1, total_turns)

    def _calculate_usage_efficiency(self, conversations: list[dict[str, Any]]) -> float:
        """Calculate overall usage efficiency score based on multiple factors."""
        if not conversations:
            return 0.0

        efficiency_factors = []

        for conv in conversations:
            # Factor 1: Turn utilization (avoid both under and over-utilization)
            turns_used = conv.get("turn_count", 0)
            max_turns = conv.get("max_turns", 1)
            turn_utilization = turns_used / max_turns

            # Optimal range is 0.3 to 0.8 (30% to 80% of limit)
            if 0.3 <= turn_utilization <= 0.8:
                turn_efficiency = 1.0
            elif turn_utilization < 0.3:
                turn_efficiency = turn_utilization / 0.3  # Penalize under-utilization
            else:
                turn_efficiency = (1.0 - turn_utilization) / 0.2  # Penalize over-utilization

            efficiency_factors.append(max(0.0, turn_efficiency))

        return sum(efficiency_factors) / len(efficiency_factors)

    def _analyze_turn_distribution(self, conversations: list[dict[str, Any]]) -> dict[str, int]:
        """Analyze distribution of turn usage across conversations."""
        distribution = {
            "low_usage_0_25": 0,  # 0-25% of limit
            "optimal_25_75": 0,  # 25-75% of limit
            "high_usage_75_100": 0,  # 75-100% of limit
            "limit_reached": 0,  # Exactly at limit
        }

        for conv in conversations:
            turns_used = conv.get("turn_count", 0)
            max_turns = conv.get("max_turns", 1)
            utilization = turns_used / max_turns

            if utilization >= 1.0:
                distribution["limit_reached"] += 1
            elif utilization >= 0.75:
                distribution["high_usage_75_100"] += 1
            elif utilization >= 0.25:
                distribution["optimal_25_75"] += 1
            else:
                distribution["low_usage_0_25"] += 1

        return distribution

    def _is_cache_valid(self) -> bool:
        """Check if cached results are still valid."""
        return (
            time.time() - self._cache_timestamp
        ) < self._cache_ttl and "usage_metrics" in self._optimization_cache

    def _cache_results(self, metrics: ConversationUsageMetrics):
        """Cache metrics results for performance optimization."""
        self._optimization_cache = {"usage_metrics": metrics}
        self._cache_timestamp = time.time()

    def get_usage_optimization_recommendations(self) -> list[UsageOptimizationRecommendation]:
        """
        Get detailed optimization recommendations based on usage patterns.

        Returns:
            List of actionable optimization recommendations
        """
        metrics = self.get_conversation_stats()
        recommendations = []

        # Turn allocation optimization
        if metrics.average_turns_per_conversation > 50:
            recommendations.append(
                UsageOptimizationRecommendation(
                    optimization_type="turn_allocation",
                    description="Reduce average turns per conversation through context optimization",
                    potential_savings_pct=15.0,
                    implementation_effort="medium",
                    impact_score=0.8,
                    specific_actions=[
                        "Implement context compression when approaching turn limits",
                        "Use conversation summarization for long interactions",
                        "Optimize prompt efficiency to reduce back-and-forth",
                    ],
                )
            )

        # Success rate optimization
        if metrics.success_rate < 0.9:
            recommendations.append(
                UsageOptimizationRecommendation(
                    optimization_type="error_reduction",
                    description="Improve conversation success rate through better error handling",
                    potential_savings_pct=10.0,
                    implementation_effort="low",
                    impact_score=0.9,
                    specific_actions=[
                        "Enhance error recovery mechanisms",
                        "Implement retry logic for transient failures",
                        "Add conversation health monitoring",
                    ],
                )
            )

        return recommendations

    def get_session_summary(self) -> dict[str, Any]:
        """
        Get comprehensive session summary for reporting.

        Returns:
            Dictionary with session-wide usage analytics and insights
        """
        metrics = self.get_conversation_stats()
        session_duration = time.time() - self._session_start_time

        return {
            "session_duration_minutes": session_duration / 60,
            "usage_metrics": metrics,
            "optimization_recommendations": self.get_usage_optimization_recommendations(),
            "efficiency_assessment": {
                "overall_score": metrics.usage_efficiency_score,
                "turn_utilization": metrics.turn_distribution,
                "performance_targets_met": {
                    "success_rate_90pct": metrics.success_rate >= 0.9,
                    "response_time_1s": metrics.average_response_time_ms <= 1000,
                    "usage_efficiency_70pct": metrics.usage_efficiency_score >= 0.7,
                },
            },
        }

    def reset_session_tracking(self):
        """Reset session-level tracking for fresh analytics."""
        with self._lock:
            self._session_start_time = time.time()
            self._peak_concurrent = 0
            self._total_completed = 0
            self._total_failed = 0
            self._optimization_cache.clear()
            self._cache_timestamp = 0.0

        logger.info("Session tracking reset for ConversationLifecycleManager")
