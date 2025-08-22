#!/usr/bin/env python3
"""
Test script to demonstrate memory integration functionality.

Shows how predictive planning components store optimization patterns
and prediction accuracy results in the memory system.
"""

import asyncio
from datetime import datetime

from src.claudelearnspokemon.predictive_planning import (
    BayesianPredictor,
    ContingencyGenerator,
    ExecutionPattern,
    ExecutionPatternAnalyzer,
    PredictionCache,
)


async def test_memory_integration():
    """Demonstrate memory integration with predictive planning components."""
    print("🧠 Testing Memory Integration for Predictive Planning Components")
    print("=" * 70)

    # Test BayesianPredictor memory storage
    print("\n📊 Testing BayesianPredictor Memory Storage...")

    predictor = BayesianPredictor()

    # Simulate a successful prediction result
    experiment_id = "test_experiment_001"

    # Store a prediction accuracy result
    predicted_outcome = {
        "success_probability": 0.85,
        "estimated_execution_time_ms": 150.0,
        "expected_performance_score": 0.75,
    }

    actual_outcome = {
        "success_rate": 1.0,  # Successful experiment
        "actual_execution_time_ms": 140.0,
        "actual_performance_score": 0.78,
    }

    accuracy_score = 0.92

    memory_id = await predictor.store_prediction_accuracy(
        experiment_id, predicted_outcome, actual_outcome, accuracy_score
    )

    if memory_id:
        print(f"  ✅ Stored prediction accuracy in memory: {memory_id}")
    else:
        print("  ⚠️  Memory storage simulated (no actual MCP connection)")

    # Test ExecutionPatternAnalyzer memory storage
    print("\n🔍 Testing ExecutionPatternAnalyzer Memory Storage...")

    analyzer = ExecutionPatternAnalyzer()

    # Create a successful execution pattern
    pattern = ExecutionPattern(
        pattern_id="pattern_001",
        features={
            "experiment_count": 0.4,
            "avg_priority": 0.8,
            "cmd_move_freq": 0.3,
            "cmd_battle_freq": 0.2,
        },
        success_rate=0.85,
        avg_execution_time=120.0,
        frequency=5,
        last_seen=datetime.utcnow(),
    )

    pattern_data = {
        "pattern_id": pattern.pattern_id,
        "features": pattern.features,
        "success_rate": pattern.success_rate,
        "avg_execution_time": pattern.avg_execution_time,
        "frequency": pattern.frequency,
    }

    performance_metrics = {
        "success_rate": pattern.success_rate,
        "avg_execution_time_ms": pattern.avg_execution_time,
        "frequency": pattern.frequency,
    }

    memory_id = await analyzer.store_optimization_pattern(
        "execution_pattern",
        pattern_data,
        performance_metrics,
        pattern.success_rate,
    )

    if memory_id:
        print(f"  ✅ Stored execution pattern in memory: {memory_id}")
    else:
        print("  ⚠️  Memory storage simulated (no actual MCP connection)")

    # Test PredictionCache memory storage
    print("\n⚡ Testing PredictionCache Memory Storage...")

    cache = PredictionCache()

    cache_metrics = {
        "avg_retrieval_time_ms": 3.5,
        "p90_retrieval_time_ms": 7.2,
        "hit_rate": 0.85,
        "cache_size": 150,
    }

    memory_id = await cache.store_performance_insight(
        "cache_optimization",
        "Cache achieving sub-10ms performance with 85% hit rate",
        cache_metrics,
        "Maintain current cache configuration for optimal performance",
    )

    if memory_id:
        print(f"  ✅ Stored cache optimization insight in memory: {memory_id}")
    else:
        print("  ⚠️  Memory storage simulated (no actual MCP connection)")

    # Test ContingencyGenerator memory storage
    print("\n🛡️  Testing ContingencyGenerator Memory Storage...")

    generator = ContingencyGenerator()

    strategy_data = {
        "contingency_types": ["execution_failure", "performance_degradation"],
        "priorities": [3, 2],
        "activation_thresholds": [0.7, 0.5],
    }

    contingency_metrics = {
        "generation_time_ms": 35.0,
        "contingencies_generated": 4,
        "scenarios_identified": 2,
        "avg_activation_probability": 0.6,
    }

    memory_id = await generator.store_optimization_pattern(
        "contingency_strategy",
        strategy_data,
        contingency_metrics,
        0.8,
    )

    if memory_id:
        print(f"  ✅ Stored contingency strategy in memory: {memory_id}")
    else:
        print("  ⚠️  Memory storage simulated (no actual MCP connection)")

    print("\n🎯 Memory Integration Test Complete!")
    print("=" * 70)
    print("All predictive planning components now store optimization patterns")
    print("and performance insights in the memory system for continuous learning.")


if __name__ == "__main__":
    asyncio.run(test_memory_integration())
