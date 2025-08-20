#!/usr/bin/env python3
"""
Performance benchmark for OpusStrategist game state processing.

Validates that game state processing meets the <50ms performance target.
"""

import sys
import time
from pathlib import Path
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claudelearnspokemon.opus_strategist import OpusStrategist


def run_performance_benchmark():
    """Run performance benchmark for game state processing."""
    print("=== OpusStrategist Game State Processing Performance Benchmark ===")

    # Create mock strategist
    mock_manager = Mock()
    mock_manager.get_strategic_process.return_value = Mock()
    strategist = OpusStrategist(mock_manager)

    # Test performance with realistic game state
    game_state = {
        "tiles": [[i % 256 for i in range(18)] for _ in range(20)],
        "position": {"x": 10, "y": 5},
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
        },
        {
            "success": True,
            "execution_time": 0.9,
            "discovered_patterns": ["speed_optimization"],
            "final_state": {"x": 11, "y": 7},
        },
    ]

    # Performance test - multiple runs
    print("Running performance test with 100 iterations...")
    times = []

    for _ in range(100):
        start_time = time.perf_counter()
        strategist.format_game_state_for_context(
            game_state, execution_results, "strategic_analysis"
        )
        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000
        times.append(processing_time)

    # Calculate statistics
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    p50_time = sorted(times)[len(times) // 2]
    p95_time = sorted(times)[int(0.95 * len(times))]
    p99_time = sorted(times)[int(0.99 * len(times))]

    # Performance results
    print("\n📊 Performance Results (100 runs):")
    print(f"Average: {avg_time:.2f}ms")
    print(f"Minimum: {min_time:.2f}ms")
    print(f"Maximum: {max_time:.2f}ms")
    print(f"P50 (Median): {p50_time:.2f}ms")
    print(f"P95: {p95_time:.2f}ms")
    print(f"P99: {p99_time:.2f}ms")
    print("\n🎯 Target: <50ms")

    # Validation
    if max_time < 50:
        print("✅ PASS: All executions under 50ms target")
        success = True
    else:
        print(f"❌ FAIL: Max time {max_time:.2f}ms exceeds 50ms target")
        success = False

    if p95_time < 50:
        print(f"✅ P95 under target: {p95_time:.2f}ms")
    else:
        print(f"⚠️  P95 exceeds target: {p95_time:.2f}ms")

    # Load test with larger data
    print("\n🔧 Load Test - Complex Game State:")
    large_game_state = {
        "tiles": [[i % 256 for i in range(18)] for _ in range(20)],
        "position": {"x": 10, "y": 5},
        "map_id": "complex_dungeon_level_5_with_long_name",
        "inventory": {f"item_{i}": i for i in range(20)},  # Large inventory
        "health": 150,
        "level": 45,
        "pokemon_count": 6,
    }

    large_execution_results = [
        {
            "success": i % 3 != 0,
            "execution_time": 1.0 + (i * 0.1),
            "discovered_patterns": [f"pattern_{i}", f"pattern_{i+1}"],
            "final_state": {"x": 10 + i, "y": 5 + i},
        }
        for i in range(10)
    ]

    load_times = []
    for _ in range(20):
        start_time = time.perf_counter()
        strategist.format_game_state_for_context(large_game_state, large_execution_results)
        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000
        load_times.append(processing_time)

    load_avg = sum(load_times) / len(load_times)
    load_max = max(load_times)

    print(f"Load test average: {load_avg:.2f}ms")
    print(f"Load test maximum: {load_max:.2f}ms")

    if load_max < 100:  # More lenient for complex data
        print("✅ Load test PASS: Under 100ms for complex data")
    else:
        print(f"⚠️  Load test exceeds 100ms: {load_max:.2f}ms")

    return success


if __name__ == "__main__":
    success = run_performance_benchmark()
    sys.exit(0 if success else 1)
