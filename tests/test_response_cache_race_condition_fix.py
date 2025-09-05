"""
Tests for ResponseCache race condition fix (Issue #239).
Validates dictionary iteration race condition is resolved.
"""

import concurrent.futures
import time
import pytest
from claudelearnspokemon.strategy_response_cache import ResponseCache
from claudelearnspokemon.strategy_response import StrategyResponse


class TestResponseCacheRaceConditionFix:
    """Test suite validating Issue #239 race condition fix."""

    def test_concurrent_operations_no_errors(self):
        """Test concurrent operations produce no race condition errors."""
        cache = ResponseCache(max_size=20, default_ttl=0.1, cleanup_interval=0.05)
        time.sleep(0.1)  # Let background thread start
        
        error_count = 0
        
        def cache_operations():
            nonlocal error_count
            try:
                for i in range(50):
                    response = StrategyResponse(
                        strategy_id="test_strategy",
                        experiments=[],
                        strategic_insights=["test_insight"],
                        next_checkpoints=["test_checkpoint"]
                    )
                    cache.put(f"key_{i}", response, ttl=0.1)
                    cache.get(f"key_{i}")
                    time.sleep(0.001)
            except Exception as e:
                error_count += 1
                print(f"Error: {e}")
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(cache_operations) for _ in range(15)]
            time.sleep(1.0)
            concurrent.futures.wait(futures)
        
        assert error_count == 0, f"Race condition errors detected: {error_count}"
        assert cache.get_stats()["cleanup_runs"] > 0, "Background cleanup should run"
    
    def test_extreme_stress_no_race_conditions(self):
        """Extreme stress test validates 0 race conditions under high load."""
        cache = ResponseCache(max_size=10, default_ttl=0.05, cleanup_interval=0.02)
        time.sleep(0.1)
        
        results = {"errors": 0, "operations": 0}
        
        def stress_operations():
            try:
                for i in range(100):
                    key = f"stress_{i % 20}"
                    response = StrategyResponse(
                        strategy_id="stress_test",
                        experiments=[],
                        strategic_insights=["stress_insight"], 
                        next_checkpoints=["stress_checkpoint"]
                    )
                    cache.put(key, response)
                    cache.get(key)
                    cache.invalidate(key)
                    results["operations"] += 3
            except Exception as e:
                results["errors"] += 1
                print(f"Stress error: {e}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(stress_operations) for _ in range(25)]
            time.sleep(2.0)
            concurrent.futures.wait(futures)
        
        assert results["errors"] == 0, f"Race conditions detected: {results['errors']}"
        assert results["operations"] > 1000, "Insufficient stress test operations"