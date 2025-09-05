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
    
    def test_toctou_vulnerability_protection(self):
        """Test TOCTOU protection in cleanup deletion loop."""
        import threading
        cache = ResponseCache(max_size=50, default_ttl=0.1, cleanup_interval=0.05)
        time.sleep(0.1)  # Let background thread start
        
        results = {"toctou_errors": 0, "operations": 0}
        
        def aggressive_operations():
            """Aggressively add/remove entries to create TOCTOU scenarios."""
            try:
                for i in range(200):
                    key = f"toctou_{i % 10}"
                    response = StrategyResponse(
                        strategy_id="toctou_test",
                        experiments=[],
                        strategic_insights=["toctou_insight"], 
                        next_checkpoints=["toctou_checkpoint"]
                    )
                    cache.put(key, response, ttl=0.05)  # Very short TTL
                    
                    # Immediately try operations that could conflict with cleanup
                    cache.get(key)
                    cache.invalidate(key)  # This removes keys that cleanup might try to remove
                    results["operations"] += 3
                    
                    if i % 20 == 0:
                        time.sleep(0.001)  # Brief pause to let cleanup run
                        
            except KeyError as e:
                # This is what we're testing against - should not happen with TOCTOU fix
                results["toctou_errors"] += 1
                print(f"TOCTOU KeyError detected: {e}")
            except Exception as e:
                print(f"Other error: {e}")
        
        # Run multiple threads to create contention
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(aggressive_operations) for _ in range(10)]
            time.sleep(1.5)  # Let them run and create contention scenarios
            concurrent.futures.wait(futures)
        
        # With TOCTOU protection, we should have 0 KeyErrors
        assert results["toctou_errors"] == 0, f"TOCTOU vulnerabilities detected: {results['toctou_errors']}"
        assert results["operations"] > 1000, "Insufficient TOCTOU test operations"
        assert cache.get_stats()["cleanup_runs"] > 5, "Cleanup should run multiple times during test"