"""
Focused test for ResponseCache race condition fix (Issue #239).
Tests the specific dictionary iteration race condition that was causing 5,600+ errors.
"""

import concurrent.futures
import time
import pytest
from claudelearnspokemon.strategy_response_cache import ResponseCache
from claudelearnspokemon.strategy_response import StrategyResponse


def test_race_condition_fix():
    """Test that dictionary iteration race condition is resolved."""
    cache = ResponseCache(max_size=20, default_ttl=0.1, cleanup_interval=0.05)
    time.sleep(0.1)  # Let background thread start
    
    error_count = 0
    operation_count = 0
    
    def cache_operations():
        nonlocal error_count, operation_count
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
                operation_count += 1
                time.sleep(0.001)  # Brief pause to allow cleanup thread to run
        except RuntimeError as e:
            if "dictionary changed size during iteration" in str(e):
                error_count += 1
                print(f"Race condition detected: {e}")
        except Exception as e:
            print(f"Other error: {e}")
    
    # Run concurrent operations to stress test the cleanup method
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(cache_operations) for _ in range(15)]
        concurrent.futures.wait(futures)
    
    # Verify no race condition errors occurred
    assert error_count == 0, f"Race condition errors detected: {error_count}"
    assert operation_count > 500, f"Insufficient operations performed: {operation_count}"
    
    # Verify background cleanup is working
    stats = cache.get_stats()
    assert stats["cleanup_runs"] > 0, "Background cleanup should have run"
    
    print(f"✅ Race condition test passed: {operation_count} operations, 0 errors")


def test_toctou_protection():
    """Test TOCTOU (Time-of-check Time-of-use) protection in cleanup method."""
    cache = ResponseCache(max_size=10, default_ttl=0.05, cleanup_interval=0.02)
    time.sleep(0.05)  # Let background thread start
    
    toctou_errors = 0
    operations = 0
    
    def aggressive_operations():
        nonlocal toctou_errors, operations
        try:
            for i in range(100):
                key = f"toctou_{i % 20}"
                response = StrategyResponse(
                    strategy_id="toctou_test",
                    experiments=[],
                    strategic_insights=["toctou_insight"], 
                    next_checkpoints=["toctou_checkpoint"]
                )
                cache.put(key, response, ttl=0.05)  # Very short TTL
                cache.get(key)
                cache.invalidate(key)  # Remove keys that cleanup might try to remove
                operations += 3
                
                if i % 20 == 0:
                    time.sleep(0.01)  # Let cleanup run
                    
        except KeyError as e:
            # This is what TOCTOU protection prevents
            toctou_errors += 1
            print(f"TOCTOU error detected: {e}")
        except Exception as e:
            print(f"Other error: {e}")
    
    # Run multiple threads to create TOCTOU scenarios
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(aggressive_operations) for _ in range(10)]
        concurrent.futures.wait(futures)
    
    # With TOCTOU protection, we should have 0 KeyErrors
    assert toctou_errors == 0, f"TOCTOU errors detected: {toctou_errors}"
    assert operations > 1000, f"Insufficient operations: {operations}"
    
    print(f"✅ TOCTOU test passed: {operations} operations, 0 TOCTOU errors")


if __name__ == "__main__":
    print("Running focused ResponseCache race condition tests...")
    test_race_condition_fix()
    test_toctou_protection()
    print("🎉 All race condition tests passed!")