#!/usr/bin/env python3
"""Quick validation that ResponseCache fix works correctly."""

import sys
import os
import time
import threading

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from claudelearnspokemon.strategy_response_cache import ResponseCache
    from claudelearnspokemon.strategy_response import StrategyResponse
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def main():
    print("=== ResponseCache Thread Leak Fix Validation ===")
    
    # Test 1: Basic functionality
    print("\n1. Testing basic functionality...")
    cache = ResponseCache(max_size=5, default_ttl=0.5)
    
    response = StrategyResponse(
        strategy_id="test",
        experiments=[],
        strategic_insights=["test"],
        next_checkpoints=["test"]
    )
    
    cache.put("test_key", response)
    retrieved = cache.get("test_key")
    
    if retrieved and retrieved.strategy_id == "test":
        print("✅ Basic put/get functionality works")
    else:
        print("❌ Basic functionality failed")
        return False
    
    # Test 2: Thread cleanup
    print("\n2. Testing thread cleanup...")
    initial_threads = threading.active_count()
    
    # Create multiple caches
    caches = []
    for i in range(3):
        c = ResponseCache(max_size=5, default_ttl=0.1)
        caches.append(c)
    
    time.sleep(0.1)  # Let threads start
    peak_threads = threading.active_count()
    
    # Shutdown all caches
    for c in caches:
        c.shutdown()
    
    del caches
    time.sleep(0.2)  # Let cleanup complete
    
    final_threads = threading.active_count()
    
    print(f"Initial threads: {initial_threads}")
    print(f"Peak threads: {peak_threads}")
    print(f"Final threads: {final_threads}")
    
    if final_threads == initial_threads:
        print("✅ Thread cleanup works correctly")
    else:
        print(f"❌ Thread leak detected: {final_threads - initial_threads} threads")
        return False
    
    # Test 3: Shutdown idempotency
    print("\n3. Testing shutdown idempotency...")
    cache = ResponseCache(max_size=3, default_ttl=0.1)
    time.sleep(0.1)
    
    try:
        cache.shutdown()
        cache.shutdown()  # Should not raise error
        cache.shutdown()  # Should not raise error
        print("✅ Multiple shutdown() calls work safely")
    except Exception as e:
        print(f"❌ Shutdown idempotency failed: {e}")
        return False
    
    print("\n✅ All validation tests passed!")
    print("ResponseCache thread leak fix is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)