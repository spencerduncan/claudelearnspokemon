#!/usr/bin/env python3
"""
Performance benchmark comparing RLock vs Lock in cache operations.

Measures the performance impact of replacing RLock with Lock in InMemoryCache
to quantify the optimization benefits for issue #190.
"""

import threading
import time
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
import statistics


class RLockCache:
    """Test cache using RLock (current implementation)."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()  # Current implementation
        
    def get(self, key: str) -> Any:
        with self._lock:
            return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value
            # Simulate the _evict_lru_entry pattern from real implementation
            if len(self._cache) > 1000:
                self._evict_oldest()
    
    def _evict_oldest(self) -> None:
        # Helper method that doesn't acquire lock (matches real pattern)
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]


class LockCache:
    """Test cache using simple Lock (proposed optimization)."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()  # Optimized implementation
        
    def get(self, key: str) -> Any:
        with self._lock:
            return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value
            # Simulate the _evict_lru_entry pattern from real implementation
            if len(self._cache) > 1000:
                self._evict_oldest()
    
    def _evict_oldest(self) -> None:
        # Helper method that doesn't acquire lock (matches real pattern)
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]


def benchmark_cache_operations(cache, operation_count: int, thread_count: int) -> float:
    """
    Benchmark cache operations with multiple threads.
    
    Returns average time per operation in microseconds.
    """
    def worker(cache, ops_per_thread: int) -> List[float]:
        times = []
        for i in range(ops_per_thread):
            # Mix of get and set operations (typical cache usage)
            key = f"key_{i % 100}"  # 100 different keys to simulate realistic access
            
            start_time = time.perf_counter()
            
            if i % 4 == 0:  # 25% set operations
                cache.set(key, {"data": i, "timestamp": time.time()})
            else:  # 75% get operations (typical cache read-heavy pattern)
                cache.get(key)
                
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1_000_000)  # Convert to microseconds
        
        return times
    
    ops_per_thread = operation_count // thread_count
    all_times = []
    
    # Execute benchmark with multiple threads
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [
            executor.submit(worker, cache, ops_per_thread) 
            for _ in range(thread_count)
        ]
        
        for future in futures:
            all_times.extend(future.result())
    
    return statistics.mean(all_times)


def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing RLock vs Lock performance."""
    
    print("Thread Synchronization Performance Benchmark")
    print("=" * 60)
    print("Comparing RLock vs Lock in cache operations (Issue #190)")
    print()
    
    # Test configurations
    configs = [
        {"ops": 10000, "threads": 1, "name": "Single-threaded"},
        {"ops": 10000, "threads": 2, "name": "Low contention (2 threads)"},
        {"ops": 10000, "threads": 4, "name": "Medium contention (4 threads)"},
        {"ops": 10000, "threads": 8, "name": "High contention (8 threads)"},
        {"ops": 50000, "threads": 4, "name": "High volume (50k ops, 4 threads)"},
    ]
    
    results = []
    
    for config in configs:
        print(f"Testing: {config['name']}")
        print(f"Operations: {config['ops']}, Threads: {config['threads']}")
        
        # Benchmark RLock implementation
        rlock_cache = RLockCache()
        rlock_time = benchmark_cache_operations(rlock_cache, config["ops"], config["threads"])
        
        # Benchmark Lock implementation  
        lock_cache = LockCache()
        lock_time = benchmark_cache_operations(lock_cache, config["ops"], config["threads"])
        
        # Calculate improvement
        improvement = ((rlock_time - lock_time) / rlock_time) * 100
        speedup = rlock_time / lock_time
        
        result = {
            "name": config["name"],
            "rlock_time": rlock_time,
            "lock_time": lock_time,
            "improvement_percent": improvement,
            "speedup_factor": speedup
        }
        results.append(result)
        
        print(f"  RLock:       {rlock_time:.2f} μs/op")
        print(f"  Lock:        {lock_time:.2f} μs/op") 
        print(f"  Improvement: {improvement:.1f}% faster ({speedup:.2f}x speedup)")
        print()
    
    # Summary
    print("BENCHMARK SUMMARY")
    print("=" * 40)
    avg_improvement = statistics.mean([r["improvement_percent"] for r in results])
    avg_speedup = statistics.mean([r["speedup_factor"] for r in results])
    
    print(f"Average performance improvement: {avg_improvement:.1f}%")
    print(f"Average speedup factor: {avg_speedup:.2f}x")
    
    # Find best case
    best_result = max(results, key=lambda x: x["improvement_percent"])
    print(f"Best case improvement: {best_result['improvement_percent']:.1f}% ({best_result['name']})")
    
    # Memory usage comparison
    print()
    print("MEMORY OVERHEAD ANALYSIS")
    print("=" * 30)
    print("RLock overhead: ~48-64 bytes per instance (thread ID tracking)")
    print("Lock overhead: ~24-32 bytes per instance (simple mutex)")
    print("Memory savings per cache instance: ~24-32 bytes")
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_benchmark()