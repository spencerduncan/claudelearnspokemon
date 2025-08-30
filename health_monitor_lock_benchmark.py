#!/usr/bin/env python3
"""
Baseline performance benchmark for HealthMonitor lock optimization.
Measures lock acquisition times before and after RLock -> Lock conversion.
"""

import statistics
import threading
import time
from unittest.mock import Mock

from src.claudelearnspokemon.health_monitor import HealthMonitor


def create_test_monitor():
    """Create HealthMonitor for testing."""
    mock_pool = Mock()
    mock_pool.clients_by_port = {
        8081: Mock(port=8081, container_id="container1-abc123"),
        8082: Mock(port=8082, container_id="container2-def456"),
    }
    mock_pool.get_status.return_value = {
        "available_count": 2,
        "busy_count": 0,
        "total_count": 2,
        "status": "healthy",
    }
    
    return HealthMonitor(
        emulator_pool=mock_pool,
        check_interval=60.0,  # Long interval to avoid timer interference
        health_timeout=1.0,
    )


def benchmark_lock_acquisition(monitor, iterations=1000):
    """Benchmark lock acquisition times."""
    times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        with monitor._lock:
            # Simulate minimal work inside lock
            pass
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1_000_000)  # Convert to microseconds
    
    return times


def benchmark_concurrent_access(monitor, threads=4, operations_per_thread=250):
    """Benchmark concurrent lock access patterns."""
    results = []
    
    def worker():
        thread_times = []
        for _ in range(operations_per_thread):
            start_time = time.perf_counter()
            with monitor._lock:
                # Simulate realistic work: read some state
                _ = monitor._running
                _ = monitor._check_count
            end_time = time.perf_counter()
            thread_times.append((end_time - start_time) * 1_000_000)  # microseconds
        results.append(thread_times)
    
    threads_list = []
    for _ in range(threads):
        thread = threading.Thread(target=worker)
        threads_list.append(thread)
    
    # Start all threads
    start_time = time.perf_counter()
    for thread in threads_list:
        thread.start()
    
    # Wait for completion
    for thread in threads_list:
        thread.join()
    total_time = time.perf_counter() - start_time
    
    # Flatten results
    all_times = []
    for thread_times in results:
        all_times.extend(thread_times)
    
    return all_times, total_time


def benchmark_stats_access(monitor, iterations=1000):
    """Benchmark get_stats() method that uses lock."""
    times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = monitor.get_stats()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1_000_000)  # microseconds
    
    return times


def main():
    print("🔒 HealthMonitor Lock Performance Baseline")
    print("=" * 50)
    
    monitor = create_test_monitor()
    
    # Benchmark 1: Basic lock acquisition
    print("\n📊 BASIC LOCK ACQUISITION (1000 iterations)")
    print("-" * 40)
    
    lock_times = benchmark_lock_acquisition(monitor)
    avg_lock_time = statistics.mean(lock_times)
    lock_stdev = statistics.stdev(lock_times)
    
    print(f"Average lock time: {avg_lock_time:.3f}μs")
    print(f"Standard deviation: {lock_stdev:.3f}μs") 
    print(f"95th percentile: {sorted(lock_times)[int(0.95 * len(lock_times))]:.3f}μs")
    print(f"99th percentile: {sorted(lock_times)[int(0.99 * len(lock_times))]:.3f}μs")
    
    # Benchmark 2: Concurrent access
    print("\n📊 CONCURRENT ACCESS (4 threads, 250 ops each)")
    print("-" * 40)
    
    concurrent_times, total_time = benchmark_concurrent_access(monitor)
    avg_concurrent_time = statistics.mean(concurrent_times)
    concurrent_stdev = statistics.stdev(concurrent_times)
    
    print(f"Average concurrent lock time: {avg_concurrent_time:.3f}μs")
    print(f"Standard deviation: {concurrent_stdev:.3f}μs")
    print(f"Total execution time: {total_time:.3f}s")
    print(f"Throughput: {len(concurrent_times) / total_time:.0f} ops/sec")
    
    # Benchmark 3: get_stats() method
    print("\n📊 GET_STATS() METHOD (1000 iterations)")
    print("-" * 40)
    
    stats_times = benchmark_stats_access(monitor)
    avg_stats_time = statistics.mean(stats_times)
    stats_stdev = statistics.stdev(stats_times)
    
    print(f"Average stats access time: {avg_stats_time:.3f}μs")
    print(f"Standard deviation: {stats_stdev:.3f}μs")
    print(f"95th percentile: {sorted(stats_times)[int(0.95 * len(stats_times))]:.3f}μs")
    
    # Store baseline metrics
    baseline_metrics = {
        "basic_lock_avg": avg_lock_time,
        "basic_lock_stdev": lock_stdev,
        "concurrent_avg": avg_concurrent_time,
        "concurrent_stdev": concurrent_stdev,
        "concurrent_throughput": len(concurrent_times) / total_time,
        "stats_avg": avg_stats_time,
        "stats_stdev": stats_stdev,
    }
    
    print("\n🎯 BASELINE METRICS SUMMARY")
    print("-" * 30)
    for key, value in baseline_metrics.items():
        if 'throughput' in key:
            print(f"{key}: {value:.0f} ops/sec")
        else:
            print(f"{key}: {value:.3f}μs")
    
    print("\n✅ Baseline established!")
    print("   • Target: 10-20% improvement after optimization")
    print("   • Focus: Concurrent access throughput")
    print("   • Acceptable: <5% degradation tolerance")
    
    return baseline_metrics


if __name__ == "__main__":
    main()