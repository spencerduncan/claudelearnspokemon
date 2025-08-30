#!/usr/bin/env python3
"""
Baseline performance benchmark for CircuitBreaker lock optimization.
Measures lock acquisition times before RLock -> Lock conversion.
"""

import statistics
import threading
import time

from src.claudelearnspokemon.circuit_breaker import CircuitBreaker, CircuitConfig


def create_test_circuit_breaker():
    """Create CircuitBreaker for testing."""
    config = CircuitConfig(
        failure_threshold=5,
        recovery_timeout=10.0,
        success_threshold=2,
        sliding_window_size=10,
    )
    
    return CircuitBreaker(name="test-circuit", config=config)


def benchmark_lock_acquisition(breaker, iterations=1000):
    """Benchmark basic lock acquisition times."""
    times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        with breaker._lock:
            # Simulate minimal work inside lock
            _ = breaker._state
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1_000_000)  # Convert to microseconds
    
    return times


def benchmark_state_operations(breaker, iterations=1000):
    """Benchmark state read operations."""
    times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = breaker.get_state()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1_000_000)
    
    return times


def benchmark_metrics_operations(breaker, iterations=1000):
    """Benchmark metrics read operations."""
    times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = breaker.get_metrics()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1_000_000)
    
    return times


def benchmark_health_status_operations(breaker, iterations=1000):
    """Benchmark health status operations (tests reentrancy)."""
    times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = breaker.get_health_status()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1_000_000)
    
    return times


def benchmark_concurrent_access(breaker, threads=4, operations_per_thread=250):
    """Benchmark concurrent circuit breaker operations."""
    results = []
    
    def worker():
        thread_times = []
        for i in range(operations_per_thread):
            start_time = time.perf_counter()
            
            # Mix of different operations to simulate real usage
            if i % 4 == 0:
                _ = breaker.get_state()
            elif i % 4 == 1:
                _ = breaker.get_metrics()
            elif i % 4 == 2:
                _ = breaker.is_available()
            else:
                _ = breaker.get_health_status()  # Tests reentrancy
                
            end_time = time.perf_counter()
            thread_times.append((end_time - start_time) * 1_000_000)
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


def main():
    print("🔒 CircuitBreaker Lock Performance Baseline")
    print("=" * 50)
    
    breaker = create_test_circuit_breaker()
    
    # Benchmark 1: Basic lock acquisition
    print("\n📊 BASIC LOCK ACQUISITION (1000 iterations)")
    print("-" * 40)
    
    lock_times = benchmark_lock_acquisition(breaker)
    avg_lock_time = statistics.mean(lock_times)
    lock_stdev = statistics.stdev(lock_times)
    
    print(f"Average lock time: {avg_lock_time:.3f}μs")
    print(f"Standard deviation: {lock_stdev:.3f}μs") 
    print(f"95th percentile: {sorted(lock_times)[int(0.95 * len(lock_times))]:.3f}μs")
    print(f"99th percentile: {sorted(lock_times)[int(0.99 * len(lock_times))]:.3f}μs")
    
    # Benchmark 2: State operations
    print("\n📊 STATE OPERATIONS (1000 iterations)")
    print("-" * 40)
    
    state_times = benchmark_state_operations(breaker)
    avg_state_time = statistics.mean(state_times)
    state_stdev = statistics.stdev(state_times)
    
    print(f"Average state access time: {avg_state_time:.3f}μs")
    print(f"Standard deviation: {state_stdev:.3f}μs")
    
    # Benchmark 3: Metrics operations
    print("\n📊 METRICS OPERATIONS (1000 iterations)")
    print("-" * 40)
    
    metrics_times = benchmark_metrics_operations(breaker)
    avg_metrics_time = statistics.mean(metrics_times)
    metrics_stdev = statistics.stdev(metrics_times)
    
    print(f"Average metrics access time: {avg_metrics_time:.3f}μs")
    print(f"Standard deviation: {metrics_stdev:.3f}μs")
    
    # Benchmark 4: Health status operations (reentrancy test)
    print("\n📊 HEALTH STATUS OPERATIONS (1000 iterations)")
    print("-" * 40)
    
    health_times = benchmark_health_status_operations(breaker)
    avg_health_time = statistics.mean(health_times)
    health_stdev = statistics.stdev(health_times)
    
    print(f"Average health status time: {avg_health_time:.3f}μs")
    print(f"Standard deviation: {health_stdev:.3f}μs")
    
    # Benchmark 5: Concurrent access
    print("\n📊 CONCURRENT ACCESS (4 threads, 250 ops each)")
    print("-" * 40)
    
    concurrent_times, total_time = benchmark_concurrent_access(breaker)
    avg_concurrent_time = statistics.mean(concurrent_times)
    concurrent_stdev = statistics.stdev(concurrent_times)
    
    print(f"Average concurrent operation time: {avg_concurrent_time:.3f}μs")
    print(f"Standard deviation: {concurrent_stdev:.3f}μs")
    print(f"Total execution time: {total_time:.3f}s")
    print(f"Throughput: {len(concurrent_times) / total_time:.0f} ops/sec")
    
    # Store baseline metrics
    baseline_metrics = {
        "basic_lock_avg": avg_lock_time,
        "basic_lock_stdev": lock_stdev,
        "state_avg": avg_state_time,
        "state_stdev": state_stdev,
        "metrics_avg": avg_metrics_time,
        "metrics_stdev": metrics_stdev,
        "health_avg": avg_health_time,
        "health_stdev": health_stdev,
        "concurrent_avg": avg_concurrent_time,
        "concurrent_stdev": concurrent_stdev,
        "concurrent_throughput": len(concurrent_times) / total_time,
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
    print("   • Focus: Concurrent access throughput and health status reentrancy")
    print("   • Acceptable: <5% degradation tolerance")
    
    return baseline_metrics


if __name__ == "__main__":
    main()