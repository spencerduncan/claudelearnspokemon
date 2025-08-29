#!/usr/bin/env python3
"""
Redis Cache Implementation Usage Example

This example demonstrates how to use the production-ready RedisCache
implementation with all its advanced features including:

- Connection pooling for high performance
- Circuit breaker pattern for automatic failover
- Fallback to InMemoryCache when Redis is unavailable
- JSON serialization with optional compression
- Health monitoring and performance metrics
- Thread-safe operations with proper error handling

Author: Claude Code (Innovator) - Production-First Engineering
"""

import logging
import time
from threading import Thread

from src.claudelearnspokemon.compatibility.cache_strategies import (
    RedisCache,
    create_cache_strategy,
)

# Configure logging to see Redis cache operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def basic_redis_cache_usage():
    """Demonstrate basic Redis cache usage with fallback."""
    print("=== Basic Redis Cache Usage ===")
    
    # Create Redis cache with fallback enabled (recommended for production)
    cache = RedisCache(
        redis_url="redis://localhost:6379",
        default_ttl_seconds=300,
        fallback_enabled=True,  # Falls back to InMemoryCache when Redis unavailable
        max_connections=10,
        health_check_interval=30.0,
    )
    
    # Test basic operations
    test_data = {
        "strategy": "aggressive_gym_battle",
        "pokemon": ["Charizard", "Blastoise", "Venusaur"],
        "success_rate": 0.95,
        "timestamp": time.time(),
    }
    
    print(f"Cache health status: {cache.is_healthy()}")
    
    # Set data
    result = cache.set("battle_strategy_1", test_data)
    print(f"Set operation result: {result}")
    
    # Get data
    retrieved = cache.get("battle_strategy_1")
    print(f"Retrieved data: {retrieved}")
    
    # Check if data matches
    print(f"Data integrity check: {retrieved == test_data}")
    
    # Get cache statistics
    stats = cache.get_stats()
    print(f"Cache statistics: {stats}")
    
    # Clear specific key
    cache.clear("battle_strategy_1")
    print("Cleared specific key")
    
    # Verify key is gone
    print(f"Key after clear: {cache.get('battle_strategy_1')}")


def redis_cache_with_compression():
    """Demonstrate Redis cache with compression for large data."""
    print("\n=== Redis Cache with Compression ===")
    
    cache = RedisCache(
        redis_url="redis://localhost:6379",
        fallback_enabled=True,
        compress_values=True,  # Enable compression for large values
    )
    
    # Create large data that would benefit from compression
    large_strategy = {
        "name": "Complete Speedrun Strategy",
        "route": "x" * 2000,  # Large route description
        "detailed_steps": [f"Step {i}: Detailed action description here" for i in range(100)],
        "metadata": {
            "complexity": "high",
            "estimated_time": 7200,
            "success_probability": 0.87,
        }
    }
    
    print(f"Large data size (estimated): {len(str(large_strategy))} characters")
    
    # Store large data (will be compressed automatically)
    start_time = time.time()
    cache.set("large_speedrun_strategy", large_strategy)
    set_time = time.time() - start_time
    
    print(f"Set large data in {set_time:.3f} seconds")
    
    # Retrieve and verify
    start_time = time.time()
    retrieved_large = cache.get("large_speedrun_strategy")
    get_time = time.time() - start_time
    
    print(f"Retrieved large data in {get_time:.3f} seconds")
    print(f"Data integrity check: {retrieved_large == large_strategy}")


def redis_cache_resilience_demo():
    """Demonstrate Redis cache resilience features."""
    print("\n=== Redis Cache Resilience Demo ===")
    
    # Create cache that will fail to connect to Redis (invalid port)
    cache = RedisCache(
        redis_url="redis://localhost:9999",  # Invalid Redis instance
        fallback_enabled=True,
        health_check_interval=5.0,
    )
    
    print(f"Initial health status: {cache.is_healthy()}")
    
    # Operations will automatically use fallback cache
    test_data = {"resilience": "test", "fallback": True}
    
    cache.set("resilience_test", test_data)
    retrieved = cache.get("resilience_test")
    
    print(f"Retrieved with fallback: {retrieved}")
    
    # Check statistics to see fallback usage
    stats = cache.get_stats()
    print(f"Fallback operations: {stats['fallback_operations']}")
    print(f"Circuit breaker state: {stats['circuit_breaker_state']}")
    print(f"Cache status: {stats['status']}")


def cache_factory_usage():
    """Demonstrate using the cache factory for environment-aware setup."""
    print("\n=== Cache Factory Usage ===")
    
    # Auto-detect cache strategy based on environment
    cache_auto = create_cache_strategy("auto")
    print(f"Auto-detected strategy: {cache_auto.__class__.__name__}")
    
    # Explicit Redis cache with custom parameters
    cache_redis = create_cache_strategy(
        "redis",
        redis_url="redis://localhost:6379",
        fallback_enabled=True,
        max_connections=20,
        compress_values=True,
    )
    print(f"Redis cache created: {cache_redis.__class__.__name__}")
    
    # Test both caches
    test_data = {"factory": "test", "timestamp": time.time()}
    
    for name, cache in [("auto", cache_auto), ("redis", cache_redis)]:
        cache.set(f"factory_test_{name}", test_data)
        retrieved = cache.get(f"factory_test_{name}")
        print(f"{name} cache test successful: {retrieved == test_data}")


def concurrent_cache_usage():
    """Demonstrate thread-safe concurrent cache usage."""
    print("\n=== Concurrent Cache Usage ===")
    
    cache = RedisCache(
        redis_url="redis://localhost:6379",
        fallback_enabled=True,
        max_connections=20,  # Higher connection pool for concurrency
    )
    
    def worker(worker_id: int, operations: int = 50):
        """Worker function for concurrent cache operations."""
        for i in range(operations):
            key = f"worker_{worker_id}_item_{i}"
            data = {
                "worker_id": worker_id,
                "operation": i,
                "timestamp": time.time(),
            }
            
            # Set data
            cache.set(key, data)
            
            # Get data back
            retrieved = cache.get(key)
            
            # Verify integrity
            if retrieved != data:
                print(f"Data integrity issue in worker {worker_id}, operation {i}")
            
            # Clear data
            cache.clear(key)
    
    # Start multiple worker threads
    threads = []
    num_workers = 5
    
    print(f"Starting {num_workers} concurrent workers...")
    start_time = time.time()
    
    for worker_id in range(num_workers):
        thread = Thread(target=worker, args=(worker_id, 20))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    elapsed_time = time.time() - start_time
    total_operations = num_workers * 20 * 3  # set, get, clear per operation
    
    print(f"Completed {total_operations} operations in {elapsed_time:.2f} seconds")
    print(f"Operations per second: {total_operations / elapsed_time:.0f}")
    
    # Check final statistics
    stats = cache.get_stats()
    print(f"Final cache statistics:")
    print(f"  Total operations: {stats['total_operations']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit ratio: {stats['hit_ratio']:.3f}")
    print(f"  Errors: {stats['errors']}")


def health_monitoring_demo():
    """Demonstrate health monitoring and metrics collection."""
    print("\n=== Health Monitoring Demo ===")
    
    cache = RedisCache(
        redis_url="redis://localhost:6379",
        fallback_enabled=True,
        health_check_interval=1.0,  # Check health every second
    )
    
    print("Monitoring cache health over 5 seconds...")
    
    for i in range(5):
        health_status = cache.is_healthy()
        stats = cache.get_stats()
        
        print(f"Second {i+1}:")
        print(f"  Healthy: {health_status}")
        print(f"  Status: {stats['status']}")
        print(f"  Last health check: {stats['last_health_check']}")
        
        # Perform some operations to generate metrics
        cache.set(f"health_test_{i}", {"check": i})
        cache.get(f"health_test_{i}")
        
        time.sleep(1.0)
    
    # Get connection pool statistics (if available)
    pool_stats = cache.get_connection_pool_stats()
    print(f"Connection pool stats: {pool_stats}")


if __name__ == "__main__":
    print("Redis Cache Implementation Demo")
    print("=" * 50)
    
    try:
        basic_redis_cache_usage()
        redis_cache_with_compression()
        redis_cache_resilience_demo()
        cache_factory_usage()
        concurrent_cache_usage()
        health_monitoring_demo()
        
        print("\n=== Demo Complete ===")
        print("All Redis cache features demonstrated successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Error during demo: {e}")