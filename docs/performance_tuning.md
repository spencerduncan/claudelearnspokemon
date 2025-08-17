# Pokemon Gym Adapter Performance Tuning Guide

This guide provides comprehensive performance optimization strategies, benchmarking techniques, and tuning recommendations for the Pokemon Gym adapter system. Follow these guidelines to achieve optimal throughput and minimize latency in your Pokemon Red speedrun learning experiments.

## Table of Contents

1. [Performance Targets](#performance-targets)
2. [System Architecture for Performance](#system-architecture-for-performance)
3. [Container Optimization](#container-optimization)
4. [Script Compilation Optimization](#script-compilation-optimization)
5. [Checkpoint Performance](#checkpoint-performance)
6. [Network and I/O Optimization](#network-and-io-optimization)
7. [Memory Management](#memory-management)
8. [Parallel Execution Patterns](#parallel-execution-patterns)
9. [Benchmarking and Monitoring](#benchmarking-and-monitoring)
10. [Production Deployment](#production-deployment)

---

## Performance Targets

The Pokemon Gym adapter is designed to meet specific performance requirements for high-throughput experimentation:

### Core Performance Metrics

| Operation | Target | Measurement | Impact |
|-----------|---------|-------------|--------|
| Script compilation | < 100ms | Time to compile DSL to instructions | Affects experiment startup |
| Checkpoint saving | < 500ms | Time to compress and write state | Enables frequent checkpointing |
| Checkpoint loading | < 500ms | Time to read and decompress state | Affects experiment restart speed |
| Tile observation | < 50ms | Time to capture and analyze game state | Critical for real-time decisions |
| Full execution cycle | < 5s | End-to-end script execution | Overall experiment throughput |
| Emulator acquisition | < 1s | Time to get available emulator | Pool efficiency |

### Throughput Targets

- **Parallel Scripts**: 50+ scripts/minute across 4 emulators
- **Checkpoint Operations**: 120+ saves/minute with compression
- **State Analysis**: 1000+ tile observations/minute
- **Container Restarts**: < 30s recovery time

---

## System Architecture for Performance

### Hardware Recommendations

#### Development Environment
```
CPU: 4+ cores (Intel i5/AMD Ryzen 5 or better)
Memory: 8GB+ RAM
Storage: SSD with 10GB+ free space
Network: Local Docker (no network overhead)
```

#### Production Environment
```
CPU: 8+ cores (Intel i7/AMD Ryzen 7 or better)
Memory: 16GB+ RAM
Storage: NVMe SSD with 50GB+ free space
Network: High-speed local network for distributed setups
```

### Docker Host Optimization

```bash
# Configure Docker daemon for performance
sudo tee /etc/docker/daemon.json <<EOF
{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
EOF

sudo systemctl restart docker

# Optimize kernel parameters for containers
echo 'net.core.somaxconn = 1024' | sudo tee -a /etc/sysctl.conf
echo 'vm.max_map_count = 262144' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

---

## Container Optimization

### Optimal Container Configuration

```python
# High-performance EmulatorPool configuration
pool = EmulatorPool(
    pool_size=8,                    # Scale based on CPU cores
    base_port=8080,                 # Start from standard port
    image_name="pokemon-gym:latest",
    startup_timeout=60,             # Allow for slower startup
    default_timeout=10.0            # Fast failover for busy pool
)

# Container resource limits in _start_single_container()
container = self.client.containers.run(
    image=self.image_name,
    ports={"8080/tcp": port},
    detach=True,
    remove=True,
    name=f"pokemon-emulator-{port}",
    restart_policy={"Name": "unless-stopped"},  # More resilient
    mem_limit="1g",                             # Increased from 512m
    cpu_count=1,
    cpu_shares=1024,                           # Fair CPU scheduling
    shm_size="128m",                           # Shared memory for emulator
)
```

### Container Image Optimization

```dockerfile
# Optimized Dockerfile for pokemon-gym
FROM python:3.10-slim

# Use multi-stage build to reduce image size
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optimize Python bytecode
ENV PYTHONOPTIMIZE=1
ENV PYTHONDONTWRITEBYTECODE=1

# Use non-root user for security and performance
RUN useradd -m pokemon && chown -R pokemon:pokemon /app
USER pokemon

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080
CMD ["python", "app.py"]
```

### Container Pool Sizing Strategy

```python
import multiprocessing
import psutil

def optimal_pool_size():
    """Calculate optimal pool size based on system resources."""
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)

    # Conservative sizing: 1 emulator per 2 CPU cores, max 1 per 2GB RAM
    cpu_limit = max(1, cpu_count // 2)
    memory_limit = max(1, int(memory_gb // 2))

    # Never exceed practical limits
    optimal_size = min(cpu_limit, memory_limit, 12)  # Max 12 containers

    print(f"System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    print(f"Optimal pool size: {optimal_size}")

    return optimal_size

# Use optimal sizing
pool_size = optimal_pool_size()
pool = EmulatorPool(pool_size=pool_size)
```

---

## Script Compilation Optimization

### Pre-compilation Strategies

```python
class ScriptCache:
    """Cache compiled scripts to avoid repeated compilation."""

    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []

    def get_compiled_script(self, pool, script_text):
        """Get compiled script from cache or compile if needed."""
        script_hash = hash(script_text)

        if script_hash in self.cache:
            # Move to end of access order (LRU)
            self.access_order.remove(script_hash)
            self.access_order.append(script_hash)
            return self.cache[script_hash]

        # Compile new script
        compiled_script = pool.compile_script(script_text)

        # Add to cache
        self.cache[script_hash] = compiled_script
        self.access_order.append(script_hash)

        # Evict oldest if cache is full
        if len(self.cache) > self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        return compiled_script

# Usage
script_cache = ScriptCache()

# Compile once, use many times
common_scripts = [
    "A B START",
    "UP UP DOWN DOWN LEFT RIGHT LEFT RIGHT",
    "PRESS A WAIT 60 PRESS B"
]

for script_text in common_scripts:
    compiled = script_cache.get_compiled_script(pool, script_text)
```

### Batch Compilation

```python
def batch_compile_scripts(pool, script_texts):
    """Compile multiple scripts efficiently."""
    import concurrent.futures
    from functools import partial

    # Use ThreadPoolExecutor for I/O-bound compilation
    compile_func = partial(pool.compile_script)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        compiled_scripts = list(executor.map(compile_func, script_texts))

    return compiled_scripts

# Pre-compile experiment scripts
experiment_scripts = [
    "MOVE UP PRESS A",
    "MOVE DOWN PRESS A",
    "MOVE LEFT PRESS A",
    "MOVE RIGHT PRESS A"
]

compiled_experiments = batch_compile_scripts(pool, experiment_scripts)
```

---

## Checkpoint Performance

### Checkpoint Optimization Strategies

```python
class OptimizedCheckpointManager(CheckpointManager):
    """Enhanced CheckpointManager with performance optimizations."""

    def __init__(self, checkpoint_dir=None, compression_level=3):
        super().__init__(checkpoint_dir)
        self.compression_level = compression_level  # LZ4 compression level
        self._write_buffer_size = 1024 * 1024  # 1MB write buffer

    def save_checkpoint_optimized(self, game_state, metadata):
        """Optimized checkpoint saving with better compression."""
        import lz4.frame
        import json
        import time
        import uuid

        start_time = time.monotonic()

        checkpoint_id = str(uuid.uuid4())
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.lz4"

        # Prepare data with minimal serialization overhead
        checkpoint_data = {
            "version": "1.1",  # Updated version for optimized format
            "checkpoint_id": checkpoint_id,
            "timestamp": time.time(),
            "game_state": game_state,
            "metadata": metadata,
        }

        # Optimize JSON serialization
        json_data = json.dumps(
            checkpoint_data,
            separators=(',', ':'),  # No spaces for smaller size
            default=str  # Handle any non-serializable objects
        ).encode('utf-8')

        # Use optimized compression
        compressed_data = lz4.frame.compress(
            json_data,
            compression_level=self.compression_level,
            block_size=lz4.frame.BLOCKSIZE_MAX1MB,
            auto_flush=True
        )

        # Atomic write with buffering
        temp_file = checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'wb', buffering=self._write_buffer_size) as f:
            f.write(compressed_data)

        # Atomic rename
        temp_file.rename(checkpoint_file)

        save_time = time.monotonic() - start_time
        self._save_times.append(save_time)

        # Performance monitoring
        compression_ratio = len(json_data) / len(compressed_data)
        if save_time > 0.5:  # Log slow saves
            print(f"⚠️ Slow checkpoint save: {save_time:.3f}s, ratio: {compression_ratio:.2f}x")

        return checkpoint_id
```

### Checkpoint Caching Strategy

```python
class CheckpointCache:
    """LRU cache for frequently accessed checkpoints."""

    def __init__(self, max_size=50, max_memory_mb=500):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = {}
        self.access_order = []
        self.memory_usage = 0

    def get_checkpoint(self, checkpoint_mgr, checkpoint_id):
        """Get checkpoint from cache or load from disk."""
        if checkpoint_id in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(checkpoint_id)
            self.access_order.append(checkpoint_id)
            return self.cache[checkpoint_id]

        # Load from disk
        checkpoint_data = checkpoint_mgr.load_checkpoint(checkpoint_id)

        # Estimate memory usage
        estimated_size = len(str(checkpoint_data).encode('utf-8'))

        # Evict if necessary
        while (len(self.cache) >= self.max_size or
               self.memory_usage + estimated_size > self.max_memory_bytes):
            if not self.access_order:
                break

            oldest_id = self.access_order.pop(0)
            old_data = self.cache.pop(oldest_id)
            old_size = len(str(old_data).encode('utf-8'))
            self.memory_usage -= old_size

        # Add to cache
        self.cache[checkpoint_id] = checkpoint_data
        self.access_order.append(checkpoint_id)
        self.memory_usage += estimated_size

        return checkpoint_data

# Usage
checkpoint_cache = CheckpointCache(max_size=100)

def cached_execute_with_checkpoint(pool, script_text, checkpoint_id):
    """Execute script with cached checkpoint loading."""
    if checkpoint_id:
        checkpoint_data = checkpoint_cache.get_checkpoint(checkpoint_mgr, checkpoint_id)
        # Use cached data for execution

    return pool.execute_script(script_text, checkpoint_id)
```

---

## Network and I/O Optimization

### HTTP Client Optimization

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class OptimizedPokemonGymClient(PokemonGymClient):
    """Optimized client with connection pooling and retries."""

    def __init__(self, port, container_id):
        super().__init__(port, container_id)

        # Configure optimized session
        self.session = requests.Session()

        # Connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=Retry(
                total=3,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504]
            )
        )

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Keep-alive and other optimizations
        self.session.headers.update({
            'Connection': 'keep-alive',
            'User-Agent': 'PokemonGymClient/1.0'
        })

    def send_input_optimized(self, input_sequence):
        """Send input with optimized HTTP parameters."""
        payload = {"inputs": input_sequence}

        response = self.session.post(
            f"{self.base_url}/input",
            json=payload,
            timeout=(5, 30),  # (connection_timeout, read_timeout)
            stream=False  # Don't stream small responses
        )

        response.raise_for_status()
        return response.json()
```

### Asynchronous Operations

```python
import asyncio
import aiohttp
import time

class AsyncEmulatorClient:
    """Asynchronous client for high-throughput operations."""

    def __init__(self, ports):
        self.ports = ports
        self.connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=10,
            keepalive_timeout=30
        )
        self.session = aiohttp.ClientSession(connector=self.connector)

    async def send_input_async(self, port, input_sequence):
        """Send input asynchronously."""
        url = f"http://localhost:{port}/input"
        payload = {"inputs": input_sequence}

        async with self.session.post(url, json=payload, timeout=10) as response:
            return await response.json()

    async def execute_scripts_parallel(self, script_port_pairs):
        """Execute multiple scripts in parallel across emulators."""
        tasks = []
        for script, port in script_port_pairs:
            task = self.send_input_async(port, script)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def close(self):
        """Clean up resources."""
        await self.session.close()

# Usage
async def high_throughput_execution():
    """Example of high-throughput parallel execution."""
    client = AsyncEmulatorClient([8081, 8082, 8083, 8084])

    # Prepare many scripts
    scripts = ["A B", "UP DOWN", "LEFT RIGHT", "START SELECT"] * 25  # 100 scripts
    ports = [8081, 8082, 8083, 8084] * 25  # Cycle through emulators

    script_port_pairs = list(zip(scripts, ports))

    start_time = time.time()
    results = await client.execute_scripts_parallel(script_port_pairs)
    duration = time.time() - start_time

    successful_results = [r for r in results if not isinstance(r, Exception)]
    throughput = len(successful_results) / duration

    print(f"Executed {len(successful_results)} scripts in {duration:.2f}s")
    print(f"Throughput: {throughput:.1f} scripts/second")

    await client.close()

# Run async example
# asyncio.run(high_throughput_execution())
```

---

## Memory Management

### Memory-Efficient Data Structures

```python
import sys
from typing import Dict, Any
import gc

class MemoryEfficientExecutionResult:
    """Memory-optimized execution result using __slots__."""

    __slots__ = ['success', 'output', 'error', 'execution_time', 'checkpoint_reached']

    def __init__(self, success, output, error=None, execution_time=None, checkpoint_reached=None):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time
        self.checkpoint_reached = checkpoint_reached

    def memory_usage(self):
        """Calculate approximate memory usage."""
        total = sys.getsizeof(self)
        for attr in self.__slots__:
            value = getattr(self, attr)
            if value is not None:
                total += sys.getsizeof(value)
        return total

def monitor_memory_usage():
    """Monitor and report memory usage patterns."""
    import psutil
    import os

    process = psutil.Process(os.getpid())

    def get_memory_info():
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }

    return get_memory_info

# Memory monitoring decorator
def track_memory_usage(func):
    """Decorator to track memory usage of functions."""
    def wrapper(*args, **kwargs):
        monitor = monitor_memory_usage()

        mem_before = monitor()
        result = func(*args, **kwargs)
        mem_after = monitor()

        mem_diff = mem_after['rss_mb'] - mem_before['rss_mb']

        if mem_diff > 10:  # Log significant memory usage
            print(f"🧠 {func.__name__} used {mem_diff:.1f}MB")

        return result

    return wrapper

# Memory cleanup strategies
class MemoryManager:
    def __init__(self, cleanup_threshold_mb=100):
        self.cleanup_threshold_mb = cleanup_threshold_mb
        self.last_cleanup = time.time()

    def should_cleanup(self):
        """Determine if memory cleanup is needed."""
        current_memory = monitor_memory_usage()()['rss_mb']
        time_since_cleanup = time.time() - self.last_cleanup

        return (current_memory > self.cleanup_threshold_mb or
                time_since_cleanup > 300)  # 5 minutes

    def cleanup(self):
        """Perform memory cleanup."""
        if self.should_cleanup():
            # Force garbage collection
            collected = gc.collect()
            self.last_cleanup = time.time()

            print(f"🧹 Collected {collected} objects during cleanup")

            # Report memory after cleanup
            mem_after = monitor_memory_usage()()['rss_mb']
            print(f"🧠 Memory after cleanup: {mem_after:.1f}MB")
```

---

## Parallel Execution Patterns

### Optimal Parallelization Strategies

```python
import concurrent.futures
import queue
import threading
from typing import List, Callable

class HighPerformanceExecutor:
    """High-performance parallel execution with load balancing."""

    def __init__(self, pool, max_workers=None):
        self.pool = pool
        self.max_workers = max_workers or pool.pool_size
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

    def execute_scripts_load_balanced(self, scripts: List[str]) -> List[dict]:
        """Execute scripts with intelligent load balancing."""

        # Submit tasks to queue
        for i, script in enumerate(scripts):
            self.task_queue.put((i, script))

        # Worker function
        def worker():
            results = []
            while True:
                try:
                    task_id, script = self.task_queue.get(timeout=1)
                except queue.Empty:
                    break

                try:
                    # Use pool's execute_script for automatic client management
                    start_time = time.time()
                    result = self.pool.execute_script(script)
                    duration = time.time() - start_time

                    results.append({
                        'task_id': task_id,
                        'script': script,
                        'result': result,
                        'duration': duration
                    })

                except Exception as e:
                    results.append({
                        'task_id': task_id,
                        'script': script,
                        'result': None,
                        'error': str(e),
                        'duration': None
                    })
                finally:
                    self.task_queue.task_done()

            return results

        # Start workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            worker_futures = [executor.submit(worker) for _ in range(self.max_workers)]

            # Wait for completion
            all_results = []
            for future in concurrent.futures.as_completed(worker_futures):
                worker_results = future.result()
                all_results.extend(worker_results)

        # Sort by task_id to maintain order
        all_results.sort(key=lambda x: x['task_id'])

        return all_results

# Advanced parallel patterns
class ExperimentPipeline:
    """Pipeline for complex parallel experiment workflows."""

    def __init__(self, pool, checkpoint_mgr):
        self.pool = pool
        self.checkpoint_mgr = checkpoint_mgr

    def parallel_checkpoint_experiments(self, base_checkpoint_id, experiments):
        """Run multiple experiments in parallel from same checkpoint."""

        def run_single_experiment(experiment_data):
            experiment_id, script_text = experiment_data

            try:
                # Execute from checkpoint
                result = self.pool.execute_script(script_text, base_checkpoint_id)

                if result.success:
                    # Save result state as new checkpoint
                    with self.pool.acquire_emulator(timeout=10) as client:
                        final_state = client.get_state()

                    result_checkpoint = self.checkpoint_mgr.save_checkpoint(
                        game_state=final_state,
                        metadata={
                            'experiment_id': experiment_id,
                            'base_checkpoint': base_checkpoint_id,
                            'script': script_text
                        }
                    )

                    return {
                        'experiment_id': experiment_id,
                        'success': True,
                        'execution_time': result.execution_time,
                        'result_checkpoint': result_checkpoint
                    }
                else:
                    return {
                        'experiment_id': experiment_id,
                        'success': False,
                        'error': result.error
                    }

            except Exception as e:
                return {
                    'experiment_id': experiment_id,
                    'success': False,
                    'error': str(e)
                }

        # Execute experiments in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.pool.pool_size) as executor:
            experiment_data = [(i, script) for i, script in enumerate(experiments)]
            results = list(executor.map(run_single_experiment, experiment_data))

        return results
```

---

## Benchmarking and Monitoring

### Comprehensive Benchmarking Suite

```python
import statistics
import time
from contextlib import contextmanager

class PerformanceBenchmark:
    """Comprehensive performance benchmarking for the Pokemon Gym adapter."""

    def __init__(self, pool, checkpoint_mgr):
        self.pool = pool
        self.checkpoint_mgr = checkpoint_mgr
        self.results = {}

    @contextmanager
    def timer(self, operation_name):
        """Context manager for timing operations."""
        start_time = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - start_time
            if operation_name not in self.results:
                self.results[operation_name] = []
            self.results[operation_name].append(duration)

    def benchmark_script_compilation(self, iterations=100):
        """Benchmark script compilation performance."""
        test_scripts = [
            "A B START",
            "UP UP DOWN DOWN LEFT RIGHT LEFT RIGHT A B A B",
            "PRESS A WAIT 60 PRESS B WAIT 30 PRESS START"
        ]

        for script in test_scripts:
            script_name = f"compile_{len(script)}_chars"

            for _ in range(iterations):
                with self.timer(script_name):
                    compiled = self.pool.compile_script(script)

    def benchmark_checkpoint_operations(self, iterations=50):
        """Benchmark checkpoint save/load performance."""
        # Create test data of various sizes
        test_data_sets = [
            {"small": {"player": {"x": 1, "y": 2}}},
            {"medium": {"player": {"x": 1, "y": 2}, "items": ["potion"] * 100}},
            {"large": {"player": {"x": 1, "y": 2}, "items": ["potion"] * 1000, "map": [[0] * 100 for _ in range(100)]}
        ]

        for data_name, test_data in test_data_sets:
            checkpoint_ids = []

            # Benchmark saves
            for _ in range(iterations):
                with self.timer(f"checkpoint_save_{data_name}"):
                    checkpoint_id = self.checkpoint_mgr.save_checkpoint(
                        test_data, {"benchmark": True, "size": data_name}
                    )
                    checkpoint_ids.append(checkpoint_id)

            # Benchmark loads
            for checkpoint_id in checkpoint_ids[:10]:  # Test subset for loads
                with self.timer(f"checkpoint_load_{data_name}"):
                    loaded_data = self.checkpoint_mgr.load_checkpoint(checkpoint_id)

            # Clean up
            for checkpoint_id in checkpoint_ids:
                self.checkpoint_mgr.delete_checkpoint(checkpoint_id)

    def benchmark_parallel_execution(self, script_counts=[10, 50, 100]):
        """Benchmark parallel execution scalability."""
        test_script = "A B START"

        for count in script_counts:
            scripts = [test_script] * count

            # Sequential execution
            with self.timer(f"sequential_{count}_scripts"):
                for script in scripts[:min(count, 20)]:  # Limit sequential test
                    result = self.pool.execute_script(script)

            # Parallel execution
            with self.timer(f"parallel_{count}_scripts"):
                executor = HighPerformanceExecutor(self.pool)
                results = executor.execute_scripts_load_balanced(scripts)

    def generate_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*80)

        for operation, times in sorted(self.results.items()):
            if not times:
                continue

            avg_time = statistics.mean(times)
            median_time = statistics.median(times)
            min_time = min(times)
            max_time = max(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0

            print(f"\n📊 {operation.upper()}")
            print(f"   Samples: {len(times)}")
            print(f"   Average: {avg_time*1000:.1f}ms")
            print(f"   Median:  {median_time*1000:.1f}ms")
            print(f"   Min/Max: {min_time*1000:.1f}ms / {max_time*1000:.1f}ms")
            print(f"   Std Dev: {std_dev*1000:.1f}ms")

            # Performance target analysis
            if "compile" in operation:
                target = 0.1  # 100ms
                status = "✅" if avg_time < target else "❌"
                print(f"   Target:  {target*1000:.0f}ms {status}")
            elif "checkpoint" in operation:
                target = 0.5  # 500ms
                status = "✅" if avg_time < target else "❌"
                print(f"   Target:  {target*1000:.0f}ms {status}")

# Usage
def run_comprehensive_benchmark(pool, checkpoint_mgr):
    """Run full performance benchmark suite."""
    benchmark = PerformanceBenchmark(pool, checkpoint_mgr)

    print("🚀 Starting comprehensive performance benchmark...")
    print("This may take several minutes...")

    # Run all benchmarks
    benchmark.benchmark_script_compilation()
    benchmark.benchmark_checkpoint_operations()
    benchmark.benchmark_parallel_execution()

    # Generate report
    benchmark.generate_report()

    return benchmark.results
```

### Real-time Performance Monitoring

```python
import threading
import time
from collections import deque

class PerformanceMonitor:
    """Real-time performance monitoring for production systems."""

    def __init__(self, pool, update_interval=10):
        self.pool = pool
        self.update_interval = update_interval
        self.metrics = {
            'operations_per_second': deque(maxlen=60),  # 10 minutes of data
            'average_response_time': deque(maxlen=60),
            'pool_utilization': deque(maxlen=60),
            'error_rate': deque(maxlen=60)
        }
        self.monitoring = False
        self.monitor_thread = None

        # Operation counters
        self.operation_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.last_reset = time.time()

    def record_operation(self, duration, success=True):
        """Record an operation for monitoring."""
        self.operation_count += 1
        self.total_response_time += duration

        if not success:
            self.error_count += 1

    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("📊 Performance monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("📊 Performance monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # Calculate metrics for this interval
                now = time.time()
                interval = now - self.last_reset

                if interval > 0:
                    ops_per_second = self.operation_count / interval
                    avg_response = (self.total_response_time / self.operation_count
                                  if self.operation_count > 0 else 0)
                    error_rate = (self.error_count / self.operation_count
                                if self.operation_count > 0 else 0)

                    # Pool utilization
                    pool_status = self.pool.get_status()
                    utilization = pool_status['busy_count'] / pool_status['total_count']

                    # Store metrics
                    self.metrics['operations_per_second'].append(ops_per_second)
                    self.metrics['average_response_time'].append(avg_response * 1000)  # Convert to ms
                    self.metrics['pool_utilization'].append(utilization * 100)  # Convert to percentage
                    self.metrics['error_rate'].append(error_rate * 100)  # Convert to percentage

                # Reset counters
                self.operation_count = 0
                self.error_count = 0
                self.total_response_time = 0.0
                self.last_reset = now

                time.sleep(self.update_interval)

            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(1)

    def get_current_metrics(self):
        """Get current performance metrics."""
        if not self.metrics['operations_per_second']:
            return None

        return {
            'operations_per_second': self.metrics['operations_per_second'][-1],
            'average_response_time_ms': self.metrics['average_response_time'][-1],
            'pool_utilization_percent': self.metrics['pool_utilization'][-1],
            'error_rate_percent': self.metrics['error_rate'][-1]
        }

    def print_status(self):
        """Print current performance status."""
        metrics = self.get_current_metrics()
        if not metrics:
            print("📊 No metrics available yet")
            return

        print("\n📊 CURRENT PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Operations/sec:    {metrics['operations_per_second']:.1f}")
        print(f"Avg response time: {metrics['average_response_time_ms']:.1f}ms")
        print(f"Pool utilization:  {metrics['pool_utilization_percent']:.1f}%")
        print(f"Error rate:        {metrics['error_rate_percent']:.1f}%")

# Monitored execution wrapper
class MonitoredEmulatorPool:
    """EmulatorPool wrapper with built-in performance monitoring."""

    def __init__(self, pool):
        self.pool = pool
        self.monitor = PerformanceMonitor(pool)
        self.monitor.start_monitoring()

    def execute_script(self, script_text, checkpoint_id=None):
        """Execute script with performance monitoring."""
        start_time = time.time()

        try:
            result = self.pool.execute_script(script_text, checkpoint_id)
            duration = time.time() - start_time

            self.monitor.record_operation(duration, result.success)
            return result

        except Exception as e:
            duration = time.time() - start_time
            self.monitor.record_operation(duration, success=False)
            raise

    def get_performance_metrics(self):
        """Get current performance metrics."""
        return self.monitor.get_current_metrics()

    def print_performance_status(self):
        """Print current performance status."""
        self.monitor.print_status()

    def shutdown(self):
        """Shutdown with monitoring cleanup."""
        self.monitor.stop_monitoring()
        self.pool.shutdown()
```

---

## Production Deployment

### Production Configuration

```python
# production_config.py
import os
from dataclasses import dataclass

@dataclass
class ProductionConfig:
    # Pool configuration
    pool_size: int = int(os.getenv('POKEMON_POOL_SIZE', 8))
    base_port: int = int(os.getenv('POKEMON_BASE_PORT', 8080))
    startup_timeout: int = int(os.getenv('POKEMON_STARTUP_TIMEOUT', 120))

    # Performance tuning
    default_timeout: float = float(os.getenv('POKEMON_DEFAULT_TIMEOUT', 30.0))
    max_script_cache_size: int = int(os.getenv('POKEMON_SCRIPT_CACHE_SIZE', 1000))
    checkpoint_compression_level: int = int(os.getenv('POKEMON_COMPRESSION_LEVEL', 6))

    # Resource limits
    container_memory_limit: str = os.getenv('POKEMON_CONTAINER_MEMORY', '1g')
    container_cpu_limit: float = float(os.getenv('POKEMON_CONTAINER_CPU', 1.0))

    # Monitoring
    enable_monitoring: bool = os.getenv('POKEMON_MONITORING', 'true').lower() == 'true'
    monitoring_interval: int = int(os.getenv('POKEMON_MONITORING_INTERVAL', 10))

    # Paths
    checkpoint_dir: str = os.getenv('POKEMON_CHECKPOINT_DIR', '/var/lib/pokemon-checkpoints')
    log_level: str = os.getenv('POKEMON_LOG_LEVEL', 'INFO')

def create_production_pool(config: ProductionConfig):
    """Create optimally configured EmulatorPool for production."""

    # Set up logging
    import logging
    logging.basicConfig(level=getattr(logging, config.log_level))

    # Create optimized checkpoint manager
    checkpoint_mgr = OptimizedCheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        compression_level=config.checkpoint_compression_level
    )

    # Create pool with production settings
    pool = EmulatorPool(
        pool_size=config.pool_size,
        base_port=config.base_port,
        startup_timeout=config.startup_timeout,
        default_timeout=config.default_timeout,
        checkpoint_manager=checkpoint_mgr
    )

    # Wrap with monitoring if enabled
    if config.enable_monitoring:
        pool = MonitoredEmulatorPool(pool)

    return pool

# Usage
config = ProductionConfig()
production_pool = create_production_pool(config)
```

### Health Checks and Alerting

```python
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

class HealthMonitor:
    """Production health monitoring with alerting."""

    def __init__(self, pool, alert_email=None):
        self.pool = pool
        self.alert_email = alert_email
        self.last_alert = {}
        self.alert_cooldown = 300  # 5 minutes between alerts

    def check_system_health(self):
        """Comprehensive system health check."""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'issues': []
        }

        # Check pool health
        pool_health = self.pool.health_check()
        if pool_health['status'] != 'healthy':
            health_report['overall_status'] = 'degraded'
            health_report['issues'].append({
                'component': 'emulator_pool',
                'issue': f"Pool unhealthy: {pool_health['healthy_count']}/{pool_health['total_count']} containers"
            })

        # Check resource usage
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent

        if cpu_percent > 90:
            health_report['issues'].append({
                'component': 'system',
                'issue': f"High CPU usage: {cpu_percent}%"
            })

        if memory_percent > 90:
            health_report['issues'].append({
                'component': 'system',
                'issue': f"High memory usage: {memory_percent}%"
            })

        if disk_usage > 90:
            health_report['issues'].append({
                'component': 'system',
                'issue': f"High disk usage: {disk_usage}%"
            })

        # Check performance metrics if monitoring is enabled
        if hasattr(self.pool, 'get_performance_metrics'):
            metrics = self.pool.get_performance_metrics()
            if metrics:
                if metrics['error_rate_percent'] > 5:  # More than 5% errors
                    health_report['issues'].append({
                        'component': 'performance',
                        'issue': f"High error rate: {metrics['error_rate_percent']:.1f}%"
                    })

                if metrics['average_response_time_ms'] > 5000:  # More than 5 seconds
                    health_report['issues'].append({
                        'component': 'performance',
                        'issue': f"Slow response time: {metrics['average_response_time_ms']:.1f}ms"
                    })

        # Set overall status based on issues
        if health_report['issues']:
            health_report['overall_status'] = 'degraded' if len(health_report['issues']) < 3 else 'critical'

        # Send alerts if necessary
        if health_report['overall_status'] in ['degraded', 'critical']:
            self._send_alert(health_report)

        return health_report

    def _send_alert(self, health_report):
        """Send alert email if conditions are met."""
        if not self.alert_email:
            return

        alert_key = health_report['overall_status']
        now = time.time()

        # Check cooldown
        if (alert_key in self.last_alert and
            now - self.last_alert[alert_key] < self.alert_cooldown):
            return

        # Prepare alert message
        subject = f"Pokemon Gym Adapter Alert: {health_report['overall_status'].upper()}"

        body = f"""
Pokemon Gym Adapter Health Alert

Status: {health_report['overall_status'].upper()}
Timestamp: {health_report['timestamp']}

Issues Detected:
"""

        for issue in health_report['issues']:
            body += f"- {issue['component']}: {issue['issue']}\n"

        body += f"\nPlease investigate the system immediately."

        # Send email (simplified - would use proper SMTP configuration)
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = 'pokemon-system@example.com'
            msg['To'] = self.alert_email

            # This is a placeholder - configure with real SMTP settings
            print(f"📧 ALERT: {subject}")
            print(body)

            self.last_alert[alert_key] = now

        except Exception as e:
            print(f"❌ Failed to send alert: {e}")

# Continuous health monitoring
def start_health_monitoring(pool, check_interval=60):
    """Start continuous health monitoring."""
    health_monitor = HealthMonitor(pool)

    def monitor_loop():
        while True:
            try:
                health_report = health_monitor.check_system_health()

                if health_report['overall_status'] != 'healthy':
                    print(f"⚠️ System status: {health_report['overall_status']}")
                    for issue in health_report['issues']:
                        print(f"   - {issue['component']}: {issue['issue']}")

                time.sleep(check_interval)

            except Exception as e:
                print(f"❌ Health monitoring error: {e}")
                time.sleep(10)

    monitor_thread = threading.Thread(target=monitor_loop)
    monitor_thread.daemon = True
    monitor_thread.start()

    return health_monitor
```

---

This comprehensive performance tuning guide provides the strategies and tools needed to optimize the Pokemon Gym adapter for high-throughput production use. Implement these optimizations incrementally, measuring performance impacts at each step to achieve optimal results for your specific use case.

For additional support, refer to the [API documentation](pokemon_gym_adapter.md), [troubleshooting guide](troubleshooting.md), and [usage examples](../examples/pokemon_gym_usage.py).
