# Pokemon Gym Adapter Troubleshooting Guide

This guide covers common issues, error messages, and solutions for the Pokemon Gym adapter system. Use this as a reference when encountering problems during installation, configuration, or runtime operations.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Docker Problems](#docker-problems)
3. [Container Startup Failures](#container-startup-failures)
4. [Connection and Communication Issues](#connection-and-communication-issues)
5. [Performance Problems](#performance-problems)
6. [Checkpoint Issues](#checkpoint-issues)
7. [Memory and Resource Problems](#memory-and-resource-problems)
8. [Error Messages Reference](#error-messages-reference)
9. [Diagnostic Commands](#diagnostic-commands)
10. [Advanced Debugging](#advanced-debugging)

---

## Installation Issues

### Problem: `docker: command not found`

**Symptoms:**
```bash
./setup.sh
docker: command not found
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker

# macOS
brew install --cask docker

# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker ps
```

### Problem: `Permission denied` when running Docker commands

**Symptoms:**
```
docker: Got permission denied while trying to connect to the Docker daemon socket
```

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Refresh group membership (logout/login or use newgrp)
newgrp docker

# Alternative: Use sudo (not recommended for development)
sudo docker ps

# Verify permissions
docker run hello-world
```

### Problem: Python version compatibility

**Symptoms:**
```
Python 3.10+ required, found Python 3.8
```

**Solution:**
```bash
# Ubuntu - install Python 3.10+
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Use specific Python version
python3.10 -m venv venv
source venv/bin/activate

# macOS - use pyenv
brew install pyenv
pyenv install 3.10.0
pyenv local 3.10.0

# Verify version
python --version
```

### Problem: Virtual environment creation fails

**Symptoms:**
```
Error: Unable to create virtual environment
python3: No module named venv
```

**Solution:**
```bash
# Ubuntu/Debian - install venv module
sudo apt install python3-venv python3-pip

# Alternative: use virtualenv
pip install virtualenv
virtualenv venv

# Verify venv works
python3 -m venv test_env
rm -rf test_env  # cleanup
```

---

## Docker Problems

### Problem: Docker daemon not running

**Symptoms:**
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

**Solution:**
```bash
# Linux - start Docker service
sudo systemctl start docker
sudo systemctl enable docker  # Start on boot

# macOS - start Docker Desktop
open /Applications/Docker.app

# Verify Docker is running
docker info
docker ps
```

### Problem: Docker image not found

**Symptoms:**
```
EmulatorPoolError: Pokemon-gym image not found: pokemon-gym:latest
```

**Solution:**
```bash
# Clone and build pokemon-gym image
git clone https://github.com/benchflow-ai/pokemon-gym.git
cd pokemon-gym

# Build the Docker image
docker build -t pokemon-gym:latest .

# Verify image exists
docker images | grep pokemon-gym

# Test image works
docker run --rm pokemon-gym:latest echo "Image works"
```

### Problem: Docker build fails

**Symptoms:**
```
failed to solve with frontend dockerfile.v0
```

**Solution:**
```bash
# Update Docker to latest version
sudo apt update && sudo apt install docker.io

# Clear Docker build cache
docker builder prune -a

# Build with no cache
docker build --no-cache -t pokemon-gym:latest .

# Check available space
df -h
docker system df

# Clean up Docker if space is low
docker system prune -a
```

---

## Container Startup Failures

### Problem: Port already in use

**Symptoms:**
```
Docker API error starting container: port is already allocated
EmulatorPoolError: Check port 8081 availability
```

**Solution:**
```bash
# Find process using the port
sudo lsof -i :8081
sudo netstat -tulpn | grep :8081

# Kill process using port
sudo kill -9 <PID>

# Or use different ports
pool = EmulatorPool(base_port=9001)

# Check available ports
python3 -c "
import socket
for port in range(8080, 8090):
    try:
        s = socket.socket()
        s.bind(('localhost', port))
        s.close()
        print(f'Port {port} available')
    except:
        print(f'Port {port} in use')
"
```

### Problem: Container startup timeout

**Symptoms:**
```
Container startup timeout (30s) exceeded
Container status: creating
```

**Solution:**
```bash
# Increase timeout for slower systems
pool = EmulatorPool(startup_timeout=120)

# Check system resources
free -h          # Memory
df -h            # Disk space
docker stats     # Docker resource usage

# Check Docker logs for specific container
docker logs <container-id>

# Restart Docker daemon
sudo systemctl restart docker

# Test container manually
docker run --rm -p 8080:8080 pokemon-gym:latest
```

### Problem: Container exits immediately

**Symptoms:**
```
Container failed to start (status: exited)
Check container logs for details
```

**Solution:**
```bash
# Check container logs
docker logs <container-id>

# Run container interactively for debugging
docker run -it --rm pokemon-gym:latest /bin/bash

# Check if image is corrupted
docker pull pokemon-gym:latest  # If using remote image
docker build --no-cache -t pokemon-gym:latest .  # If building locally

# Test with minimal container
docker run --rm pokemon-gym:latest echo "Container can start"
```

---

## Connection and Communication Issues

### Problem: HTTP connection refused

**Symptoms:**
```
Failed to send input to emulator on port 8081: Connection refused
requests.exceptions.ConnectionError
```

**Solution:**
```bash
# Verify container is running
docker ps | grep pokemon

# Check container health
docker exec <container-id> curl http://localhost:8080/health

# Test connection from host
curl http://localhost:8081/health

# Check port mapping
docker port <container-id>

# Restart specific emulator
pool.restart_emulator(8081)

# Manual container restart
docker restart <container-id>
```

### Problem: HTTP timeout errors

**Symptoms:**
```
Failed to get state from emulator on port 8081: Read timed out
```

**Solution:**
```python
# Increase timeout for slow operations
client.session.timeout = 30  # Not recommended - use timeouts per request

# Better: Use EmulatorPool with longer timeouts
pool = EmulatorPool(default_timeout=60.0)

# Check if emulator is overloaded
health = pool.health_check()
print(health)

# Monitor emulator resource usage
import time
for _ in range(10):
    with pool.acquire_emulator(timeout=5) as client:
        start = time.time()
        client.get_state()
        print(f"State request took {time.time() - start:.2f}s")
        time.sleep(1)
```

### Problem: Inconsistent responses from emulator

**Symptoms:**
```
Sometimes works, sometimes fails with same input
EmulatorPoolError: Unexpected response format
```

**Solution:**
```python
# Add retry logic with exponential backoff
import time
import random

def robust_send_input(client, input_sequence, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.send_input(input_sequence)
            # Validate response format
            if isinstance(response, dict):
                return response
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)

    raise RuntimeError("All retry attempts failed")

# Check emulator health before operations
if not client.is_healthy():
    pool.restart_emulator(client.port)
    client = pool.acquire()
```

---

## Performance Problems

### Problem: Slow script execution

**Symptoms:**
```
Script execution taking > 10 seconds
Performance targets not met
```

**Solution:**
```python
# Profile execution components
import time

def profile_execution(pool, script_text):
    timings = {}

    # Compilation time
    start = time.time()
    compiled_script = pool.compile_script(script_text)
    timings['compilation'] = time.time() - start

    # Acquisition time
    start = time.time()
    client = pool.acquire(timeout=30)
    timings['acquisition'] = time.time() - start

    try:
        # Execution time
        start = time.time()
        response = client.send_input(script_text)
        timings['execution'] = time.time() - start

        # State retrieval time
        start = time.time()
        state = client.get_state()
        timings['state_retrieval'] = time.time() - start

    finally:
        pool.release(client)

    return timings

# Check what's slow
timings = profile_execution(pool, "A B START")
for component, duration in timings.items():
    print(f"{component}: {duration*1000:.1f}ms")
```

### Problem: High memory usage

**Symptoms:**
```
System running out of memory
Docker containers using excessive RAM
```

**Solution:**
```bash
# Monitor memory usage
docker stats
htop  # or top

# Limit container memory
docker run -m 512m pokemon-gym:latest

# In EmulatorPool, containers already have memory limits:
# mem_limit="512m" in _start_single_container()

# Check for memory leaks in checkpoints
import os
import psutil

def check_checkpoint_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Monitor during checkpoint operations
checkpoint_mgr = CheckpointManager()
check_checkpoint_memory()

# Save large checkpoint
large_state = {"data": ["x"] * 100000}
checkpoint_id = checkpoint_mgr.save_checkpoint(large_state, {})
check_checkpoint_memory()

# Load checkpoint
loaded_data = checkpoint_mgr.load_checkpoint(checkpoint_id)
check_checkpoint_memory()
```

### Problem: Pool resource exhaustion

**Symptoms:**
```
No emulators available within timeout
All 4 emulators are currently busy
```

**Solution:**
```python
# Monitor pool utilization
def monitor_pool_usage(pool, duration_seconds=60):
    import time

    start_time = time.time()
    stats = {"busy_count": [], "available_count": []}

    while time.time() - start_time < duration_seconds:
        status = pool.get_status()
        stats["busy_count"].append(status["busy_count"])
        stats["available_count"].append(status["available_count"])
        time.sleep(1)

    avg_busy = sum(stats["busy_count"]) / len(stats["busy_count"])
    max_busy = max(stats["busy_count"])

    print(f"Average busy emulators: {avg_busy:.1f}")
    print(f"Peak busy emulators: {max_busy}")

    if avg_busy > pool.pool_size * 0.8:
        print("⚠️ Pool heavily utilized - consider increasing pool_size")

# Increase pool size for high-throughput scenarios
pool = EmulatorPool(pool_size=8)  # Instead of default 4

# Use shorter timeouts to fail fast
with pool.acquire_emulator(timeout=5) as client:
    # Operations...
    pass
```

---

## Checkpoint Issues

### Problem: Checkpoint corruption

**Symptoms:**
```
CheckpointCorruptionError: Checkpoint file is corrupted
Error decompressing LZ4 data
```

**Solution:**
```python
# Check checkpoint file integrity
import lz4.frame
from pathlib import Path

def verify_checkpoint_integrity(checkpoint_mgr, checkpoint_id):
    checkpoint_file = checkpoint_mgr.checkpoint_dir / f"{checkpoint_id}.lz4"

    if not checkpoint_file.exists():
        print(f"❌ Checkpoint file not found: {checkpoint_file}")
        return False

    try:
        with open(checkpoint_file, 'rb') as f:
            compressed_data = f.read()

        # Try to decompress
        decompressed_data = lz4.frame.decompress(compressed_data)

        # Try to parse JSON
        import json
        data = json.loads(decompressed_data)

        print(f"✅ Checkpoint {checkpoint_id} is valid")
        return True

    except Exception as e:
        print(f"❌ Checkpoint {checkpoint_id} is corrupted: {e}")
        return False

# Repair strategy: re-create checkpoint
def repair_checkpoint_system():
    # Get current game state
    with pool.acquire_emulator() as client:
        current_state = client.get_state()

    # Create new checkpoint
    new_checkpoint = checkpoint_mgr.save_checkpoint(
        game_state=current_state,
        metadata={"type": "repair", "description": "Recovered state"}
    )

    return new_checkpoint
```

### Problem: Checkpoint directory permissions

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: '/.claudelearnspokemon/checkpoints'
```

**Solution:**
```bash
# Check current permissions
ls -la ~/.claudelearnspokemon/

# Fix permissions
chmod 755 ~/.claudelearnspokemon/
chmod 755 ~/.claudelearnspokemon/checkpoints/

# Or use custom directory with correct permissions
mkdir -p ./my-checkpoints
chmod 755 ./my-checkpoints

# Use custom directory
checkpoint_mgr = CheckpointManager("./my-checkpoints")
```

### Problem: Disk space issues with checkpoints

**Symptoms:**
```
OSError: [Errno 28] No space left on device
```

**Solution:**
```python
# Monitor checkpoint disk usage
def check_checkpoint_disk_usage(checkpoint_mgr):
    import shutil

    total, used, free = shutil.disk_usage(checkpoint_mgr.checkpoint_dir)

    print(f"Checkpoint directory: {checkpoint_mgr.checkpoint_dir}")
    print(f"Total space: {total // (1024**3)} GB")
    print(f"Used space: {used // (1024**3)} GB")
    print(f"Free space: {free // (1024**3)} GB")

    if free < 1024**3:  # Less than 1GB
        print("⚠️ Low disk space - consider cleanup")

# Automated cleanup strategy
def cleanup_old_checkpoints(checkpoint_mgr, max_age_days=7):
    import time
    from pathlib import Path

    cutoff_time = time.time() - (max_age_days * 24 * 3600)
    cleaned_count = 0

    for checkpoint_file in checkpoint_mgr.checkpoint_dir.glob("*.lz4"):
        if checkpoint_file.stat().st_mtime < cutoff_time:
            checkpoint_id = checkpoint_file.stem
            checkpoint_mgr.delete_checkpoint(checkpoint_id)
            cleaned_count += 1

    print(f"Cleaned up {cleaned_count} old checkpoints")

# Regular maintenance
check_checkpoint_disk_usage(checkpoint_mgr)
cleanup_old_checkpoints(checkpoint_mgr, max_age_days=3)
```

---

## Memory and Resource Problems

### Problem: Memory leaks in long-running processes

**Symptoms:**
```
Memory usage continuously increasing
System becomes unresponsive after hours of operation
```

**Solution:**
```python
# Memory monitoring decorator
import functools
import psutil
import os

def monitor_memory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_diff = mem_after - mem_before

        if mem_diff > 50:  # More than 50MB increase
            print(f"⚠️ High memory usage in {func.__name__}: +{mem_diff:.1f}MB")

        return result
    return wrapper

# Apply to operations
@monitor_memory
def monitored_execute_script(pool, script_text):
    return pool.execute_script(script_text)

# Resource cleanup strategy
class ResourceManager:
    def __init__(self, pool, checkpoint_mgr):
        self.pool = pool
        self.checkpoint_mgr = checkpoint_mgr
        self.operation_count = 0

    def periodic_cleanup(self):
        """Perform cleanup every N operations."""
        self.operation_count += 1

        if self.operation_count % 100 == 0:  # Every 100 operations
            # Force garbage collection
            import gc
            gc.collect()

            # Check pool health and restart unhealthy emulators
            health = self.pool.health_check()
            for port, status in health["emulators"].items():
                if not status["healthy"]:
                    self.pool.restart_emulator(int(port))

            print(f"🧹 Performed cleanup after {self.operation_count} operations")

# Usage
resource_mgr = ResourceManager(pool, checkpoint_mgr)

# In operation loop
for script in scripts:
    result = monitored_execute_script(pool, script)
    resource_mgr.periodic_cleanup()
```

---

## Error Messages Reference

### EmulatorPoolError Messages

| Error Message | Cause | Solution |
|---------------|--------|----------|
| `Docker daemon unavailable` | Docker not running | Start Docker service |
| `Pokemon-gym image not found` | Image not built | Build pokemon-gym image |
| `Port is already allocated` | Port conflict | Change base_port or kill process |
| `Container startup timeout exceeded` | Slow system/resource limits | Increase startup_timeout |
| `No emulators available within timeout` | Pool exhausted | Increase pool_size or reduce load |
| `EmulatorPool not initialized` | Called methods before initialize() | Call pool.initialize() first |

### CheckpointError Messages

| Error Message | Cause | Solution |
|---------------|--------|----------|
| `Checkpoint not found` | Invalid checkpoint ID | Check checkpoint ID exists |
| `Checkpoint corrupted` | File corruption/disk issues | Verify file integrity, re-create |
| `Permission denied` | Directory permissions | Fix directory permissions |
| `No space left on device` | Disk full | Clean up old checkpoints |

### HTTP Connection Errors

| Error Message | Cause | Solution |
|---------------|--------|----------|
| `Connection refused` | Container not running | Check container status |
| `Read timed out` | Slow emulator response | Increase request timeout |
| `Bad Gateway` | Container networking issue | Restart container |

---

## Diagnostic Commands

### System Health Check

```bash
#!/bin/bash
# comprehensive_health_check.sh

echo "=== Pokemon Gym Adapter Health Check ==="

# Check Docker
echo "🐳 Docker Status:"
docker --version
docker info | grep "Server Version"
docker ps | grep pokemon || echo "No Pokemon containers running"

# Check Python environment
echo -e "\n🐍 Python Environment:"
python --version
pip list | grep -E "(claudelearnspokemon|docker|requests)" || echo "Packages not installed"

# Check ports
echo -e "\n🔌 Port Status:"
for port in 8081 8082 8083 8084; do
    nc -z localhost $port && echo "Port $port: OPEN" || echo "Port $port: CLOSED"
done

# Check disk space
echo -e "\n💾 Disk Space:"
df -h | grep -E "(Filesystem|/$|/home)"

# Check memory
echo -e "\n🧠 Memory Usage:"
free -h

echo -e "\n=== Health Check Complete ==="
```

### Python Diagnostic Script

```python
#!/usr/bin/env python3
"""
diagnostic_script.py - Comprehensive system diagnostic
"""

import docker
import requests
import time
from pathlib import Path

def diagnose_docker():
    """Diagnose Docker-related issues."""
    print("🐳 Docker Diagnostics:")

    try:
        client = docker.from_env()
        print(f"✅ Docker client connected")

        # Check images
        images = client.images.list("pokemon-gym")
        print(f"✅ Pokemon-gym images found: {len(images)}")

        # Check containers
        containers = client.containers.list(all=True,
                                          filters={"name": "pokemon-emulator"})
        print(f"📦 Pokemon containers: {len(containers)}")

        for container in containers:
            print(f"   {container.name}: {container.status}")

    except Exception as e:
        print(f"❌ Docker diagnostic failed: {e}")

def diagnose_network():
    """Diagnose network connectivity to emulators."""
    print("\n🌐 Network Diagnostics:")

    ports = [8081, 8082, 8083, 8084]
    for port in ports:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=3)
            print(f"✅ Port {port}: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Port {port}: Connection refused")
        except requests.exceptions.Timeout:
            print(f"⚠️ Port {port}: Timeout")
        except Exception as e:
            print(f"❌ Port {port}: {e}")

def diagnose_checkpoints():
    """Diagnose checkpoint system."""
    print("\n💾 Checkpoint Diagnostics:")

    checkpoint_dir = Path.home() / ".claudelearnspokemon" / "checkpoints"

    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.lz4"))
        print(f"✅ Checkpoint directory exists: {checkpoint_dir}")
        print(f"📄 Checkpoint files: {len(checkpoints)}")

        # Check directory permissions
        if checkpoint_dir.stat().st_mode & 0o200:  # Write permission
            print("✅ Directory writable")
        else:
            print("❌ Directory not writable")
    else:
        print(f"⚠️ Checkpoint directory not found: {checkpoint_dir}")

if __name__ == "__main__":
    diagnose_docker()
    diagnose_network()
    diagnose_checkpoints()
    print("\n🏁 Diagnostics complete!")
```

---

## Advanced Debugging

### Enable Debug Logging

```python
import logging
import structlog

# Enable debug logging for all components
logging.basicConfig(level=logging.DEBUG)

# Configure structlog for detailed checkpoint logs
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Now all operations will have detailed logging
pool = EmulatorPool(pool_size=2)
pool.initialize()
```

### Network Traffic Analysis

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time

class DebugHTTPAdapter(HTTPAdapter):
    def send(self, request, **kwargs):
        print(f"🌐 HTTP Request: {request.method} {request.url}")
        start_time = time.time()

        try:
            response = super().send(request, **kwargs)
            duration = time.time() - start_time
            print(f"✅ HTTP Response: {response.status_code} ({duration:.3f}s)")
            return response
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ HTTP Error: {e} ({duration:.3f}s)")
            raise

# Use debug adapter
session = requests.Session()
session.mount("http://", DebugHTTPAdapter())
session.mount("https://", DebugHTTPAdapter())

# Now all HTTP requests will be logged
client.session = session
```

### Container Log Analysis

```bash
# Get real-time logs from all Pokemon containers
docker ps | grep pokemon-emulator | awk '{print $1}' | xargs -I {} docker logs -f {}

# Search for specific errors in logs
docker ps | grep pokemon-emulator | awk '{print $1}' | xargs -I {} sh -c 'echo "=== Container {} ===" && docker logs {} 2>&1 | grep -i error'

# Check container resource usage over time
docker stats $(docker ps | grep pokemon-emulator | awk '{print $1}')
```

### Performance Profiling

```python
import cProfile
import pstats
from io import StringIO

def profile_operation(func, *args, **kwargs):
    """Profile a function call and return statistics."""
    pr = cProfile.Profile()
    pr.enable()

    result = func(*args, **kwargs)

    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()

    print(s.getvalue())
    return result

# Profile script execution
result = profile_operation(pool.execute_script, "A B START")

# Profile checkpoint operations
checkpoint_id = profile_operation(
    checkpoint_mgr.save_checkpoint,
    {"test": "data"},
    {"type": "profile_test"}
)
```

---

## Getting Help

If you've tried the solutions in this guide and are still experiencing issues:

1. **Check the GitHub Issues**: Search for similar problems in the project repository
2. **Enable Debug Logging**: Use the debug logging examples above to get detailed information
3. **Create Minimal Reproduction**: Try to reproduce the issue with minimal code
4. **Gather System Information**: Include OS, Docker version, Python version in bug reports
5. **Include Full Error Messages**: Copy the complete error message and stack trace

### Information to Include in Bug Reports

```python
# System information script
import sys
import platform
import docker
import pkg_resources

print("=== System Information ===")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print(f"Docker: {docker.from_env().version()['Version']}")

print("\n=== Package Versions ===")
packages = ['claudelearnspokemon', 'docker', 'requests', 'lz4', 'structlog']
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f"{pkg}: {version}")
    except:
        print(f"{pkg}: Not installed")
```

This troubleshooting guide should help resolve most common issues with the Pokemon Gym adapter system. For additional support, refer to the [API documentation](pokemon_gym_adapter.md) and [usage examples](../examples/pokemon_gym_usage.py).
