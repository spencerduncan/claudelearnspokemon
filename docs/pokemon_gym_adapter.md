# Pokemon Gym Adapter API Documentation

This document provides comprehensive API reference for the Pokemon Gym adapter system, including all classes, methods, and usage patterns for integrating with benchflow-ai/pokemon-gym emulators.

## Table of Contents

1. [EmulatorPool](#emulatorpool) - Container orchestration and management
2. [PokemonGymClient](#pokemongymclient) - HTTP client for emulator communication
3. [CheckpointManager](#checkpointmanager) - Game state persistence
4. [ScriptCompiler](#scriptcompiler) - DSL compilation system
5. [ExecutionResult](#executionresult) - Execution outcome data
6. [Error Handling](#error-handling) - Exception types and handling
7. [Performance Considerations](#performance-considerations)
8. [Advanced Usage Patterns](#advanced-usage-patterns)

---

## EmulatorPool

The `EmulatorPool` class orchestrates multiple Pokemon-gym Docker containers for parallel execution. It provides production-grade container lifecycle management with proper error handling and resource cleanup.

### Constructor

```python
EmulatorPool(
    pool_size: int = 4,
    base_port: int = 8081,
    image_name: str = "pokemon-gym:latest",
    startup_timeout: int = 30,
    checkpoint_manager: CheckpointManager | None = None,
    default_timeout: float | None = None
)
```

**Parameters:**
- `pool_size`: Number of containers in pool (default: 4)
- `base_port`: Starting port for sequential allocation (8081, 8082, 8083, 8084)
- `image_name`: Docker image name for containers
- `startup_timeout`: Maximum seconds to wait for container startup
- `checkpoint_manager`: Optional CheckpointManager for state loading
- `default_timeout`: Default timeout for acquire operations (None = block indefinitely)

**Example:**
```python
from claudelearnspokemon import EmulatorPool, CheckpointManager

# Basic pool with 4 emulators
pool = EmulatorPool()

# Custom configuration
checkpoint_mgr = CheckpointManager()
pool = EmulatorPool(
    pool_size=8,
    base_port=9001,
    image_name="custom-pokemon-gym:v2",
    startup_timeout=60,
    checkpoint_manager=checkpoint_mgr,
    default_timeout=30.0
)
```

### Core Methods

#### initialize()

Initialize the container pool with production-grade error handling.

```python
def initialize(self, pool_size: int | None = None) -> None
```

**Parameters:**
- `pool_size`: Override default pool size if provided

**Raises:**
- `EmulatorPoolError`: On any initialization failure with actionable message

**Example:**
```python
pool = EmulatorPool(pool_size=4)
try:
    pool.initialize()
    print("Pool initialized successfully")
except EmulatorPoolError as e:
    print(f"Initialization failed: {e}")
```

#### shutdown()

Gracefully shutdown all containers with production-grade error handling.

```python
def shutdown() -> None
```

**Features:**
- Continues shutdown process even if individual containers fail
- Idempotent operation - safe to call multiple times
- Comprehensive logging of shutdown results

**Example:**
```python
# Always call shutdown to prevent resource leaks
try:
    # Use pool for operations...
    pass
finally:
    pool.shutdown()
```

#### acquire()

Acquire an available emulator client from the pool (thread-safe).

```python
def acquire(self, timeout: float | None = None) -> PokemonGymClient
```

**Parameters:**
- `timeout`: Maximum seconds to wait (None = block indefinitely)

**Returns:**
- `PokemonGymClient` for exclusive use

**Raises:**
- `EmulatorPoolError`: If no emulators available within timeout

**Example:**
```python
# Blocking acquisition
client = pool.acquire()

# Timeout-based acquisition
try:
    client = pool.acquire(timeout=10.0)
    # Use client...
finally:
    pool.release(client)
```

#### acquire_emulator()

Acquire emulator as context manager for automatic resource cleanup.

```python
def acquire_emulator(self, timeout: float | None = None) -> EmulatorContext
```

**Returns:**
- `EmulatorContext` for use in `with` statement

**Example:**
```python
# Automatic resource management
with pool.acquire_emulator(timeout=30) as client:
    response = client.send_input("A B START")
    state = client.get_state()
# Client automatically released
```

#### execute_script()

Compile and execute script text with automatic client management.

```python
def execute_script(
    self,
    script_text: str,
    checkpoint_id: str | None = None
) -> ExecutionResult
```

**Parameters:**
- `script_text`: DSL script to compile and execute
- `checkpoint_id`: Optional checkpoint to load before execution

**Returns:**
- `ExecutionResult` with execution details and outcome

**Example:**
```python
# Basic script execution
result = pool.execute_script("PRESS A WAIT 60 PRESS START")
if result.success:
    print(f"Execution completed in {result.execution_time:.2f}s")
else:
    print(f"Execution failed: {result.error}")

# Execute with checkpoint
result = pool.execute_script(
    script_text="MOVE RIGHT PRESS A",
    checkpoint_id="checkpoint-uuid-here"
)
```

#### health_check()

Verify all emulators are responsive and healthy.

```python
def health_check() -> dict[str, Any]
```

**Returns:**
```python
{
    "status": "healthy" | "degraded" | "not_initialized",
    "healthy_count": int,
    "total_count": int,
    "emulators": {
        8081: {
            "healthy": bool,
            "container_id": str,
            "error": str | None
        }
        # ... for each port
    }
}
```

**Example:**
```python
health = pool.health_check()
if health["status"] != "healthy":
    print(f"Pool degraded: {health['healthy_count']}/{health['total_count']} healthy")

    # Restart unhealthy emulators
    for port, status in health["emulators"].items():
        if not status["healthy"]:
            pool.restart_emulator(int(port))
```

### Advanced Methods

#### compile_script()

Compile DSL script to CompiledScript using high-performance ScriptCompiler.

```python
def compile_script(self, script_text: str) -> CompiledScript
```

**Parameters:**
- `script_text`: DSL script to compile

**Returns:**
- `CompiledScript` with instructions, frame estimates, and metadata

**Example:**
```python
# Pre-compile scripts for batch execution
scripts = [
    "MOVE UP PRESS A",
    "MOVE DOWN PRESS B",
    "PRESS START WAIT 30"
]

compiled_scripts = []
for script_text in scripts:
    compiled = pool.compile_script(script_text)
    compiled_scripts.append(compiled)

# Execute compiled scripts with different emulators
for script in compiled_scripts:
    with pool.acquire_emulator() as client:
        result = pool.execute_compiled_script(client, script)
```

#### restart_emulator()

Restart specific emulator instance due to failure.

```python
def restart_emulator(self, port: int) -> None
```

**Parameters:**
- `port`: Port of the emulator to restart

**Example:**
```python
# Restart emulator on port 8081
try:
    pool.restart_emulator(8081)
    print("Emulator restarted successfully")
except EmulatorPoolError as e:
    print(f"Restart failed: {e}")
```

---

## PokemonGymClient

HTTP client wrapper for Pokemon-gym emulator communication. Provides clean interface for script execution and state management.

### Constructor

```python
PokemonGymClient(port: int, container_id: str)
```

**Parameters:**
- `port`: HTTP port for emulator communication
- `container_id`: Docker container ID for this emulator

**Note:** Typically created internally by `EmulatorPool`, not directly instantiated.

### Methods

#### send_input()

Send input sequence to the emulator.

```python
def send_input(self, input_sequence: str) -> dict[str, Any]
```

**Parameters:**
- `input_sequence`: Button inputs (A, B, START, UP, DOWN, LEFT, RIGHT, SELECT)

**Returns:**
- Response data from emulator

**Example:**
```python
with pool.acquire_emulator() as client:
    # Single input
    response = client.send_input("A")

    # Input sequence
    response = client.send_input("A B START")

    # Directional inputs
    response = client.send_input("UP UP RIGHT A")
```

#### get_state()

Get current game state from emulator.

```python
def get_state() -> dict[str, Any]
```

**Returns:**
- Current game state data including player position, items, progress flags

**Example:**
```python
with pool.acquire_emulator() as client:
    state = client.get_state()

    # Analyze game state
    player_pos = state.get("player", {})
    x, y = player_pos.get("x", 0), player_pos.get("y", 0)
    print(f"Player at ({x}, {y})")

    # Check inventory
    items = state.get("items", [])
    print(f"Player has {len(items)} items")
```

#### reset_game()

Reset the game to initial state.

```python
def reset_game() -> dict[str, Any]
```

**Returns:**
- Reset confirmation from emulator

**Example:**
```python
with pool.acquire_emulator() as client:
    # Reset to beginning
    reset_response = client.reset_game()

    # Verify reset worked
    state = client.get_state()
    assert state.get("player", {}).get("x") == 0  # Starting position
```

#### is_healthy()

Check if emulator is responding to health checks.

```python
def is_healthy() -> bool
```

**Returns:**
- `True` if emulator is healthy, `False` otherwise

**Example:**
```python
# Manual health check
if not client.is_healthy():
    print("Emulator appears unhealthy")
    # Handle appropriately - restart, skip, etc.
```

---

## CheckpointManager

Manages Pokemon game state checkpoints with LZ4 compression for efficient save/load operations.

### Constructor

```python
CheckpointManager(checkpoint_dir: str | None = None)
```

**Parameters:**
- `checkpoint_dir`: Directory for checkpoint storage (default: `~/.claudelearnspokemon/checkpoints`)

**Example:**
```python
# Default location
checkpoint_mgr = CheckpointManager()

# Custom location
checkpoint_mgr = CheckpointManager("/path/to/my/checkpoints")
```

### Methods

#### save_checkpoint()

Save game state with metadata to compressed checkpoint file.

```python
def save_checkpoint(
    self,
    game_state: dict[str, Any],
    metadata: dict[str, Any]
) -> str
```

**Parameters:**
- `game_state`: Complete game state dictionary from emulator
- `metadata`: Checkpoint metadata (location, progress, description, etc.)

**Returns:**
- Checkpoint identifier (UUID string)

**Performance:** Target < 500ms

**Example:**
```python
# Get current state from emulator
with pool.acquire_emulator() as client:
    current_state = client.get_state()

# Save with descriptive metadata
checkpoint_id = checkpoint_mgr.save_checkpoint(
    game_state=current_state,
    metadata={
        "location": "pallet_town",
        "progress": "got_pokedex",
        "description": "Just received Pokedex from Oak",
        "timestamp": time.time(),
        "player_level": 5
    }
)

print(f"Saved checkpoint: {checkpoint_id}")
```

#### load_checkpoint()

Load compressed checkpoint file and return game state with metadata.

```python
def load_checkpoint(self, checkpoint_id: str) -> dict[str, Any]
```

**Parameters:**
- `checkpoint_id`: UUID string of checkpoint to load

**Returns:**
- Dictionary with `game_state`, `metadata`, `timestamp`, etc.

**Raises:**
- `CheckpointNotFoundError`: If checkpoint doesn't exist
- `CheckpointCorruptionError`: If checkpoint is corrupted

**Example:**
```python
try:
    checkpoint_data = checkpoint_mgr.load_checkpoint(checkpoint_id)

    game_state = checkpoint_data["game_state"]
    metadata = checkpoint_data["metadata"]

    print(f"Loaded checkpoint from {metadata['location']}")

    # Use loaded state for execution
    result = pool.execute_script(
        "PRESS A WAIT 30 PRESS B",
        checkpoint_id=checkpoint_id
    )

except CheckpointNotFoundError:
    print(f"Checkpoint {checkpoint_id} not found")
except CheckpointCorruptionError:
    print(f"Checkpoint {checkpoint_id} is corrupted")
```

#### list_checkpoints()

List all available checkpoints with basic metadata.

```python
def list_checkpoints() -> list[dict[str, Any]]
```

**Returns:**
- List of checkpoint info dictionaries

**Example:**
```python
checkpoints = checkpoint_mgr.list_checkpoints()
for checkpoint in checkpoints:
    print(f"{checkpoint['id']}: {checkpoint['metadata']['description']}")
    print(f"  Location: {checkpoint['metadata']['location']}")
    print(f"  Saved: {checkpoint['timestamp']}")
```

#### delete_checkpoint()

Delete a specific checkpoint file.

```python
def delete_checkpoint(self, checkpoint_id: str) -> None
```

**Example:**
```python
# Clean up old checkpoints
old_checkpoints = checkpoint_mgr.list_checkpoints()
for cp in old_checkpoints:
    # Delete checkpoints older than 1 week
    if time.time() - cp['timestamp'] > 7 * 24 * 3600:
        checkpoint_mgr.delete_checkpoint(cp['id'])
```

---

## ExecutionResult

Container for script execution results with comprehensive outcome data.

### Constructor

```python
ExecutionResult(
    success: bool,
    output: Any,
    error: str | None = None,
    execution_time: float | None = None,
    checkpoint_reached: str | None = None
)
```

### Attributes

- `success: bool` - Whether execution completed successfully
- `output: Any` - Execution output data (emulator responses, final state, etc.)
- `error: str | None` - Error message if execution failed
- `execution_time: float | None` - Total execution time in seconds
- `checkpoint_reached: str | None` - Significant checkpoint reached during execution

### Methods

#### __str__()

Human-readable string representation.

**Example:**
```python
result = pool.execute_script("PRESS A WAIT 60 PRESS START")
print(result)  # ExecutionResult(SUCCESS, time=2.34s)
```

### Usage Patterns

```python
# Basic success/failure handling
result = pool.execute_script("MOVE UP PRESS A")
if result.success:
    final_state = result.output["final_state"]
    print(f"Completed in {result.execution_time:.2f}s")

    # Check if significant progress made
    if result.checkpoint_reached:
        print(f"Reached checkpoint: {result.checkpoint_reached}")
else:
    print(f"Execution failed: {result.error}")
    # Handle failure - retry, restart emulator, etc.

# Performance monitoring
execution_times = []
for script in script_list:
    result = pool.execute_script(script)
    if result.execution_time:
        execution_times.append(result.execution_time)

avg_time = sum(execution_times) / len(execution_times)
print(f"Average execution time: {avg_time:.3f}s")
```

---

## Error Handling

The adapter provides specific exception types for different failure modes.

### Exception Hierarchy

```python
Exception
├── EmulatorPoolError          # General pool operation failures
├── CheckpointError            # Base checkpoint exception
│   ├── CheckpointNotFoundError    # Checkpoint doesn't exist
│   └── CheckpointCorruptionError  # Checkpoint file corrupted
└── requests.RequestException  # HTTP communication failures (from requests library)
```

### EmulatorPoolError

Raised for pool-related failures with actionable error messages.

**Common Causes:**
- Docker daemon unavailable
- Port conflicts
- Container startup failures
- Resource exhaustion

**Example:**
```python
try:
    pool = EmulatorPool()
    pool.initialize()
except EmulatorPoolError as e:
    print(f"Pool initialization failed: {e}")
    # Error message includes specific failure cause and remediation
```

### CheckpointError

Base class for checkpoint-related failures.

**Example:**
```python
try:
    checkpoint_data = checkpoint_mgr.load_checkpoint(checkpoint_id)
except CheckpointNotFoundError:
    print("Checkpoint not found - using default state")
    checkpoint_data = None
except CheckpointCorruptionError:
    print("Checkpoint corrupted - creating new save")
    # Handle corrupted checkpoint appropriately
```

### Comprehensive Error Handling Pattern

```python
def robust_script_execution(pool, script_text, max_retries=3):
    """Execute script with comprehensive error handling and retries."""

    for attempt in range(max_retries):
        try:
            result = pool.execute_script(script_text)
            if result.success:
                return result
            else:
                print(f"Execution failed (attempt {attempt + 1}): {result.error}")

        except EmulatorPoolError as e:
            if "no emulators available" in str(e).lower():
                print(f"Pool busy, waiting... (attempt {attempt + 1})")
                time.sleep(5)
            else:
                print(f"Pool error: {e}")
                break

        except requests.RequestException as e:
            print(f"Communication error: {e}")
            # Possibly restart emulator

    raise RuntimeError(f"Script execution failed after {max_retries} attempts")
```

---

## Performance Considerations

### Target Performance Metrics

- **Script compilation**: < 100ms
- **Checkpoint saving**: < 500ms
- **Checkpoint loading**: < 500ms
- **Full execution cycle**: < 5 seconds
- **Emulator acquisition**: < 1 second (under normal load)

### Optimization Strategies

#### 1. Pre-compile Scripts for Batch Execution

```python
# Inefficient - compiles each time
scripts = ["PRESS A", "PRESS B", "PRESS START"]
for script_text in scripts:
    result = pool.execute_script(script_text)  # Compilation overhead

# Efficient - compile once, execute many times
compiled_scripts = []
for script_text in scripts:
    compiled_scripts.append(pool.compile_script(script_text))

# Execute pre-compiled scripts
for script in compiled_scripts:
    with pool.acquire_emulator() as client:
        result = pool.execute_compiled_script(client, script)
```

#### 2. Parallel Execution Patterns

```python
import concurrent.futures

def execute_script_parallel(script_text):
    with pool.acquire_emulator(timeout=30) as client:
        return pool.execute_compiled_script(client, compiled_script)

# Execute multiple scripts in parallel
script_texts = ["MOVE UP PRESS A", "MOVE DOWN PRESS B", "PRESS START"]
compiled_scripts = [pool.compile_script(text) for text in script_texts]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for script in compiled_scripts:
        future = executor.submit(execute_script_parallel, script)
        futures.append(future)

    # Collect results
    results = [future.result() for future in futures]
```

#### 3. Checkpoint Strategy Optimization

```python
# Efficient checkpoint usage
checkpoint_cache = {}

def get_or_create_checkpoint(state_key, game_state, metadata):
    if state_key in checkpoint_cache:
        return checkpoint_cache[state_key]

    checkpoint_id = checkpoint_mgr.save_checkpoint(game_state, metadata)
    checkpoint_cache[state_key] = checkpoint_id
    return checkpoint_id

# Reuse checkpoints for similar states
common_states = {
    "pallet_town_start": {"location": "pallet_town", "progress": "start"},
    "route_1_entered": {"location": "route_1", "progress": "left_town"}
}

for state_key, metadata in common_states.items():
    checkpoint_id = get_or_create_checkpoint(state_key, game_state, metadata)
```

#### 4. Connection Pooling and Reuse

```python
# Long-running operations - reuse clients
class LongRunningExperiment:
    def __init__(self, pool):
        self.pool = pool
        self.client = None

    def __enter__(self):
        self.client = self.pool.acquire(timeout=60)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.pool.release(self.client)

    def run_experiment(self, scripts):
        results = []
        for script_text in scripts:
            # Reuse same client for entire experiment
            input_seq = self.pool._compile_script(script_text)
            response = self.client.send_input(input_seq)
            results.append(response)
        return results

# Usage
with LongRunningExperiment(pool) as experiment:
    results = experiment.run_experiment(["PRESS A", "WAIT 30", "PRESS B"])
```

### Performance Monitoring

```python
import time
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.timings = defaultdict(list)

    def measure(self, operation_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.monotonic()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.monotonic() - start
                    self.timings[operation_name].append(duration)
            return wrapper
        return decorator

    def report(self):
        for operation, times in self.timings.items():
            avg = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            print(f"{operation}: avg={avg:.3f}s, max={max_time:.3f}s, min={min_time:.3f}s")

# Usage
monitor = PerformanceMonitor()

@monitor.measure("script_execution")
def execute_with_monitoring(pool, script):
    return pool.execute_script(script)

# Run experiments
for script in test_scripts:
    execute_with_monitoring(pool, script)

monitor.report()
```

---

## Advanced Usage Patterns

### 1. Multi-Stage Experiment Pipeline

```python
class ExperimentPipeline:
    def __init__(self, pool, checkpoint_mgr):
        self.pool = pool
        self.checkpoint_mgr = checkpoint_mgr
        self.stage_checkpoints = {}

    def setup_stage(self, stage_name, setup_script):
        """Setup a reusable experiment stage."""
        result = self.pool.execute_script(setup_script)
        if result.success:
            # Save stage state for reuse
            with self.pool.acquire_emulator() as client:
                state = client.get_state()

            checkpoint_id = self.checkpoint_mgr.save_checkpoint(
                game_state=state,
                metadata={"stage": stage_name, "type": "experiment_stage"}
            )
            self.stage_checkpoints[stage_name] = checkpoint_id
            return checkpoint_id
        else:
            raise RuntimeError(f"Failed to setup stage {stage_name}: {result.error}")

    def run_from_stage(self, stage_name, experiment_script):
        """Run experiment starting from specific stage."""
        if stage_name not in self.stage_checkpoints:
            raise ValueError(f"Stage {stage_name} not setup")

        checkpoint_id = self.stage_checkpoints[stage_name]
        return self.pool.execute_script(experiment_script, checkpoint_id=checkpoint_id)

# Usage
pipeline = ExperimentPipeline(pool, checkpoint_mgr)

# Setup reusable stages
pipeline.setup_stage("pallet_town_ready", "PRESS A WAIT 60 PRESS START")
pipeline.setup_stage("route_1_entry", "MOVE UP UP UP PRESS A")

# Run experiments from different stages
result1 = pipeline.run_from_stage("pallet_town_ready", "MOVE LEFT PRESS A")
result2 = pipeline.run_from_stage("route_1_entry", "MOVE RIGHT PRESS B")
```

### 2. Adaptive Timeout Management

```python
class AdaptiveTimeoutPool:
    def __init__(self, base_pool):
        self.pool = base_pool
        self.timeout_history = []
        self.base_timeout = 30.0

    def smart_acquire(self, urgency="normal"):
        """Acquire emulator with adaptive timeout based on history."""
        if urgency == "low":
            timeout = self.base_timeout * 2
        elif urgency == "high":
            timeout = self.base_timeout * 0.5
        else:
            # Use historical data to predict good timeout
            if self.timeout_history:
                avg_wait = sum(self.timeout_history[-10:]) / min(len(self.timeout_history), 10)
                timeout = max(avg_wait * 1.5, self.base_timeout)
            else:
                timeout = self.base_timeout

        start_time = time.monotonic()
        try:
            client = self.pool.acquire(timeout=timeout)
            actual_wait = time.monotonic() - start_time
            self.timeout_history.append(actual_wait)
            return client
        except EmulatorPoolError:
            # Track failed waits too
            self.timeout_history.append(timeout)
            raise
```

### 3. Fault-Tolerant Execution

```python
class FaultTolerantExecutor:
    def __init__(self, pool, max_retries=3):
        self.pool = pool
        self.max_retries = max_retries
        self.failed_emulators = set()

    def execute_with_failover(self, script_text, checkpoint_id=None):
        """Execute script with automatic failover on emulator failure."""

        for attempt in range(self.max_retries):
            try:
                # Try to avoid known-bad emulators
                client = None
                max_acquire_attempts = 5

                for _ in range(max_acquire_attempts):
                    client = self.pool.acquire(timeout=10)
                    if client.port not in self.failed_emulators:
                        break
                    else:
                        # This emulator has failed before, try another
                        self.pool.release(client)
                        client = None
                        time.sleep(1)

                if client is None:
                    raise EmulatorPoolError("No healthy emulators available")

                # Execute with this client
                if checkpoint_id:
                    # Load checkpoint manually for more control
                    checkpoint_data = self.pool.checkpoint_manager.load_checkpoint(checkpoint_id)
                    # Would need to implement state loading in PokemonGymClient

                response = client.send_input(script_text)
                final_state = client.get_state()

                # Success - remove from failed list if it was there
                self.failed_emulators.discard(client.port)

                return ExecutionResult(
                    success=True,
                    output={"response": response, "final_state": final_state},
                    execution_time=time.monotonic() - start_time
                )

            except Exception as e:
                if client:
                    # Mark this emulator as problematic
                    self.failed_emulators.add(client.port)
                    self.pool.release(client)

                if attempt == self.max_retries - 1:
                    # Final attempt failed
                    return ExecutionResult(
                        success=False,
                        output=None,
                        error=f"All {self.max_retries} attempts failed. Last error: {e}"
                    )

                # Wait before retry
                time.sleep(2 ** attempt)  # Exponential backoff
```

### 4. State-based Experiment Design

```python
class StateBasedExperiment:
    """Run experiments that branch based on game state."""

    def __init__(self, pool, checkpoint_mgr):
        self.pool = pool
        self.checkpoint_mgr = checkpoint_mgr

    def conditional_execution(self, script_text, state_conditions, checkpoint_id=None):
        """Execute script with different paths based on game state."""

        result = self.pool.execute_script(script_text, checkpoint_id)
        if not result.success:
            return result

        # Analyze final state
        final_state = result.output.get("final_state", {})

        # Check conditions and execute appropriate follow-up
        for condition, follow_up_script in state_conditions.items():
            if self._check_condition(final_state, condition):
                # Save intermediate state
                intermediate_checkpoint = self.checkpoint_mgr.save_checkpoint(
                    game_state=final_state,
                    metadata={"type": "conditional_branch", "condition": condition}
                )

                # Execute follow-up
                follow_up_result = self.pool.execute_script(
                    follow_up_script,
                    checkpoint_id=intermediate_checkpoint
                )

                # Combine results
                return ExecutionResult(
                    success=follow_up_result.success,
                    output={
                        "initial_result": result.output,
                        "condition_met": condition,
                        "follow_up_result": follow_up_result.output
                    },
                    execution_time=result.execution_time + follow_up_result.execution_time,
                    error=follow_up_result.error
                )

        # No conditions met - return original result
        return result

    def _check_condition(self, game_state, condition):
        """Check if game state meets specified condition."""
        # Example conditions
        if condition == "in_battle":
            return game_state.get("battle_active", False)
        elif condition == "in_menu":
            return game_state.get("menu_open", False)
        elif condition.startswith("location:"):
            location = condition.split(":")[1]
            return game_state.get("location", {}).get("name") == location

        return False

# Usage
experiment = StateBasedExperiment(pool, checkpoint_mgr)

result = experiment.conditional_execution(
    script_text="PRESS A WAIT 60",
    state_conditions={
        "in_battle": "PRESS B PRESS B",  # Escape from battle
        "in_menu": "PRESS START",       # Close menu
        "location:pallet_town": "MOVE UP UP"  # Move north if in Pallet Town
    },
    checkpoint_id="some-checkpoint-id"
)
```

---

## Configuration Reference

### Environment Variables

```bash
# Checkpoint storage
CLAUDE_POKEMON_CHECKPOINT_DIR="/custom/path/checkpoints"

# Docker configuration
DOCKER_HOST="unix:///var/run/docker.sock"
DOCKER_API_VERSION="1.40"

# Performance tuning
CLAUDE_POKEMON_MAX_CONTAINERS=8
CLAUDE_POKEMON_DEFAULT_TIMEOUT=60

# Logging
CLAUDE_POKEMON_LOG_LEVEL=INFO
CLAUDE_POKEMON_LOG_FORMAT=json
```

### Configuration File Support

```python
# Load configuration from file
from pydantic import BaseSettings

class PokemonGymConfig(BaseSettings):
    pool_size: int = 4
    base_port: int = 8081
    image_name: str = "pokemon-gym:latest"
    startup_timeout: int = 30
    checkpoint_dir: str = None

    class Config:
        env_prefix = "CLAUDE_POKEMON_"
        env_file = ".env"

# Usage
config = PokemonGymConfig()
pool = EmulatorPool(
    pool_size=config.pool_size,
    base_port=config.base_port,
    image_name=config.image_name,
    startup_timeout=config.startup_timeout
)
```

---

This comprehensive API documentation covers all aspects of the Pokemon Gym adapter system. For working code examples, see [examples/pokemon_gym_usage.py](../examples/pokemon_gym_usage.py).
