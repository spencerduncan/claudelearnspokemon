# Claude Learns Pokemon

A Pokemon Red speedrun learning agent system that discovers optimal strategies through parallel empirical experimentation. The system uses Claude AI for strategic planning and tactical script development, coordinating multiple pokemon-gym emulator instances for high-throughput learning.

## Overview

This project implements a sophisticated Pokemon Red speedrun optimization system with the following key components:

- **Parallel Execution**: Orchestrates 4 simultaneous Pokemon-gym emulator instances
- **AI-Driven Strategy**: Uses Claude Opus for strategic planning and Claude Sonnet for tactical execution
- **Checkpoint System**: Deterministic save/load system for reproducible experiments
- **DSL Compilation**: Domain-specific language for Pokemon gameplay automation
- **Performance Optimization**: Millisecond-level operation targeting for production-grade performance

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Claude Opus    │    │ EmulatorPool     │    │ CheckpointManager│
│  (Strategy)     │◄──►│ (4 Containers)   │◄──►│ (State Management)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ ScriptCompiler  │    │ PokemonGymClient │    │ TileObserver    │
│ (DSL→Inputs)    │    │ (HTTP Interface) │    │ (State Analysis)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Pokemon-gym Installation

### Prerequisites

1. **Docker**: Required for running pokemon-gym emulator containers
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install docker.io docker-compose
   sudo usermod -aG docker $USER  # Add user to docker group
   newgrp docker  # Refresh group membership

   # macOS (using Homebrew)
   brew install --cask docker

   # Verify installation
   docker --version
   ```

2. **Python 3.10+**: Required for the learning agent system
   ```bash
   python3 --version  # Should be 3.10 or higher
   ```

### Installing Pokemon-gym

1. **Clone benchflow-ai/pokemon-gym repository**:
   ```bash
   git clone https://github.com/benchflow-ai/pokemon-gym.git
   cd pokemon-gym
   ```

2. **Build the Docker image**:
   ```bash
   # Build the base pokemon-gym image
   docker build -t pokemon-gym:latest .

   # Verify the image was built successfully
   docker images | grep pokemon-gym
   ```

3. **Test single emulator instance**:
   ```bash
   # Run a test container
   docker run -d -p 8080:8080 --name test-pokemon-gym pokemon-gym:latest

   # Test HTTP endpoint
   curl http://localhost:8080/health

   # Cleanup test container
   docker stop test-pokemon-gym && docker rm test-pokemon-gym
   ```

### Installing Claude Learns Pokemon

1. **Clone this repository**:
   ```bash
   git clone <repository-url>
   cd claudelearnspokemon
   ```

2. **Run the setup script**:
   ```bash
   ./setup.sh
   ```

   This automatically:
   - Creates and activates virtual environment
   - Installs all dependencies
   - Sets up pre-commit hooks
   - Verifies environment correctness

3. **Verify installation**:
   ```bash
   python verify_setup.py
   ```

## Configuration

### EmulatorPool Configuration

The `EmulatorPool` class manages Docker containers for pokemon-gym instances:

```python
from claudelearnspokemon import EmulatorPool

# Basic configuration
pool = EmulatorPool(
    pool_size=4,                    # Number of parallel emulators
    base_port=8081,                # Starting port for sequential allocation
    image_name="pokemon-gym:latest", # Docker image name
    startup_timeout=30,             # Container startup timeout (seconds)
    default_timeout=None            # Default acquire timeout (None = block)
)
```

### CheckpointManager Configuration

The `CheckpointManager` handles save/load operations:

```python
from claudelearnspokemon import CheckpointManager

# Initialize with custom directory
checkpoint_mgr = CheckpointManager(
    checkpoint_dir="./my-checkpoints"  # Default: ~/.claudelearnspokemon/checkpoints
)
```

### Environment Variables

Optional environment variables for configuration:

```bash
# Checkpoint storage location
export CLAUDE_POKEMON_CHECKPOINT_DIR="/path/to/checkpoints"

# Docker configuration
export DOCKER_HOST="unix:///var/run/docker.sock"  # Docker daemon socket

# Performance tuning
export CLAUDE_POKEMON_MAX_CONTAINERS=8  # Override default pool size
```

## Quick Start

### 1. Initialize the System

```python
from claudelearnspokemon import EmulatorPool, CheckpointManager

# Initialize components
pool = EmulatorPool(pool_size=4)
pool.initialize()

# Verify all emulators are healthy
health_status = pool.health_check()
print(f"Pool status: {health_status}")
```

### 2. Execute a Simple Script

```python
# Execute basic Pokemon inputs
result = pool.execute_script("A A START B")
print(f"Execution result: {result}")

# Execute with checkpoint loading
result = pool.execute_script(
    script_text="MOVE RIGHT PRESS A",
    checkpoint_id="saved-game-state-uuid"
)
```

### 3. Use Context Manager for Safe Resource Handling

```python
# Automatic acquisition and release
with pool.acquire_emulator(timeout=30) as client:
    # Send inputs directly to emulator
    response = client.send_input("A B START")

    # Get current game state
    state = client.get_state()

    # Reset game if needed
    client.reset_game()
```

### 4. Checkpoint Management

```python
from claudelearnspokemon import CheckpointManager

checkpoint_mgr = CheckpointManager()

# Save game state
game_state = {"player": {"x": 100, "y": 150}, "items": ["potion"]}
metadata = {"location": "pallet_town", "progress": "started"}

checkpoint_id = checkpoint_mgr.save_checkpoint(game_state, metadata)
print(f"Saved checkpoint: {checkpoint_id}")

# Load checkpoint later
loaded_data = checkpoint_mgr.load_checkpoint(checkpoint_id)
restored_state = loaded_data["game_state"]
```

## Docker Container Management

### Manual Container Operations

```bash
# Start 4 pokemon-gym containers manually
for port in 8081 8082 8083 8084; do
    docker run -d \
        -p ${port}:8080 \
        --name pokemon-emulator-${port} \
        --restart on-failure:3 \
        --memory 512m \
        --cpus 1.0 \
        pokemon-gym:latest
done

# Check container status
docker ps --filter "name=pokemon-emulator"

# Stop all containers
docker stop $(docker ps -q --filter "name=pokemon-emulator")
```

### Container Health Monitoring

```python
# Check individual emulator health
health = pool.health_check()
for port, status in health["emulators"].items():
    if not status["healthy"]:
        print(f"Port {port} unhealthy: {status['error']}")

        # Restart unhealthy emulator
        pool.restart_emulator(int(port))
```

## Performance Requirements

The system is designed with specific performance targets:

- **Script compilation**: < 100ms
- **Checkpoint loading**: < 500ms
- **Tile observation**: < 50ms
- **Pattern queries**: < 100ms
- **Full execution cycle**: < 5 seconds

### Performance Monitoring

```python
# Monitor checkpoint manager performance
checkpoint_mgr = CheckpointManager()

# Performance stats are tracked automatically
save_times = checkpoint_mgr._save_times
load_times = checkpoint_mgr._load_times

avg_save = sum(save_times) / len(save_times) if save_times else 0
avg_load = sum(load_times) / len(load_times) if load_times else 0

print(f"Average save time: {avg_save:.3f}s")
print(f"Average load time: {avg_load:.3f}s")
```

## Development Workflow

### Project Structure

```
claudelearnspokemon/
├── src/claudelearnspokemon/     # Core source code
│   ├── emulator_pool.py         # Docker container management
│   ├── checkpoint_manager.py    # Save/load operations
│   ├── script_compiler.py       # DSL compilation
│   ├── tile_observer.py         # Game state analysis
│   └── dsl_ast.py              # DSL abstract syntax tree
├── tests/                       # Unit and integration tests
├── examples/                    # Usage examples
├── docs/                        # Documentation
├── scripts/                     # Utility scripts
└── checkpoints/                 # Default checkpoint storage
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific component tests
pytest tests/test_emulator_pool.py -v

# Run with coverage reporting
pytest --cov=claudelearnspokemon tests/

# Run performance benchmarks
pytest tests/test_performance.py -v --benchmark
```

### Using Worktrees for Development

This project uses git worktrees for parallel development:

```bash
# List available worktrees
ls /home/sd/worktrees/

# Navigate to issue-specific worktree
cd /home/sd/worktrees/issue-139

# Each worktree has its own environment
./setup.sh
```

## Troubleshooting

### Common Issues

1. **Docker Permission Denied**:
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

2. **Port Already in Use**:
   ```bash
   # Find process using port
   sudo lsof -i :8081

   # Kill process or change base_port
   pool = EmulatorPool(base_port=9081)
   ```

3. **Container Startup Timeout**:
   ```python
   # Increase timeout for slower systems
   pool = EmulatorPool(startup_timeout=60)
   ```

4. **Memory Issues**:
   ```bash
   # Check available memory
   free -h

   # Reduce pool size if needed
   pool = EmulatorPool(pool_size=2)
   ```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# EmulatorPool will now provide detailed logs
pool = EmulatorPool(pool_size=2)
pool.initialize()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run pre-commit hooks: `pre-commit run --all-files`
5. Submit pull request

## License

[License information to be added]

## See Also

- [API Documentation](docs/pokemon_gym_adapter.md) - Detailed API reference
- [Usage Examples](examples/pokemon_gym_usage.py) - Working code examples
- [Performance Tuning Guide](docs/performance_tuning.md) - Optimization tips
- [Troubleshooting Guide](docs/troubleshooting.md) - Common issues and solutions

---

For detailed API documentation and advanced usage patterns, see the [Pokemon Gym Adapter Documentation](docs/pokemon_gym_adapter.md).
