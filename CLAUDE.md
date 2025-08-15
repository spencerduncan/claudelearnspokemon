# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Pokemon Red speedrun learning agent system that discovers optimal strategies through parallel empirical experimentation. The system uses Claude Opus for strategic planning and Claude Sonnet for tactical script development, all through the Claude Code CLI.

## Architecture Overview

The system consists of these core components that need to be implemented:

**Parallel Execution Layer**
- `ParallelExecutionCoordinator` - Orchestrates 4 parallel Pokemon-gym emulator instances
- `EmulatorPool` - Manages Docker containers running on sequential ports
- `ClaudeCodeManager` - Manages 1 Opus + 4 Sonnet CLI conversations

**Strategic Intelligence**
- `OpusStrategist` - High-level planning and pattern synthesis via Claude Opus
- `SonnetWorkerPool` - Pool of 4 Sonnet workers for tactical script development
- `ExperimentSelector` - Prioritizes and selects diverse parallel experiments

**Script Development**
- `ScriptCompiler` - Translates DSL to Pokemon-gym input sequences
- `PatternDiscovery` - Identifies reusable patterns from execution results

**State Management**
- `CheckpointManager` - Saves/loads game states for deterministic replay
- `TileObserver` - Captures and analyzes 20x18 tile grids
- `MemoryGraph` - Memgraph-based persistent storage for patterns and knowledge

**Lifecycle Management**
- `ConversationLifecycleManager` - Handles Claude conversation turn limits and compression

## Development Setup

**IMPORTANT: Always run `./setup.sh` first when starting work on this project!**

The project includes automatic environment configuration:

```bash
# Quick setup (run this first in every new Claude instance):
./setup.sh

# This automatically:
# - Creates and activates the virtual environment
# - Installs all dependencies
# - Sets up pre-commit hooks
# - Verifies the environment is correct
```

### Automatic Environment Activation

The project supports multiple ways to ensure the correct environment:

1. **Manual activation**: `source venv/bin/activate`
2. **Direnv (recommended)**: Automatically activates when you cd into the project
3. **Python scripts**: Import `activate_env` at the top of scripts
4. **Verification**: Run `python verify_setup.py` to check everything is correct

### Environment Setup Commands

```bash
# First-time setup or environment repair
./setup.sh

# Verify environment is correct
python verify_setup.py

# For direnv users (auto-activation on cd)
direnv allow  # After installing direnv
```

## Testing Strategy

Each component has detailed unit tests specified in the design document. When implementing:

1. Start with the test file for a component
2. Write tests based on the specifications in pokemon-speedrun-design-v2.md
3. Implement the component to pass tests
4. Integration tests should verify parallel execution coordination

Example test structure:
```python
# tests/test_claude_code_manager.py
def test_claude_code_manager_initializes_one_opus_four_sonnet_processes():
    # Verify 5 total Claude processes start correctly
    pass
```

## Key Implementation Patterns

**Parallel Execution Model**
- 4 simultaneous Pokemon-gym instances
- Non-blocking script development while previous scripts execute
- Aggregated results sent to Opus for pattern synthesis

**DSL Evolution**
- Start with primitive inputs (A, B, START, etc.)
- Discover patterns through gameplay
- Evolve language based on successful patterns
- Compile DSL to input sequences

**Checkpoint Strategy**
- Save states at strategic locations
- Maximum 100 checkpoints with automatic pruning
- Enable deterministic replay from any checkpoint

## Claude Code CLI Integration

All Claude interactions use the CLI, not API:
- Opus: 100-turn conversations for strategic planning
- Sonnet: 20-turn conversations for tactical tasks
- Automatic context compression at turn limits
- Persistent conversations with restart capability

## Performance Requirements

- Script compilation: < 100ms
- Checkpoint loading: < 500ms
- Tile observation: < 50ms
- Pattern queries: < 100ms
- Full execution cycle: < 5 seconds

## Pokemon-gym Integration

The system interfaces with Pokemon-gym emulators via Docker:
- 4 containers on sequential ports
- Thread-safe acquisition/release
- Automatic restart on failure
- Checkpoint isolation between instances

## Component Implementation Order

Recommended build sequence:
1. `EmulatorPool` - Get Pokemon-gym containers running
2. `CheckpointManager` - Enable save/load functionality
3. `ScriptCompiler` - Basic DSL to input translation
4. `ClaudeCodeManager` - Claude CLI integration
5. `ParallelExecutionCoordinator` - Wire up parallel execution
6. Add remaining components incrementally

## Common Commands

```bash
# Run all tests
pytest

# Run specific component tests
pytest tests/test_emulator_pool.py

# Start Pokemon-gym containers (when EmulatorPool is ready)
docker run -d -p 8081:8080 pokemon-gym
docker run -d -p 8082:8080 pokemon-gym
docker run -d -p 8083:8080 pokemon-gym
docker run -d -p 8084:8080 pokemon-gym

# Connect to Memgraph (when MemoryGraph is ready)
# Requires Memgraph installation first
```

## Current Status

Project is in initial implementation phase. The design specification (pokemon-speedrun-design-v2.md) contains detailed component interfaces and test cases. Begin by implementing core infrastructure components before moving to intelligent agents.
