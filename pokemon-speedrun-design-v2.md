# Pokemon Speedrun Learning Agent - Design Specification v2.0

## Executive Summary

This specification defines a learning agent system that masters Pokemon Red speedrunning through empirical discovery and pattern recognition. The system leverages parallel execution capabilities and Claude Code CLI integration to accelerate learning while operating within a Claude Max subscription. The agent develops its own domain-specific language through gameplay experience, discovering optimal routes and techniques through iterative script execution from saved checkpoints.

## System Architecture

### Core Architecture Principles

The system employs a three-layer architecture with parallel execution capabilities. The strategic layer utilizes Claude Opus through Claude Code CLI for high-level planning and pattern synthesis. The compilation layer translates an evolving domain-specific language into executable input sequences. The execution layer manages multiple Pokemon-gym emulator instances for parallel experimentation.

### Parallel Execution Model

The system operates with four parallel execution streams, each running an independent Pokemon-gym emulator instance. A single Claude Opus conversation provides strategic coordination across all parallel experiments, while four Claude Sonnet conversations handle tactical script development for each execution stream. This architecture enables the system to test multiple hypotheses simultaneously while maintaining unified strategic direction.

### Claude Code Integration

All Claude interactions occur through the Claude Code CLI using a Claude Max subscription, eliminating API costs. The system maintains persistent conversations with automatic context compression when approaching turn limits. Claude Opus handles strategic planning with a 100-turn conversation lifecycle, while Claude Sonnet workers operate with 20-turn cycles for tactical tasks.

## Component Specifications

### ClaudeCodeManager

**Purpose**: Manages lifecycle and communication for all Claude Code CLI conversations.

**Responsibilities**: The ClaudeCodeManager initializes and maintains one Opus conversation and four Sonnet conversations at system startup. It routes messages to appropriate conversations based on task type, tracks conversation turn counts, and handles context compression and restart when turn limits are reached. The manager ensures continuous availability of Claude resources for both strategic planning and tactical execution.

**Interface Methods**:
- `initialize()`: Starts all Claude conversations with appropriate system prompts
- `send_to_opus(message: str) -> str`: Sends strategic planning request to Opus
- `send_to_sonnet(worker_id: str, message: str) -> str`: Sends task to specific Sonnet worker
- `acquire_sonnet_worker() -> str`: Returns ID of available Sonnet worker
- `release_sonnet_worker(worker_id: str)`: Marks worker as available
- `restart_conversation(conversation_id: str)`: Compresses context and restarts conversation
- `shutdown()`: Gracefully terminates all conversations

**Unit Test Cases**:
- test_claude_code_manager_initializes_one_opus_four_sonnet_processes
- test_claude_code_manager_routes_strategic_tasks_to_opus
- test_claude_code_manager_distributes_tactical_tasks_across_sonnet_workers
- test_claude_code_manager_tracks_conversation_turn_counts
- test_claude_code_manager_compresses_context_at_turn_limit
- test_claude_code_manager_restarts_individual_conversations_without_affecting_others
- test_claude_code_manager_handles_worker_acquisition_when_all_busy
- test_claude_code_manager_gracefully_handles_conversation_failure

### ParallelExecutionCoordinator

**Purpose**: Orchestrates parallel script execution across multiple emulator instances.

**Responsibilities**: The coordinator manages the primary execution loop, coordinating between Claude strategic planning and parallel script execution. It maintains synchronization between Opus strategic decisions and Sonnet tactical implementations, aggregates results from parallel executions for pattern discovery, and ensures efficient resource utilization across all parallel streams.

**Interface Methods**:
- `run()`: Main execution loop coordinating all components
- `get_opus_strategy(game_state: dict) -> dict`: Requests strategic plan from Opus
- `develop_and_execute(experiment: dict, worker_id: str) -> dict`: Develops and executes single experiment
- `opus_analyze_results(results: list) -> list`: Sends parallel results to Opus for analysis
- `propagate_learnings(discoveries: list)`: Updates all systems with new discoveries
- `emergency_stop()`: Halts all parallel executions immediately

**Unit Test Cases**:
- test_parallel_coordinator_maintains_four_simultaneous_executions
- test_parallel_coordinator_aggregates_results_from_all_streams
- test_parallel_coordinator_routes_discoveries_to_memory_graph
- test_parallel_coordinator_handles_individual_stream_failure
- test_parallel_coordinator_synchronizes_checkpoint_usage
- test_parallel_coordinator_prevents_checkpoint_conflicts
- test_parallel_coordinator_balances_load_across_workers

### EmulatorPool

**Purpose**: Manages pool of Pokemon-gym emulator instances for parallel execution.

**Responsibilities**: The EmulatorPool maintains four Pokemon-gym Docker containers on sequential ports, tracks emulator availability and current assignments, provides thread-safe emulator acquisition and release, and monitors emulator health with automatic restart on failure.

**Interface Methods**:
- `initialize(pool_size: int)`: Starts specified number of emulator containers
- `acquire() -> PokemonGymClient`: Returns available emulator client
- `release(client: PokemonGymClient)`: Returns emulator to available pool
- `execute_script(client: PokemonGymClient, script: CompiledScript, checkpoint_id: str) -> ExecutionResult`: Executes script on specific emulator
- `health_check()`: Verifies all emulators are responsive
- `restart_emulator(port: int)`: Restarts specific emulator instance
- `shutdown()`: Stops all emulator containers

**Unit Test Cases**:
- test_emulator_pool_starts_containers_on_sequential_ports
- test_emulator_pool_tracks_emulator_availability
- test_emulator_pool_blocks_acquisition_when_all_busy
- test_emulator_pool_handles_concurrent_acquisition_requests
- test_emulator_pool_restarts_failed_emulator_automatically
- test_emulator_pool_maintains_checkpoint_isolation_between_instances
- test_emulator_pool_gracefully_shuts_down_all_containers

### OpusStrategist

**Purpose**: Encapsulates Claude Opus conversation for strategic planning.

**Responsibilities**: The OpusStrategist analyzes overall game progression to identify optimization opportunities, synthesizes learnings from parallel script executions, proposes language evolution based on discovered patterns, and generates specifications for parallel experiments. It maintains long-term context about successful strategies and patterns while managing its conversation lifecycle to maximize value from the Claude Max subscription.

**Interface Methods**:
- `request_strategy(game_state: dict, recent_results: list) -> dict`: Gets strategic direction
- `analyze_parallel_results(results: list) -> list`: Analyzes results from parallel trials
- `propose_language_evolution(patterns: list) -> dict`: Suggests DSL improvements
- `summarize_learnings() -> str`: Creates comprehensive summary for context compression
- `think_ahead(current_experiments: list) -> dict`: Plans contingencies while scripts execute

**Unit Test Cases**:
- test_opus_strategist_formats_game_state_for_context
- test_opus_strategist_compresses_results_summary_efficiently
- test_opus_strategist_parses_json_strategy_response
- test_opus_strategist_extracts_parallel_experiments_from_response
- test_opus_strategist_identifies_patterns_across_parallel_results
- test_opus_strategist_maintains_context_continuity_across_messages
- test_opus_strategist_generates_diverse_experiment_strategies

### SonnetWorkerPool

**Purpose**: Manages pool of Claude Sonnet conversations for parallel script development.

**Responsibilities**: The pool maintains four Sonnet conversation processes for tactical tasks, assigns script development tasks to available workers, develops executable DSL scripts from strategic objectives, and provides script refinements based on execution results. Each worker operates independently while sharing discovered patterns and successful techniques.

**Interface Methods**:
- `initialize(worker_count: int)`: Starts specified number of Sonnet workers
- `assign_task(task: dict) -> str`: Assigns task to available worker, returns worker ID
- `develop_script(worker_id: str, task: dict) -> dict`: Develops script for given task
- `analyze_result(worker_id: str, result: ExecutionResult) -> dict`: Gets refinement suggestions
- `get_worker_status(worker_id: str) -> dict`: Returns current worker state
- `restart_worker(worker_id: str)`: Restarts specific worker conversation

**Unit Test Cases**:
- test_sonnet_pool_initializes_specified_number_of_workers
- test_sonnet_pool_assigns_tasks_to_available_workers
- test_sonnet_pool_queues_tasks_when_all_workers_busy
- test_sonnet_pool_develops_valid_dsl_scripts
- test_sonnet_pool_maintains_worker_independence
- test_sonnet_pool_shares_discovered_patterns_across_workers
- test_sonnet_pool_handles_individual_worker_restart

### ScriptCompiler

**Purpose**: Translates evolving domain-specific language into executable input sequences.

**Responsibilities**: The compiler parses DSL scripts into abstract syntax trees, expands macros and language patterns into primitive inputs, handles conditional constructs based on probability models, maintains the current language specification, and evolves compilation rules based on discovered patterns.

**Interface Methods**:
- `compile(script_text: str) -> CompiledScript`: Transforms DSL to input sequence
- `register_pattern(name: str, expansion: list)`: Adds new language pattern
- `validate_syntax(script_text: str) -> bool`: Checks script syntax validity
- `get_language_spec() -> dict`: Returns current DSL specification
- `estimate_frames(script: CompiledScript) -> int`: Predicts execution time
- `add_observation_points(script: CompiledScript, density: float) -> CompiledScript`: Inserts tile capture points

**Unit Test Cases**:
- test_compiler_tokenizes_dsl_syntax_into_ast_nodes
- test_compiler_expands_macros_to_primitive_sequences
- test_compiler_handles_nested_macro_expansion
- test_compiler_detects_recursive_macro_definitions
- test_compiler_validates_parameter_types_in_patterns
- test_compiler_estimates_frame_count_accurately
- test_compiler_inserts_observation_points_at_specified_density
- test_compiler_maintains_compilation_determinism

### PatternDiscovery

**Purpose**: Identifies reusable patterns and strategies from execution results.

**Responsibilities**: Pattern discovery analyzes execution sequences to identify repeated successful subsequences, calculates pattern success rates and performance improvements, promotes consistent patterns to strategies, identifies combinable patterns for complex operations, and maintains pattern relationships in the memory graph.

**Interface Methods**:
- `analyze_execution(result: ExecutionResult) -> list`: Extracts patterns from single execution
- `find_common_patterns(results: list) -> list`: Identifies patterns across multiple executions
- `calculate_pattern_similarity(pattern1: dict, pattern2: dict) -> float`: Measures pattern similarity
- `promote_to_strategy(pattern: dict) -> dict`: Elevates pattern to strategy status
- `combine_patterns(patterns: list) -> dict`: Creates complex pattern from simple ones
- `evaluate_pattern_effectiveness(pattern: dict, context: dict) -> float`: Assesses pattern value

**Unit Test Cases**:
- test_pattern_discovery_identifies_repeated_subsequences
- test_pattern_discovery_calculates_success_correlation
- test_pattern_discovery_filters_by_minimum_occurrence
- test_pattern_discovery_generalizes_specific_values_to_parameters
- test_pattern_discovery_detects_conditional_branches
- test_pattern_discovery_measures_time_improvement
- test_pattern_discovery_combines_compatible_patterns
- test_pattern_discovery_maintains_pattern_provenance

### MemoryGraph

**Purpose**: Provides persistent storage for discovered patterns and game knowledge using Memgraph.

**Responsibilities**: The memory graph stores tile semantics learned through interaction, maintains script performance metrics and success rates, tracks pattern relationships and combinations, manages checkpoint network for starting positions, and provides efficient querying for pattern retrieval and analysis.

**Interface Methods**:
- `store_discovery(discovery: dict)`: Persists new pattern discovery
- `query_patterns(criteria: dict) -> list`: Retrieves patterns matching criteria
- `update_script_performance(script_id: str, result: dict)`: Updates success metrics
- `get_tile_properties(tile_id: str, map_context: str) -> dict`: Returns learned tile semantics
- `find_checkpoint_path(start: str, end: str) -> list`: Finds checkpoint sequence
- `get_failure_analysis(location: str) -> dict`: Returns common failure patterns
- `compact_patterns()`: Consolidates similar patterns

**Unit Test Cases**:
- test_memory_graph_stores_discovery_with_relationships
- test_memory_graph_queries_patterns_by_success_rate
- test_memory_graph_updates_script_metrics_incrementally
- test_memory_graph_maintains_tile_semantics_per_map
- test_memory_graph_finds_shortest_checkpoint_path
- test_memory_graph_aggregates_failure_patterns
- test_memory_graph_handles_transaction_rollback
- test_memory_graph_compacts_redundant_patterns

### TileObserver

**Purpose**: Captures and analyzes tile-based game state representations.

**Responsibilities**: The tile observer captures 20x18 tile grids from game state, converts tile identifiers to semantic interpretations, learns tile properties through collision and interaction, detects patterns and structures in tile arrangements, and maintains tile knowledge in the memory graph.

**Interface Methods**:
- `capture_tiles(game_state: dict) -> numpy.ndarray`: Gets current tile representation
- `analyze_tile_grid(tiles: numpy.ndarray) -> dict`: Converts tiles to semantic analysis
- `detect_patterns(tiles: numpy.ndarray, pattern: numpy.ndarray) -> list`: Finds pattern occurrences
- `learn_tile_properties(observations: list)`: Updates tile semantic knowledge
- `identify_npcs(tiles: numpy.ndarray) -> list`: Locates NPCs in tile grid
- `find_path(tiles: numpy.ndarray, start: tuple, end: tuple) -> list`: Calculates walkable path

**Unit Test Cases**:
- test_tile_observer_captures_correct_grid_dimensions
- test_tile_observer_identifies_player_position
- test_tile_observer_detects_npc_positions
- test_tile_observer_learns_solid_tiles_from_collisions
- test_tile_observer_identifies_walkable_paths
- test_tile_observer_detects_repeating_patterns
- test_tile_observer_maintains_map_specific_semantics
- test_tile_observer_handles_menu_overlay_tiles

### CheckpointManager

**Purpose**: Manages saved game states for deterministic replay and experimentation.

**Responsibilities**: The checkpoint manager saves and loads game states to disk with compression, maintains checkpoint metadata including location and game progress, prunes low-value checkpoints to manage storage, provides checkpoint discovery for strategic planning, and ensures checkpoint consistency across parallel executions.

**Interface Methods**:
- `save_checkpoint(game_state: dict, metadata: dict) -> str`: Saves state with metadata
- `load_checkpoint(checkpoint_id: str) -> dict`: Loads game state from disk
- `list_checkpoints(criteria: dict) -> list`: Returns checkpoints matching criteria
- `prune_checkpoints(max_count: int)`: Removes low-value checkpoints
- `get_checkpoint_metadata(checkpoint_id: str) -> dict`: Returns checkpoint information
- `validate_checkpoint(checkpoint_id: str) -> bool`: Verifies checkpoint integrity
- `find_nearest_checkpoint(location: str) -> str`: Finds closest checkpoint to location

**Unit Test Cases**:
- test_checkpoint_manager_saves_with_compression
- test_checkpoint_manager_loads_compressed_checkpoints
- test_checkpoint_manager_generates_unique_identifiers
- test_checkpoint_manager_stores_metadata_correctly
- test_checkpoint_manager_prunes_by_value_score
- test_checkpoint_manager_validates_checkpoint_integrity
- test_checkpoint_manager_handles_concurrent_access
- test_checkpoint_manager_finds_nearest_by_game_progress

### ConversationLifecycleManager

**Purpose**: Manages Claude conversation lifecycles to maximize value from subscription.

**Responsibilities**: The lifecycle manager tracks turn counts for all active conversations, triggers context compression at configurable thresholds, preserves critical discoveries during compression, coordinates conversation restarts without disrupting parallel execution, and optimizes conversation usage within subscription limits.

**Interface Methods**:
- `track_turn(conversation_id: str)`: Increments turn counter
- `should_compress(conversation_id: str) -> bool`: Checks if compression needed
- `compress_context(conversation_id: str) -> str`: Creates compressed summary
- `restart_with_context(conversation_id: str, context: str)`: Restarts with summary
- `get_conversation_stats() -> dict`: Returns usage statistics
- `optimize_turn_allocation()`: Adjusts turn limits based on usage patterns

**Unit Test Cases**:
- test_lifecycle_manager_tracks_individual_turn_counts
- test_lifecycle_manager_triggers_compression_at_threshold
- test_lifecycle_manager_preserves_discoveries_in_summary
- test_lifecycle_manager_restarts_without_message_loss
- test_lifecycle_manager_maintains_separate_thresholds_per_model
- test_lifecycle_manager_calculates_usage_statistics
- test_lifecycle_manager_adjusts_thresholds_dynamically

### ExperimentSelector

**Purpose**: Prioritizes and selects experiments for parallel execution.

**Responsibilities**: The experiment selector maintains a priority queue of potential experiments, generates variations of successful experiments, balances exploration and exploitation strategies, ensures diversity across parallel trials, and tracks experiment history to avoid repetition.

**Interface Methods**:
- `add_experiment(experiment: dict, priority: float)`: Adds prioritized experiment
- `select_next_experiments(count: int) -> list`: Selects experiments for parallel execution
- `generate_variations(base_experiment: dict) -> list`: Creates experiment variations
- `calculate_priority(experiment: dict) -> float`: Computes experiment priority
- `mark_completed(experiment_id: str, result: dict)`: Records experiment completion
- `get_experiment_history() -> list`: Returns completed experiments

**Unit Test Cases**:
- test_experiment_selector_maintains_priority_ordering
- test_experiment_selector_selects_diverse_experiments
- test_experiment_selector_generates_valid_variations
- test_experiment_selector_avoids_duplicate_experiments
- test_experiment_selector_balances_exploration_exploitation
- test_experiment_selector_tracks_completion_history
- test_experiment_selector_adjusts_priorities_based_on_results

## Data Flow Architecture

### Strategic Planning Flow

The system aggregates current game state, recent execution results, and discovered patterns into a structured request for Claude Opus. Opus analyzes this information and returns a strategic plan containing parallel experiment specifications and language evolution proposals. The system parses this response and creates individual script development tasks for distribution to Sonnet workers.

### Parallel Script Development

Each experiment specification from Opus is assigned to an available Sonnet worker through the ClaudeCodeManager. Sonnet workers operate independently, developing executable DSL scripts based on their assigned objectives. Completed scripts are compiled and queued for execution on available emulators. This parallel development ensures continuous script generation while previous scripts execute.

### Execution and Learning

Compiled scripts execute on the emulator pool with real-time tile observation capture. Execution results stream to both the originating Sonnet worker for tactical refinement and to Opus for strategic analysis. The PatternDiscovery component analyzes results to identify reusable patterns, which are stored in the memory graph. Successful patterns trigger language evolution proposals, creating an empirical learning loop.

### Context Compression and Continuity

As conversations approach turn limits, the ConversationLifecycleManager initiates context compression. Critical discoveries, successful patterns, and current objectives are preserved in a compressed summary. Conversations restart with this compressed context, maintaining continuity while staying within subscription limits.

## Performance Requirements

### Timing Constraints

Script compilation must complete within 100 milliseconds to maintain execution flow. Checkpoint loading should finish within 500 milliseconds to minimize emulator idle time. Tile observation capture must occur within 50 milliseconds to avoid frame drops. Pattern queries from the memory graph must return within 100 milliseconds. Each complete execution cycle should finish within 5 seconds.

### Concurrency Requirements

The system maintains four parallel execution streams continuously. Claude Opus processes strategic planning while scripts execute. Sonnet workers develop new scripts while previous scripts run. The memory graph handles concurrent reads from all components. Checkpoint access uses file locking to prevent corruption.

### Storage Constraints

The checkpoint library maintains a maximum of 100 checkpoints with automatic pruning. Pattern storage allows up to 1000 unique patterns with consolidation of similar patterns. Execution history keeps a rolling window of 500 recent executions. Tile semantic storage has no limit but uses consolidation to manage growth.

## Configuration Structure

The system configuration divides into several sections. Claude Code configuration specifies model usage with Opus for strategic tasks using 100-turn conversations and Sonnet for tactical tasks using 20-turn conversations across four workers. Execution configuration defines checkpoint library size, script timeout values, and observation density. Learning parameters set pattern success thresholds, language evolution rates, and experiment batch sizes. Memory graph configuration provides connection details and query timeout settings.

## Error Handling Strategy

The system implements comprehensive error handling across all components. Script failures trigger checkpoint rollback with detailed failure analysis. Compilation errors result in pattern refinement without system crashes. Network failures use exponential backoff retry logic. Graph inconsistencies are resolved through periodic validation and self-healing. Checkpoint corruption triggers automatic fallback to previous valid states. Conversation failures initiate automatic restart with compressed context.

## System Boundaries and Constraints

The system operates exclusively with Pokemon Red, focusing on any-percent completion optimization. It supports single-player gameplay only without multiplayer or trading features. Glitch discovery occurs opportunistically through normal gameplay exploration. The system assumes standard hardware timing without frame-perfect optimization. All learning occurs through empirical gameplay rather than theoretical modeling.

## Success Metrics

Learning metrics track the discovery rate of new patterns per 100 executions, script success rate percentages, optimization delta showing time improvement per iteration, and language complexity measured by unique DSL constructs created. Performance metrics monitor completion time for full game runs, segment times for individual sections, consistency of execution at each checkpoint, and parallel efficiency across execution streams. Knowledge metrics measure pattern library growth, pattern reuse frequency across different contexts, reduction in repeated failures, and depth of strategy abstraction achieved.

## Conclusion

This specification defines a comprehensive learning system that discovers optimal Pokemon speedrun strategies through parallel empirical experimentation. The integration of Claude Code CLI with parallel execution architecture enables rapid learning while operating within subscription constraints. The system's modular design, clear interfaces, and comprehensive testing ensure maintainable implementation. Through iterative discovery and pattern recognition, the agent progresses from basic gameplay to advanced speedrunning techniques, developing its own domain-specific language grounded in actual gameplay experience.
