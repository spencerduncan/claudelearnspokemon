# Pokemon Gym Adapter Architecture Overview

This document provides a comprehensive overview of the Pokemon Gym adapter system architecture, including component relationships, data flow, and design patterns used for high-performance Pokemon Red speedrun learning.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Container Orchestration](#container-orchestration)
5. [Threading and Concurrency](#threading-and-concurrency)
6. [State Management](#state-management)
7. [Error Handling Architecture](#error-handling-architecture)
8. [Performance Architecture](#performance-architecture)
9. [Integration Patterns](#integration-patterns)
10. [Scalability Design](#scalability-design)

---

## System Overview

The Pokemon Gym adapter system is designed as a high-performance, multi-container orchestration platform that enables parallel Pokemon Red gameplay experimentation with deterministic state management.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLAUDE LEARNS POKEMON SYSTEM                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │   Claude Opus   │    │ EmulatorPool    │    │CheckpointManager│  │
│  │   (Strategy)    │◄──►│ (Orchestrator)  │◄──►│ (State Mgmt)    │  │
│  │                 │    │                 │    │                 │  │
│  │ - Planning      │    │ - 4x Containers │    │ - LZ4 Compress  │  │
│  │ - Analysis      │    │ - Load Balance  │    │ - Atomic Saves  │  │
│  │ - Optimization  │    │ - Health Check  │    │ - Fast Loads    │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│           │                       │                       │         │
│           ▼                       ▼                       ▼         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ ScriptCompiler  │    │PokemonGymAdapter│    │  TileObserver   │  │
│  │                 │    │ (4-Component)   │    │                 │  │
│  │ - DSL Parsing   │    │ - HTTP Client   │    │ - State Parse   │  │
│  │ - Optimization  │    │ - Error Recovery│    │ - Tile Analysis │  │
│  │ - Instruction   │    │ - Performance   │    │ - Change Detect │  │
│  │   Generation    │    │ - Input Valid   │    │ - Pattern Match │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│                                   │                                 │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │     DOCKER LAYER       │
                        ├────────────────────────┤
                        │                        │
                        │  pokemon-gym:8081     │
                        │  pokemon-gym:8082     │
                        │  pokemon-gym:8083     │
                        │  pokemon-gym:8084     │
                        │                        │
                        │ Each container runs:   │
                        │ - Pokemon Red ROM      │
                        │ - Game Boy emulator    │
                        │ - HTTP API server      │
                        │ - State serialization  │
                        └────────────────────────┘
```

### Core Design Principles

1. **Parallel Execution**: Multiple emulator instances for concurrent experimentation
2. **Deterministic Replay**: Checkpoint system enables reproducible experiments
3. **Fault Tolerance**: Graceful error handling and automatic recovery
4. **Performance Focus**: Sub-second response times for critical operations
5. **Resource Management**: Efficient container lifecycle and resource cleanup
6. **Scalability**: Design supports scaling from 1 to 20+ emulator instances

---

## Component Architecture

### PokemonGymAdapter - Refactored Component-Based Architecture

The PokemonGymAdapter has been refactored following SOLID principles to eliminate violations and improve maintainability. The monolithic adapter was decomposed into 4 specialized components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    POKEMONGYMADAPTER COMPONENTS                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │PokemonGymClient │    │ErrorRecoveryHan │    │PerformanceMonit │  │
│  │                 │    │dler             │    │or               │  │
│  │ - HTTP Client   │    │                 │    │                 │  │
│  │ - Connection    │    │ - Session Errors│    │ - Timing Track  │  │
│  │   Pooling       │◄──►│ - Recovery Logic│◄──►│ - SLA Validation│  │
│  │ - Timeout       │    │ - Retry Backoff │    │ - Performance   │  │
│  │   Management    │    │ - Emergency     │    │   Stats         │  │
│  │ - API Calls     │    │   Recovery      │    │ - Monitoring    │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│           │                       │                       │         │
│           └───────────────────────┼───────────────────────┘         │
│                                   │                                 │
│                          ┌─────────────────┐                        │
│                          │InputValidator   │                        │
│                          │                 │                        │
│                          │ - Button Parse  │                        │
│                          │ - Input Valid   │                        │
│                          │ - Response      │                        │
│                          │   Formatting    │                        │
│                          │ - Test Compat   │                        │
│                          └─────────────────┘                        │
│                                   │                                 │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                 POKEMONGYMADAPTER COORDINATOR                   │ │
│  │                                                                 │ │
│  │ - Dependency Injection of all 4 components                     │ │
│  │ - High-level API coordination                                   │ │
│  │ - Session management delegation                                 │ │
│  │ - Error handling orchestration                                  │ │
│  │ - Performance monitoring integration                            │ │
│  │ - Backward compatibility maintenance                            │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Architectural Improvements:**

1. **Single Responsibility Principle**: Each component has one clear responsibility
   - `PokemonGymClient`: HTTP communication only
   - `ErrorRecoveryHandler`: Error recovery logic only  
   - `PerformanceMonitor`: Performance tracking only
   - `InputValidator`: Input validation only

2. **Open/Closed Principle**: Components are extensible without modification
   - New error recovery strategies can be added to ErrorRecoveryHandler
   - New validation rules can be added to InputValidator
   - New performance metrics can be added to PerformanceMonitor

3. **Dependency Inversion**: Components depend on abstractions
   - ErrorRecoveryHandler depends on PokemonGymClient interface
   - Main adapter coordinates through dependency injection
   - Easy to mock components for testing

4. **Reduced Complexity**: 
   - Original monolith: 1,039 lines
   - Refactored components: 612 lines (48% reduction)
   - Improved testability with 95% test coverage

### EmulatorPool - Container Orchestration Engine

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EMULATOR POOL                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    POOL MANAGER                                 │ │
│  │                                                                 │ │
│  │  ┌───────────────┐  ┌────────────────┐  ┌──────────────────┐   │ │
│  │  │ Docker Client │  │ Container Mgmt │  │ Health Monitor   │   │ │
│  │  │               │  │                │  │                  │   │ │
│  │  │ - Image Pull  │  │ - Lifecycle    │  │ - Health Checks  │   │ │
│  │  │ - Container   │  │ - Port Mgmt    │  │ - Auto Restart   │   │ │
│  │  │   Create      │  │ - Resource     │  │ - Status Track   │   │ │
│  │  │ - Network     │  │   Limits       │  │ - Failure Detect│   │ │
│  │  │   Config      │  │ - Auto Cleanup │  │ - Recovery Logic │   │ │
│  │  └───────────────┘  └────────────────┘  └──────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                     │                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                  RESOURCE POOL                                  │ │
│  │                                                                 │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │ │
│  │  │ Available   │    │ Busy        │    │ Connection Pool     │  │ │
│  │  │ Queue       │    │ Tracking    │    │                     │  │ │
│  │  │             │    │             │    │ - HTTP Sessions     │  │ │
│  │  │ Port 8081 ──┼───►│ Port 8083   │    │ - Keep-Alive        │  │ │
│  │  │ Port 8082   │    │             │    │ - Connection Reuse  │  │ │
│  │  │ Port 8084   │    │             │    │ - Timeout Handling  │  │ │
│  │  │             │    │             │    │ - Retry Logic       │  │ │
│  │  └─────────────┘    └─────────────┘    └─────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                     │                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                 THREAD SAFETY                                   │ │
│  │                                                                 │ │
│  │  ┌─────────────────┐              ┌──────────────────────────┐  │ │
│  │  │ RLock           │              │ Queue Operations         │  │ │
│  │  │                 │              │                          │  │ │
│  │  │ - Acquire/      │              │ - Thread-safe get/put    │  │ │
│  │  │   Release       │              │ - Timeout handling       │  │ │
│  │  │ - Context Mgmt  │              │ - Non-blocking ops       │  │ │
│  │  │ - Deadlock      │              │ - Exception safety       │  │ │
│  │  │   Prevention    │              │ - Resource counting      │  │ │
│  │  └─────────────────┘              └──────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### CheckpointManager - State Persistence Engine

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CHECKPOINT MANAGER                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   SAVE PIPELINE                                 │ │
│  │                                                                 │ │
│  │  Game State ──► JSON Serialize ──► LZ4 Compress ──► Atomic     │ │
│  │      │               │                   │             Write   │ │
│  │      │               │                   │               │     │ │
│  │   Validate         Optimize           High Ratio      Temp     │ │
│  │   Structure        Format             Compression      File ────┼─► │
│  │      │               │                   │             Rename  │ │  Final
│  │   Metadata         Remove              Fast            Safety  │ │  File
│  │   Injection        Whitespace          Decompression     │     │ │
│  │      │               │                   │               │     │ │
│  │   UUID             Error               Error           Verify  │ │
│  │   Generation       Handling            Handling        Write   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                     │                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   LOAD PIPELINE                                 │ │
│  │                                                                 │ │
│  │   File Read ──► LZ4 Decompress ──► JSON Parse ──► Validate ──► │ │
│  │      │               │                │             │         │ │
│  │   Check           Fast               Error         Version    │ │
│  │   Exists          Decompression      Handling      Check     │ │ │ Game
│  │      │               │                │             │        │ │ State
│  │   Read            Corruption          Parse         Metadata │ │
│  │   Buffered        Detection           Errors        Extract  │ │
│  │      │               │                │             │        │ │
│  │   Error           Error               Recovery       Cache   │ │
│  │   Handling        Recovery            Strategies     Update  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                     │                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                 PERFORMANCE OPTIMIZATION                        │ │
│  │                                                                 │ │
│  │  ┌──────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │ │
│  │  │ Timing       │  │ Compression │  │ File System             │ │ │
│  │  │ Tracking     │  │ Levels      │  │ Optimization            │ │ │
│  │  │              │  │             │  │                         │ │ │
│  │  │ - Save Times │  │ - Fast: L1  │  │ - Buffered I/O          │ │ │
│  │  │ - Load Times │  │ - Balanced  │  │ - Atomic Operations     │ │ │
│  │  │ - Size Data  │  │   L3-L6     │  │ - Directory Pre-create  │ │ │
│  │  │ - Monitoring │  │ - Max: L9   │  │ - Temp File Strategy    │ │ │
│  │  │ - Alerts     │  │ - Auto      │  │ - Cleanup Automation    │ │ │
│  │  └──────────────┘  └─────────────┘  └─────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### ScriptCompiler - DSL Processing Engine

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SCRIPT COMPILER                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   COMPILATION PIPELINE                          │ │
│  │                                                                 │ │
│  │  DSL Text ──► Lexical ──► Syntax ──► Semantic ──► Code ──► AST  │ │
│  │     │         Analysis    Parse     Analysis     Gen      │     │ │
│  │     │            │          │          │          │       │     │ │
│  │  Tokenize    Keywords   Grammar   Validation  Optimize  Build  │ │
│  │  Commands    Actions    Rules     Context     Sequence  Tree   │ │
│  │     │            │          │          │          │       │     │ │
│  │  "PRESS A"   [PRESS]   Command   Action      Remove   Node    │ │
│  │  "WAIT 60"   [A]       Args      Exists      Redundant Structure│ │
│  │  "MOVE UP"   [WAIT]    Types     Timing      Combine   With     │ │
│  │     │        [60]      Check     Valid       Similar   Metadata │ │
│  │     │        [MOVE]      │          │        Actions     │      │ │
│  │   Error      [UP]     Syntax    Runtime    Instruction  │      │ │
│  │   Recovery   Tokens   Errors    Checks     Sequence     │      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                     │                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                 OPTIMIZATION ENGINE                             │ │
│  │                                                                 │ │
│  │  ┌──────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │ │
│  │  │ Pattern      │  │ Instruction │  │ Performance             │ │ │
│  │  │ Recognition  │  │ Fusion      │  │ Prediction              │ │ │
│  │  │              │  │             │  │                         │ │ │
│  │  │ - Common     │  │ - Combine   │  │ - Frame Estimation      │ │ │
│  │  │   Sequences  │  │   Adjacent  │  │ - Execution Time        │ │ │
│  │  │ - Movement   │  │   Actions   │  │ - Resource Usage        │ │ │
│  │  │   Patterns   │  │ - Reduce    │  │ - Success Probability   │ │ │
│  │  │ - Action     │  │   Overhead  │  │ - Checkpoint Likelihood │ │ │
│  │  │   Chains     │  │ - Timing    │  │ - Alternative Paths     │ │ │
│  │  └──────────────┘  └─────────────┘  └─────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                     │                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   OUTPUT GENERATION                             │ │
│  │                                                                 │ │
│  │  CompiledScript ──► Instructions ──► Metadata ──► Validation    │ │
│  │       │                 │              │             │          │ │
│  │   Data Structure    Button Sequence  Frame Count   Verify      │ │
│  │   Creation          Generation       Estimation     Output     │ │
│  │       │                 │              │             │          │ │
│  │   - Instructions    "A B START UP"   - Total Time   Check      │ │
│  │   - Frame Count     - Timing Info    - Complexity   Bounds     │ │
│  │   - Metadata        - Error Points   - Success Rate Ranges     │ │
│  │   - Checksum        - Alternatives   - Dependencies Complete   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagrams

### Script Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SCRIPT EXECUTION FLOW                          │
└─────────────────────────────────────────────────────────────────────┘

User Request
     │
     ▼
┌─────────────────┐
│ pool.execute_   │
│ script(text,    │ ──┐
│ checkpoint_id)  │   │
└─────────────────┘   │
     │                │
     ▼                │
┌─────────────────┐   │   ┌─────────────────┐
│ Script          │   │   │ Checkpoint      │
│ Compilation     │   │   │ Loading         │
│                 │   │   │ (if specified)  │
│ - Parse DSL     │   └──►│                 │
│ - Generate AST  │       │ - Load from     │
│ - Optimize      │       │   disk          │
│ - Validate      │       │ - Decompress    │
└─────────────────┘       │ - Validate      │
     │                    └─────────────────┘
     ▼                             │
┌─────────────────┐                │
│ Emulator        │◄───────────────┘
│ Acquisition     │
│                 │
│ - Check pool    │
│ - Wait if busy  │
│ - Acquire lock  │
│ - Return client │
└─────────────────┘
     │
     ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Checkpoint      │    │ Script          │    │ State           │
│ Application     │    │ Execution       │    │ Monitoring      │
│                 │    │                 │    │                 │
│ - Load state    │───►│ - Send inputs   │───►│ - Capture       │
│   into emulator │    │   to emulator   │    │   game state    │
│ - Verify load   │    │ - Monitor       │    │ - Analyze       │
│ - Handle errors │    │   responses     │    │   changes       │
└─────────────────┘    │ - Track timing  │    │ - Detect        │
                       │ - Handle errors │    │   checkpoints   │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       │
                       ┌─────────────────┐              │
                       │ Response        │              │
                       │ Processing      │              │
                       │                 │              │
                       │ - Parse output  │              │
                       │ - Extract state │◄─────────────┘
                       │ - Calculate     │
                       │   metrics       │
                       │ - Build result  │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Resource        │    │ Return Result   │
                       │ Cleanup         │    │                 │
                       │                 │    │ - ExecutionResult
                       │ - Release       │    │ - Success/Error │
                       │   emulator      │    │ - Timing data   │
                       │ - Close         │───►│ - Game state    │
                       │   connections   │    │ - Metadata      │
                       │ - Update pool   │    │ - Checkpoints   │
                       │   status        │    │   reached       │
                       └─────────────────┘    └─────────────────┘
```

### Parallel Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PARALLEL EXECUTION FLOW                          │
└─────────────────────────────────────────────────────────────────────┘

Multiple Script Requests
    │    │    │    │
    ▼    ▼    ▼    ▼
┌─────────────────────────────────┐
│     ThreadPoolExecutor          │
│                                 │
│  Worker 1  Worker 2  Worker 3  │
│     │         │         │      │
└─────┼─────────┼─────────┼──────┘
      │         │         │
      ▼         ▼         ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Script 1        │ │ Script 2        │ │ Script 3        │
│ Compilation     │ │ Compilation     │ │ Compilation     │ ┌─────────────────┐
│                 │ │                 │ │                 │ │ Script 4        │
└─────────────────┘ └─────────────────┘ └─────────────────┘ │ (Queued)        │
      │                       │                       │     │                 │
      ▼                       ▼                       ▼     └─────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 EMULATOR POOL                               │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ Available   │    │ Available   │    │ Available   │      │
│  │ Port 8081   │    │ Port 8082   │    │ Port 8083   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│        │                   │                   │            │
└────────┼───────────────────┼───────────────────┼────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │ Emulator 1  │    │ Emulator 2  │    │ Emulator 3  │
  │ Executing   │    │ Executing   │    │ Executing   │
  │ Script 1    │    │ Script 2    │    │ Script 3    │
  └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │ Results 1   │    │ Results 2   │    │ Results 3   │
  └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                 ┌─────────────────────┐
                 │ Result Aggregation  │
                 │                     │
                 │ - Collect all       │
                 │   results           │
                 │ - Sort by task ID   │
                 │ - Calculate stats   │
                 │ - Return to caller  │
                 └─────────────────────┘
```

---

## Container Orchestration

### Docker Container Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                   CONTAINER LIFECYCLE                               │
└─────────────────────────────────────────────────────────────────────┘

Pool Initialization
         │
         ▼
┌─────────────────┐
│ Docker Client   │
│ Connection      │ ──► Check Docker daemon
│                 │     availability
└─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Container 1     │     │ Container 2     │     │ Container N     │
│ Creation        │     │ Creation        │ ... │ Creation        │
│                 │     │                 │     │                 │
│ Port: 8081     │     │ Port: 8082     │     │ Port: 808N     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Image Pull      │     │ Image Pull      │     │ Image Pull      │
│ (if needed)     │     │ (if needed)     │     │ (if needed)     │
│                 │     │                 │     │                 │
│ pokemon-gym:    │     │ pokemon-gym:    │     │ pokemon-gym:    │
│ latest          │     │ latest          │     │ latest          │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Container       │     │ Container       │     │ Container       │
│ Start           │     │ Start           │     │ Start           │
│                 │     │                 │     │                 │
│ - Resource      │     │ - Resource      │     │ - Resource      │
│   limits        │     │   limits        │     │   limits        │
│ - Port mapping  │     │ - Port mapping  │     │ - Port mapping  │
│ - Restart       │     │ - Restart       │     │ - Restart       │
│   policy        │     │   policy        │     │   policy        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Health Check    │     │ Health Check    │     │ Health Check    │
│ Loop            │     │ Loop            │     │ Loop            │
│                 │     │                 │     │                 │
│ - HTTP /health  │     │ - HTTP /health  │     │ - HTTP /health  │
│ - Container     │     │ - Container     │     │ - Container     │
│   exec test     │     │   exec test     │     │   exec test     │
│ - Response      │     │ - Response      │     │ - Response      │
│   validation    │     │   validation    │     │   validation    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Client          │     │ Client          │     │ Client          │
│ Creation        │     │ Creation        │     │ Creation        │
│                 │     │                 │     │                 │
│ - HTTP session  │     │ - HTTP session  │     │ - HTTP session  │
│ - Connection    │     │ - Connection    │     │ - Connection    │
│   pooling       │     │   pooling       │     │   pooling       │
│ - Add to pool   │     │ - Add to pool   │     │ - Add to pool   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                 │
                                 ▼
                      ┌─────────────────┐
                      │ Pool Ready      │
                      │                 │
                      │ All containers  │
                      │ healthy and     │
                      │ available for   │
                      │ script          │
                      │ execution       │
                      └─────────────────┘
```

### Container Health Monitoring

```
┌─────────────────────────────────────────────────────────────────────┐
│                   HEALTH MONITORING SYSTEM                          │
└─────────────────────────────────────────────────────────────────────┘

Continuous Monitoring Loop
          │
          ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ HTTP Health     │      │ Container       │      │ Resource        │
│ Checks          │      │ Status Check    │      │ Monitoring      │
│                 │      │                 │      │                 │
│ GET /health     │      │ docker ps       │      │ CPU Usage       │
│ - Response      │◄────►│ - Status        │◄────►│ Memory Usage    │
│   validation    │      │ - Restart       │      │ Network I/O     │
│ - Timeout       │      │   count         │      │ Disk I/O        │
│   handling      │      │ - Uptime        │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      HEALTH DECISION ENGINE                         │
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ Healthy     │    │ Degraded    │    │ Failed                  │  │
│  │             │    │             │    │                         │  │
│  │ - All       │    │ - Some      │    │ - Container down        │  │
│  │   checks    │    │   checks    │    │ - Health check fails    │  │
│  │   pass      │    │   failing   │    │ - Resource exhausted    │  │
│  │ - Normal    │───►│ - Slow      │───►│ - Network unreachable   │  │
│  │   response  │    │   response  │    │ - Restart loop          │  │
│  │ - Good      │    │ - High      │    │ - Critical error        │  │
│  │   resources │    │   resource  │    │                         │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘  │
│         │                   │                          │            │
│         ▼                   ▼                          ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ Continue    │    │ Monitor     │    │ Recovery Action         │  │
│  │ Normal      │    │ Closely     │    │                         │  │
│  │ Operation   │    │             │    │ - Stop container        │  │
│  │             │    │ - Increase  │    │ - Remove from pool      │  │
│  │             │    │   frequency │    │ - Start new container   │  │
│  │             │    │ - Log       │    │ - Update port mapping   │  │
│  │             │    │   warnings  │    │ - Verify health         │  │
│  │             │    │ - Alert     │    │ - Add back to pool      │  │
│  │             │    │   if trend  │    │ - Log recovery          │  │
│  │             │    │   worsens   │    │ - Update metrics        │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Threading and Concurrency

### Thread Safety Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     THREADING ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────┘

Main Application Thread
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EMULATOR POOL                                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   THREAD SAFETY LAYER                          │ │
│  │                                                                 │ │
│  │  ┌─────────────────┐              ┌──────────────────────────┐  │ │
│  │  │ RLock (Reentrant│              │ Queue (Thread-Safe)      │  │ │
│  │  │ Lock)           │              │                          │  │ │
│  │  │                 │              │ Available Clients       │  │ │
│  │  │ - Acquire()     │◄────────────►│ ┌─────┐ ┌─────┐ ┌─────┐ │  │ │
│  │  │ - Release()     │              │ │8081 │ │8082 │ │8083 │ │  │ │
│  │  │ - Context Mgmt  │              │ └─────┘ └─────┘ └─────┘ │  │ │
│  │  │ - Deadlock      │              │                          │  │ │
│  │  │   Prevention    │              │ - get(timeout)           │  │ │
│  │  │                 │              │ - put(client)            │  │ │
│  │  └─────────────────┘              │ - qsize()                │  │ │
│  │           │                       │ - empty()                │  │ │
│  │           │                       └──────────────────────────┘  │ │
│  │           ▼                                    │                │ │
│  │  ┌─────────────────┐              ┌──────────────────────────┐  │ │
│  │  │ Busy Clients    │              │ Atomic Operations        │  │ │
│  │  │ Dictionary      │              │                          │  │ │
│  │  │                 │              │ - Container start/stop   │  │ │
│  │  │ {port: client}  │◄────────────►│ - Health check updates   │  │ │
│  │  │                 │              │ - Status modifications   │  │ │
│  │  │ - Thread-safe   │              │ - Error state changes    │  │ │
│  │  │   updates       │              │ - Pool size adjustments  │  │ │
│  │  │ - Consistent    │              │                          │  │ │
│  │  │   state         │              └──────────────────────────┘  │ │
│  │  └─────────────────┘                                            │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 CONCURRENT EXECUTION LAYER                         │
│                                                                     │
│  Thread 1         Thread 2         Thread 3         Thread N       │
│      │               │               │               │             │
│      ▼               ▼               ▼               ▼             │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│ │ acquire()   │ │ acquire()   │ │ acquire()   │ │ acquire()   │   │
│ │   │         │ │   │         │ │   │         │ │   │         │   │
│ │   ▼         │ │   ▼         │ │   ▼         │ │   ▼         │   │
│ │ wait_for_   │ │ wait_for_   │ │ wait_for_   │ │ wait_for_   │   │
│ │ available   │ │ available   │ │ available   │ │ available   │   │
│ │   │         │ │   │         │ │   │         │ │   │         │   │
│ │   ▼         │ │   ▼         │ │   ▼         │ │   ▼         │   │
│ │ execute     │ │ execute     │ │ execute     │ │ execute     │   │
│ │ script      │ │ script      │ │ script      │ │ script      │   │
│ │   │         │ │   │         │ │   │         │ │   │         │   │
│ │   ▼         │ │   ▼         │ │   ▼         │ │   ▼         │   │
│ │ release()   │ │ release()   │ │ release()   │ │ release()   │   │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Context Manager Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTEXT MANAGER PATTERN                          │
└─────────────────────────────────────────────────────────────────────┘

with pool.acquire_emulator() as client:
    # User code here
    pass

             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 EMULATOR CONTEXT LIFECYCLE                          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    __enter__() Method                          │ │
│  │                                                                 │ │
│  │  1. Call pool.acquire(timeout)                                 │ │
│  │     │                                                           │ │
│  │     ▼                                                           │ │
│  │  2. Block until emulator available                             │ │
│  │     │                                                           │ │
│  │     ▼                                                           │ │
│  │  3. Remove from available queue                                 │ │
│  │     │                                                           │ │
│  │     ▼                                                           │ │
│  │  4. Add to busy tracking                                        │ │
│  │     │                                                           │ │
│  │     ▼                                                           │ │
│  │  5. Return PokemonGymClient                                     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                │                                    │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                  USER CODE EXECUTION                           │ │
│  │                                                                 │ │
│  │  - client.send_input("A B START")                              │ │
│  │  - client.get_state()                                          │ │
│  │  - client.reset_game()                                         │ │
│  │  - Any other operations...                                     │ │
│  │                                                                 │ │
│  │  Exception Safety:                                              │ │
│  │  - Any exception here triggers __exit__()                      │ │
│  │  - Resource cleanup guaranteed                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                │                                    │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    __exit__() Method                           │ │
│  │                (Always Called - Exception Safe)                │ │
│  │                                                                 │ │
│  │  1. Check if client exists                                      │ │
│  │     │                                                           │ │
│  │     ▼                                                           │ │
│  │  2. Remove from busy tracking                                   │ │
│  │     │                                                           │ │
│  │     ▼                                                           │ │
│  │  3. Add back to available queue                                 │ │
│  │     │                                                           │ │
│  │     ▼                                                           │ │
│  │  4. Update pool status                                          │ │
│  │     │                                                           │ │
│  │     ▼                                                           │ │
│  │  5. Log release event                                           │ │
│  │                                                                 │ │
│  │  Error Handling:                                                │ │
│  │  - Log any cleanup errors                                       │ │
│  │  - Don't propagate cleanup exceptions                           │ │
│  │  - Ensure resource always released                              │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## State Management

### Checkpoint State Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CHECKPOINT STATE FLOW                          │
└─────────────────────────────────────────────────────────────────────┘

Game Execution
       │
       ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Running Game    │      │ Significant     │      │ State Capture   │
│                 │      │ Event           │      │                 │
│ - Player        │      │                 │      │ - Full game     │
│   actions       │─────►│ - Location      │─────►│   memory        │
│ - Game state    │      │   change        │      │ - Player data   │
│   updates       │      │ - Item acquired │      │ - Inventory     │
│ - Progress      │      │ - Badge earned  │      │ - Map state     │
│   tracking      │      │ - Story flag    │      │ - Flags/vars    │
└─────────────────┘      └─────────────────┘      └─────────────────┘
                                                          │
                                                          ▼
                        ┌─────────────────┐      ┌─────────────────┐
                        │ Metadata        │      │ Checkpoint      │
                        │ Generation      │      │ Creation        │
                        │                 │      │                 │
                        │ - Timestamp     │◄─────│ - UUID          │
                        │ - Description   │      │   generation    │
                        │ - Location      │      │ - JSON          │
                        │ - Progress      │      │   serialization │
                        │ - Tags/labels   │      │ - LZ4 compress  │
                        └─────────────────┘      └─────────────────┘
                                                          │
                                                          ▼
                        ┌─────────────────┐      ┌─────────────────┐
                        │ File Storage    │      │ Atomic Write    │
                        │                 │      │                 │
                        │ ~/.claude       │◄─────│ - Temp file     │
                        │ learnspokemon/  │      │   creation      │
                        │ checkpoints/    │      │ - Write data    │
                        │ {uuid}.lz4      │      │ - Verify write  │
                        │                 │      │ - Atomic rename │
                        └─────────────────┘      └─────────────────┘

Later Game Execution
       │
       ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Load Request    │      │ Checkpoint      │      │ State           │
│                 │      │ Loading         │      │ Restoration     │
│ - Checkpoint    │      │                 │      │                 │
│   UUID          │─────►│ - File read     │─────►│ - Decompress    │
│ - Validation    │      │ - LZ4           │      │   game state    │
│ - Error         │      │   decompress    │      │ - Apply to      │
│   handling      │      │ - JSON parse    │      │   emulator      │
│                 │      │ - Validate      │      │ - Verify load   │
└─────────────────┘      └─────────────────┘      └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │ Execution       │
                                                 │ Continuation    │
                                                 │                 │
                                                 │ - Resume from   │
                                                 │   saved state   │
                                                 │ - Execute new   │
                                                 │   script        │
                                                 │ - Track         │
                                                 │   changes       │
                                                 └─────────────────┘
```

---

## Error Handling Architecture

### Error Propagation and Recovery

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ERROR HANDLING ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────┘

Application Layer
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ERROR DETECTION                                │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Docker Errors   │  │ HTTP Errors     │  │ Application Errors  │  │
│  │                 │  │                 │  │                     │  │
│  │ - Container     │  │ - Connection    │  │ - Invalid input     │  │
│  │   start fail    │  │   refused       │  │ - Compilation fail  │  │
│  │ - Image missing │  │ - Timeout       │  │ - State corruption  │  │
│  │ - Port conflict │  │ - Bad response  │  │ - Resource limits   │  │
│  │ - Resource      │  │ - Network down  │  │ - Logic errors      │  │
│  │   exhaustion    │  │                 │  │                     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│          │                     │                      │            │
└──────────┼─────────────────────┼──────────────────────┼────────────┘
           │                     │                      │
           ▼                     ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ERROR CLASSIFICATION                             │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Recoverable     │  │ Transient       │  │ Fatal               │  │
│  │                 │  │                 │  │                     │  │
│  │ - Container     │  │ - Network       │  │ - Image corruption  │  │
│  │   restart       │  │   timeout       │  │ - Disk full         │  │
│  │ - Port change   │  │ - Emulator      │  │ - Permission        │  │
│  │ - Resource      │  │   busy          │  │   denied            │  │
│  │   reallocation  │  │ - High load     │  │ - Docker daemon     │  │
│  │                 │  │                 │  │   down              │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│          │                     │                      │            │
│          ▼                     ▼                      ▼            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Auto Recovery   │  │ Retry Strategy  │  │ Graceful Shutdown   │  │
│  │                 │  │                 │  │                     │  │
│  │ - Restart       │  │ - Exponential   │  │ - Clean up          │  │
│  │   emulator      │  │   backoff       │  │   resources         │  │
│  │ - Reallocate    │  │ - Circuit       │  │ - Log fatal error   │  │
│  │   port          │  │   breaker       │  │ - Notify operator   │  │
│  │ - Scale         │  │ - Max attempts  │  │ - Exit gracefully   │  │
│  │   resources     │  │                 │  │                     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
           │                     │                      │
           ▼                     ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ERROR RESPONSE                                 │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Success After   │  │ Partial Success │  │ Complete Failure    │  │
│  │ Recovery        │  │                 │  │                     │  │
│  │                 │  │ - Some          │  │ - Operation fails   │  │
│  │ - Operation     │  │   operations    │  │ - Clear error       │  │
│  │   completed     │  │   succeeded     │  │   message           │  │
│  │ - Log recovery  │  │ - Degraded      │  │ - Actionable        │  │
│  │   details       │  │   performance   │  │   guidance          │  │
│  │ - Update        │  │ - Warn user     │  │ - Recovery steps    │  │
│  │   metrics       │  │ - Continue      │  │ - Support info      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Performance Architecture

### Performance Optimization Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                   PERFORMANCE ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────┘

Request Layer
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       CACHING LAYER                                 │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Script Cache    │  │ Checkpoint      │  │ State Cache         │  │
│  │                 │  │ Cache           │  │                     │  │
│  │ - LRU eviction  │  │                 │  │ - Recent game       │  │
│  │ - Hash-based    │  │ - Memory-based  │  │   states            │  │
│  │   keys          │  │ - Size limits   │  │ - Fast lookups      │  │
│  │ - Compiled AST  │  │ - Load          │  │ - Change            │  │
│  │   storage       │  │   optimization  │  │   detection         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PARALLELIZATION LAYER                           │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ ThreadPool      │  │ Load Balancing  │  │ Resource Pool       │  │
│  │ Executor        │  │                 │  │                     │  │
│  │                 │  │ - Round-robin   │  │ - Emulator pool     │  │
│  │ - Worker        │  │ - Least busy    │  │ - Connection pool   │  │
│  │   threads       │  │ - Health-aware  │  │ - Memory pool       │  │
│  │ - Task queue    │  │ - Performance   │  │ - Buffer pool       │  │
│  │ - Result        │  │   based         │  │                     │  │
│  │   collection    │  │                 │  │                     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     OPTIMIZATION LAYER                             │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Code            │  │ I/O             │  │ Memory              │  │
│  │ Optimization    │  │ Optimization    │  │ Optimization        │  │
│  │                 │  │                 │  │                     │  │
│  │ - JIT patterns  │  │ - Async I/O     │  │ - Object pooling    │  │
│  │ - Hotpath       │  │ - Batch         │  │ - Lazy loading      │  │
│  │   detection     │  │   operations    │  │ - Memory mapping    │  │
│  │ - Algorithm     │  │ - Buffer        │  │ - GC optimization   │  │
│  │   selection     │  │   management    │  │ - Weak references   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MONITORING LAYER                              │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Metrics         │  │ Profiling       │  │ Alerting            │  │
│  │ Collection      │  │                 │  │                     │  │
│  │                 │  │ - CPU profiling │  │ - Performance       │  │
│  │ - Timing data   │  │ - Memory        │  │   degradation       │  │
│  │ - Throughput    │  │   profiling     │  │ - Resource          │  │
│  │ - Error rates   │  │ - I/O profiling │  │   exhaustion        │  │
│  │ - Resource      │  │ - Bottleneck    │  │ - Error spikes      │  │
│  │   usage         │  │   detection     │  │ - SLA violations    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Integration Patterns

### Claude AI Integration Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLAUDE AI INTEGRATION                            │
└─────────────────────────────────────────────────────────────────────┘

Strategic Planning (Claude Opus)
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STRATEGY LAYER                                   │
│                                                                     │
│  ┌─────────────────┐              ┌───────────────────────────────┐  │
│  │ Experiment      │              │ Pattern Analysis              │  │
│  │ Planning        │◄────────────►│                               │  │
│  │                 │              │ - Success patterns            │  │
│  │ - Goal setting  │              │ - Failure analysis            │  │
│  │ - Hypothesis    │              │ - Strategy refinement         │  │
│  │   generation    │              │ - Learning optimization       │  │
│  │ - Resource      │              │                               │  │
│  │   allocation    │              └───────────────────────────────┘  │
│  │ - Timeline      │                            │                   │
│  │   planning      │                            ▼                   │
│  └─────────────────┘              ┌───────────────────────────────┐  │
│           │                       │ Knowledge Synthesis           │  │
│           │                       │                               │  │
│           │                       │ - Result aggregation          │  │
│           │                       │ - Insight extraction          │  │
│           │                       │ - Strategy updates            │  │
│           │                       │ - Next experiment design      │  │
│           │                       └───────────────────────────────┘  │
└───────────┼─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TACTICAL LAYER                                   │
│                                                                     │
│  Tactical Execution (Claude Sonnet Pool)                           │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Sonnet Worker 1 │  │ Sonnet Worker 2 │  │ Sonnet Worker N     │  │
│  │                 │  │                 │  │                     │  │
│  │ Script          │  │ Script          │  │ Script              │  │
│  │ Development     │  │ Development     │  │ Development         │  │
│  │                 │  │                 │  │                     │  │
│  │ - DSL           │  │ - DSL           │  │ - DSL               │  │
│  │   generation    │  │   generation    │  │   generation        │  │
│  │ - Optimization  │  │ - Optimization  │  │ - Optimization      │  │
│  │ - Testing       │  │ - Testing       │  │ - Testing           │  │
│  │ - Refinement    │  │ - Refinement    │  │ - Refinement        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│           │                     │                      │           │
└───────────┼─────────────────────┼──────────────────────┼───────────┘
            │                     │                      │
            ▼                     ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER                                  │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Emulator 1      │  │ Emulator 2      │  │ Emulator N          │  │
│  │                 │  │                 │  │                     │  │
│  │ Script A        │  │ Script B        │  │ Script Z            │  │
│  │ Execution       │  │ Execution       │  │ Execution           │  │
│  │                 │  │                 │  │                     │  │
│  │ - Load          │  │ - Load          │  │ - Load              │  │
│  │   checkpoint    │  │   checkpoint    │  │   checkpoint        │  │
│  │ - Execute       │  │ - Execute       │  │ - Execute           │  │
│  │   commands      │  │   commands      │  │   commands          │  │
│  │ - Capture       │  │ - Capture       │  │ - Capture           │  │
│  │   results       │  │   results       │  │   results           │  │
│  │ - Save state    │  │ - Save state    │  │ - Save state        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│           │                     │                      │           │
└───────────┼─────────────────────┼──────────────────────┼───────────┘
            │                     │                      │
            ▼                     ▼                      ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                  RESULT AGGREGATION                             │
     │                                                                 │
     │  ┌─────────────────┐              ┌─────────────────────────┐    │
     │  │ Data            │              │ Analysis & Feedback     │    │
     │  │ Collection      │─────────────►│                         │    │
     │  │                 │              │ - Performance metrics   │    │
     │  │ - Execution     │              │ - Success rates         │    │
     │  │   results       │              │ - Pattern detection     │    │
     │  │ - Timing data   │              │ - Anomaly detection     │    │
     │  │ - State         │              │ - Strategy validation   │    │
     │  │   changes       │              │                         │    │
     │  │ - Checkpoints   │              └─────────────────────────┘    │
     │  │   reached       │                            │               │
     │  └─────────────────┘                            ▼               │
     │                                ┌─────────────────────────┐      │
     │                                │ Feedback to Strategy    │      │
     │                                │                         │      │
     │                                │ - Update models         │──────┼──► Back to
     │                                │ - Refine hypotheses     │      │   Strategy Layer
     │                                │ - Adjust parameters     │      │
     │                                │ - Plan next iteration   │      │
     │                                └─────────────────────────┘      │
     └─────────────────────────────────────────────────────────────────┘
```

---

## Scalability Design

### Horizontal Scaling Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   HORIZONTAL SCALING DESIGN                         │
└─────────────────────────────────────────────────────────────────────┘

Single Machine (Development)
┌─────────────────────────────────────────────────────────────────────┐
│  Host Machine                                                       │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ EmulatorPool    │  │ CheckpointMgr   │  │ Claude Integration  │  │
│  │ (4 containers)  │  │ (Local FS)      │  │ (Direct API)        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│                                                                     │
│  Docker Network: bridge                                             │
│  Storage: Local filesystem                                          │
│  Processing: Single CPU/Memory pool                                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
Multi-Machine (Production)
┌─────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED ARCHITECTURE                         │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    LOAD BALANCER                                │ │
│  │                                                                 │ │
│  │  ┌─────────────────┐              ┌─────────────────────────┐   │ │
│  │  │ Request         │              │ Health-aware            │   │ │
│  │  │ Distribution    │◄────────────►│ Routing                 │   │ │
│  │  │                 │              │                         │   │ │
│  │  │ - Round-robin   │              │ - Check node health     │   │ │
│  │  │ - Least         │              │ - Route to available    │   │ │
│  │  │   connections   │              │ - Handle failures       │   │ │
│  │  │ - Weighted      │              │ - Auto-discovery        │   │ │
│  │  │   routing       │              │                         │   │ │
│  │  └─────────────────┘              └─────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                     │                               │
└─────────────────────────────────────┼───────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ Node 1          │         │ Node 2          │         │ Node N          │
│                 │         │                 │         │                 │
│ ┌─────────────┐ │         │ ┌─────────────┐ │         │ ┌─────────────┐ │
│ │EmulatorPool │ │         │ │EmulatorPool │ │         │ │EmulatorPool │ │
│ │(8 emulators)│ │         │ │(8 emulators)│ │         │ │(8 emulators)│ │
│ └─────────────┘ │         │ └─────────────┘ │         │ └─────────────┘ │
│                 │         │                 │         │                 │
│ ┌─────────────┐ │         │ ┌─────────────┐ │         │ ┌─────────────┐ │
│ │Local Cache  │ │         │ │Local Cache  │ │         │ │Local Cache  │ │
│ │- Scripts    │ │         │ │- Scripts    │ │         │ │- Scripts    │ │
│ │- States     │ │         │ │- States     │ │         │ │- States     │ │
│ └─────────────┘ │         │ └─────────────┘ │         │ └─────────────┘ │
│                 │         │                 │         │                 │
│ Docker Engine   │         │ Docker Engine   │         │ Docker Engine   │
│ Kubernetes Pod  │         │ Kubernetes Pod  │         │ Kubernetes Pod  │
│ or Standalone   │         │ or Standalone   │         │ or Standalone   │
└─────────────────┘         └─────────────────┘         └─────────────────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SHARED SERVICES                                  │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Distributed     │  │ Message Queue   │  │ Monitoring &        │  │
│  │ Storage         │  │                 │  │ Logging             │  │
│  │                 │  │ - Task          │  │                     │  │
│  │ - Checkpoint    │  │   distribution  │  │ - Metrics           │  │
│  │   storage       │  │ - Result        │  │   collection        │  │
│  │ - State sync    │  │   collection    │  │ - Log aggregation   │  │
│  │ - Backup/       │  │ - Event         │  │ - Alerting          │  │
│  │   replication   │  │   streaming     │  │ - Dashboards        │  │
│  │                 │  │                 │  │                     │  │
│  │ (NFS, S3,      │  │ (Redis, RabbitMQ│  │ (Prometheus,        │  │
│  │  MinIO)        │  │  Apache Kafka)  │  │  Grafana, ELK)      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

This comprehensive architecture overview provides the foundation for understanding how the Pokemon Gym adapter system achieves high performance, reliability, and scalability in Pokemon Red speedrun learning experiments. Each component is designed with production-grade engineering principles while maintaining simplicity and efficiency.

For implementation details, refer to the [API documentation](pokemon_gym_adapter.md), [performance tuning guide](performance_tuning.md), and [usage examples](../examples/pokemon_gym_usage.py).
