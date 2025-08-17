#!/usr/bin/env python3
"""
Pokemon Gym Adapter Usage Examples

This module demonstrates practical usage patterns for the Pokemon Gym adapter system,
including basic operations, advanced patterns, error handling, and performance optimization.

Run examples:
    python examples/pokemon_gym_usage.py --example basic
    python examples/pokemon_gym_usage.py --example checkpoints
    python examples/pokemon_gym_usage.py --example parallel
    python examples/pokemon_gym_usage.py --example all

Prerequisites:
    1. Docker running with pokemon-gym:latest image available
    2. Python environment with claudelearnspokemon installed
    3. Run: ./setup.sh to configure environment
"""

import argparse
import concurrent.futures
import logging
import time

from claudelearnspokemon import CheckpointManager, EmulatorPool, ExecutionResult


def setup_logging():
    """Configure logging for examples."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def example_basic_usage():
    """Demonstrate basic EmulatorPool operations."""
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)

    # Initialize pool with default settings
    pool = EmulatorPool(pool_size=2, startup_timeout=60)

    try:
        # Initialize container pool
        print("🚀 Initializing EmulatorPool...")
        pool.initialize()

        # Verify pool health
        health = pool.health_check()
        print(
            f"📊 Pool Status: {health['status']} ({health['healthy_count']}/{health['total_count']} healthy)"
        )

        # Basic script execution
        print("🎮 Executing basic script...")
        result = pool.execute_script("A A START B")

        if result.success:
            print(f"✅ Script executed successfully in {result.execution_time:.2f}s")
            print(f"📄 Output keys: {list(result.output.keys()) if result.output else 'None'}")
        else:
            print(f"❌ Script execution failed: {result.error}")

        # Manual client usage with context manager
        print("🔧 Using manual client management...")
        with pool.acquire_emulator(timeout=30) as client:
            print(f"📡 Acquired client on port {client.port}")

            # Test individual operations
            state = client.get_state()
            print(f"🎯 Current game state keys: {list(state.keys()) if state else 'None'}")

            # Send inputs step by step
            response1 = client.send_input("A")
            print(f"🎮 Sent 'A', response type: {type(response1)}")

            response2 = client.send_input("B START")
            print(f"🎮 Sent 'B START', response type: {type(response2)}")

            # Check if emulator is healthy
            is_healthy = client.is_healthy()
            print(f"💚 Emulator health: {'Healthy' if is_healthy else 'Unhealthy'}")

        print("✅ Basic usage example completed successfully!")

    except Exception as e:
        print(f"❌ Basic usage example failed: {e}")
        logging.exception("Basic usage example error details:")
    finally:
        print("🧹 Shutting down EmulatorPool...")
        pool.shutdown()


def example_checkpoint_management():
    """Demonstrate checkpoint save/load operations."""
    print("=" * 60)
    print("CHECKPOINT MANAGEMENT EXAMPLE")
    print("=" * 60)

    # Initialize components
    checkpoint_mgr = CheckpointManager()
    pool = EmulatorPool(pool_size=1, checkpoint_manager=checkpoint_mgr, startup_timeout=60)

    try:
        print("🚀 Initializing system...")
        pool.initialize()

        # Get initial game state
        print("📸 Capturing initial game state...")
        with pool.acquire_emulator() as client:
            initial_state = client.get_state()

        # Save initial checkpoint
        print("💾 Saving initial checkpoint...")
        checkpoint_id = checkpoint_mgr.save_checkpoint(
            game_state=initial_state,
            metadata={
                "description": "Game start state",
                "location": "initial",
                "progress": "beginning",
                "created_by": "usage_examples",
                "timestamp": time.time(),
            },
        )
        print(f"💾 Saved checkpoint: {checkpoint_id}")

        # Execute some actions to change state
        print("🎮 Executing actions to modify game state...")
        result = pool.execute_script("A WAIT 30 B WAIT 30 START")

        if result.success:
            # Get modified state
            with pool.acquire_emulator() as client:
                modified_state = client.get_state()

            # Save second checkpoint
            checkpoint_id_2 = checkpoint_mgr.save_checkpoint(
                game_state=modified_state,
                metadata={
                    "description": "After initial inputs",
                    "location": "modified",
                    "progress": "inputs_sent",
                    "created_by": "usage_examples",
                    "timestamp": time.time(),
                },
            )
            print(f"💾 Saved second checkpoint: {checkpoint_id_2}")

            # Demonstrate loading checkpoint
            print("📂 Loading first checkpoint...")
            loaded_data = checkpoint_mgr.load_checkpoint(checkpoint_id)
            loaded_metadata = loaded_data["metadata"]

            print("📂 Loaded checkpoint metadata:")
            for key, value in loaded_metadata.items():
                print(f"   {key}: {value}")

            # Execute script starting from checkpoint
            print("🎮 Executing script from checkpoint...")
            result_from_checkpoint = pool.execute_script(
                script_text="RIGHT RIGHT A", checkpoint_id=checkpoint_id
            )

            if result_from_checkpoint.success:
                print(
                    f"✅ Checkpoint execution completed in {result_from_checkpoint.execution_time:.2f}s"
                )
            else:
                print(f"❌ Checkpoint execution failed: {result_from_checkpoint.error}")

            # List all checkpoints
            print("📋 Listing all checkpoints...")
            checkpoints = checkpoint_mgr.list_checkpoints()
            for i, cp in enumerate(checkpoints, 1):
                print(f"   {i}. {cp['id'][:8]}... - {cp['metadata']['description']}")

        print("✅ Checkpoint management example completed!")

    except Exception as e:
        print(f"❌ Checkpoint management example failed: {e}")
        logging.exception("Checkpoint management error details:")
    finally:
        print("🧹 Shutting down...")
        pool.shutdown()


def example_parallel_execution():
    """Demonstrate parallel script execution across multiple emulators."""
    print("=" * 60)
    print("PARALLEL EXECUTION EXAMPLE")
    print("=" * 60)

    # Initialize pool with 4 emulators for parallel execution
    pool = EmulatorPool(pool_size=4, startup_timeout=60)

    try:
        print("🚀 Initializing pool with 4 emulators...")
        pool.initialize()

        # Define test scripts to run in parallel
        test_scripts = [
            "A A B START",
            "UP UP DOWN DOWN LEFT RIGHT LEFT RIGHT",
            "START SELECT A B A B",
            "A WAIT 60 B WAIT 60 START WAIT 30 SELECT",
        ]

        # Method 1: Using ThreadPoolExecutor for parallel execution
        print("🔄 Method 1: ThreadPoolExecutor parallel execution...")

        def execute_single_script(script_text):
            """Execute a single script and return results."""
            try:
                result = pool.execute_script(script_text)
                return {
                    "script": script_text,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "error": result.error,
                }
            except Exception as e:
                return {
                    "script": script_text,
                    "success": False,
                    "execution_time": None,
                    "error": str(e),
                }

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all scripts for parallel execution
            futures = []
            for script in test_scripts:
                future = executor.submit(execute_single_script, script)
                futures.append(future)

            # Collect results as they complete
            results = []
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                results.append(result)

                status = "✅" if result["success"] else "❌"
                time_str = f"{result['execution_time']:.2f}s" if result["execution_time"] else "N/A"
                print(f"   {status} Script {i}: {result['script'][:20]}... ({time_str})")

        total_parallel_time = time.time() - start_time
        successful_results = [r for r in results if r["success"]]

        print(f"🏁 Parallel execution completed in {total_parallel_time:.2f}s")
        print(f"📊 Success rate: {len(successful_results)}/{len(results)} scripts")

        if successful_results:
            avg_execution_time = sum(r["execution_time"] for r in successful_results) / len(
                successful_results
            )
            print(f"📊 Average script execution time: {avg_execution_time:.2f}s")

        # Method 2: Manual parallel execution with context managers
        print("\n🔄 Method 2: Manual context manager approach...")

        def execute_with_context_manager(script_text):
            """Execute script using context manager for resource control."""
            try:
                with pool.acquire_emulator(timeout=10) as client:
                    start = time.time()
                    response = client.send_input(script_text)
                    execution_time = time.time() - start

                    return {
                        "script": script_text,
                        "success": True,
                        "execution_time": execution_time,
                        "response_keys": list(response.keys()) if response else [],
                    }
            except Exception as e:
                return {"script": script_text, "success": False, "error": str(e)}

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(execute_with_context_manager, script)
                for script in test_scripts[:2]  # Just test 2 for brevity
            ]

            manual_results = [future.result() for future in futures]

        manual_time = time.time() - start_time
        manual_successful = [r for r in manual_results if r["success"]]

        print(f"🏁 Manual execution completed in {manual_time:.2f}s")
        print(f"📊 Success rate: {len(manual_successful)}/{len(manual_results)} scripts")

        # Pool status after parallel operations
        print("\n📊 Pool status after parallel execution:")
        status = pool.get_status()
        print(f"   Available: {status['available_count']}")
        print(f"   Busy: {status['busy_count']}")
        print(f"   Total: {status['total_count']}")
        print(f"   Status: {status['status']}")

        print("✅ Parallel execution example completed!")

    except Exception as e:
        print(f"❌ Parallel execution example failed: {e}")
        logging.exception("Parallel execution error details:")
    finally:
        print("🧹 Shutting down pool...")
        pool.shutdown()


def example_error_handling():
    """Demonstrate comprehensive error handling patterns."""
    print("=" * 60)
    print("ERROR HANDLING EXAMPLE")
    print("=" * 60)

    pool = EmulatorPool(pool_size=1, startup_timeout=60)

    try:
        print("🚀 Initializing for error handling demo...")
        pool.initialize()

        # Test 1: Invalid script execution
        print("🧪 Test 1: Handling invalid script execution...")
        result = pool.execute_script("")  # Empty script
        if not result.success:
            print(f"   ✅ Correctly handled empty script: {result.error}")

        # Test 2: Timeout handling
        print("🧪 Test 2: Testing acquisition timeout...")
        try:
            # Fill up all emulators
            clients = []
            for _ in range(pool.pool_size):
                clients.append(pool.acquire(timeout=5))

            # Now try to acquire one more (should timeout)
            pool.acquire(timeout=1)
            print("   ❌ Timeout test failed - should have timed out")
        except Exception as e:
            print(f"   ✅ Correctly handled timeout: {e}")
        finally:
            # Release all clients
            for client in clients:
                pool.release(client)

        # Test 3: Health check and recovery
        print("🧪 Test 3: Health monitoring...")
        health = pool.health_check()
        if health["status"] == "healthy":
            print("   ✅ All emulators healthy")
        else:
            print(f"   ⚠️ Pool degraded: {health}")

        # Test 4: Robust execution with retries
        print("🧪 Test 4: Robust execution pattern...")

        def robust_execute(script_text, max_retries=3):
            for attempt in range(max_retries):
                try:
                    result = pool.execute_script(script_text)
                    if result.success:
                        return result
                    else:
                        print(f"   Attempt {attempt + 1} failed: {result.error}")
                except Exception as e:
                    print(f"   Attempt {attempt + 1} exception: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry

            return ExecutionResult(
                success=False, output=None, error=f"All {max_retries} attempts failed"
            )

        robust_result = robust_execute("A B START")
        if robust_result.success:
            print(f"   ✅ Robust execution succeeded in {robust_result.execution_time:.2f}s")
        else:
            print(f"   ❌ Robust execution ultimately failed: {robust_result.error}")

        print("✅ Error handling example completed!")

    except Exception as e:
        print(f"❌ Error handling example failed: {e}")
        logging.exception("Error handling example error details:")
    finally:
        print("🧹 Shutting down...")
        pool.shutdown()


def example_performance_optimization():
    """Demonstrate performance optimization techniques."""
    print("=" * 60)
    print("PERFORMANCE OPTIMIZATION EXAMPLE")
    print("=" * 60)

    checkpoint_mgr = CheckpointManager()
    pool = EmulatorPool(
        pool_size=4, checkpoint_manager=checkpoint_mgr, default_timeout=30, startup_timeout=60
    )

    try:
        print("🚀 Initializing performance test environment...")
        pool.initialize()

        # Performance Test 1: Script compilation overhead
        print("⚡ Test 1: Script compilation performance...")

        test_script = "UP UP DOWN DOWN LEFT RIGHT LEFT RIGHT A B A B START SELECT"

        # Time individual compilation
        start_time = time.time()
        for _ in range(10):
            pool.compile_script(test_script)
        compilation_time = (time.time() - start_time) / 10

        print(f"   📊 Average compilation time: {compilation_time * 1000:.1f}ms")
        if compilation_time < 0.1:  # Target < 100ms
            print("   ✅ Compilation performance target met")
        else:
            print("   ⚠️ Compilation slower than target (100ms)")

        # Performance Test 2: Checkpoint operations
        print("⚡ Test 2: Checkpoint save/load performance...")

        # Create test state
        test_state = {
            "player": {"x": 100, "y": 200, "level": 5},
            "items": ["potion", "pokeball"] * 50,  # Make it substantial
            "progress": {"badges": 0, "pokedex": 5},
        }
        test_metadata = {"location": "test", "type": "performance_test"}

        # Time checkpoint save
        save_times = []
        for _i in range(5):
            start = time.time()
            checkpoint_id = checkpoint_mgr.save_checkpoint(test_state, test_metadata)
            save_time = time.time() - start
            save_times.append(save_time)

            # Clean up
            checkpoint_mgr.delete_checkpoint(checkpoint_id)

        avg_save_time = sum(save_times) / len(save_times)
        print(f"   📊 Average checkpoint save time: {avg_save_time * 1000:.1f}ms")

        if avg_save_time < 0.5:  # Target < 500ms
            print("   ✅ Checkpoint save performance target met")
        else:
            print("   ⚠️ Checkpoint save slower than target (500ms)")

        # Performance Test 3: Parallel execution efficiency
        print("⚡ Test 3: Parallel execution efficiency...")

        scripts = ["A B"] * 8  # 8 simple scripts

        # Sequential execution time
        start_sequential = time.time()
        sequential_results = []
        for script in scripts[:4]:  # Test with 4 scripts
            result = pool.execute_script(script)
            sequential_results.append(result)
        sequential_time = time.time() - start_sequential

        # Parallel execution time
        start_parallel = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            parallel_futures = [
                executor.submit(pool.execute_script, script) for script in scripts[:4]
            ]
            [f.result() for f in parallel_futures]
        parallel_time = time.time() - start_parallel

        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        efficiency = speedup / 4  # With 4 workers

        print(f"   📊 Sequential time: {sequential_time:.2f}s")
        print(f"   📊 Parallel time: {parallel_time:.2f}s")
        print(f"   📊 Speedup: {speedup:.2f}x")
        print(f"   📊 Parallel efficiency: {efficiency:.1%}")

        if speedup > 2.0:
            print("   ✅ Good parallel performance")
        else:
            print("   ⚠️ Parallel execution not optimal")

        # Performance Test 4: Resource utilization
        print("⚡ Test 4: Resource utilization monitoring...")

        # Monitor resource usage during operations
        pool.get_status()

        # Simulate workload
        workload_scripts = ["UP DOWN LEFT RIGHT A"] * 6

        start_workload = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            workload_futures = [
                executor.submit(pool.execute_script, script) for script in workload_scripts
            ]

            # Check status while work is happening
            time.sleep(0.1)  # Let some work start
            status_during = pool.get_status()

            # Wait for completion
            workload_results = [f.result() for f in workload_futures]

        workload_time = time.time() - start_workload
        pool.get_status()

        successful_workload = sum(1 for r in workload_results if r.success)
        throughput = successful_workload / workload_time

        print("   📊 Resource usage during workload:")
        print(f"      Available: {status_during['available_count']}/{status_during['total_count']}")
        print(f"      Busy: {status_during['busy_count']}/{status_during['total_count']}")
        print(f"   📊 Throughput: {throughput:.1f} scripts/second")
        print(f"   📊 Success rate: {successful_workload}/{len(workload_scripts)}")

        # Final health check
        final_health = pool.health_check()
        print(f"   📊 Final pool health: {final_health['status']}")

        print("✅ Performance optimization example completed!")

    except Exception as e:
        print(f"❌ Performance optimization example failed: {e}")
        logging.exception("Performance optimization error details:")
    finally:
        print("🧹 Shutting down...")
        pool.shutdown()


def example_advanced_patterns():
    """Demonstrate advanced usage patterns and best practices."""
    print("=" * 60)
    print("ADVANCED PATTERNS EXAMPLE")
    print("=" * 60)

    checkpoint_mgr = CheckpointManager()
    pool = EmulatorPool(pool_size=2, checkpoint_manager=checkpoint_mgr, startup_timeout=60)

    try:
        print("🚀 Initializing advanced patterns demo...")
        pool.initialize()

        # Pattern 1: State-based execution
        print("🧠 Pattern 1: State-based execution...")

        def execute_with_state_analysis(script_text, checkpoint_id=None):
            """Execute script and analyze resulting state for next actions."""
            result = pool.execute_script(script_text, checkpoint_id)

            if result.success and result.output:
                final_state = result.output.get("final_state", {})

                # Simulate state analysis
                analysis = {
                    "execution_time": result.execution_time,
                    "state_keys": list(final_state.keys()) if final_state else [],
                    "checkpoint_reached": result.checkpoint_reached,
                    "suggested_next_action": "continue",  # Would be more sophisticated
                }

                return {"result": result, "analysis": analysis}
            else:
                return {"result": result, "analysis": {"error": result.error}}

        analyzed_result = execute_with_state_analysis("A WAIT 30 B")
        print(f"   📊 State analysis keys: {analyzed_result['analysis'].get('state_keys', [])}")
        print(f"   📊 Execution time: {analyzed_result['analysis'].get('execution_time', 0):.2f}s")

        # Pattern 2: Checkpoint-based experiment chains
        print("🧠 Pattern 2: Checkpoint-based experiment chains...")

        # Create base checkpoint
        with pool.acquire_emulator() as client:
            base_state = client.get_state()

        base_checkpoint = checkpoint_mgr.save_checkpoint(
            game_state=base_state,
            metadata={
                "description": "Base state for experiments",
                "type": "experiment_base",
                "created_for": "advanced_patterns_demo",
            },
        )
        print(f"   💾 Created base checkpoint: {base_checkpoint[:8]}...")

        # Run experiment chain
        experiment_scripts = ["UP A", "DOWN B", "LEFT START"]

        experiment_results = []
        for i, script in enumerate(experiment_scripts, 1):
            print(f"   🧪 Running experiment {i}: {script}")

            # Each experiment starts from the same base state
            result = pool.execute_script(script, checkpoint_id=base_checkpoint)

            if result.success:
                # Save intermediate result as new checkpoint
                with pool.acquire_emulator() as client:
                    exp_state = client.get_state()

                exp_checkpoint = checkpoint_mgr.save_checkpoint(
                    game_state=exp_state,
                    metadata={
                        "description": f"Experiment {i} result",
                        "type": "experiment_result",
                        "base_checkpoint": base_checkpoint,
                        "script": script,
                    },
                )

                experiment_results.append(
                    {
                        "experiment": i,
                        "script": script,
                        "checkpoint": exp_checkpoint,
                        "execution_time": result.execution_time,
                    }
                )

                print(f"   ✅ Experiment {i} completed in {result.execution_time:.2f}s")
            else:
                print(f"   ❌ Experiment {i} failed: {result.error}")

        print(f"   📊 Completed {len(experiment_results)} successful experiments")

        # Pattern 3: Resource pooling with retry logic
        print("🧠 Pattern 3: Smart resource management...")

        class SmartResourceManager:
            def __init__(self, pool, max_retries=3):
                self.pool = pool
                self.max_retries = max_retries
                self.failed_ports = set()

            def smart_execute(self, script_text):
                for attempt in range(self.max_retries):
                    try:
                        # Try to get a client that hasn't failed recently
                        client = None
                        for _ in range(5):  # Try up to 5 times to get good client
                            candidate = self.pool.acquire(timeout=5)
                            if candidate.port not in self.failed_ports:
                                client = candidate
                                break
                            else:
                                self.pool.release(candidate)
                                time.sleep(0.5)

                        if not client:
                            client = self.pool.acquire(timeout=10)  # Fall back to any client

                        # Try execution
                        response = client.send_input(script_text)

                        # Success - remove from failed list
                        self.failed_ports.discard(client.port)

                        return {
                            "success": True,
                            "response": response,
                            "port": client.port,
                            "attempt": attempt + 1,
                        }

                    except Exception as e:
                        if client:
                            self.failed_ports.add(client.port)
                            self.pool.release(client)

                        if attempt == self.max_retries - 1:
                            return {"success": False, "error": str(e), "attempts": self.max_retries}

                        time.sleep(1)  # Wait before retry
                    finally:
                        if client:
                            self.pool.release(client)

        smart_manager = SmartResourceManager(pool)
        smart_result = smart_manager.smart_execute("A B START")

        if smart_result["success"]:
            print(
                f"   ✅ Smart execution succeeded on port {smart_result['port']} (attempt {smart_result['attempt']})"
            )
        else:
            print(f"   ❌ Smart execution failed after {smart_result['attempts']} attempts")

        # Cleanup experiment checkpoints
        print("🧹 Cleaning up experiment checkpoints...")
        for exp_result in experiment_results:
            checkpoint_mgr.delete_checkpoint(exp_result["checkpoint"])
        checkpoint_mgr.delete_checkpoint(base_checkpoint)

        print("✅ Advanced patterns example completed!")

    except Exception as e:
        print(f"❌ Advanced patterns example failed: {e}")
        logging.exception("Advanced patterns error details:")
    finally:
        print("🧹 Shutting down...")
        pool.shutdown()


def main():
    """Main function to run examples based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pokemon Gym Adapter Usage Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/pokemon_gym_usage.py --example basic
    python examples/pokemon_gym_usage.py --example checkpoints
    python examples/pokemon_gym_usage.py --example parallel
    python examples/pokemon_gym_usage.py --example error_handling
    python examples/pokemon_gym_usage.py --example performance
    python examples/pokemon_gym_usage.py --example advanced
    python examples/pokemon_gym_usage.py --example all
        """,
    )

    parser.add_argument(
        "--example",
        choices=[
            "basic",
            "checkpoints",
            "parallel",
            "error_handling",
            "performance",
            "advanced",
            "all",
        ],
        default="basic",
        help="Which example to run",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("🎮 Pokemon Gym Adapter Usage Examples")
    print("====================================")
    print(f"Running example: {args.example}")
    print()

    # Run selected example(s)
    examples = {
        "basic": example_basic_usage,
        "checkpoints": example_checkpoint_management,
        "parallel": example_parallel_execution,
        "error_handling": example_error_handling,
        "performance": example_performance_optimization,
        "advanced": example_advanced_patterns,
    }

    if args.example == "all":
        for name, func in examples.items():
            print(f"\n{'=' * 80}")
            print(f"RUNNING: {name.upper()}")
            print(f"{'=' * 80}")
            try:
                func()
            except KeyboardInterrupt:
                print(f"\n⚠️ Example {name} interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Example {name} failed: {e}")
                logging.exception(f"Example {name} failed with exception:")
            print(f"\n{'=' * 80}")
            print(f"COMPLETED: {name.upper()}")
            print(f"{'=' * 80}")
    else:
        examples[args.example]()

    print("\n🏁 Examples completed!")
    print("For more information, see:")
    print("  - README.md - Installation and basic usage")
    print("  - docs/pokemon_gym_adapter.md - Comprehensive API documentation")
    print("  - docs/troubleshooting.md - Common issues and solutions")


if __name__ == "__main__":
    main()
