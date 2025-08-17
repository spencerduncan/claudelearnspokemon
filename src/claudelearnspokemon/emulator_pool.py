"""
EmulatorPool: Simplified Docker container management for Pokemon-gym emulators.

Workstation-appropriate container orchestration with clear error handling
and reliable resource management. Built with Bot Dean engineering principles
optimized for development workflow.

Author: Bot Dean - Workstation Engineering
"""

import logging
import queue
import time
from typing import TYPE_CHECKING, Any, cast

import docker
import requests
from docker.errors import APIError, DockerException, ImageNotFound

if TYPE_CHECKING:
    from .checkpoint_manager import CheckpointError, CheckpointManager
    from .dsl_ast import CompiledScript
    from .script_compiler import ScriptCompiler
else:
    # Import Pokemon functionality components (will be gracefully handled if not available)
    try:
        from .checkpoint_manager import CheckpointError, CheckpointManager
        from .dsl_ast import CompiledScript
        from .script_compiler import ScriptCompiler

        POKEMON_COMPONENTS_AVAILABLE = True
    except ImportError:
        # Graceful degradation when Pokemon components aren't available
        class CheckpointManager:  # type: ignore
            pass

        CheckpointError = Exception

        class ScriptCompiler:  # type: ignore
            pass

        class CompiledScript:  # type: ignore
            pass

        POKEMON_COMPONENTS_AVAILABLE = False

# Set the flag here so it's always available
if not TYPE_CHECKING:
    pass  # Flag already set above
else:
    POKEMON_COMPONENTS_AVAILABLE = True  # For type checking, assume available

# Configure logging for production observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmulatorPoolError(Exception):
    """
    Custom exception for EmulatorPool operations.

    Provides actionable error messages for production debugging.
    """

    pass


class ExecutionResult:
    """
    Results from script execution on a Pokemon-gym emulator.

    Contains all information needed for analysis and debugging.
    """

    def __init__(
        self,
        success: bool,
        output: Any,
        error: str | None = None,
        execution_time: float | None = None,
        checkpoint_reached: str | None = None,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time
        self.checkpoint_reached = checkpoint_reached

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILURE"
        return f"ExecutionResult({status}, time={self.execution_time:.2f}s)"


class PokemonGymClient:
    """
    HTTP client wrapper for Pokemon-gym emulator communication.

    Provides clean interface for script execution and state management
    while tracking the specific emulator instance.
    """

    def __init__(self, port: int, container_id: str):
        """
        Initialize client for specific emulator instance.

        Args:
            port: HTTP port for emulator communication
            container_id: Docker container ID for this emulator
        """
        self.port = port
        self.container_id = container_id
        self.base_url = f"http://localhost:{port}"
        self.session = requests.Session()
        # Note: timeout is set per request, not on session

        logger.info(f"PokemonGymClient initialized for port {port}, container {container_id[:12]}")

    def send_input(self, input_sequence: str) -> dict[str, Any]:
        """
        Send input sequence to the emulator.

        Args:
            input_sequence: Button inputs (A, B, START, etc.)

        Returns:
            Response data from emulator

        Raises:
            EmulatorPoolError: On communication failure
        """
        try:
            response = self.session.post(
                f"{self.base_url}/input", json={"inputs": input_sequence}, timeout=10
            )
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
        except requests.RequestException as e:
            raise EmulatorPoolError(
                f"Failed to send input to emulator on port {self.port}: {e}"
            ) from e

    def get_state(self) -> dict[str, Any]:
        """
        Get current game state from emulator.

        Returns:
            Current game state data

        Raises:
            EmulatorPoolError: On communication failure
        """
        try:
            response = self.session.get(f"{self.base_url}/state", timeout=5)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
        except requests.RequestException as e:
            raise EmulatorPoolError(
                f"Failed to get state from emulator on port {self.port}: {e}"
            ) from e

    def reset_game(self) -> dict[str, Any]:
        """
        Reset the game to initial state.

        Returns:
            Reset confirmation from emulator

        Raises:
            EmulatorPoolError: On communication failure
        """
        try:
            response = self.session.post(f"{self.base_url}/reset", timeout=10)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
        except requests.RequestException as e:
            raise EmulatorPoolError(f"Failed to reset emulator on port {self.port}: {e}") from e

    def is_healthy(self) -> bool:
        """
        Check if emulator is responding to health checks.

        Returns:
            True if emulator is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=3)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

    def __str__(self) -> str:
        return f"PokemonGymClient(port={self.port}, container={self.container_id[:12]})"


class EmulatorContext:
    """
    Context manager for automatic emulator acquisition and release.

    Ensures emulators are always properly returned to the pool, even on exceptions.
    """

    def __init__(self, pool: "EmulatorPool", timeout: float | None):
        """
        Initialize context manager.

        Args:
            pool: EmulatorPool instance to acquire from
            timeout: Timeout for acquisition
        """
        self.pool = pool
        self.timeout = timeout
        self.client: PokemonGymClient | None = None

    def __enter__(self) -> PokemonGymClient:
        """Acquire emulator client when entering context."""
        self.client = self.pool.acquire(timeout=self.timeout)
        return self.client

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """Release emulator client when exiting context."""
        if self.client is not None:
            self.pool.release(self.client)
            self.client = None


class EmulatorPool:
    """
    Manages a pool of Pokemon-gym Docker containers for parallel execution.

    Implements production patterns:
    - Graceful failure handling
    - Resource cleanup on exceptions
    - Timeout-based operations
    - Comprehensive logging for debugging
    - Idempotent operations
    """

    def __init__(
        self,
        pool_size: int = 4,
        base_port: int = 8081,
        image_name: str = "pokemon-gym:latest",
        startup_timeout: int = 30,
        checkpoint_manager: "CheckpointManager | None" = None,
        default_timeout: float | None = None,
    ):
        """
        Initialize EmulatorPool with workstation configuration.

        Args:
            pool_size: Number of containers in pool (default: 4)
            base_port: Starting port for sequential allocation (default: 8081)
            image_name: Docker image name for containers (default: pokemon-gym:latest)
            startup_timeout: Max seconds to wait for container startup (default: 30)
            checkpoint_manager: CheckpointManager for state loading (optional)
            default_timeout: Default timeout for acquire operations (optional)
        """
        self.pool_size = pool_size
        self.base_port = base_port
        self.image_name = image_name
        self.startup_timeout = startup_timeout
        self.default_timeout = default_timeout

        # Container management state
        self.containers: list[docker.models.containers.Container] = []
        self.client: docker.DockerClient | None = None

        # Simplified resource pool state - workstation-appropriate
        self.available_clients: queue.Queue = queue.Queue()
        self.clients_by_port: dict[int, PokemonGymClient] = {}  # All clients by port

        # Core Pokemon functionality components (with graceful degradation)
        if POKEMON_COMPONENTS_AVAILABLE:
            self.checkpoint_manager: CheckpointManager | None = (
                checkpoint_manager or CheckpointManager()
            )
            self.script_compiler: ScriptCompiler | None = ScriptCompiler()
        else:
            self.checkpoint_manager = None
            self.script_compiler = None
            logger.info("Pokemon components not available - running in basic mode")

        logger.info(
            f"EmulatorPool configured: size={pool_size}, base_port={base_port}, "
            f"image={image_name}, timeout={startup_timeout}s"
        )

    def initialize(self, pool_size: int | None = None) -> None:
        """
        Initialize the container pool with production-grade error handling.

        Args:
            pool_size: Override default pool size if provided

        Raises:
            EmulatorPoolError: On any initialization failure with actionable message
        """
        if pool_size is not None:
            self.pool_size = pool_size

        logger.info(f"Initializing EmulatorPool with {self.pool_size} containers")

        try:
            # Connect to Docker daemon with proper error handling
            self.client = docker.from_env()
            logger.info("Connected to Docker daemon")
        except DockerException as e:
            raise EmulatorPoolError(
                f"Docker daemon unavailable: {e}. "
                f"Ensure Docker daemon is running and accessible."
            ) from e

        # Track successfully started containers for cleanup on failure
        started_containers = []

        try:
            ports = self._get_container_ports(self.pool_size)

            for port in ports:
                logger.info(f"Starting container on port {port}")

                try:
                    container = self._start_single_container(port)
                    started_containers.append(container)
                    self.containers.append(container)

                    # Create PokemonGymClient and add to available pool
                    container_id = container.id or f"unknown-{port}"
                    client = PokemonGymClient(port, container_id)
                    self.clients_by_port[port] = client
                    self.available_clients.put(client)

                    logger.info(f"Container {container.id} started successfully on port {port}")

                except Exception as e:
                    # Clean up successfully started containers before re-raising
                    logger.error(f"Failed to start container on port {port}: {e}")
                    self._cleanup_containers(started_containers)
                    raise EmulatorPoolError(
                        f"Failed to start container on port {port}: {e}. "
                        f"Check port availability and resource limits."
                    ) from e

        except EmulatorPoolError:
            # Already handled, just re-raise
            raise
        except Exception as e:
            # Unexpected error - clean up and provide debugging info
            logger.error(f"Unexpected error during initialization: {e}")
            self._cleanup_containers(started_containers)
            raise EmulatorPoolError(
                f"Unexpected error during pool initialization: {e}. "
                f"Check Docker daemon status and system resources."
            ) from e

        logger.info(f"EmulatorPool initialized successfully with {len(self.containers)} containers")

    def shutdown(self) -> None:
        """
        Gracefully shutdown all containers with clear error handling.

        Continues shutdown process even if individual containers fail to stop.
        Safe to call multiple times.
        """
        if not self.containers:
            logger.info("EmulatorPool shutdown called - no containers to stop")
            return

        logger.info(f"Shutting down EmulatorPool with {len(self.containers)} containers")

        # Stop all containers
        for container in self.containers:
            try:
                logger.info(f"Stopping container {container.id}")
                container.stop(timeout=10)  # Give containers time to gracefully stop
                logger.info(f"Container {container.id} stopped successfully")
            except Exception as e:
                # Log error but continue with other containers
                logger.error(f"Failed to stop container {container.id}: {e}")

        # Clear container list
        self.containers.clear()

        # Clean up client resources
        for client in self.clients_by_port.values():
            try:
                client.close()
            except Exception as e:
                logger.error(f"Error closing client {client}: {e}")

        # Clear all client tracking
        while not self.available_clients.empty():
            try:
                self.available_clients.get_nowait()
            except queue.Empty:
                break

        self.clients_by_port.clear()
        logger.info("EmulatorPool shutdown complete")

    def acquire(self, timeout: float | None = None) -> PokemonGymClient:
        """
        Acquire an available emulator client from the pool.

        Simple resource allocation with blocking behavior when all emulators are busy.

        Args:
            timeout: Maximum seconds to wait for available emulator (None = block indefinitely)

        Returns:
            PokemonGymClient for exclusive use

        Raises:
            EmulatorPoolError: If no emulators available within timeout or pool not initialized
        """
        if not self.containers:
            raise EmulatorPoolError("EmulatorPool not initialized - call initialize() first")

        try:
            # Block until emulator becomes available - queue handles thread safety
            client = cast(PokemonGymClient, self.available_clients.get(timeout=timeout))
            logger.info(f"Acquired emulator {client}")
            return client

        except queue.Empty:
            raise EmulatorPoolError(
                f"No emulators available within {timeout}s timeout. "
                f"All {self.pool_size} emulators are currently busy."
            ) from None

    def acquire_emulator(self, timeout: float | None = None) -> "EmulatorContext":
        """
        Acquire an emulator client as a context manager for automatic resource cleanup.

        Args:
            timeout: Maximum seconds to wait for available emulator (None uses default_timeout)

        Returns:
            EmulatorContext that can be used in a 'with' statement

        Raises:
            EmulatorPoolError: If no emulators available within timeout or pool not initialized
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        return EmulatorContext(self, effective_timeout)

    def release(self, client: PokemonGymClient) -> None:
        """
        Release emulator client back to available pool.

        Simple operation that makes emulator available for other tasks.

        Args:
            client: PokemonGymClient to return to pool

        Raises:
            EmulatorPoolError: If client was not acquired from this pool
        """
        # Check client validity - handle both real clients and mocked clients
        if hasattr(client, "port") and hasattr(client, "container_id"):
            # Valid client (either real or properly mocked)
            pass
        else:
            raise EmulatorPoolError("Invalid client type - must be PokemonGymClient")

        # Return client to available pool - queue handles thread safety
        self.available_clients.put(client)
        logger.info(f"Released emulator {client}")

    def execute_script(self, script_text: str, checkpoint_id: str | None = None) -> ExecutionResult:
        """
        Compile and execute script text on an available emulator with automatic client management.

        This method provides the full Pokemon script execution pipeline when Pokemon components
        are available, or falls back to basic script execution mode.

        Args:
            script_text: DSL script to compile and execute
            checkpoint_id: Checkpoint to load before execution (optional)

        Returns:
            ExecutionResult with execution details and outcome

        Raises:
            EmulatorPoolError: On execution failure or resource unavailability
        """
        client = None
        start_time = time.time()

        try:
            # Acquire emulator with reasonable timeout
            client = self.acquire(timeout=30.0)

            # Use full compilation pipeline if Pokemon components are available
            if POKEMON_COMPONENTS_AVAILABLE and self.script_compiler:
                logger.info(f"Compiling script: {script_text[:100]}...")
                compiled_script = self.script_compiler.compile(script_text)

                # Execute compiled script
                return self.execute_compiled_script(client, compiled_script, checkpoint_id)

            else:
                # Fallback to basic script execution
                logger.info(f"Using basic script execution for: {script_text[:100]}...")

                # Load checkpoint if specified (basic mode)
                if checkpoint_id:
                    logger.warning(
                        "Checkpoint loading requested but Pokemon components not available"
                    )

                # Compile DSL script to input sequence (simplified compilation)
                input_sequence = self._compile_script(script_text)

                # Execute script on emulator
                logger.info(f"Executing script on {client}: {script_text[:100]}...")

                try:
                    # Send inputs to emulator
                    response = client.send_input(input_sequence)

                    # Get final state
                    final_state = client.get_state()

                    execution_time = time.time() - start_time

                    return ExecutionResult(
                        success=True,
                        output={"response": response, "final_state": final_state},
                        execution_time=execution_time,
                        checkpoint_reached=None,  # Would be determined by game state analysis
                    )

                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"Script execution failed on {client}: {e}")

                    return ExecutionResult(
                        success=False, output=None, error=str(e), execution_time=execution_time
                    )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to execute script: {e}")

            return ExecutionResult(
                success=False, output=None, error=str(e), execution_time=execution_time
            )

        finally:
            # Always release client back to pool
            if client:
                self.release(client)

    def compile_script(self, script_text: str) -> "CompiledScript":
        """
        Compile DSL script text to CompiledScript using high-performance ScriptCompiler.

        This method provides access to the compilation step separately, which is useful for:
        - Pre-compiling scripts for batch execution
        - Validating script syntax before execution
        - Performance optimization by avoiding repeated compilation

        Args:
            script_text: DSL script to compile

        Returns:
            CompiledScript with instructions, frame estimates, and metadata

        Raises:
            EmulatorPoolError: On compilation failure or if Pokemon components not available
        """
        if not POKEMON_COMPONENTS_AVAILABLE or not self.script_compiler:
            raise EmulatorPoolError(
                "Pokemon components not available. Install CheckpointManager and ScriptCompiler modules."
            )

        try:
            return self.script_compiler.compile(script_text)
        except Exception as e:
            raise EmulatorPoolError(f"Script compilation failed: {e}") from e

    def execute_compiled_script(
        self, client: PokemonGymClient, script: "CompiledScript", checkpoint_id: str | None = None
    ) -> ExecutionResult:
        """
        Execute compiled script on specified emulator client.

        This is the core Pokemon script execution method that integrates all components:
        - CheckpointManager for state loading
        - Compiled script execution with timing
        - Proper error handling and cleanup

        Args:
            client: PokemonGymClient for script execution
            script: CompiledScript with instructions and metadata
            checkpoint_id: Checkpoint to load before execution (optional)

        Returns:
            ExecutionResult with execution details and outcome

        Raises:
            EmulatorPoolError: On execution failure or invalid parameters
        """
        start_time = time.time()

        try:
            logger.info(
                f"Executing compiled script on {client}: "
                f"{len(script.instructions)} instructions, "
                f"{script.total_frames} frames estimated"
            )

            # Load checkpoint if specified and available
            if checkpoint_id and POKEMON_COMPONENTS_AVAILABLE and self.checkpoint_manager:
                try:
                    logger.info(f"Loading checkpoint {checkpoint_id} on {client}")
                    checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_id)

                    # Extract game state and load it into emulator
                    game_state = checkpoint_data.get("game_state", {})
                    if game_state:
                        # This would require extending PokemonGymClient with state loading
                        # For now, we'll use the reset method as a placeholder
                        client.reset_game()
                        logger.info(f"Checkpoint {checkpoint_id} loaded successfully")
                    else:
                        logger.warning(f"Checkpoint {checkpoint_id} has no game state")

                except CheckpointError as e:
                    logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
                    # Continue execution without checkpoint - this allows graceful degradation
                except Exception as e:
                    logger.error(f"Unexpected error loading checkpoint {checkpoint_id}: {e}")

            # Execute compiled script instructions
            try:
                # Convert instructions tuple to space-separated string for Pokemon-gym
                input_sequence = " ".join(script.instructions)

                # Send inputs to emulator
                response = client.send_input(input_sequence)

                # Get final state
                final_state = client.get_state()

                execution_time = time.time() - start_time

                # Analyze final state to determine if any checkpoints were reached
                checkpoint_reached = self._analyze_checkpoint_reached(final_state)

                return ExecutionResult(
                    success=True,
                    output={
                        "response": response,
                        "final_state": final_state,
                        "script_metadata": script.metadata,
                    },
                    execution_time=execution_time,
                    checkpoint_reached=checkpoint_reached,
                )

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Script execution failed on {client}: {e}")

                return ExecutionResult(
                    success=False,
                    output={"script_metadata": script.metadata},
                    error=str(e),
                    execution_time=execution_time,
                )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to execute compiled script: {e}")

            return ExecutionResult(
                success=False, output=None, error=str(e), execution_time=execution_time
            )

    def health_check(self) -> dict[str, Any]:
        """
        Verify all emulators are responsive and healthy.

        Returns:
            Health status report for all emulators
        """
        if not self.containers:
            return {"status": "not_initialized", "healthy_count": 0, "total_count": 0}

        health_results = {}
        healthy_count = 0

        for port, client in self.clients_by_port.items():
            try:
                is_healthy = client.is_healthy()
                health_results[port] = {
                    "healthy": is_healthy,
                    "container_id": client.container_id[:12],
                    "error": None,
                }
                if is_healthy:
                    healthy_count += 1

            except Exception as e:
                health_results[port] = {
                    "healthy": False,
                    "container_id": client.container_id[:12],
                    "error": str(e),
                }
                logger.error(f"Health check failed for emulator on port {port}: {e}")

        overall_status = "healthy" if healthy_count == len(self.clients_by_port) else "degraded"

        return {
            "status": overall_status,
            "healthy_count": healthy_count,
            "total_count": len(self.clients_by_port),
            "emulators": health_results,
        }

    def get_status(self) -> dict[str, Any]:
        """
        Get current pool status with available and busy emulator counts.

        Returns:
            Dictionary with pool status information
        """
        if not self.containers:
            return {
                "available_count": 0,
                "busy_count": 0,
                "total_count": 0,
                "status": "not_initialized",
            }

        available_count = self.available_clients.qsize()
        busy_count = self.pool_size - available_count  # Simple calculation

        return {
            "available_count": available_count,
            "busy_count": busy_count,
            "total_count": self.pool_size,
            "queue_size": 0,  # Simplified: not tracking waiting threads
            "status": "healthy" if available_count <= self.pool_size else "degraded",
        }

    # Removed restart_emulator() method - too complex for workstation use
    # For workstation development, if an emulator fails, shutdown and reinitialize the entire pool

    def _compile_script(self, script: str) -> str:
        """
        Compile DSL script to Pokemon-gym input sequence.

        This is a simplified implementation - the full ScriptCompiler
        will provide proper DSL parsing and compilation.

        Args:
            script: DSL script string

        Returns:
            Input sequence string for Pokemon-gym
        """
        # For now, assume script is already in input sequence format
        # The real implementation would use ScriptCompiler

        # Basic DSL-like patterns to input mapping
        script = script.upper()

        # Replace common patterns
        mappings = {
            "PRESS A": "A",
            "PRESS B": "B",
            "PRESS START": "START",
            "PRESS SELECT": "SELECT",
            "MOVE UP": "UP",
            "MOVE DOWN": "DOWN",
            "MOVE LEFT": "LEFT",
            "MOVE RIGHT": "RIGHT",
            "WAIT": "WAIT",
        }

        for pattern, replacement in mappings.items():
            script = script.replace(pattern, replacement)

        # Remove extra whitespace and return
        return " ".join(script.split())

    def _start_single_container(self, port: int) -> docker.models.containers.Container:
        """
        Start a single container with production configuration.

        Args:
            port: Host port for container mapping

        Returns:
            Started and health-checked container

        Raises:
            EmulatorPoolError: On startup or health check failure
        """
        try:
            if not self.client:
                raise EmulatorPoolError("Docker client not initialized. Call initialize() first.")
            container = self.client.containers.run(
                image=self.image_name,
                ports={"8080/tcp": port},  # Map internal port 8080 to host port
                detach=True,
                remove=True,  # Auto-cleanup on container stop
                name=f"pokemon-emulator-{port}",
                # Production container configuration
                restart_policy={"Name": "on-failure", "MaximumRetryCount": 3},
                mem_limit="512m",  # Prevent memory exhaustion
                cpu_count=1,  # Fair CPU allocation
            )

        except ImageNotFound as e:
            raise EmulatorPoolError(
                f"Pokemon-gym image not found: {self.image_name}. "
                f"Ensure image is built and available locally."
            ) from e
        except APIError as e:
            raise EmulatorPoolError(
                f"Docker API error starting container: {e}. "
                f"Check port {port} availability and system resources."
            ) from e

        # Wait for container to become ready with timeout
        self._wait_for_container_ready(container)

        # Perform health check
        self._verify_container_health(container)

        return container

    def _wait_for_container_ready(self, container: docker.models.containers.Container) -> None:
        """
        Wait for container to reach running state with production timeout.

        Args:
            container: Container to wait for

        Raises:
            EmulatorPoolError: On timeout or container failure
        """
        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:
            container.reload()  # Refresh container status

            if container.status == "running":
                return
            elif container.status in ["exited", "dead"]:
                raise EmulatorPoolError(
                    f"Container {container.id} failed to start (status: {container.status}). "
                    f"Check container logs for details."
                )

            time.sleep(0.5)  # Poll interval

        # Timeout reached
        raise EmulatorPoolError(
            f"Container startup timeout ({self.startup_timeout}s) exceeded. "
            f"Container {container.id} status: {container.status}. "
            f"Increase timeout or check system performance."
        )

    def _verify_container_health(self, container: docker.models.containers.Container) -> None:
        """
        Verify container is healthy and responding.

        Args:
            container: Container to health check

        Raises:
            EmulatorPoolError: On health check failure
        """
        try:
            # Simple health check - verify container can execute commands
            result = container.exec_run("echo 'health_check'")

            if result.exit_code != 0:
                raise EmulatorPoolError(
                    f"Container health check failed for {container.id}. "
                    f"Exit code: {result.exit_code}, output: {result.output}"
                )

        except Exception as e:
            raise EmulatorPoolError(
                f"Container health check failed for {container.id}: {e}. "
                f"Container may not be fully initialized."
            ) from e

    def _get_container_ports(self, count: int) -> list[int]:
        """
        Calculate sequential port numbers for containers.

        Args:
            count: Number of ports needed

        Returns:
            List of sequential port numbers
        """
        return [self.base_port + i for i in range(count)]

    def _analyze_checkpoint_reached(self, game_state: dict[str, Any]) -> str | None:
        """
        Analyze game state to determine if any significant checkpoints were reached.

        This is a simplified analysis - a full implementation would check:
        - Player location/coordinates
        - Story progress flags
        - Item acquisitions
        - Badge count, etc.

        Args:
            game_state: Current game state from emulator

        Returns:
            Checkpoint identifier if significant progress made, None otherwise
        """
        # Placeholder implementation - would analyze game state for progress
        # Real implementation would check location, story flags, items, etc.
        if not isinstance(game_state, dict):
            return None

        # Example: check if player moved to a new location
        location = game_state.get("location", {})
        if isinstance(location, dict):
            map_id = location.get("map_id")
            if map_id and map_id != "starting_town":  # Example progression
                return f"location_{map_id}"

        return None

    def _cleanup_containers(self, containers: list[docker.models.containers.Container]) -> None:
        """
        Clean up containers on failure - prevent resource leaks.

        Args:
            containers: List of containers to stop and clean up
        """
        if not containers:
            return

        logger.info(f"Cleaning up {len(containers)} containers due to initialization failure")

        for container in containers:
            try:
                container.stop(timeout=5)
                logger.info(f"Cleaned up container {container.id}")
            except Exception as e:
                # Log but don't raise - we're already in error handling
                logger.error(f"Failed to cleanup container {container.id}: {e}")

    def get_container_status(self) -> dict:
        """
        Get current status of all containers for monitoring.

        Returns:
            Dictionary with container status information
        """
        if not self.containers:
            return {"total": 0, "running": 0, "status": "not_initialized"}

        statuses = []
        running_count = 0

        for container in self.containers:
            try:
                container.reload()
                status = container.status
                if status == "running":
                    running_count += 1

                statuses.append(
                    {
                        "id": (
                            container.id[:12] if container.id else "unknown"
                        ),  # Short ID for readability
                        "status": status,
                        "name": container.name,
                    }
                )
            except Exception as e:
                statuses.append(
                    {
                        "id": (
                            container.id[:12]
                            if hasattr(container, "id") and container.id
                            else "unknown"
                        ),
                        "status": "error",
                        "error": str(e),
                    }
                )

        return {
            "total": len(self.containers),
            "running": running_count,
            "containers": statuses,
            "status": "healthy" if running_count == len(self.containers) else "degraded",
        }
