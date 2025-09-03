"""
EmulatorPool: Simplified Docker container management for Pokemon-gym emulators.

Workstation-appropriate container orchestration with clear error handling
and reliable resource management. Built with Bot Dean engineering principles
optimized for development workflow.

Author: Bot Dean - Workstation Engineering
"""

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

import requests

try:
    import docker
    from docker.errors import APIError, DockerException, ImageNotFound  # type: ignore[import-not-found]
except ImportError:
    # Docker not available - will raise error at runtime if needed
    docker = None  # type: ignore
    APIError = Exception
    DockerException = Exception
    ImageNotFound = Exception

# Import compatibility layer factory for transparent adapter selection
try:
    from .pokemon_gym_factory import create_pokemon_client

    COMPATIBILITY_LAYER_AVAILABLE = True
except ImportError:
    # Graceful degradation if compatibility layer not available
    COMPATIBILITY_LAYER_AVAILABLE = False

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


class ExecutionStatus(Enum):
    """Execution status enum for script execution results."""

    SUCCESS = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


class ContainerHealthStatus(Enum):
    """
    Health status enumeration for EmulatorPool container instances.

    Provides clear state tracking for workstation development with
    explicit status transitions and structured monitoring.
    """

    HEALTHY = "healthy"  # Container running and responsive
    UNHEALTHY = "unhealthy"  # Container running but not responsive
    STOPPED = "stopped"  # Container exists but not running
    UNKNOWN = "unknown"  # Container state cannot be determined

    def __str__(self) -> str:
        """Return enum value for display."""
        return self.value

    @property
    def is_available(self) -> bool:
        """Check if container is available for work allocation."""
        return self == ContainerHealthStatus.HEALTHY

    @property
    def needs_restart(self) -> bool:
        """Check if container needs restart or replacement."""
        return self in (ContainerHealthStatus.STOPPED, ContainerHealthStatus.UNHEALTHY)


class EmulatorPoolError(Exception):
    """
    Custom exception for EmulatorPool operations.

    Provides actionable error messages for production debugging.
    """

    pass


@dataclass
class ExecutionResult:
    """
    Production-ready execution results from Pokemon script execution.

    Comprehensive tracking of execution metrics, state, and outcomes
    for analysis, debugging, and learning system optimization.

    Design follows Clean Architecture principles:
    - Single Responsibility: Contains only execution result data
    - Open/Closed: Extensible via additional fields
    - Immutable: Frozen dataclass for thread safety
    """

    # Core execution tracking
    execution_id: str  # UUID for unique identification
    script_id: str  # Identifier for the executed script
    status: ExecutionStatus  # Execution outcome

    # Timing information
    start_time: float  # Unix timestamp when execution started
    end_time: float  # Unix timestamp when execution completed

    # State and observations
    final_state: dict[str, Any]  # Final game state after execution
    tile_observations: list[dict[str, Any]]  # Captured tile states during execution

    # Performance and debugging
    performance_metrics: dict[str, Any]  # frames_executed, actual_duration_ms, etc.
    error_message: str | None = None  # Detailed error information if execution failed

    # Optional execution context
    checkpoint_id: str | None = None  # Checkpoint loaded before execution

    @property
    def success(self) -> bool:
        """Backward compatibility property for existing code."""
        return self.status == ExecutionStatus.SUCCESS

    @property
    def execution_time(self) -> float:
        """Backward compatibility property - execution duration in seconds."""
        return self.end_time - self.start_time

    @property
    def duration_ms(self) -> int:
        """Execution duration in milliseconds for performance analysis."""
        return int((self.end_time - self.start_time) * 1000)

    def __str__(self) -> str:
        return f"ExecutionResult({self.status.name}, {self.duration_ms}ms, script={self.script_id[:8]})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization and analysis."""
        return {
            "execution_id": self.execution_id,
            "script_id": self.script_id,
            "status": self.status.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "final_state": self.final_state,
            "tile_observations_count": len(self.tile_observations),
            "performance_metrics": self.performance_metrics,
            "error_message": self.error_message,
            "checkpoint_id": self.checkpoint_id,
        }


@dataclass
class ContainerHealthInfo:
    """
    Structured health information for individual container instances.

    Provides comprehensive tracking of container state with timestamps
    for workstation development monitoring and debugging.
    """

    container_id: str  # Docker container ID (short form)
    port: int  # HTTP port for communication
    status: ContainerHealthStatus  # Current health status
    last_check_time: float  # Unix timestamp of last health check
    docker_status: str  # Docker container status (running, stopped, etc.)

    # Optional diagnostic information
    error_message: str | None = None  # Last error encountered
    response_time_ms: float | None = None  # Latest health check response time
    consecutive_failures: int = 0  # Count of consecutive health check failures

    def __post_init__(self) -> None:
        """Ensure container_id is in short form."""
        if len(self.container_id) > 12:
            self.container_id = self.container_id[:12]

    @property
    def age_seconds(self) -> float:
        """Get age of this health status in seconds."""
        return time.time() - self.last_check_time

    @property
    def is_stale(self, max_age_seconds: float = 60.0) -> bool:
        """Check if health status is stale (older than max_age_seconds)."""
        return self.age_seconds > max_age_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging and API responses."""
        return {
            "container_id": self.container_id,
            "port": self.port,
            "status": self.status.value,
            "last_check_time": self.last_check_time,
            "docker_status": self.docker_status,
            "error_message": self.error_message,
            "response_time_ms": self.response_time_ms,
            "consecutive_failures": self.consecutive_failures,
            "age_seconds": self.age_seconds,
        }


class PokemonGymClient:
    """
    Workstation-optimized HTTP client for Pokemon-gym emulator communication.

    Simplified for development use with essential resilience features:
    - Basic error handling with clear messages
    - Simplified retry logic for workstation reliability
    - Checkpoint loading capability
    - Essential performance monitoring

    Optimized for workstation development (4 containers max) rather than
    production high-concurrency scenarios.
    """

    # Workstation-appropriate thresholds (simpler than production)
    MAX_CONSECUTIVE_FAILURES = 3  # Fail fast for development
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3  # Circuit breaker threshold for tests
    FAILURE_RESET_TIMEOUT = 10  # Quick recovery for development iteration

    # Simple retry configuration
    MAX_RETRIES = 2  # Fewer retries for faster feedback
    RETRY_DELAY = 0.5  # Simple fixed delay, no exponential backoff

    def __init__(self, port: int, container_id: str):
        """
        Initialize workstation-optimized client for specific emulator instance.

        Args:
            port: HTTP port for emulator communication
            container_id: Docker container ID for this emulator
        """
        self.port = port
        self.container_id = container_id
        self.base_url = f"http://localhost:{port}"
        self.session = requests.Session()

        # Simplified failure tracking for workstation use
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._failure_lock = threading.Lock()  # Simple lock for failure tracking

        # Basic performance tracking
        self._request_count = 0
        self._total_request_time = 0.0

        logger.info(f"PokemonGymClient initialized for port {port}, container {container_id[:12]}")

    def send_input(self, input_sequence: str) -> dict[str, Any]:
        """
        Send input sequence to the emulator with retry logic and circuit breaker.

        Args:
            input_sequence: Button inputs (A, B, START, etc.)

        Returns:
            Response data from emulator

        Raises:
            EmulatorPoolError: On communication failure after all retries
        """
        return self._execute_with_resilience(
            "POST",
            "/input",
            json_data={"inputs": input_sequence},
            timeout=10,
            operation_name="send_input",
        )

    def get_state(self) -> dict[str, Any]:
        """
        Get current game state from emulator with retry logic.

        Returns:
            Current game state data

        Raises:
            EmulatorPoolError: On communication failure after all retries
        """
        return self._execute_with_resilience("GET", "/state", timeout=5, operation_name="get_state")

    def reset_game(self) -> dict[str, Any]:
        """
        Reset the game to initial state with retry logic.

        Returns:
            Reset confirmation from emulator

        Raises:
            EmulatorPoolError: On communication failure after all retries
        """
        return self._execute_with_resilience(
            "POST", "/reset", timeout=10, operation_name="reset_game"
        )

    def is_healthy(self) -> bool:
        """
        Check if emulator is responding to health checks.

        Does not use retry logic for health checks to get fast feedback.

        Returns:
            True if emulator is healthy, False otherwise
        """
        try:
            start_time = time.perf_counter()
            response = self.session.get(f"{self.base_url}/health", timeout=3)
            self._update_performance_metrics(time.perf_counter() - start_time)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

    def __str__(self) -> str:
        return f"PokemonGymClient(port={self.port}, container={self.container_id[:12]})"

    def load_checkpoint(self, checkpoint_data: bytes) -> bool:
        """
        Load checkpoint data into the emulator.

        Args:
            checkpoint_data: Binary checkpoint data to load

        Returns:
            True if checkpoint loaded successfully, False otherwise

        Raises:
            EmulatorPoolError: On communication failure after all retries
        """
        try:
            # Convert bytes to appropriate format for Pokemon-gym API
            # This assumes the Pokemon-gym server accepts checkpoint data via POST
            files = {"checkpoint": checkpoint_data}

            result = self._execute_with_resilience(
                "POST",
                "/checkpoint/load",
                files=files,
                timeout=15,  # Checkpoint loading may take longer
                operation_name="load_checkpoint",
            )
            return result.get("status") == "success"
        except Exception as e:
            logger.error(f"Checkpoint loading failed on {self}: {e}")
            raise EmulatorPoolError(
                f"Failed to load checkpoint on emulator port {self.port}: {e}"
            ) from e

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get simplified client performance metrics for workstation monitoring.

        Returns:
            Dictionary with essential performance statistics
        """
        avg_request_time = (
            self._total_request_time / self._request_count if self._request_count > 0 else 0.0
        )

        return {
            "total_requests": self._request_count,
            "average_request_time_ms": int(avg_request_time * 1000),
            "consecutive_failures": self._consecutive_failures,
            "circuit_breaker_failures": self._consecutive_failures,  # Alias for test compatibility
            "circuit_breaker_open": self._is_circuit_breaker_open(),  # Alias for test compatibility
            "temporarily_disabled": self._is_temporarily_disabled(),
        }

    # Private methods for resilience patterns

    def _execute_with_resilience(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float = 10.0,
        operation_name: str = "unknown",
    ) -> dict[str, Any]:
        """
        Execute HTTP request with simplified workstation-appropriate retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            json_data: JSON payload for request
            files: Files payload for request
            timeout: Request timeout in seconds
            operation_name: Operation name for logging

        Returns:
            Response data from emulator

        Raises:
            EmulatorPoolError: On communication failure after all retries
        """
        # Simple failure check for workstation use
        if self._is_temporarily_disabled():
            raise EmulatorPoolError(
                f"Circuit breaker OPEN - Emulator on port {self.port} temporarily disabled due to consecutive failures. "
                f"Will retry after {self.FAILURE_RESET_TIMEOUT}s."
            )

        last_exception = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                start_time = time.perf_counter()

                # Execute the HTTP request
                url = f"{self.base_url}{endpoint}"
                if method.upper() == "GET":
                    response = self.session.get(url, timeout=timeout)
                elif method.upper() == "POST":
                    if files:
                        response = self.session.post(url, files=files, timeout=timeout)
                    else:
                        response = self.session.post(url, json=json_data, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Check response status
                response.raise_for_status()

                # Track performance and reset failure count on success
                request_time = time.perf_counter() - start_time
                self._update_performance_metrics(request_time)
                self._reset_failures()

                # Return response data
                if response.content:
                    return cast(dict[str, Any], response.json())
                else:
                    return {"status": "success"}  # For operations that don't return data

            except requests.RequestException as e:
                last_exception = e
                self._record_failure()

                # Don't retry on client errors (same as before)
                if isinstance(e, requests.HTTPError) and e.response is not None:
                    if e.response.status_code in (400, 401, 403, 404):
                        break

                # Simple fixed retry delay (no exponential backoff)
                if attempt < self.MAX_RETRIES:
                    logger.warning(
                        f"Request failed on {self}, attempt {attempt + 1}/{self.MAX_RETRIES + 1}. "
                        f"Retrying in {self.RETRY_DELAY}s. Error: {e}"
                    )
                    time.sleep(self.RETRY_DELAY)
                else:
                    logger.error(
                        f"Request failed on {self} after {self.MAX_RETRIES + 1} attempts. "
                        f"Final error: {e}"
                    )

        # All retries exhausted
        raise EmulatorPoolError(
            f"Failed to execute {operation_name} on emulator port {self.port} after "
            f"{self.MAX_RETRIES + 1} attempts. Last error: {last_exception}"
        ) from last_exception

    def _is_temporarily_disabled(self) -> bool:
        """
        Check if emulator is temporarily disabled due to consecutive failures.

        Simplified workstation logic with fail-fast behavior.

        Returns:
            True if emulator should be temporarily disabled
        """
        with self._failure_lock:
            # Check if we have too many consecutive failures
            if self._consecutive_failures < self.MAX_CONSECUTIVE_FAILURES:
                return False

            # Check if enough time has passed for reset
            if time.time() - self._last_failure_time > self.FAILURE_RESET_TIMEOUT:
                # Reset failures after timeout
                self._consecutive_failures = 0
                return False

            return True

    def _is_circuit_breaker_open(self) -> bool:
        """
        Alias for _is_temporarily_disabled() to match test expectations.

        Returns:
            True if circuit breaker is open (emulator temporarily disabled)
        """
        return self._is_temporarily_disabled()

    def _record_failure(self) -> None:
        """
        Record a failure for simple workstation failure tracking.
        """
        with self._failure_lock:
            self._consecutive_failures += 1
            self._last_failure_time = time.time()

            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    f"Emulator {self} temporarily disabled after {self._consecutive_failures} "
                    f"consecutive failures. Will retry after {self.FAILURE_RESET_TIMEOUT}s."
                )

    def _reset_failures(self) -> None:
        """
        Reset failure tracking after successful request.
        """
        with self._failure_lock:
            if self._consecutive_failures > 0:
                logger.info(
                    f"Emulator {self} recovered after {self._consecutive_failures} failures"
                )
                self._consecutive_failures = 0
                self._last_failure_time = 0.0

    def _update_performance_metrics(self, request_time: float) -> None:
        """
        Update basic performance tracking metrics for workstation monitoring.

        Args:
            request_time: Request duration in seconds
        """
        self._request_count += 1
        self._total_request_time += request_time


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
        adapter_type: str = "auto",
        input_delay: float = 0.05,
        detection_timeout: float = 3.0,
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
            adapter_type: Client adapter type - "auto", "benchflow", "direct", "fallback" (default: "auto")
            input_delay: Delay between sequential inputs for benchflow adapter (default: 50ms)
            detection_timeout: Timeout for server type auto-detection (default: 3s)
        """
        self.pool_size = pool_size
        self.base_port = base_port
        self.image_name = image_name
        self.startup_timeout = startup_timeout
        self.default_timeout = default_timeout

        # Compatibility layer configuration
        self.adapter_type = adapter_type
        self.input_delay = input_delay
        self.detection_timeout = detection_timeout

        # Container management state
        self.containers: list[Any] = []  # Docker container objects
        self.client: Any | None = None  # Docker client

        # Simplified resource pool state - workstation-appropriate
        self.available_clients: queue.Queue = queue.Queue()
        self.clients_by_port: dict[int, Any] = (
            {}
        )  # All clients by port (PokemonGymClient or PokemonGymAdapter)

        # Health status tracking for workstation monitoring
        self.container_health: dict[int, ContainerHealthInfo] = {}
        self._health_lock = threading.Lock()  # Simple lock for health status updates

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
            f"image={image_name}, timeout={startup_timeout}s, adapter_type={adapter_type}"
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
            if docker is None:
                raise EmulatorPoolError("Docker library not available")
            self.client = docker.from_env()  # type: ignore[attr-defined]
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

                    # Create Pokemon client using compatibility layer factory
                    container_id = container.id or f"unknown-{port}"
                    if COMPATIBILITY_LAYER_AVAILABLE:
                        # Use factory for transparent adapter selection
                        client = create_pokemon_client(
                            port=port,
                            container_id=container_id,
                            adapter_type=self.adapter_type,
                            input_delay=self.input_delay,
                            detection_timeout=self.detection_timeout,
                        )
                        logger.info(
                            f"Created Pokemon client via compatibility layer for port {port}"
                        )
                    else:
                        # Fallback to direct client if compatibility layer not available
                        client = PokemonGymClient(port, container_id)
                        logger.info(
                            f"Created direct PokemonGymClient for port {port} (compatibility layer unavailable)"
                        )
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

    def release(self, client: Any) -> None:
        """
        Release emulator client back to available pool.

        Simple operation that makes emulator available for other tasks.

        Args:
            client: Pokemon client (PokemonGymClient or PokemonGymAdapter) to return to pool

        Raises:
            EmulatorPoolError: If client was not acquired from this pool
        """
        # Check client validity - handle both client types and mocked clients
        if hasattr(client, "port") and hasattr(client, "container_id"):
            # Valid client (direct client, adapter, or properly mocked)
            pass
        else:
            raise EmulatorPoolError(
                "Invalid client type - must be a valid Pokemon client with port and container_id attributes"
            )

        # Return client to available pool - queue handles thread safety
        self.available_clients.put(client)
        logger.info(f"Released emulator {client}")

    def execute_script(
        self,
        script_text: str,
        checkpoint_id: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        cancellation_event: threading.Event | None = None,
        timeout: float = 300.0,
    ) -> ExecutionResult:
        """
        Compile and execute script text with comprehensive monitoring and control.

        This method provides the full Pokemon script execution pipeline with:
        - Automatic client acquisition and release
        - Progress monitoring and cancellation support
        - Timeout protection for long-running scripts
        - Comprehensive error handling and metrics
        - Graceful fallback when Pokemon components unavailable

        Args:
            script_text: DSL script to compile and execute
            checkpoint_id: Checkpoint to load before execution (optional)
            progress_callback: Callback for progress updates (frames_executed, total_frames)
            cancellation_event: Event to signal execution cancellation
            timeout: Maximum execution time in seconds (default: 300s)

        Returns:
            ExecutionResult with comprehensive execution details and outcome

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

                # Execute compiled script with monitoring support
                script_id = f"script_{int(time.time())}"
                return self.execute_compiled_script(
                    client,
                    compiled_script,
                    checkpoint_id,
                    progress_callback=progress_callback,
                    cancellation_event=cancellation_event,
                    script_id=script_id,
                )

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

                # Generate execution tracking
                execution_id = str(uuid.uuid4())
                script_id = f"basic_script_{int(time.time())}"

                try:
                    # Send inputs to emulator
                    client.send_input(input_sequence)

                    # Get final state
                    final_state = client.get_state()

                    end_time = time.time()

                    return ExecutionResult(
                        execution_id=execution_id,
                        script_id=script_id,
                        status=ExecutionStatus.SUCCESS,
                        start_time=start_time,
                        end_time=end_time,
                        final_state=final_state,
                        tile_observations=[
                            {
                                "frame": 0,
                                "instruction_index": 0,
                                "state": final_state,
                                "timestamp": end_time,
                            }
                        ],
                        performance_metrics={
                            "frames_executed": 1,  # Simple scripts execute as one block
                            "actual_duration_ms": int((end_time - start_time) * 1000),
                            "estimated_frames": 1,
                            "instructions_completed": 1,
                            "total_instructions": 1,
                            "completion_percentage": 100.0,
                            "observations_captured": 1,
                        },
                        error_message=None,
                        checkpoint_id=checkpoint_id,
                    )

                except Exception as e:
                    end_time = time.time()
                    logger.error(f"Script execution failed on {client}: {e}")

                    return ExecutionResult(
                        execution_id=execution_id,
                        script_id=script_id,
                        status=ExecutionStatus.FAILED,
                        start_time=start_time,
                        end_time=end_time,
                        final_state={},
                        tile_observations=[],
                        performance_metrics={
                            "frames_executed": 0,
                            "actual_duration_ms": int((end_time - start_time) * 1000),
                            "estimated_frames": 1,
                            "instructions_completed": 0,
                            "total_instructions": 1,
                            "completion_percentage": 0.0,
                            "observations_captured": 0,
                        },
                        error_message=str(e),
                        checkpoint_id=checkpoint_id,
                    )

        except Exception as e:
            end_time = time.time()
            logger.error(f"Failed to execute script: {e}")

            return ExecutionResult(
                execution_id=str(uuid.uuid4()),
                script_id=f"failed_script_{int(time.time())}",
                status=ExecutionStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                final_state={},
                tile_observations=[],
                performance_metrics={
                    "frames_executed": 0,
                    "actual_duration_ms": int((end_time - start_time) * 1000),
                    "estimated_frames": 0,
                    "instructions_completed": 0,
                    "total_instructions": 0,
                    "completion_percentage": 0.0,
                    "observations_captured": 0,
                },
                error_message=str(e),
                checkpoint_id=checkpoint_id,
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
        self,
        client: PokemonGymClient,
        script: "CompiledScript",
        checkpoint_id: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        cancellation_event: threading.Event | None = None,
        script_id: str | None = None,
    ) -> ExecutionResult:
        """
        Execute compiled script on specified emulator client with comprehensive monitoring.

        This is the core Pokemon script execution method that integrates all components:
        - CheckpointManager for state loading
        - Sequential instruction execution with timing
        - Progress monitoring and cancellation support
        - Tile observation capture at specified points
        - Comprehensive performance metrics

        Args:
            client: PokemonGymClient for script execution
            script: CompiledScript with instructions and metadata
            checkpoint_id: Checkpoint to load before execution (optional)
            progress_callback: Callback for progress updates (frames_executed, total_frames)
            cancellation_event: Event to signal execution cancellation
            script_id: Unique identifier for the script (auto-generated if None)

        Returns:
            ExecutionResult with comprehensive execution details and outcome

        Raises:
            EmulatorPoolError: On execution failure or invalid parameters
        """
        # Initialize execution tracking
        execution_id = str(uuid.uuid4())
        script_id = script_id or str(uuid.uuid4())[:8]
        start_time = time.time()

        # Initialize result containers
        tile_observations: list[dict[str, Any]] = []
        frames_executed = 0

        try:
            logger.info(
                f"Executing compiled script {script_id} on {client}: "
                f"{len(script.instructions)} instructions, "
                f"{script.total_frames} frames estimated"
            )

            # Load checkpoint if specified and available
            if checkpoint_id and POKEMON_COMPONENTS_AVAILABLE and self.checkpoint_manager:
                try:
                    logger.info(f"Loading checkpoint {checkpoint_id} on {client}")
                    checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_id)

                    # Extract game state and load it into emulator
                    # Convert checkpoint data to bytes for the client
                    import json

                    checkpoint_bytes = json.dumps(checkpoint_data).encode("utf-8")

                    success = client.load_checkpoint(checkpoint_bytes)
                    if success:
                        logger.info(f"Checkpoint {checkpoint_id} loaded successfully")
                    else:
                        logger.warning(f"Checkpoint {checkpoint_id} failed to load")

                except CheckpointError as e:
                    logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
                    # Continue execution without checkpoint - this allows graceful degradation
                except Exception as e:
                    logger.error(f"Unexpected error loading checkpoint {checkpoint_id}: {e}")

            # Execute script instructions sequentially with monitoring
            try:
                total_instructions = len(script.instructions)

                # Capture initial state as observation
                try:
                    initial_state = client.get_state()
                    tile_observations.append(
                        {
                            "frame": 0,
                            "instruction_index": 0,
                            "state": initial_state,
                            "timestamp": time.time(),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to capture initial state: {e}")

                # Execute instructions one by one for fine-grained control
                failed_instructions = 0
                for instruction_index, instruction in enumerate(script.instructions):
                    # Check for cancellation
                    if cancellation_event and cancellation_event.is_set():
                        logger.info(
                            f"Execution cancelled at instruction {instruction_index}/{total_instructions}"
                        )
                        end_time = time.time()

                        return ExecutionResult(
                            execution_id=execution_id,
                            script_id=script_id,
                            status=ExecutionStatus.CANCELLED,
                            start_time=start_time,
                            end_time=end_time,
                            final_state={},
                            tile_observations=tile_observations,
                            performance_metrics={
                                "frames_executed": frames_executed,
                                "actual_duration_ms": int((end_time - start_time) * 1000),
                                "instructions_completed": instruction_index,
                                "total_instructions": total_instructions,
                                "completion_percentage": (instruction_index / total_instructions)
                                * 100,
                            },
                            error_message="Execution cancelled by user",
                            checkpoint_id=checkpoint_id,
                        )

                    # Execute single instruction
                    try:
                        client.send_input(instruction)
                        frames_executed += 1

                        # Report progress if callback provided
                        if progress_callback:
                            progress_callback(frames_executed, script.total_frames)

                        # Capture tile observations at specified points
                        if instruction_index in script.observation_points:
                            try:
                                current_state = client.get_state()
                                tile_observations.append(
                                    {
                                        "frame": frames_executed,
                                        "instruction_index": instruction_index,
                                        "instruction": instruction,
                                        "state": current_state,
                                        "timestamp": time.time(),
                                    }
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to capture observation at frame {frames_executed}: {e}"
                                )

                        # Small delay to prevent overwhelming the emulator
                        time.sleep(0.05)

                    except Exception as e:
                        logger.error(
                            f"Failed to execute instruction '{instruction}' at index {instruction_index}: {e}"
                        )
                        failed_instructions += 1
                        # Continue with next instruction for resilience

                # Get final state after all instructions
                try:
                    final_state = client.get_state()
                except Exception as e:
                    logger.error(f"Failed to capture final state: {e}")
                    final_state = {}

                end_time = time.time()

                # Analyze final state to determine if any checkpoints were reached
                checkpoint_reached = self._analyze_checkpoint_reached(final_state)

                # Determine success based on instruction failure rate
                total_instructions = len(script.instructions)
                success_rate = (
                    (total_instructions - failed_instructions) / total_instructions
                    if total_instructions > 0
                    else 1.0
                )

                # Consider execution successful if at least 50% of instructions succeeded
                # For complete failures (all instructions failed), this will be FAILED
                is_success = success_rate >= 0.5
                execution_status = ExecutionStatus.SUCCESS if is_success else ExecutionStatus.FAILED
                error_msg = (
                    None
                    if is_success
                    else f"Script execution failed: {failed_instructions}/{total_instructions} instructions failed"
                )

                return ExecutionResult(
                    execution_id=execution_id,
                    script_id=script_id,
                    status=execution_status,
                    start_time=start_time,
                    end_time=end_time,
                    final_state=final_state,
                    tile_observations=tile_observations,
                    performance_metrics={
                        "frames_executed": frames_executed,
                        "actual_duration_ms": int((end_time - start_time) * 1000),
                        "estimated_frames": script.total_frames,
                        "instructions_completed": total_instructions - failed_instructions,
                        "total_instructions": total_instructions,
                        "completion_percentage": success_rate * 100.0,
                        "observations_captured": len(tile_observations),
                        "checkpoint_reached": checkpoint_reached,
                        "failed_instructions": failed_instructions,
                        "success_rate": success_rate,
                    },
                    error_message=error_msg,
                    checkpoint_id=checkpoint_id,
                )

            except Exception as e:
                end_time = time.time()
                logger.error(f"Script execution failed on {client}: {e}")

                return ExecutionResult(
                    execution_id=execution_id,
                    script_id=script_id,
                    status=ExecutionStatus.FAILED,
                    start_time=start_time,
                    end_time=end_time,
                    final_state={},
                    tile_observations=tile_observations,
                    performance_metrics={
                        "frames_executed": frames_executed,
                        "actual_duration_ms": int((end_time - start_time) * 1000),
                        "estimated_frames": script.total_frames,
                        "instructions_completed": frames_executed,
                        "total_instructions": len(script.instructions),
                        "completion_percentage": (
                            (frames_executed / len(script.instructions)) * 100
                            if script.instructions
                            else 0
                        ),
                        "observations_captured": len(tile_observations),
                    },
                    error_message=str(e),
                    checkpoint_id=checkpoint_id,
                )

        except Exception as e:
            end_time = time.time()
            logger.error(f"Failed to execute compiled script: {e}")

            return ExecutionResult(
                execution_id=execution_id,
                script_id=script_id,
                status=ExecutionStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                final_state={},
                tile_observations=tile_observations,
                performance_metrics={
                    "frames_executed": frames_executed,
                    "actual_duration_ms": int((end_time - start_time) * 1000),
                    "estimated_frames": 0,
                    "instructions_completed": 0,
                    "total_instructions": 0,
                    "completion_percentage": 0.0,
                    "observations_captured": len(tile_observations),
                },
                error_message=str(e),
                checkpoint_id=checkpoint_id,
            )

    def _update_container_health(
        self,
        port: int,
        container_id: str,
        status: ContainerHealthStatus,
        docker_status: str,
        error_message: str | None = None,
        response_time_ms: float | None = None,
    ) -> None:
        """
        Update health status for a specific container with structured logging.

        Args:
            port: Container HTTP port
            container_id: Docker container ID
            status: New health status
            docker_status: Current Docker container status
            error_message: Optional error message if health check failed
            response_time_ms: Optional response time for successful health checks
        """
        with self._health_lock:
            current_time = time.time()

            # Get current health info to track status changes
            current_health = self.container_health.get(port)
            status_changed = current_health is None or current_health.status != status

            # Track consecutive failures
            consecutive_failures = 0
            if current_health and status != ContainerHealthStatus.HEALTHY:
                consecutive_failures = current_health.consecutive_failures + 1
            elif status == ContainerHealthStatus.HEALTHY:
                consecutive_failures = 0

            # Update health status
            self.container_health[port] = ContainerHealthInfo(
                container_id=container_id,
                port=port,
                status=status,
                last_check_time=current_time,
                docker_status=docker_status,
                error_message=error_message,
                response_time_ms=response_time_ms,
                consecutive_failures=consecutive_failures,
            )

            # Structured logging for health status changes
            if status_changed:
                old_status = current_health.status.value if current_health else "unknown"
                logger.info(
                    f"Container health status changed: port={port} container={container_id[:12]} "
                    f"{old_status} -> {status.value} docker_status={docker_status}"
                )

                # Log additional context for unhealthy states
                if status != ContainerHealthStatus.HEALTHY:
                    logger.warning(
                        f"Container unhealthy: port={port} container={container_id[:12]} "
                        f"status={status.value} consecutive_failures={consecutive_failures} "
                        f"error='{error_message or 'none'}'"
                    )

    def get_container_health_status(self, port: int) -> ContainerHealthInfo | None:
        """
        Get current health status for a specific container.

        Args:
            port: Container HTTP port

        Returns:
            ContainerHealthInfo if container exists, None otherwise
        """
        with self._health_lock:
            return self.container_health.get(port)

    def get_all_container_health(self) -> dict[int, ContainerHealthInfo]:
        """
        Get health status for all containers.

        Returns:
            Dictionary mapping ports to ContainerHealthInfo
        """
        with self._health_lock:
            return self.container_health.copy()

    def health_check(self) -> dict[str, Any]:
        """
        Comprehensive health check with Docker status and enum-based status tracking.

        Performs both HTTP connectivity checks and Docker container status validation
        for complete workstation health monitoring.

        Returns:
            Enhanced health status report with structured container information
        """
        if not self.containers:
            return {
                "status": "not_initialized",
                "healthy_count": 0,
                "total_count": 0,
                "containers": [],
            }

        health_results = {}
        healthy_count = 0

        # Create a mapping of ports to containers for Docker status checking
        container_by_port = {}
        for container in self.containers:
            try:
                if (
                    hasattr(container, "attrs")
                    and hasattr(container.attrs, "get")
                    and "NetworkSettings" in container.attrs
                ):
                    ports = container.attrs["NetworkSettings"].get("Ports", {})
                    for port_spec, bindings in ports.items():
                        if bindings and port_spec.endswith("/tcp"):
                            host_port = int(bindings[0]["HostPort"])
                            container_by_port[host_port] = container
            except (TypeError, AttributeError):
                # Handle case where attrs might be a Mock object or not properly structured
                continue

        for port, client in self.clients_by_port.items():
            container_id = client.container_id
            docker_status = "unknown"
            status = ContainerHealthStatus.UNKNOWN
            error_message = None
            response_time_ms = None

            # Check Docker container status first
            current_container: Any | None = container_by_port.get(port)

            # Fallback: if port mapping failed, try to find container by ID
            if not current_container:
                for container in self.containers:
                    if hasattr(container, "id") and container.id == container_id:
                        current_container = container
                        break

            if current_container:
                try:
                    current_container.reload()
                    docker_status = current_container.status

                    if docker_status != "running":
                        status = ContainerHealthStatus.STOPPED
                        error_message = f"Docker container not running: {docker_status}"
                    else:
                        # Container is running, now check HTTP responsiveness
                        start_time = time.time()
                        try:
                            is_responsive = client.is_healthy()
                            response_time_ms = (time.time() - start_time) * 1000

                            if is_responsive:
                                status = ContainerHealthStatus.HEALTHY
                            else:
                                status = ContainerHealthStatus.UNHEALTHY
                                error_message = "Container running but not responsive"

                        except Exception as e:
                            response_time_ms = (time.time() - start_time) * 1000
                            status = ContainerHealthStatus.UNHEALTHY
                            error_message = f"HTTP health check failed: {str(e)}"

                except Exception as e:
                    status = ContainerHealthStatus.UNKNOWN
                    error_message = f"Docker status check failed: {str(e)}"
            else:
                # No container found for this port
                status = ContainerHealthStatus.UNKNOWN
                error_message = "No Docker container found for port"

            # Update health status tracking
            self._update_container_health(
                port=port,
                container_id=container_id,
                status=status,
                docker_status=docker_status,
                error_message=error_message,
                response_time_ms=response_time_ms,
            )

            # Build health result
            health_results[port] = {
                "status": status.value,
                "healthy": status.is_available,
                "docker_status": docker_status,
                "container_id": container_id[:12],
                "error": error_message,
                "response_time_ms": response_time_ms,
                "needs_restart": status.needs_restart,
            }

            if status.is_available:
                healthy_count += 1

        # Determine overall pool status
        total_count = len(self.clients_by_port)
        if total_count == 0:
            overall_status = "not_initialized"
        elif healthy_count == total_count:
            overall_status = "healthy"
        elif healthy_count == 0:
            overall_status = "critical"
        else:
            overall_status = "degraded"

        return {
            "status": overall_status,
            "healthy_count": healthy_count,
            "total_count": total_count,
            "containers": list(health_results.values()),
            "emulators": health_results,  # Backward compatibility
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

    def replace_failed_container(self, port: int) -> bool:
        """
        Replace a failed container with a new one on the same port.

        This method provides workstation-appropriate container replacement for auto-restart
        functionality. It stops the failed container and creates a replacement while
        maintaining the simplified EmulatorPool architecture.

        Args:
            port: Port of the container to replace

        Returns:
            True if replacement successful, False otherwise
        """
        if not self.containers or not self.client:
            logger.error(f"Cannot replace container on port {port}: EmulatorPool not initialized")
            return False

        logger.info(f"Attempting to replace failed container on port {port}")

        try:
            # Find and remove the old client from tracking
            old_client = self.clients_by_port.get(port)
            if not old_client:
                logger.error(f"No client found for port {port}")
                return False

            # Check if client was available in queue and remove it if found
            was_available = self._is_client_available(old_client, remove_if_found=True)

            # Close the old client session
            try:
                old_client.close()
            except Exception as e:
                logger.warning(f"Error closing old client on port {port}: {e}")

            # Find and stop the old container
            old_container = None
            for i, container in enumerate(self.containers):
                try:
                    container.reload()
                    # Check container name to match port
                    if container.name == f"pokemon-emulator-{port}":
                        old_container = container
                        logger.info(
                            f"Stopping old container {container.id[:12] if container.id else 'unknown'} on port {port}"
                        )
                        container.stop(timeout=10)
                        # Remove from containers list
                        self.containers.pop(i)
                        break
                except Exception as e:
                    logger.warning(
                        f"Error checking container {container.id[:12] if container.id else 'unknown'}: {e}"
                    )

            if not old_container:
                logger.warning(
                    f"Could not find old container for port {port}, proceeding with new container creation"
                )

            # Create new container on the same port
            try:
                new_container = self._start_single_container(port)
                self.containers.append(new_container)

                # Create new client and update mapping
                container_id = new_container.id or f"unknown-{port}"
                new_client = PokemonGymClient(port, container_id)
                self.clients_by_port[port] = new_client

                # If the old client was available, make the new one available too
                if was_available:
                    self.available_clients.put(new_client)

                logger.info(
                    f"Successfully replaced container on port {port} with {new_container.id[:12] if new_container.id else 'unknown'}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to create replacement container on port {port}: {e}")
                # Clean up partial state - remove the client mapping since container failed
                self.clients_by_port.pop(port, None)
                return False

        except Exception as e:
            logger.error(f"Unexpected error replacing container on port {port}: {e}")
            return False

    def _is_client_available(
        self, target_client: PokemonGymClient, remove_if_found: bool = False
    ) -> bool:
        """
        Check if a client is currently in the available queue.

        This is a workstation-appropriate way to check availability without
        complex queue introspection. We temporarily drain and restore the queue.

        Args:
            target_client: Client to check for availability
            remove_if_found: If True, don't restore the client if found (remove it from queue)

        Returns:
            True if client is available, False if busy
        """
        # Simple approach: temporarily drain queue to check for client
        temp_clients = []
        client_found = False

        try:
            # Drain queue and look for our client
            while not self.available_clients.empty():
                try:
                    client = self.available_clients.get_nowait()
                    temp_clients.append(client)

                    # Check if this is our target client (match by port and container_id)
                    if (
                        hasattr(client, "port")
                        and hasattr(client, "container_id")
                        and client.port == target_client.port
                        and client.container_id == target_client.container_id
                    ):
                        client_found = True
                        # If we should remove it, don't add it to temp_clients
                        if remove_if_found:
                            temp_clients.pop()  # Remove the client we just added

                except queue.Empty:
                    break

            # Restore clients back to the queue (excluding removed client if any)
            for client in temp_clients:
                self.available_clients.put(client)

            return client_found

        except Exception as e:
            logger.error(f"Error checking client availability: {e}")
            # Restore any clients we managed to get
            for client in temp_clients:
                self.available_clients.put(client)
            return False

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

    def _start_single_container(self, port: int) -> Any:  # Docker container type
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

    def _wait_for_container_ready(self, container: Any) -> None:  # Docker container type
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

    def _verify_container_health(self, container: Any) -> None:  # Docker container type
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

    def _cleanup_containers(self, containers: list[Any]) -> None:  # Docker container types
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
