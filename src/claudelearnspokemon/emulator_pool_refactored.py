"""
EmulatorPool - Refactored Pokemon-gym emulator management with reduced duplication.

This refactored version uses common utilities to eliminate duplication:
- EmulatorCircuitBreakerMixin for circuit breaker patterns
- BaseExceptionHandler for standardized error handling
- ComponentLogger for consistent logging
- StandardMetricsMixin for metrics recording
- RetryableExceptionHandler for retry logic
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import requests

try:
    import docker
    import docker.errors  # type: ignore[import-not-found]
except ImportError:
    # Docker not available - will raise error at runtime if needed
    docker = None  # type: ignore

from .common_circuit_breaker import EmulatorCircuitBreakerMixin
from .common_error_handling import ErrorSeverity, RetryableExceptionHandler, ComponentErrorHandler
from .common_logging import ComponentLogger, logged_operation
from .common_metrics import StandardMetricsMixin


class ContainerHealthStatus(Enum):
    """Health status for individual container instances."""

    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

    @property
    def is_available(self) -> bool:
        """Check if container is available for work allocation."""
        return self == ContainerHealthStatus.HEALTHY

    @property
    def needs_restart(self) -> bool:
        """Check if container needs restart or replacement."""
        return self in (ContainerHealthStatus.STOPPED, ContainerHealthStatus.UNHEALTHY)


class ExecutionStatus(Enum):
    """Execution status enumeration."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


class EmulatorPoolError(Exception):
    """Custom exception for EmulatorPool operations."""

    pass


@dataclass
class ExecutionResult:
    """Production-ready execution results from Pokemon script execution."""

    execution_id: str
    script_id: str
    status: ExecutionStatus
    start_time: float
    end_time: float
    final_state: dict[str, Any]
    tile_observations: list[dict[str, Any]]
    performance_metrics: dict[str, Any]
    error_message: str | None = None
    checkpoint_id: str | None = None

    @property
    def success(self) -> bool:
        """Backward compatibility property."""
        return self.status == ExecutionStatus.SUCCESS

    @property
    def execution_time(self) -> float:
        """Execution duration in seconds."""
        return self.end_time - self.start_time

    @property
    def duration_ms(self) -> int:
        """Execution duration in milliseconds."""
        return int(self.execution_time * 1000)


@dataclass
class ContainerHealthInfo:
    """Structured health information for individual container instances."""

    container_id: str
    port: int
    status: ContainerHealthStatus
    last_check_time: float
    docker_status: str
    error_message: str | None = None
    response_time_ms: float | None = None
    consecutive_failures: int = 0

    def __post_init__(self) -> None:
        """Ensure container_id is in short form."""
        if len(self.container_id) > 12:
            self.container_id = self.container_id[:12]

    @property
    def age_seconds(self) -> float:
        """Get age of health status in seconds."""
        return time.time() - self.last_check_time

    @property
    def is_stale(self, max_age_seconds: float = 60.0) -> bool:
        """Check if health status is stale."""
        return self.age_seconds > max_age_seconds


class PokemonGymClient(
    RetryableExceptionHandler,
    ComponentErrorHandler,
    ComponentLogger,
    StandardMetricsMixin,
):
    """
    Refactored HTTP client for Pokemon-gym emulator communication.

    Uses common utilities to reduce duplication:
    - ComponentErrorHandler for error handling and context creation
    - ComponentLogger for consistent logging
    - StandardMetricsMixin for metrics recording
    """

    def __init__(self, port: int, container_id: str):
        # Initialize mixins explicitly to handle multiple inheritance properly
        RetryableExceptionHandler.__init__(self)
        ComponentErrorHandler.__init__(self, component_name="PokemonGymClient")
        ComponentLogger.__init__(self)
        StandardMetricsMixin.__init__(self)

        self.port = port
        self.container_id = container_id
        self.base_url = f"http://localhost:{port}"
        self.session = requests.Session()

        self.log_initialization(
            {
                "port": port,
                "container_id": container_id[:12],
                "base_url": self.base_url,
            }
        )

    def create_fallback_response(self, operation_name: str, error: Exception) -> Any:
        """Create component-specific fallback response for emulator operations."""
        return {"status": "error", "message": f"{operation_name} failed: {str(error)}"}

    @logged_operation("send_input")
    def send_input(self, input_sequence: str) -> dict[str, Any] | None:
        """Send input sequence to the emulator."""
        return self._execute_http_operation(
            method="POST",
            endpoint="/input",
            json_data={"input": input_sequence},
            operation_name="send_input",
        )

    @logged_operation("get_state")
    def get_state(self) -> dict[str, Any] | None:
        """Get current game state from emulator."""
        return self._execute_http_operation(
            method="GET", endpoint="/state", operation_name="get_state", timeout=5.0
        )

    @logged_operation("reset_game")
    def reset_game(self) -> dict[str, Any] | None:
        """Reset the game to initial state."""
        return self._execute_http_operation(
            method="POST", endpoint="/reset", operation_name="reset_game"
        )

    @logged_operation("load_checkpoint")
    def load_checkpoint(self, checkpoint_data: bytes) -> bool:
        """Load checkpoint data into the emulator."""
        try:
            response = self._execute_http_operation(
                method="POST",
                endpoint="/checkpoint/load",
                files={"checkpoint": checkpoint_data},
                operation_name="load_checkpoint",
            )
            return response is not None and response.get("status") == "success"
        except Exception as e:
            context = self.create_context("load_checkpoint", {"data_size": len(checkpoint_data)})
            self.handle_exception(e, context, ErrorSeverity.HIGH, reraise_as=EmulatorPoolError)
            return False

    @logged_operation("health_check")
    def health_check(self) -> bool:
        """Perform health check on emulator."""
        try:
            response = self._execute_http_operation(
                method="GET", endpoint="/health", timeout=2.0, operation_name="health_check"
            )
            return response is not None and response.get("status") == "healthy"
        except Exception:
            return False

    def _execute_http_operation(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float = 10.0,
        operation_name: str = "http_operation",
    ) -> dict[str, Any] | None:
        """
        Consolidated HTTP operation execution with retry logic.

        This method eliminates the duplication across all client methods
        by providing a common HTTP execution pattern with standardized:
        - Error handling
        - Retry logic
        - Metrics recording
        - Logging
        """
        context = self.create_context(
            operation_name,
            {
                "method": method,
                "endpoint": endpoint,
                "timeout": timeout,
                "has_json": json_data is not None,
                "has_files": files is not None,
            },
        )

        def http_operation() -> Any:
            url = f"{self.base_url}{endpoint}"

            # Prepare request arguments
            kwargs = {"timeout": timeout}
            if json_data:
                kwargs["json"] = json_data
            if files:
                kwargs["files"] = files

            # Execute HTTP request
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()

            # Parse JSON response
            return response.json()

        return self.execute_with_retry(
            operation=http_operation,
            context=context,
            retryable_exceptions=(
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            ),
            non_retryable_exceptions=(requests.exceptions.JSONDecodeError,),
            reraise_as=EmulatorPoolError,
        )


class EmulatorPool(
    EmulatorCircuitBreakerMixin,
    ComponentErrorHandler,
    ComponentLogger,
    StandardMetricsMixin,
):
    """
    Refactored EmulatorPool with reduced duplication.

    Uses common utilities to eliminate repeated patterns:
    - EmulatorCircuitBreakerMixin for fault tolerance
    - ComponentErrorHandler for error handling and context creation
    - ComponentLogger for consistent logging
    - StandardMetricsMixin for metrics
    """

    def __init__(self, pool_size: int = 4, startup_timeout: float = 30.0):
        # Initialize mixins explicitly to handle multiple inheritance properly
        EmulatorCircuitBreakerMixin.__init__(self)
        ComponentErrorHandler.__init__(self, component_name="EmulatorPool")
        ComponentLogger.__init__(self)
        StandardMetricsMixin.__init__(self)

        self.pool_size = pool_size
        self.startup_timeout = startup_timeout
        self.base_port = 8081

        # Container management
        self._containers: list[Any] = []  # Docker container objects
        self._clients: list[PokemonGymClient] = []
        self._available_clients: list[bool] = []
        self._docker_client: Any | None = None

        # Thread safety
        self._pool_lock = threading.RLock()

        self.log_initialization(
            {
                "pool_size": pool_size,
                "base_port": self.base_port,
                "startup_timeout": startup_timeout,
            }
        )

    @logged_operation("initialize_pool", log_results=True)
    def initialize(self, pool_size: int | None = None) -> None:
        """Initialize the emulator pool."""
        if pool_size is not None:
            self.pool_size = pool_size

        context = self.create_context("initialize", {"pool_size": self.pool_size})

        def init_operation() -> None:
            with self._pool_lock:
                self._connect_to_docker()
                self._create_containers()
                self._wait_for_containers_ready()
                self._create_clients()

        self.safe_execute(
            operation=init_operation,
            context=context,
            allowed_exceptions=(docker.errors.DockerException, EmulatorPoolError),
            severity=ErrorSeverity.CRITICAL,
            reraise_as=EmulatorPoolError,
        )

    @logged_operation("acquire_emulator")
    def acquire(self, timeout: float = 5.0) -> PokemonGymClient | None:
        """Acquire an available emulator client."""
        context = self.create_context("acquire", {"timeout": timeout})

        def acquire_operation() -> Optional[PokemonGymClient]:
            with self._pool_lock:
                for i, (client, available) in enumerate(
                    zip(self._clients, self._available_clients, strict=False)
                ):
                    if available and client.health_check():
                        self._available_clients[i] = False
                        self.record_metric("emulator_acquired", 1)
                        return client

                # No available clients
                self.record_metric("acquisition_failures", 1)
                return None

        return self.safe_execute(
            operation=acquire_operation,
            context=context,
            fallback_value=None,
            severity=ErrorSeverity.MEDIUM,
        )

    @logged_operation("release_emulator")
    def release(self, client: PokemonGymClient) -> None:
        """Release an emulator client back to the pool."""
        context = self.create_context("release", {"client_port": client.port})

        def release_operation() -> None:
            with self._pool_lock:
                for i, pool_client in enumerate(self._clients):
                    if pool_client.port == client.port:
                        self._available_clients[i] = True
                        self.record_metric("emulator_released", 1)
                        return

                # Client not found in pool
                self.record_metric("release_errors", 1)
                raise EmulatorPoolError(f"Client {client.port} not found in pool")

        self.safe_execute(operation=release_operation, context=context, severity=ErrorSeverity.LOW)

    @logged_operation("shutdown_pool")
    def shutdown(self) -> None:
        """Shutdown all emulator containers."""
        context = self.create_context("shutdown", {"container_count": len(self._containers)})

        def shutdown_operation() -> None:
            with self._pool_lock:
                for container in self._containers:
                    try:
                        container.stop()
                        self.record_metric("containers_stopped", 1)
                    except Exception as e:
                        self.logger.log_operation_failure(self.log_start("container_stop"), e)
                        self.record_metric("stop_failures", 1)

                self._containers.clear()
                self._clients.clear()
                self._available_clients.clear()

        self.safe_execute(
            operation=shutdown_operation, context=context, severity=ErrorSeverity.MEDIUM
        )

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        with self._pool_lock:
            health_info = []
            for i, (container, _client) in enumerate(
                zip(self._containers, self._clients, strict=False)
            ):
                info = ContainerHealthInfo(
                    container_id=container.id,
                    port=self.base_port + i,
                    status=(
                        ContainerHealthStatus.HEALTHY
                        if self._available_clients[i]
                        else ContainerHealthStatus.UNHEALTHY
                    ),
                    last_check_time=time.time(),
                    docker_status=getattr(container, "status", "unknown"),
                )
                health_info.append(info.to_dict() if hasattr(info, "to_dict") else vars(info))

        return {
            "pool_size": self.pool_size,
            "available_count": sum(self._available_clients),
            "containers": health_info,
            "circuit_breaker": self.get_circuit_health_status(),
            "metrics": self.get_metrics(),  # type: ignore[attr-defined]
        }

    def _connect_to_docker(self) -> None:
        """Connect to Docker daemon."""
        if docker is None:
            raise EmulatorPoolError("Docker library not available")
            
        try:
            self._docker_client = docker.from_env()  # type: ignore[attr-defined]
            self.logger.log_operation_success(
                self.log_start("docker_connect"), "Connected to Docker daemon"
            )
        except docker.errors.DockerException as e:
            raise EmulatorPoolError(f"Docker daemon unavailable: {e}") from e

    def _create_containers(self) -> None:
        """Create and start Docker containers."""
        if self._docker_client is None:
            raise EmulatorPoolError("Docker client not initialized. Call _connect_to_docker() first.")
            
        for i in range(self.pool_size):
            port = self.base_port + i
            try:
                container = self._docker_client.containers.run(
                    image="pokemon-gym:latest",
                    ports={"8080/tcp": port},
                    detach=True,
                    remove=True,
                    name=f"pokemon-emulator-{port}",
                )
                self._containers.append(container)
                if hasattr(self, 'record_metric'):
                    self.record_metric("containers_created", 1)

            except Exception as e:
                # Clean up already created containers
                self._cleanup_containers()
                raise EmulatorPoolError(f"Failed to start container on port {port}: {e}") from e

    def _wait_for_containers_ready(self) -> None:
        """Wait for all containers to be ready."""
        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:
            all_ready = True

            for container in self._containers:
                if container.status != "running":
                    all_ready = False
                    break

                # Check health
                try:
                    result = container.exec_run("echo health_check")
                    if result.exit_code != 0:
                        all_ready = False
                        break
                except Exception:
                    all_ready = False
                    break

            if all_ready:
                return

            time.sleep(1)

        raise EmulatorPoolError("Container startup timeout")

    def _create_clients(self) -> None:
        """Create HTTP clients for containers."""
        self._clients = []
        self._available_clients = []

        for i, container in enumerate(self._containers):
            port = self.base_port + i
            client = PokemonGymClient(port, container.id)
            self._clients.append(client)
            self._available_clients.append(True)

    def _cleanup_containers(self) -> None:
        """Clean up created containers."""
        for container in self._containers:
            try:
                container.stop()
            except Exception:
                pass  # Best effort cleanup
        self._containers.clear()
