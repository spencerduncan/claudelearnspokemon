"""
EmulatorPool - Thread-safe Pokemon-gym Docker container management

Combines production-grade Docker container orchestration with kernel-quality
concurrent access handling, synchronization primitives, and deadlock prevention.

Provides thread-safe access to a pool of Pokemon-gym Docker containers with:
- Proper locking and condition variables for concurrent acquisition
- Priority queue management with FIFO fairness
- Timeout handling and graceful cleanup
- Docker container lifecycle management
- Production-grade error handling and logging
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import docker
from docker.errors import APIError, DockerException, ImageNotFound

# Type aliases for API compatibility
PokemonGymClient = Any  # Docker container or API client
CompiledScript = Any
ExecutionResult = dict[str, Any]

# Configure logging for concurrent debugging and production monitoring
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EmulatorPoolError(Exception):
    """Custom exception for EmulatorPool operations with actionable error messages."""

    pass


class EmulatorState(Enum):
    """Emulator instance state tracking"""

    AVAILABLE = "available"
    BUSY = "busy"
    FAILED = "failed"
    RESTARTING = "restarting"


@dataclass
class EmulatorInstance:
    """Container for emulator instance state with Docker integration"""

    port: int
    container: docker.models.containers.Container | None = None
    client: PokemonGymClient | None = None
    state: EmulatorState = EmulatorState.AVAILABLE
    owner_thread_id: int | None = None
    acquired_at: float | None = None

    def is_available(self) -> bool:
        """Check if emulator is available for acquisition"""
        return self.state == EmulatorState.AVAILABLE and self.owner_thread_id is None


@dataclass
class AcquisitionRequest:
    """Request for emulator acquisition with priority and timeout handling"""

    thread_id: int
    priority: int = 0
    requested_at: float | None = None
    timeout: float | None = None

    def __post_init__(self) -> None:
        if self.requested_at is None:
            self.requested_at = time.time()


class EmulatorPool:
    """
    Thread-safe pool manager for Pokemon-gym Docker container instances

    Combines Docker container lifecycle management with concurrent access handling:
    - Thread-safe acquisition/release with RLock and condition variables
    - Priority queue management with FIFO fairness semantics
    - Configurable timeout handling with graceful cleanup
    - Production Docker container orchestration
    - Comprehensive error handling and deadlock prevention
    """

    def __init__(
        self,
        pool_size: int = 4,
        default_timeout: float = 30.0,
        base_port: int = 8081,
        image_name: str = "pokemon-gym:latest",
        startup_timeout: int = 30,
    ) -> None:
        """Initialize emulator pool with Docker and concurrent access configuration"""
        # Pool configuration
        self.pool_size = pool_size
        self.default_timeout = default_timeout
        self.base_port = base_port
        self.image_name = image_name
        self.startup_timeout = startup_timeout

        # Core synchronization primitives (kernel-quality thread safety)
        self._lock = threading.RLock()  # Reentrant for deadlock prevention
        self._condition = threading.Condition(self._lock)

        # Docker client for container management
        self.docker_client: docker.DockerClient | None = None

        # Emulator tracking with thread-safe state management
        self._emulators: dict[int, EmulatorInstance] = {}
        self._available_ports: set[int] = set()
        self._busy_ports: set[int] = set()

        # Concurrent acquisition queue management
        self._acquisition_queue: queue.PriorityQueue[Any] = queue.PriorityQueue()
        self._thread_requests: dict[int, AcquisitionRequest] = {}

        # Pool lifecycle state
        self._initialized = False
        self._shutdown = False

        # Thread-local storage for debugging context
        self._thread_local = threading.local()

        logger.info(
            "EmulatorPool initialized: size=%d, timeout=%.1f, base_port=%d, image=%s",
            pool_size,
            default_timeout,
            base_port,
            image_name,
        )

    def initialize(self, pool_size: int | None = None) -> None:
        """
        Initialize Docker container pool with thread-safe state management

        Combines Docker container startup with concurrent access initialization.
        """
        with self._lock:
            if self._initialized:
                logger.warning("EmulatorPool already initialized")
                return

            if pool_size is not None:
                self.pool_size = pool_size

            logger.info("Initializing %d Docker emulator containers...", self.pool_size)

            try:
                # Connect to Docker daemon with proper error handling
                self.docker_client = docker.from_env()
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
                    logger.info("Starting Docker container on port %d", port)

                    try:
                        container = self._start_single_container(port)
                        started_containers.append(container)

                        # Create emulator instance with Docker integration
                        emulator = EmulatorInstance(port=port, container=container)
                        self._emulators[port] = emulator
                        self._available_ports.add(port)

                        logger.info(
                            "Container %s started successfully on port %d",
                            container.id[:12] if container.id else "unknown",
                            port,
                        )

                    except Exception as e:
                        logger.error("Failed to start container on port %d: %s", port, e)
                        self._cleanup_containers(started_containers)
                        raise EmulatorPoolError(
                            f"Failed to start container on port {port}: {e}. "
                            f"Check port availability and resource limits."
                        ) from e

                self._initialized = True
                self._condition.notify_all()  # Wake up any waiting threads

                logger.info("EmulatorPool initialized with %d containers", len(self._emulators))

            except EmulatorPoolError:
                # Already handled, just re-raise
                raise
            except Exception as e:
                logger.error("Unexpected error during initialization: %s", e)
                self._cleanup_containers(started_containers)
                raise EmulatorPoolError(
                    f"Unexpected error during pool initialization: {e}. "
                    f"Check Docker daemon status and system resources."
                ) from e

    def acquire(self, timeout: float | None = None, priority: int = 0) -> PokemonGymClient | None:
        """
        Acquire an available emulator with thread-safe queuing and Docker integration

        Uses kernel-quality synchronization primitives for concurrent access.
        """
        if timeout is None:
            timeout = self.default_timeout

        thread_id = threading.get_ident()

        # Create acquisition request
        request = AcquisitionRequest(thread_id=thread_id, priority=priority, timeout=timeout)

        logger.debug(
            "Thread %d requesting emulator acquisition (priority=%d, timeout=%.1f)",
            thread_id,
            priority,
            timeout,
        )

        with self._lock:
            if self._shutdown:
                logger.warning("Thread %d acquisition rejected: pool shutdown", thread_id)
                return None

            if not self._initialized:
                logger.warning("Thread %d waiting for pool initialization...", thread_id)
                if not self._condition.wait_for(
                    lambda: self._initialized or self._shutdown, timeout=timeout
                ):
                    logger.error("Thread %d timed out waiting for initialization", thread_id)
                    return None

                if self._shutdown:
                    return None

            # Check for immediate availability
            available_port = self._get_available_port()
            if available_port is not None:
                return self._assign_emulator(available_port, thread_id)

            # Add to queue and wait
            self._thread_requests[thread_id] = request
            self._acquisition_queue.put((priority, request.requested_at, request))

            logger.debug(
                "Thread %d added to acquisition queue (queue_size=%d)",
                thread_id,
                self._acquisition_queue.qsize(),
            )

            # Wait for availability or timeout
            deadline = time.time() + timeout
            while True:
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    # Timeout - clean up request
                    self._cleanup_request(thread_id)
                    logger.warning(
                        "Thread %d acquisition timed out after %.1fs",
                        thread_id,
                        timeout,
                    )
                    return None

                if self._shutdown:
                    self._cleanup_request(thread_id)
                    logger.info("Thread %d acquisition cancelled: shutdown", thread_id)
                    return None

                # Wait for notification or timeout
                if self._condition.wait(timeout=min(remaining_time, 1.0)):
                    # Check if we got an emulator
                    available_port = self._get_available_port()
                    if available_port is not None and self._is_highest_priority_thread(thread_id):
                        self._cleanup_request(thread_id)
                        return self._assign_emulator(available_port, thread_id)

        return None

    def release(self, client: PokemonGymClient) -> None:
        """Release emulator back to available pool with thread ownership validation"""
        thread_id = threading.get_ident()

        with self._lock:
            if self._shutdown:
                logger.debug("Thread %d release during shutdown", thread_id)
                return

            # Find emulator by client (Docker container)
            port = None
            for p, emulator in self._emulators.items():
                if emulator.client is client or emulator.container is client:
                    port = p
                    break

            if port is None:
                logger.error("Thread %d attempted to release unknown client", thread_id)
                return

            emulator = self._emulators[port]

            # Verify ownership (critical for thread safety)
            if emulator.owner_thread_id != thread_id:
                logger.error(
                    "Thread %d attempted to release emulator owned by thread %s",
                    thread_id,
                    emulator.owner_thread_id,
                )
                return

            # Release emulator
            emulator.state = EmulatorState.AVAILABLE
            emulator.owner_thread_id = None
            emulator.acquired_at = None
            emulator.client = None

            self._busy_ports.discard(port)
            self._available_ports.add(port)

            logger.debug("Thread %d released emulator on port %d", thread_id, port)

            # Notify waiting threads - use notify_all for priority queue processing
            self._condition.notify_all()

    @contextmanager
    def acquire_emulator(self, timeout: float | None = None, priority: int = 0) -> Any:
        """Context manager for automatic emulator acquisition and release"""
        client = self.acquire(timeout=timeout, priority=priority)
        try:
            yield client
        finally:
            if client is not None:
                self.release(client)

    def execute_script(
        self, client: PokemonGymClient, script: CompiledScript, checkpoint_id: str
    ) -> ExecutionResult:
        """Execute script on specific emulator Docker container with ownership validation"""
        thread_id = threading.get_ident()

        with self._lock:
            # Verify client ownership
            port = None
            for p, emulator in self._emulators.items():
                if (
                    emulator.client is client or emulator.container is client
                ) and emulator.owner_thread_id == thread_id:
                    port = p
                    break

            if port is None:
                raise ValueError(f"Thread {thread_id} does not own the specified client")

        logger.debug("Thread %d executing script on port %d", thread_id, port)

        # TODO: Implement actual script execution on Docker container
        # This would interface with the Pokemon-gym Docker container via API

        # Placeholder implementation
        return {
            "success": True,
            "port": port,
            "thread_id": thread_id,
            "script_id": getattr(script, "id", "unknown"),
            "checkpoint_id": checkpoint_id,
        }

    def health_check(self) -> dict[int, bool]:
        """Verify all Docker containers are responsive with thread-safe access"""
        with self._lock:
            health_status = {}

            for port, emulator in self._emulators.items():
                try:
                    # Failed state overrides container status
                    if emulator.state == EmulatorState.FAILED:
                        health_status[port] = False
                    elif emulator.container:
                        emulator.container.reload()
                        health_status[port] = emulator.container.status == "running"
                    else:
                        # No container and not failed - consider healthy if available
                        health_status[port] = emulator.state == EmulatorState.AVAILABLE
                except Exception as e:
                    logger.error("Health check failed for port %d: %s", port, e)
                    health_status[port] = False

            logger.debug("Health check completed: %s", health_status)
            return health_status

    def restart_emulator(self, port: int) -> bool:
        """Restart specific Docker container with thread-safe state management"""
        with self._lock:
            if port not in self._emulators:
                logger.error("Cannot restart non-existent emulator on port %d", port)
                return False

            emulator = self._emulators[port]

            if emulator.state == EmulatorState.BUSY:
                logger.warning("Cannot restart busy emulator on port %d", port)
                return False

            logger.info("Restarting Docker container on port %d", port)

            emulator.state = EmulatorState.RESTARTING
            self._available_ports.discard(port)

            try:
                # Stop existing container
                if emulator.container:
                    emulator.container.stop(timeout=5)
                    emulator.container = None

                # Start new container
                new_container = self._start_single_container(port)
                emulator.container = new_container
                emulator.state = EmulatorState.AVAILABLE
                self._available_ports.add(port)

                logger.info("Successfully restarted container on port %d", port)
                self._condition.notify()
                return True

            except Exception as e:
                logger.error("Failed to restart container on port %d: %s", port, e)
                emulator.state = EmulatorState.FAILED
                return False

    def get_status(self) -> dict[str, Any]:
        """Get thread-safe status report with Docker container information"""
        with self._lock:
            status: dict[str, Any] = {
                "initialized": self._initialized,
                "shutdown": self._shutdown,
                "pool_size": self.pool_size,
                "available_count": len(self._available_ports),
                "busy_count": len(self._busy_ports),
                "queue_size": self._acquisition_queue.qsize(),
                "emulators": {},
            }

            for port, emulator in self._emulators.items():
                container_info = {}
                if emulator.container:
                    try:
                        emulator.container.reload()
                        container_info = {
                            "container_id": (
                                emulator.container.id[:12] if emulator.container.id else "unknown"
                            ),
                            "container_status": emulator.container.status,
                        }
                    except Exception as e:
                        container_info = {"container_error": str(e)}

                status["emulators"][port] = {
                    "state": emulator.state.value,
                    "owner_thread": emulator.owner_thread_id,
                    "acquired_at": emulator.acquired_at,
                    **container_info,
                }

            return status

    def shutdown(self) -> None:
        """Gracefully shutdown all Docker containers and clean up thread state"""
        logger.info("Shutting down EmulatorPool...")

        with self._lock:
            if self._shutdown:
                logger.warning("EmulatorPool already shutdown")
                return

            self._shutdown = True

            # Wake up all waiting threads
            self._condition.notify_all()

            # Stop all Docker containers
            containers_to_cleanup = []
            for port, emulator in self._emulators.items():
                if emulator.container:
                    containers_to_cleanup.append(emulator.container)
                    logger.debug("Stopping Docker container on port %d", port)

            # Clean up containers
            self._cleanup_containers(containers_to_cleanup)

            # Clear all state
            self._emulators.clear()
            self._available_ports.clear()
            self._busy_ports.clear()

            # Clear acquisition queue
            while not self._acquisition_queue.empty():
                try:
                    self._acquisition_queue.get_nowait()
                except queue.Empty:
                    break

            self._thread_requests.clear()

            logger.info("EmulatorPool shutdown complete")

    # Private helper methods for Docker container management

    def _get_container_ports(self, count: int) -> list[int]:
        """Calculate sequential port numbers for containers"""
        return [self.base_port + i for i in range(count)]

    def _start_single_container(self, port: int) -> docker.models.containers.Container:
        """
        Start single Docker container with production configuration

        Raises:
            EmulatorPoolError: On startup or health check failure
        """
        try:
            if not self.docker_client:
                raise EmulatorPoolError("Docker client not initialized. Call initialize() first.")

            container = self.docker_client.containers.run(
                image=self.image_name,
                ports={"8080/tcp": port},  # Map internal port 8080 to host port
                detach=True,
                remove=True,  # Auto-cleanup on container stop
                name=f"pokemon-emulator-{port}",
                # Production container configuration
                # NOTE: restart_policy conflicts with remove=True
                mem_limit="512m",  # Prevent memory exhaustion
                cpu_count=1,  # Fair CPU allocation
            )

            # Wait for container to be ready with timeout
            start_time = time.time()
            while time.time() - start_time < self.startup_timeout:
                container.reload()
                if container.status == "running":
                    break
                time.sleep(0.5)
            else:
                container.stop()
                raise EmulatorPoolError(
                    f"Container startup timeout on port {port} after {self.startup_timeout}s"
                )

            # Simple health check - verify container can execute commands
            result = container.exec_run("echo 'health_check'")
            if result.exit_code != 0:
                container.stop()
                raise EmulatorPoolError(f"Container health check failed on port {port}")

            return container

        except ImageNotFound as e:
            raise EmulatorPoolError(
                f"Pokemon-gym Docker image '{self.image_name}' not found. "
                f"Build the image or update image_name parameter."
            ) from e
        except APIError as e:
            if "port is already allocated" in str(e):
                raise EmulatorPoolError(
                    f"Port {port} already in use. Check for conflicting services."
                ) from e
            raise EmulatorPoolError(
                f"Docker API error starting container on port {port}: {e}"
            ) from e

    def _cleanup_containers(self, containers: list[docker.models.containers.Container]) -> None:
        """Clean up Docker containers with proper error handling"""
        if not containers:
            return

        logger.info("Cleaning up %d containers due to shutdown/failure", len(containers))

        for container in containers:
            try:
                container.stop(timeout=5)
                logger.info(
                    "Cleaned up container %s", container.id[:12] if container.id else "unknown"
                )
            except Exception as e:
                # Log but don't raise - we're already in error handling
                logger.error(
                    "Failed to cleanup container %s: %s",
                    container.id[:12] if container.id else "unknown",
                    e,
                )

    # Private helper methods for concurrent access management

    def _get_available_port(self) -> int | None:
        """Get next available port (must hold lock)"""
        if not self._available_ports:
            return None
        return next(iter(self._available_ports))

    def _assign_emulator(self, port: int, thread_id: int) -> PokemonGymClient:
        """Assign emulator to thread (must hold lock)"""
        emulator = self._emulators[port]

        # Update emulator state
        emulator.state = EmulatorState.BUSY
        emulator.owner_thread_id = thread_id
        emulator.acquired_at = time.time()

        # Update port tracking
        self._available_ports.discard(port)
        self._busy_ports.add(port)

        # Set client to Docker container for API compatibility
        emulator.client = emulator.container
        client = emulator.container

        logger.debug("Assigned Docker container on port %d to thread %d", port, thread_id)
        return client

    def _is_highest_priority_thread(self, thread_id: int) -> bool:
        """Check if thread has highest priority among waiting threads (must hold lock)"""
        if thread_id not in self._thread_requests:
            return False

        current_thread_priority = self._thread_requests[thread_id].priority

        # Check all other waiting threads
        for other_thread_id, other_request in self._thread_requests.items():
            if other_thread_id != thread_id:
                # Lower number = higher priority
                if other_request.priority < current_thread_priority:
                    return False
                # If same priority, check timestamp (FIFO)
                elif (
                    other_request.priority == current_thread_priority
                    and other_request.requested_at is not None
                    and self._thread_requests[thread_id].requested_at is not None
                ):
                    # Type-safe comparison after None checks
                    other_time = other_request.requested_at
                    current_time = self._thread_requests[thread_id].requested_at
                    # Both values are guaranteed to be non-None due to checks above
                    assert other_time is not None and current_time is not None
                    if other_time < current_time:
                        return False

        return True

    def _cleanup_request(self, thread_id: int) -> None:
        """Clean up acquisition request (must hold lock)"""
        self._thread_requests.pop(thread_id, None)

        # Remove from queue (expensive but necessary for correctness)
        temp_queue: queue.PriorityQueue[Any] = queue.PriorityQueue()
        while not self._acquisition_queue.empty():
            try:
                item = self._acquisition_queue.get_nowait()
                if item[2].thread_id != thread_id:
                    temp_queue.put(item)
            except queue.Empty:
                break

        self._acquisition_queue = temp_queue
