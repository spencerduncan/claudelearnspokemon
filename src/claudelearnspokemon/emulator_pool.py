"""
EmulatorPool - Thread-safe Pokemon-gym emulator instance management

Provides concurrent access to a pool of emulator instances with proper
synchronization, timeout handling, and deadlock prevention.
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

# Type aliases for future implementation
PokemonGymClient = Any
CompiledScript = Any
ExecutionResult = dict[str, Any]

# Configure logging for concurrent debugging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EmulatorState(Enum):
    """Emulator instance state tracking"""

    AVAILABLE = "available"
    BUSY = "busy"
    FAILED = "failed"
    RESTARTING = "restarting"


@dataclass
class EmulatorInstance:
    """Container for emulator instance state"""

    port: int
    client: PokemonGymClient | None = None
    state: EmulatorState = EmulatorState.AVAILABLE
    owner_thread_id: int | None = None
    acquired_at: float | None = None

    def is_available(self) -> bool:
        """Check if emulator is available for acquisition"""
        return self.state == EmulatorState.AVAILABLE and self.owner_thread_id is None


@dataclass
class AcquisitionRequest:
    """Request for emulator acquisition with priority"""

    thread_id: int
    priority: int = 0
    requested_at: float | None = None
    timeout: float | None = None

    def __post_init__(self) -> None:
        if self.requested_at is None:
            self.requested_at = time.time()


class EmulatorPool:
    """
    Thread-safe pool manager for Pokemon-gym emulator instances

    Provides concurrent access with proper locking, queue management,
    timeout handling, and deadlock prevention.
    """

    def __init__(self, pool_size: int = 4, default_timeout: float = 30.0) -> None:
        """Initialize emulator pool with configuration parameters"""
        self.pool_size = pool_size
        self.default_timeout = default_timeout

        # Core synchronization primitives
        self._lock = threading.RLock()  # Reentrant for deadlock prevention
        self._condition = threading.Condition(self._lock)

        # Emulator tracking
        self._emulators: dict[int, EmulatorInstance] = {}
        self._available_ports: set[int] = set()
        self._busy_ports: set[int] = set()

        # Acquisition queue management
        self._acquisition_queue: queue.PriorityQueue[Any] = queue.PriorityQueue()
        self._thread_requests: dict[int, AcquisitionRequest] = {}

        # State tracking
        self._initialized = False
        self._shutdown = False

        # Thread-local storage for debugging context
        self._thread_local = threading.local()

        logger.info(
            "EmulatorPool initialized with size=%d, timeout=%.1f",
            pool_size,
            default_timeout,
        )

    def initialize(self, pool_size: int | None = None) -> None:
        """Initialize emulator pool with Docker containers"""
        with self._lock:
            if self._initialized:
                logger.warning("EmulatorPool already initialized")
                return

            if pool_size is not None:
                self.pool_size = pool_size

            logger.info("Initializing %d emulator instances...", self.pool_size)

            # Create emulator instances on sequential ports
            base_port = 8081
            for i in range(self.pool_size):
                port = base_port + i
                emulator = EmulatorInstance(port=port)
                self._emulators[port] = emulator
                self._available_ports.add(port)

                # TODO: Start Docker container
                logger.debug("Created emulator instance on port %d", port)

            self._initialized = True
            self._condition.notify_all()  # Wake up any waiting threads

            logger.info("EmulatorPool initialized with %d instances", len(self._emulators))

    def acquire(self, timeout: float | None = None, priority: int = 0) -> PokemonGymClient | None:
        """Acquire an available emulator with thread-safe queuing"""
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
        """Release emulator back to available pool"""
        thread_id = threading.get_ident()

        with self._lock:
            if self._shutdown:
                logger.debug("Thread %d release during shutdown", thread_id)
                return

            # Find emulator by client
            port = None
            for p, emulator in self._emulators.items():
                if emulator.client is client:
                    port = p
                    break

            if port is None:
                logger.error("Thread %d attempted to release unknown client", thread_id)
                return

            emulator = self._emulators[port]

            # Verify ownership
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
        """Execute script on specific emulator instance"""
        thread_id = threading.get_ident()

        with self._lock:
            # Verify client ownership
            port = None
            for p, emulator in self._emulators.items():
                if emulator.client is client and emulator.owner_thread_id == thread_id:
                    port = p
                    break

            if port is None:
                raise ValueError(f"Thread {thread_id} does not own the specified client")

        logger.debug("Thread %d executing script on port %d", thread_id, port)

        # TODO: Implement actual script execution
        # This would interface with the Pokemon-gym Docker container

        # Placeholder implementation
        return {
            "success": True,
            "port": port,
            "thread_id": thread_id,
            "script_id": getattr(script, "id", "unknown"),
            "checkpoint_id": checkpoint_id,
        }

    def health_check(self) -> dict[int, bool]:
        """Verify all emulators are responsive"""
        with self._lock:
            health_status = {}

            for port, emulator in self._emulators.items():
                # TODO: Implement actual health check
                health_status[port] = emulator.state != EmulatorState.FAILED

            logger.debug("Health check completed: %s", health_status)
            return health_status

    def restart_emulator(self, port: int) -> bool:
        """Restart specific emulator instance"""
        with self._lock:
            if port not in self._emulators:
                logger.error("Cannot restart non-existent emulator on port %d", port)
                return False

            emulator = self._emulators[port]

            if emulator.state == EmulatorState.BUSY:
                logger.warning("Cannot restart busy emulator on port %d", port)
                return False

            logger.info("Restarting emulator on port %d", port)

            emulator.state = EmulatorState.RESTARTING
            self._available_ports.discard(port)

            try:
                # TODO: Implement actual Docker container restart
                time.sleep(0.1)  # Simulate restart time

                emulator.state = EmulatorState.AVAILABLE
                self._available_ports.add(port)

                logger.info("Successfully restarted emulator on port %d", port)
                self._condition.notify()
                return True

            except Exception as e:
                logger.error("Failed to restart emulator on port %d: %s", port, e)
                emulator.state = EmulatorState.FAILED
                return False

    def get_status(self) -> dict[str, Any]:
        """Get thread-safe status report"""
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
                status["emulators"][port] = {
                    "state": emulator.state.value,
                    "owner_thread": emulator.owner_thread_id,
                    "acquired_at": emulator.acquired_at,
                }

            return status

    def shutdown(self) -> None:
        """Gracefully shutdown emulator pool"""
        logger.info("Shutting down EmulatorPool...")

        with self._lock:
            if self._shutdown:
                logger.warning("EmulatorPool already shutdown")
                return

            self._shutdown = True

            # Wake up all waiting threads
            self._condition.notify_all()

            # TODO: Stop all Docker containers
            for port in list(self._emulators.keys()):
                logger.debug("Stopping emulator container on port %d", port)

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

    # Private helper methods

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

        # TODO: Create actual PokemonGymClient
        client = f"PokemonGymClient(port={port})"
        emulator.client = client

        logger.debug("Assigned emulator on port %d to thread %d", port, thread_id)
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
                    and other_request.requested_at < self._thread_requests[thread_id].requested_at
                ):
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
