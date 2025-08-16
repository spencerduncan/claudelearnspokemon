"""
CheckpointManager - Thread-safe game state checkpoint management.

Thread-safe checkpoint manager with comprehensive functionality:
- LZ4 compression for efficiency
- UUID-based checkpoint identifiers
- Atomic write operations with file locking
- Reader-writer locks for concurrent access
- Per-checkpoint locks for fine-grained control
- Performance target: <500ms for save/load operations
- Cross-platform file locking (Unix fcntl, Windows msvcrt)
- Deadlock prevention with timeout-based acquisition
"""

import fcntl  # Unix file locking
import json
import os
import platform
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Condition, Lock, RLock
from typing import Any

import lz4.frame
import structlog

# Windows-specific imports
if platform.system() == "Windows":
    import msvcrt
else:
    # Type: ignore for cross-platform compatibility
    import types

    msvcrt = types.ModuleType("msvcrt")
    msvcrt.LK_NBLCK = 1  # type: ignore
    msvcrt.LK_UNLCK = 2  # type: ignore

    def dummy_locking(fd: int, mode: int, nbytes: int) -> None:
        pass

    msvcrt.locking = dummy_locking  # type: ignore

logger = structlog.get_logger(__name__)


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""

    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint cannot be found."""

    pass


class CheckpointCorruptionError(CheckpointError):
    """Raised when a checkpoint is corrupted or invalid."""

    pass


class CheckpointLockTimeoutError(CheckpointError):
    """Raised when lock acquisition times out."""

    pass


class ReadWriteLock:
    """Reader-writer lock with writer preference to prevent starvation.

    Allows multiple concurrent readers OR one exclusive writer.
    Writers have preference to prevent writer starvation.
    """

    def __init__(self):
        self._readers_lock = Lock()  # Protects reader count
        self._writers_lock = Lock()  # Only one writer at a time
        self._readers_count = 0
        self._writers_waiting = 0
        self._writer_condition = Condition(self._writers_lock)

    @contextmanager
    def read_lock(self, timeout: float = 30.0) -> Iterator[None]:
        """Acquire read lock with timeout."""
        acquired = False
        try:
            # Check for waiting writers (writer preference)
            if not self._writers_lock.acquire(timeout=timeout):
                raise CheckpointLockTimeoutError("Read lock timeout: writers waiting")

            try:
                # If no writers waiting, allow readers
                if self._writers_waiting == 0:
                    with self._readers_lock:
                        self._readers_count += 1
                        acquired = True
                else:
                    # Writers are waiting - block readers to prevent starvation
                    raise CheckpointLockTimeoutError("Read lock timeout: writer preference")
            finally:
                self._writers_lock.release()

            yield

        finally:
            if acquired:
                with self._readers_lock:
                    self._readers_count -= 1
                    if self._readers_count == 0:
                        # Last reader - notify waiting writers
                        with self._writer_condition:
                            self._writer_condition.notify_all()

    @contextmanager
    def write_lock(self, timeout: float = 30.0) -> Iterator[None]:
        """Acquire write lock with timeout."""
        acquired = False
        try:
            if not self._writers_lock.acquire(timeout=timeout):
                raise CheckpointLockTimeoutError("Write lock timeout")
            acquired = True

            # Signal that a writer is waiting
            self._writers_waiting += 1

            try:
                # Wait for all readers to finish
                with self._writer_condition:
                    start_time = time.time()
                    while self._readers_count > 0:
                        remaining = timeout - (time.time() - start_time)
                        if remaining <= 0:
                            raise CheckpointLockTimeoutError("Write lock timeout: readers active")
                        self._writer_condition.wait(timeout=remaining)

                # Now have exclusive access
                yield

            finally:
                self._writers_waiting -= 1

        finally:
            if acquired:
                self._writers_lock.release()


class FileLock:
    """Cross-platform exclusive file locking.

    Uses fcntl on Unix systems and msvcrt on Windows.
    """

    def __init__(self, file_path: Path, timeout: float = 30.0):
        self.file_path = file_path
        self.timeout = timeout
        self._lock_file: Any = None
        self._lock_fd: int | None = None

    @contextmanager
    def exclusive_lock(self) -> Iterator[None]:
        """Acquire exclusive file lock with timeout."""
        lock_file_path = self.file_path.with_suffix(".lock")

        try:
            # Open lock file
            self._lock_file = lock_file_path.open("w")
            self._lock_fd = self._lock_file.fileno()

            # Platform-specific locking
            start_time = time.time()
            locked = False

            while not locked and (time.time() - start_time) < self.timeout:
                try:
                    if platform.system() == "Windows":
                        # Windows locking
                        msvcrt.locking(self._lock_fd, msvcrt.LK_NBLCK, 1)  # type: ignore
                        locked = True
                    else:
                        # Unix locking
                        fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = True
                except OSError:
                    # Lock busy - wait a bit
                    time.sleep(0.01)

            if not locked:
                raise CheckpointLockTimeoutError(f"File lock timeout: {self.file_path}")

            # Write process info for debugging
            self._lock_file.write(f"pid:{os.getpid()}\nthread:{threading.get_ident()}\n")
            self._lock_file.flush()

            yield

        finally:
            # Release lock
            if self._lock_file:
                try:
                    if platform.system() == "Windows" and self._lock_fd:
                        msvcrt.locking(self._lock_fd, msvcrt.LK_UNLCK, 1)  # type: ignore
                    # fcntl locks are released automatically when file is closed
                    self._lock_file.close()
                except OSError:
                    pass  # Ignore errors during cleanup

            # Remove lock file
            try:
                if lock_file_path.exists():
                    lock_file_path.unlink()
            except OSError:
                pass  # Ignore cleanup errors


class CheckpointManager:
    """
    Thread-safe Pokemon game state checkpoint manager with LZ4 compression.

    Provides atomic save/load operations with comprehensive thread safety:
    - ReadWriteLock for metadata operations
    - FileLock for exclusive file access
    - Per-checkpoint locks for fine-grained control
    - Cross-platform file locking
    - Deadlock prevention with timeouts

    Designed for high-frequency use in parallel execution environments.
    """

    def __init__(self, checkpoint_dir: str | None = None):
        """
        Initialize thread-safe checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint storage.
                          Defaults to ~/.claudelearnspokemon/checkpoints
        """
        if checkpoint_dir is None:
            self.checkpoint_dir = Path.home() / ".claudelearnspokemon" / "checkpoints"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety infrastructure
        self._metadata_lock = ReadWriteLock()  # For metadata operations
        self._checkpoint_locks: dict[str, RLock] = {}  # Per-checkpoint locks
        self._checkpoint_locks_lock = Lock()  # Protects checkpoint_locks dict

        # Performance tracking (thread-safe lists)
        self._save_times: list[float] = []
        self._load_times: list[float] = []
        self._stats_lock = Lock()  # Protects performance stats

        logger.info(
            "Thread-safe CheckpointManager initialized",
            checkpoint_dir=str(self.checkpoint_dir),
            thread_id=threading.get_ident(),
        )

    def _get_checkpoint_lock(self, checkpoint_id: str) -> RLock:
        """Get or create per-checkpoint lock for fine-grained locking."""
        with self._checkpoint_locks_lock:
            if checkpoint_id not in self._checkpoint_locks:
                self._checkpoint_locks[checkpoint_id] = RLock()
            return self._checkpoint_locks[checkpoint_id]

    def save_checkpoint(self, game_state: dict[str, Any], metadata: dict[str, Any]) -> str:
        """
        Thread-safe save of game state with metadata to compressed checkpoint file.

        Args:
            game_state: Complete game state dictionary
            metadata: Checkpoint metadata (location, progress, etc.)

        Returns:
            Checkpoint identifier (UUID string)

        Raises:
            CheckpointError: If save operation fails
            CheckpointLockTimeoutError: If locks cannot be acquired
        """
        start_time = time.monotonic()

        # Generate unique checkpoint ID
        checkpoint_id = str(uuid.uuid4())
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.lz4"

        # Get checkpoint-specific lock for fine-grained control
        checkpoint_lock = self._get_checkpoint_lock(checkpoint_id)

        try:
            # Acquire per-checkpoint lock with timeout
            if not checkpoint_lock.acquire(timeout=30.0):
                raise CheckpointLockTimeoutError(
                    f"Could not acquire checkpoint lock: {checkpoint_id}"
                )

            try:
                # Use file-level locking for atomic write protection
                with FileLock(checkpoint_file, timeout=30.0).exclusive_lock():
                    # Prepare checkpoint data structure
                    checkpoint_data = {
                        "version": "1.0",
                        "checkpoint_id": checkpoint_id,
                        "timestamp": time.time(),
                        "thread_id": threading.get_ident(),
                        "game_state": game_state,
                        "metadata": metadata,
                    }

                    # Serialize to JSON
                    json_data = json.dumps(checkpoint_data, separators=(",", ":"))
                    json_bytes = json_data.encode("utf-8")

                    # Compress with LZ4 for speed
                    compressed_data = lz4.frame.compress(json_bytes)

                    # Atomic write: write to temp file, then rename
                    temp_file = checkpoint_file.with_suffix(".tmp")
                    try:
                        with temp_file.open("wb") as f:
                            f.write(compressed_data)
                            f.flush()
                            os.fsync(f.fileno())  # Ensure data reaches disk

                        # Atomic rename - critical for thread safety
                        temp_file.rename(checkpoint_file)

                    except Exception:
                        # Cleanup temp file on failure
                        if temp_file.exists():
                            temp_file.unlink()
                        raise

                    # Track performance (thread-safe)
                    duration = time.monotonic() - start_time
                    with self._stats_lock:
                        self._save_times.append(duration)

                    logger.info(
                        "Checkpoint saved",
                        checkpoint_id=checkpoint_id,
                        thread_id=threading.get_ident(),
                        duration_ms=int(duration * 1000),
                        compressed_size=len(compressed_data),
                        original_size=len(json_bytes),
                    )

                    return checkpoint_id

            finally:
                checkpoint_lock.release()

        except Exception as e:
            logger.error(
                "Failed to save checkpoint",
                error=str(e),
                checkpoint_id=checkpoint_id,
                thread_id=threading.get_ident(),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )
            if isinstance(e, CheckpointLockTimeoutError):
                raise
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    def load_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """
        Thread-safe load of game state from compressed checkpoint file.

        Args:
            checkpoint_id: Checkpoint identifier (UUID string)

        Returns:
            Game state dictionary

        Raises:
            CheckpointNotFoundError: If checkpoint file doesn't exist
            CheckpointCorruptionError: If checkpoint is corrupted or invalid
            CheckpointLockTimeoutError: If locks cannot be acquired
        """
        start_time = time.monotonic()

        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.lz4"

        # Use read lock for concurrent read access
        with self._metadata_lock.read_lock(timeout=30.0):
            if not checkpoint_file.exists():
                raise CheckpointNotFoundError(f"Checkpoint {checkpoint_id} not found")

        # Get checkpoint-specific lock
        checkpoint_lock = self._get_checkpoint_lock(checkpoint_id)

        try:
            if not checkpoint_lock.acquire(timeout=30.0):
                raise CheckpointLockTimeoutError(
                    f"Could not acquire checkpoint lock: {checkpoint_id}"
                )

            try:
                # Read compressed data with file lock for consistency
                with FileLock(checkpoint_file, timeout=30.0).exclusive_lock():
                    with checkpoint_file.open("rb") as f:
                        compressed_data = f.read()

                if not compressed_data:
                    raise CheckpointCorruptionError(f"Checkpoint {checkpoint_id} is empty")

                # Decompress
                try:
                    json_bytes = lz4.frame.decompress(compressed_data)
                except (RuntimeError, Exception) as e:
                    # LZ4 raises RuntimeError for decompression failures
                    if "LZ4F_" in str(e) or "decompress" in str(e).lower():
                        raise CheckpointCorruptionError(
                            f"Failed to decompress checkpoint {checkpoint_id}"
                        ) from e
                    else:
                        raise

                # Parse JSON
                try:
                    json_data = json_bytes.decode("utf-8")
                    checkpoint_data = json.loads(json_data)
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    raise CheckpointCorruptionError(
                        f"Failed to parse checkpoint {checkpoint_id}"
                    ) from e

                # Validate checkpoint structure
                required_fields = ["version", "checkpoint_id", "game_state", "metadata"]
                for field in required_fields:
                    if field not in checkpoint_data:
                        raise CheckpointCorruptionError(
                            f"Checkpoint {checkpoint_id} missing field: {field}"
                        )

                # Validate checkpoint ID matches
                if checkpoint_data["checkpoint_id"] != checkpoint_id:
                    raise CheckpointCorruptionError(
                        f"Checkpoint ID mismatch: file {checkpoint_id}, content {checkpoint_data['checkpoint_id']}"
                    )

                # Extract game state
                game_state: dict[str, Any] = checkpoint_data["game_state"]

                # Track performance (thread-safe)
                duration = time.monotonic() - start_time
                with self._stats_lock:
                    self._load_times.append(duration)

                logger.info(
                    "Checkpoint loaded",
                    checkpoint_id=checkpoint_id,
                    thread_id=threading.get_ident(),
                    duration_ms=int(duration * 1000),
                    compressed_size=len(compressed_data),
                    decompressed_size=len(json_bytes),
                )

                return game_state

            finally:
                checkpoint_lock.release()

        except (CheckpointNotFoundError, CheckpointCorruptionError, CheckpointLockTimeoutError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            logger.error(
                "Failed to load checkpoint",
                error=str(e),
                checkpoint_id=checkpoint_id,
                thread_id=threading.get_ident(),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )
            raise CheckpointError(f"Failed to load checkpoint {checkpoint_id}: {e}") from e

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Thread-safe get performance statistics for checkpoint operations.

        Returns:
            Dictionary with performance metrics
        """
        with self._stats_lock:
            # Create copies to avoid race conditions
            save_times = list(self._save_times)
            load_times = list(self._load_times)

        stats = {
            "save_operations": len(save_times),
            "load_operations": len(load_times),
            "thread_info": {
                "current_thread_id": threading.get_ident(),
                "active_checkpoint_locks": len(self._checkpoint_locks),
            },
        }

        if save_times:
            stats.update(
                {
                    "avg_save_time_ms": int(sum(save_times) * 1000 / len(save_times)),
                    "max_save_time_ms": int(max(save_times) * 1000),
                    "min_save_time_ms": int(min(save_times) * 1000),
                }
            )

        if load_times:
            stats.update(
                {
                    "avg_load_time_ms": int(sum(load_times) * 1000 / len(load_times)),
                    "max_load_time_ms": int(max(load_times) * 1000),
                    "min_load_time_ms": int(min(load_times) * 1000),
                }
            )

        return stats

    def checkpoint_exists(self, checkpoint_id: str) -> bool:
        """
        Thread-safe check if a checkpoint exists.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            True if checkpoint exists, False otherwise
        """
        with self._metadata_lock.read_lock(timeout=30.0):
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.lz4"
            return checkpoint_file.exists()

    def get_checkpoint_size(self, checkpoint_id: str) -> int:
        """
        Thread-safe get compressed size of checkpoint file.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Size in bytes

        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
            CheckpointLockTimeoutError: If locks cannot be acquired
        """
        with self._metadata_lock.read_lock(timeout=30.0):
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.lz4"

            if not checkpoint_file.exists():
                raise CheckpointNotFoundError(f"Checkpoint {checkpoint_id} not found")

            return checkpoint_file.stat().st_size

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Thread-safe delete checkpoint file.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            CheckpointLockTimeoutError: If locks cannot be acquired
        """
        checkpoint_lock = self._get_checkpoint_lock(checkpoint_id)

        try:
            if not checkpoint_lock.acquire(timeout=30.0):
                raise CheckpointLockTimeoutError(
                    f"Could not acquire checkpoint lock: {checkpoint_id}"
                )

            try:
                with self._metadata_lock.write_lock(timeout=30.0):
                    checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.lz4"

                    if not checkpoint_file.exists():
                        logger.warning("Checkpoint already deleted", checkpoint_id=checkpoint_id)
                        return False

                    # Use file lock during deletion
                    try:
                        with FileLock(checkpoint_file, timeout=30.0).exclusive_lock():
                            checkpoint_file.unlink()
                    except CheckpointLockTimeoutError:
                        # File might be in use - try without lock
                        checkpoint_file.unlink()

                    logger.info("Checkpoint deleted", checkpoint_id=checkpoint_id)
                    return True

            finally:
                checkpoint_lock.release()

        except Exception as e:
            logger.error("Failed to delete checkpoint", error=str(e), checkpoint_id=checkpoint_id)
            return False

    def list_checkpoints(self) -> list[str]:
        """
        Thread-safe list all checkpoint IDs.

        Returns:
            List of checkpoint IDs
        """
        with self._metadata_lock.read_lock(timeout=30.0):
            checkpoint_files = list(self.checkpoint_dir.glob("*.lz4"))
            return [f.stem for f in checkpoint_files]

    def cleanup_orphaned_locks(self) -> int:
        """
        Clean up orphaned checkpoint locks (for maintenance).

        Returns:
            Number of locks cleaned up
        """
        with self._checkpoint_locks_lock:
            existing_checkpoints = set(self.list_checkpoints())
            orphaned_locks = []

            for checkpoint_id in self._checkpoint_locks:
                if checkpoint_id not in existing_checkpoints:
                    orphaned_locks.append(checkpoint_id)

            for checkpoint_id in orphaned_locks:
                del self._checkpoint_locks[checkpoint_id]

            if orphaned_locks:
                logger.info(
                    "Cleaned up orphaned locks",
                    count=len(orphaned_locks),
                    checkpoint_ids=orphaned_locks,
                )

            return len(orphaned_locks)
