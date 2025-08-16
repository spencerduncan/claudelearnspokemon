"""
Unit tests for thread-safe CheckpointManager functionality.

Tests all aspects of checkpoint storage with comprehensive thread safety validation,
performance requirements, and concurrent access scenarios.
"""

import json
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lz4.frame
import pytest
from claudelearnspokemon.checkpoint_manager import (
    CheckpointCorruptionError,
    CheckpointLockTimeoutError,
    CheckpointManager,
    CheckpointNotFoundError,
    FileLock,
    ReadWriteLock,
)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create CheckpointManager with temporary directory."""
    return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)


@pytest.fixture
def sample_game_state():
    """Sample game state data."""
    return {
        "player": {"name": "RED", "position": {"x": 100, "y": 150}, "level": 25, "health": 80},
        "pokemon": [
            {
                "name": "PIKACHU",
                "level": 25,
                "health": 65,
                "moves": ["THUNDERBOLT", "QUICK_ATTACK"],
            },
            {"name": "CHARMANDER", "level": 12, "health": 39, "moves": ["SCRATCH", "GROWL"]},
        ],
        "inventory": {"pokeball": 10, "potion": 5, "rare_candy": 1},
        "flags": {"gym_badges": ["boulder", "cascade"], "story_progress": "cerulean_city"},
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "game_location": "pallet_town",
        "progress_markers": ["got_starter", "visited_oak"],
        "performance_metrics": {"execution_time": 5.2, "success_rate": 0.85},
        "tags": ["tutorial", "important"],
        "custom_fields": {"strategy_type": "speedrun", "difficulty": "normal"},
    }


class TestCheckpointManagerBasics:
    """Basic functionality tests."""

    def test_initialization(self, temp_checkpoint_dir):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        assert manager.checkpoint_dir == Path(temp_checkpoint_dir)
        assert manager.checkpoint_dir.exists()
        assert isinstance(manager._metadata_lock, ReadWriteLock)
        assert len(manager._checkpoint_locks) == 0

    def test_save_and_load_roundtrip(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test basic save and load functionality."""
        # Save checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        assert checkpoint_id is not None
        assert len(checkpoint_id) == 36  # UUID length

        # Verify file exists
        assert checkpoint_manager.checkpoint_exists(checkpoint_id)

        # Load and verify data
        loaded_state = checkpoint_manager.load_checkpoint(checkpoint_id)
        assert loaded_state == sample_game_state

    def test_checkpoint_not_found(self, checkpoint_manager):
        """Test loading non-existent checkpoint raises proper exception."""
        fake_id = "nonexistent-checkpoint-id"

        with pytest.raises(CheckpointNotFoundError):
            checkpoint_manager.load_checkpoint(fake_id)

        assert not checkpoint_manager.checkpoint_exists(fake_id)

    def test_performance_stats(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test performance statistics tracking."""
        # Initial stats should be empty
        stats = checkpoint_manager.get_performance_stats()
        assert stats["save_operations"] == 0
        assert stats["load_operations"] == 0

        # Save a checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Load the checkpoint
        checkpoint_manager.load_checkpoint(checkpoint_id)

        # Check updated stats
        stats = checkpoint_manager.get_performance_stats()
        assert stats["save_operations"] == 1
        assert stats["load_operations"] == 1
        assert "avg_save_time_ms" in stats
        assert "avg_load_time_ms" in stats
        assert "thread_info" in stats

    def test_get_checkpoint_size(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test getting checkpoint file size."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        size = checkpoint_manager.get_checkpoint_size(checkpoint_id)
        assert size > 0
        assert isinstance(size, int)

    def test_delete_checkpoint(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test checkpoint deletion."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        assert checkpoint_manager.checkpoint_exists(checkpoint_id)

        success = checkpoint_manager.delete_checkpoint(checkpoint_id)
        assert success is True
        assert not checkpoint_manager.checkpoint_exists(checkpoint_id)

    def test_list_checkpoints(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test listing checkpoints."""
        # Initially empty
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 0

        # Save some checkpoints
        id1 = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 2
        assert id1 in checkpoints
        assert id2 in checkpoints


class TestThreadSafetyInfrastructure:
    """Test the thread safety mechanisms themselves."""

    def test_read_write_lock_concurrent_readers(self):
        """Test ReadWriteLock allows concurrent readers."""
        lock = ReadWriteLock()
        results = []

        def reader_task(reader_id):
            with lock.read_lock():
                results.append(f"reader_{reader_id}_start")
                time.sleep(0.1)  # Hold lock briefly
                results.append(f"reader_{reader_id}_end")

        # Start multiple readers concurrently
        threads = []
        for i in range(5):
            t = threading.Thread(target=reader_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All readers should have been able to run concurrently
        assert len([r for r in results if "start" in r]) == 5
        assert len([r for r in results if "end" in r]) == 5

    def test_read_write_lock_writer_exclusion(self):
        """Test ReadWriteLock provides writer exclusion."""
        lock = ReadWriteLock()
        results = []

        def writer_task():
            with lock.write_lock():
                results.append("writer_start")
                time.sleep(0.2)
                results.append("writer_end")

        def reader_task():
            # This should wait for writer to finish
            with lock.read_lock():
                results.append("reader_after_writer")

        # Start writer first, then reader
        writer_thread = threading.Thread(target=writer_task)
        reader_thread = threading.Thread(target=reader_task)

        writer_thread.start()
        time.sleep(0.05)  # Small delay to ensure writer gets lock first
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        # Writer should complete before reader starts
        assert results == ["writer_start", "writer_end", "reader_after_writer"]

    def test_file_lock_exclusive_access(self, temp_checkpoint_dir):
        """Test FileLock provides exclusive file access."""
        test_file = Path(temp_checkpoint_dir) / "test_lock.txt"
        results = []

        def lock_task(task_id):
            try:
                with FileLock(test_file).exclusive_lock():
                    results.append(f"task_{task_id}_acquired")
                    time.sleep(0.1)
                    results.append(f"task_{task_id}_released")
            except CheckpointLockTimeoutError:
                results.append(f"task_{task_id}_timeout")

        # Start two tasks that try to lock the same file
        threads = []
        for i in range(2):
            t = threading.Thread(target=lock_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Only one task should have acquired the lock
        acquired_count = len([r for r in results if "acquired" in r])
        assert acquired_count == 1


class TestConcurrentOperations:
    """Test concurrent checkpoint operations."""

    def test_concurrent_saves_different_checkpoints(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test concurrent saves of different checkpoints."""
        results = []

        def save_task(task_id):
            # Modify metadata slightly to make each save unique
            metadata = sample_metadata.copy()
            metadata["task_id"] = task_id

            try:
                checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, metadata)
                results.append(("success", task_id, checkpoint_id))
            except Exception as e:
                results.append(("error", task_id, type(e).__name__))

        # Start multiple concurrent saves
        threads = []
        num_tasks = 8
        for i in range(num_tasks):
            t = threading.Thread(target=save_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All saves should succeed with unique IDs
        successful_saves = [r for r in results if r[0] == "success"]
        assert len(successful_saves) == num_tasks

        checkpoint_ids = [r[2] for r in successful_saves]
        assert len(set(checkpoint_ids)) == num_tasks  # All IDs should be unique

    def test_concurrent_load_same_checkpoint(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test concurrent loads of the same checkpoint."""
        # First create a checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        results = []

        def load_task(task_id):
            try:
                loaded_state = checkpoint_manager.load_checkpoint(checkpoint_id)
                results.append(("success", task_id, loaded_state == sample_game_state))
            except Exception as e:
                results.append(("error", task_id, type(e).__name__))

        # Start multiple concurrent loads
        threads = []
        num_tasks = 5
        for i in range(num_tasks):
            t = threading.Thread(target=load_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All loads should succeed
        successful_loads = [r for r in results if r[0] == "success"]
        assert len(successful_loads) == num_tasks

        # All should have loaded correct data
        all_data_correct = all(r[2] for r in successful_loads)
        assert all_data_correct

    def test_concurrent_save_delete_operations(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test concurrent save and delete operations."""
        results = []

        def save_and_delete_task(task_id):
            try:
                # Save checkpoint
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    sample_game_state, sample_metadata
                )
                results.append(("saved", task_id, checkpoint_id))

                # Small delay
                time.sleep(0.05)

                # Try to delete it
                deleted = checkpoint_manager.delete_checkpoint(checkpoint_id)
                results.append(("deleted", task_id, deleted))

            except Exception as e:
                results.append(("error", task_id, type(e).__name__))

        # Start multiple concurrent save/delete operations
        threads = []
        num_tasks = 3
        for i in range(num_tasks):
            t = threading.Thread(target=save_and_delete_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have some successful saves and deletes
        saves = [r for r in results if r[0] == "saved"]
        deletes = [r for r in results if r[0] == "deleted"]

        assert len(saves) >= num_tasks
        assert len(deletes) >= num_tasks

    def test_high_concurrency_stress(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test high concurrency with mixed operations."""
        results = []
        shared_checkpoint_ids = []
        lock = threading.Lock()

        def mixed_operations_task(task_id):
            try:
                # Save operation
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    sample_game_state, sample_metadata
                )
                with lock:
                    shared_checkpoint_ids.append(checkpoint_id)
                results.append(("save_success", task_id))

                # Load operation on own checkpoint
                loaded_state = checkpoint_manager.load_checkpoint(checkpoint_id)
                assert loaded_state == sample_game_state
                results.append(("load_success", task_id))

                # Try to load from shared pool if available
                with lock:
                    if len(shared_checkpoint_ids) > 1:
                        other_id = (
                            shared_checkpoint_ids[0]
                            if shared_checkpoint_ids[0] != checkpoint_id
                            else shared_checkpoint_ids[1]
                        )
                        loaded_state = checkpoint_manager.load_checkpoint(other_id)
                        results.append(("shared_load_success", task_id))

            except Exception:
                results.append(("error", task_id))

        # Run high-concurrency stress test
        threads = []
        num_tasks = 20

        start_time = time.time()

        for i in range(num_tasks):
            t = threading.Thread(target=mixed_operations_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        duration = time.time() - start_time

        # Analyze results
        errors = [r for r in results if r[0] == "error"]
        save_successes = [r for r in results if r[0] == "save_success"]
        load_successes = [r for r in results if r[0] == "load_success"]

        # Should complete in reasonable time with no errors
        assert duration < 10.0  # Should complete within 10 seconds
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(save_successes) == num_tasks
        assert len(load_successes) == num_tasks

        # Check performance requirements
        stats = checkpoint_manager.get_performance_stats()
        assert stats["avg_save_time_ms"] < 500  # <500ms requirement
        assert stats["avg_load_time_ms"] < 500  # <500ms requirement

    def test_deadlock_prevention(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test that operations don't deadlock under concurrent load."""
        results = []

        def complex_operations_task(task_id):
            try:
                # Create checkpoint
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    sample_game_state, sample_metadata
                )

                # Mixed operations that could potentially deadlock
                checkpoint_manager.checkpoint_exists(checkpoint_id)
                checkpoint_manager.get_checkpoint_size(checkpoint_id)
                checkpoint_manager.load_checkpoint(checkpoint_id)

                if task_id % 2 == 0:  # Delete some checkpoints
                    checkpoint_manager.delete_checkpoint(checkpoint_id)

                results.append(("completed", task_id))

            except CheckpointLockTimeoutError:
                results.append(("timeout", task_id))
            except Exception:
                results.append(("error", task_id))

        # Start many concurrent complex operations
        threads = []
        num_tasks = 15

        start_time = time.time()

        for i in range(num_tasks):
            t = threading.Thread(target=complex_operations_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=15.0)  # Don't wait forever

        duration = time.time() - start_time

        # Should complete without deadlocks
        assert duration < 15.0

        completed = [r for r in results if r[0] == "completed"]
        timeouts = [r for r in results if r[0] == "timeout"]
        errors = [r for r in results if r[0] == "error"]

        # Most operations should complete successfully
        assert len(completed) >= num_tasks * 0.8  # At least 80% success
        assert len(errors) == 0, f"Unexpected errors: {errors}"

        # Some timeouts are acceptable under high load
        if timeouts:
            assert len(timeouts) < num_tasks * 0.2  # Less than 20% timeouts


class TestCorruptionHandling:
    """Test handling of corrupted checkpoint files."""

    def test_corrupted_lz4_data(self, checkpoint_manager, temp_checkpoint_dir):
        """Test handling of corrupted LZ4 data."""
        # Create a file with invalid LZ4 data
        fake_id = "corrupted-checkpoint"
        corrupted_file = Path(temp_checkpoint_dir) / f"{fake_id}.lz4"
        corrupted_file.write_bytes(b"invalid lz4 data here")

        with pytest.raises(CheckpointCorruptionError):
            checkpoint_manager.load_checkpoint(fake_id)

    def test_corrupted_json_data(self, checkpoint_manager, temp_checkpoint_dir):
        """Test handling of corrupted JSON data."""
        # Create a file with valid LZ4 but invalid JSON
        fake_id = "corrupted-json"
        corrupted_file = Path(temp_checkpoint_dir) / f"{fake_id}.lz4"

        invalid_json = b"invalid json content"
        compressed_data = lz4.frame.compress(invalid_json)
        corrupted_file.write_bytes(compressed_data)

        with pytest.raises(CheckpointCorruptionError):
            checkpoint_manager.load_checkpoint(fake_id)

    def test_missing_required_fields(self, checkpoint_manager, temp_checkpoint_dir):
        """Test handling of JSON missing required fields."""
        fake_id = "missing-fields"
        corrupted_file = Path(temp_checkpoint_dir) / f"{fake_id}.lz4"

        # Create JSON missing required fields
        incomplete_data = {"version": "1.0"}  # Missing required fields
        json_data = json.dumps(incomplete_data).encode("utf-8")
        compressed_data = lz4.frame.compress(json_data)
        corrupted_file.write_bytes(compressed_data)

        with pytest.raises(CheckpointCorruptionError):
            checkpoint_manager.load_checkpoint(fake_id)


class TestPerformanceRequirements:
    """Test performance requirements are met."""

    def test_save_performance_requirement(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test save operations meet <500ms requirement."""
        start_time = time.monotonic()
        checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        duration = time.monotonic() - start_time

        assert duration < 0.5  # <500ms requirement

    def test_load_performance_requirement(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test load operations meet <500ms requirement."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        start_time = time.monotonic()
        checkpoint_manager.load_checkpoint(checkpoint_id)
        duration = time.monotonic() - start_time

        assert duration < 0.5  # <500ms requirement

    def test_performance_under_concurrent_load(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test performance requirements are maintained under concurrent load."""

        def timed_operations():
            # Save
            start = time.monotonic()
            checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
            save_time = time.monotonic() - start

            # Load
            start = time.monotonic()
            checkpoint_manager.load_checkpoint(checkpoint_id)
            load_time = time.monotonic() - start

            return save_time, load_time

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(timed_operations) for _ in range(20)]

            for future in as_completed(futures):
                save_time, load_time = future.result()

                # Each operation should meet performance requirements
                assert save_time < 0.5, f"Save time {save_time:.3f}s exceeds 500ms"
                assert load_time < 0.5, f"Load time {load_time:.3f}s exceeds 500ms"


class TestMaintenanceOperations:
    """Test maintenance and utility operations."""

    def test_cleanup_orphaned_locks(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test cleanup of orphaned checkpoint locks."""
        # Create some checkpoints to generate locks
        id1 = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Access them to ensure locks are created
        checkpoint_manager.load_checkpoint(id1)
        checkpoint_manager.load_checkpoint(id2)

        # Verify locks exist
        assert len(checkpoint_manager._checkpoint_locks) >= 2

        # Delete one checkpoint file directly (simulating external deletion)
        checkpoint_file = checkpoint_manager.checkpoint_dir / f"{id1}.lz4"
        checkpoint_file.unlink()

        # Cleanup should remove orphaned lock
        cleaned_count = checkpoint_manager.cleanup_orphaned_locks()
        assert cleaned_count >= 1

        # Verify orphaned lock is removed
        assert id1 not in checkpoint_manager._checkpoint_locks
        assert id2 in checkpoint_manager._checkpoint_locks  # Should still exist
