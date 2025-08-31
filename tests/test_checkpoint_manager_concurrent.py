"""
Concurrent tests for CheckpointManager thread safety.

Tests race conditions, concurrent access patterns, and thread safety
of shared data structures under heavy concurrent load.
"""

import concurrent.futures
import shutil
import tempfile
import threading
import time
from collections import defaultdict

import pytest

from claudelearnspokemon.checkpoint_manager import CheckpointManager


@pytest.mark.fast
@pytest.mark.medium
class TestCheckpointManagerConcurrency:
    """Test suite for CheckpointManager thread safety and concurrent access."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create CheckpointManager instance for testing."""
        return CheckpointManager(storage_dir=temp_dir)

    @pytest.fixture
    def sample_game_state(self):
        """Sample game state for testing."""
        return {
            "player_position": {"x": 100, "y": 200},
            "inventory": ["pokeball", "potion"],
            "pokemon": [{"name": "pikachu", "level": 25}],
            "game_time": 12345,
        }

    def test_concurrent_cache_access(self, checkpoint_manager, sample_game_state):
        """Test concurrent cache read/write operations don't cause race conditions."""
        # Pre-populate with some checkpoints
        checkpoint_ids = []
        for i in range(10):
            metadata = {
                "game_location": f"location_{i}",
                "tags": [f"tag_{i}"],
                "performance_metrics": {"score": i * 10},
            }
            checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, metadata)
            checkpoint_ids.append(checkpoint_id)

        # Define concurrent access function
        access_counts = defaultdict(int)
        errors = []

        def concurrent_cache_access(thread_id):
            """Function to run concurrent cache operations."""
            try:
                for i in range(20):  # Each thread does 20 operations
                    # Mix of cache hits and misses
                    checkpoint_id = checkpoint_ids[i % len(checkpoint_ids)]

                    # Read from cache (should be thread-safe)
                    metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
                    if metadata:
                        access_counts[thread_id] += 1

                    # Small delay to increase chance of race conditions
                    time.sleep(0.001)

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Run concurrent access with multiple threads
        num_threads = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(concurrent_cache_access, thread_id)
                for thread_id in range(num_threads)
            ]
            concurrent.futures.wait(futures)

        # Verify no errors occurred
        assert not errors, f"Concurrent access errors: {errors}"

        # Verify all threads completed successfully
        assert len(access_counts) == num_threads
        for thread_id, count in access_counts.items():
            assert count == 20, f"Thread {thread_id} only completed {count}/20 operations"

    def test_concurrent_metadata_updates(self, checkpoint_manager, sample_game_state):
        """Test concurrent metadata updates don't cause corruption."""
        # Create a checkpoint to update
        metadata = {
            "game_location": "test_location",
            "tags": ["initial"],
            "performance_metrics": {"score": 0},
            "custom_fields": {"counter": 0},
        }
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, metadata)

        update_counts = defaultdict(int)
        errors = []

        def concurrent_update(thread_id):
            """Function to run concurrent metadata updates."""
            try:
                for i in range(10):
                    # Each thread tries to update different fields
                    updates = {
                        "tags": [f"thread_{thread_id}_update_{i}"],
                        "custom_fields": {f"thread_{thread_id}": i},
                    }

                    success = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
                    if success:
                        update_counts[thread_id] += 1

                    time.sleep(0.001)  # Small delay

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Run concurrent updates
        num_threads = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(concurrent_update, thread_id) for thread_id in range(num_threads)
            ]
            concurrent.futures.wait(futures)

        # Verify no errors
        assert not errors, f"Concurrent update errors: {errors}"

        # Verify final state is consistent
        final_metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert final_metadata is not None
        assert "tags" in final_metadata
        assert "custom_fields" in final_metadata

    def test_concurrent_save_load_operations(self, checkpoint_manager, sample_game_state):
        """Test concurrent save and load operations don't interfere."""
        saved_ids = []
        load_results = []
        errors = []
        lock = threading.Lock()

        def concurrent_save_load(thread_id):
            """Function to run concurrent save/load operations."""
            try:
                for i in range(5):
                    # Save operation
                    metadata = {
                        "game_location": f"location_{thread_id}_{i}",
                        "tags": [f"thread_{thread_id}"],
                        "custom_fields": {"thread_id": thread_id, "iteration": i},
                    }

                    # Modify game state slightly for each thread
                    modified_state = sample_game_state.copy()
                    modified_state["thread_marker"] = f"thread_{thread_id}_{i}"

                    checkpoint_id = checkpoint_manager.save_checkpoint(modified_state, metadata)

                    with lock:
                        saved_ids.append(checkpoint_id)

                    # Immediate load to verify
                    loaded_state = checkpoint_manager.load_checkpoint(checkpoint_id)

                    with lock:
                        load_results.append(
                            {
                                "thread_id": thread_id,
                                "checkpoint_id": checkpoint_id,
                                "loaded_correctly": loaded_state["thread_marker"]
                                == f"thread_{thread_id}_{i}",
                            }
                        )

                    time.sleep(0.001)

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Run concurrent save/load operations
        num_threads = 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(concurrent_save_load, thread_id) for thread_id in range(num_threads)
            ]
            concurrent.futures.wait(futures)

        # Verify no errors
        assert not errors, f"Concurrent save/load errors: {errors}"

        # Verify all saves succeeded
        assert len(saved_ids) == num_threads * 5

        # Verify all loads returned correct data
        assert len(load_results) == num_threads * 5
        for result in load_results:
            assert result["loaded_correctly"], f"Load failed for {result}"

    def test_concurrent_cache_eviction(self, checkpoint_manager, sample_game_state):
        """Test that cache eviction works correctly under concurrent access."""
        # Fill cache to near capacity
        cache_size = checkpoint_manager.METADATA_CACHE_SIZE
        checkpoint_ids = []

        # Fill cache to 90% capacity
        for i in range(int(cache_size * 0.9)):
            metadata = {"game_location": f"location_{i}", "tags": [f"tag_{i}"]}
            checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, metadata)
            checkpoint_ids.append(checkpoint_id)

        errors = []
        access_results = defaultdict(list)

        def concurrent_cache_eviction_trigger(thread_id):
            """Function to trigger cache eviction concurrently."""
            try:
                # Each thread creates new checkpoints to trigger eviction
                for i in range(10):
                    metadata = {
                        "game_location": f"eviction_test_{thread_id}_{i}",
                        "tags": [f"eviction_{thread_id}"],
                    }

                    checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, metadata)
                    checkpoint_ids.append(checkpoint_id)

                    # Immediately try to access some old checkpoints
                    for old_id in checkpoint_ids[:5]:  # Access first 5 old checkpoints
                        result = checkpoint_manager.get_checkpoint_metadata(old_id)
                        access_results[thread_id].append(result is not None)

                    time.sleep(0.001)

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Run concurrent eviction triggers
        num_threads = 6
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(concurrent_cache_eviction_trigger, thread_id)
                for thread_id in range(num_threads)
            ]
            concurrent.futures.wait(futures)

        # Verify no errors during cache eviction
        assert not errors, f"Cache eviction errors: {errors}"

        # Verify cache still functions correctly
        total_checkpoints = checkpoint_manager.list_checkpoints({})
        assert len(total_checkpoints) > 0

    def test_concurrent_search_operations(self, checkpoint_manager, sample_game_state):
        """Test concurrent search operations don't cause issues."""
        # Create diverse checkpoints for searching
        locations = ["pallet_town", "viridian_city", "pewter_city", "cerulean_city"]
        tags_options = [["tutorial"], ["battle"], ["gym"], ["important"]]

        for i in range(20):
            metadata = {
                "game_location": locations[i % len(locations)],
                "tags": tags_options[i % len(tags_options)],
                "performance_metrics": {"score": i * 10},
                "custom_fields": {"group": i // 5},
            }
            checkpoint_manager.save_checkpoint(sample_game_state, metadata)

        search_results = defaultdict(list)
        errors = []

        def concurrent_search(thread_id):
            """Function to run concurrent search operations."""
            try:
                search_criteria = [
                    {"game_location": "pallet_town"},
                    {"tags": ["tutorial"]},
                    {"performance_min": 50},
                    {"custom_fields": {"group": 1}},
                ]

                for i in range(15):  # Each thread does 15 searches
                    criteria = search_criteria[i % len(search_criteria)]
                    results = checkpoint_manager.search_checkpoints(criteria)
                    search_results[thread_id].append(len(results))

                    time.sleep(0.001)

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Run concurrent searches
        num_threads = 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(concurrent_search, thread_id) for thread_id in range(num_threads)
            ]
            concurrent.futures.wait(futures)

        # Verify no errors
        assert not errors, f"Concurrent search errors: {errors}"

        # Verify all threads completed their searches
        assert len(search_results) == num_threads
        for thread_id, results in search_results.items():
            assert (
                len(results) == 15
            ), f"Thread {thread_id} only completed {len(results)}/15 searches"

    def test_stress_test_high_concurrency(self, checkpoint_manager, sample_game_state):
        """Stress test with high concurrency and mixed operations."""
        errors = []
        operation_counts = defaultdict(int)

        def mixed_operations(thread_id):
            """Function that performs mixed concurrent operations."""
            try:
                checkpoint_ids = []

                # Mix of save, load, update, search operations
                for i in range(20):
                    operation_type = i % 4

                    if operation_type == 0:  # Save
                        metadata = {
                            "game_location": f"stress_test_{thread_id}_{i}",
                            "tags": [f"stress_{thread_id}"],
                            "performance_metrics": {"score": i},
                        }
                        checkpoint_id = checkpoint_manager.save_checkpoint(
                            sample_game_state, metadata
                        )
                        checkpoint_ids.append(checkpoint_id)
                        operation_counts[f"thread_{thread_id}_save"] += 1

                    elif operation_type == 1 and checkpoint_ids:  # Load
                        checkpoint_id = checkpoint_ids[-1]
                        checkpoint_manager.load_checkpoint(checkpoint_id)
                        operation_counts[f"thread_{thread_id}_load"] += 1

                    elif operation_type == 2 and checkpoint_ids:  # Update
                        checkpoint_id = checkpoint_ids[-1]
                        updates = {"tags": [f"updated_{thread_id}_{i}"]}
                        checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
                        operation_counts[f"thread_{thread_id}_update"] += 1

                    elif operation_type == 3:  # Search
                        criteria = {"tags": [f"stress_{thread_id}"]}
                        checkpoint_manager.search_checkpoints(criteria)
                        operation_counts[f"thread_{thread_id}_search"] += 1

                    # Very small delay
                    time.sleep(0.0005)

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Run high concurrency stress test
        num_threads = 15
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(mixed_operations, thread_id) for thread_id in range(num_threads)
            ]
            concurrent.futures.wait(futures)

        # Verify no errors under high concurrency
        assert not errors, f"High concurrency errors: {errors}"

        # Verify operations completed
        assert len(operation_counts) > 0

        # Verify checkpoint manager is still functional
        all_checkpoints = checkpoint_manager.list_checkpoints({})
        assert len(all_checkpoints) > 0

    def test_race_condition_in_cache_lru_eviction(self, checkpoint_manager, sample_game_state):
        """Test race conditions specifically in LRU cache eviction logic."""
        # Set a small cache size for easier testing
        original_cache_size = checkpoint_manager.METADATA_CACHE_SIZE
        checkpoint_manager.METADATA_CACHE_SIZE = 20

        try:
            # Pre-fill cache
            checkpoint_ids = []
            for i in range(15):
                metadata = {"game_location": f"location_{i}"}
                checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, metadata)
                checkpoint_ids.append(checkpoint_id)

            errors = []

            def trigger_lru_eviction(thread_id):
                """Function that triggers LRU eviction through cache access patterns."""
                try:
                    # Each thread accesses existing checkpoints to update access times
                    # while also creating new ones to trigger eviction
                    for i in range(10):
                        # Access old checkpoints (updates access time)
                        if checkpoint_ids:
                            old_id = checkpoint_ids[i % len(checkpoint_ids)]
                            checkpoint_manager.get_checkpoint_metadata(old_id)

                        # Create new checkpoint (may trigger eviction)
                        metadata = {"game_location": f"eviction_{thread_id}_{i}"}
                        checkpoint_manager.save_checkpoint(sample_game_state, metadata)

                        time.sleep(0.001)

                except Exception as e:
                    errors.append(f"Thread {thread_id}: {e}")

            # Run LRU eviction race condition test
            num_threads = 8
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(trigger_lru_eviction, thread_id)
                    for thread_id in range(num_threads)
                ]
                concurrent.futures.wait(futures)

            # Verify no race condition errors
            assert not errors, f"LRU eviction race condition errors: {errors}"

        finally:
            # Restore original cache size
            checkpoint_manager.METADATA_CACHE_SIZE = original_cache_size
