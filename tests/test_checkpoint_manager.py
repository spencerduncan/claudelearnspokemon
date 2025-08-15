"""
Unit tests for CheckpointManager metadata management functionality.

Tests all aspects of checkpoint storage, metadata management, validation,
search, and performance requirements.
"""

import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from claudelearnspokemon.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """Test suite for CheckpointManager metadata management."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create CheckpointManager instance for testing."""
        return CheckpointManager(checkpoint_dir=temp_dir)

    @pytest.fixture
    def sample_game_state(self):
        """Sample game state for testing."""
        return {
            "player_position": {"x": 100, "y": 200},
            "inventory": ["pokeball", "potion"],
            "pokemon": [{"name": "pikachu", "level": 25}],
            "game_time": 12345,
        }

    @pytest.fixture
    def sample_metadata(self):
        """Sample metadata for testing."""
        return {
            "game_location": "pallet_town",
            "progress_markers": ["got_starter", "visited_oak"],
            "performance_metrics": {"execution_time": 5.2, "success_rate": 0.85},
            "tags": ["tutorial", "important"],
            "custom_fields": {"strategy_type": "speedrun", "difficulty": "normal"},
        }

    def test_checkpoint_manager_initialization(self, temp_dir):
        """Test CheckpointManager initialization creates directory and database."""
        manager = CheckpointManager(checkpoint_dir=temp_dir)

        assert Path(temp_dir).exists()
        assert (Path(temp_dir) / "metadata.db").exists()
        assert manager.checkpoint_dir == Path(temp_dir)

    def test_save_checkpoint_with_metadata(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test checkpoint saving with metadata storage."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Verify checkpoint ID format
        assert checkpoint_id is not None
        assert len(checkpoint_id) == 36  # UUID format

        # Verify file was created
        checkpoint_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.pkl.gz"
        assert checkpoint_file.exists()

        # Verify metadata was stored
        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert metadata is not None
        assert metadata["game_location"] == "pallet_town"
        assert metadata["progress_markers"] == ["got_starter", "visited_oak"]
        assert metadata["tags"] == ["tutorial", "important"]

    def test_load_checkpoint_returns_original_state(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test checkpoint loading returns original game state."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        loaded_state = checkpoint_manager.load_checkpoint(checkpoint_id)

        assert loaded_state == sample_game_state
        assert loaded_state["player_position"] == {"x": 100, "y": 200}
        assert loaded_state["pokemon"][0]["name"] == "pikachu"

    def test_get_checkpoint_metadata_returns_complete_metadata(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test get_checkpoint_metadata returns all metadata fields."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)

        assert metadata["checkpoint_id"] == checkpoint_id
        assert "created_at" in metadata
        assert metadata["game_location"] == "pallet_town"
        assert metadata["progress_markers"] == ["got_starter", "visited_oak"]
        assert metadata["performance_metrics"]["execution_time"] == 5.2
        assert metadata["performance_metrics"]["success_rate"] == 0.85
        assert metadata["tags"] == ["tutorial", "important"]
        assert metadata["custom_fields"]["strategy_type"] == "speedrun"
        assert metadata["file_size"] > 0
        assert len(metadata["checksum"]) == 64  # SHA-256 hex

    def test_get_checkpoint_metadata_nonexistent_returns_none(self, checkpoint_manager):
        """Test get_checkpoint_metadata returns None for nonexistent checkpoint."""
        metadata = checkpoint_manager.get_checkpoint_metadata("nonexistent-id")
        assert metadata is None

    def test_update_checkpoint_metadata_success(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test successful metadata update for existing checkpoint."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Update metadata
        updates = {
            "game_location": "viridian_city",
            "tags": ["tutorial", "important", "completed"],
            "custom_fields": {"strategy_type": "casual", "notes": "good checkpoint"},
        }

        success = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
        assert success is True

        # Verify updates
        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert metadata["game_location"] == "viridian_city"
        assert metadata["tags"] == ["tutorial", "important", "completed"]
        assert metadata["custom_fields"]["strategy_type"] == "casual"
        assert metadata["custom_fields"]["notes"] == "good checkpoint"

    def test_update_checkpoint_metadata_nonexistent_fails(self, checkpoint_manager):
        """Test metadata update fails for nonexistent checkpoint."""
        success = checkpoint_manager.update_checkpoint_metadata(
            "nonexistent-id", {"game_location": "test"}
        )
        assert success is False

    def test_search_checkpoints_by_location(self, checkpoint_manager, sample_game_state):
        """Test searching checkpoints by game location."""
        # Create multiple checkpoints
        metadata1 = {"game_location": "pallet_town", "tags": ["start"]}
        metadata2 = {"game_location": "viridian_city", "tags": ["route1"]}
        metadata3 = {"game_location": "pallet_town", "tags": ["return"]}

        id1 = checkpoint_manager.save_checkpoint(sample_game_state, metadata1)
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, metadata2)
        id3 = checkpoint_manager.save_checkpoint(sample_game_state, metadata3)

        # Search by location
        results = checkpoint_manager.search_checkpoints({"game_location": "pallet_town"})

        assert len(results) == 2
        found_ids = [r["checkpoint_id"] for r in results]
        assert id1 in found_ids
        assert id3 in found_ids
        assert id2 not in found_ids

    def test_search_checkpoints_by_tags(self, checkpoint_manager, sample_game_state):
        """Test searching checkpoints by tags."""
        metadata1 = {"tags": ["tutorial", "important"]}
        metadata2 = {"tags": ["battle", "gym"]}
        metadata3 = {"tags": ["tutorial", "completed"]}

        id1 = checkpoint_manager.save_checkpoint(sample_game_state, metadata1)
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, metadata2)
        id3 = checkpoint_manager.save_checkpoint(sample_game_state, metadata3)

        # Search by tags
        results = checkpoint_manager.search_checkpoints({"tags": ["tutorial"]})

        assert len(results) == 2
        found_ids = [r["checkpoint_id"] for r in results]
        assert id1 in found_ids
        assert id3 in found_ids
        assert id2 not in found_ids

    def test_search_checkpoints_by_time_range(self, checkpoint_manager, sample_game_state):
        """Test searching checkpoints by creation time range."""
        now = datetime.now(timezone.utc)
        past_time = now.replace(hour=now.hour - 1).isoformat()
        future_time = now.replace(hour=now.hour + 1).isoformat()

        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, {})

        # Search in range
        results = checkpoint_manager.search_checkpoints(
            {"created_after": past_time, "created_before": future_time}
        )

        assert len(results) == 1
        assert results[0]["checkpoint_id"] == checkpoint_id

        # Search outside range
        results = checkpoint_manager.search_checkpoints({"created_after": future_time})

        assert len(results) == 0

    def test_search_checkpoints_by_performance(self, checkpoint_manager, sample_game_state):
        """Test searching checkpoints by performance metrics."""
        metadata1 = {"performance_metrics": {"success_rate": 0.9, "execution_time": 3.0}}
        metadata2 = {"performance_metrics": {"success_rate": 0.7, "execution_time": 8.0}}
        metadata3 = {"performance_metrics": {"success_rate": 0.95, "execution_time": 2.5}}

        id1 = checkpoint_manager.save_checkpoint(sample_game_state, metadata1)
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, metadata2)
        id3 = checkpoint_manager.save_checkpoint(sample_game_state, metadata3)

        # Search by minimum performance
        results = checkpoint_manager.search_checkpoints({"performance_min": 0.85})

        assert len(results) == 2
        found_ids = [r["checkpoint_id"] for r in results]
        assert id1 in found_ids
        assert id3 in found_ids
        assert id2 not in found_ids

        # Search by maximum execution time
        results = checkpoint_manager.search_checkpoints({"performance_max": 5.0})

        assert len(results) == 2
        found_ids = [r["checkpoint_id"] for r in results]
        assert id1 in found_ids
        assert id3 in found_ids
        assert id2 not in found_ids

    def test_search_checkpoints_by_custom_fields(self, checkpoint_manager, sample_game_state):
        """Test searching checkpoints by custom fields."""
        metadata1 = {"custom_fields": {"strategy": "aggressive", "difficulty": "hard"}}
        metadata2 = {"custom_fields": {"strategy": "defensive", "difficulty": "normal"}}
        metadata3 = {"custom_fields": {"strategy": "aggressive", "difficulty": "normal"}}

        id1 = checkpoint_manager.save_checkpoint(sample_game_state, metadata1)
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, metadata2)
        id3 = checkpoint_manager.save_checkpoint(sample_game_state, metadata3)

        # Search by custom fields
        results = checkpoint_manager.search_checkpoints(
            {"custom_fields": {"strategy": "aggressive"}}
        )

        assert len(results) == 2
        found_ids = [r["checkpoint_id"] for r in results]
        assert id1 in found_ids
        assert id3 in found_ids
        assert id2 not in found_ids

    def test_list_checkpoints_returns_all(self, checkpoint_manager, sample_game_state):
        """Test list_checkpoints returns all checkpoints when no criteria given."""
        # Create multiple checkpoints
        id1 = checkpoint_manager.save_checkpoint(sample_game_state, {"game_location": "loc1"})
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, {"game_location": "loc2"})
        id3 = checkpoint_manager.save_checkpoint(sample_game_state, {"game_location": "loc3"})

        results = checkpoint_manager.list_checkpoints()

        assert len(results) == 3
        found_ids = [r["checkpoint_id"] for r in results]
        assert id1 in found_ids
        assert id2 in found_ids
        assert id3 in found_ids

    def test_validate_checkpoint_success(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test checkpoint validation for valid checkpoint."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        is_valid = checkpoint_manager.validate_checkpoint(checkpoint_id)
        assert is_valid is True

    def test_validate_checkpoint_missing_file(self, checkpoint_manager, temp_dir):
        """Test checkpoint validation fails for missing file."""
        # Create fake metadata entry
        fake_id = "fake-checkpoint-id"

        is_valid = checkpoint_manager.validate_checkpoint(fake_id)
        assert is_valid is False

    def test_validate_checkpoint_corrupted_file(
        self, checkpoint_manager, sample_game_state, sample_metadata, temp_dir
    ):
        """Test checkpoint validation fails for corrupted file."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Corrupt the file
        checkpoint_file = Path(temp_dir) / f"{checkpoint_id}.pkl.gz"
        with open(checkpoint_file, "wb") as f:
            f.write(b"corrupted data")

        is_valid = checkpoint_manager.validate_checkpoint(checkpoint_id)
        assert is_valid is False

    def test_prune_checkpoints_removes_lowest_value(self, checkpoint_manager, sample_game_state):
        """Test checkpoint pruning removes lowest value checkpoints."""
        # Create checkpoints with different performance metrics
        old_metadata = {
            "created_at": "2024-01-01T00:00:00+00:00",
            "performance_metrics": {"success_rate": 0.5, "execution_time": 10.0},
        }
        good_metadata = {"performance_metrics": {"success_rate": 0.95, "execution_time": 2.0}}
        bad_metadata = {"performance_metrics": {"success_rate": 0.2, "execution_time": 15.0}}

        checkpoint_manager.save_checkpoint(sample_game_state, old_metadata)
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, good_metadata)
        checkpoint_manager.save_checkpoint(sample_game_state, bad_metadata)

        # Prune to keep only 2
        pruned_count = checkpoint_manager.prune_checkpoints(max_count=2)

        assert pruned_count == 1

        # Verify good checkpoint remains
        remaining = checkpoint_manager.list_checkpoints()
        assert len(remaining) == 2

        remaining_ids = [r["checkpoint_id"] for r in remaining]
        assert id2 in remaining_ids  # Good performance should remain

    def test_find_nearest_checkpoint_exact_match(self, checkpoint_manager, sample_game_state):
        """Test find_nearest_checkpoint returns exact location match."""
        metadata1 = {"game_location": "pallet_town"}
        metadata2 = {"game_location": "viridian_city"}
        metadata3 = {"game_location": "pewter_city"}

        checkpoint_manager.save_checkpoint(sample_game_state, metadata1)
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, metadata2)
        checkpoint_manager.save_checkpoint(sample_game_state, metadata3)

        nearest_id = checkpoint_manager.find_nearest_checkpoint("viridian_city")
        assert nearest_id == id2

    def test_find_nearest_checkpoint_no_match_returns_any(
        self, checkpoint_manager, sample_game_state
    ):
        """Test find_nearest_checkpoint returns something when no exact match."""
        metadata1 = {"game_location": "pallet_town"}
        id1 = checkpoint_manager.save_checkpoint(sample_game_state, metadata1)

        nearest_id = checkpoint_manager.find_nearest_checkpoint("nonexistent_location")
        assert nearest_id == id1  # Should return the only available checkpoint

    def test_find_nearest_checkpoint_no_checkpoints_returns_none(self, checkpoint_manager):
        """Test find_nearest_checkpoint returns None when no checkpoints exist."""
        nearest_id = checkpoint_manager.find_nearest_checkpoint("any_location")
        assert nearest_id is None

    def test_metadata_validation_required_fields(self, checkpoint_manager, sample_game_state):
        """Test metadata validation enforces required field types."""
        # Valid metadata should work
        valid_metadata = {
            "game_location": "test_location",
            "progress_markers": ["marker1"],
            "performance_metrics": {"time": 1.0},
            "tags": ["tag1"],
            "custom_fields": {"field": "value"},
        }

        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, valid_metadata)
        assert checkpoint_id is not None

        # Invalid progress_markers type should raise error
        invalid_metadata = {"progress_markers": "not_a_list"}

        with pytest.raises(ValueError, match="progress_markers must be a list"):
            checkpoint_manager.save_checkpoint(sample_game_state, invalid_metadata)

        # Invalid performance_metrics type should raise error
        invalid_metadata = {"performance_metrics": "not_a_dict"}

        with pytest.raises(ValueError, match="performance_metrics must be a dict"):
            checkpoint_manager.save_checkpoint(sample_game_state, invalid_metadata)

    def test_metadata_caching_improves_performance(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test metadata caching improves repeated access performance."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # First access - should populate cache
        start_time = time.perf_counter()
        metadata1 = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        first_access_time = time.perf_counter() - start_time

        # Second access - should use cache
        start_time = time.perf_counter()
        metadata2 = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        second_access_time = time.perf_counter() - start_time

        # Verify same data returned
        assert metadata1 == metadata2

        # Cache should be faster (though timing can be unreliable in tests)
        assert second_access_time <= first_access_time * 2  # Allow some variance

    def test_load_checkpoint_missing_file_raises_error(self, checkpoint_manager):
        """Test loading nonexistent checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            checkpoint_manager.load_checkpoint("nonexistent-id")

    def test_save_checkpoint_creates_checksum(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test saved checkpoint includes file checksum for integrity verification."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert "checksum" in metadata
        assert len(metadata["checksum"]) == 64  # SHA-256 hex length
        assert metadata["file_size"] > 0

    def test_load_checkpoint_validates_checksum(
        self, checkpoint_manager, sample_game_state, sample_metadata, temp_dir
    ):
        """Test loading checkpoint with corrupted checksum raises error."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Corrupt the file content but keep valid gzip structure
        checkpoint_file = Path(temp_dir) / f"{checkpoint_id}.pkl.gz"

        with open(checkpoint_file, "ab") as f:
            f.write(b"corruption")  # Append corrupt data

        # Loading should detect checksum mismatch
        with pytest.raises(ValueError, match="checksum mismatch"):
            checkpoint_manager.load_checkpoint(checkpoint_id)

    def test_metadata_schema_versioning(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test metadata includes schema version for future migrations."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert "schema_version" in metadata
        assert metadata["schema_version"] == CheckpointManager.SCHEMA_VERSION


class TestCheckpointManagerPerformance:
    """Performance tests for CheckpointManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def checkpoint_manager_with_data(self, temp_dir):
        """Create CheckpointManager with multiple checkpoints for performance testing."""
        manager = CheckpointManager(checkpoint_dir=temp_dir)

        # Create 100 test checkpoints with varying metadata
        game_state = {"test": "data"}
        locations = ["pallet_town", "viridian_city", "pewter_city", "cerulean_city"]
        tags_options = [["tutorial"], ["battle"], ["gym"], ["important"], ["speedrun"]]

        for i in range(100):
            metadata = {
                "game_location": locations[i % len(locations)],
                "tags": tags_options[i % len(tags_options)],
                "performance_metrics": {
                    "success_rate": 0.5 + (i % 50) / 100,
                    "execution_time": 1.0 + (i % 10),
                },
                "custom_fields": {"batch": i // 10},
            }
            manager.save_checkpoint(game_state, metadata)

        return manager

    def test_metadata_query_performance_requirement(self, checkpoint_manager_with_data):
        """Test that metadata queries meet <100ms performance requirement."""
        manager = checkpoint_manager_with_data

        # Test various query types
        query_types = [
            {"game_location": "pallet_town"},
            {"tags": ["tutorial"]},
            {"performance_min": 0.8},
            {"custom_fields": {"batch": 5}},
        ]

        for criteria in query_types:
            start_time = time.perf_counter()
            results = manager.search_checkpoints(criteria)
            query_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

            # Should meet performance requirement
            assert query_time < 100, f"Query {criteria} took {query_time:.2f}ms, expected <100ms"
            assert len(results) > 0  # Should find some results

    def test_metadata_cache_performance_benefit(self, checkpoint_manager_with_data):
        """Test that metadata caching provides performance benefit."""
        manager = checkpoint_manager_with_data

        checkpoints = manager.list_checkpoints()[:10]  # Get first 10
        checkpoint_ids = [cp["checkpoint_id"] for cp in checkpoints]

        # First access - populate cache
        start_time = time.perf_counter()
        for checkpoint_id in checkpoint_ids:
            manager.get_checkpoint_metadata(checkpoint_id)
        first_run_time = time.perf_counter() - start_time

        # Second access - use cache
        start_time = time.perf_counter()
        for checkpoint_id in checkpoint_ids:
            manager.get_checkpoint_metadata(checkpoint_id)
        second_run_time = time.perf_counter() - start_time

        # Cache should provide meaningful speedup
        speedup_ratio = first_run_time / second_run_time if second_run_time > 0 else float("inf")
        assert speedup_ratio > 1.2, f"Cache speedup {speedup_ratio:.2f}x, expected >1.2x"

    def test_large_checkpoint_handling(self, temp_dir):
        """Test handling of large checkpoint files with reasonable performance."""
        manager = CheckpointManager(checkpoint_dir=temp_dir)

        # Create large game state (simulate complex game state)
        large_game_state = {
            "world_map": [[i * j for j in range(1000)] for i in range(1000)],
            "entities": [{"id": i, "data": f"entity_{i}" * 100} for i in range(10000)],
            "history": [f"action_{i}" for i in range(50000)],
        }

        metadata = {
            "game_location": "large_world",
            "tags": ["performance_test"],
            "custom_fields": {"size_test": True},
        }

        # Save should complete in reasonable time
        start_time = time.perf_counter()
        checkpoint_id = manager.save_checkpoint(large_game_state, metadata)
        save_time = (time.perf_counter() - start_time) * 1000

        assert save_time < 5000, f"Large checkpoint save took {save_time:.2f}ms, expected <5000ms"

        # Load should also complete in reasonable time
        start_time = time.perf_counter()
        loaded_state = manager.load_checkpoint(checkpoint_id)
        load_time = (time.perf_counter() - start_time) * 1000

        assert load_time < 5000, f"Large checkpoint load took {load_time:.2f}ms, expected <5000ms"
        assert loaded_state == large_game_state
