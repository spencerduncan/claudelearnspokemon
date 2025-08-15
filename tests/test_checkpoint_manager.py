"""
Comprehensive test suite for CheckpointManager - Production-grade testing.

Tests cover:
- Core save/load operations with performance validation
- Pruning algorithm with various scenarios
- Integrity validation with corruption simulation
- Error handling and edge cases
- Metrics and observability
- Concurrent access patterns

Author: Bot Dean - Production-First Testing
"""

import tempfile
import time
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from claudelearnspokemon.checkpoint_manager import CheckpointManager, CheckpointMetadata


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def checkpoint_manager(temp_storage_dir):
    """Create CheckpointManager instance for testing."""
    return CheckpointManager(
        storage_dir=temp_storage_dir,
        max_checkpoints=5,  # Small limit for testing
        enable_metrics=True,
    )


@pytest.fixture
def sample_game_state():
    """Sample game state for testing."""
    return {
        "player": {
            "x": 100,
            "y": 150,
            "level": 5,
            "pokemon": [{"name": "Pikachu", "level": 8, "hp": 30}],
        },
        "map": {"current": "Route 1", "tiles": [[1, 2, 3], [4, 5, 6]]},
        "inventory": ["pokeball", "potion"],
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "location": "Route 1",
        "progress_markers": {"badges": 0, "pokemon_caught": 1, "story_flags": ["intro_complete"]},
    }


class TestCheckpointManagerCore:
    """Test core save/load functionality."""

    def test_initialization(self, temp_storage_dir):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(
            storage_dir=temp_storage_dir, max_checkpoints=100, enable_metrics=True
        )

        assert manager.storage_dir == temp_storage_dir.resolve()
        assert manager.max_checkpoints == 100
        assert manager.enable_metrics is True
        assert temp_storage_dir.exists()

        # Check initial metrics
        metrics = manager.get_metrics()
        assert metrics["saves_total"] == 0
        assert metrics["loads_total"] == 0
        assert metrics["checkpoint_count"] == 0

    def test_save_checkpoint_success(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test successful checkpoint saving."""
        start_time = time.perf_counter()

        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Validate checkpoint ID format
        assert checkpoint_id.startswith("cp_")
        assert len(checkpoint_id) > 20

        # Check files were created
        checkpoint_path = checkpoint_manager._get_checkpoint_path(checkpoint_id)
        metadata_path = checkpoint_manager._get_metadata_path(checkpoint_id)
        assert checkpoint_path.exists()
        assert metadata_path.exists()

        # Performance requirement: saving should be reasonably fast
        assert elapsed_ms < 1000  # Allow 1s for test environment

        # Check metrics updated
        metrics = checkpoint_manager.get_metrics()
        assert metrics["saves_total"] == 1
        assert metrics["checkpoint_count"] == 1

    def test_save_checkpoint_validation(self, checkpoint_manager):
        """Test input validation for save_checkpoint."""

        # Test invalid game_state
        with pytest.raises(ValueError, match="game_state must be a dictionary"):
            checkpoint_manager.save_checkpoint("invalid", {})

        with pytest.raises(ValueError, match="game_state cannot be empty"):
            checkpoint_manager.save_checkpoint({}, {})

        # Test invalid metadata
        with pytest.raises(ValueError, match="metadata must be a dictionary"):
            checkpoint_manager.save_checkpoint({"valid": "state"}, "invalid")

    def test_load_checkpoint_success(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test successful checkpoint loading."""
        # Save first
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Load and validate
        start_time = time.perf_counter()
        loaded_state = checkpoint_manager.load_checkpoint(checkpoint_id)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert loaded_state == sample_game_state

        # Performance requirement: < 500ms
        assert elapsed_ms < checkpoint_manager.MAX_LOAD_TIME_MS

        # Check metrics
        metrics = checkpoint_manager.get_metrics()
        assert metrics["loads_total"] == 1

    def test_load_checkpoint_not_found(self, checkpoint_manager):
        """Test loading non-existent checkpoint."""
        with pytest.raises(FileNotFoundError, match="Checkpoint nonexistent not found"):
            checkpoint_manager.load_checkpoint("nonexistent")

    def test_load_checkpoint_validation_errors(self, checkpoint_manager):
        """Test load validation errors."""
        with pytest.raises(ValueError, match="checkpoint_id must be a non-empty string"):
            checkpoint_manager.load_checkpoint("")

        with pytest.raises(ValueError, match="checkpoint_id must be a non-empty string"):
            checkpoint_manager.load_checkpoint(None)


class TestCheckpointValidation:
    """Test checkpoint integrity validation."""

    def test_validate_checkpoint_success(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test successful checkpoint validation."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        start_time = time.perf_counter()
        is_valid = checkpoint_manager.validate_checkpoint(checkpoint_id)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert is_valid is True

        # Performance requirement: < 100ms
        assert elapsed_ms < checkpoint_manager.MAX_VALIDATION_TIME_MS

        # Check metrics
        metrics = checkpoint_manager.get_metrics()
        assert metrics["validations_total"] == 1

    def test_validate_checkpoint_missing_files(self, checkpoint_manager):
        """Test validation with missing files."""
        assert checkpoint_manager.validate_checkpoint("nonexistent") is False

    def test_validate_checkpoint_corrupted_data(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test validation with corrupted checkpoint data."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        checkpoint_path = checkpoint_manager._get_checkpoint_path(checkpoint_id)

        # Corrupt the checkpoint file
        with open(checkpoint_path, "wb") as f:
            f.write(b"corrupted data")

        # Should fail validation due to CRC mismatch
        assert checkpoint_manager.validate_checkpoint(checkpoint_id) is False

        # Check corruption metric
        metrics = checkpoint_manager.get_metrics()
        assert metrics["corruption_events"] > 0

    def test_validate_checkpoint_corrupted_compression(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test validation with corrupted compressed data."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        checkpoint_path = checkpoint_manager._get_checkpoint_path(checkpoint_id)

        # Get valid CRC but corrupt the compressed data content
        metadata = checkpoint_manager._load_metadata(checkpoint_id)
        valid_crc = metadata.crc32_checksum

        # Write data that matches CRC length but is invalid compressed data
        corrupted_data = b"x" * 50  # Invalid zlib data
        actual_crc = hex(zlib.crc32(corrupted_data) & 0xFFFFFFFF)

        with open(checkpoint_path, "wb") as f:
            f.write(corrupted_data)

        # Update metadata with new CRC
        metadata.crc32_checksum = actual_crc
        checkpoint_manager._save_metadata(checkpoint_id, metadata)

        # Should fail validation due to decompression error
        assert checkpoint_manager.validate_checkpoint(checkpoint_id) is False

    def test_validate_empty_checkpoint_id(self, checkpoint_manager):
        """Test validation with empty checkpoint ID."""
        assert checkpoint_manager.validate_checkpoint("") is False
        assert checkpoint_manager.validate_checkpoint(None) is False


class TestCheckpointPruning:
    """Test checkpoint pruning algorithm."""

    def test_prune_checkpoints_no_action_needed(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test pruning when no action is needed."""
        # Save fewer checkpoints than the limit
        checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        result = checkpoint_manager.prune_checkpoints(max_count=5)

        assert result["action"] == "no_pruning_needed"
        assert result["total_checkpoints"] == 2
        assert len(result["removed"]) == 0
        assert len(result["retained"]) == 2

    def test_prune_checkpoints_dry_run(self, temp_storage_dir, sample_game_state, sample_metadata):
        """Test pruning in dry-run mode."""
        # Create checkpoint manager with high limit to prevent auto-pruning
        checkpoint_manager = CheckpointManager(
            storage_dir=temp_storage_dir,
            max_checkpoints=20,  # High limit to prevent auto-pruning
            enable_metrics=True,
        )

        # Create more checkpoints than the target limit
        checkpoint_ids = []
        for i in range(7):
            modified_state = sample_game_state.copy()
            modified_state["step"] = i
            checkpoint_id = checkpoint_manager.save_checkpoint(modified_state, sample_metadata)
            checkpoint_ids.append(checkpoint_id)

        result = checkpoint_manager.prune_checkpoints(max_count=3, dry_run=True)

        assert result["action"] == "dry_run"
        assert result["total_checkpoints"] == 7
        assert len(result["removed"]) == 4
        assert len(result["retained"]) == 3

        # Verify no files were actually removed
        for checkpoint_id in checkpoint_ids:
            checkpoint_path = checkpoint_manager._get_checkpoint_path(checkpoint_id)
            assert checkpoint_path.exists()

    def test_prune_checkpoints_execution(
        self, temp_storage_dir, sample_game_state, sample_metadata
    ):
        """Test actual pruning execution."""
        # Create checkpoint manager with high limit to prevent auto-pruning
        checkpoint_manager = CheckpointManager(
            storage_dir=temp_storage_dir,
            max_checkpoints=20,  # High limit to prevent auto-pruning
            enable_metrics=True,
        )

        # Create checkpoints with different access patterns
        checkpoint_ids = []
        for i in range(6):
            modified_state = sample_game_state.copy()
            modified_state["step"] = i
            checkpoint_id = checkpoint_manager.save_checkpoint(modified_state, sample_metadata)
            checkpoint_ids.append(checkpoint_id)

        # Access some checkpoints to affect scoring
        checkpoint_manager.load_checkpoint(checkpoint_ids[0])  # High access
        checkpoint_manager.load_checkpoint(checkpoint_ids[0])  # High access
        checkpoint_manager.load_checkpoint(checkpoint_ids[2])  # Medium access

        start_time = time.perf_counter()
        result = checkpoint_manager.prune_checkpoints(max_count=3, dry_run=False)
        elapsed_s = time.perf_counter() - start_time

        assert result["action"] == "pruning_executed"
        assert len(result["removed"]) == 3
        assert len(result["retained"]) == 3

        # Performance requirement: < 2s
        assert elapsed_s < checkpoint_manager.MAX_PRUNING_TIME_S

        # Verify files were actually removed
        retained_ids = set(result["retained"])
        removed_ids = set(result["removed"])

        for checkpoint_id in retained_ids:
            assert checkpoint_manager._get_checkpoint_path(checkpoint_id).exists()

        for checkpoint_id in removed_ids:
            assert not checkpoint_manager._get_checkpoint_path(checkpoint_id).exists()

        # Check metrics
        metrics = checkpoint_manager.get_metrics()
        assert metrics["pruning_operations"] == 1
        assert metrics["checkpoint_count"] == 3

    def test_prune_checkpoints_invalid_max_count(self, checkpoint_manager):
        """Test pruning with invalid max_count."""
        with pytest.raises(ValueError, match="max_count must be at least 1"):
            checkpoint_manager.prune_checkpoints(max_count=0)

        with pytest.raises(ValueError, match="max_count must be at least 1"):
            checkpoint_manager.prune_checkpoints(max_count=-1)

    def test_automatic_pruning_on_save(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test automatic pruning when checkpoint limit is exceeded."""
        # Save exactly at limit
        for i in range(5):
            modified_state = sample_game_state.copy()
            modified_state["step"] = i
            checkpoint_manager.save_checkpoint(modified_state, sample_metadata)

        assert checkpoint_manager.get_metrics()["checkpoint_count"] == 5

        # Save one more - should trigger auto-pruning
        checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Should have pruned back to limit
        metrics = checkpoint_manager.get_metrics()
        assert metrics["checkpoint_count"] == 5
        assert metrics["pruning_operations"] == 1


class TestValueScoring:
    """Test the value scoring algorithm for pruning decisions."""

    def test_value_scoring_access_frequency(self, checkpoint_manager):
        """Test value scoring based on access frequency."""
        # Create metadata with different access patterns
        base_time = datetime.now(timezone.utc)

        high_access = CheckpointMetadata(
            checkpoint_id="high_access",
            created_at=base_time,
            game_state_hash="hash1",
            file_size_bytes=1000,
            location="test",
            progress_markers={},
            access_count=20,
            last_accessed=base_time - timedelta(hours=1),
        )

        low_access = CheckpointMetadata(
            checkpoint_id="low_access",
            created_at=base_time,
            game_state_hash="hash2",
            file_size_bytes=1000,
            location="test",
            progress_markers={},
            access_count=1,
            last_accessed=base_time - timedelta(hours=1),
        )

        high_score = checkpoint_manager._calculate_value_score(high_access)
        low_score = checkpoint_manager._calculate_value_score(low_access)

        assert high_score > low_score

    def test_value_scoring_recency(self, checkpoint_manager):
        """Test value scoring based on access recency."""
        base_time = datetime.now(timezone.utc)

        recent = CheckpointMetadata(
            checkpoint_id="recent",
            created_at=base_time,
            game_state_hash="hash1",
            file_size_bytes=1000,
            location="test",
            progress_markers={},
            access_count=5,
            last_accessed=base_time - timedelta(minutes=30),  # Very recent
        )

        old = CheckpointMetadata(
            checkpoint_id="old",
            created_at=base_time,
            game_state_hash="hash2",
            file_size_bytes=1000,
            location="test",
            progress_markers={},
            access_count=5,
            last_accessed=base_time - timedelta(days=10),  # Old
        )

        recent_score = checkpoint_manager._calculate_value_score(recent)
        old_score = checkpoint_manager._calculate_value_score(old)

        assert recent_score > old_score

    def test_value_scoring_strategic_value(self, checkpoint_manager):
        """Test value scoring based on strategic importance."""
        base_time = datetime.now(timezone.utc)

        strategic = CheckpointMetadata(
            checkpoint_id="strategic",
            created_at=base_time,
            game_state_hash="hash1",
            file_size_bytes=1000,
            location="gym",
            progress_markers={},
            strategic_value=0.9,  # High strategic value
        )

        regular = CheckpointMetadata(
            checkpoint_id="regular",
            created_at=base_time,
            game_state_hash="hash2",
            file_size_bytes=1000,
            location="route",
            progress_markers={},
            strategic_value=0.1,  # Low strategic value
        )

        strategic_score = checkpoint_manager._calculate_value_score(strategic)
        regular_score = checkpoint_manager._calculate_value_score(regular)

        assert strategic_score > regular_score

    def test_value_scoring_storage_efficiency(self, checkpoint_manager):
        """Test value scoring storage efficiency multiplier."""
        base_time = datetime.now(timezone.utc)

        small_file = CheckpointMetadata(
            checkpoint_id="small",
            created_at=base_time,
            game_state_hash="hash1",
            file_size_bytes=1024,  # 1KB
            location="test",
            progress_markers={},
            success_rate=0.5,
        )

        large_file = CheckpointMetadata(
            checkpoint_id="large",
            created_at=base_time,
            game_state_hash="hash2",
            file_size_bytes=50 * 1024 * 1024,  # 50MB
            location="test",
            progress_markers={},
            success_rate=0.5,
        )

        small_score = checkpoint_manager._calculate_value_score(small_file)
        large_score = checkpoint_manager._calculate_value_score(large_file)

        # Small files should have slight efficiency bonus
        assert small_score >= large_score


class TestCheckpointListing:
    """Test checkpoint listing and querying functionality."""

    def test_list_checkpoints_empty(self, checkpoint_manager):
        """Test listing when no checkpoints exist."""
        result = checkpoint_manager.list_checkpoints({})
        assert result == []

    def test_list_checkpoints_all(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test listing all checkpoints."""
        # Create multiple checkpoints
        checkpoint_ids = []
        for i in range(3):
            modified_state = sample_game_state.copy()
            modified_state["step"] = i
            checkpoint_id = checkpoint_manager.save_checkpoint(modified_state, sample_metadata)
            checkpoint_ids.append(checkpoint_id)

        result = checkpoint_manager.list_checkpoints({})

        assert len(result) == 3
        assert all("checkpoint_id" in cp for cp in result)
        assert all("value_score" in cp for cp in result)

    def test_list_checkpoints_by_location(self, checkpoint_manager, sample_game_state):
        """Test filtering checkpoints by location."""
        # Create checkpoints at different locations
        route1_meta = {"location": "Route 1", "progress_markers": {}}
        route2_meta = {"location": "Route 2", "progress_markers": {}}

        checkpoint_manager.save_checkpoint(sample_game_state, route1_meta)
        checkpoint_manager.save_checkpoint(sample_game_state, route2_meta)
        checkpoint_manager.save_checkpoint(sample_game_state, route1_meta)

        route1_results = checkpoint_manager.list_checkpoints({"location": "Route 1"})
        route2_results = checkpoint_manager.list_checkpoints({"location": "Route 2"})

        assert len(route1_results) == 2
        assert len(route2_results) == 1
        assert all(cp["location"] == "Route 1" for cp in route1_results)
        assert all(cp["location"] == "Route 2" for cp in route2_results)

    def test_list_checkpoints_by_min_score(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test filtering checkpoints by minimum score."""
        # Create checkpoints and access some to increase scores
        cp1 = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        cp2 = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Access cp1 multiple times to increase its score
        checkpoint_manager.load_checkpoint(cp1)
        checkpoint_manager.load_checkpoint(cp1)
        checkpoint_manager.load_checkpoint(cp1)

        # Filter by minimum score
        all_results = checkpoint_manager.list_checkpoints({})
        high_score_results = checkpoint_manager.list_checkpoints({"min_score": 0.5})

        assert len(all_results) == 2
        assert len(high_score_results) <= len(all_results)

        # All returned checkpoints should meet minimum score
        for cp in high_score_results:
            assert cp["value_score"] >= 0.5

    def test_get_checkpoint_metadata(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test getting detailed checkpoint metadata."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)

        assert metadata["checkpoint_id"] == checkpoint_id
        assert metadata["location"] == "Route 1"
        assert "value_score" in metadata
        assert "file_path" in metadata
        assert "created_at" in metadata

    def test_get_checkpoint_metadata_nonexistent(self, checkpoint_manager):
        """Test getting metadata for non-existent checkpoint."""
        metadata = checkpoint_manager.get_checkpoint_metadata("nonexistent")
        assert metadata == {}

    def test_find_nearest_checkpoint(self, checkpoint_manager, sample_game_state):
        """Test finding nearest checkpoint by location."""
        # Create checkpoints at different locations
        route1_meta = {"location": "Route 1", "progress_markers": {}}
        route2_meta = {"location": "Route 2", "progress_markers": {}}

        cp1 = checkpoint_manager.save_checkpoint(sample_game_state, route1_meta)
        cp2 = checkpoint_manager.save_checkpoint(sample_game_state, route2_meta)
        cp3 = checkpoint_manager.save_checkpoint(sample_game_state, route1_meta)

        # Access cp3 to make it higher-scored
        checkpoint_manager.load_checkpoint(cp3)
        checkpoint_manager.load_checkpoint(cp3)

        nearest = checkpoint_manager.find_nearest_checkpoint("Route 1")

        # Should return the highest-scored checkpoint at Route 1
        assert nearest == cp3

        nearest_route2 = checkpoint_manager.find_nearest_checkpoint("Route 2")
        assert nearest_route2 == cp2

        # Non-existent location
        nearest_none = checkpoint_manager.find_nearest_checkpoint("NonExistent")
        assert nearest_none == ""


class TestMetricsAndObservability:
    """Test metrics collection and system observability."""

    def test_get_metrics_disabled(self, temp_storage_dir):
        """Test metrics when disabled."""
        manager = CheckpointManager(storage_dir=temp_storage_dir, enable_metrics=False)

        metrics = manager.get_metrics()
        assert metrics == {"metrics_disabled": True}

    def test_get_metrics_enabled(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test comprehensive metrics collection."""
        # Perform various operations
        cp1 = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        cp2 = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        checkpoint_manager.load_checkpoint(cp1)
        checkpoint_manager.validate_checkpoint(cp1)
        checkpoint_manager.prune_checkpoints(max_count=1)

        metrics = checkpoint_manager.get_metrics()

        # Check all expected metrics
        assert metrics["saves_total"] == 2
        assert metrics["loads_total"] == 1
        assert metrics["validations_total"] >= 1  # Validation called during load too
        assert metrics["pruning_operations"] == 1
        assert metrics["checkpoint_count"] == 1  # After pruning
        assert metrics["storage_bytes_used"] > 0
        assert metrics["storage_utilization"] > 0
        assert "storage_dir" in metrics
        assert "max_checkpoints" in metrics

    def test_metrics_corruption_tracking(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test corruption event metrics."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Corrupt the file
        checkpoint_path = checkpoint_manager._get_checkpoint_path(checkpoint_id)
        with open(checkpoint_path, "wb") as f:
            f.write(b"corrupted")

        # Trigger validation failure
        checkpoint_manager.validate_checkpoint(checkpoint_id)

        metrics = checkpoint_manager.get_metrics()
        assert metrics["corruption_events"] > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_storage_directory_creation(self, temp_storage_dir):
        """Test automatic storage directory creation."""
        nested_dir = temp_storage_dir / "nested" / "storage"
        manager = CheckpointManager(storage_dir=nested_dir)

        assert nested_dir.exists()
        assert manager.storage_dir == nested_dir.resolve()

    def test_concurrent_access_simulation(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test concurrent access patterns."""
        import threading

        results = []
        errors = []

        def save_checkpoint(index):
            try:
                modified_state = sample_game_state.copy()
                modified_state["thread"] = index
                checkpoint_id = checkpoint_manager.save_checkpoint(modified_state, sample_metadata)
                results.append(checkpoint_id)
            except Exception as e:
                errors.append(e)

        # Simulate concurrent saves
        threads = []
        for i in range(5):
            thread = threading.Thread(target=save_checkpoint, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(errors) == 0
        assert len(results) == 5
        assert len(set(results)) == 5  # All unique IDs

    def test_disk_space_simulation(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test behavior when storage operations fail."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Simulate disk full by making directory read-only
        checkpoint_manager.storage_dir.chmod(0o444)

        try:
            with pytest.raises((OSError, PermissionError)):
                checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        finally:
            # Restore permissions
            checkpoint_manager.storage_dir.chmod(0o755)

    def test_malformed_metadata_handling(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test handling of malformed metadata files."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        metadata_path = checkpoint_manager._get_metadata_path(checkpoint_id)

        # Corrupt metadata file
        with open(metadata_path, "w") as f:
            f.write("invalid json content")

        # Clear cache to force reload from corrupted disk file
        checkpoint_manager._metadata_cache.clear()
        checkpoint_manager._cache_loaded = False

        # Should handle gracefully
        metadata = checkpoint_manager._load_metadata(checkpoint_id)
        assert metadata is None

        # Validation should fail
        assert checkpoint_manager.validate_checkpoint(checkpoint_id) is False


class TestProductionPerformance:
    """Test production performance requirements."""

    def test_load_performance_requirement(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test that load operations meet < 500ms requirement."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Measure load time
        start_time = time.perf_counter()
        checkpoint_manager.load_checkpoint(checkpoint_id)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < checkpoint_manager.MAX_LOAD_TIME_MS

    def test_validation_performance_requirement(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test that validation meets < 100ms requirement."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Measure validation time
        start_time = time.perf_counter()
        checkpoint_manager.validate_checkpoint(checkpoint_id)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < checkpoint_manager.MAX_VALIDATION_TIME_MS

    def test_pruning_performance_requirement(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test that pruning meets < 2s requirement for 100 checkpoints."""
        # Create multiple checkpoints (scaled down for test speed)
        for i in range(20):  # Scaled down from 100 for test efficiency
            modified_state = sample_game_state.copy()
            modified_state["step"] = i
            checkpoint_manager.save_checkpoint(modified_state, sample_metadata)

        # Measure pruning time
        start_time = time.perf_counter()
        checkpoint_manager.prune_checkpoints(max_count=10)
        elapsed_s = time.perf_counter() - start_time

        # Should be well under 2s even for this smaller test
        assert elapsed_s < 1.0  # Conservative for test environment

    @pytest.mark.parametrize("checkpoint_count", [10, 25, 50])
    def test_scaling_performance(
        self, temp_storage_dir, sample_game_state, sample_metadata, checkpoint_count
    ):
        """Test performance scaling with checkpoint count."""
        manager = CheckpointManager(
            storage_dir=temp_storage_dir, max_checkpoints=checkpoint_count + 10
        )

        # Create checkpoints
        creation_times = []
        for i in range(checkpoint_count):
            start_time = time.perf_counter()
            modified_state = sample_game_state.copy()
            modified_state["step"] = i
            manager.save_checkpoint(modified_state, sample_metadata)
            creation_times.append(time.perf_counter() - start_time)

        # Performance should remain reasonable as count increases
        avg_creation_time = sum(creation_times) / len(creation_times)
        assert avg_creation_time < 0.1  # 100ms average

        # Test pruning performance scales
        start_time = time.perf_counter()
        manager.prune_checkpoints(max_count=checkpoint_count // 2)
        pruning_time = time.perf_counter() - start_time

        # Pruning should scale reasonably
        assert pruning_time < (checkpoint_count * 0.01)  # 10ms per checkpoint max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
