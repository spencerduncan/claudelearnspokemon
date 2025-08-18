"""
Comprehensive test suite for CheckpointManager.

Tests cover production-grade features:
- Core save/load functionality with performance validation
- Checkpoint pruning algorithm with value scoring
- Integrity validation with corruption simulation
- Error handling and edge cases
- Metrics and observability
- Concurrent access patterns

Author: Bot Dean - Production-First Testing
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import lz4.frame
import pytest

from claudelearnspokemon.checkpoint_manager import (
    CheckpointCorruptionError,
    CheckpointError,
    CheckpointManager,
    CheckpointNotFoundError,
)


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def checkpoint_manager(temp_storage_dir):
    """Create CheckpointManager instance for testing."""
    return CheckpointManager(
        temp_storage_dir,
        max_checkpoints=5,  # Small limit for testing
        enable_metrics=True,
    )


@pytest.fixture
def checkpoint_manager_no_auto_prune(temp_storage_dir):
    """Create CheckpointManager instance for pruning tests without auto-pruning interference."""
    return CheckpointManager(
        temp_storage_dir,
        max_checkpoints=100,  # High limit to prevent auto-pruning during tests
        enable_metrics=True,
    )


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_game_state():
    """Sample game state for testing."""
    return {
        "player": {
            "name": "RED",
            "position": {"x": 100, "y": 150},
            "level": 25,
            "health": 80,
        },
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
        "location": "cerulean_city",
        "progress_percentage": 15.5,
        "strategy": "speed_run_route",
        "notes": "After defeating Misty",
        "progress_markers": {
            "badges": 2,
            "pokemon_caught": 2,
            "story_flags": ["gym_leader_misty_defeated"],
        },
    }


@pytest.mark.medium
class TestCheckpointManagerCore:
    """Test core save/load functionality."""

    def test_initialization(self, temp_storage_dir):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(temp_storage_dir, max_checkpoints=100, enable_metrics=True)

        assert manager.storage_dir == temp_storage_dir.resolve()
        assert manager.max_checkpoints == 100
        assert manager.enable_metrics is True
        assert manager.storage_dir.exists()

    def test_checkpoint_manager_initialization(self, temp_checkpoint_dir) -> None:
        """Test CheckpointManager initializes with correct directory."""
        manager = CheckpointManager(temp_checkpoint_dir)
        assert manager.storage_dir == Path(temp_checkpoint_dir)
        assert manager.storage_dir.exists()

    def test_checkpoint_manager_default_directory(self) -> None:
        """Test CheckpointManager uses default directory when none specified."""
        manager = CheckpointManager()
        expected_dir = Path.home() / ".claudelearnspokemon" / "checkpoints"
        assert manager.storage_dir == expected_dir

    def test_save_checkpoint_success(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test successful checkpoint saving."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        assert checkpoint_id is not None
        assert isinstance(checkpoint_id, str)
        assert len(checkpoint_id) == 36  # UUID length

        # Verify file exists
        checkpoint_file = checkpoint_manager.storage_dir / f"{checkpoint_id}.lz4"
        metadata_file = checkpoint_manager.storage_dir / f"{checkpoint_id}.metadata.json"

        assert checkpoint_file.exists()
        assert metadata_file.exists()
        assert checkpoint_file.stat().st_size > 0

    def test_save_checkpoint_invalid_input(self, checkpoint_manager):
        """Test checkpoint saving with invalid inputs."""
        # Test invalid game state
        with pytest.raises(ValueError, match="game_state must be a dictionary"):
            checkpoint_manager.save_checkpoint("invalid", {"location": "test"})

        # Test invalid metadata
        with pytest.raises(ValueError, match="metadata must be a dictionary"):
            checkpoint_manager.save_checkpoint({"valid": "state"}, "invalid")

        # Test empty game state
        with pytest.raises(ValueError, match="game_state cannot be empty"):
            checkpoint_manager.save_checkpoint({}, {"location": "test"})

    def test_save_checkpoint_returns_uuid(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test save_checkpoint returns valid UUID string."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # UUID format validation
        assert isinstance(checkpoint_id, str)
        assert len(checkpoint_id) == 36  # UUID length
        assert checkpoint_id.count("-") == 4  # UUID has 4 hyphens

    def test_save_checkpoint_creates_file(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test save_checkpoint creates compressed file on disk."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        checkpoint_file = checkpoint_manager.storage_dir / f"{checkpoint_id}.lz4"
        assert checkpoint_file.exists()
        assert checkpoint_file.stat().st_size > 0

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

    def test_load_checkpoint_returns_game_state(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test load_checkpoint returns original game state."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        loaded_state = checkpoint_manager.load_checkpoint(checkpoint_id)

        assert loaded_state == sample_game_state

    def test_checkpoint_round_trip_preserves_data(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test save/load round trip preserves all data exactly."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        loaded_state = checkpoint_manager.load_checkpoint(checkpoint_id)

        # Deep comparison of nested structures
        assert loaded_state["player"]["name"] == sample_game_state["player"]["name"]
        assert loaded_state["pokemon"][0]["moves"] == sample_game_state["pokemon"][0]["moves"]
        assert loaded_state["inventory"] == sample_game_state["inventory"]
        assert loaded_state["flags"] == sample_game_state["flags"]

    def test_checkpoint_files_are_lz4_compressed(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test checkpoint files are valid LZ4 compressed format."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        checkpoint_file = checkpoint_manager.storage_dir / f"{checkpoint_id}.lz4"
        with checkpoint_file.open("rb") as f:
            compressed_data = f.read()

        # Should be able to decompress with LZ4
        decompressed_data = lz4.frame.decompress(compressed_data)

        # Should contain valid JSON
        checkpoint_data = json.loads(decompressed_data.decode("utf-8"))
        assert checkpoint_data["game_state"] == sample_game_state

    def test_compression_reduces_file_size(self, checkpoint_manager) -> None:
        """Test compression significantly reduces file size for large states."""
        # Create large game state with repetitive data
        large_game_state = {
            "large_inventory": ["potion"] * 1000,
            "repetitive_data": "x" * 10000,
            "nested_data": [{"same_structure": i} for i in range(500)],
        }

        checkpoint_id = checkpoint_manager.save_checkpoint(large_game_state, {})

        # Calculate original JSON size
        original_json = json.dumps(
            {
                "version": "1.0",
                "checkpoint_id": checkpoint_id,
                "timestamp": time.time(),
                "game_state": large_game_state,
                "metadata": {},
            },
            separators=(",", ":"),
        )
        original_size = len(original_json.encode("utf-8"))

        # Get compressed file size
        compressed_size = checkpoint_manager.get_checkpoint_size(checkpoint_id)

        # Should achieve significant compression
        compression_ratio = compressed_size / original_size
        assert compression_ratio < 0.1, f"Poor compression ratio: {compression_ratio}"

    def test_load_checkpoint_not_found(self, checkpoint_manager):
        """Test loading non-existent checkpoint."""
        with pytest.raises(CheckpointNotFoundError):
            checkpoint_manager.load_checkpoint("nonexistent-id")


@pytest.mark.medium
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
        """Test validation of missing checkpoint."""
        assert checkpoint_manager.validate_checkpoint("missing-id") is False

    def test_validate_checkpoint_corrupted_data(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test validation with corrupted checkpoint data."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Corrupt the checkpoint file
        checkpoint_file = checkpoint_manager.storage_dir / f"{checkpoint_id}.lz4"
        with checkpoint_file.open("wb") as f:
            f.write(b"corrupted data that is not valid LZ4")

        # Should fail validation due to decompression error
        assert checkpoint_manager.validate_checkpoint(checkpoint_id) is False

    def test_validate_empty_checkpoint_id(self, checkpoint_manager):
        """Test validation with empty checkpoint ID."""
        assert checkpoint_manager.validate_checkpoint("") is False
        assert checkpoint_manager.validate_checkpoint(None) is False


@pytest.mark.medium
class TestCheckpointPruning:
    """Test checkpoint pruning algorithm."""

    def test_prune_checkpoints_no_action_needed(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test pruning when no action is needed."""
        # Save fewer checkpoints than the limit
        for i in range(3):
            checkpoint_manager.save_checkpoint(sample_game_state, {"location": f"location_{i}"})

        result = checkpoint_manager.prune_checkpoints(5)

        assert result["action"] == "no_pruning_needed"
        assert result["total_checkpoints"] == 3
        assert len(result["removed"]) == 0

    def test_prune_checkpoints_removes_excess(
        self, checkpoint_manager_no_auto_prune, sample_game_state, sample_metadata
    ):
        """Test pruning removes excess checkpoints."""
        # Save more checkpoints than the limit
        checkpoint_ids = []
        for i in range(10):
            checkpoint_id = checkpoint_manager_no_auto_prune.save_checkpoint(
                sample_game_state, {"location": f"location_{i}"}
            )
            checkpoint_ids.append(checkpoint_id)

        result = checkpoint_manager_no_auto_prune.prune_checkpoints(5)

        assert result["action"] == "pruning_executed"
        assert result["total_checkpoints"] == 10
        assert result["target_count"] == 5
        assert len(result["removed"]) == 5
        assert len(result["retained"]) == 5

        # Verify files are actually removed
        for checkpoint_id in result["removed"]:
            checkpoint_file = checkpoint_manager_no_auto_prune.storage_dir / f"{checkpoint_id}.lz4"
            assert not checkpoint_file.exists()

    def test_prune_checkpoints_dry_run(
        self, checkpoint_manager_no_auto_prune, sample_game_state, sample_metadata
    ):
        """Test pruning dry run doesn't remove files."""
        # Save checkpoints
        for i in range(8):
            checkpoint_manager_no_auto_prune.save_checkpoint(
                sample_game_state, {"location": f"location_{i}"}
            )

        result = checkpoint_manager_no_auto_prune.prune_checkpoints(5, dry_run=True)

        assert result["action"] == "dry_run"
        assert len(result["removed"]) == 3

        # Verify files still exist
        all_checkpoint_ids = checkpoint_manager_no_auto_prune._get_all_checkpoint_ids()
        assert len(all_checkpoint_ids) == 8

    def test_prune_checkpoints_performance(
        self, checkpoint_manager_no_auto_prune, sample_game_state
    ):
        """Test pruning performance meets requirements with production-grade timeout protection."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Performance test exceeded 3-second timeout - indicates bottleneck")

        # Production-grade circuit breaker: 3-second hard timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(3)

        try:
            # Create many checkpoints to test performance
            for i in range(50):
                checkpoint_manager_no_auto_prune.save_checkpoint(
                    sample_game_state, {"location": f"loc_{i}"}
                )

            start_time = time.perf_counter()
            checkpoint_manager_no_auto_prune.prune_checkpoints(25)
            pruning_time = time.perf_counter() - start_time

            # Performance requirement: < 2s for pruning operation
            assert pruning_time < checkpoint_manager_no_auto_prune.MAX_PRUNING_TIME_S

            # Production validation: Log actual performance for monitoring
            print(
                f"✅ PRODUCTION PERF: Pruning completed in {pruning_time:.3f}s (target: {checkpoint_manager_no_auto_prune.MAX_PRUNING_TIME_S}s)"
            )

        finally:
            # Always clean up timeout handler - production hygiene
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def test_prune_checkpoints_value_scoring(
        self, checkpoint_manager_no_auto_prune, sample_game_state, sample_metadata
    ):
        """Test pruning uses value scoring correctly."""
        checkpoint_manager = checkpoint_manager_no_auto_prune
        # Create checkpoints with different access patterns
        checkpoint_ids = []
        for i in range(6):
            checkpoint_id = checkpoint_manager.save_checkpoint(
                sample_game_state, {"location": f"important_location_{i}"}
            )
            checkpoint_ids.append(checkpoint_id)

        # Simulate different access patterns to create score differences
        # Access some checkpoints more frequently
        for _ in range(5):
            checkpoint_manager.load_checkpoint(checkpoint_ids[0])  # High access

        for _ in range(2):
            checkpoint_manager.load_checkpoint(checkpoint_ids[1])  # Medium access

        # checkpoint_ids[2:] remain unaccessed (low access)

        result = checkpoint_manager.prune_checkpoints(3)

        # High-access checkpoints should be retained
        assert checkpoint_ids[0] in result["retained"]
        assert checkpoint_ids[1] in result["retained"]

    def test_prune_invalid_max_count(self, checkpoint_manager):
        """Test pruning with invalid max_count."""
        with pytest.raises(ValueError, match="max_count must be at least 1"):
            checkpoint_manager.prune_checkpoints(0)

        with pytest.raises(ValueError, match="max_count must be at least 1"):
            checkpoint_manager.prune_checkpoints(-1)


@pytest.mark.medium
class TestCheckpointListing:
    """Test checkpoint listing and querying."""

    def test_list_checkpoints_empty(self, checkpoint_manager):
        """Test listing when no checkpoints exist."""
        result = checkpoint_manager.list_checkpoints({})
        assert result == []

    def test_list_checkpoints_with_criteria(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test listing checkpoints with filter criteria."""
        # Create checkpoints at different locations
        locations = ["route_1", "cerulean_city", "route_1", "lavender_town"]
        for location in locations:
            checkpoint_manager.save_checkpoint(sample_game_state, {"location": location})

        # Filter by location
        route_checkpoints = checkpoint_manager.list_checkpoints({"location": "route_1"})
        assert len(route_checkpoints) == 2

        # All results should have the correct location
        for checkpoint in route_checkpoints:
            assert checkpoint["location"] == "route_1"

    def test_find_nearest_checkpoint(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test finding nearest checkpoint by location."""
        # Create checkpoints
        locations = ["route_1", "cerulean_city", "lavender_town"]
        checkpoint_ids = []
        for location in locations:
            checkpoint_id = checkpoint_manager.save_checkpoint(
                sample_game_state, {"location": location}
            )
            checkpoint_ids.append(checkpoint_id)

        # Find exact match
        found_id = checkpoint_manager.find_nearest_checkpoint("cerulean_city")
        assert found_id in checkpoint_ids

        # Non-existent location
        not_found_id = checkpoint_manager.find_nearest_checkpoint("nonexistent_location")
        assert not_found_id == ""


@pytest.mark.medium
class TestMetricsAndObservability:
    """Test metrics collection and observability features."""

    def test_metrics_collection(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test comprehensive metrics collection."""
        # Initial metrics
        metrics = checkpoint_manager.get_metrics()
        assert metrics["saves_total"] == 0
        assert metrics["loads_total"] == 0
        assert metrics["validations_total"] == 0

        # Perform operations
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        checkpoint_manager.load_checkpoint(checkpoint_id)
        checkpoint_manager.validate_checkpoint(checkpoint_id)

        # Check updated metrics
        final_metrics = checkpoint_manager.get_metrics()
        assert final_metrics["saves_total"] == 1
        assert final_metrics["loads_total"] == 1
        assert final_metrics["validations_total"] >= 1  # validation called during load too

    def test_metrics_disabled(self, temp_storage_dir):
        """Test behavior when metrics are disabled."""
        manager = CheckpointManager(temp_storage_dir, enable_metrics=False)

        metrics = manager.get_metrics()
        assert metrics["metrics_disabled"] is True

    def test_storage_utilization_tracking(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test storage utilization metrics."""
        # Save checkpoints
        for i in range(3):
            checkpoint_manager.save_checkpoint(sample_game_state, {"location": f"loc_{i}"})

        metrics = checkpoint_manager.get_metrics()
        assert metrics["checkpoint_count"] == 3
        assert metrics["storage_utilization"] == 3 / 5  # 3 checkpoints, max 5
        assert metrics["storage_bytes_used"] > 0

    def test_corruption_event_tracking(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test corruption event metrics."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Corrupt the checkpoint file
        checkpoint_file = checkpoint_manager.storage_dir / f"{checkpoint_id}.lz4"
        with checkpoint_file.open("wb") as f:
            f.write(b"corrupted")

        # Trigger validation failure
        checkpoint_manager.validate_checkpoint(checkpoint_id)

        metrics = checkpoint_manager.get_metrics()
        assert metrics["corruption_events"] > 0


@pytest.mark.medium
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_storage_directory_creation(self, temp_storage_dir):
        """Test automatic storage directory creation."""
        nested_dir = temp_storage_dir / "nested" / "storage"
        manager = CheckpointManager(nested_dir)

        assert nested_dir.exists()
        assert manager.storage_dir == nested_dir.resolve()

    def test_load_nonexistent_checkpoint_raises_not_found(self, checkpoint_manager) -> None:
        """Test loading non-existent checkpoint raises CheckpointNotFoundError."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        with pytest.raises(CheckpointNotFoundError) as exc_info:
            checkpoint_manager.load_checkpoint(fake_id)

        assert fake_id in str(exc_info.value)

    def test_load_corrupted_lz4_data_raises_corruption_error(
        self, checkpoint_manager, temp_checkpoint_dir
    ) -> None:
        """Test loading corrupted LZ4 data raises CheckpointCorruptionError."""
        # Create file with invalid LZ4 data
        fake_id = "corrupted-checkpoint-id"
        corrupt_file = checkpoint_manager.storage_dir / f"{fake_id}.lz4"

        with corrupt_file.open("wb") as f:
            f.write(b"this is not valid LZ4 data")

        with pytest.raises(CheckpointCorruptionError) as exc_info:
            checkpoint_manager.load_checkpoint(fake_id)

        assert "decompress" in str(exc_info.value).lower()

    def test_load_corrupted_json_raises_corruption_error(
        self, checkpoint_manager, temp_checkpoint_dir
    ) -> None:
        """Test loading corrupted JSON data raises CheckpointCorruptionError."""
        fake_id = "corrupted-json-id"
        corrupt_file = checkpoint_manager.storage_dir / f"{fake_id}.lz4"

        # Create valid LZ4 file with invalid JSON
        invalid_json = b"this is not valid JSON data"
        compressed_data = lz4.frame.compress(invalid_json)

        with corrupt_file.open("wb") as f:
            f.write(compressed_data)

        with pytest.raises(CheckpointCorruptionError) as exc_info:
            checkpoint_manager.load_checkpoint(fake_id)

        assert "parse" in str(exc_info.value).lower()

    def test_load_missing_required_fields_raises_corruption_error(
        self, checkpoint_manager, temp_checkpoint_dir
    ) -> None:
        """Test loading checkpoint missing required fields raises corruption error."""
        fake_id = "missing-fields-id"
        corrupt_file = checkpoint_manager.storage_dir / f"{fake_id}.lz4"

        # Create checkpoint missing required fields
        incomplete_data = {
            "version": "1.0",
            # Missing checkpoint_id, game_state, metadata
        }
        json_data = json.dumps(incomplete_data).encode("utf-8")
        compressed_data = lz4.frame.compress(json_data)

        with corrupt_file.open("wb") as f:
            f.write(compressed_data)

        with pytest.raises(CheckpointCorruptionError) as exc_info:
            checkpoint_manager.load_checkpoint(fake_id)

        assert "missing field" in str(exc_info.value).lower()

    def test_load_mismatched_checkpoint_id_raises_corruption_error(
        self, checkpoint_manager, temp_checkpoint_dir
    ) -> None:
        """Test loading checkpoint with mismatched ID raises corruption error."""
        file_id = "file-checkpoint-id"
        content_id = "content-checkpoint-id"

        corrupt_file = checkpoint_manager.storage_dir / f"{file_id}.lz4"

        # Create checkpoint with mismatched ID
        mismatched_data = {
            "version": "1.0",
            "checkpoint_id": content_id,  # Different from file name
            "game_state": {},
            "metadata": {},
        }
        json_data = json.dumps(mismatched_data).encode("utf-8")
        compressed_data = lz4.frame.compress(json_data)

        with corrupt_file.open("wb") as f:
            f.write(compressed_data)

        with pytest.raises(CheckpointCorruptionError) as exc_info:
            checkpoint_manager.load_checkpoint(file_id)

        assert "mismatch" in str(exc_info.value).lower()

    def test_save_with_invalid_data_types_handles_gracefully(self, checkpoint_manager) -> None:
        """Test saving data that can't be JSON serialized raises CheckpointError."""
        # Create data that can't be JSON serialized
        invalid_state = {
            "function_object": lambda x: x,  # Can't serialize functions
        }

        with pytest.raises(CheckpointError):
            checkpoint_manager.save_checkpoint(invalid_state, {})

    def test_empty_checkpoint_file_raises_corruption_error(
        self, checkpoint_manager, temp_checkpoint_dir
    ) -> None:
        """Test loading empty checkpoint file raises CheckpointCorruptionError."""
        empty_id = "empty-checkpoint-id"
        empty_file = checkpoint_manager.storage_dir / f"{empty_id}.lz4"

        # Create empty file
        empty_file.touch()

        with pytest.raises(CheckpointCorruptionError) as exc_info:
            checkpoint_manager.load_checkpoint(empty_id)

        assert "empty" in str(exc_info.value).lower()

    def test_concurrent_checkpoint_operations(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ):
        """Test concurrent checkpoint operations."""
        import threading

        results = []
        errors = []

        def save_checkpoint(index):
            try:
                modified_state = sample_game_state.copy()
                modified_state["player"]["name"] = f"Player_{index}"
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    modified_state, {"location": f"location_{index}"}
                )
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


@pytest.mark.medium
class TestCheckpointManagerPerformance:
    """Test performance requirements."""

    def test_save_checkpoint_under_500ms(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test save_checkpoint completes within 500ms performance requirement."""
        start_time = time.monotonic()
        checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        duration = time.monotonic() - start_time

        assert duration < 0.5, f"Save took {duration:.3f}s, exceeds 500ms limit"

    def test_load_checkpoint_under_500ms(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test load_checkpoint completes within 500ms performance requirement."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        start_time = time.monotonic()
        checkpoint_manager.load_checkpoint(checkpoint_id)
        duration = time.monotonic() - start_time

        assert duration < 0.5, f"Load took {duration:.3f}s, exceeds 500ms limit"

    def test_large_game_state_performance(self, checkpoint_manager) -> None:
        """Test performance with large game states (simulating complex Pokemon data)."""
        # Create large game state similar to full Pokemon game state
        large_state = {
            "player_data": {
                "name": "RED",
                "position": {"x": 100, "y": 150, "map": "route_1"},
                "stats": {"level": 50, "money": 999999},
            },
            "pokemon_team": [
                {
                    "species": f"POKEMON_{i}",
                    "level": 50 + i,
                    "moves": [f"MOVE_{j}" for j in range(4)],
                    "stats": {"hp": 200 + i, "attack": 150 + i, "defense": 120 + i},
                    "experience": 125000 + i * 1000,
                }
                for i in range(6)
            ],
            "pokemon_box": [
                {
                    "species": f"BOXED_POKEMON_{i}",
                    "level": 10 + (i % 20),
                    "moves": [f"MOVE_{j}" for j in range(4)],
                }
                for i in range(240)  # Full PC box
            ],
            "inventory": {f"ITEM_{i}": (i % 99) + 1 for i in range(100)},
            "game_flags": {f"FLAG_{i}": bool(i % 2) for i in range(1000)},
            "map_data": {
                "visited_locations": [f"LOCATION_{i}" for i in range(200)],
                "npc_interactions": {
                    f"NPC_{i}": {"talked": True, "state": i % 10} for i in range(500)
                },
            },
        }
        metadata = {"location": "end_game", "completion": 95.5}

        # Test save performance
        start_time = time.monotonic()
        checkpoint_id = checkpoint_manager.save_checkpoint(large_state, metadata)
        save_duration = time.monotonic() - start_time

        # Test load performance
        start_time = time.monotonic()
        loaded_state = checkpoint_manager.load_checkpoint(checkpoint_id)
        load_duration = time.monotonic() - start_time

        assert save_duration < 0.5, f"Large state save took {save_duration:.3f}s"
        assert load_duration < 0.5, f"Large state load took {load_duration:.3f}s"
        assert loaded_state == large_state


@pytest.mark.medium
class TestCheckpointManagerAtomicOperations:
    """Test atomic write operations."""

    @patch("os.fsync")
    def test_save_uses_atomic_writes(
        self, mock_fsync, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test save operation uses atomic writes with fsync."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Verify fsync was called (ensures data reaches disk)
        mock_fsync.assert_called()

        # Verify final file exists and temp file doesn't
        final_file = checkpoint_manager.storage_dir / f"{checkpoint_id}.lz4"
        temp_file = checkpoint_manager.storage_dir / f"{checkpoint_id}.tmp"

        assert final_file.exists()
        assert not temp_file.exists()

    def test_save_cleans_up_temp_file_on_failure(
        self, checkpoint_manager, temp_checkpoint_dir
    ) -> None:
        """Test temporary file is cleaned up if save operation fails."""

        # Mock Path.rename to simulate failure after temp file creation
        original_rename = Path.rename

        def failing_rename(self, target):
            if str(self).endswith(".tmp"):
                raise OSError("Simulated rename failure")
            return original_rename(self, target)

        with patch.object(Path, "rename", failing_rename):
            with pytest.raises(CheckpointError):
                checkpoint_manager.save_checkpoint({"test": "data"}, {})

        # Verify no temp files remain
        temp_files = list(checkpoint_manager.storage_dir.glob("*.tmp"))
        assert len(temp_files) == 0, f"Temp files not cleaned up: {temp_files}"


@pytest.mark.medium
class TestCheckpointManagerUtilityMethods:
    """Test utility methods."""

    def test_checkpoint_exists_returns_true_for_existing(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test checkpoint_exists returns True for existing checkpoints."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        assert checkpoint_manager.checkpoint_exists(checkpoint_id) is True

    def test_checkpoint_exists_returns_false_for_nonexistent(self, checkpoint_manager) -> None:
        """Test checkpoint_exists returns False for non-existent checkpoints."""
        fake_id = "nonexistent-checkpoint-id"
        assert checkpoint_manager.checkpoint_exists(fake_id) is False

    def test_get_checkpoint_size_returns_correct_size(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test get_checkpoint_size returns correct file size."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        size = checkpoint_manager.get_checkpoint_size(checkpoint_id)

        # Verify size matches actual file
        checkpoint_file = checkpoint_manager.storage_dir / f"{checkpoint_id}.lz4"
        actual_size = checkpoint_file.stat().st_size

        assert size == actual_size
        assert size > 0

    def test_get_checkpoint_size_raises_not_found_for_nonexistent(self, checkpoint_manager) -> None:
        """Test get_checkpoint_size raises CheckpointNotFoundError for non-existent checkpoints."""
        fake_id = "nonexistent-checkpoint-id"

        with pytest.raises(CheckpointNotFoundError):
            checkpoint_manager.get_checkpoint_size(fake_id)

    def test_get_performance_stats_tracks_operations(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test get_performance_stats tracks save/load operations."""
        # Initially no operations
        stats = checkpoint_manager.get_performance_stats()
        assert stats["save_operations"] == 0
        assert stats["load_operations"] == 0

        # After save operation
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        stats = checkpoint_manager.get_performance_stats()
        assert stats["save_operations"] == 1
        assert stats["load_operations"] == 0
        assert "avg_save_time_ms" in stats

        # After load operation
        checkpoint_manager.load_checkpoint(checkpoint_id)
        stats = checkpoint_manager.get_performance_stats()
        assert stats["save_operations"] == 1
        assert stats["load_operations"] == 1
        assert "avg_load_time_ms" in stats


@pytest.mark.medium
class TestCheckpointManagerUniqueIdentifiers:
    """Test UUID generation uniqueness."""

    def test_save_checkpoint_generates_unique_identifiers(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test multiple saves generate unique checkpoint identifiers."""
        ids = []

        for _i in range(10):
            checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
            ids.append(checkpoint_id)

        # All IDs should be unique
        assert len(set(ids)) == len(ids), "Generated non-unique checkpoint IDs"

    def test_concurrent_saves_generate_unique_identifiers(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test concurrent saves generate unique identifiers."""
        import threading

        ids = []
        ids_lock = threading.Lock()

        def save_checkpoint():
            checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
            with ids_lock:
                ids.append(checkpoint_id)

        # Run concurrent saves
        threads = []
        for _i in range(5):
            thread = threading.Thread(target=save_checkpoint)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All IDs should be unique
        assert len(set(ids)) == len(ids), f"Generated non-unique IDs: {ids}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
