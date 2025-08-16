"""
Unit tests for CheckpointManager metadata management functionality.

Tests all aspects of checkpoint storage, metadata management, validation,
and basic querying capabilities.
"""

import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
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
def temp_checkpoint_dir():
    """Create temporary directory for checkpoint tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


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
    """Sample checkpoint metadata."""
    return {
        "game_location": "cerulean_city",
        "progress_markers": ["got_starter", "visited_oak"],
        "performance_metrics": {"execution_time": 5.2, "success_rate": 0.85},
        "tags": ["tutorial", "important"],
        "custom_fields": {"strategy_type": "speedrun", "difficulty": "normal"},
    }


class TestCheckpointManagerBasics:
    """Test basic save/load functionality."""

    def test_checkpoint_manager_initialization(self, temp_checkpoint_dir) -> None:
        """Test CheckpointManager initializes with correct directory."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        assert manager.checkpoint_dir == Path(temp_checkpoint_dir)
        assert manager.checkpoint_dir.exists()

    def test_checkpoint_manager_default_directory(self) -> None:
        """Test CheckpointManager uses default directory when none specified."""
        manager = CheckpointManager()
        expected_dir = Path.home() / ".claudelearnspokemon" / "checkpoints"
        assert manager.checkpoint_dir == expected_dir

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

        checkpoint_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.lz4"
        assert checkpoint_file.exists()

    def test_load_checkpoint_returns_game_state(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test load_checkpoint returns original game state."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        loaded_state = checkpoint_manager.load_checkpoint(checkpoint_id)

        assert loaded_state == sample_game_state
        assert loaded_state["player"]["name"] == sample_game_state["player"]["name"]
        assert loaded_state["pokemon"][0]["moves"] == sample_game_state["pokemon"][0]["moves"]

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

    def test_get_checkpoint_metadata_returns_enhanced_metadata(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test get_checkpoint_metadata returns metadata with added fields."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)

        assert metadata["checkpoint_id"] == checkpoint_id
        assert "created_at" in metadata
        assert metadata["game_location"] == "cerulean_city"
        assert metadata["progress_markers"] == ["got_starter", "visited_oak"]
        assert metadata["performance_metrics"]["execution_time"] == 5.2
        assert metadata["performance_metrics"]["success_rate"] == 0.85
        assert metadata["tags"] == ["tutorial", "important"]
        assert metadata["custom_fields"]["strategy_type"] == "speedrun"

    def test_checkpoint_files_are_lz4_compressed(
        self, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test checkpoint files are valid LZ4 compressed format."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        checkpoint_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.lz4"
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
                "metadata": {
                    "checkpoint_id": checkpoint_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            },
            separators=(",", ":"),
        )
        original_size = len(original_json.encode("utf-8"))

        # Get compressed file size
        compressed_size = checkpoint_manager.get_checkpoint_size(checkpoint_id)

        # Should achieve significant compression
        compression_ratio = compressed_size / original_size
        assert compression_ratio < 0.1, f"Poor compression ratio: {compression_ratio}"

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
        corrupt_file = Path(temp_checkpoint_dir) / f"{fake_id}.lz4"

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
        corrupt_file = Path(temp_checkpoint_dir) / f"{fake_id}.lz4"

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
        corrupt_file = Path(temp_checkpoint_dir) / f"{fake_id}.lz4"

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

    def test_save_with_invalid_data_types_handles_gracefully(self, checkpoint_manager) -> None:
        """Test saving data that can't be JSON serialized raises CheckpointError."""
        # Create data that can't be JSON serialized
        invalid_state = {
            "function_object": lambda x: x,  # Can't serialize functions
        }

        with pytest.raises(CheckpointError):
            checkpoint_manager.save_checkpoint(invalid_state, {})


class TestCheckpointManagerQuerying:
    """Test basic querying functionality."""

    def test_list_checkpoints_returns_all(self, checkpoint_manager, sample_game_state) -> None:
        """Test list_checkpoints returns all checkpoints when no criteria given."""
        # Create multiple checkpoints
        id1 = checkpoint_manager.save_checkpoint(
            sample_game_state, {"game_location": "pallet_town"}
        )
        id2 = checkpoint_manager.save_checkpoint(
            sample_game_state, {"game_location": "viridian_city"}
        )
        id3 = checkpoint_manager.save_checkpoint(
            sample_game_state, {"game_location": "pewter_city"}
        )

        results = checkpoint_manager.list_checkpoints()

        assert len(results) == 3
        found_ids = [r["checkpoint_id"] for r in results]
        assert id1 in found_ids
        assert id2 in found_ids
        assert id3 in found_ids

    def test_list_checkpoints_filters_by_location(
        self, checkpoint_manager, sample_game_state
    ) -> None:
        """Test list_checkpoints filters by game location."""
        id1 = checkpoint_manager.save_checkpoint(
            sample_game_state, {"game_location": "pallet_town"}
        )
        id2 = checkpoint_manager.save_checkpoint(
            sample_game_state, {"game_location": "viridian_city"}
        )
        id3 = checkpoint_manager.save_checkpoint(
            sample_game_state, {"game_location": "pallet_town_gym"}
        )

        # Filter by location substring
        results = checkpoint_manager.list_checkpoints(criteria={"location": "pallet"})

        assert len(results) == 2
        found_ids = [r["checkpoint_id"] for r in results]
        assert id1 in found_ids
        assert id3 in found_ids
        assert id2 not in found_ids

    def test_list_checkpoints_filters_by_tags(self, checkpoint_manager, sample_game_state) -> None:
        """Test list_checkpoints filters by tags."""
        id1 = checkpoint_manager.save_checkpoint(
            sample_game_state, {"tags": ["tutorial", "important"]}
        )
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, {"tags": ["battle", "gym"]})
        id3 = checkpoint_manager.save_checkpoint(
            sample_game_state, {"tags": ["tutorial", "completed"]}
        )

        # Filter by tags
        results = checkpoint_manager.list_checkpoints(criteria={"tags": ["tutorial"]})

        assert len(results) == 2
        found_ids = [r["checkpoint_id"] for r in results]
        assert id1 in found_ids
        assert id3 in found_ids
        assert id2 not in found_ids

    def test_list_checkpoints_respects_limit(self, checkpoint_manager, sample_game_state) -> None:
        """Test list_checkpoints respects limit parameter."""
        # Create 5 checkpoints
        for i in range(5):
            checkpoint_manager.save_checkpoint(
                sample_game_state, {"game_location": f"location_{i}"}
            )

        results = checkpoint_manager.list_checkpoints(limit=3)
        assert len(results) == 3

    def test_count_checkpoints_returns_correct_count(
        self, checkpoint_manager, sample_game_state
    ) -> None:
        """Test count_checkpoints returns accurate count."""
        # Create checkpoints
        checkpoint_manager.save_checkpoint(sample_game_state, {"game_location": "pallet_town"})
        checkpoint_manager.save_checkpoint(sample_game_state, {"game_location": "viridian_city"})
        checkpoint_manager.save_checkpoint(sample_game_state, {"game_location": "pallet_town"})

        # Total count
        total_count = checkpoint_manager.count_checkpoints()
        assert total_count == 3

        # Filtered count
        filtered_count = checkpoint_manager.count_checkpoints(criteria={"location": "pallet"})
        assert filtered_count == 2


class TestCheckpointManagerPerformance:
    """Performance tests for CheckpointManager."""

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
        metadata = {"game_location": "end_game", "completion": 95.5}

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

    @patch("os.fsync")
    def test_save_uses_atomic_writes(
        self, mock_fsync, checkpoint_manager, sample_game_state, sample_metadata
    ) -> None:
        """Test save operation uses atomic writes with fsync."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)

        # Verify fsync was called (ensures data reaches disk)
        mock_fsync.assert_called_once()

        # Verify final file exists and temp file doesn't
        final_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.lz4"
        temp_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.tmp"

        assert final_file.exists()
        assert not temp_file.exists()


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
        checkpoint_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.lz4"
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
