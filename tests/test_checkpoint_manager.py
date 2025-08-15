"""
Comprehensive unit tests for CheckpointManager core save/load operations.

Tests cover:
- Basic save/load functionality
- LZ4 compression validation
- UUID generation uniqueness
- Error handling and edge cases
- Performance requirements (<500ms)
- Atomic write operations
- Large game state handling
"""

import json
import lz4.frame
import os
import pytest
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

from claudelearnspokemon.checkpoint_manager import (
    CheckpointManager,
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointCorruptionError,
)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create CheckpointManager with temporary directory."""
    return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)


@pytest.fixture
def sample_game_state():
    """Sample game state data."""
    return {
        "player": {
            "name": "RED",
            "position": {"x": 100, "y": 150},
            "level": 25,
            "health": 80
        },
        "pokemon": [
            {"name": "PIKACHU", "level": 25, "health": 65, "moves": ["THUNDERBOLT", "QUICK_ATTACK"]},
            {"name": "CHARMANDER", "level": 12, "health": 39, "moves": ["SCRATCH", "GROWL"]}
        ],
        "inventory": {
            "pokeball": 10,
            "potion": 5,
            "rare_candy": 1
        },
        "flags": {
            "gym_badges": ["boulder", "cascade"],
            "story_progress": "cerulean_city"
        }
    }


@pytest.fixture
def sample_metadata():
    """Sample checkpoint metadata."""
    return {
        "location": "cerulean_city",
        "progress_percentage": 15.5,
        "strategy": "speed_run_route",
        "notes": "After defeating Misty"
    }


class TestCheckpointManagerBasics:
    """Test basic save/load functionality."""

    def test_checkpoint_manager_initialization(self, temp_checkpoint_dir):
        """Test CheckpointManager initializes with correct directory."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        assert manager.checkpoint_dir == Path(temp_checkpoint_dir)
        assert manager.checkpoint_dir.exists()

    def test_checkpoint_manager_default_directory(self):
        """Test CheckpointManager uses default directory when none specified."""
        manager = CheckpointManager()
        expected_dir = Path.home() / ".claudelearnspokemon" / "checkpoints"
        assert manager.checkpoint_dir == expected_dir

    def test_save_checkpoint_returns_uuid(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test save_checkpoint returns valid UUID string."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        # UUID format validation
        assert isinstance(checkpoint_id, str)
        assert len(checkpoint_id) == 36  # UUID length
        assert checkpoint_id.count('-') == 4  # UUID has 4 hyphens

    def test_save_checkpoint_creates_file(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test save_checkpoint creates compressed file on disk."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        checkpoint_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.lz4"
        assert checkpoint_file.exists()
        assert checkpoint_file.stat().st_size > 0

    def test_load_checkpoint_returns_game_state(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test load_checkpoint returns original game state."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        loaded_state = checkpoint_manager.load_checkpoint(checkpoint_id)
        
        assert loaded_state == sample_game_state

    def test_checkpoint_round_trip_preserves_data(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test save/load round trip preserves all data exactly."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        loaded_state = checkpoint_manager.load_checkpoint(checkpoint_id)
        
        # Deep comparison of nested structures
        assert loaded_state["player"]["name"] == sample_game_state["player"]["name"]
        assert loaded_state["pokemon"][0]["moves"] == sample_game_state["pokemon"][0]["moves"]
        assert loaded_state["inventory"] == sample_game_state["inventory"]
        assert loaded_state["flags"] == sample_game_state["flags"]


class TestCheckpointManagerCompression:
    """Test LZ4 compression functionality."""

    def test_checkpoint_files_are_lz4_compressed(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test checkpoint files are valid LZ4 compressed format."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        checkpoint_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.lz4"
        with checkpoint_file.open('rb') as f:
            compressed_data = f.read()
        
        # Should be able to decompress with LZ4
        decompressed_data = lz4.frame.decompress(compressed_data)
        
        # Should contain valid JSON
        checkpoint_data = json.loads(decompressed_data.decode('utf-8'))
        assert checkpoint_data["game_state"] == sample_game_state

    def test_compression_reduces_file_size(self, checkpoint_manager):
        """Test compression significantly reduces file size for large states."""
        # Create large game state with repetitive data
        large_game_state = {
            "large_inventory": ["potion"] * 1000,
            "repetitive_data": "x" * 10000,
            "nested_data": [{"same_structure": i} for i in range(500)]
        }
        
        checkpoint_id = checkpoint_manager.save_checkpoint(large_game_state, {})
        
        # Calculate original JSON size
        original_json = json.dumps({
            "version": "1.0",
            "checkpoint_id": checkpoint_id,
            "timestamp": time.time(),
            "game_state": large_game_state,
            "metadata": {}
        }, separators=(',', ':'))
        original_size = len(original_json.encode('utf-8'))
        
        # Get compressed file size
        compressed_size = checkpoint_manager.get_checkpoint_size(checkpoint_id)
        
        # Should achieve significant compression
        compression_ratio = compressed_size / original_size
        assert compression_ratio < 0.1, f"Poor compression ratio: {compression_ratio}"


class TestCheckpointManagerErrors:
    """Test error handling and edge cases."""

    def test_load_nonexistent_checkpoint_raises_not_found(self, checkpoint_manager):
        """Test loading non-existent checkpoint raises CheckpointNotFoundError."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        
        with pytest.raises(CheckpointNotFoundError) as exc_info:
            checkpoint_manager.load_checkpoint(fake_id)
        
        assert fake_id in str(exc_info.value)

    def test_load_corrupted_lz4_data_raises_corruption_error(self, checkpoint_manager, temp_checkpoint_dir):
        """Test loading corrupted LZ4 data raises CheckpointCorruptionError."""
        # Create file with invalid LZ4 data
        fake_id = "corrupted-checkpoint-id"
        corrupt_file = Path(temp_checkpoint_dir) / f"{fake_id}.lz4"
        
        with corrupt_file.open('wb') as f:
            f.write(b"this is not valid LZ4 data")
        
        with pytest.raises(CheckpointCorruptionError) as exc_info:
            checkpoint_manager.load_checkpoint(fake_id)
        
        assert "decompress" in str(exc_info.value).lower()

    def test_load_corrupted_json_raises_corruption_error(self, checkpoint_manager, temp_checkpoint_dir):
        """Test loading corrupted JSON data raises CheckpointCorruptionError."""
        fake_id = "corrupted-json-id"
        corrupt_file = Path(temp_checkpoint_dir) / f"{fake_id}.lz4"
        
        # Create valid LZ4 file with invalid JSON
        invalid_json = b"this is not valid JSON data"
        compressed_data = lz4.frame.compress(invalid_json)
        
        with corrupt_file.open('wb') as f:
            f.write(compressed_data)
        
        with pytest.raises(CheckpointCorruptionError) as exc_info:
            checkpoint_manager.load_checkpoint(fake_id)
        
        assert "parse" in str(exc_info.value).lower()

    def test_load_missing_required_fields_raises_corruption_error(self, checkpoint_manager, temp_checkpoint_dir):
        """Test loading checkpoint missing required fields raises corruption error."""
        fake_id = "missing-fields-id"
        corrupt_file = Path(temp_checkpoint_dir) / f"{fake_id}.lz4"
        
        # Create checkpoint missing required fields
        incomplete_data = {
            "version": "1.0",
            # Missing checkpoint_id, game_state, metadata
        }
        json_data = json.dumps(incomplete_data).encode('utf-8')
        compressed_data = lz4.frame.compress(json_data)
        
        with corrupt_file.open('wb') as f:
            f.write(compressed_data)
        
        with pytest.raises(CheckpointCorruptionError) as exc_info:
            checkpoint_manager.load_checkpoint(fake_id)
        
        assert "missing field" in str(exc_info.value).lower()

    def test_load_mismatched_checkpoint_id_raises_corruption_error(self, checkpoint_manager, temp_checkpoint_dir):
        """Test loading checkpoint with mismatched ID raises corruption error."""
        file_id = "file-checkpoint-id"
        content_id = "content-checkpoint-id"
        
        corrupt_file = Path(temp_checkpoint_dir) / f"{file_id}.lz4"
        
        # Create checkpoint with mismatched ID
        mismatched_data = {
            "version": "1.0",
            "checkpoint_id": content_id,  # Different from file name
            "game_state": {},
            "metadata": {}
        }
        json_data = json.dumps(mismatched_data).encode('utf-8')
        compressed_data = lz4.frame.compress(json_data)
        
        with corrupt_file.open('wb') as f:
            f.write(compressed_data)
        
        with pytest.raises(CheckpointCorruptionError) as exc_info:
            checkpoint_manager.load_checkpoint(file_id)
        
        assert "mismatch" in str(exc_info.value).lower()

    def test_save_with_invalid_data_types_handles_gracefully(self, checkpoint_manager):
        """Test saving data that can't be JSON serialized raises CheckpointError."""
        # Create data that can't be JSON serialized
        invalid_state = {
            "function_object": lambda x: x,  # Can't serialize functions
        }
        
        with pytest.raises(CheckpointError):
            checkpoint_manager.save_checkpoint(invalid_state, {})

    def test_empty_checkpoint_file_raises_corruption_error(self, checkpoint_manager, temp_checkpoint_dir):
        """Test loading empty checkpoint file raises CheckpointCorruptionError."""
        empty_id = "empty-checkpoint-id"
        empty_file = Path(temp_checkpoint_dir) / f"{empty_id}.lz4"
        
        # Create empty file
        empty_file.touch()
        
        with pytest.raises(CheckpointCorruptionError) as exc_info:
            checkpoint_manager.load_checkpoint(empty_id)
        
        assert "empty" in str(exc_info.value).lower()


class TestCheckpointManagerPerformance:
    """Test performance requirements."""

    def test_save_checkpoint_under_500ms(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test save_checkpoint completes within 500ms performance requirement."""
        start_time = time.monotonic()
        checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        duration = time.monotonic() - start_time
        
        assert duration < 0.5, f"Save took {duration:.3f}s, exceeds 500ms limit"

    def test_load_checkpoint_under_500ms(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test load_checkpoint completes within 500ms performance requirement."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        start_time = time.monotonic()
        checkpoint_manager.load_checkpoint(checkpoint_id)
        duration = time.monotonic() - start_time
        
        assert duration < 0.5, f"Load took {duration:.3f}s, exceeds 500ms limit"

    def test_large_game_state_performance(self, checkpoint_manager):
        """Test performance with large game states (simulating complex Pokemon data)."""
        # Create large game state similar to full Pokemon game state
        large_state = {
            "player_data": {
                "name": "RED",
                "position": {"x": 100, "y": 150, "map": "route_1"},
                "stats": {"level": 50, "money": 999999}
            },
            "pokemon_team": [
                {
                    "species": f"POKEMON_{i}",
                    "level": 50 + i,
                    "moves": [f"MOVE_{j}" for j in range(4)],
                    "stats": {"hp": 200 + i, "attack": 150 + i, "defense": 120 + i},
                    "experience": 125000 + i * 1000
                }
                for i in range(6)
            ],
            "pokemon_box": [
                {
                    "species": f"BOXED_POKEMON_{i}",
                    "level": 10 + (i % 20),
                    "moves": [f"MOVE_{j}" for j in range(4)]
                }
                for i in range(240)  # Full PC box
            ],
            "inventory": {f"ITEM_{i}": (i % 99) + 1 for i in range(100)},
            "game_flags": {f"FLAG_{i}": bool(i % 2) for i in range(1000)},
            "map_data": {
                "visited_locations": [f"LOCATION_{i}" for i in range(200)],
                "npc_interactions": {f"NPC_{i}": {"talked": True, "state": i % 10} for i in range(500)}
            }
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


class TestCheckpointManagerAtomicOperations:
    """Test atomic write operations."""

    @patch('os.fsync')
    def test_save_uses_atomic_writes(self, mock_fsync, checkpoint_manager, sample_game_state, sample_metadata):
        """Test save operation uses atomic writes with fsync."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        # Verify fsync was called (ensures data reaches disk)
        mock_fsync.assert_called_once()
        
        # Verify final file exists and temp file doesn't
        final_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.lz4"
        temp_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.tmp"
        
        assert final_file.exists()
        assert not temp_file.exists()

    def test_save_cleans_up_temp_file_on_failure(self, checkpoint_manager, temp_checkpoint_dir):
        """Test temporary file is cleaned up if save operation fails."""
        
        # Mock Path.rename to simulate failure after temp file creation
        original_rename = Path.rename
        
        def failing_rename(self, target):
            if str(self).endswith('.tmp'):
                raise OSError("Simulated rename failure")
            return original_rename(self, target)
        
        with patch.object(Path, 'rename', failing_rename):
            with pytest.raises(CheckpointError):
                checkpoint_manager.save_checkpoint({"test": "data"}, {})
        
        # Verify no temp files remain
        temp_files = list(Path(temp_checkpoint_dir).glob("*.tmp"))
        assert len(temp_files) == 0, f"Temp files not cleaned up: {temp_files}"


class TestCheckpointManagerUtilityMethods:
    """Test utility methods."""

    def test_checkpoint_exists_returns_true_for_existing(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test checkpoint_exists returns True for existing checkpoints."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        assert checkpoint_manager.checkpoint_exists(checkpoint_id) is True

    def test_checkpoint_exists_returns_false_for_nonexistent(self, checkpoint_manager):
        """Test checkpoint_exists returns False for non-existent checkpoints."""
        fake_id = "nonexistent-checkpoint-id"
        assert checkpoint_manager.checkpoint_exists(fake_id) is False

    def test_get_checkpoint_size_returns_correct_size(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test get_checkpoint_size returns correct file size."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        size = checkpoint_manager.get_checkpoint_size(checkpoint_id)
        
        # Verify size matches actual file
        checkpoint_file = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.lz4"
        actual_size = checkpoint_file.stat().st_size
        
        assert size == actual_size
        assert size > 0

    def test_get_checkpoint_size_raises_not_found_for_nonexistent(self, checkpoint_manager):
        """Test get_checkpoint_size raises CheckpointNotFoundError for non-existent checkpoints."""
        fake_id = "nonexistent-checkpoint-id"
        
        with pytest.raises(CheckpointNotFoundError):
            checkpoint_manager.get_checkpoint_size(fake_id)

    def test_get_performance_stats_tracks_operations(self, checkpoint_manager, sample_game_state, sample_metadata):
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


class TestCheckpointManagerUniqueIdentifiers:
    """Test UUID generation uniqueness."""

    def test_save_checkpoint_generates_unique_identifiers(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test multiple saves generate unique checkpoint identifiers."""
        ids = []
        
        for i in range(10):
            checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
            ids.append(checkpoint_id)
        
        # All IDs should be unique
        assert len(set(ids)) == len(ids), "Generated non-unique checkpoint IDs"

    def test_concurrent_saves_generate_unique_identifiers(self, checkpoint_manager, sample_game_state, sample_metadata):
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
        for i in range(5):
            thread = threading.Thread(target=save_checkpoint)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All IDs should be unique
        assert len(set(ids)) == len(ids), f"Generated non-unique IDs: {ids}"