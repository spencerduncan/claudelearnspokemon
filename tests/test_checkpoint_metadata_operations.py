"""
Test suite for CheckpointManager metadata operations.

Tests the newly implemented methods:
- update_checkpoint_metadata
- search_checkpoints
- enhanced get_checkpoint_metadata

Following TDD methodology and comprehensive testing patterns.
"""

import tempfile
import pytest

from claudelearnspokemon.checkpoint_manager import (
    CheckpointManager,
    CheckpointError,
    CheckpointNotFoundError,
)


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def checkpoint_manager(temp_storage_dir):
    """Create CheckpointManager instance for testing."""
    return CheckpointManager(
        temp_storage_dir,
        max_checkpoints=50,
        enable_metrics=True,
    )


@pytest.fixture
def sample_game_state():
    """Sample game state for testing."""
    return {
        "player": {
            "name": "RED",
            "position": {"x": 100, "y": 150},
            "level": 25,
        },
        "pokemon": [
            {
                "name": "PIKACHU",
                "level": 25,
                "health": 65,
                "moves": ["THUNDERBOLT", "QUICK_ATTACK"],
            },
        ],
        "inventory": {"pokeball": 10, "potion": 5},
        "flags": {"gym_badges": ["boulder"], "story_progress": "pewter_city"},
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "location": "pewter_city",
        "progress_percentage": 25.0,
        "strategy": "speed_run",
        "progress_markers": {
            "badges": 1,
            "pokemon_caught": 1,
        },
    }


class TestUpdateCheckpointMetadata:
    """Test update_checkpoint_metadata method."""

    def test_update_tags_successful(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test successful tags update."""
        # Create checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        # Update tags
        updates = {"tags": ["gym_battle", "important"]}
        result = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
        
        assert result is True
        
        # Verify update
        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert metadata is not None
        assert metadata["tags"] == ["gym_battle", "important"]

    def test_update_custom_fields_successful(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test successful custom fields update."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        updates = {"custom_fields": {"difficulty": "hard", "attempts": 3}}
        result = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
        
        assert result is True
        
        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert metadata["custom_fields"]["difficulty"] == "hard"
        assert metadata["custom_fields"]["attempts"] == 3

    def test_update_performance_metrics_successful(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test successful performance metrics update."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        updates = {"performance_metrics": {"score": 850, "time": 120.5}}
        result = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
        
        assert result is True
        
        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert metadata["performance_metrics"]["score"] == 850
        assert metadata["performance_metrics"]["time"] == 120.5

    def test_update_game_location_successful(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test successful game location update."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        updates = {"game_location": "cerulean_city"}
        result = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
        
        assert result is True
        
        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert metadata["game_location"] == "cerulean_city"
        assert metadata["location"] == "cerulean_city"  # Both fields should match

    def test_update_strategic_value_successful(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test successful strategic value update."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        updates = {"strategic_value": 0.75}
        result = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
        
        assert result is True
        
        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert metadata["strategic_value"] == 0.75

    def test_update_success_rate_successful(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test successful success rate update with clamping."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        # Test normal value
        updates = {"success_rate": 0.85}
        result = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
        
        assert result is True
        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert metadata["success_rate"] == 0.85
        
        # Test clamping - value too high
        updates = {"success_rate": 1.5}
        result = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
        
        assert result is True
        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert metadata["success_rate"] == 1.0
        
        # Test clamping - value too low
        updates = {"success_rate": -0.1}
        result = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
        
        assert result is True
        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert metadata["success_rate"] == 0.0

    def test_update_multiple_fields_successful(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test updating multiple fields simultaneously."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        updates = {
            "tags": ["multi_update"],
            "custom_fields": {"test": True},
            "game_location": "viridian_city",
            "strategic_value": 0.9
        }
        result = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
        
        assert result is True
        
        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        assert metadata["tags"] == ["multi_update"]
        assert metadata["custom_fields"]["test"] is True
        assert metadata["game_location"] == "viridian_city"
        assert metadata["strategic_value"] == 0.9

    def test_update_nonexistent_checkpoint_returns_false(self, checkpoint_manager):
        """Test updating nonexistent checkpoint returns False."""
        fake_id = "nonexistent-checkpoint-id"
        updates = {"tags": ["test"]}
        
        result = checkpoint_manager.update_checkpoint_metadata(fake_id, updates)
        assert result is False

    def test_update_invalid_checkpoint_id_raises_error(self, checkpoint_manager):
        """Test invalid checkpoint ID raises ValueError."""
        updates = {"tags": ["test"]}
        
        with pytest.raises(ValueError, match="checkpoint_id must be a non-empty string"):
            checkpoint_manager.update_checkpoint_metadata("", updates)
        
        with pytest.raises(ValueError, match="checkpoint_id must be a non-empty string"):
            checkpoint_manager.update_checkpoint_metadata(None, updates)

    def test_update_invalid_updates_raises_error(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test invalid updates parameter raises ValueError."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        with pytest.raises(ValueError, match="updates must be a non-empty dictionary"):
            checkpoint_manager.update_checkpoint_metadata(checkpoint_id, {})
        
        with pytest.raises(ValueError, match="updates must be a non-empty dictionary"):
            checkpoint_manager.update_checkpoint_metadata(checkpoint_id, None)
        
        with pytest.raises(ValueError, match="updates must be a non-empty dictionary"):
            checkpoint_manager.update_checkpoint_metadata(checkpoint_id, "invalid")

    def test_update_unsupported_field_logs_warning(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test unsupported field update logs warning but doesn't fail."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        updates = {"unsupported_field": "value"}
        result = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
        
        # Should return False as no valid updates were applied
        assert result is False

    def test_update_wrong_type_logs_warning(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test wrong type for supported field logs warning."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        updates = {"tags": "should_be_list"}  # Wrong type
        result = checkpoint_manager.update_checkpoint_metadata(checkpoint_id, updates)
        
        # Should return False as no valid updates were applied
        assert result is False


class TestSearchCheckpoints:
    """Test search_checkpoints method."""

    def test_search_empty_criteria_returns_all(self, checkpoint_manager, sample_game_state):
        """Test empty criteria returns all checkpoints."""
        # Create multiple checkpoints
        for i in range(3):
            checkpoint_manager.save_checkpoint(
                sample_game_state, 
                {"location": f"location_{i}"}
            )
        
        results = checkpoint_manager.search_checkpoints({})
        assert len(results) == 3

    def test_search_by_game_location_exact_match(self, checkpoint_manager, sample_game_state):
        """Test search by exact game location match."""
        # Create checkpoints with different locations
        locations = ["pallet_town", "viridian_city", "pallet_town", "pewter_city"]
        checkpoint_ids = []
        
        for location in locations:
            checkpoint_id = checkpoint_manager.save_checkpoint(
                sample_game_state,
                {"location": location}
            )
            checkpoint_ids.append(checkpoint_id)
        
        # Search for pallet_town
        results = checkpoint_manager.search_checkpoints({"game_location": "pallet_town"})
        
        assert len(results) == 2
        for result in results:
            assert result["location"] == "pallet_town"

    def test_search_by_tags_any_match(self, checkpoint_manager, sample_game_state):
        """Test search by tags with any match logic."""
        # Create checkpoints with different tags
        checkpoint_ids = []
        
        # Checkpoint 1: tags [gym, battle]
        id1 = checkpoint_manager.save_checkpoint(sample_game_state, {"location": "gym_1"})
        checkpoint_manager.update_checkpoint_metadata(id1, {"tags": ["gym", "battle"]})
        checkpoint_ids.append(id1)
        
        # Checkpoint 2: tags [tutorial]
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, {"location": "start"})
        checkpoint_manager.update_checkpoint_metadata(id2, {"tags": ["tutorial"]})
        checkpoint_ids.append(id2)
        
        # Checkpoint 3: tags [gym, important]
        id3 = checkpoint_manager.save_checkpoint(sample_game_state, {"location": "gym_2"})
        checkpoint_manager.update_checkpoint_metadata(id3, {"tags": ["gym", "important"]})
        checkpoint_ids.append(id3)
        
        # Search for checkpoints with "gym" tag
        results = checkpoint_manager.search_checkpoints({"tags": ["gym"]})
        
        assert len(results) == 2
        found_ids = [r["checkpoint_id"] for r in results]
        assert id1 in found_ids
        assert id3 in found_ids
        assert id2 not in found_ids

    def test_search_by_performance_min(self, checkpoint_manager, sample_game_state):
        """Test search by minimum performance score."""
        checkpoint_ids = []
        
        # Create checkpoints with different performance scores
        scores = [100, 250, 300, 150]
        for i, score in enumerate(scores):
            checkpoint_id = checkpoint_manager.save_checkpoint(
                sample_game_state,
                {"location": f"location_{i}"}
            )
            checkpoint_manager.update_checkpoint_metadata(
                checkpoint_id,
                {"performance_metrics": {"score": score}}
            )
            checkpoint_ids.append(checkpoint_id)
        
        # Search for scores >= 200
        results = checkpoint_manager.search_checkpoints({"performance_min": 200})
        
        assert len(results) == 2
        for result in results:
            performance = result.get("performance_metrics", {})
            assert performance.get("score", 0) >= 200

    def test_search_by_custom_fields(self, checkpoint_manager, sample_game_state):
        """Test search by custom fields."""
        checkpoint_ids = []
        
        # Create checkpoints with different custom fields
        id1 = checkpoint_manager.save_checkpoint(sample_game_state, {"location": "loc_1"})
        checkpoint_manager.update_checkpoint_metadata(id1, {"custom_fields": {"type": "boss", "difficulty": "hard"}})
        
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, {"location": "loc_2"})
        checkpoint_manager.update_checkpoint_metadata(id2, {"custom_fields": {"type": "regular", "difficulty": "easy"}})
        
        id3 = checkpoint_manager.save_checkpoint(sample_game_state, {"location": "loc_3"})
        checkpoint_manager.update_checkpoint_metadata(id3, {"custom_fields": {"type": "boss", "difficulty": "medium"}})
        
        # Search for boss battles
        results = checkpoint_manager.search_checkpoints({"custom_fields": {"type": "boss"}})
        
        assert len(results) == 2
        found_ids = [r["checkpoint_id"] for r in results]
        assert id1 in found_ids
        assert id3 in found_ids
        assert id2 not in found_ids

    def test_search_by_min_score(self, checkpoint_manager, sample_game_state):
        """Test search by minimum value score."""
        # Create checkpoints and access some to increase their value scores
        id1 = checkpoint_manager.save_checkpoint(sample_game_state, {"location": "high_value"})
        id2 = checkpoint_manager.save_checkpoint(sample_game_state, {"location": "low_value"})
        
        # Access first checkpoint multiple times to increase its value score
        for _ in range(5):
            checkpoint_manager.load_checkpoint(id1)
        
        # Search with minimum score that should filter out low-value checkpoints
        results = checkpoint_manager.search_checkpoints({"min_score": 0.1})
        
        # At least the high-value checkpoint should be included
        found_ids = [r["checkpoint_id"] for r in results]
        assert id1 in found_ids

    def test_search_multiple_criteria(self, checkpoint_manager, sample_game_state):
        """Test search with multiple criteria (AND logic)."""
        # Create test checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, {"location": "test_location"})
        
        # Add tags and performance metrics
        checkpoint_manager.update_checkpoint_metadata(checkpoint_id, {
            "tags": ["important", "boss"],
            "performance_metrics": {"score": 500}
        })
        
        # Search with multiple criteria
        results = checkpoint_manager.search_checkpoints({
            "game_location": "test_location",
            "tags": ["important"],
            "performance_min": 400
        })
        
        assert len(results) == 1
        assert results[0]["checkpoint_id"] == checkpoint_id

    def test_search_no_matches(self, checkpoint_manager, sample_game_state):
        """Test search that returns no matches."""
        # Create checkpoint
        checkpoint_manager.save_checkpoint(sample_game_state, {"location": "existing_location"})
        
        # Search for non-existent location
        results = checkpoint_manager.search_checkpoints({"game_location": "nonexistent_location"})
        
        assert len(results) == 0

    def test_search_results_sorted_by_value_score(self, checkpoint_manager, sample_game_state):
        """Test search results are sorted by value score (descending)."""
        # Create multiple checkpoints
        ids = []
        for i in range(3):
            checkpoint_id = checkpoint_manager.save_checkpoint(
                sample_game_state,
                {"location": "test_location"}
            )
            ids.append(checkpoint_id)
        
        # Access checkpoints different numbers of times to create different value scores
        for _ in range(5):
            checkpoint_manager.load_checkpoint(ids[2])  # Highest score
        
        for _ in range(2):
            checkpoint_manager.load_checkpoint(ids[1])  # Medium score
        
        # ids[0] not accessed - lowest score
        
        results = checkpoint_manager.search_checkpoints({"game_location": "test_location"})
        
        assert len(results) == 3
        # Results should be sorted by value_score (descending)
        scores = [r["value_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        
        # Most accessed should be first
        assert results[0]["checkpoint_id"] == ids[2]

    def test_search_invalid_criteria_raises_error(self, checkpoint_manager):
        """Test invalid criteria raises ValueError."""
        with pytest.raises(ValueError, match="criteria must be a dictionary"):
            checkpoint_manager.search_checkpoints("invalid")

    def test_search_includes_metadata_fields(self, checkpoint_manager, sample_game_state):
        """Test search results include all expected metadata fields."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, {"location": "test"})
        
        results = checkpoint_manager.search_checkpoints({})
        
        assert len(results) == 1
        result = results[0]
        
        # Check that all expected fields are present
        required_fields = [
            "checkpoint_id", "created_at", "location", "progress_markers",
            "value_score", "file_path"
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"


class TestGetCheckpointMetadataEnhancements:
    """Test enhancements to get_checkpoint_metadata method."""

    def test_get_metadata_includes_backward_compatibility_fields(self, checkpoint_manager, sample_game_state, sample_metadata):
        """Test get_checkpoint_metadata includes backward compatibility fields."""
        checkpoint_id = checkpoint_manager.save_checkpoint(sample_game_state, sample_metadata)
        
        # Add some metadata through updates
        checkpoint_manager.update_checkpoint_metadata(checkpoint_id, {
            "tags": ["test_tag"],
            "custom_fields": {"test_field": "test_value"},
            "performance_metrics": {"score": 100}
        })
        
        metadata = checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        
        assert metadata is not None
        
        # Check backward compatibility fields
        assert "game_location" in metadata
        assert metadata["game_location"] == sample_metadata["location"]
        
        assert "tags" in metadata
        assert metadata["tags"] == ["test_tag"]
        
        assert "custom_fields" in metadata
        assert metadata["custom_fields"]["test_field"] == "test_value"
        
        assert "performance_metrics" in metadata
        assert metadata["performance_metrics"]["score"] == 100

    def test_get_metadata_nonexistent_checkpoint_returns_none(self, checkpoint_manager):
        """Test get_checkpoint_metadata returns None for nonexistent checkpoint."""
        result = checkpoint_manager.get_checkpoint_metadata("nonexistent-id")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])