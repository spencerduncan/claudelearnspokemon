"""
Tests for PositionTracker component - Part of TileObserver Position Detection System.

Following TDD principles for Issue #28:
- Test position history tracking with timestamps
- Test movement pattern analysis
- Test NPC tracking across frames
- Test performance requirements
"""

# Import after path modification
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Note: This will be implemented as part of Issue #28
from claudelearnspokemon.position_tracker import (
    EntityPosition,
    Position,
    PositionTracker,
)


@pytest.mark.fast
class TestPositionDataStructures:
    """Test basic data structures for position tracking."""

    @pytest.mark.fast
    def test_position_creation(self):
        """Test Position dataclass creation."""
        pos = Position(x=10, y=5, timestamp=time.time())
        assert pos.x == 10
        assert pos.y == 5
        assert isinstance(pos.timestamp, float)

    @pytest.mark.fast
    def test_entity_position_creation(self):
        """Test EntityPosition dataclass creation with entity metadata."""
        entity_pos = EntityPosition(
            x=15,
            y=8,
            timestamp=time.time(),
            entity_type="player",
            entity_id="player_1",
            facing_direction="up",
            sprite_id=255,
        )
        assert entity_pos.x == 15
        assert entity_pos.y == 8
        assert entity_pos.entity_type == "player"
        assert entity_pos.facing_direction == "up"
        assert entity_pos.sprite_id == 255


@pytest.mark.fast
class TestPositionTracker:
    """Test PositionTracker functionality."""

    @pytest.mark.fast
    def test_position_tracker_initialization(self):
        """Test PositionTracker initializes correctly."""
        tracker = PositionTracker()
        assert tracker is not None

    @pytest.mark.fast
    def test_track_player_position_basic(self):
        """Test basic player position tracking."""
        tracker = PositionTracker()
        timestamp = time.time()

        tracker.track_player_position(x=10, y=5, timestamp=timestamp, facing_direction="up")

        current_pos = tracker.get_current_player_position()
        assert current_pos.x == 10
        assert current_pos.y == 5
        assert current_pos.entity_type == "player"
        assert current_pos.facing_direction == "up"

    @pytest.mark.fast
    def test_track_npc_position_basic(self):
        """Test basic NPC position tracking."""
        tracker = PositionTracker()
        timestamp = time.time()

        tracker.track_npc_position(
            npc_id="npc_trainer_1",
            x=8,
            y=12,
            timestamp=timestamp,
            npc_type="trainer",
            sprite_id=205,
        )

        npcs = tracker.get_current_npc_positions()
        assert len(npcs) == 1
        assert npcs[0].entity_id == "npc_trainer_1"
        assert npcs[0].x == 8
        assert npcs[0].y == 12
        assert npcs[0].entity_type == "trainer"

    @pytest.mark.fast
    def test_position_history_tracking(self):
        """Test that position history is maintained."""
        tracker = PositionTracker()
        base_time = time.time()

        # Track multiple positions
        tracker.track_player_position(10, 5, base_time, "up")
        tracker.track_player_position(10, 4, base_time + 1, "up")
        tracker.track_player_position(10, 3, base_time + 2, "up")

        history = tracker.get_player_position_history()
        assert len(history) == 3
        assert history[0].y == 5  # First position
        assert history[1].y == 4  # Second position
        assert history[2].y == 3  # Third position

    @pytest.mark.fast
    def test_movement_pattern_detection(self):
        """Test movement pattern detection."""
        tracker = PositionTracker()
        base_time = time.time()

        # Create clear movement pattern (moving up)
        positions = [(10, 10), (10, 9), (10, 8), (10, 7), (10, 6)]
        for i, (x, y) in enumerate(positions):
            tracker.track_player_position(x, y, base_time + i, "up")

        pattern = tracker.analyze_movement_pattern("player")
        assert pattern.direction == "up"
        assert pattern.speed > 0
        assert pattern.consistency > 0.8  # High consistency for straight line

    @pytest.mark.fast
    def test_npc_type_differentiation(self):
        """Test different NPC types are tracked separately."""
        tracker = PositionTracker()
        timestamp = time.time()

        tracker.track_npc_position("npc1", 5, 5, timestamp, "trainer", 205)
        tracker.track_npc_position("npc2", 8, 8, timestamp, "wild_pokemon", 210)
        tracker.track_npc_position("npc3", 12, 3, timestamp, "town_npc", 202)

        trainers = tracker.get_npcs_by_type("trainer")
        wild_pokemon = tracker.get_npcs_by_type("wild_pokemon")
        town_npcs = tracker.get_npcs_by_type("town_npc")

        assert len(trainers) == 1
        assert len(wild_pokemon) == 1
        assert len(town_npcs) == 1
        assert trainers[0].sprite_id == 205
        assert wild_pokemon[0].sprite_id == 210

    @pytest.mark.fast
    def test_position_prediction(self):
        """Test position prediction based on movement patterns."""
        tracker = PositionTracker()
        base_time = time.time()

        # Create predictable movement pattern
        tracker.track_player_position(0, 0, base_time, "right")
        tracker.track_player_position(1, 0, base_time + 1, "right")
        tracker.track_player_position(2, 0, base_time + 2, "right")

        predicted = tracker.predict_next_position("player", prediction_time=base_time + 3)
        assert predicted.x == 3  # Continuing pattern
        assert predicted.y == 0

    @pytest.mark.fast
    def test_clear_old_positions(self):
        """Test clearing old position data."""
        tracker = PositionTracker(max_history_size=5)
        base_time = time.time()

        # Add more positions than max history
        for i in range(10):
            tracker.track_player_position(i, 0, base_time + i, "right")

        history = tracker.get_player_position_history()
        assert len(history) <= 5  # Should be limited by max_history_size

    @pytest.mark.fast
    def test_performance_requirements(self):
        """Test that position tracking meets performance requirements."""
        tracker = PositionTracker()

        # Simulate realistic workload
        start_time = time.perf_counter()

        for i in range(100):  # 100 position updates
            tracker.track_player_position(i % 20, i % 18, time.time() + i, "up")
            tracker.track_npc_position(
                f"npc_{i}", i % 15, i % 12, time.time() + i, "trainer", 200 + (i % 20)
            )

        # Analysis operations
        tracker.get_current_player_position()
        tracker.get_current_npc_positions()
        tracker.analyze_movement_pattern("player")

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000

        # Should complete well within performance budget
        assert elapsed_ms < 25, f"Position tracking took {elapsed_ms:.2f}ms, should be < 25ms"


@pytest.mark.fast
class TestMovementPatternAnalysis:
    """Test movement pattern analysis capabilities."""

    @pytest.mark.fast
    def test_detect_straight_line_movement(self):
        """Test detection of straight-line movement patterns."""
        tracker = PositionTracker()
        base_time = time.time()

        # Straight line movement north
        positions = [(5, 10), (5, 9), (5, 8), (5, 7), (5, 6)]
        for i, (x, y) in enumerate(positions):
            tracker.track_player_position(x, y, base_time + i, "up")

        pattern = tracker.analyze_movement_pattern("player")
        assert pattern.direction == "up"
        assert pattern.pattern_type == "linear"

    @pytest.mark.fast
    def test_detect_circular_movement(self):
        """Test detection of circular movement patterns."""
        tracker = PositionTracker()
        base_time = time.time()

        # Circular movement pattern
        positions = [(5, 5), (6, 5), (6, 6), (5, 6), (5, 5)]
        for i, (x, y) in enumerate(positions):
            tracker.track_player_position(x, y, base_time + i, "right")

        pattern = tracker.analyze_movement_pattern("player")
        assert pattern.pattern_type == "circular"

    @pytest.mark.fast
    def test_detect_random_movement(self):
        """Test detection of random/erratic movement."""
        tracker = PositionTracker()
        base_time = time.time()

        # Random movement positions
        positions = [(1, 1), (8, 3), (2, 9), (15, 2), (7, 12)]
        for i, (x, y) in enumerate(positions):
            tracker.track_player_position(x, y, base_time + i, "up")

        pattern = tracker.analyze_movement_pattern("player")
        assert pattern.pattern_type == "random"
        assert pattern.consistency < 0.7  # Lower consistency for random movement


@pytest.mark.fast
class TestIntegration:
    """Test integration scenarios."""

    @pytest.mark.fast
    def test_multi_frame_tracking_scenario(self):
        """Test tracking multiple entities across multiple frames."""
        tracker = PositionTracker()

        # Simulate 10 frames of game data
        for frame in range(10):
            timestamp = time.time() + frame * 0.1

            # Player moving
            tracker.track_player_position(frame, 5, timestamp, "right")

            # Trainer NPC stationary
            tracker.track_npc_position("trainer_1", 10, 10, timestamp, "trainer", 205)

            # Wild Pokemon moving randomly
            wild_x = 5 + (frame % 3) - 1  # Slight random movement
            wild_y = 8 + (frame % 2)
            tracker.track_npc_position("wild_1", wild_x, wild_y, timestamp, "wild_pokemon", 210)

        # Verify tracking
        player_pattern = tracker.analyze_movement_pattern("player")
        trainer_pattern = tracker.analyze_movement_pattern("trainer_1")
        wild_pattern = tracker.analyze_movement_pattern("wild_1")

        assert player_pattern.direction == "right"
        assert trainer_pattern.pattern_type == "stationary"
        # Wild pokemon has constrained movement so might be classified as circular
        assert wild_pattern.pattern_type in ["random", "erratic", "circular"]

    @pytest.mark.fast
    def test_cross_map_position_tracking(self):
        """Test position tracking across different map areas."""
        tracker = PositionTracker()
        timestamp = time.time()

        # Track position in Route 1
        tracker.track_player_position(10, 5, timestamp, "up", map_id="route_1")

        # Simulate map transition
        tracker.track_player_position(10, 0, timestamp + 1, "up", map_id="pallet_town")

        # Check position history includes map context
        history = tracker.get_player_position_history()
        assert len(history) == 2
        assert history[0].map_id == "route_1"
        assert history[1].map_id == "pallet_town"
