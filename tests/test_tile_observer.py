"""
Tests for TileObserver component.

Following Uncle Bob's TDD principles:
- Red: Write failing tests first
- Green: Implement minimal code to pass
- Refactor: Clean up while keeping tests green

Test coverage based on design specification requirements:
- Capture 20x18 tile grids in <50ms
- Identify player and NPC positions
- Learn solid vs walkable tiles from collisions
- Detect repeating patterns in tile arrangements
- Handle menu overlay tiles appropriately
- Pattern detection in <100ms
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import after path modification
# ruff: noqa: E402
from claudelearnspokemon.tile_observer import (
    GamePosition,
    GameState,
    GameStateInterface,
    TileInfo,
    TileObserver,
)


@pytest.mark.fast
@pytest.mark.medium
class TestTileObserverBasicStructure:
    """Test TileObserver class initialization and basic structure."""

    @pytest.mark.fast
    def test_tile_observer_initializes_correctly(self) -> None:
        """Test that TileObserver initializes without errors."""
        observer = TileObserver()
        assert observer is not None

    @pytest.mark.fast
    def test_tile_observer_has_required_methods(self) -> None:
        """Test that TileObserver has all required interface methods."""
        observer = TileObserver()

        # Methods defined in design specification
        assert hasattr(observer, "capture_tiles")
        assert hasattr(observer, "analyze_tile_grid")
        assert hasattr(observer, "detect_patterns")
        assert hasattr(observer, "learn_tile_properties")
        assert hasattr(observer, "identify_npcs")
        assert hasattr(observer, "find_path")

    @pytest.mark.fast
    def test_tile_observer_initializes_with_empty_tile_knowledge(self) -> None:
        """Test that TileObserver starts with no learned tile properties."""
        observer = TileObserver()
        assert len(observer._tile_semantics) == 0
        assert len(observer._map_context) == 0


@pytest.mark.fast
@pytest.mark.medium
class TestTileCapture:
    """Test tile grid capture functionality."""

    @pytest.mark.fast
    def test_capture_tiles_returns_correct_dimensions(self) -> None:
        """Test that capture_tiles returns 20x18 numpy array."""
        observer = TileObserver()
        game_state = self._create_mock_game_state()

        tiles = observer.capture_tiles(game_state)

        assert isinstance(tiles, np.ndarray)
        assert tiles.shape == (20, 18), f"Expected (20, 18), got {tiles.shape}"

    @pytest.mark.fast
    def test_capture_tiles_performance_under_50ms(self) -> None:
        """Test that tile capture completes within 50ms performance requirement."""
        observer = TileObserver()
        game_state = self._create_mock_game_state()

        start_time = time.perf_counter()
        tiles = observer.capture_tiles(game_state)  # noqa: F841
        end_time = time.perf_counter()

        capture_time_ms = (end_time - start_time) * 1000
        assert capture_time_ms < 50, f"Capture took {capture_time_ms:.2f}ms, must be < 50ms"

    @pytest.mark.fast
    def test_capture_tiles_handles_invalid_game_state(self) -> None:
        """Test that capture_tiles handles invalid game state gracefully."""
        observer = TileObserver()

        with pytest.raises(ValueError, match="Invalid game state"):
            observer.capture_tiles({})

    @pytest.mark.fast
    def test_capture_tiles_extracts_correct_tile_data(self) -> None:
        """Test that capture_tiles extracts tile data correctly from game state."""
        observer = TileObserver()
        expected_tiles = np.random.randint(0, 256, (20, 18), dtype=np.uint8)
        game_state = {"tiles": expected_tiles.tolist()}

        result = observer.capture_tiles(game_state)

        np.testing.assert_array_equal(result, expected_tiles)

    def _create_mock_game_state(self) -> dict:
        """Create a mock game state for testing."""
        return {
            "tiles": np.random.randint(0, 256, (20, 18), dtype=np.uint8).tolist(),
            "player_position": (10, 9),
            "menu_active": False,
        }


@pytest.mark.fast
@pytest.mark.medium
class TestPositionDetection:
    """Test player and NPC position identification."""

    @pytest.mark.fast
    def test_identify_player_position_from_tiles(self) -> None:
        """Test that player position is correctly identified in tile grid."""
        observer = TileObserver()
        tiles = np.zeros((20, 18), dtype=np.uint8)
        player_tile_id = 255  # Assume 255 represents player
        tiles[10, 9] = player_tile_id

        result = observer.analyze_tile_grid(tiles)

        assert "player_position" in result
        assert result["player_position"] == (10, 9)

    @pytest.mark.fast
    def test_identify_npcs_returns_list_of_positions(self) -> None:
        """Test that NPC identification returns list of NPC positions."""
        observer = TileObserver()
        tiles = np.zeros((20, 18), dtype=np.uint8)
        npc_tile_id = 200
        # Place NPCs at known positions
        tiles[5, 5] = npc_tile_id
        tiles[15, 10] = npc_tile_id

        npcs = observer.identify_npcs(tiles)

        assert isinstance(npcs, list)
        assert len(npcs) == 2
        assert (5, 5) in npcs
        assert (15, 10) in npcs

    @pytest.mark.fast
    def test_identify_npcs_handles_no_npcs(self) -> None:
        """Test that NPC identification handles grids with no NPCs."""
        observer = TileObserver()
        tiles = np.zeros((20, 18), dtype=np.uint8)

        npcs = observer.identify_npcs(tiles)

        assert isinstance(npcs, list)
        assert len(npcs) == 0

    @pytest.mark.fast
    def test_analyze_tile_grid_identifies_entities(self) -> None:
        """Test that analyze_tile_grid identifies all entities correctly."""
        observer = TileObserver()
        tiles = np.zeros((20, 18), dtype=np.uint8)
        tiles[10, 9] = 255  # Player
        tiles[5, 5] = 200  # NPC

        analysis = observer.analyze_tile_grid(tiles)

        assert "player_position" in analysis
        assert "npcs" in analysis
        assert analysis["player_position"] == (10, 9)
        assert (5, 5) in analysis["npcs"]


@pytest.mark.fast
@pytest.mark.medium
class TestCollisionLearning:
    """Test learning tile semantics through collision detection."""

    @pytest.mark.fast
    def test_learn_tile_properties_from_collision_observations(self) -> None:
        """Test that tile properties are learned from collision data."""
        observer = TileObserver()
        observations = [
            {"tile_id": 50, "position": (5, 5), "collision": True, "context": "route_1"},
            {"tile_id": 51, "position": (5, 6), "collision": False, "context": "route_1"},
        ]

        observer.learn_tile_properties(observations)

        # Check that tile semantics were updated
        assert 50 in observer._tile_semantics["route_1"]
        assert 51 in observer._tile_semantics["route_1"]
        assert observer._tile_semantics["route_1"][50]["walkable"] is False
        assert observer._tile_semantics["route_1"][51]["walkable"] is True

    @pytest.mark.fast
    def test_learn_tile_properties_maintains_map_context(self) -> None:
        """Test that tile learning maintains separate contexts per map."""
        observer = TileObserver()
        observations_route1 = [{"tile_id": 50, "collision": True, "context": "route_1"}]
        observations_route2 = [{"tile_id": 50, "collision": False, "context": "route_2"}]

        observer.learn_tile_properties(observations_route1)
        observer.learn_tile_properties(observations_route2)

        assert observer._tile_semantics["route_1"][50]["walkable"] is False
        assert observer._tile_semantics["route_2"][50]["walkable"] is True

    @pytest.mark.fast
    def test_learn_tile_properties_updates_confidence_scores(self) -> None:
        """Test that repeated observations increase confidence in tile properties."""
        observer = TileObserver()
        # Multiple observations of the same tile being solid
        observations = [
            {"tile_id": 50, "collision": True, "context": "route_1"},
            {"tile_id": 50, "collision": True, "context": "route_1"},
            {"tile_id": 50, "collision": True, "context": "route_1"},
        ]

        observer.learn_tile_properties(observations)

        tile_properties = observer._tile_semantics["route_1"][50]
        assert tile_properties["confidence"] > 0.5
        assert tile_properties["observation_count"] == 3


@pytest.mark.fast
@pytest.mark.medium
class TestPatternDetection:
    """Test pattern detection in tile arrangements."""

    @pytest.mark.fast
    def test_detect_patterns_finds_exact_matches(self) -> None:
        """Test that detect_patterns finds exact pattern matches in tiles."""
        observer = TileObserver()
        tiles = np.zeros((20, 18), dtype=np.uint8)
        # Create a simple 2x2 pattern
        pattern = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        # Place pattern at known location
        tiles[5:7, 5:7] = pattern
        tiles[10:12, 10:12] = pattern  # Second occurrence

        matches = observer.detect_patterns(tiles, pattern)

        assert isinstance(matches, list)
        assert len(matches) == 2
        assert (5, 5) in matches
        assert (10, 10) in matches

    @pytest.mark.fast
    def test_detect_patterns_performance_under_100ms(self) -> None:
        """Test that pattern detection completes within 100ms requirement."""
        observer = TileObserver()
        tiles = np.random.randint(0, 10, (20, 18), dtype=np.uint8)
        pattern = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        start_time = time.perf_counter()
        matches = observer.detect_patterns(tiles, pattern)  # noqa: F841
        end_time = time.perf_counter()

        detection_time_ms = (end_time - start_time) * 1000
        assert (
            detection_time_ms < 100
        ), f"Pattern detection took {detection_time_ms:.2f}ms, must be < 100ms"

    @pytest.mark.fast
    def test_detect_patterns_handles_no_matches(self) -> None:
        """Test that pattern detection handles cases with no matches."""
        observer = TileObserver()
        tiles = np.zeros((20, 18), dtype=np.uint8)
        pattern = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        matches = observer.detect_patterns(tiles, pattern)

        assert isinstance(matches, list)
        assert len(matches) == 0

    @pytest.mark.fast
    def test_detect_patterns_handles_edge_cases(self) -> None:
        """Test that pattern detection handles patterns at grid edges."""
        observer = TileObserver()
        tiles = np.zeros((20, 18), dtype=np.uint8)
        pattern = np.array([[1, 2]], dtype=np.uint8)

        # Place pattern at edge
        tiles[0:1, 16:18] = pattern

        matches = observer.detect_patterns(tiles, pattern)

        assert len(matches) == 1
        assert (0, 16) in matches


@pytest.mark.fast
@pytest.mark.medium
class TestMenuHandling:
    """Test handling of menu overlay tiles."""

    @pytest.mark.fast
    def test_capture_tiles_filters_menu_overlays(self) -> None:
        """Test that menu overlay tiles are handled appropriately."""
        observer = TileObserver()
        game_state = {
            "tiles": np.random.randint(0, 256, (20, 18), dtype=np.uint8).tolist(),
            "menu_active": True,
            "menu_tiles": [(5, 5), (5, 6), (6, 5), (6, 6)],
        }

        tiles = observer.capture_tiles(game_state)

        # Menu overlay positions should be handled specially
        assert isinstance(tiles, np.ndarray)
        # Additional menu handling logic would be tested here

    @pytest.mark.fast
    def test_analyze_tile_grid_ignores_menu_areas(self) -> None:
        """Test that analysis ignores menu overlay areas when appropriate."""
        observer = TileObserver()
        tiles = np.ones((20, 18), dtype=np.uint8) * 100  # Background tiles
        # Simulate menu overlay in corner
        tiles[0:5, 0:8] = 254  # Menu tiles
        tiles[10, 9] = 255  # Player outside menu area

        analysis = observer.analyze_tile_grid(tiles, menu_areas=[(0, 0, 5, 8)])

        assert "player_position" in analysis
        assert analysis["player_position"] == (10, 9)


@pytest.mark.fast
@pytest.mark.medium
class TestPathfinding:
    """Test walkable path identification."""

    @pytest.mark.fast
    def test_find_path_uses_learned_tile_semantics(self) -> None:
        """Test that pathfinding uses learned walkable/solid tile properties."""
        observer = TileObserver()

        # Set up learned tile semantics
        observer._tile_semantics["test_map"] = {
            0: {"walkable": True, "confidence": 0.9},  # Walkable
            1: {"walkable": False, "confidence": 0.9},  # Solid
        }

        # Create simple grid with path
        tiles = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=np.uint8)

        path = observer.find_path(tiles, (0, 0), (2, 2), context="test_map")

        assert isinstance(path, list)
        assert len(path) > 0
        assert (0, 0) in path
        assert (2, 2) in path

    @pytest.mark.fast
    def test_find_path_returns_empty_for_blocked_destination(self) -> None:
        """Test that pathfinding returns empty path when destination is blocked."""
        observer = TileObserver()

        observer._tile_semantics["test_map"] = {
            0: {"walkable": True, "confidence": 0.9},
            1: {"walkable": False, "confidence": 0.9},
        }

        # Grid with no path to destination
        tiles = np.array(
            [[0, 1, 1], [1, 1, 1], [1, 1, 0]],  # Destination is walkable but isolated
            dtype=np.uint8,
        )

        path = observer.find_path(tiles, (0, 0), (2, 2), context="test_map")

        assert isinstance(path, list)
        assert len(path) == 0

    @pytest.mark.fast
    def test_find_path_handles_unknown_tiles(self) -> None:
        """Test that pathfinding handles tiles with unknown walkability."""
        observer = TileObserver()
        tiles = np.array([[99, 99], [99, 99]], dtype=np.uint8)  # Unknown tiles

        # Should use conservative approach for unknown tiles
        path = observer.find_path(tiles, (0, 0), (1, 1), context="unknown_map")

        assert isinstance(path, list)


@pytest.mark.fast
@pytest.mark.medium
class TestIntegrationScenarios:
    """Test integrated scenarios combining multiple TileObserver capabilities."""

    @pytest.mark.fast
    def test_complete_tile_analysis_workflow(self) -> None:
        """Test complete workflow: capture -> analyze -> learn -> pathfind."""
        observer = TileObserver()

        # 1. Capture tiles
        game_state = self._create_complex_game_state()
        tiles = observer.capture_tiles(game_state)

        # 2. Analyze grid
        analysis = observer.analyze_tile_grid(tiles)

        # 3. Learn from observations
        observations = [
            {"tile_id": 50, "collision": True, "context": "complex_map"},
            {"tile_id": 51, "collision": False, "context": "complex_map"},
        ]
        observer.learn_tile_properties(observations)

        # 4. Find path using learned knowledge
        start = analysis["player_position"]
        destination = (15, 15)
        path = observer.find_path(tiles, start, destination, context="complex_map")

        assert tiles.shape == (20, 18)
        assert "player_position" in analysis
        assert isinstance(path, list)

    @pytest.mark.fast
    def test_performance_under_load(self) -> None:
        """Test that all operations maintain performance under realistic load."""
        observer = TileObserver()

        # Simulate multiple rapid captures and analyses
        for _ in range(10):
            game_state = self._create_complex_game_state()

            start_time = time.perf_counter()
            tiles = observer.capture_tiles(game_state)
            analysis = observer.analyze_tile_grid(tiles)  # noqa: F841
            pattern = np.array([[1, 2], [3, 4]], dtype=np.uint8)
            matches = observer.detect_patterns(tiles, pattern)  # noqa: F841
            end_time = time.perf_counter()

            total_time_ms = (end_time - start_time) * 1000
            # Total cycle should be fast for real-time gameplay
            assert total_time_ms < 150, f"Full cycle took {total_time_ms:.2f}ms"

    def _create_complex_game_state(self) -> dict:
        """Create a complex game state for integration testing."""
        tiles = np.random.randint(0, 100, (20, 18), dtype=np.uint8)
        tiles[10, 9] = 255  # Player
        tiles[5, 5] = 200  # NPC
        tiles[15, 10] = 200  # Another NPC

        return {
            "tiles": tiles.tolist(),
            "player_position": (10, 9),
            "menu_active": False,
            "context": "complex_map",
        }


@pytest.mark.fast
@pytest.mark.medium
class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    @pytest.mark.fast
    def test_handles_empty_tile_grids(self) -> None:
        """Test handling of empty or invalid tile grids."""
        observer = TileObserver()

        with pytest.raises(ValueError):
            observer.analyze_tile_grid(np.array([]))

    @pytest.mark.fast
    def test_handles_wrong_sized_grids(self) -> None:
        """Test handling of incorrectly sized tile grids."""
        observer = TileObserver()
        wrong_sized_tiles = np.zeros((10, 10), dtype=np.uint8)

        # Should handle gracefully or raise informative error
        with pytest.raises(ValueError, match="Expected grid shape"):
            observer.analyze_tile_grid(wrong_sized_tiles)

    @pytest.mark.fast
    def test_handles_concurrent_learning(self) -> None:
        """Test that tile learning handles concurrent observations safely."""
        observer = TileObserver()

        # This would be more relevant with actual threading,
        # but tests basic concurrent observation handling
        observations1 = [{"tile_id": 50, "collision": True, "context": "route_1"}]
        observations2 = [{"tile_id": 50, "collision": False, "context": "route_1"}]

        observer.learn_tile_properties(observations1)
        observer.learn_tile_properties(observations2)

        # Should have combined the observations appropriately
        tile_props = observer._tile_semantics["route_1"][50]
        assert tile_props["observation_count"] == 2


class TestGameStateInterface:
    """Test GameState interface classes and functionality."""

    @pytest.mark.fast
    def test_gamestate_interface_initializes(self):
        """Test GameStateInterface initializes correctly."""
        interface = GameStateInterface()
        assert interface is not None

    @pytest.mark.fast
    def test_gameposition_dataclass(self):
        """Test GamePosition dataclass creation."""
        pos = GamePosition(x=10, y=5, map_id="route_1", facing_direction="up")
        assert pos.x == 10
        assert pos.y == 5
        assert pos.map_id == "route_1"
        assert pos.facing_direction == "up"

    @pytest.mark.fast
    def test_gamestate_dataclass(self):
        """Test GameState dataclass creation."""
        position = GamePosition(0, 0, "test", "down")
        tiles = np.zeros((20, 18), dtype=np.uint8)
        state = GameState(
            position=position,
            tiles=tiles,
            npcs=[],
            menu_state=None,
            inventory={},
            timestamp=time.time(),
            frame_count=0,
        )
        assert state.position == position
        assert state.tiles.shape == (20, 18)

    @pytest.mark.fast
    def test_tileinfo_dataclass(self):
        """Test TileInfo dataclass creation."""
        tile_info = TileInfo(
            tile_id=42, passable=True, interaction_type="npc", semantic_tags={"walkable", "grass"}
        )
        assert tile_info.tile_id == 42
        assert tile_info.passable is True
        assert tile_info.interaction_type == "npc"
        assert "walkable" in tile_info.semantic_tags

    @pytest.mark.fast
    def test_extract_tile_grid_success(self):
        """Test successful tile grid extraction."""
        interface = GameStateInterface()
        position = GamePosition(0, 0, "test", "down")
        tiles = np.ones((20, 18), dtype=np.uint8) * 42
        state = GameState(position, tiles, [], None, {}, time.time(), 0)

        result = interface.extract_tile_grid(state)
        assert result.shape == (20, 18)
        assert np.all(result == 42)

    @pytest.mark.fast
    def test_extract_tile_grid_invalid_state(self):
        """Test tile grid extraction with invalid state."""
        interface = GameStateInterface()

        with pytest.raises(ValueError, match="Invalid GameState object"):
            interface.extract_tile_grid("not a gamestate")

    @pytest.mark.fast
    def test_get_player_position_success(self):
        """Test successful player position retrieval."""
        interface = GameStateInterface()
        position = GamePosition(15, 8, "route_2", "right")
        tiles = np.zeros((20, 18), dtype=np.uint8)
        state = GameState(position, tiles, [], None, {}, time.time(), 0)

        result = interface.get_player_position(state)
        assert result.x == 15
        assert result.y == 8
        assert result.map_id == "route_2"
        assert result.facing_direction == "right"

    @pytest.mark.fast
    def test_serialize_state_success(self):
        """Test successful state serialization."""
        interface = GameStateInterface()
        position = GamePosition(5, 10, "cerulean", "left")
        tiles = np.random.randint(0, 256, (20, 18), dtype=np.uint8)
        inventory = {"pokeball": 5, "potion": 3}
        state = GameState(position, tiles, [], None, inventory, time.time(), 100)

        serialized = interface.serialize_state(state)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    @pytest.mark.fast
    def test_capture_current_state_mock_client(self):
        """Test state capture with mock emulator client."""
        interface = GameStateInterface()

        # Create mock client
        class MockClient:
            def get_state(self):
                return {
                    "tiles": np.ones((20, 18), dtype=np.uint8) * 123,
                    "player_position": (7, 3),
                    "map_id": "viridian_forest",
                    "facing_direction": "up",
                    "npcs": [{"id": 1, "x": 5, "y": 8}],
                    "inventory": {"pokeball": 2},
                    "frame_count": 500,
                }

        mock_client = MockClient()
        result = interface.capture_current_state(mock_client)

        assert isinstance(result, GameState)
        assert result.position.x == 7
        assert result.position.y == 3
        assert result.position.map_id == "viridian_forest"
        assert result.position.facing_direction == "up"
        assert result.tiles.shape == (20, 18)
        assert np.all(result.tiles == 123)
        assert result.frame_count == 500
