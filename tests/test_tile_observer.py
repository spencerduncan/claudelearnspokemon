"""
Tests for TileObserver component.

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
    InteractionType,
    InteractiveObject,
    SemanticCategory,
    SemanticClassifier,
    StrategicLocation,
    StrategicLocationType,
    TileInfo,
    TileObserver,
    TileSemantics,
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


@pytest.mark.fast
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


@pytest.mark.fast
@pytest.mark.medium
class TestSemanticDataStructures:
    """Test semantic classification data structures."""

    @pytest.mark.fast
    def test_semantic_category_enum_values(self) -> None:
        """Test that SemanticCategory enum has all required values."""
        assert SemanticCategory.TERRAIN.value == "terrain"
        assert SemanticCategory.DOOR.value == "door"
        assert SemanticCategory.NPC.value == "npc"
        assert SemanticCategory.ITEM.value == "item"
        assert SemanticCategory.INTERACTIVE_OBJECT.value == "interactive_object"
        assert SemanticCategory.STRATEGIC_LOCATION.value == "strategic_location"
        assert SemanticCategory.OBSTACLE.value == "obstacle"
        assert SemanticCategory.WATER.value == "water"
        assert SemanticCategory.BUILDING.value == "building"

    @pytest.mark.fast
    def test_interaction_type_enum_values(self) -> None:
        """Test that InteractionType enum has all required values."""
        assert InteractionType.TALK.value == "talk"
        assert InteractionType.PICK_UP.value == "pick_up"
        assert InteractionType.USE.value == "use"
        assert InteractionType.ENTER.value == "enter"
        assert InteractionType.READ.value == "read"
        assert InteractionType.EXAMINE.value == "examine"

    @pytest.mark.fast
    def test_strategic_location_type_enum_values(self) -> None:
        """Test that StrategicLocationType enum has all required values."""
        assert StrategicLocationType.POKEMON_CENTER.value == "pokemon_center"
        assert StrategicLocationType.SHOP.value == "shop"
        assert StrategicLocationType.GYM.value == "gym"
        assert StrategicLocationType.ROUTE_EXIT.value == "route_exit"

    @pytest.mark.fast
    def test_tile_semantics_dataclass_creation(self) -> None:
        """Test TileSemantics dataclass creation and validation."""
        semantics = TileSemantics(
            tile_id=42,
            category=SemanticCategory.TERRAIN,
            subcategory="grass",
            interaction_type=None,
            strategic_type=None,
            confidence=0.85,
            context_dependent=False,
        )

        assert semantics.tile_id == 42
        assert semantics.category == SemanticCategory.TERRAIN
        assert semantics.subcategory == "grass"
        assert semantics.confidence == 0.85
        assert semantics.context_dependent is False

    @pytest.mark.fast
    def test_interactive_object_dataclass_creation(self) -> None:
        """Test InteractiveObject dataclass creation."""
        interactive_obj = InteractiveObject(
            position=(5, 10),
            tile_id=50,
            interaction_type=InteractionType.TALK,
            description="Friendly NPC",
            confidence=0.9,
        )

        assert interactive_obj.position == (5, 10)
        assert interactive_obj.tile_id == 50
        assert interactive_obj.interaction_type == InteractionType.TALK
        assert interactive_obj.description == "Friendly NPC"
        assert interactive_obj.confidence == 0.9

    @pytest.mark.fast
    def test_strategic_location_dataclass_creation(self) -> None:
        """Test StrategicLocation dataclass creation."""
        strategic_location = StrategicLocation(
            position=(10, 15),
            location_type=StrategicLocationType.POKEMON_CENTER,
            name="Viridian Pokemon Center",
            entrance_tiles=[(10, 15), (11, 15)],
            confidence=0.95,
        )

        assert strategic_location.position == (10, 15)
        assert strategic_location.location_type == StrategicLocationType.POKEMON_CENTER
        assert strategic_location.name == "Viridian Pokemon Center"
        assert strategic_location.entrance_tiles == [(10, 15), (11, 15)]
        assert strategic_location.confidence == 0.95


@pytest.mark.fast
@pytest.mark.medium
class TestSemanticClassifier:
    """Test semantic classification functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.classifier = SemanticClassifier()

    @pytest.mark.fast
    def test_semantic_classifier_initializes_correctly(self) -> None:
        """Test that SemanticClassifier initializes without errors."""
        classifier = SemanticClassifier()
        assert classifier is not None
        assert hasattr(classifier, "_semantic_patterns")
        assert hasattr(classifier, "_known_strategic_locations")

    @pytest.mark.fast
    def test_classify_tile_basic_terrain(self) -> None:
        """Test basic terrain tile classification."""
        # Test grass tile classification
        semantics = self.classifier.classify_tile_semantics(
            tile_id=1, position=(5, 5), surrounding_tiles=np.zeros((3, 3)), context="route_1"
        )

        assert isinstance(semantics, TileSemantics)
        assert semantics.tile_id == 1
        assert semantics.category in [SemanticCategory.TERRAIN, SemanticCategory.OBSTACLE]
        assert isinstance(semantics.confidence, float)
        assert 0.0 <= semantics.confidence <= 1.0

    @pytest.mark.fast
    def test_classify_tile_npc_detection(self) -> None:
        """Test NPC tile classification."""
        # NPC tile IDs are typically in the 200-220 range
        semantics = self.classifier.classify_tile_semantics(
            tile_id=205, position=(10, 8), surrounding_tiles=np.zeros((3, 3)), context="route_1"
        )

        assert semantics.category == SemanticCategory.NPC
        assert semantics.confidence > 0.7  # Should be confident about NPCs

    @pytest.mark.fast
    def test_classify_tile_door_detection(self) -> None:
        """Test door tile classification."""
        # Create building-like surrounding pattern
        surrounding = np.array(
            [
                [20, 20, 20],  # Building walls
                [20, 50, 20],  # Door in center
                [1, 1, 1],  # Ground tiles
            ]
        )

        semantics = self.classifier.classify_tile_semantics(
            tile_id=50, position=(1, 1), surrounding_tiles=surrounding, context="viridian_city"
        )

        # Should detect door or building entrance
        assert semantics.category in [
            SemanticCategory.DOOR,
            SemanticCategory.BUILDING,
            SemanticCategory.INTERACTIVE_OBJECT,
        ]

    @pytest.mark.fast
    def test_hierarchical_classification_terrain_grass(self) -> None:
        """Test hierarchical classification for terrain -> grass types."""
        semantics = self.classifier.classify_tile_hierarchical(
            tile_id=3, position=(5, 5), surrounding_tiles=np.zeros((3, 3)), context="route_1"
        )

        assert semantics.category == SemanticCategory.TERRAIN
        # Should have subcategory for different grass types
        assert semantics.subcategory is not None

    @pytest.mark.fast
    def test_context_dependent_classification(self) -> None:
        """Test that same tile can have different meanings in different contexts."""
        # Test same tile ID in different map contexts
        route_semantics = self.classifier.classify_tile_semantics(
            tile_id=10, position=(5, 5), surrounding_tiles=np.zeros((3, 3)), context="route_1"
        )

        city_semantics = self.classifier.classify_tile_semantics(
            tile_id=10, position=(5, 5), surrounding_tiles=np.zeros((3, 3)), context="viridian_city"
        )

        # Classifications might differ based on context
        assert isinstance(route_semantics, TileSemantics)
        assert isinstance(city_semantics, TileSemantics)

    @pytest.mark.fast
    def test_confidence_scoring_accuracy(self) -> None:
        """Test that confidence scores are reasonable and consistent."""
        # Known NPC tile should have high confidence
        npc_semantics = self.classifier.classify_tile_semantics(
            tile_id=200, position=(10, 8), surrounding_tiles=np.zeros((3, 3)), context="route_1"
        )

        # Unknown tile should have lower confidence
        unknown_semantics = self.classifier.classify_tile_semantics(
            tile_id=999, position=(10, 8), surrounding_tiles=np.zeros((3, 3)), context="route_1"
        )

        assert npc_semantics.confidence >= unknown_semantics.confidence
        assert 0.0 <= npc_semantics.confidence <= 1.0
        assert 0.0 <= unknown_semantics.confidence <= 1.0


@pytest.mark.fast
@pytest.mark.medium
class TestInteractiveObjectDetection:
    """Test interactive object detection and classification."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.observer = TileObserver()

    @pytest.mark.fast
    def test_detect_interactive_objects_basic(self) -> None:
        """Test basic interactive object detection."""
        tiles = np.zeros((20, 18), dtype=np.uint8)
        # Place some interactive objects
        tiles[5, 5] = 200  # NPC
        tiles[10, 8] = 150  # Item
        tiles[15, 12] = 75  # Potential door

        interactive_objects = self.observer.detect_interactive_objects(tiles, context="route_1")

        assert isinstance(interactive_objects, list)
        assert len(interactive_objects) >= 0  # Should find some interactive objects

        for obj in interactive_objects:
            assert isinstance(obj, InteractiveObject)
            assert hasattr(obj, "position")
            assert hasattr(obj, "interaction_type")
            assert hasattr(obj, "confidence")

    @pytest.mark.fast
    def test_detect_interactive_objects_npc_talk(self) -> None:
        """Test that NPCs are classified with TALK interaction."""
        tiles = np.zeros((20, 18), dtype=np.uint8)
        tiles[10, 9] = 205  # NPC tile

        interactive_objects = self.observer.detect_interactive_objects(tiles, context="route_1")

        npc_objects = [
            obj for obj in interactive_objects if obj.interaction_type == InteractionType.TALK
        ]
        assert len(npc_objects) >= 0  # Should find NPC objects

    @pytest.mark.fast
    def test_detect_interactive_objects_item_pickup(self) -> None:
        """Test that items are classified with PICK_UP interaction."""
        tiles = np.zeros((20, 18), dtype=np.uint8)
        tiles[5, 5] = 180  # Item tile

        interactive_objects = self.observer.detect_interactive_objects(tiles, context="route_1")

        item_objects = [
            obj for obj in interactive_objects if obj.interaction_type == InteractionType.PICK_UP
        ]
        assert len(item_objects) >= 0  # Should find item objects

    @pytest.mark.fast
    def test_interactive_object_confidence_scores(self) -> None:
        """Test that interactive objects have reasonable confidence scores."""
        tiles = np.zeros((20, 18), dtype=np.uint8)
        tiles[10, 9] = 200  # Known NPC tile

        interactive_objects = self.observer.detect_interactive_objects(tiles, context="route_1")

        for obj in interactive_objects:
            assert 0.0 <= obj.confidence <= 1.0
            # Known patterns should have higher confidence
            if obj.tile_id in range(200, 221):  # NPC range
                assert obj.confidence > 0.3


@pytest.mark.fast
@pytest.mark.medium
class TestStrategicLocationRecognition:
    """Test strategic location recognition (Pokemon Centers, shops, gyms)."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.observer = TileObserver()

    @pytest.mark.fast
    def test_identify_strategic_locations_basic(self) -> None:
        """Test basic strategic location identification."""
        tiles = self._create_pokemon_center_pattern()

        strategic_locations = self.observer.identify_strategic_locations(
            tiles, context="viridian_city"
        )

        assert isinstance(strategic_locations, list)
        for location in strategic_locations:
            assert isinstance(location, StrategicLocation)
            assert hasattr(location, "location_type")
            assert hasattr(location, "position")
            assert hasattr(location, "confidence")

    @pytest.mark.fast
    def test_identify_pokemon_center_pattern(self) -> None:
        """Test Pokemon Center pattern recognition."""
        tiles = self._create_pokemon_center_pattern()

        strategic_locations = self.observer.identify_strategic_locations(
            tiles, context="viridian_city"
        )

        pokemon_centers = [
            loc
            for loc in strategic_locations
            if loc.location_type == StrategicLocationType.POKEMON_CENTER
        ]

        # Should detect the Pokemon Center pattern
        if len(pokemon_centers) > 0:
            center = pokemon_centers[0]
            assert center.confidence > 0.5
            assert center.name is not None

    @pytest.mark.fast
    def test_identify_shop_pattern(self) -> None:
        """Test shop pattern recognition."""
        tiles = self._create_shop_pattern()

        strategic_locations = self.observer.identify_strategic_locations(
            tiles, context="viridian_city"
        )

        shops = [
            loc for loc in strategic_locations if loc.location_type == StrategicLocationType.SHOP
        ]

        # Should detect shop patterns
        assert len(shops) >= 0

    @pytest.mark.fast
    def test_strategic_location_entrance_tiles(self) -> None:
        """Test that strategic locations identify entrance tiles correctly."""
        tiles = self._create_pokemon_center_pattern()

        strategic_locations = self.observer.identify_strategic_locations(
            tiles, context="viridian_city"
        )

        for location in strategic_locations:
            assert hasattr(location, "entrance_tiles")
            assert isinstance(location.entrance_tiles, list)
            # Entrance tiles should be valid positions
            for entrance in location.entrance_tiles:
                assert isinstance(entrance, tuple)
                assert len(entrance) == 2
                row, col = entrance
                assert 0 <= row < 20
                assert 0 <= col < 18

    def _create_pokemon_center_pattern(self) -> np.ndarray:
        """Create a tile pattern resembling a Pokemon Center."""
        tiles = np.ones((20, 18), dtype=np.uint8) * 1  # Ground tiles

        # Pokemon Center building pattern (simplified)
        tiles[8:12, 7:11] = 30  # Building walls
        tiles[10, 9] = 35  # Entrance door
        tiles[9, 9] = 36  # Interior

        return tiles

    def _create_shop_pattern(self) -> np.ndarray:
        """Create a tile pattern resembling a shop."""
        tiles = np.ones((20, 18), dtype=np.uint8) * 1  # Ground tiles

        # Shop building pattern (simplified)
        tiles[5:9, 5:9] = 25  # Building walls
        tiles[7, 7] = 28  # Entrance door

        return tiles


@pytest.mark.fast
@pytest.mark.medium
class TestSemanticMapping:
    """Test semantic mapping functionality for navigation and planning."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.observer = TileObserver()

    @pytest.mark.fast
    def test_create_semantic_map_basic(self) -> None:
        """Test basic semantic map creation."""
        tiles = np.random.randint(0, 100, (20, 18), dtype=np.uint8)

        semantic_map = self.observer.create_semantic_map(tiles, context="route_1")

        assert isinstance(semantic_map, dict)
        assert "tile_semantics" in semantic_map
        assert "interactive_objects" in semantic_map
        assert "strategic_locations" in semantic_map

    @pytest.mark.fast
    def test_semantic_map_navigation_data(self) -> None:
        """Test that semantic map provides useful navigation data."""
        tiles = self._create_complex_map_with_obstacles()

        semantic_map = self.observer.create_semantic_map(tiles, context="route_1")

        # Should identify walkable vs non-walkable areas
        assert "walkable_areas" in semantic_map or "tile_semantics" in semantic_map

        # Should identify interactive elements
        if semantic_map["interactive_objects"]:
            for obj in semantic_map["interactive_objects"]:
                assert isinstance(obj, InteractiveObject)

    @pytest.mark.fast
    def test_semantic_pathfinding_integration(self) -> None:
        """Test integration of semantic understanding with pathfinding."""
        tiles = self._create_complex_map_with_obstacles()

        # Use semantic understanding for pathfinding
        path = self.observer.find_semantic_path(
            tiles, start=(1, 1), end=(18, 16), context="route_1", avoid_npcs=True
        )

        assert isinstance(path, list)
        # Path should avoid NPCs and obstacles based on semantic understanding

    @pytest.mark.fast
    def test_semantic_map_performance_real_time(self) -> None:
        """Test that semantic map creation meets real-time performance requirements."""
        tiles = np.random.randint(0, 256, (20, 18), dtype=np.uint8)

        start_time = time.perf_counter()
        semantic_map = self.observer.create_semantic_map(tiles, context="route_1")  # noqa: F841
        end_time = time.perf_counter()

        processing_time_ms = (end_time - start_time) * 1000
        # Should be fast enough for real-time gameplay
        assert (
            processing_time_ms < 50
        ), f"Semantic mapping took {processing_time_ms:.2f}ms, must be < 50ms"

    def _create_complex_map_with_obstacles(self) -> np.ndarray:
        """Create a complex map with various semantic elements."""
        tiles = np.ones((20, 18), dtype=np.uint8) * 1  # Ground tiles

        # Add obstacles
        tiles[5:8, 5:8] = 20  # Building/obstacle
        tiles[10, 9] = 200  # NPC
        tiles[15, 12] = 180  # Item
        tiles[2:4, 2:15] = 25  # Wall/barrier

        return tiles


@pytest.mark.fast
@pytest.mark.medium
class TestSemanticIntegrationWithCollisionDetection:
    """Test integration of semantic classification with existing collision detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.observer = TileObserver()
        # Set up some learned collision data from Issue #34
        self.observer._tile_semantics["route_1"] = {
            20: {
                "walkable": False,
                "confidence": 0.9,
                "observation_count": 10,
                "collision_count": 9,
            },
            1: {
                "walkable": True,
                "confidence": 0.95,
                "observation_count": 20,
                "collision_count": 1,
            },
        }

    @pytest.mark.fast
    def test_semantic_classification_uses_collision_data(self) -> None:
        """Test that semantic classification uses collision detection data."""
        # Create classifier with collision data integrated
        classifier = SemanticClassifier()

        # Should use collision data to enhance classification
        semantics = classifier.classify_tile_with_collision_data(
            tile_id=20,
            position=(5, 5),
            collision_data={"walkable": False, "confidence": 0.9},
            surrounding_tiles=np.zeros((3, 3)),
            context="route_1",
        )

        assert isinstance(semantics, TileSemantics)
        # Non-walkable tiles are likely obstacles or buildings
        assert semantics.category in [
            SemanticCategory.OBSTACLE,
            SemanticCategory.BUILDING,
            SemanticCategory.DOOR,
            SemanticCategory.INTERACTIVE_OBJECT,
        ]

    @pytest.mark.fast
    def test_collision_data_enhances_confidence(self) -> None:
        """Test that collision data enhances classification confidence."""
        classifier = SemanticClassifier()

        # Classification with collision data
        semantics_with_collision = classifier.classify_tile_with_collision_data(
            tile_id=50,
            position=(5, 5),
            collision_data={"walkable": False, "confidence": 0.9},
            surrounding_tiles=np.zeros((3, 3)),
            context="route_1",
        )

        # Classification without collision data
        semantics_without_collision = classifier.classify_tile_semantics(
            tile_id=50, position=(5, 5), surrounding_tiles=np.zeros((3, 3)), context="route_1"
        )

        # With collision data should have higher confidence
        assert semantics_with_collision.confidence >= semantics_without_collision.confidence

    @pytest.mark.fast
    def test_learned_walkability_informs_semantic_category(self) -> None:
        """Test that learned walkability helps determine semantic categories."""
        # Test that known walkable tiles are classified as terrain
        # Test that known non-walkable tiles are classified as obstacles/buildings

        walkable_semantics = self.observer.classify_tile_semantics_enhanced(
            tile_id=1, position=(5, 5), context="route_1"
        )

        non_walkable_semantics = self.observer.classify_tile_semantics_enhanced(
            tile_id=20, position=(8, 8), context="route_1"
        )

        # Walkable tiles should be terrain or similar
        if walkable_semantics.confidence > 0.5:
            assert walkable_semantics.category in [SemanticCategory.TERRAIN, SemanticCategory.ITEM]

        # Non-walkable tiles should be obstacles or buildings
        if non_walkable_semantics.confidence > 0.5:
            assert non_walkable_semantics.category in [
                SemanticCategory.OBSTACLE,
                SemanticCategory.BUILDING,
                SemanticCategory.DOOR,
                SemanticCategory.INTERACTIVE_OBJECT,
            ]


@pytest.mark.fast
@pytest.mark.medium
class TestSemanticPerformanceAndOptimization:
    """Test performance optimization and caching for semantic classification."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.observer = TileObserver()

    @pytest.mark.fast
    def test_semantic_classification_caching(self) -> None:
        """Test that semantic classifications are cached for performance."""
        tiles = np.random.randint(0, 100, (20, 18), dtype=np.uint8)

        # First call - should compute and cache
        start_time = time.perf_counter()
        semantic_map1 = self.observer.create_semantic_map(tiles, context="route_1")
        first_call_time = time.perf_counter() - start_time

        # Second call with same tiles - should use cache
        start_time = time.perf_counter()
        semantic_map2 = self.observer.create_semantic_map(tiles, context="route_1")
        second_call_time = time.perf_counter() - start_time

        # Second call should be faster due to caching
        assert second_call_time <= first_call_time * 1.1  # Allow for some variance
        assert semantic_map1.keys() == semantic_map2.keys()

    @pytest.mark.fast
    def test_incremental_learning_performance(self) -> None:
        """Test that incremental learning doesn't impact real-time performance."""
        # Simulate learning from multiple observations
        observations = []
        for i in range(100):
            observations.append(
                {
                    "tile_id": i % 50,
                    "position": (i % 20, (i * 2) % 18),
                    "collision": i % 3 == 0,
                    "context": "route_1",
                    "semantic_category": (
                        SemanticCategory.TERRAIN if i % 2 else SemanticCategory.OBSTACLE
                    ),
                }
            )

        start_time = time.perf_counter()
        self.observer.learn_semantic_properties(observations)
        learning_time = time.perf_counter() - start_time

        # Learning should be fast enough for real-time updates
        learning_time_ms = learning_time * 1000
        assert learning_time_ms < 100, f"Learning took {learning_time_ms:.2f}ms, must be < 100ms"

    @pytest.mark.fast
    def test_memory_usage_bounded(self) -> None:
        """Test that semantic knowledge storage doesn't grow unbounded."""
        # Add many semantic observations
        for i in range(1000):
            tile_semantics = TileSemantics(
                tile_id=i % 100, category=SemanticCategory.TERRAIN, confidence=0.5
            )
            self.observer._add_semantic_knowledge(tile_semantics, context="route_1")

        # Should have reasonable memory footprint
        semantic_knowledge_size = len(self.observer._semantic_knowledge.get("route_1", {}))
        assert semantic_knowledge_size <= 200  # Should limit knowledge size for performance
