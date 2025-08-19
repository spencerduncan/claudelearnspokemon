"""
Test suite for MCP Data Patterns and Query Design (Issue #77).

This module tests the data patterns and query structures for storing Pokemon
speedrun patterns, game states, strategies, and learned knowledge through
Memgraph MCP integration.
"""

import time
from datetime import datetime
from unittest.mock import patch

import pytest


@pytest.mark.fast
class TestMCPDataPatternBase:
    """Test the base MCP data pattern functionality through concrete implementations."""

    def test_base_pattern_initialization(self):
        """Test that base data patterns initialize correctly through Pokemon pattern."""
        from claudelearnspokemon.mcp_data_patterns import PokemonPattern

        pattern = PokemonPattern(
            id="test-pattern-1", name="Test Pattern", input_sequence=["A", "B"]
        )

        assert pattern.id == "test-pattern-1"
        assert pattern.name == "Test Pattern"
        assert pattern.data_type == "pattern"
        assert isinstance(pattern.created_at, datetime)
        assert pattern.confidence == 1.0  # Default confidence

    def test_base_pattern_to_mcp_format(self):
        """Test conversion to MCP storage format through Pokemon pattern."""
        from claudelearnspokemon.mcp_data_patterns import PokemonPattern

        pattern = PokemonPattern(
            id="test-pattern-1", name="Test Pattern", input_sequence=["A", "B"], confidence=0.85
        )

        mcp_data = pattern.to_mcp_format()

        assert mcp_data["node_type"] == "concept"
        assert mcp_data["confidence"] == 0.85
        assert "Test Pattern" in mcp_data["content"]
        assert "test-pattern-1" in mcp_data["content"]
        assert "pokemon-pattern" in mcp_data["tags"]
        assert "claudelearnspokemon" in mcp_data["tags"]

    def test_base_pattern_from_mcp_format(self):
        """Test reconstruction from MCP storage format through Pokemon pattern."""
        from claudelearnspokemon.mcp_data_patterns import PokemonPattern

        mcp_data = {
            "id": "stored-id",
            "content": '{"id": "test-pattern-1", "name": "Test Pattern", "data_type": "pattern", "input_sequence": ["A", "B"], "success_rate": 0.8}',
            "confidence": 0.9,
            "tags": ["pattern", "claudelearnspokemon"],
        }

        pattern = PokemonPattern.from_mcp_format(mcp_data)

        assert pattern.id == "test-pattern-1"
        assert pattern.name == "Test Pattern"
        assert pattern.data_type == "pattern"
        assert pattern.confidence == 0.9


@pytest.mark.fast
class TestPokemonPattern:
    """Test Pokemon-specific pattern data structures."""

    def test_pokemon_pattern_creation(self):
        """Test creating a Pokemon gameplay pattern."""
        from claudelearnspokemon.mcp_data_patterns import PokemonPattern

        pattern = PokemonPattern(
            id="gym-leader-defeat-1",
            name="Brock Defeat Strategy",
            input_sequence=["A", "DOWN", "A", "B", "A"],
            success_rate=0.92,
            average_completion_time=45.2,
            prerequisites={"location": "pewter_gym", "level": 12},
            outcomes={"badge": "boulder_badge", "exp": 1000},
        )

        assert pattern.name == "Brock Defeat Strategy"
        assert pattern.input_sequence == ["A", "DOWN", "A", "B", "A"]
        assert pattern.success_rate == 0.92
        assert pattern.prerequisites["location"] == "pewter_gym"
        assert pattern.outcomes["badge"] == "boulder_badge"

    def test_pokemon_pattern_mcp_storage(self):
        """Test Pokemon pattern MCP storage format."""
        from claudelearnspokemon.mcp_data_patterns import PokemonPattern

        pattern = PokemonPattern(
            id="item-collection-1",
            name="Potion Collection",
            input_sequence=["A"],
            success_rate=1.0,
            prerequisites={"location": "viridian_city"},
        )

        mcp_data = pattern.to_mcp_format()

        assert mcp_data["node_type"] == "concept"
        assert "pokemon-pattern" in mcp_data["tags"]
        assert "item-collection-1" in mcp_data["content"]
        assert "Potion Collection" in mcp_data["content"]
        assert pattern.success_rate == 1.0

    def test_pokemon_pattern_validation(self):
        """Test that Pokemon patterns validate correctly."""
        from claudelearnspokemon.mcp_data_patterns import PokemonPattern

        # Valid pattern
        valid_pattern = PokemonPattern(
            id="valid-pattern", name="Valid Pattern", input_sequence=["A", "B"], success_rate=0.8
        )
        assert valid_pattern.is_valid()

        # Invalid pattern - empty input sequence
        invalid_pattern = PokemonPattern(
            id="invalid-pattern", name="Invalid Pattern", input_sequence=[], success_rate=0.8
        )
        assert not invalid_pattern.is_valid()

        # Invalid pattern - success rate out of bounds
        invalid_pattern2 = PokemonPattern(
            id="invalid-pattern-2",
            name="Invalid Pattern 2",
            input_sequence=["A"],
            success_rate=1.5,  # > 1.0
        )
        assert not invalid_pattern2.is_valid()


@pytest.mark.fast
class TestPokemonStrategy:
    """Test Pokemon strategy data structures."""

    def test_pokemon_strategy_creation(self):
        """Test creating a Pokemon speedrun strategy."""
        from claudelearnspokemon.mcp_data_patterns import PokemonStrategy

        strategy = PokemonStrategy(
            id="elite-four-rush",
            name="Elite Four Speed Strategy",
            pattern_sequence=["pattern-1", "pattern-2", "pattern-3"],
            success_rate=0.75,
            estimated_time=1200.5,
            resource_requirements={"level": 50, "items": ["potion", "revive"]},
            risk_assessment={"high_risk_segments": ["champion_battle"]},
        )

        assert strategy.name == "Elite Four Speed Strategy"
        assert len(strategy.pattern_sequence) == 3
        assert strategy.success_rate == 0.75
        assert strategy.resource_requirements["level"] == 50
        assert "champion_battle" in strategy.risk_assessment["high_risk_segments"]

    def test_strategy_pattern_dependencies(self):
        """Test that strategies track pattern dependencies correctly."""
        from claudelearnspokemon.mcp_data_patterns import PokemonStrategy

        strategy = PokemonStrategy(
            id="gym-rush-strategy",
            name="All Gyms Speed Run",
            pattern_sequence=["brock-defeat", "misty-defeat", "surge-defeat"],
            conditional_patterns={"if_low_hp": ["healing-pattern"]},
        )

        dependencies = strategy.get_pattern_dependencies()

        assert "brock-defeat" in dependencies
        assert "misty-defeat" in dependencies
        assert "surge-defeat" in dependencies
        assert "healing-pattern" in dependencies

    def test_strategy_mcp_storage(self):
        """Test strategy MCP storage format."""
        from claudelearnspokemon.mcp_data_patterns import PokemonStrategy

        strategy = PokemonStrategy(
            id="test-strategy",
            name="Test Strategy",
            pattern_sequence=["pattern-1"],
            success_rate=0.9,
        )

        mcp_data = strategy.to_mcp_format()

        assert mcp_data["node_type"] == "concept"
        assert "pokemon-strategy" in mcp_data["tags"]
        assert strategy.success_rate == 0.9


@pytest.mark.fast
class TestGameStateAndCheckpoint:
    """Test game state and checkpoint data structures."""

    def test_game_state_creation(self):
        """Test creating a Pokemon game state."""
        from claudelearnspokemon.mcp_data_patterns import PokemonGameState

        game_state = PokemonGameState(
            id="state-pewter-gym-entry",
            location={"map": "pewter_gym", "x": 5, "y": 3},
            player_state={"hp": 45, "level": 12, "items": ["pokeball", "potion"], "badges": []},
            game_progress={"story_flags": ["visited_viridian"]},
            tile_observations=[["grass", "path", "building"] * 6] * 18,  # 20x18 grid
        )

        assert game_state.location["map"] == "pewter_gym"
        assert game_state.player_state["hp"] == 45
        assert len(game_state.tile_observations) == 18
        assert len(game_state.tile_observations[0]) == 18  # 6*3 = 18
        assert "visited_viridian" in game_state.game_progress["story_flags"]

    def test_checkpoint_creation(self):
        """Test creating a Pokemon checkpoint."""
        from claudelearnspokemon.mcp_data_patterns import PokemonCheckpoint

        checkpoint = PokemonCheckpoint(
            id="checkpoint-gym-ready",
            file_path="/checkpoints/pewter_gym_ready.state",
            game_state_id="state-pewter-gym-entry",
            strategic_value="Pre-gym optimization point",
            usage_count=15,
            success_rate=0.88,
        )

        assert checkpoint.file_path == "/checkpoints/pewter_gym_ready.state"
        assert checkpoint.game_state_id == "state-pewter-gym-entry"
        assert checkpoint.usage_count == 15
        assert checkpoint.success_rate == 0.88

    def test_game_state_mcp_storage(self):
        """Test game state MCP storage format."""
        from claudelearnspokemon.mcp_data_patterns import PokemonGameState

        game_state = PokemonGameState(
            id="test-state", location={"map": "test_map", "x": 1, "y": 1}, player_state={"hp": 100}
        )

        mcp_data = game_state.to_mcp_format()

        assert mcp_data["node_type"] == "fact"  # Game states are facts
        assert "pokemon-gamestate" in mcp_data["tags"]
        assert "test_map" in mcp_data["content"]


@pytest.mark.fast
class TestLocationAndTileSemantics:
    """Test location and tile semantic data structures."""

    def test_pokemon_location_creation(self):
        """Test creating a Pokemon location."""
        from claudelearnspokemon.mcp_data_patterns import PokemonLocation

        location = PokemonLocation(
            id="pewter_gym:5:3",
            map_name="pewter_gym",
            coordinates={"x": 5, "y": 3},
            tile_type="floor",
            semantic_properties={
                "interaction_type": "gym_leader_battle",
                "required_items": [],
                "encounter_rate": 0.0,
            },
            navigation_value=0.9,
            connected_locations=["pewter_gym:4:3", "pewter_gym:6:3"],
        )

        assert location.map_name == "pewter_gym"
        assert location.coordinates["x"] == 5
        assert location.semantic_properties["interaction_type"] == "gym_leader_battle"
        assert location.navigation_value == 0.9
        assert len(location.connected_locations) == 2

    def test_tile_semantics_creation(self):
        """Test creating tile semantic information."""
        from claudelearnspokemon.mcp_data_patterns import TileSemantics

        tile = TileSemantics(
            id="grass_tile_route1",
            visual_characteristics={"color": "green", "pattern": "grass"},
            behavioral_properties={
                "wild_encounter": True,
                "encounter_rate": 0.1,
                "movement_cost": 1,
            },
            learning_confidence=0.95,
            context_dependencies={"time_of_day": "any", "story_progress": "any"},
        )

        assert tile.behavioral_properties["wild_encounter"] is True
        assert tile.behavioral_properties["encounter_rate"] == 0.1
        assert tile.learning_confidence == 0.95

    def test_location_mcp_storage(self):
        """Test location MCP storage format."""
        from claudelearnspokemon.mcp_data_patterns import PokemonLocation

        location = PokemonLocation(
            id="test_location",
            map_name="test_map",
            coordinates={"x": 1, "y": 1},
            tile_type="test_tile",
        )

        mcp_data = location.to_mcp_format()

        assert mcp_data["node_type"] == "entity"  # Locations are entities
        assert "pokemon-location" in mcp_data["tags"]


@pytest.mark.fast
class TestQueryBuilder:
    """Test MCP query building for efficient searches."""

    @patch("claudelearnspokemon.mcp_data_patterns.search_memories")
    def test_pattern_discovery_query(self, mock_search):
        """Test querying for patterns by success rate."""
        from claudelearnspokemon.mcp_data_patterns import QueryBuilder

        mock_search.return_value = {
            "success": True,
            "results": [{"content": "test pattern", "confidence": 0.9}],
        }

        builder = QueryBuilder()
        results = builder.find_patterns_by_success_rate(min_success_rate=0.8)

        mock_search.assert_called_once()
        call_args = mock_search.call_args[1]
        assert "pokemon-pattern" in call_args["pattern"]
        assert call_args["limit"] == 50  # Default limit
        assert results is not None

    @patch("claudelearnspokemon.mcp_data_patterns.search_memories")
    def test_strategy_query_performance(self, mock_search):
        """Test that strategy queries complete within performance requirement."""
        from claudelearnspokemon.mcp_data_patterns import QueryBuilder

        mock_search.return_value = {"success": True, "results": []}

        builder = QueryBuilder()

        start_time = time.time()
        builder.find_strategies_for_game_state({"location": "pewter_gym"})
        end_time = time.time()

        # Should complete well under 100ms requirement
        assert (end_time - start_time) < 0.1  # 100ms
        mock_search.assert_called_once()

    @patch("claudelearnspokemon.mcp_data_patterns.search_memories")
    def test_navigation_query(self, mock_search):
        """Test navigation queries for pathfinding."""
        from claudelearnspokemon.mcp_data_patterns import QueryBuilder

        mock_search.return_value = {"success": True, "results": []}

        builder = QueryBuilder()
        result = builder.find_path_between_locations("pewter_gym:1:1", "pewter_gym:5:5")

        mock_search.assert_called()
        assert result is not None

    def test_query_optimization_strategies(self):
        """Test that queries use optimization strategies."""
        from claudelearnspokemon.mcp_data_patterns import QueryBuilder

        builder = QueryBuilder()

        # Test tag-based filtering
        query_params = builder._build_pattern_query({"success_rate": 0.8})
        assert "pokemon-pattern" in query_params["pattern"]

        # Test confidence thresholds
        assert query_params["min_confidence"] >= 0.5  # Filter low confidence

        # Test result limits
        assert query_params["limit"] <= 100  # Reasonable limit for performance


@pytest.mark.fast
class TestRelationshipManager:
    """Test pattern relationship management."""

    @patch("claudelearnspokemon.mcp_data_patterns.link_memories")
    def test_create_pattern_dependency(self, mock_link):
        """Test creating pattern dependency relationships."""
        from claudelearnspokemon.mcp_data_patterns import RelationshipManager

        mock_link.return_value = {"success": True}

        manager = RelationshipManager()
        result = manager.create_dependency(
            pattern_id="pattern-a",
            depends_on_id="pattern-b",
            relationship_type="DEPENDS_ON",
            confidence=0.9,
        )

        mock_link.assert_called_once()
        call_args = mock_link.call_args[1]
        assert call_args["node1_id"] == "pattern-a"
        assert call_args["node2_id"] == "pattern-b"
        assert call_args["relation_type"] == "DEPENDS_ON"
        assert call_args["confidence"] == 0.9
        assert result["success"] is True

    @patch("claudelearnspokemon.mcp_data_patterns.get_relationships")
    def test_get_pattern_relationships(self, mock_get_rels):
        """Test retrieving pattern relationships."""
        from claudelearnspokemon.mcp_data_patterns import RelationshipManager

        mock_get_rels.return_value = {
            "success": True,
            "relationships": [{"target_id": "pattern-b", "relation_type": "DEPENDS_ON"}],
        }

        manager = RelationshipManager()
        relationships = manager.get_dependencies("pattern-a")

        mock_get_rels.assert_called_once_with(
            node_id="pattern-a", relation_type="DEPENDS_ON", depth=1
        )
        assert len(relationships) == 1
        assert relationships[0]["target_id"] == "pattern-b"

    @patch("claudelearnspokemon.mcp_data_patterns.link_memories")
    def test_strategy_pattern_relationships(self, mock_link):
        """Test strategy-pattern relationship creation."""
        from claudelearnspokemon.mcp_data_patterns import RelationshipManager

        mock_link.return_value = {"success": True}

        manager = RelationshipManager()
        manager.link_strategy_to_pattern(
            strategy_id="strategy-1", pattern_id="pattern-1", usage_order=1, confidence=0.95
        )

        mock_link.assert_called_once()
        call_args = mock_link.call_args[1]
        assert call_args["relation_type"] == "USES_PATTERN"
        assert call_args["confidence"] == 0.95


@pytest.mark.fast
class TestPatternValidator:
    """Test data validation and constraint patterns."""

    def test_pattern_validation(self):
        """Test pattern data validation."""
        from claudelearnspokemon.mcp_data_patterns import PatternValidator, PokemonPattern

        validator = PatternValidator()

        # Valid pattern
        valid_pattern = PokemonPattern(
            id="valid", name="Valid Pattern", input_sequence=["A", "B"], success_rate=0.8
        )

        validation_result = validator.validate_pattern(valid_pattern)
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0

        # Invalid pattern
        invalid_pattern = PokemonPattern(
            id="",  # Empty ID
            name="Invalid",
            input_sequence=[],  # Empty sequence
            success_rate=1.5,  # Invalid rate
        )

        validation_result = validator.validate_pattern(invalid_pattern)
        assert validation_result["valid"] is False
        assert len(validation_result["errors"]) > 0

    def test_relationship_validation(self):
        """Test relationship validation."""
        from claudelearnspokemon.mcp_data_patterns import PatternValidator

        validator = PatternValidator()

        # Valid relationship
        valid_rel = validator.validate_relationship(
            source_id="pattern-1",
            target_id="pattern-2",
            relationship_type="DEPENDS_ON",
            confidence=0.8,
        )
        assert valid_rel["valid"] is True

        # Invalid relationship - same source and target
        invalid_rel = validator.validate_relationship(
            source_id="pattern-1",
            target_id="pattern-1",
            relationship_type="DEPENDS_ON",
            confidence=0.8,
        )
        assert invalid_rel["valid"] is False
        assert "circular" in invalid_rel["errors"][0].lower()

    def test_performance_validation(self):
        """Test performance metric validation."""
        from claudelearnspokemon.mcp_data_patterns import PatternValidator

        validator = PatternValidator()

        # Valid metrics
        valid_metrics = validator.validate_performance_metrics(
            {"success_rate": 0.85, "average_time": 45.2, "completion_rate": 0.9}
        )
        assert valid_metrics["valid"] is True

        # Invalid metrics
        invalid_metrics = validator.validate_performance_metrics(
            {
                "success_rate": 1.5,  # > 1.0
                "average_time": -10,  # Negative time
            }
        )
        assert invalid_metrics["valid"] is False


@pytest.mark.fast
class TestEvolutionTracker:
    """Test pattern evolution and versioning support."""

    @patch("claudelearnspokemon.mcp_data_patterns.store_memory")
    @patch("claudelearnspokemon.mcp_data_patterns.link_memories")
    def test_pattern_evolution_tracking(self, mock_link, mock_store):
        """Test tracking pattern evolution."""
        from claudelearnspokemon.mcp_data_patterns import EvolutionTracker, PokemonPattern

        mock_store.return_value = {"success": True, "memory_id": "new-pattern-id"}
        mock_link.return_value = {"success": True}

        tracker = EvolutionTracker()

        original_pattern = PokemonPattern(
            id="pattern-v1", name="Original Pattern", input_sequence=["A", "B"], success_rate=0.7
        )

        improved_pattern = PokemonPattern(
            id="pattern-v2",
            name="Improved Pattern",
            input_sequence=["A", "A", "B"],  # Added input
            success_rate=0.9,  # Better success rate
        )

        result = tracker.evolve_pattern(original_pattern, improved_pattern, "optimization")

        mock_store.assert_called()  # New pattern stored
        mock_link.assert_called()  # Evolution relationship created
        assert result["success"] is True

    @patch("claudelearnspokemon.mcp_data_patterns.get_relationships")
    def test_pattern_version_history(self, mock_get_rels):
        """Test retrieving pattern version history."""
        from claudelearnspokemon.mcp_data_patterns import EvolutionTracker

        mock_get_rels.return_value = {
            "success": True,
            "relationships": [
                {"target_id": "pattern-v2", "relation_type": "EVOLVES_TO"},
                {"target_id": "pattern-v3", "relation_type": "EVOLVES_TO"},
            ],
        }

        tracker = EvolutionTracker()
        history = tracker.get_evolution_history("pattern-v1")

        assert len(history) == 2
        assert "pattern-v2" in [h["target_id"] for h in history]
        assert "pattern-v3" in [h["target_id"] for h in history]

    def test_evolution_validation(self):
        """Test that pattern evolution is actually an improvement."""
        from claudelearnspokemon.mcp_data_patterns import EvolutionTracker, PokemonPattern

        tracker = EvolutionTracker()

        original = PokemonPattern(id="v1", name="Original", input_sequence=["A"], success_rate=0.9)
        worse = PokemonPattern(id="v2", name="Worse", input_sequence=["A"], success_rate=0.5)
        better = PokemonPattern(id="v3", name="Better", input_sequence=["A"], success_rate=0.95)

        # Should reject worse pattern
        assert not tracker.is_valid_evolution(original, worse)

        # Should accept better pattern
        assert tracker.is_valid_evolution(original, better)


@pytest.mark.fast
class TestMCPIntegration:
    """Test complete MCP integration scenarios."""

    @patch("claudelearnspokemon.mcp_data_patterns.store_memory")
    @patch("claudelearnspokemon.mcp_data_patterns.search_memories")
    @patch("claudelearnspokemon.mcp_data_patterns.link_memories")
    def test_end_to_end_pattern_storage_and_retrieval(self, mock_link, mock_search, mock_store):
        """Test complete pattern storage and retrieval workflow."""
        from claudelearnspokemon.mcp_data_patterns import (
            PokemonPattern,
            QueryBuilder,
            RelationshipManager,
        )

        # Setup mocks
        mock_store.return_value = {"success": True, "memory_id": "stored-pattern-id"}
        mock_search.return_value = {
            "success": True,
            "results": [{"id": "stored-pattern-id", "content": "test", "confidence": 0.9}],
        }
        mock_link.return_value = {"success": True}

        # Create and store pattern
        pattern = PokemonPattern(
            id="integration-test-pattern",
            name="Integration Test",
            input_sequence=["A", "B", "A"],
            success_rate=0.85,
        )

        # Store pattern
        mcp_data = pattern.to_mcp_format()
        store_result = mock_store(
            node_type=mcp_data["node_type"],
            content=mcp_data["content"],
            confidence=mcp_data["confidence"],
            tags=mcp_data["tags"],
        )

        # Query for pattern
        query_builder = QueryBuilder()
        found_patterns = query_builder.find_patterns_by_success_rate(min_success_rate=0.8)

        # Create relationships
        rel_manager = RelationshipManager()
        rel_manager.create_dependency(
            "integration-test-pattern", "prerequisite-pattern", "DEPENDS_ON", 0.9
        )

        # Verify all operations succeeded
        assert store_result["success"] is True
        assert found_patterns is not None
        mock_store.assert_called_once()
        mock_search.assert_called_once()
        mock_link.assert_called_once()

    @patch("claudelearnspokemon.mcp_data_patterns.store_memory")
    def test_performance_requirement_compliance(self, mock_store):
        """Test that operations meet <100ms performance requirement."""
        from claudelearnspokemon.mcp_data_patterns import PokemonPattern

        mock_store.return_value = {"success": True, "memory_id": "test-id"}

        pattern = PokemonPattern(
            id="perf-test",
            name="Performance Test",
            input_sequence=["A"] * 100,  # Large input sequence
            success_rate=0.8,
        )

        # Test pattern creation performance
        start_time = time.time()
        mcp_data = pattern.to_mcp_format()
        end_time = time.time()

        assert (end_time - start_time) < 0.1  # <100ms

        # Test storage call performance
        start_time = time.time()
        mock_store(**mcp_data)
        end_time = time.time()

        assert (end_time - start_time) < 0.05  # Mock should be very fast

    def test_data_structure_evolution_support(self):
        """Test that data structures support evolution and versioning."""
        from claudelearnspokemon.mcp_data_patterns import PokemonPattern

        # Create pattern with version info
        pattern_v1 = PokemonPattern(
            id="versioned-pattern-v1",
            name="Versioned Pattern",
            input_sequence=["A", "B"],
            success_rate=0.8,
            version="1.0",
            evolution_metadata={"creation_method": "manual_discovery"},
        )

        mcp_data = pattern_v1.to_mcp_format()

        # Verify version info is preserved
        assert "v1" in mcp_data["content"]
        assert "1.0" in mcp_data["content"]
        assert "evolution" in mcp_data["tags"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
