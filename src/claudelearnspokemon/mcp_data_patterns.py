"""
MCP Data Patterns and Query Design for Pokemon Speedrun Learning System.

This module implements data patterns and query structures for storing Pokemon
speedrun patterns, game states, strategies, and learned knowledge through
Memgraph MCP integration.

Design follows Clean Code principles:
- Single Responsibility: Each class has one clear purpose
- Interface Segregation: Clean interfaces for different pattern types
- Dependency Inversion: Abstractions over concrete MCP operations
- Open/Closed: Extensible for new pattern types

Performance target: <100ms query response time
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# MCP Integration - Import the memory functions
try:
    from mcp__memgraph_memory__get_relationships import get_relationships  # type: ignore
    from mcp__memgraph_memory__link_memories import link_memories  # type: ignore
    from mcp__memgraph_memory__search_memories import search_memories  # type: ignore
    from mcp__memgraph_memory__store_memory import store_memory  # type: ignore
except ImportError:
    # For testing - create mock functions
    def store_memory(*args, **kwargs):
        return {"success": True, "memory_id": "mock-id"}

    def search_memories(*args, **kwargs):
        return {"success": True, "results": []}

    def link_memories(*args, **kwargs):
        return {"success": True}

    def get_relationships(*args, **kwargs):
        return {"success": True, "relationships": []}


class MCPDataPattern(ABC):
    """
    Base class for all MCP data patterns.

    Provides common functionality for converting to/from MCP storage format
    and validation. Follows Single Responsibility Principle.
    """

    def __init__(self, id: str, name: str, data_type: str, confidence: float = 1.0):
        """Initialize base pattern data."""
        self.id = id
        self.name = name
        self.data_type = data_type
        self.confidence = confidence
        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at

    def to_mcp_format(self) -> dict[str, Any]:
        """
        Convert pattern to MCP storage format.

        Returns:
            Dict containing node_type, content, confidence, tags for MCP storage
        """
        # Determine node type based on data pattern type
        node_type_mapping = {
            "pattern": "concept",
            "strategy": "concept",
            "gamestate": "fact",
            "checkpoint": "fact",
            "location": "entity",
            "tile": "entity",
        }

        node_type = node_type_mapping.get(self.data_type, "concept")

        # Create structured content
        content = self._serialize_content()

        # Generate appropriate tags
        tags = self._generate_tags()

        return {
            "node_type": node_type,
            "content": content,
            "confidence": self.confidence,
            "tags": tags,
            "source": "pokemon_speedrun_system",
        }

    @classmethod
    def from_mcp_format(cls, mcp_data: dict[str, Any]) -> "MCPDataPattern":
        """
        Reconstruct pattern from MCP storage format.

        Args:
            mcp_data: Dictionary containing MCP stored data

        Returns:
            Reconstructed pattern instance
        """
        # Parse content - assume JSON format
        content_data = json.loads(mcp_data["content"])

        # This method should be overridden in concrete classes
        # For now, return a basic reconstruction
        data_type = content_data.get("data_type", "pattern")

        if data_type == "pattern":
            # Import here to avoid circular imports
            return PokemonPattern.from_mcp_format(mcp_data)
        elif data_type == "strategy":
            return PokemonStrategy.from_mcp_format(mcp_data)
        elif data_type == "gamestate":
            return PokemonGameState.from_mcp_format(mcp_data)
        elif data_type == "checkpoint":
            return PokemonCheckpoint.from_mcp_format(mcp_data)
        elif data_type == "location":
            return PokemonLocation.from_mcp_format(mcp_data)
        elif data_type == "tile":
            return TileSemantics.from_mcp_format(mcp_data)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    @abstractmethod
    def _serialize_content(self) -> str:
        """Serialize pattern-specific content for storage."""
        pass

    def _generate_tags(self) -> list[str]:
        """Generate standard tags for MCP storage."""
        base_tags = ["claudelearnspokemon", f"pokemon-{self.data_type}", "speedrun-learning"]

        # Add specific tags based on pattern type
        specific_tags = self._get_specific_tags()

        return base_tags + specific_tags

    def _get_specific_tags(self) -> list[str]:
        """Get pattern-specific tags. Override in subclasses."""
        return []

    def is_valid(self) -> bool:
        """Basic validation for all patterns."""
        return bool(self.id) and bool(self.name) and 0.0 <= self.confidence <= 1.0


@dataclass
class PokemonPattern(MCPDataPattern):
    """
    Represents a reusable Pokemon gameplay sequence.

    Patterns are discovered through gameplay and represent effective
    input sequences for achieving specific outcomes.
    """

    input_sequence: list[str] = field(default_factory=list)
    success_rate: float = 0.0
    average_completion_time: float | None = None
    frame_count: int | None = None
    prerequisites: dict[str, Any] = field(default_factory=dict)
    outcomes: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    discovery_method: str = "unknown"
    version: str = "1.0"
    evolution_metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self, id: str, name: str, input_sequence: list[str], success_rate: float = 0.0, **kwargs
    ):
        """Initialize Pokemon pattern."""
        super().__init__(id, name, "pattern", kwargs.get("confidence", 1.0))
        self.input_sequence = input_sequence
        self.success_rate = success_rate
        self.average_completion_time = kwargs.get("average_completion_time")
        self.frame_count = kwargs.get("frame_count")
        self.prerequisites = kwargs.get("prerequisites", {})
        self.outcomes = kwargs.get("outcomes", {})
        self.context = kwargs.get("context", {})
        self.discovery_method = kwargs.get("discovery_method", "unknown")
        self.version = kwargs.get("version", "1.0")
        self.evolution_metadata = kwargs.get("evolution_metadata", {})

    def _serialize_content(self) -> str:
        """Serialize pattern content for MCP storage."""
        data = {
            "id": self.id,
            "name": self.name,
            "data_type": self.data_type,
            "input_sequence": self.input_sequence,
            "success_rate": self.success_rate,
            "average_completion_time": self.average_completion_time,
            "frame_count": self.frame_count,
            "prerequisites": self.prerequisites,
            "outcomes": self.outcomes,
            "context": self.context,
            "discovery_method": self.discovery_method,
            "version": self.version,
            "evolution_metadata": self.evolution_metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        return json.dumps(data, indent=2)

    def _get_specific_tags(self) -> list[str]:
        """Get pattern-specific tags."""
        tags = ["gameplay-sequence"]

        if self.success_rate >= 0.9:
            tags.append("high-success")
        elif self.success_rate >= 0.7:
            tags.append("medium-success")
        else:
            tags.append("low-success")

        if "location" in self.prerequisites:
            tags.append(f"location-{self.prerequisites['location']}")

        if self.version != "1.0" or self.evolution_metadata:
            tags.append("evolution")

        return tags

    def is_valid(self) -> bool:
        """Validate Pokemon pattern data."""
        return (
            super().is_valid() and len(self.input_sequence) > 0 and 0.0 <= self.success_rate <= 1.0
        )

    @classmethod
    def from_mcp_format(cls, mcp_data: dict[str, Any]) -> "PokemonPattern":
        """Reconstruct Pokemon pattern from MCP storage format."""
        content_data = json.loads(mcp_data["content"])

        return cls(
            id=content_data["id"],
            name=content_data["name"],
            input_sequence=content_data.get("input_sequence", []),
            success_rate=content_data.get("success_rate", 0.0),
            average_completion_time=content_data.get("average_completion_time"),
            frame_count=content_data.get("frame_count"),
            prerequisites=content_data.get("prerequisites", {}),
            outcomes=content_data.get("outcomes", {}),
            context=content_data.get("context", {}),
            discovery_method=content_data.get("discovery_method", "unknown"),
            version=content_data.get("version", "1.0"),
            evolution_metadata=content_data.get("evolution_metadata", {}),
            confidence=mcp_data.get("confidence", 1.0),
        )


@dataclass
class PokemonStrategy(MCPDataPattern):
    """
    Represents a high-level Pokemon speedrun strategy.

    Strategies combine multiple patterns into cohesive game plans
    with conditional logic and fallback options.
    """

    pattern_sequence: list[str] = field(default_factory=list)
    conditional_patterns: dict[str, list[str]] = field(default_factory=dict)
    success_rate: float = 0.0
    estimated_time: float | None = None
    resource_requirements: dict[str, Any] = field(default_factory=dict)
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    alternatives: list[str] = field(default_factory=list)
    optimization_history: list[dict[str, Any]] = field(default_factory=list)

    def __init__(
        self, id: str, name: str, pattern_sequence: list[str], success_rate: float = 0.0, **kwargs
    ):
        """Initialize Pokemon strategy."""
        super().__init__(id, name, "strategy", kwargs.get("confidence", 1.0))
        self.pattern_sequence = pattern_sequence
        self.conditional_patterns = kwargs.get("conditional_patterns", {})
        self.success_rate = success_rate
        self.estimated_time = kwargs.get("estimated_time")
        self.resource_requirements = kwargs.get("resource_requirements", {})
        self.risk_assessment = kwargs.get("risk_assessment", {})
        self.alternatives = kwargs.get("alternatives", [])
        self.optimization_history = kwargs.get("optimization_history", [])

    def get_pattern_dependencies(self) -> list[str]:
        """Get all patterns that this strategy depends on."""
        dependencies = list(self.pattern_sequence)

        # Add conditional patterns
        for condition_patterns in self.conditional_patterns.values():
            dependencies.extend(condition_patterns)

        return list(set(dependencies))  # Remove duplicates

    def _serialize_content(self) -> str:
        """Serialize strategy content for MCP storage."""
        data = {
            "id": self.id,
            "name": self.name,
            "data_type": self.data_type,
            "pattern_sequence": self.pattern_sequence,
            "conditional_patterns": self.conditional_patterns,
            "success_rate": self.success_rate,
            "estimated_time": self.estimated_time,
            "resource_requirements": self.resource_requirements,
            "risk_assessment": self.risk_assessment,
            "alternatives": self.alternatives,
            "optimization_history": self.optimization_history,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        return json.dumps(data, indent=2)

    def _get_specific_tags(self) -> list[str]:
        """Get strategy-specific tags."""
        tags = ["high-level-plan"]

        if len(self.pattern_sequence) > 10:
            tags.append("complex-strategy")
        else:
            tags.append("simple-strategy")

        if self.risk_assessment.get("high_risk_segments"):
            tags.append("high-risk")

        return tags


@dataclass
class PokemonGameState(MCPDataPattern):
    """
    Represents a snapshot of Pokemon game state.

    Game states capture the complete state of the game at a specific
    moment for analysis and checkpointing.
    """

    location: dict[str, Any] = field(default_factory=dict)
    player_state: dict[str, Any] = field(default_factory=dict)
    game_progress: dict[str, Any] = field(default_factory=dict)
    tile_observations: list[list[str]] = field(default_factory=list)
    performance_context: dict[str, Any] = field(default_factory=dict)
    strategic_significance: str = ""

    def __init__(
        self,
        id: str,
        location: dict[str, Any],
        player_state: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize Pokemon game state."""
        super().__init__(id, f"GameState-{id}", "gamestate", kwargs.get("confidence", 1.0))
        self.location = location
        self.player_state = player_state or {}
        self.game_progress = kwargs.get("game_progress", {})
        self.tile_observations = kwargs.get("tile_observations", [])
        self.performance_context = kwargs.get("performance_context", {})
        self.strategic_significance = kwargs.get("strategic_significance", "")

    def _serialize_content(self) -> str:
        """Serialize game state content for MCP storage."""
        data = {
            "id": self.id,
            "name": self.name,
            "data_type": self.data_type,
            "location": self.location,
            "player_state": self.player_state,
            "game_progress": self.game_progress,
            "tile_observations": self.tile_observations,
            "performance_context": self.performance_context,
            "strategic_significance": self.strategic_significance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        return json.dumps(data, indent=2)

    def _get_specific_tags(self) -> list[str]:
        """Get game state specific tags."""
        tags = ["game-snapshot"]

        if "map" in self.location:
            tags.append(f"map-{self.location['map']}")

        if self.strategic_significance:
            tags.append("strategic-checkpoint")

        return tags


@dataclass
class PokemonCheckpoint(MCPDataPattern):
    """
    Represents a Pokemon game checkpoint for deterministic replay.

    Checkpoints enable loading specific game states for experimentation
    and pattern development.
    """

    file_path: str = ""
    game_state_id: str = ""
    strategic_value: str = ""
    usage_count: int = 0
    success_rate: float = 0.0
    branching_strategies: list[str] = field(default_factory=list)

    def __init__(self, id: str, file_path: str, game_state_id: str, **kwargs):
        """Initialize Pokemon checkpoint."""
        super().__init__(id, f"Checkpoint-{id}", "checkpoint", kwargs.get("confidence", 1.0))
        self.file_path = file_path
        self.game_state_id = game_state_id
        self.strategic_value = kwargs.get("strategic_value", "")
        self.usage_count = kwargs.get("usage_count", 0)
        self.success_rate = kwargs.get("success_rate", 0.0)
        self.branching_strategies = kwargs.get("branching_strategies", [])

    def _serialize_content(self) -> str:
        """Serialize checkpoint content for MCP storage."""
        data = {
            "id": self.id,
            "name": self.name,
            "data_type": self.data_type,
            "file_path": self.file_path,
            "game_state_id": self.game_state_id,
            "strategic_value": self.strategic_value,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "branching_strategies": self.branching_strategies,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        return json.dumps(data, indent=2)

    def _get_specific_tags(self) -> list[str]:
        """Get checkpoint specific tags."""
        tags = ["save-state"]

        if self.usage_count > 10:
            tags.append("frequently-used")

        if self.success_rate >= 0.8:
            tags.append("reliable-checkpoint")

        return tags


@dataclass
class PokemonLocation(MCPDataPattern):
    """
    Represents a Pokemon game location with learned semantics.

    Locations capture spatial knowledge including navigation value,
    interaction types, and strategic importance.
    """

    map_name: str = ""
    coordinates: dict[str, int] = field(default_factory=dict)
    tile_type: str = ""
    semantic_properties: dict[str, Any] = field(default_factory=dict)
    navigation_value: float = 0.0
    risk_factors: list[str] = field(default_factory=list)
    connected_locations: list[str] = field(default_factory=list)
    strategic_importance: float = 0.0

    def __init__(
        self, id: str, map_name: str, coordinates: dict[str, int], tile_type: str, **kwargs
    ):
        """Initialize Pokemon location."""
        super().__init__(
            id,
            f"Location-{map_name}:{coordinates.get('x', 0)}:{coordinates.get('y', 0)}",
            "location",
            kwargs.get("confidence", 1.0),
        )
        self.map_name = map_name
        self.coordinates = coordinates
        self.tile_type = tile_type
        self.semantic_properties = kwargs.get("semantic_properties", {})
        self.navigation_value = kwargs.get("navigation_value", 0.0)
        self.risk_factors = kwargs.get("risk_factors", [])
        self.connected_locations = kwargs.get("connected_locations", [])
        self.strategic_importance = kwargs.get("strategic_importance", 0.0)

    def _serialize_content(self) -> str:
        """Serialize location content for MCP storage."""
        data = {
            "id": self.id,
            "name": self.name,
            "data_type": self.data_type,
            "map_name": self.map_name,
            "coordinates": self.coordinates,
            "tile_type": self.tile_type,
            "semantic_properties": self.semantic_properties,
            "navigation_value": self.navigation_value,
            "risk_factors": self.risk_factors,
            "connected_locations": self.connected_locations,
            "strategic_importance": self.strategic_importance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        return json.dumps(data, indent=2)

    def _get_specific_tags(self) -> list[str]:
        """Get location specific tags."""
        tags = ["game-location"]

        tags.append(f"map-{self.map_name}")
        tags.append(f"tile-{self.tile_type}")

        if self.strategic_importance > 0.8:
            tags.append("high-importance")

        return tags


@dataclass
class TileSemantics(MCPDataPattern):
    """
    Represents learned semantics for Pokemon game tiles.

    Tile semantics capture behavioral properties and interaction
    patterns for specific tile types in different contexts.
    """

    visual_characteristics: dict[str, Any] = field(default_factory=dict)
    behavioral_properties: dict[str, Any] = field(default_factory=dict)
    context_dependencies: dict[str, Any] = field(default_factory=dict)
    learning_confidence: float = 0.5
    interaction_history: list[dict[str, Any]] = field(default_factory=list)
    performance_impact: dict[str, float] = field(default_factory=dict)

    def __init__(
        self,
        id: str,
        visual_characteristics: dict[str, Any],
        behavioral_properties: dict[str, Any],
        **kwargs,
    ):
        """Initialize tile semantics."""
        super().__init__(id, f"Tile-{id}", "tile", kwargs.get("confidence", 1.0))
        self.visual_characteristics = visual_characteristics
        self.behavioral_properties = behavioral_properties
        self.context_dependencies = kwargs.get("context_dependencies", {})
        self.learning_confidence = kwargs.get("learning_confidence", 0.5)
        self.interaction_history = kwargs.get("interaction_history", [])
        self.performance_impact = kwargs.get("performance_impact", {})

    def _serialize_content(self) -> str:
        """Serialize tile semantics content for MCP storage."""
        data = {
            "id": self.id,
            "name": self.name,
            "data_type": self.data_type,
            "visual_characteristics": self.visual_characteristics,
            "behavioral_properties": self.behavioral_properties,
            "context_dependencies": self.context_dependencies,
            "learning_confidence": self.learning_confidence,
            "interaction_history": self.interaction_history,
            "performance_impact": self.performance_impact,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        return json.dumps(data, indent=2)

    def _get_specific_tags(self) -> list[str]:
        """Get tile semantics specific tags."""
        tags = ["tile-behavior"]

        if self.learning_confidence >= 0.9:
            tags.append("high-confidence")
        elif self.learning_confidence >= 0.7:
            tags.append("medium-confidence")
        else:
            tags.append("low-confidence")

        return tags


class QueryBuilder:
    """
    Builds optimized queries for MCP pattern retrieval.

    Implements query patterns for <100ms performance requirement.
    Follows Single Responsibility Principle for query construction.
    """

    def __init__(self):
        """Initialize query builder."""
        self.default_limit = 50
        self.max_limit = 100
        self.min_confidence_threshold = 0.5

    def find_patterns_by_success_rate(
        self, min_success_rate: float = 0.8, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Find patterns with success rate above threshold.

        Args:
            min_success_rate: Minimum success rate filter
            limit: Maximum results to return

        Returns:
            List of matching patterns
        """
        query_params = self._build_pattern_query({"success_rate": min_success_rate})
        query_params["limit"] = limit or self.default_limit

        result = search_memories(**query_params)
        return result.get("results", []) if result.get("success") else []

    def find_strategies_for_game_state(
        self, game_state: dict[str, Any], limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Find strategies applicable to current game state.

        Args:
            game_state: Current game state context
            limit: Maximum results to return

        Returns:
            List of applicable strategies
        """
        # Build location-based search
        location = game_state.get("location", "")
        pattern = f"pokemon-strategy {location}"

        query_params = {
            "pattern": pattern,
            "limit": limit or self.default_limit,
            "min_confidence": self.min_confidence_threshold,
        }

        result = search_memories(**query_params)
        return result.get("results", []) if result.get("success") else []

    def find_path_between_locations(self, start_location: str, end_location: str) -> list[str]:
        """
        Find navigation path between locations.

        Args:
            start_location: Starting location ID
            end_location: Destination location ID

        Returns:
            List of location IDs forming path
        """
        # Search for location connections
        pattern = f"pokemon-location {start_location} {end_location}"

        query_params = {
            "pattern": pattern,
            "limit": 20,  # Smaller limit for path finding
            "min_confidence": 0.7,
        }

        result = search_memories(**query_params)

        # Simplified pathfinding - return direct connections found
        if result.get("success") and result.get("results"):
            return [start_location, end_location]  # Placeholder implementation

        return []

    def store_pattern(self, pattern: MCPDataPattern) -> dict[str, Any]:
        """
        Store a pattern in the MCP memory system.

        Args:
            pattern: Pattern to store

        Returns:
            Dictionary with success status and memory ID
        """
        try:
            mcp_data = pattern.to_mcp_format()
            result = store_memory(**mcp_data)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_patterns(self, query: str, limit: int | None = None) -> dict[str, Any]:
        """
        Search for patterns using query string.

        Args:
            query: Search query string
            limit: Maximum results to return

        Returns:
            Dictionary with search results
        """
        try:
            query_params = {
                "pattern": query,
                "limit": limit or self.default_limit,
                "min_confidence": self.min_confidence_threshold,
            }
            result = search_memories(**query_params)
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "results": []}

    def _build_pattern_query(self, filters: dict[str, Any]) -> dict[str, Any]:
        """
        Build optimized query parameters for pattern search.

        Args:
            filters: Search criteria

        Returns:
            Dictionary of query parameters
        """
        # Start with base pattern search
        pattern_parts = ["pokemon-pattern"]

        # Add filter-specific terms
        if "success_rate" in filters:
            if filters["success_rate"] >= 0.9:
                pattern_parts.append("high-success")
            elif filters["success_rate"] >= 0.7:
                pattern_parts.append("medium-success")

        return {
            "pattern": " ".join(pattern_parts),
            "min_confidence": self.min_confidence_threshold,
            "limit": self.default_limit,
        }


class RelationshipManager:
    """
    Manages relationships between data patterns.

    Implements Clean Code relationship management with clear
    interfaces for different relationship types.
    """

    RELATIONSHIP_TYPES = {
        "DEPENDS_ON": "dependency",
        "COMBINES_WITH": "combination",
        "CONFLICTS_WITH": "conflict",
        "EVOLVES_TO": "evolution",
        "SUCCEEDS": "succession",
        "USES_PATTERN": "usage",
        "FALLBACK_TO": "fallback",
        "OPTIMIZES": "optimization",
        "LEADS_TO": "transition",
        "LOCATED_AT": "spatial",
    }

    def create_dependency(
        self, pattern_id: str, depends_on_id: str, relationship_type: str, confidence: float
    ) -> dict[str, Any]:
        """
        Create dependency relationship between patterns.

        Args:
            pattern_id: Source pattern ID
            depends_on_id: Target pattern ID
            relationship_type: Type of relationship
            confidence: Confidence in relationship

        Returns:
            Result of relationship creation
        """
        return link_memories(
            node1_id=pattern_id,
            node2_id=depends_on_id,
            relation_type=relationship_type,
            confidence=confidence,
        )

    def link_strategy_to_pattern(
        self, strategy_id: str, pattern_id: str, usage_order: int, confidence: float
    ) -> dict[str, Any]:
        """
        Link strategy to its constituent patterns.

        Args:
            strategy_id: Strategy ID
            pattern_id: Pattern ID
            usage_order: Order of pattern in strategy
            confidence: Confidence in relationship

        Returns:
            Result of relationship creation
        """
        return link_memories(
            node1_id=strategy_id,
            node2_id=pattern_id,
            relation_type="USES_PATTERN",
            confidence=confidence,
        )

    def get_dependencies(self, pattern_id: str, depth: int = 1) -> list[dict[str, Any]]:
        """
        Get all dependencies for a pattern.

        Args:
            pattern_id: Pattern to find dependencies for
            depth: Traversal depth

        Returns:
            List of dependency relationships
        """
        result = get_relationships(node_id=pattern_id, relation_type="DEPENDS_ON", depth=depth)

        return result.get("relationships", []) if result.get("success") else []


class PatternValidator:
    """
    Validates data patterns and relationships.

    Implements comprehensive validation following Clean Code principles
    with clear error reporting and validation rules.
    """

    def validate_pattern(self, pattern: MCPDataPattern) -> dict[str, Any]:
        """
        Validate a data pattern.

        Args:
            pattern: Pattern to validate

        Returns:
            Dictionary with validation results and errors
        """
        errors = []

        # Basic validation
        if not pattern.id:
            errors.append("Pattern ID cannot be empty")

        if not pattern.name:
            errors.append("Pattern name cannot be empty")

        if not (0.0 <= pattern.confidence <= 1.0):
            errors.append("Confidence must be between 0.0 and 1.0")

        # Pattern-specific validation
        if isinstance(pattern, PokemonPattern):
            errors.extend(self._validate_pokemon_pattern(pattern))
        elif isinstance(pattern, PokemonStrategy):
            errors.extend(self._validate_pokemon_strategy(pattern))

        return {"valid": len(errors) == 0, "errors": errors}

    def validate_relationship(
        self, source_id: str, target_id: str, relationship_type: str, confidence: float
    ) -> dict[str, Any]:
        """
        Validate a relationship between patterns.

        Args:
            source_id: Source pattern ID
            target_id: Target pattern ID
            relationship_type: Type of relationship
            confidence: Relationship confidence

        Returns:
            Dictionary with validation results
        """
        errors = []

        if source_id == target_id:
            errors.append("Circular relationship not allowed")

        if relationship_type not in RelationshipManager.RELATIONSHIP_TYPES:
            errors.append(f"Invalid relationship type: {relationship_type}")

        if not (0.0 <= confidence <= 1.0):
            errors.append("Confidence must be between 0.0 and 1.0")

        return {"valid": len(errors) == 0, "errors": errors}

    def validate_performance_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Validate performance metrics.

        Args:
            metrics: Performance metrics to validate

        Returns:
            Dictionary with validation results
        """
        errors = []

        if "success_rate" in metrics:
            if not (0.0 <= metrics["success_rate"] <= 1.0):
                errors.append("Success rate must be between 0.0 and 1.0")

        if "average_time" in metrics:
            if metrics["average_time"] < 0:
                errors.append("Average time cannot be negative")

        if "completion_rate" in metrics:
            if not (0.0 <= metrics["completion_rate"] <= 1.0):
                errors.append("Completion rate must be between 0.0 and 1.0")

        return {"valid": len(errors) == 0, "errors": errors}

    def _validate_pokemon_pattern(self, pattern: PokemonPattern) -> list[str]:
        """Validate Pokemon-specific pattern requirements."""
        errors = []

        if len(pattern.input_sequence) == 0:
            errors.append("Input sequence cannot be empty")

        if not (0.0 <= pattern.success_rate <= 1.0):
            errors.append("Success rate must be between 0.0 and 1.0")

        return errors

    def _validate_pokemon_strategy(self, strategy: PokemonStrategy) -> list[str]:
        """Validate Pokemon-specific strategy requirements."""
        errors = []

        if len(strategy.pattern_sequence) == 0:
            errors.append("Strategy must include at least one pattern")

        if not (0.0 <= strategy.success_rate <= 1.0):
            errors.append("Success rate must be between 0.0 and 1.0")

        return errors


class EvolutionTracker:
    """
    Tracks pattern evolution and versioning.

    Implements Clean Code evolution management with clear
    interfaces for tracking improvements and version history.
    """

    def evolve_pattern(
        self,
        original_pattern: MCPDataPattern,
        improved_pattern: MCPDataPattern,
        evolution_type: str,
    ) -> dict[str, Any]:
        """
        Create evolution relationship between patterns.

        Args:
            original_pattern: Original pattern
            improved_pattern: Improved pattern
            evolution_type: Type of evolution (optimization, fix, etc.)

        Returns:
            Result of evolution tracking
        """
        # Validate evolution
        if not self.is_valid_evolution(original_pattern, improved_pattern):
            return {"success": False, "error": "Invalid evolution"}

        # Store improved pattern
        store_result = store_memory(**improved_pattern.to_mcp_format())

        if not store_result.get("success"):
            return {"success": False, "error": "Failed to store improved pattern"}

        # Create evolution relationship
        link_result = link_memories(
            node1_id=original_pattern.id,
            node2_id=improved_pattern.id,
            relation_type="EVOLVES_TO",
            confidence=0.9,
        )

        return {
            "success": True,
            "evolution_id": store_result.get("memory_id"),
            "relationship_created": link_result.get("success", False),
        }

    def get_evolution_history(self, pattern_id: str) -> list[dict[str, Any]]:
        """
        Get complete evolution history for a pattern.

        Args:
            pattern_id: Pattern to get history for

        Returns:
            List of evolution relationships
        """
        result = get_relationships(
            node_id=pattern_id,
            relation_type="EVOLVES_TO",
            depth=3,  # Allow multiple evolution generations
        )

        return result.get("relationships", []) if result.get("success") else []

    def is_valid_evolution(self, original: MCPDataPattern, improved: MCPDataPattern) -> bool:
        """
        Check if pattern evolution represents actual improvement.

        Args:
            original: Original pattern
            improved: Improved pattern

        Returns:
            True if evolution is valid improvement
        """
        # Basic validation - improved pattern should be valid
        if not improved.is_valid():
            return False

        # Pattern-specific evolution validation
        if isinstance(original, PokemonPattern) and isinstance(improved, PokemonPattern):
            return improved.success_rate >= original.success_rate

        if isinstance(original, PokemonStrategy) and isinstance(improved, PokemonStrategy):
            return improved.success_rate >= original.success_rate

        return True  # Default to allowing evolution


# Export main classes for clean imports
__all__ = [
    "MCPDataPattern",
    "PokemonPattern",
    "PokemonStrategy",
    "PokemonGameState",
    "PokemonCheckpoint",
    "PokemonLocation",
    "TileSemantics",
    "QueryBuilder",
    "RelationshipManager",
    "PatternValidator",
    "EvolutionTracker",
]
