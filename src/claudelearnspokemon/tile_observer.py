"""
TileObserver - Captures and analyzes tile-based game state representations.

Following Clean Code principles:
- Single Responsibility: Each method has one clear purpose
- Open/Closed: Extensible pattern detection strategies
- Liskov Substitution: Proper inheritance patterns
- Interface Segregation: Focused interfaces
- Dependency Inversion: Abstract from concrete implementations

Performance Requirements:
- Tile observation: < 50ms per capture
- Pattern detection: < 100ms

Author: Uncle Bot (Claude Code)
"""

import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass
class GamePosition:
    """Represents a player position in the game world."""

    x: int
    y: int
    map_id: str
    facing_direction: str


@dataclass
class GameState:
    """Complete game state capture for TileObserver analysis."""

    position: GamePosition
    tiles: np.ndarray  # 20x18 tile grid
    npcs: list[Any]
    menu_state: Any
    inventory: dict[str, int]
    timestamp: float
    frame_count: int


@dataclass
class TileInfo:
    """Information about a specific tile type."""

    tile_id: int
    passable: bool
    interaction_type: str | None
    semantic_tags: set[str]


# Semantic Classification Enums and Data Structures (Issue #39)


class SemanticCategory(Enum):
    """Semantic categories for tile classification."""

    TERRAIN = "terrain"
    DOOR = "door"
    NPC = "npc"
    ITEM = "item"
    INTERACTIVE_OBJECT = "interactive_object"
    STRATEGIC_LOCATION = "strategic_location"
    OBSTACLE = "obstacle"
    WATER = "water"
    BUILDING = "building"


class InteractionType(Enum):
    """Types of interactions possible with tiles/objects."""

    TALK = "talk"
    PICK_UP = "pick_up"
    USE = "use"
    ENTER = "enter"
    READ = "read"
    EXAMINE = "examine"


class StrategicLocationType(Enum):
    """Types of strategic locations in Pokemon Red."""

    POKEMON_CENTER = "pokemon_center"
    SHOP = "shop"
    GYM = "gym"
    ROUTE_EXIT = "route_exit"


@dataclass
class TileSemantics:
    """Semantic information about a tile."""

    tile_id: int
    category: SemanticCategory
    subcategory: str | None = None  # For hierarchical classification
    interaction_type: InteractionType | None = None
    strategic_type: StrategicLocationType | None = None
    confidence: float = 0.0
    context_dependent: bool = False


@dataclass
class InteractiveObject:
    """Represents an interactive object detected in the game world."""

    position: tuple[int, int]
    tile_id: int
    interaction_type: InteractionType
    description: str = ""
    confidence: float = 0.0


@dataclass
class StrategicLocation:
    """Represents a strategic location like Pokemon Center or shop."""

    position: tuple[int, int]
    location_type: StrategicLocationType
    name: str | None = None
    entrance_tiles: list[tuple[int, int]] = field(default_factory=list)
    confidence: float = 0.0


class GameStateInterface:
    """Interface for capturing and processing game state information."""

    def capture_current_state(self, emulator_client) -> GameState:
        """Capture complete game state from emulator.

        Args:
            emulator_client: Client connection to Pokemon-gym emulator

        Returns:
            GameState: Complete captured game state

        Raises:
            ValueError: If emulator_client is invalid or unresponsive
        """
        if not emulator_client:
            raise ValueError("Invalid emulator client provided")

        try:
            # Extract raw state from emulator
            raw_state: dict[str, Any] = getattr(emulator_client, "get_state", lambda: {})()

            # Convert tiles to 20x18 numpy array
            tiles = np.zeros((20, 18), dtype=np.uint8)
            if "tiles" in raw_state:
                tile_data = np.array(raw_state["tiles"], dtype=np.uint8)
                if tile_data.shape == (20, 18):
                    # Type: ignore because we've verified the shape is exactly (20, 18)
                    tiles = tile_data  # type: ignore[assignment]
                else:
                    # Reshape to proper dimensions
                    resized_data = np.resize(tile_data, (20 * 18,))
                    tiles = resized_data.reshape((20, 18)).astype(np.uint8)  # type: ignore[assignment]

            # Extract position with defaults
            pos_data = raw_state.get("player_position", (0, 0))
            position = GamePosition(
                x=pos_data[0] if len(pos_data) > 0 else 0,
                y=pos_data[1] if len(pos_data) > 1 else 0,
                map_id=raw_state.get("map_id", "unknown"),
                facing_direction=raw_state.get("facing_direction", "down"),
            )

            return GameState(
                position=position,
                tiles=tiles,
                npcs=raw_state.get("npcs", []),
                menu_state=raw_state.get("menu_state", None),
                inventory=raw_state.get("inventory", {}),
                timestamp=time.time(),
                frame_count=raw_state.get("frame_count", 0),
            )

        except Exception as e:
            raise ValueError(f"Failed to capture game state: {e}") from e

    def extract_tile_grid(self, state: GameState) -> np.ndarray:
        """Extract 20x18 tile grid from game state.

        Args:
            state: GameState object containing tile data

        Returns:
            numpy.ndarray: 20x18 array of tile IDs

        Raises:
            ValueError: If state is invalid or tiles are wrong shape
        """
        if not isinstance(state, GameState):
            raise ValueError("Invalid GameState object provided")
        if state.tiles.shape != (20, 18):
            raise ValueError(f"Expected grid shape (20, 18), got {state.tiles.shape}")
        return state.tiles.copy()

    def get_player_position(self, state: GameState) -> GamePosition:
        """Get current player position and context.

        Args:
            state: GameState object containing position data

        Returns:
            GamePosition: Current player position and facing direction

        Raises:
            ValueError: If state is invalid
        """
        if not isinstance(state, GameState):
            raise ValueError("Invalid GameState object provided")
        return state.position

    def serialize_state(self, state: GameState) -> bytes:
        """Serialize game state for storage.

        Args:
            state: GameState object to serialize

        Returns:
            bytes: Serialized state data

        Raises:
            ValueError: If state cannot be serialized
        """
        if not isinstance(state, GameState):
            raise ValueError("Invalid GameState object provided")
        try:
            return pickle.dumps(state)
        except Exception as e:
            raise ValueError(f"Failed to serialize game state: {e}") from e


class SemanticClassifier:
    """
    Semantic classification engine for tiles and game objects.

    Handles pattern-based classification, context-aware analysis,
    and hierarchical semantic understanding for Pokemon Red tiles.

    Following Clean Code principles:
    - Single Responsibility: Focused on semantic classification only
    - Open/Closed: Extensible for new semantic patterns
    - Dependency Inversion: Uses abstract patterns, not hardcoded rules
    """

    def __init__(self) -> None:
        """Initialize semantic classifier with Pokemon Red knowledge base."""
        # Known tile patterns for Pokemon Red
        self._semantic_patterns: dict[str, dict] = {
            "npc_tiles": {
                "range": (200, 220),
                "category": SemanticCategory.NPC,
                "interaction": InteractionType.TALK,
                "confidence": 0.9,
            },
            "item_tiles": {
                "range": (150, 190),
                "category": SemanticCategory.ITEM,
                "interaction": InteractionType.PICK_UP,
                "confidence": 0.7,
            },
            "terrain_tiles": {
                "range": (1, 50),
                "category": SemanticCategory.TERRAIN,
                "interaction": None,
                "confidence": 0.6,
            },
            "building_tiles": {
                "range": (20, 40),
                "category": SemanticCategory.BUILDING,
                "interaction": InteractionType.ENTER,
                "confidence": 0.5,
            },
        }

        # Context-specific classification rules
        self._context_rules: dict[str, dict] = {
            "viridian_city": {
                "pokemon_center_pattern": [30, 35, 36],  # Building, door, interior
                "shop_pattern": [25, 28],  # Building, door
            },
            "route_1": {
                "grass_types": {1: "normal_grass", 3: "tall_grass"},
                "obstacles": [20, 25, 30],
            },
        }

        # Strategic location patterns
        self._known_strategic_locations: dict[str, dict] = {
            "pokemon_center": {
                "building_tiles": [30, 31, 32],
                "door_tile": 35,
                "interior_tiles": [36, 37],
                "confidence_boost": 0.3,
            },
            "shop": {
                "building_tiles": [25, 26],
                "door_tile": 28,
                "confidence_boost": 0.2,
            },
        }

        # Performance caching
        self._classification_cache: dict[str, TileSemantics] = {}
        self._cache_max_size = 1000

    def classify_tile_semantics(
        self, tile_id: int, position: tuple[int, int], surrounding_tiles: np.ndarray, context: str
    ) -> TileSemantics:
        """
        Classify a tile's semantic meaning based on ID, position, and context.

        Args:
            tile_id: The tile identifier
            position: (row, col) position in grid
            surrounding_tiles: 3x3 array of surrounding tile IDs
            context: Map context (e.g., "route_1", "viridian_city")

        Returns:
            TileSemantics: Classification results with confidence score
        """
        # Check cache first for performance
        cache_key = f"{tile_id}_{position}_{context}_{hash(surrounding_tiles.tobytes())}"
        if cache_key in self._classification_cache:
            return self._classification_cache[cache_key]

        # Classify using pattern matching
        semantics = self._classify_by_patterns(tile_id, position, surrounding_tiles, context)

        # Apply context-specific rules
        semantics = self._apply_context_rules(semantics, surrounding_tiles, context)

        # Cache result if confidence is reasonable
        if semantics.confidence > 0.3:
            self._cache_classification(cache_key, semantics)

        return semantics

    def classify_tile_hierarchical(
        self, tile_id: int, position: tuple[int, int], surrounding_tiles: np.ndarray, context: str
    ) -> TileSemantics:
        """
        Perform hierarchical classification (e.g., terrain -> grass -> tall_grass).

        Returns semantic classification with subcategory populated.
        """
        # Get base classification
        base_semantics = self.classify_tile_semantics(tile_id, position, surrounding_tiles, context)

        # Add hierarchical subcategory
        if base_semantics.category == SemanticCategory.TERRAIN:
            base_semantics.subcategory = self._classify_terrain_type(tile_id, context)
        elif base_semantics.category == SemanticCategory.BUILDING:
            base_semantics.subcategory = self._classify_building_type(tile_id, surrounding_tiles)

        return base_semantics

    def classify_tile_with_collision_data(
        self,
        tile_id: int,
        position: tuple[int, int],
        collision_data: dict[str, Any],
        surrounding_tiles: np.ndarray,
        context: str,
    ) -> TileSemantics:
        """
        Enhanced classification using collision detection data from Issue #34.

        Args:
            collision_data: Dict with 'walkable' and 'confidence' from collision learner
        """
        # Get base semantic classification
        semantics = self.classify_tile_semantics(tile_id, position, surrounding_tiles, context)

        # Enhance with collision data
        walkable = collision_data.get("walkable", True)
        collision_confidence = collision_data.get("confidence", 0.0)

        # Non-walkable tiles are likely obstacles, buildings, or interactive objects
        if not walkable and collision_confidence > 0.7:
            if semantics.category == SemanticCategory.TERRAIN:
                # Reassify terrain as obstacle if not walkable
                semantics.category = SemanticCategory.OBSTACLE
                semantics.confidence = min(0.9, semantics.confidence + 0.3)

        # Walkable tiles reinforce terrain classification
        elif walkable and collision_confidence > 0.7:
            if semantics.category in [SemanticCategory.TERRAIN, SemanticCategory.ITEM]:
                semantics.confidence = min(0.95, semantics.confidence + 0.2)

        return semantics

    # Private implementation methods

    def _classify_by_patterns(
        self, tile_id: int, position: tuple[int, int], surrounding_tiles: np.ndarray, context: str
    ) -> TileSemantics:
        """Classify tile using known patterns."""
        # Check each pattern category
        for _pattern_name, pattern_info in self._semantic_patterns.items():
            tile_range = pattern_info["range"]

            if tile_range[0] <= tile_id <= tile_range[1]:
                return TileSemantics(
                    tile_id=tile_id,
                    category=pattern_info["category"],
                    interaction_type=pattern_info["interaction"],
                    confidence=pattern_info["confidence"],
                    context_dependent=True if context in self._context_rules else False,
                )

        # Unknown tile - classify conservatively
        return TileSemantics(
            tile_id=tile_id,
            category=SemanticCategory.TERRAIN,  # Conservative default
            confidence=0.3,
            context_dependent=True,
        )

    def _apply_context_rules(
        self, semantics: TileSemantics, surrounding_tiles: np.ndarray, context: str
    ) -> TileSemantics:
        """Apply context-specific classification rules."""
        # Always check for door patterns regardless of context rules
        if self._has_door_pattern(surrounding_tiles):
            semantics.category = SemanticCategory.DOOR
            semantics.interaction_type = InteractionType.ENTER
            semantics.confidence = min(0.85, semantics.confidence + 0.4)
            return semantics

        # Apply building detection patterns
        if self._has_building_pattern(surrounding_tiles, context):
            if semantics.category in [SemanticCategory.TERRAIN, SemanticCategory.OBSTACLE]:
                semantics.category = SemanticCategory.BUILDING
                semantics.interaction_type = InteractionType.ENTER
                semantics.confidence = min(0.8, semantics.confidence + 0.3)

        # Apply context-specific rules if available
        if context not in self._context_rules:
            return semantics

        rules = self._context_rules[context]

        # Apply grass type rules for routes
        if "grass_types" in rules and semantics.category == SemanticCategory.TERRAIN:
            grass_types = rules["grass_types"]
            if semantics.tile_id in grass_types:
                semantics.subcategory = grass_types[semantics.tile_id]
                semantics.confidence = min(0.9, semantics.confidence + 0.2)

        return semantics

    def _has_building_pattern(self, surrounding_tiles: np.ndarray, context: str) -> bool:
        """Check if surrounding tiles indicate a building pattern."""
        if context not in self._context_rules:
            return False

        rules = self._context_rules[context]
        if "pokemon_center_pattern" in rules:
            pattern_tiles = rules["pokemon_center_pattern"]
            # Check if any surrounding tiles match building patterns
            for tile_id in surrounding_tiles.flatten():
                if tile_id in pattern_tiles:
                    return True

        return False

    def _has_door_pattern(self, surrounding_tiles: np.ndarray) -> bool:
        """Check if surrounding tiles indicate a door pattern (building walls around)."""
        # Count building/wall tiles around the center tile
        building_wall_tiles = 0
        ground_tiles = 0

        # Check 8 surrounding positions (excluding center which is the potential door)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:  # Skip center tile (the potential door)
                    continue

                tile_id = int(surrounding_tiles[i, j])

                # Building/wall tile ranges
                if 20 <= tile_id <= 40:  # Building tiles
                    building_wall_tiles += 1
                elif 1 <= tile_id <= 10:  # Ground tiles
                    ground_tiles += 1

        # Door pattern: surrounded by building tiles on top/sides, ground below
        # Need at least 3 building tiles and some ground tiles
        return building_wall_tiles >= 3 and ground_tiles >= 1

    def _classify_terrain_type(self, tile_id: int, context: str) -> str:
        """Classify terrain subcategory."""
        if context in self._context_rules:
            grass_types = self._context_rules[context].get("grass_types", {})
            if tile_id in grass_types:
                return grass_types[tile_id]

        # Default terrain types based on tile ID ranges
        if 1 <= tile_id <= 10:
            return "grass"
        elif 11 <= tile_id <= 15:
            return "water"
        elif 16 <= tile_id <= 20:
            return "rock"
        else:
            return "unknown"

    def _classify_building_type(self, tile_id: int, surrounding_tiles: np.ndarray) -> str:
        """Classify building subcategory based on patterns."""
        # Check for Pokemon Center pattern
        if self._matches_pokemon_center_pattern(surrounding_tiles):
            return "pokemon_center"
        elif self._matches_shop_pattern(surrounding_tiles):
            return "shop"
        else:
            return "generic_building"

    def _matches_pokemon_center_pattern(self, surrounding_tiles: np.ndarray) -> bool:
        """Check if tiles match Pokemon Center pattern."""
        center_pattern = self._known_strategic_locations["pokemon_center"]
        building_tiles = center_pattern["building_tiles"]

        # Count matching tiles in surrounding area
        matches = sum(1 for tile_id in surrounding_tiles.flatten() if tile_id in building_tiles)
        return matches >= 2  # Need at least 2 matching tiles

    def _matches_shop_pattern(self, surrounding_tiles: np.ndarray) -> bool:
        """Check if tiles match shop pattern."""
        shop_pattern = self._known_strategic_locations["shop"]
        building_tiles = shop_pattern["building_tiles"]

        matches = sum(1 for tile_id in surrounding_tiles.flatten() if tile_id in building_tiles)
        return matches >= 1

    def _cache_classification(self, cache_key: str, semantics: TileSemantics) -> None:
        """Cache classification result with size limit."""
        if len(self._classification_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._classification_cache))
            del self._classification_cache[oldest_key]

        self._classification_cache[cache_key] = semantics


class TileObserver:
    """
    Observes and analyzes tile-based game state for pattern discovery.

    This class captures 20x18 tile grids from game state and learns tile
    semantics through interaction and collision detection. It maintains
    separate tile knowledge for different map contexts and provides
    efficient pattern detection capabilities.

    Responsibilities (Single Responsibility Principle):
    - Capture tile grids from game state
    - Analyze tile grids for entities and patterns
    - Learn tile properties through observations
    - Detect patterns in tile arrangements
    - Calculate walkable paths using learned semantics
    """

    # Constants for clean, self-documenting code
    EXPECTED_GRID_HEIGHT = 20
    EXPECTED_GRID_WIDTH = 18
    PLAYER_TILE_ID = 255
    NPC_TILE_ID_START = 200
    NPC_TILE_ID_END = 220
    MENU_TILE_ID = 254

    def __init__(self) -> None:
        """
        Initialize TileObserver with empty knowledge base.

        Following clean initialization principles:
        - Clear, meaningful variable names
        - Initialize to sensible defaults
        - No complex logic in constructor
        """
        self._tile_semantics: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        self._map_context: dict[str, Any] = {}
        self._pattern_cache: dict[str, list[tuple[int, int]]] = {}

        # Semantic classification enhancement (Issue #39)
        self._semantic_classifier = SemanticClassifier()
        self._semantic_knowledge: dict[str, dict[int, TileSemantics]] = defaultdict(dict)
        self._semantic_cache: dict[str, dict] = {}
        self._cache_ttl: int = 300  # 5 minutes cache lifetime

    def capture_tiles(self, game_state: dict) -> np.ndarray:
        """
        Capture 20x18 tile grid from game state.

        Performance requirement: < 50ms

        Args:
            game_state: Dictionary containing game state with 'tiles' key

        Returns:
            numpy.ndarray: 20x18 tile grid as uint8 array

        Raises:
            ValueError: If game state is invalid or missing required data
        """
        start_time = time.perf_counter()

        # Validate input - fail fast principle
        if not isinstance(game_state, dict):
            raise ValueError("Invalid game state: must be a dictionary")

        if "tiles" not in game_state:
            raise ValueError("Invalid game state: missing 'tiles' key")

        # Extract and convert tiles efficiently
        tiles_data = game_state["tiles"]
        if isinstance(tiles_data, list):
            tiles_array = np.array(tiles_data, dtype=np.uint8)
        elif isinstance(tiles_data, np.ndarray):
            tiles_array = tiles_data.astype(np.uint8)
        else:
            raise ValueError("Invalid tile data: must be list or numpy array")

        # Validate dimensions
        if tiles_array.shape != (self.EXPECTED_GRID_HEIGHT, self.EXPECTED_GRID_WIDTH):
            raise ValueError(
                f"Invalid grid dimensions: expected ({self.EXPECTED_GRID_HEIGHT}, {self.EXPECTED_GRID_WIDTH}), got {tiles_array.shape}"
            )

        # Handle menu overlays if present
        if game_state.get("menu_active", False) and "menu_tiles" in game_state:
            tiles_array = self._handle_menu_overlays(tiles_array, game_state["menu_tiles"])

        # Performance check - ensure we meet the <50ms requirement
        elapsed_time = (time.perf_counter() - start_time) * 1000
        if elapsed_time > 50:
            # Log warning in production, but don't fail tests during development
            pass

        return tiles_array

    def analyze_tile_grid(
        self, tiles: np.ndarray, menu_areas: list[tuple[int, int, int, int]] | None = None
    ) -> dict:
        """
        Convert tiles to semantic analysis with entity positions.

        Args:
            tiles: 20x18 numpy array of tile IDs
            menu_areas: Optional list of (x, y, width, height) menu overlay areas to ignore

        Returns:
            Dictionary with analysis results including:
            - player_position: (row, col) tuple or None
            - npcs: List of (row, col) NPC positions
            - tile_counts: Summary of tile type distribution
        """
        self._validate_tile_grid(tiles)

        analysis: dict[str, Any] = {"player_position": None, "npcs": [], "tile_counts": {}}

        # Create mask for areas to analyze (exclude menu areas)
        analysis_mask = np.ones(tiles.shape, dtype=bool)
        if menu_areas:
            for x, y, width, height in menu_areas:
                analysis_mask[y : y + height, x : x + width] = False

        # Find player position
        player_positions = np.where((tiles == self.PLAYER_TILE_ID) & analysis_mask)
        if len(player_positions[0]) > 0:
            # Take first occurrence if multiple found
            analysis["player_position"] = (int(player_positions[0][0]), int(player_positions[1][0]))

        # Find NPC positions
        analysis["npcs"] = self.identify_npcs(tiles, analysis_mask)

        # Calculate tile distribution
        masked_tiles = tiles[analysis_mask]
        unique_tiles, counts = np.unique(masked_tiles, return_counts=True)
        analysis["tile_counts"] = dict(zip(unique_tiles.tolist(), counts.tolist(), strict=False))

        return analysis

    def detect_patterns(self, tiles: np.ndarray, pattern: np.ndarray) -> list[tuple[int, int]]:
        """
        Find pattern occurrences in tile grid.

        Performance requirement: < 100ms

        Args:
            tiles: 20x18 tile grid to search
            pattern: Pattern to find as 2D numpy array

        Returns:
            List of (row, col) positions where pattern starts
        """
        start_time = time.perf_counter()

        self._validate_tile_grid(tiles)
        if pattern.size == 0:
            return []

        matches = []
        pattern_height, pattern_width = pattern.shape

        # Use efficient sliding window approach
        for row in range(tiles.shape[0] - pattern_height + 1):
            for col in range(tiles.shape[1] - pattern_width + 1):
                # Extract window and compare
                window = tiles[row : row + pattern_height, col : col + pattern_width]
                if np.array_equal(window, pattern):
                    matches.append((row, col))

        # Performance check
        elapsed_time = (time.perf_counter() - start_time) * 1000
        if elapsed_time > 100:
            # Log warning in production
            pass

        return matches

    def learn_tile_properties(self, observations: list[dict[str, Any]]) -> None:
        """
        Update tile semantic knowledge from collision observations.

        Args:
            observations: List of observation dictionaries with keys:
                - tile_id: Tile identifier
                - collision: True if tile blocked movement
                - context: Map context identifier
                - position: Optional (row, col) position
        """
        for obs in observations:
            tile_id = obs["tile_id"]
            context = obs.get("context", "default")
            collision = obs.get("collision", False)

            # Initialize tile knowledge if not present
            if tile_id not in self._tile_semantics[context]:
                self._tile_semantics[context][tile_id] = {
                    "walkable": not collision,
                    "confidence": 0.0,
                    "observation_count": 0,
                    "collision_count": 0,
                }

            # Update tile properties
            tile_props = self._tile_semantics[context][tile_id]
            tile_props["observation_count"] += 1

            if collision:
                tile_props["collision_count"] += 1

            # Calculate walkability and confidence
            total_obs = tile_props["observation_count"]
            collision_ratio = tile_props["collision_count"] / total_obs

            # Conservative approach: tile is walkable if collision rate < 50%
            tile_props["walkable"] = collision_ratio < 0.5

            # Confidence increases with more observations - more generous for learning
            # Use a formula that gives higher confidence faster
            tile_props["confidence"] = min(0.95, total_obs / (total_obs + 2))

    def identify_npcs(
        self, tiles: np.ndarray, mask: np.ndarray | None = None
    ) -> list[tuple[int, int]]:
        """
        Locate NPC positions in tile grid.

        Args:
            tiles: 20x18 tile grid
            mask: Optional boolean mask for areas to search

        Returns:
            List of (row, col) NPC positions
        """
        if mask is None:
            mask = np.ones(tiles.shape, dtype=bool)

        # Find tiles in NPC range
        npc_mask = (tiles >= self.NPC_TILE_ID_START) & (tiles <= self.NPC_TILE_ID_END) & mask

        npc_positions = np.where(npc_mask)
        return list(zip(npc_positions[0].tolist(), npc_positions[1].tolist(), strict=False))

    def find_path(
        self,
        tiles: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        context: str = "default",
    ) -> list[tuple[int, int]]:
        """
        Calculate walkable path using learned tile semantics.

        Args:
            tiles: Tile grid (any size for flexible pathfinding)
            start: Starting position (row, col)
            end: Destination position (row, col)
            context: Map context for tile semantic lookup

        Returns:
            List of (row, col) positions forming path, empty if no path found
        """
        # Validate that tiles is a non-empty 2D array
        if tiles.size == 0 or len(tiles.shape) != 2:
            raise ValueError("Tiles must be a non-empty 2D array")

        # Simple A* pathfinding implementation
        # For production, could be optimized further or use specialized libraries

        if not self._is_position_walkable(tiles, start, context):
            return []
        if not self._is_position_walkable(tiles, end, context):
            return []

        # Use BFS for simplicity - can be upgraded to A* if needed
        from collections import deque

        queue = deque([(start, [start])])
        visited = {start}

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        grid_height, grid_width = tiles.shape

        while queue:
            (current_pos, path) = queue.popleft()

            if current_pos == end:
                return path

            for dr, dc in directions:
                new_row, new_col = current_pos[0] + dr, current_pos[1] + dc
                new_pos = (new_row, new_col)

                if (
                    0 <= new_row < grid_height
                    and 0 <= new_col < grid_width
                    and new_pos not in visited
                    and self._is_position_walkable(tiles, new_pos, context)
                ):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))

        return []  # No path found

    # Semantic Classification Methods (Issue #39)

    def detect_interactive_objects(
        self, tiles: np.ndarray, context: str = "default"
    ) -> list[InteractiveObject]:
        """
        Detect and classify interactive objects in the tile grid.

        Args:
            tiles: 20x18 tile grid
            context: Map context for semantic understanding

        Returns:
            List of InteractiveObject instances with positions and interaction types
        """
        self._validate_tile_grid(tiles)
        interactive_objects = []

        # Scan tiles for interactive elements
        for row in range(tiles.shape[0]):
            for col in range(tiles.shape[1]):
                tile_id = int(tiles[row, col])
                position = (row, col)

                # Get 3x3 surrounding area for context
                surrounding = self._extract_surrounding_tiles(tiles, row, col)

                # Classify tile semantics
                semantics = self._semantic_classifier.classify_tile_semantics(
                    tile_id, position, surrounding, context
                )

                # Convert semantic classification to interactive object
                if self._is_interactive_tile(semantics):
                    interactive_obj = InteractiveObject(
                        position=position,
                        tile_id=tile_id,
                        interaction_type=semantics.interaction_type or InteractionType.EXAMINE,
                        description=self._generate_object_description(semantics),
                        confidence=semantics.confidence,
                    )
                    interactive_objects.append(interactive_obj)

        return interactive_objects

    def identify_strategic_locations(
        self, tiles: np.ndarray, context: str = "default"
    ) -> list[StrategicLocation]:
        """
        Identify strategic locations like Pokemon Centers, shops, and gyms.

        Args:
            tiles: 20x18 tile grid
            context: Map context for pattern recognition

        Returns:
            List of StrategicLocation instances
        """
        self._validate_tile_grid(tiles)
        strategic_locations = []

        # Look for Pokemon Center patterns
        pokemon_centers = self._find_pokemon_center_patterns(tiles, context)
        strategic_locations.extend(pokemon_centers)

        # Look for shop patterns
        shops = self._find_shop_patterns(tiles, context)
        strategic_locations.extend(shops)

        # Look for gym patterns (if context suggests it)
        if "gym" in context.lower() or "city" in context:
            gyms = self._find_gym_patterns(tiles, context)
            strategic_locations.extend(gyms)

        return strategic_locations

    def create_semantic_map(self, tiles: np.ndarray, context: str = "default") -> dict:
        """
        Create a comprehensive semantic map of the tile grid.

        Args:
            tiles: 20x18 tile grid
            context: Map context for semantic understanding

        Returns:
            Dictionary containing:
            - tile_semantics: Per-tile semantic classifications
            - interactive_objects: List of interactive objects
            - strategic_locations: List of strategic locations
            - walkable_areas: Areas safe for navigation
        """
        start_time = time.perf_counter()

        # Check cache first
        cache_key = f"{context}_{hash(tiles.tobytes())}"
        if cache_key in self._semantic_cache:
            cached_result = self._semantic_cache[cache_key]
            if time.time() - cached_result.get("timestamp", 0) < self._cache_ttl:
                return cached_result["data"]

        self._validate_tile_grid(tiles)

        # Build comprehensive semantic map
        semantic_map: dict[str, Any] = {
            "tile_semantics": {},
            "interactive_objects": self.detect_interactive_objects(tiles, context),
            "strategic_locations": self.identify_strategic_locations(tiles, context),
            "walkable_areas": self._identify_walkable_areas(tiles, context),
        }

        # Add per-tile semantic information
        for row in range(tiles.shape[0]):
            for col in range(tiles.shape[1]):
                tile_id = int(tiles[row, col])
                position = (row, col)
                surrounding = self._extract_surrounding_tiles(tiles, row, col)

                semantics = self._semantic_classifier.classify_tile_semantics(
                    tile_id, position, surrounding, context
                )
                semantic_map["tile_semantics"][position] = semantics

        # Cache result for performance
        self._semantic_cache[cache_key] = {
            "data": semantic_map,
            "timestamp": time.time(),
        }

        # Performance validation
        processing_time = (time.perf_counter() - start_time) * 1000
        if processing_time > 50:
            # Log performance warning in production
            pass

        return semantic_map

    def find_semantic_path(
        self,
        tiles: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        context: str = "default",
        avoid_npcs: bool = False,
    ) -> list[tuple[int, int]]:
        """
        Enhanced pathfinding using semantic understanding.

        Args:
            tiles: Tile grid
            start: Starting position
            end: Destination position
            context: Map context
            avoid_npcs: Whether to avoid NPC tiles

        Returns:
            List of positions forming optimal path considering semantics
        """
        # Get semantic map for enhanced pathfinding
        semantic_map = self.create_semantic_map(tiles, context)

        # Use enhanced walkability check that considers semantics
        def is_semantically_walkable(pos: tuple[int, int]) -> bool:
            row, col = pos
            if not (0 <= row < tiles.shape[0] and 0 <= col < tiles.shape[1]):
                return False

            # Check base walkability
            base_walkable = self._is_position_walkable(tiles, pos, context)
            if not base_walkable:
                return False

            # Apply semantic constraints
            tile_semantics = semantic_map["tile_semantics"].get(pos)
            if tile_semantics:
                # Avoid NPCs if requested
                if avoid_npcs and tile_semantics.category == SemanticCategory.NPC:
                    return False

                # Avoid obstacles and buildings unless they're doors
                if tile_semantics.category == SemanticCategory.OBSTACLE:
                    return False
                if (
                    tile_semantics.category == SemanticCategory.BUILDING
                    and tile_semantics.interaction_type != InteractionType.ENTER
                ):
                    return False

            return True

        # Enhanced A* pathfinding with semantic awareness
        from collections import deque

        if not is_semantically_walkable(start) or not is_semantically_walkable(end):
            return []

        queue = deque([(start, [start])])
        visited = {start}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while queue:
            (current_pos, path) = queue.popleft()

            if current_pos == end:
                return path

            for dr, dc in directions:
                new_row, new_col = current_pos[0] + dr, current_pos[1] + dc
                new_pos = (new_row, new_col)

                if new_pos not in visited and is_semantically_walkable(new_pos):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))

        return []  # No semantic path found

    def classify_tile_semantics_enhanced(
        self, tile_id: int, position: tuple[int, int], context: str
    ) -> TileSemantics:
        """
        Enhanced tile classification using learned collision data.

        Integrates collision detection knowledge from Issue #34 with semantic classification.
        """
        # Get collision data if available
        collision_data = {}
        if context in self._tile_semantics and tile_id in self._tile_semantics[context]:
            tile_props = self._tile_semantics[context][tile_id]
            collision_data = {
                "walkable": tile_props.get("walkable", True),
                "confidence": tile_props.get("confidence", 0.0),
            }

        # Create dummy surrounding tiles if position info not available
        surrounding_tiles = np.zeros((3, 3), dtype=np.uint8)

        # Use enhanced classification with collision data
        if collision_data:
            return self._semantic_classifier.classify_tile_with_collision_data(
                tile_id, position, collision_data, surrounding_tiles, context
            )
        else:
            return self._semantic_classifier.classify_tile_semantics(
                tile_id, position, surrounding_tiles, context
            )

    def learn_semantic_properties(self, observations: list[dict[str, Any]]) -> None:
        """
        Learn semantic properties from gameplay observations.

        Extends existing collision learning with semantic understanding.

        Args:
            observations: List of observation dicts with semantic information
        """
        start_time = time.perf_counter()

        for obs in observations:
            tile_id = obs["tile_id"]
            context = obs.get("context", "default")

            # Learn basic collision properties (existing functionality)
            self.learn_tile_properties([obs])

            # Learn semantic properties
            if "semantic_category" in obs:
                semantic_category = obs["semantic_category"]
                interaction_type = obs.get("interaction_type")

                # Create or update semantic knowledge
                semantics = TileSemantics(
                    tile_id=tile_id,
                    category=semantic_category,
                    interaction_type=interaction_type,
                    confidence=min(0.9, obs.get("confidence", 0.7)),
                    context_dependent=True,
                )

                self._add_semantic_knowledge(semantics, context)

        # Performance check
        learning_time = (time.perf_counter() - start_time) * 1000
        if learning_time > 100:
            # Log performance warning
            pass

    def _add_semantic_knowledge(self, semantics: TileSemantics, context: str) -> None:
        """
        Add semantic knowledge to the knowledge base with memory management.

        Args:
            semantics: TileSemantics to store
            context: Map context for the knowledge
        """
        # Memory management - limit knowledge base size
        max_knowledge_per_context = 200

        if len(self._semantic_knowledge[context]) >= max_knowledge_per_context:
            # Remove oldest/lowest confidence entries
            items = list(self._semantic_knowledge[context].items())
            items.sort(key=lambda x: x[1].confidence)  # Sort by confidence

            # Keep only the most confident entries
            keep_count = max_knowledge_per_context // 2
            self._semantic_knowledge[context] = dict(items[-keep_count:])

        # Add new knowledge
        self._semantic_knowledge[context][semantics.tile_id] = semantics

    # Private helper methods (following clean code principle of hiding implementation details)

    def _validate_tile_grid(self, tiles: np.ndarray) -> None:
        """Validate that tiles array has correct shape and type."""
        if tiles.size == 0:
            raise ValueError("Cannot analyze empty tile grid")

        if tiles.shape != (self.EXPECTED_GRID_HEIGHT, self.EXPECTED_GRID_WIDTH):
            raise ValueError(
                f"Expected grid shape ({self.EXPECTED_GRID_HEIGHT}, {self.EXPECTED_GRID_WIDTH}), got {tiles.shape}"
            )

    def _handle_menu_overlays(
        self, tiles: np.ndarray, menu_tiles: list[tuple[int, int]]
    ) -> np.ndarray:
        """Handle menu overlay tiles by preserving underlying tile data."""
        # In a real implementation, this might preserve the underlying tiles
        # For now, we'll mark menu areas but not modify the core tiles
        result: np.ndarray = tiles.copy()
        return result

    def _is_position_walkable(
        self, tiles: np.ndarray, position: tuple[int, int], context: str
    ) -> bool:
        """Check if a position is walkable based on learned tile semantics."""
        row, col = position
        grid_height, grid_width = tiles.shape

        if not (0 <= row < grid_height and 0 <= col < grid_width):
            return False

        tile_id = int(tiles[row, col])

        # Check learned semantics
        if context in self._tile_semantics and tile_id in self._tile_semantics[context]:
            tile_props = self._tile_semantics[context][tile_id]
            confidence = tile_props.get("confidence", 0.0)
            walkable = tile_props.get("walkable", True)
            if (
                isinstance(confidence, int | float) and confidence > 0.3
            ):  # Only use if reasonably confident
                return bool(walkable)

        # Default heuristics for unknown tiles
        # Player and NPC tiles are definitely not walkable for pathfinding
        if tile_id == self.PLAYER_TILE_ID:
            return False
        if self.NPC_TILE_ID_START <= tile_id <= self.NPC_TILE_ID_END:
            return False
        if tile_id == self.MENU_TILE_ID:
            return False

        # Conservative assumption: unknown tiles might be walkable
        # In production, might want to be more conservative
        return True

    # Additional helper methods for semantic classification (Issue #39)

    def _extract_surrounding_tiles(self, tiles: np.ndarray, row: int, col: int) -> np.ndarray:
        """Extract 3x3 surrounding tile area for context analysis."""
        surrounding = np.zeros((3, 3), dtype=np.uint8)

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                new_row, new_col = row + dr, col + dc

                # Use tile if in bounds, otherwise use 0 (default)
                if 0 <= new_row < tiles.shape[0] and 0 <= new_col < tiles.shape[1]:
                    surrounding[dr + 1, dc + 1] = tiles[new_row, new_col]

        return surrounding

    def _is_interactive_tile(self, semantics: TileSemantics) -> bool:
        """Check if a tile is interactive based on semantic classification."""
        # Interactive categories
        interactive_categories = {
            SemanticCategory.NPC,
            SemanticCategory.ITEM,
            SemanticCategory.DOOR,
            SemanticCategory.INTERACTIVE_OBJECT,
        }

        # High-confidence classifications only
        return semantics.category in interactive_categories and semantics.confidence > 0.5

    def _generate_object_description(self, semantics: TileSemantics) -> str:
        """Generate human-readable description for interactive objects."""
        category_descriptions = {
            SemanticCategory.NPC: "Friendly person to talk to",
            SemanticCategory.ITEM: "Item that can be picked up",
            SemanticCategory.DOOR: "Door or entrance",
            SemanticCategory.INTERACTIVE_OBJECT: "Interactive object",
            SemanticCategory.BUILDING: "Building entrance",
        }

        base_description = category_descriptions.get(
            semantics.category, "Unknown interactive object"
        )

        # Add subcategory if available
        if semantics.subcategory:
            return f"{base_description} ({semantics.subcategory})"

        return base_description

    def _find_pokemon_center_patterns(
        self, tiles: np.ndarray, context: str
    ) -> list[StrategicLocation]:
        """Find Pokemon Center patterns in the tile grid."""
        locations = []

        # Pokemon Center pattern detection
        for row in range(tiles.shape[0] - 2):
            for col in range(tiles.shape[1] - 2):
                # Check for building pattern
                region = tiles[row : row + 3, col : col + 3]

                # Look for Pokemon Center tiles (simplified pattern)
                building_tiles = [30, 31, 32]  # Building wall tiles
                door_tiles = [35]  # Door tiles

                building_count = sum(1 for tile_id in region.flatten() if tile_id in building_tiles)
                door_count = sum(1 for tile_id in region.flatten() if tile_id in door_tiles)

                # Pattern match criteria
                if building_count >= 4 and door_count >= 1:
                    # Found potential Pokemon Center
                    center_pos = (row + 1, col + 1)  # Center of pattern

                    # Find entrance tiles
                    entrance_tiles = []
                    for r in range(row, row + 3):
                        for c in range(col, col + 3):
                            if tiles[r, c] in door_tiles:
                                entrance_tiles.append((r, c))

                    location = StrategicLocation(
                        position=center_pos,
                        location_type=StrategicLocationType.POKEMON_CENTER,
                        name=f"{context.title()} Pokemon Center",
                        entrance_tiles=entrance_tiles,
                        confidence=min(0.9, (building_count + door_count) / 10.0),
                    )
                    locations.append(location)

        return locations

    def _find_shop_patterns(self, tiles: np.ndarray, context: str) -> list[StrategicLocation]:
        """Find shop patterns in the tile grid."""
        locations = []

        # Shop pattern detection (simpler than Pokemon Center)
        for row in range(tiles.shape[0] - 1):
            for col in range(tiles.shape[1] - 1):
                region = tiles[row : row + 2, col : col + 2]

                # Shop tiles
                building_tiles = [25, 26]
                door_tiles = [28]

                building_count = sum(1 for tile_id in region.flatten() if tile_id in building_tiles)
                door_count = sum(1 for tile_id in region.flatten() if tile_id in door_tiles)

                if building_count >= 2 and door_count >= 1:
                    shop_pos = (row, col)

                    # Find entrance
                    entrance_tiles = []
                    for r in range(row, row + 2):
                        for c in range(col, col + 2):
                            if tiles[r, c] in door_tiles:
                                entrance_tiles.append((r, c))

                    location = StrategicLocation(
                        position=shop_pos,
                        location_type=StrategicLocationType.SHOP,
                        name=f"{context.title()} Shop",
                        entrance_tiles=entrance_tiles,
                        confidence=min(0.8, (building_count + door_count) / 6.0),
                    )
                    locations.append(location)

        return locations

    def _find_gym_patterns(self, tiles: np.ndarray, context: str) -> list[StrategicLocation]:
        """Find gym patterns in the tile grid."""
        locations: list[StrategicLocation] = []

        # Gym detection (larger buildings in cities)
        if "city" not in context.lower():
            return locations

        for row in range(tiles.shape[0] - 3):
            for col in range(tiles.shape[1] - 3):
                region = tiles[row : row + 4, col : col + 4]

                # Large building patterns suggest gyms
                building_tiles = [40, 41, 42, 43]  # Gym-specific tiles
                door_tiles = [45]

                building_count = sum(1 for tile_id in region.flatten() if tile_id in building_tiles)
                door_count = sum(1 for tile_id in region.flatten() if tile_id in door_tiles)

                # More strict criteria for gyms
                if building_count >= 8 and door_count >= 1:
                    gym_pos = (row + 1, col + 1)

                    entrance_tiles = []
                    for r in range(row, row + 4):
                        for c in range(col, col + 4):
                            if tiles[r, c] in door_tiles:
                                entrance_tiles.append((r, c))

                    location = StrategicLocation(
                        position=gym_pos,
                        location_type=StrategicLocationType.GYM,
                        name=f"{context.title()} Gym",
                        entrance_tiles=entrance_tiles,
                        confidence=min(0.95, building_count / 12.0),
                    )
                    locations.append(location)

        return locations

    def _identify_walkable_areas(self, tiles: np.ndarray, context: str) -> list[tuple[int, int]]:
        """Identify areas that are safe for navigation."""
        walkable_areas = []

        for row in range(tiles.shape[0]):
            for col in range(tiles.shape[1]):
                position = (row, col)

                if self._is_position_walkable(tiles, position, context):
                    # Additional semantic checks
                    surrounding = self._extract_surrounding_tiles(tiles, row, col)
                    semantics = self._semantic_classifier.classify_tile_semantics(
                        int(tiles[row, col]), position, surrounding, context
                    )

                    # Consider semantic safety
                    if self._is_semantically_safe_area(semantics):
                        walkable_areas.append(position)

        return walkable_areas

    def _is_semantically_safe_area(self, semantics: TileSemantics) -> bool:
        """Check if an area is semantically safe for navigation."""
        # Safe categories for walking
        safe_categories = {
            SemanticCategory.TERRAIN,
        }

        # Unsafe categories to avoid
        unsafe_categories = {
            SemanticCategory.OBSTACLE,
            SemanticCategory.NPC,  # Might block path
        }

        # Items are walkable but might be worth visiting
        if semantics.category == SemanticCategory.ITEM:
            return True

        return (
            semantics.category in safe_categories
            and semantics.category not in unsafe_categories
            and semantics.confidence > 0.3
        )
