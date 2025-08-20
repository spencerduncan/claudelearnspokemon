"""
PositionTracker - Enhanced position detection and tracking system for TileObserver.

Part of Issue #28: TileObserver Position Detection System
Provides position history tracking, movement pattern analysis, and NPC type differentiation.

Following Clean Code principles:
- Single Responsibility: Each class handles one aspect of position tracking
- Open/Closed: Extensible for new movement patterns and entity types
- Liskov Substitution: Proper inheritance patterns
- Interface Segregation: Focused interfaces
- Dependency Inversion: Abstract from concrete implementations

Performance Requirements:
- Position tracking: < 25ms for realistic workloads
- Pattern analysis: < 50ms
- Memory efficient with configurable history limits

Author: Linus Torbot (Claude Code) - Kernel Quality Standards Applied
"""

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class Position:
    """Basic position with timestamp."""

    x: int
    y: int
    timestamp: float
    map_id: str | None = None


@dataclass
class EntityPosition(Position):
    """Extended position for entities (player, NPCs) with metadata."""

    entity_type: str = "unknown"
    entity_id: str = ""
    facing_direction: str = "down"
    sprite_id: int = 0


@dataclass
class MovementPattern:
    """Analysis of movement patterns for an entity."""

    entity_id: str
    direction: str = "unknown"  # up, down, left, right, diagonal
    pattern_type: str = "unknown"  # linear, circular, random, stationary
    speed: float = 0.0  # tiles per second
    consistency: float = 0.0  # 0-1 score of pattern consistency
    confidence: float = 0.0  # 0-1 confidence in pattern detection
    sample_size: int = 0
    created_at: float = field(default_factory=time.time)


class PositionTracker:
    """
    Enhanced position tracking system for TileObserver.

    Responsibilities:
    - Track position history with timestamps for all entities
    - Analyze movement patterns and predict future positions
    - Differentiate between entity types (player, trainer, wild_pokemon, town_npc)
    - Provide efficient position queries and pattern analysis
    - Maintain performance requirements for real-time gameplay
    """

    # Constants for clean, self-documenting code
    DEFAULT_MAX_HISTORY = 100
    MIN_POSITIONS_FOR_PATTERN = 3
    POSITION_UPDATE_TOLERANCE = 0.001  # Seconds

    # Entity type constants
    ENTITY_PLAYER = "player"
    ENTITY_TRAINER = "trainer"
    ENTITY_WILD_POKEMON = "wild_pokemon"
    ENTITY_TOWN_NPC = "town_npc"
    ENTITY_GENERIC_NPC = "generic_npc"

    def __init__(self, max_history_size: int = DEFAULT_MAX_HISTORY) -> None:
        """
        Initialize PositionTracker.

        Args:
            max_history_size: Maximum number of positions to retain per entity
        """
        self.max_history_size = max_history_size

        # Position histories per entity (entity_id -> deque of EntityPosition)
        self._position_histories: dict[str, deque[EntityPosition]] = defaultdict(
            lambda: deque(maxlen=max_history_size)
        )

        # Current positions for quick access (entity_id -> EntityPosition)
        self._current_positions: dict[str, EntityPosition] = {}

        # Entity type mapping (entity_id -> entity_type)
        self._entity_types: dict[str, str] = {}

        # Cached movement patterns (entity_id -> MovementPattern)
        self._movement_patterns: dict[str, MovementPattern] = {}

        # Performance tracking
        self._last_pattern_analysis: dict[str, float] = {}

    def track_player_position(
        self,
        x: int,
        y: int,
        timestamp: float,
        facing_direction: str = "down",
        map_id: str | None = None,
    ) -> None:
        """
        Track player position with timestamp.

        Args:
            x: X coordinate in tile grid
            y: Y coordinate in tile grid
            timestamp: Time of position capture
            facing_direction: Direction player is facing
            map_id: Optional map identifier
        """
        entity_pos = EntityPosition(
            x=x,
            y=y,
            timestamp=timestamp,
            entity_type=self.ENTITY_PLAYER,
            entity_id="player",
            facing_direction=facing_direction,
            sprite_id=255,  # Standard player sprite ID
            map_id=map_id,
        )

        self._update_entity_position("player", entity_pos)

    def track_npc_position(
        self,
        npc_id: str,
        x: int,
        y: int,
        timestamp: float,
        npc_type: str = ENTITY_GENERIC_NPC,
        sprite_id: int = 200,
        facing_direction: str = "down",
        map_id: str | None = None,
    ) -> None:
        """
        Track NPC position with type differentiation.

        Args:
            npc_id: Unique identifier for this NPC
            x: X coordinate in tile grid
            y: Y coordinate in tile grid
            timestamp: Time of position capture
            npc_type: Type of NPC (trainer, wild_pokemon, town_npc, etc)
            sprite_id: Sprite identifier for visual recognition
            facing_direction: Direction NPC is facing
            map_id: Optional map identifier
        """
        entity_pos = EntityPosition(
            x=x,
            y=y,
            timestamp=timestamp,
            entity_type=npc_type,
            entity_id=npc_id,
            facing_direction=facing_direction,
            sprite_id=sprite_id,
            map_id=map_id,
        )

        self._entity_types[npc_id] = npc_type
        self._update_entity_position(npc_id, entity_pos)

    def get_current_player_position(self) -> EntityPosition | None:
        """Get current player position."""
        return self._current_positions.get("player")

    def get_current_npc_positions(self) -> list[EntityPosition]:
        """Get all current NPC positions."""
        return [pos for entity_id, pos in self._current_positions.items() if entity_id != "player"]

    def get_npcs_by_type(self, npc_type: str) -> list[EntityPosition]:
        """Get NPCs of specific type."""
        return [
            pos
            for entity_id, pos in self._current_positions.items()
            if self._entity_types.get(entity_id) == npc_type
        ]

    def get_player_position_history(self) -> list[EntityPosition]:
        """Get player position history ordered by timestamp."""
        return list(self._position_histories["player"])

    def get_npc_position_history(self, npc_id: str) -> list[EntityPosition]:
        """Get NPC position history."""
        return list(self._position_histories[npc_id])

    def analyze_movement_pattern(
        self, entity_id: str, force_refresh: bool = False
    ) -> MovementPattern:
        """
        Analyze movement pattern for an entity.

        Args:
            entity_id: Entity to analyze
            force_refresh: Force new analysis even if cached

        Returns:
            MovementPattern analysis results
        """
        # Check cache unless forced refresh
        if not force_refresh and entity_id in self._movement_patterns:
            last_analysis = self._last_pattern_analysis.get(entity_id, 0)
            if time.time() - last_analysis < 2.0:  # Cache for 2 seconds for better performance
                return self._movement_patterns[entity_id]

        history = self._position_histories.get(entity_id, deque())

        if len(history) < self.MIN_POSITIONS_FOR_PATTERN:
            # Not enough data for pattern analysis
            pattern = MovementPattern(
                entity_id=entity_id, pattern_type="insufficient_data", sample_size=len(history)
            )
        else:
            # Only analyze recent positions for performance (last 10)
            recent_history = list(history)[-10:] if len(history) > 10 else list(history)
            pattern = self._calculate_movement_pattern(entity_id, recent_history)

        # Cache the result
        self._movement_patterns[entity_id] = pattern
        self._last_pattern_analysis[entity_id] = time.time()

        return pattern

    def predict_next_position(
        self, entity_id: str, prediction_time: float
    ) -> EntityPosition | None:
        """
        Predict future position based on movement pattern.

        Args:
            entity_id: Entity to predict for
            prediction_time: Time to predict position for

        Returns:
            Predicted EntityPosition or None if unpredictable
        """
        current_pos = self._current_positions.get(entity_id)
        if not current_pos:
            return None

        pattern = self.analyze_movement_pattern(entity_id)

        # Predict for linear movement with reasonable confidence
        if pattern.pattern_type != "linear" or pattern.confidence < 0.3:
            return None

        time_delta = prediction_time - current_pos.timestamp
        if time_delta <= 0:
            return current_pos

        # Calculate predicted position based on direction and speed
        dx, dy = self._direction_to_vector(pattern.direction)
        distance = pattern.speed * time_delta

        predicted_x = int(current_pos.x + dx * distance)
        predicted_y = int(current_pos.y + dy * distance)

        return EntityPosition(
            x=predicted_x,
            y=predicted_y,
            timestamp=prediction_time,
            entity_type=current_pos.entity_type,
            entity_id=current_pos.entity_id,
            facing_direction=(
                pattern.direction
                if pattern.direction != "unknown"
                else current_pos.facing_direction
            ),
            sprite_id=current_pos.sprite_id,
            map_id=current_pos.map_id,
        )

    # Private implementation methods

    def _update_entity_position(self, entity_id: str, position: EntityPosition) -> None:
        """Update position for an entity efficiently."""
        # Update current position
        self._current_positions[entity_id] = position

        # Add to history (deque automatically handles max size)
        self._position_histories[entity_id].append(position)

        # Only invalidate pattern cache if position actually changed
        current = self._current_positions.get(entity_id)
        if (
            current
            and len(self._position_histories[entity_id]) > 1
            and entity_id in self._movement_patterns
        ):
            prev_pos = (
                list(self._position_histories[entity_id])[-2]
                if len(self._position_histories[entity_id]) > 1
                else None
            )
            if prev_pos and (prev_pos.x != position.x or prev_pos.y != position.y):
                # Position actually changed, invalidate cache
                del self._movement_patterns[entity_id]

    def _calculate_movement_pattern(
        self, entity_id: str, positions: list[EntityPosition]
    ) -> MovementPattern:
        """Calculate movement pattern from position history."""
        if len(positions) < 2:
            return MovementPattern(
                entity_id=entity_id, pattern_type="stationary", sample_size=len(positions)
            )

        # Calculate movement vectors
        vectors = []
        speeds = []

        for i in range(1, len(positions)):
            prev_pos = positions[i - 1]
            curr_pos = positions[i]

            dx = curr_pos.x - prev_pos.x
            dy = curr_pos.y - prev_pos.y
            dt = curr_pos.timestamp - prev_pos.timestamp

            if dt > 0:
                vectors.append((dx, dy))
                distance = math.sqrt(dx * dx + dy * dy)
                speeds.append(distance / dt)

        if not vectors:
            return MovementPattern(
                entity_id=entity_id, pattern_type="stationary", sample_size=len(positions)
            )

        # Analyze pattern type
        if all(vx == 0 and vy == 0 for vx, vy in vectors):
            pattern_type = "stationary"
            direction = "none"
            consistency = 1.0
        else:
            pattern_type = self._classify_pattern_type(vectors)
            direction = self._determine_primary_direction(vectors)
            consistency = self._calculate_consistency(vectors)

        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        confidence = min(1.0, len(positions) / 10.0)  # Higher confidence with more samples

        return MovementPattern(
            entity_id=entity_id,
            direction=direction,
            pattern_type=pattern_type,
            speed=avg_speed,
            consistency=consistency,
            confidence=confidence,
            sample_size=len(positions),
        )

    def _classify_pattern_type(self, vectors: list[tuple[int, int]]) -> str:
        """Classify movement pattern from vectors."""
        if len(vectors) < 2:
            return "linear"

        # Check for consistent direction (linear) first
        directions = [self._vector_to_direction(vx, vy) for vx, vy in vectors if vx != 0 or vy != 0]
        unique_directions = set(directions)

        # Calculate consistency of movement
        consistency = self._calculate_consistency(vectors)

        # Calculate movement metrics for classification
        movement_distances = [abs(vx) + abs(vy) for vx, vy in vectors]
        avg_distance = (
            sum(movement_distances) / len(movement_distances) if movement_distances else 0
        )

        # Random/erratic: Check for erratic movement patterns
        # Large movements with inconsistent direction or many direction changes
        if avg_distance > 8 and consistency < 0.7:
            return "random"

        # Single direction or very consistent
        if len(unique_directions) <= 1:
            return "linear"

        # Check for circular pattern (returning to same area with some movement)
        total_displacement = (sum(vx for vx, vy in vectors), sum(vy for vx, vy in vectors))

        total_distance = sum(abs(vx) + abs(vy) for vx, vy in vectors)
        net_displacement = abs(total_displacement[0]) + abs(total_displacement[1])

        # Circular: low net displacement but with actual movement
        if total_distance > 0 and net_displacement / total_distance < 0.2:
            # Only circular if movements are reasonably small (not massive jumps)
            if avg_distance < 5:
                return "circular"

        # Linear: mostly consistent direction
        if consistency > 0.6 or len(unique_directions) <= 2:
            return "linear"

        return "random"

    def _determine_primary_direction(self, vectors: list[tuple[int, int]]) -> str:
        """Determine primary movement direction."""
        total_x = sum(vx for vx, vy in vectors)
        total_y = sum(vy for vx, vy in vectors)

        if abs(total_x) < 0.1 and abs(total_y) < 0.1:
            return "stationary"

        if abs(total_x) > abs(total_y):
            return "right" if total_x > 0 else "left"
        else:
            return "down" if total_y > 0 else "up"

    def _calculate_consistency(self, vectors: list[tuple[int, int]]) -> float:
        """Calculate consistency score (0-1) of movement pattern."""
        if not vectors:
            return 0.0

        # Calculate variance in direction
        directions = [self._vector_to_direction(vx, vy) for vx, vy in vectors]
        direction_counts: dict[str, int] = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

        if not direction_counts:
            return 0.0

        # Consistency is ratio of most common direction
        max_count = max(direction_counts.values())
        return max_count / len(directions)

    def _vector_to_direction(self, dx: int, dy: int) -> str:
        """Convert movement vector to direction string."""
        if dx == 0 and dy == 0:
            return "stationary"
        elif abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

    def _direction_to_vector(self, direction: str) -> tuple[float, float]:
        """Convert direction string to unit vector."""
        direction_map = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
            "stationary": (0, 0),
            "unknown": (0, 0),
        }
        return direction_map.get(direction, (0, 0))
