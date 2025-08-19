"""
Strategy Response Data Structures

Type-safe data classes for Opus strategy responses.
Provides immutable, validated structures for strategic data.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    """
    Immutable specification for a parallel experiment.

    Encapsulates all data needed to execute a strategic experiment
    including DSL script, checkpoints, and success criteria.
    """

    id: str
    name: str
    checkpoint: str
    script_dsl: str
    expected_outcome: str
    priority: int
    directives: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate experiment specification on creation."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Experiment ID must be non-empty string")

        if not self.name or not isinstance(self.name, str):
            raise ValueError("Experiment name must be non-empty string")

        if not self.checkpoint or not isinstance(self.checkpoint, str):
            raise ValueError("Checkpoint must be non-empty string")

        if not self.script_dsl or not isinstance(self.script_dsl, str):
            raise ValueError("Script DSL must be non-empty string")

        if not isinstance(self.priority, int) or self.priority < 1:
            raise ValueError("Priority must be positive integer")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "checkpoint": self.checkpoint,
            "script_dsl": self.script_dsl,
            "expected_outcome": self.expected_outcome,
            "priority": self.priority,
            "directives": list(self.directives),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StrategyResponse:
    """
    Immutable strategic response from Claude Opus.

    Contains validated experiments, insights, and strategic direction.
    Implements content-based caching and integrity verification.
    """

    strategy_id: str
    experiments: list[ExperimentSpec]
    strategic_insights: list[str]
    next_checkpoints: list[str]
    timestamp: datetime | None = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    _content_hash: str | None = field(init=False, default=None)

    def __post_init__(self):
        """Validate strategy response and compute content hash."""
        if not self.strategy_id or not isinstance(self.strategy_id, str):
            raise ValueError("Strategy ID must be non-empty string")

        if not isinstance(self.experiments, list):
            raise ValueError("Experiments must be a list")

        if not isinstance(self.strategic_insights, list):
            raise ValueError("Strategic insights must be a list")

        if not isinstance(self.next_checkpoints, list):
            raise ValueError("Next checkpoints must be a list")

        # Ensure all experiments are valid ExperimentSpec instances
        for i, exp in enumerate(self.experiments):
            if not isinstance(exp, ExperimentSpec):
                raise ValueError(f"Experiment {i} must be ExperimentSpec instance")

        # Compute content hash for caching and integrity
        object.__setattr__(self, "_content_hash", self._compute_content_hash())

    def _compute_content_hash(self) -> str:
        """Compute SHA-256 hash of response content for integrity verification."""
        content = {
            "strategy_id": self.strategy_id,
            "experiments": [exp.to_dict() for exp in self.experiments],
            "strategic_insights": self.strategic_insights,
            "next_checkpoints": self.next_checkpoints,
        }

        content_json = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content_json.encode("utf-8")).hexdigest()

    @property
    def content_hash(self) -> str:
        """Get content hash for caching and integrity checks."""
        if self._content_hash is None:
            # This should never happen after __post_init__, but handle gracefully
            return self._compute_content_hash()
        return self._content_hash

    def get_experiments_by_priority(self) -> list[ExperimentSpec]:
        """Get experiments sorted by priority (highest first)."""
        return sorted(self.experiments, key=lambda x: x.priority, reverse=True)

    def get_high_priority_experiments(self, min_priority: int = 3) -> list[ExperimentSpec]:
        """Get experiments with priority >= min_priority."""
        return [exp for exp in self.experiments if exp.priority >= min_priority]

    def extract_directives(self) -> list[str]:
        """Extract all strategic directives from response."""
        directives = []

        # Extract from experiments
        for exp in self.experiments:
            directives.extend(exp.directives)

        # Extract from insights marked as directives
        for insight in self.strategic_insights:
            if insight.upper().startswith("DIRECTIVE:"):
                directive = insight[10:].strip()  # Remove 'DIRECTIVE:' prefix
                directives.append(directive)

        return list(set(directives))  # Remove duplicates

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy_id": self.strategy_id,
            "experiments": [exp.to_dict() for exp in self.experiments],
            "strategic_insights": self.strategic_insights,
            "next_checkpoints": self.next_checkpoints,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": dict(self.metadata),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyResponse":
        """Create StrategyResponse from dictionary."""
        experiments = []
        for exp_data in data.get("experiments", []):
            exp = ExperimentSpec(
                id=exp_data["id"],
                name=exp_data["name"],
                checkpoint=exp_data["checkpoint"],
                script_dsl=exp_data["script_dsl"],
                expected_outcome=exp_data["expected_outcome"],
                priority=exp_data["priority"],
                directives=exp_data.get("directives", []),
                metadata=exp_data.get("metadata", {}),
            )
            experiments.append(exp)

        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

        return cls(
            strategy_id=data["strategy_id"],
            experiments=experiments,
            strategic_insights=data.get("strategic_insights", []),
            next_checkpoints=data.get("next_checkpoints", []),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class FallbackStrategy:
    """
    Fallback strategy for use when Opus communication fails.

    Provides basic exploration and learning experiments to maintain
    system progress even during strategic planning failures.
    """

    @staticmethod
    def create_default_fallback(game_state: dict[str, Any]) -> StrategyResponse:
        """Create default fallback strategy based on current game state."""
        fallback_experiments = [
            ExperimentSpec(
                id="fallback_explore",
                name="Basic Exploration",
                checkpoint=game_state.get("current_checkpoint", "fallback_checkpoint"),
                script_dsl="EXPLORE; COLLECT_ITEMS; SAVE_PROGRESS",
                expected_outcome="incremental_progress",
                priority=2,
                directives=["exploration", "item_collection"],
                metadata={"type": "fallback", "reason": "opus_unavailable"},
            ),
            ExperimentSpec(
                id="fallback_level",
                name="Experience Grinding",
                checkpoint=game_state.get("current_checkpoint", "fallback_checkpoint"),
                script_dsl="BATTLE_WILD_POKEMON; GAIN_EXPERIENCE; LEVEL_UP",
                expected_outcome="level_increase",
                priority=1,
                directives=["leveling", "stat_improvement"],
                metadata={"type": "fallback", "reason": "opus_unavailable"},
            ),
        ]

        return StrategyResponse(
            strategy_id=f"fallback_{int(datetime.utcnow().timestamp())}",
            experiments=fallback_experiments,
            strategic_insights=[
                "Using fallback strategy due to Opus unavailability",
                "Focus on basic progression and data collection",
                "Resume strategic planning when Opus connection restored",
            ],
            next_checkpoints=[game_state.get("location", "unknown_location")],
            metadata={"type": "fallback", "created_by": "fallback_strategy"},
        )
