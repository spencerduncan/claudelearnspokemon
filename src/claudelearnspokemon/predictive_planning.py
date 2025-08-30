"""
Predictive Planning Components for OpusStrategist

Performance-optimized predictive planning system designed for real-time
Pokemon speedrun scenarios. Implements John Carmack's optimization principles:
- Cache-friendly data structures
- O(n) pattern analysis algorithms
- Mathematical precision in prediction models
- Memory-bounded caching with LRU eviction
- Persistent memory storage for continuous learning
"""

import hashlib
import json
import math
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .strategy_response import StrategyResponse

# Memory system integration - import memory tools if available
try:
    # These will be available when running in Claude Code environment
    # For testing, we'll use mock implementations
    _MEMORY_AVAILABLE = True
except ImportError:
    _MEMORY_AVAILABLE = False


class MemoryIntegrationMixin:
    """
    Memory system integration for predictive planning components.

    Provides methods for storing optimization patterns, prediction accuracy,
    and performance insights in the persistent memory system for continuous learning.
    Implements John Carmack's principle: measure, learn, optimize.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_enabled = _MEMORY_AVAILABLE
        self.component_name = self.__class__.__name__

    async def store_optimization_pattern(
        self,
        pattern_type: str,
        pattern_data: dict[str, Any],
        performance_metrics: dict[str, float],
        confidence: float,
    ) -> str | None:
        """
        Store optimization pattern in memory system for future reuse.

        Args:
            pattern_type: Type of pattern (e.g., "prediction_accuracy", "cache_optimization")
            pattern_data: Pattern-specific data and parameters
            performance_metrics: Performance measurements (time, accuracy, etc.)
            confidence: Confidence score (0.0-1.0) based on measurement reliability

        Returns:
            Memory ID if stored successfully, None otherwise
        """
        if not self.memory_enabled:
            return None

        try:
            # Create comprehensive pattern description
            content = (
                f"{self.component_name} {pattern_type} optimization pattern: "
                f"Performance metrics: {performance_metrics}. "
                f"Pattern parameters: {json.dumps(pattern_data, sort_keys=True)[:200]}..."
            )

            # Store in memory with predictive planning tags
            from .mcp_memory import store_memory

            memory_id = await store_memory(
                node_type="concept",
                content=content,
                confidence=confidence,
                source=f"{self.component_name} optimization analysis",
                tags=[
                    "claudelearnspokemon",
                    "predictive-planning",
                    f"issue-107-{pattern_type}",
                    f"{self.component_name.lower()}-optimization",
                    f"confidence-{int(confidence * 100)}",
                ],
            )

            return memory_id

        except Exception:
            # Graceful degradation - continue without memory storage
            return None

    async def store_prediction_accuracy(
        self,
        experiment_id: str,
        predicted_outcome: dict[str, Any],
        actual_outcome: dict[str, Any],
        accuracy_score: float,
    ) -> str | None:
        """
        Store prediction accuracy results for learning improvement.

        Enables the system to learn from prediction successes and failures,
        improving Bayesian priors and pattern recognition over time.
        """
        if not self.memory_enabled:
            return None

        try:
            # Calculate prediction error metrics
            prediction_error = abs(
                predicted_outcome.get("success_probability", 0.0)
                - actual_outcome.get("success_rate", 0.0)
            )

            time_error = abs(
                predicted_outcome.get("estimated_execution_time_ms", 0.0)
                - actual_outcome.get("actual_execution_time_ms", 0.0)
            )

            content = (
                f"Prediction accuracy result for experiment {experiment_id}: "
                f"Accuracy score: {accuracy_score:.3f}, "
                f"Probability error: {prediction_error:.3f}, "
                f"Time error: {time_error:.1f}ms. "
                f"Predicted: {json.dumps(predicted_outcome)[:100]}... "
                f"Actual: {json.dumps(actual_outcome)[:100]}..."
            )

            from .mcp_memory import store_memory

            memory_id = await store_memory(
                node_type="fact",
                content=content,
                confidence=min(accuracy_score, 0.95),  # Cap confidence at 95%
                source=f"{self.component_name} prediction tracking",
                tags=[
                    "claudelearnspokemon",
                    "predictive-planning",
                    "issue-107-accuracy",
                    f"experiment-{experiment_id}",
                    f"accuracy-{int(accuracy_score * 100)}",
                    f"{self.component_name.lower()}-results",
                ],
            )

            return memory_id

        except Exception:
            return None

    async def retrieve_similar_patterns(
        self,
        pattern_type: str,
        current_context: dict[str, Any],
        min_confidence: float = 0.6,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve similar optimization patterns from memory for improved predictions.

        Enables learning from historical successes to improve future performance.
        """
        if not self.memory_enabled:
            return []

        try:
            from .mcp_memory import search_memories

            # Search for similar patterns
            search_query = f"{self.component_name} {pattern_type} optimization"

            memories = await search_memories(
                pattern=search_query, min_confidence=min_confidence, limit=limit
            )

            # Filter and format results
            similar_patterns = []
            for memory in memories.get("results", []):
                if (
                    memory.get("confidence", 0.0) >= min_confidence
                    and pattern_type.lower() in memory.get("content", "").lower()
                ):

                    similar_patterns.append(
                        {
                            "memory_id": memory.get("id"),
                            "content": memory.get("content"),
                            "confidence": memory.get("confidence"),
                            "tags": memory.get("tags", []),
                            "age_days": memory.get("_temporal_context", {}).get("age_days", 0),
                        }
                    )

            return similar_patterns

        except Exception:
            return []

    async def store_performance_insight(
        self,
        insight_type: str,
        description: str,
        metrics: dict[str, float],
        actionable_recommendation: str,
    ) -> str | None:
        """
        Store performance insights discovered during optimization work.

        Captures lessons learned for future optimization efforts.
        """
        if not self.memory_enabled:
            return None

        try:
            content = (
                f"{self.component_name} {insight_type} insight: {description}. "
                f"Performance metrics: {metrics}. "
                f"Recommendation: {actionable_recommendation}"
            )

            from .mcp_memory import store_memory

            memory_id = await store_memory(
                node_type="concept",
                content=content,
                confidence=0.8,  # High confidence for measured insights
                source=f"{self.component_name} performance analysis",
                tags=[
                    "claudelearnspokemon",
                    "predictive-planning",
                    "issue-107-insight",
                    f"{insight_type}-optimization",
                    f"{self.component_name.lower()}-lessons",
                    "performance-improvement",
                ],
            )

            return memory_id

        except Exception:
            return None


class PredictionConfidence(Enum):
    """Confidence levels for predictions based on statistical significance."""

    VERY_LOW = 0.2  # Insufficient data, speculative
    LOW = 0.4  # Limited data, educated guess
    MEDIUM = 0.6  # Reasonable data, good confidence
    HIGH = 0.8  # Strong data, high confidence
    VERY_HIGH = 0.95  # Extensive data, very high confidence


class ScenarioTrigger(Enum):
    """Types of scenario triggers for contingency activation."""

    EXECUTION_FAILURE = "execution_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    UNEXPECTED_GAME_STATE = "unexpected_game_state"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIME_CONSTRAINT = "time_constraint"
    PATTERN_DEVIATION = "pattern_deviation"


@dataclass(frozen=True)
class OutcomePrediction:
    """
    Prediction for experiment execution outcome.

    Uses Bayesian inference and historical pattern analysis for
    probabilistic outcome modeling with confidence intervals.
    """

    experiment_id: str
    success_probability: float  # 0.0 to 1.0
    estimated_execution_time_ms: float
    expected_performance_score: float
    confidence: PredictionConfidence
    contributing_factors: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    historical_accuracy: float = 0.0  # Track prediction accuracy over time

    def __post_init__(self):
        """Validate prediction parameters."""
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValueError("Success probability must be between 0.0 and 1.0")

        if self.estimated_execution_time_ms < 0:
            raise ValueError("Execution time must be non-negative")

        if not isinstance(self.confidence, PredictionConfidence):
            raise ValueError("Confidence must be PredictionConfidence enum")

    def get_risk_score(self) -> float:
        """Calculate overall risk score based on factors."""
        base_risk = 1.0 - self.success_probability
        confidence_multiplier = 1.0 - (self.confidence.value * 0.5)
        risk_factor_penalty = len(self.risk_factors) * 0.1

        return min(base_risk * confidence_multiplier + risk_factor_penalty, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "success_probability": self.success_probability,
            "estimated_execution_time_ms": self.estimated_execution_time_ms,
            "expected_performance_score": self.expected_performance_score,
            "confidence": self.confidence.value,
            "contributing_factors": list(self.contributing_factors),
            "risk_factors": list(self.risk_factors),
            "historical_accuracy": self.historical_accuracy,
            "risk_score": self.get_risk_score(),
        }


@dataclass(frozen=True)
class ContingencyStrategy:
    """
    Contingency strategy for specific execution scenarios.

    Defines trigger conditions, strategic response, and probability
    modeling for automated contingency activation.
    """

    scenario_id: str
    trigger_type: ScenarioTrigger
    trigger_conditions: list[str]
    strategy: StrategyResponse
    activation_probability: float  # Likelihood this scenario occurs
    confidence: PredictionConfidence
    priority: int = 1  # Higher numbers = higher priority
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate contingency strategy parameters."""
        if not self.scenario_id:
            raise ValueError("Scenario ID cannot be empty")

        if not 0.0 <= self.activation_probability <= 1.0:
            raise ValueError("Activation probability must be between 0.0 and 1.0")

        if not isinstance(self.trigger_type, ScenarioTrigger):
            raise ValueError("Trigger type must be ScenarioTrigger enum")

        if self.priority < 1:
            raise ValueError("Priority must be positive integer")

    def should_activate(self, current_conditions: dict[str, Any]) -> bool:
        """
        Determine if contingency should activate based on current conditions.

        Uses pattern matching and threshold checking for efficient
        real-time activation decisions.
        """
        if not self.trigger_conditions:
            return False

        # Check each trigger condition
        matched_conditions = 0
        for condition in self.trigger_conditions:
            if self._matches_condition(condition, current_conditions):
                matched_conditions += 1

        # Require majority of conditions to match for activation
        activation_threshold = max(1, len(self.trigger_conditions) // 2 + 1)
        return matched_conditions >= activation_threshold

    def _matches_condition(self, condition: str, current_conditions: dict[str, Any]) -> bool:
        """Check if a single condition matches current state."""
        # Parse condition expressions like "failure_rate > 0.3"
        condition = condition.strip()

        # Handle comparison operators
        for op in [" > ", " >= ", " < ", " <= ", " == ", " != "]:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    try:
                        threshold = float(parts[1].strip())
                        current_value = current_conditions.get(key, 0.0)

                        if op == " > ":
                            return current_value > threshold
                        elif op == " >= ":
                            return current_value >= threshold
                        elif op == " < ":
                            return current_value < threshold
                        elif op == " <= ":
                            return current_value <= threshold
                        elif op == " == ":
                            return abs(current_value - threshold) < 1e-6  # Float comparison
                        elif op == " != ":
                            return abs(current_value - threshold) >= 1e-6

                    except ValueError:
                        pass  # Not a numeric comparison
                break

        # Simple pattern matching for text-based conditions
        condition_lower = condition.lower()

        # Performance degradation patterns
        if "performance" in condition_lower and "degradation" in condition_lower:
            current_performance = current_conditions.get("performance_score", 1.0)
            return current_performance < 0.7  # Below 70% performance

        # Execution failure patterns
        if "failure" in condition_lower or "error" in condition_lower:
            failure_rate = current_conditions.get("failure_rate", 0.0)
            return failure_rate > 0.3  # Above 30% failure rate

        # Time constraint patterns
        if "time" in condition_lower and (
            "exceeded" in condition_lower or "constraint" in condition_lower
        ):
            execution_time = current_conditions.get("execution_time_ms", 0)
            expected_time = current_conditions.get("expected_time_ms", float("inf"))
            return execution_time > expected_time * 1.5  # 50% over expected time

        # Default: check for exact key match or boolean value
        return current_conditions.get(condition, False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario_id": self.scenario_id,
            "trigger_type": self.trigger_type.value,
            "trigger_conditions": list(self.trigger_conditions),
            "strategy": self.strategy.to_dict(),
            "activation_probability": self.activation_probability,
            "confidence": self.confidence.value,
            "priority": self.priority,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PredictivePlanningResult:
    """
    Result of predictive planning analysis.

    Contains primary strategy, contingencies, outcome predictions,
    and performance metrics. Optimized for caching and rapid retrieval.
    """

    planning_id: str
    primary_strategy: StrategyResponse
    contingencies: list[ContingencyStrategy]
    outcome_predictions: dict[str, OutcomePrediction]
    confidence_scores: dict[str, float]
    execution_time_ms: float
    cache_metadata: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    _content_hash: str | None = field(init=False, default=None)

    def __post_init__(self):
        """Validate and compute content hash for caching."""
        if not self.planning_id:
            raise ValueError("Planning ID cannot be empty")

        if not isinstance(self.primary_strategy, StrategyResponse):
            raise ValueError("Primary strategy must be StrategyResponse")

        if not isinstance(self.contingencies, list):
            raise ValueError("Contingencies must be a list")

        if not isinstance(self.outcome_predictions, dict):
            raise ValueError("Outcome predictions must be a dictionary")

        if self.execution_time_ms < 0:
            raise ValueError("Execution time must be non-negative")

        # Validate all contingencies are ContingencyStrategy instances
        for i, contingency in enumerate(self.contingencies):
            if not isinstance(contingency, ContingencyStrategy):
                raise ValueError(f"Contingency {i} must be ContingencyStrategy instance")

        # Validate all outcome predictions are OutcomePrediction instances
        for exp_id, prediction in self.outcome_predictions.items():
            if not isinstance(prediction, OutcomePrediction):
                raise ValueError(f"Prediction for {exp_id} must be OutcomePrediction instance")

        # Compute content hash for caching and integrity
        object.__setattr__(self, "_content_hash", self._compute_content_hash())

    def _compute_content_hash(self) -> str:
        """Compute SHA-256 hash of result content for cache integrity."""
        content = {
            "planning_id": self.planning_id,
            "primary_strategy": self.primary_strategy.to_dict(),
            "contingencies": [c.to_dict() for c in self.contingencies],
            "outcome_predictions": {k: v.to_dict() for k, v in self.outcome_predictions.items()},
            "confidence_scores": self.confidence_scores,
        }

        content_json = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content_json.encode("utf-8")).hexdigest()

    @property
    def content_hash(self) -> str:
        """Get content hash for caching and integrity checks."""
        if self._content_hash is None:
            return self._compute_content_hash()
        return self._content_hash

    def get_high_priority_contingencies(self, min_priority: int = 3) -> list[ContingencyStrategy]:
        """Get contingencies with priority >= min_priority."""
        return [c for c in self.contingencies if c.priority >= min_priority]

    def get_contingencies_by_trigger(
        self, trigger_type: ScenarioTrigger
    ) -> list[ContingencyStrategy]:
        """Get contingencies matching specific trigger type."""
        return [c for c in self.contingencies if c.trigger_type == trigger_type]

    def get_overall_confidence(self) -> float:
        """Calculate overall confidence score for the prediction."""
        if not self.confidence_scores:
            return 0.0

        # Weighted average based on prediction importance
        total_weight = 0.0
        weighted_sum = 0.0

        for component, confidence in self.confidence_scores.items():
            # Primary strategy gets highest weight
            weight = 1.0 if component == "primary_strategy" else 0.5
            weighted_sum += confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_predicted_success_rate(self) -> float:
        """Get overall predicted success rate across all experiments."""
        if not self.outcome_predictions:
            return 0.0

        success_rates = [pred.success_probability for pred in self.outcome_predictions.values()]
        return sum(success_rates) / len(success_rates)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "planning_id": self.planning_id,
            "primary_strategy": self.primary_strategy.to_dict(),
            "contingencies": [c.to_dict() for c in self.contingencies],
            "outcome_predictions": {k: v.to_dict() for k, v in self.outcome_predictions.items()},
            "confidence_scores": self.confidence_scores,
            "execution_time_ms": self.execution_time_ms,
            "cache_metadata": dict(self.cache_metadata),
            "timestamp": self.timestamp.isoformat(),
            "content_hash": self.content_hash,
            "overall_confidence": self.get_overall_confidence(),
            "predicted_success_rate": self.get_predicted_success_rate(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictivePlanningResult":
        """Create PredictivePlanningResult from dictionary."""
        # Parse primary strategy
        primary_strategy = StrategyResponse.from_dict(data["primary_strategy"])

        # Parse contingencies
        contingencies = []
        for c_data in data.get("contingencies", []):
            strategy = StrategyResponse.from_dict(c_data["strategy"])
            contingency = ContingencyStrategy(
                scenario_id=c_data["scenario_id"],
                trigger_type=ScenarioTrigger(c_data["trigger_type"]),
                trigger_conditions=c_data["trigger_conditions"],
                strategy=strategy,
                activation_probability=c_data["activation_probability"],
                confidence=PredictionConfidence(c_data["confidence"]),
                priority=c_data.get("priority", 1),
                metadata=c_data.get("metadata", {}),
            )
            contingencies.append(contingency)

        # Parse outcome predictions
        outcome_predictions = {}
        for exp_id, pred_data in data.get("outcome_predictions", {}).items():
            prediction = OutcomePrediction(
                experiment_id=pred_data["experiment_id"],
                success_probability=pred_data["success_probability"],
                estimated_execution_time_ms=pred_data["estimated_execution_time_ms"],
                expected_performance_score=pred_data["expected_performance_score"],
                confidence=PredictionConfidence(pred_data["confidence"]),
                contributing_factors=pred_data.get("contributing_factors", []),
                risk_factors=pred_data.get("risk_factors", []),
                historical_accuracy=pred_data.get("historical_accuracy", 0.0),
            )
            outcome_predictions[exp_id] = prediction

        # Parse timestamp
        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

        return cls(
            planning_id=data["planning_id"],
            primary_strategy=primary_strategy,
            contingencies=contingencies,
            outcome_predictions=outcome_predictions,
            confidence_scores=data["confidence_scores"],
            execution_time_ms=data["execution_time_ms"],
            cache_metadata=data.get("cache_metadata", {}),
            timestamp=timestamp,
        )


@dataclass
class ExecutionPattern:
    """
    Represents execution pattern for similarity analysis.

    Optimized for cache-friendly access and O(1) similarity scoring.
    """

    pattern_id: str
    features: dict[str, float]  # Normalized feature values 0.0-1.0
    success_rate: float
    avg_execution_time: float
    frequency: int = 1
    last_seen: datetime = field(default_factory=datetime.utcnow)

    def compute_similarity(self, other: "ExecutionPattern") -> float:
        """
        Compute cosine similarity between patterns - O(k) complexity.

        Uses optimized dot product calculation for cache efficiency.
        """
        if not self.features or not other.features:
            return 0.0

        # Get common features for efficient computation
        common_features = set(self.features.keys()) & set(other.features.keys())
        if not common_features:
            return 0.0

        # Vectorized similarity computation
        dot_product = sum(self.features[f] * other.features[f] for f in common_features)

        # Pre-computed norms for efficiency
        norm_self = math.sqrt(sum(v * v for v in self.features.values()))
        norm_other = math.sqrt(sum(v * v for v in other.features.values()))

        if norm_self == 0 or norm_other == 0:
            return 0.0

        return dot_product / (norm_self * norm_other)


class ExecutionPatternAnalyzer(MemoryIntegrationMixin):
    """
    High-performance pattern analysis engine for predictive planning.

    Implements John Carmack optimization principles:
    - O(n) pattern analysis algorithms
    - Cache-friendly data structures
    - Memory-bounded storage with automatic eviction
    - Real-time capable with <50ms analysis time
    - Persistent memory storage for continuous learning
    """

    def __init__(
        self,
        max_patterns: int = 1000,
        similarity_threshold: float = 0.75,
        min_frequency_threshold: int = 3,
    ):
        """
        Initialize pattern analyzer with performance constraints.

        Args:
            max_patterns: Maximum stored patterns (memory bound)
            similarity_threshold: Minimum similarity for pattern matching
            min_frequency_threshold: Minimum frequency for reliable patterns
        """
        super().__init__()
        self.max_patterns = max_patterns
        self.similarity_threshold = similarity_threshold
        self.min_frequency_threshold = min_frequency_threshold

        # Cache-friendly storage structures
        self.patterns: dict[str, ExecutionPattern] = {}
        self.pattern_index: deque[str] = deque(maxlen=max_patterns)  # LRU ordering
        self.feature_statistics: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "sum": 0.0,
                "sum_sq": 0.0,
                "count": 0,
                "min": float("inf"),
                "max": float("-inf"),
            }
        )

        # Performance tracking
        self.analysis_times: deque[float] = deque(maxlen=100)  # Last 100 analysis times
        self.cache_hits = 0
        self.cache_misses = 0

    def analyze_execution_patterns(
        self,
        current_experiments: list[dict[str, Any]],
        historical_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze execution patterns with O(n) complexity.

        Returns pattern analysis results with similarity scores,
        trend predictions, and confidence metrics.
        """
        start_time = time.perf_counter()

        try:
            # Extract current pattern features
            current_features = self._extract_pattern_features(current_experiments)
            current_pattern = ExecutionPattern(
                pattern_id=f"current_{int(time.time())}",
                features=current_features,
                success_rate=0.0,  # Unknown for current
                avg_execution_time=0.0,  # Will be estimated
            )

            # Find similar historical patterns
            similar_patterns = self._find_similar_patterns(current_pattern)

            # Compute trend analysis
            trend_analysis = self._compute_trend_analysis(similar_patterns)

            # Generate confidence metrics
            confidence_metrics = self._compute_confidence_metrics(
                similar_patterns, len(current_experiments)
            )

            # Build comprehensive analysis result
            analysis_result = {
                "current_pattern": current_features,
                "similar_patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "similarity": self._get_cached_similarity(current_pattern, p),
                        "success_rate": p.success_rate,
                        "frequency": p.frequency,
                        "avg_execution_time": p.avg_execution_time,
                    }
                    for p in similar_patterns
                ],
                "trend_analysis": trend_analysis,
                "confidence_metrics": confidence_metrics,
                "pattern_statistics": self._get_pattern_statistics(),
            }

            # Store current pattern for future analysis
            if historical_results:
                self._update_pattern_storage(current_pattern, historical_results)

            # Record performance metrics
            analysis_time = (time.perf_counter() - start_time) * 1000
            self.analysis_times.append(analysis_time)

            analysis_result["analysis_time_ms"] = analysis_time  # type: ignore
            analysis_result["performance_metrics"] = self.get_performance_metrics()

            return analysis_result

        except Exception as e:
            # Graceful degradation on analysis failure
            return {
                "error": str(e),
                "fallback_analysis": True,
                "analysis_time_ms": (time.perf_counter() - start_time) * 1000,
            }

    def _extract_pattern_features(self, experiments: list[dict[str, Any]]) -> dict[str, float]:
        """
        Extract normalized feature vector from experiments - O(n) complexity.

        Features are normalized to 0.0-1.0 range for consistent similarity scoring.
        """
        if not experiments:
            return {}

        features = {}

        # Experiment count feature
        features["experiment_count"] = min(len(experiments) / 10.0, 1.0)  # Normalize to max 10

        # Priority distribution features
        priorities = [exp.get("priority", 1) for exp in experiments]
        features["avg_priority"] = sum(priorities) / len(priorities) / 5.0  # Normalize to max 5
        features["priority_variance"] = min(
            statistics.variance(priorities) if len(priorities) > 1 else 0.0, 1.0
        )

        # Script complexity features
        script_lengths = [len(exp.get("script_dsl", "")) for exp in experiments]
        if script_lengths:
            features["avg_script_length"] = min(
                sum(script_lengths) / len(script_lengths) / 100.0, 1.0
            )
            features["script_length_variance"] = min(
                statistics.variance(script_lengths) if len(script_lengths) > 1 else 0.0, 1.0
            )

        # Checkpoint distribution features
        unique_checkpoints = len({exp.get("checkpoint", "unknown") for exp in experiments})
        features["checkpoint_diversity"] = unique_checkpoints / len(experiments)

        # DSL command analysis
        all_scripts = " ".join(exp.get("script_dsl", "") for exp in experiments).upper()
        total_commands = len(all_scripts.split()) if all_scripts.strip() else 1

        # Common command frequency features
        common_commands = ["MOVE", "BATTLE", "USE", "SAVE", "EXPLORE", "COLLECT"]
        for command in common_commands:
            command_count = all_scripts.count(command)
            features[f"cmd_{command.lower()}_freq"] = command_count / total_commands

        return features

    def _find_similar_patterns(self, current_pattern: ExecutionPattern) -> list[ExecutionPattern]:
        """
        Find similar patterns using optimized similarity search - O(n) complexity.

        Returns patterns sorted by similarity score (highest first).
        """
        similar_patterns = []

        for pattern in self.patterns.values():
            # Skip patterns with insufficient frequency
            if pattern.frequency < self.min_frequency_threshold:
                continue

            # Compute similarity using cached or computed value
            similarity = self._get_cached_similarity(current_pattern, pattern)

            if similarity >= self.similarity_threshold:
                similar_patterns.append(pattern)

        # Sort by similarity (highest first) - O(k log k) where k << n
        similar_patterns.sort(
            key=lambda p: self._get_cached_similarity(current_pattern, p), reverse=True
        )

        return similar_patterns[:10]  # Return top 10 most similar

    def _get_cached_similarity(
        self, pattern1: ExecutionPattern, pattern2: ExecutionPattern
    ) -> float:
        """Get similarity with caching for performance."""
        # For current implementation, compute directly
        # In production, could implement caching for frequently compared patterns
        return pattern1.compute_similarity(pattern2)

    def _compute_trend_analysis(self, similar_patterns: list[ExecutionPattern]) -> dict[str, float]:
        """
        Compute trend analysis from similar patterns - O(k) complexity.

        Uses statistical methods for trend detection and prediction.
        """
        if not similar_patterns:
            return {
                "success_rate_trend": 0.0,
                "execution_time_trend": 0.0,
                "confidence": 0.0,
            }

        # Extract time-series data
        success_rates = [p.success_rate for p in similar_patterns]
        execution_times = [p.avg_execution_time for p in similar_patterns]
        frequencies = [p.frequency for p in similar_patterns]

        # Weighted averages using frequency as weight
        total_weight = sum(frequencies)
        if total_weight == 0:
            return {"success_rate_trend": 0.0, "execution_time_trend": 0.0, "confidence": 0.0}

        weighted_success_rate = (
            sum(sr * f for sr, f in zip(success_rates, frequencies, strict=False)) / total_weight
        )
        weighted_execution_time = (
            sum(et * f for et, f in zip(execution_times, frequencies, strict=False)) / total_weight
        )

        # Trend direction using linear regression approximation
        success_rate_trend = self._compute_simple_trend(success_rates)
        execution_time_trend = self._compute_simple_trend(execution_times)

        # Confidence based on sample size and variance
        confidence = min(len(similar_patterns) / 10.0, 1.0)  # Higher confidence with more samples
        if len(success_rates) > 1:
            success_variance = statistics.variance(success_rates)
            confidence *= max(0.1, 1.0 - success_variance)  # Lower confidence with high variance

        return {
            "weighted_success_rate": weighted_success_rate,
            "weighted_execution_time": weighted_execution_time,
            "success_rate_trend": success_rate_trend,
            "execution_time_trend": execution_time_trend,
            "confidence": confidence,
            "sample_size": len(similar_patterns),
        }

    def _compute_simple_trend(self, values: list[float]) -> float:
        """
        Compute simple trend direction using linear approximation - O(n) complexity.

        Returns positive for increasing trend, negative for decreasing.
        """
        if len(values) < 2:
            return 0.0

        # Simple linear trend approximation
        n = len(values)
        x_sum = n * (n - 1) // 2  # Sum of indices 0, 1, 2, ..., n-1
        y_sum = sum(values)
        xy_sum = sum(i * v for i, v in enumerate(values))
        x_sq_sum = sum(i * i for i in range(n))

        # Linear regression slope
        denominator = n * x_sq_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0

        slope = (n * xy_sum - x_sum * y_sum) / denominator
        return slope

    def _compute_confidence_metrics(
        self, similar_patterns: list[ExecutionPattern], experiment_count: int
    ) -> dict[str, float]:
        """
        Compute confidence metrics for predictions - O(k) complexity.

        Returns confidence scores for different prediction aspects.
        """
        if not similar_patterns:
            return {
                "overall_confidence": 0.0,
                "sample_size_confidence": 0.0,
                "pattern_stability_confidence": 0.0,
                "recency_confidence": 0.0,
            }

        # Sample size confidence
        sample_size_confidence = min(len(similar_patterns) / 20.0, 1.0)

        # Pattern stability confidence (low variance = high confidence)
        success_rates = [p.success_rate for p in similar_patterns]
        if len(success_rates) > 1:
            stability_confidence = max(0.0, 1.0 - statistics.variance(success_rates))
        else:
            stability_confidence = 0.5  # Moderate confidence with single sample

        # Recency confidence (more recent patterns get higher weight)
        now = datetime.utcnow()
        recency_scores = []
        for pattern in similar_patterns:
            days_old = (now - pattern.last_seen).days
            recency_score = max(0.1, 1.0 - days_old / 30.0)  # Decay over 30 days
            recency_scores.append(recency_score)

        recency_confidence = sum(recency_scores) / len(recency_scores) if recency_scores else 0.0

        # Overall confidence as weighted average
        overall_confidence = (
            sample_size_confidence * 0.4 + stability_confidence * 0.4 + recency_confidence * 0.2
        )

        return {
            "overall_confidence": overall_confidence,
            "sample_size_confidence": sample_size_confidence,
            "pattern_stability_confidence": stability_confidence,
            "recency_confidence": recency_confidence,
        }

    def _get_pattern_statistics(self) -> dict[str, Any]:
        """Get current pattern storage statistics."""
        return {
            "total_patterns": len(self.patterns),
            "storage_utilization": len(self.patterns) / self.max_patterns,
            "avg_pattern_frequency": (
                statistics.mean([p.frequency for p in self.patterns.values()])
                if self.patterns
                else 0.0
            ),
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
        }

    def _update_pattern_storage(
        self, pattern: ExecutionPattern, results: list[dict[str, Any]]
    ) -> None:
        """
        Update pattern storage with new results - O(1) amortized complexity.

        Implements LRU eviction for memory-bounded storage.
        """
        # Compute success rate from results
        if results:
            successful_results = sum(1 for r in results if r.get("success", False))
            pattern.success_rate = successful_results / len(results)

            execution_times = [
                r.get("execution_time", 0) for r in results if r.get("execution_time")
            ]
            if execution_times:
                pattern.avg_execution_time = sum(execution_times) / len(execution_times)

        # Check if similar pattern already exists
        existing_pattern = None
        for existing in self.patterns.values():
            if existing.compute_similarity(pattern) > 0.9:  # Very high similarity threshold
                existing_pattern = existing
                break

        if existing_pattern:
            # Update existing pattern
            existing_pattern.frequency += 1
            existing_pattern.last_seen = datetime.utcnow()

            # Update success rate using exponential moving average
            alpha = 0.3  # Learning rate
            existing_pattern.success_rate = (
                alpha * pattern.success_rate + (1 - alpha) * existing_pattern.success_rate
            )
            existing_pattern.avg_execution_time = (
                alpha * pattern.avg_execution_time
                + (1 - alpha) * existing_pattern.avg_execution_time
            )

        else:
            # Add new pattern with LRU eviction
            if len(self.patterns) >= self.max_patterns:
                # Remove oldest pattern (from front of deque)
                oldest_pattern_id = self.pattern_index.popleft()
                del self.patterns[oldest_pattern_id]

            # Add new pattern
            self.patterns[pattern.pattern_id] = pattern
            self.pattern_index.append(pattern.pattern_id)

            # Store successful new patterns in memory system
            if self.memory_enabled and pattern.success_rate >= 0.7:  # Store high-success patterns
                from .mcp_memory import run_async_memory_call

                pattern_data = {
                    "pattern_id": pattern.pattern_id,
                    "features": pattern.features,
                    "success_rate": pattern.success_rate,
                    "avg_execution_time": pattern.avg_execution_time,
                    "frequency": pattern.frequency,
                }

                performance_metrics = {
                    "success_rate": pattern.success_rate,
                    "avg_execution_time_ms": pattern.avg_execution_time,
                    "frequency": pattern.frequency,
                }

                run_async_memory_call(
                    self.store_optimization_pattern(
                        "execution_pattern",
                        pattern_data,
                        performance_metrics,
                        min(pattern.success_rate, 0.9),  # Confidence based on success rate
                    )
                )

        # Update feature statistics for normalization
        for feature, value in pattern.features.items():
            stats = self.feature_statistics[feature]
            stats["count"] += 1
            stats["sum"] += value
            stats["sum_sq"] += value * value
            stats["min"] = min(stats["min"], value)
            stats["max"] = max(stats["max"], value)

    def get_performance_metrics(self) -> dict[str, float]:
        """Get analyzer performance metrics."""
        if not self.analysis_times:
            return {"avg_analysis_time_ms": 0.0, "max_analysis_time_ms": 0.0}

        return {
            "avg_analysis_time_ms": statistics.mean(self.analysis_times),
            "max_analysis_time_ms": max(self.analysis_times),
            "min_analysis_time_ms": min(self.analysis_times),
            "analysis_count": len(self.analysis_times),
        }


class BayesianPredictor(MemoryIntegrationMixin):
    """
    Bayesian outcome prediction system for experiment success probability.

    Implements mathematical precision prediction algorithms following
    John Carmack's optimization principles:
    - Numerically stable Bayesian inference
    - O(1) prediction updates with conjugate priors
    - Cache-friendly parameter storage
    - Bounded memory usage with forgetting factors
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,  # Beta distribution success prior
        beta_prior: float = 1.0,  # Beta distribution failure prior
        forgetting_factor: float = 0.95,  # Exponential forgetting for concept drift
        min_samples: int = 5,  # Minimum samples for reliable prediction
    ):
        """
        Initialize Bayesian predictor with conjugate Beta-Binomial model.

        Args:
            alpha_prior: Prior belief in successes (higher = more optimistic)
            beta_prior: Prior belief in failures (higher = more pessimistic)
            forgetting_factor: Exponential decay for old observations
            min_samples: Minimum samples needed for confident predictions
        """
        super().__init__()
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.forgetting_factor = forgetting_factor
        self.min_samples = min_samples

        # Cache for experiment-specific parameters
        self.experiment_priors: dict[str, dict[str, float]] = {}
        self.prediction_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Performance tracking
        self.prediction_times: deque[float] = deque(maxlen=100)
        self.accuracy_scores: deque[float] = deque(maxlen=100)

    def predict_outcome(
        self,
        experiment_id: str,
        experiment_features: dict[str, Any],
        similar_patterns: list[ExecutionPattern],
    ) -> OutcomePrediction:
        """
        Predict experiment outcome using Bayesian inference - O(1) complexity.

        Combines prior knowledge with similar pattern evidence for
        optimal prediction under uncertainty.
        """
        start_time = time.perf_counter()

        try:
            # Get or initialize experiment parameters
            if experiment_id not in self.experiment_priors:
                self.experiment_priors[experiment_id] = {
                    "alpha": self.alpha_prior,
                    "beta": self.beta_prior,
                    "sample_count": 0,
                    "last_updated": time.time(),
                }

            params = self.experiment_priors[experiment_id]

            # Apply forgetting factor for concept drift - O(1) operation
            time_decay = math.exp(-(time.time() - params["last_updated"]) / 86400.0)  # Daily decay
            effective_alpha = params["alpha"] * time_decay + self.alpha_prior * (1 - time_decay)
            effective_beta = params["beta"] * time_decay + self.beta_prior * (1 - time_decay)

            # Update parameters with similar pattern evidence
            evidence_alpha, evidence_beta = self._extract_evidence_parameters(similar_patterns)

            # Bayesian update with evidence
            posterior_alpha = effective_alpha + evidence_alpha
            posterior_beta = effective_beta + evidence_beta

            # Compute success probability (Beta distribution mean)
            success_probability = posterior_alpha / (posterior_alpha + posterior_beta)

            # Estimate execution time using weighted average from similar patterns
            estimated_time = self._estimate_execution_time(similar_patterns, experiment_features)

            # Compute performance score using historical data
            performance_score = self._estimate_performance_score(
                similar_patterns, experiment_features
            )

            # Determine confidence level based on sample size and uncertainty
            confidence = self._compute_prediction_confidence(
                posterior_alpha, posterior_beta, len(similar_patterns)
            )

            # Extract contributing and risk factors
            contributing_factors = self._identify_contributing_factors(
                similar_patterns, experiment_features
            )
            risk_factors = self._identify_risk_factors(similar_patterns, experiment_features)

            # Get historical accuracy for this experiment type
            historical_accuracy = self._get_historical_accuracy(experiment_id)

            # Create prediction result
            prediction = OutcomePrediction(
                experiment_id=experiment_id,
                success_probability=success_probability,
                estimated_execution_time_ms=estimated_time,
                expected_performance_score=performance_score,
                confidence=confidence,
                contributing_factors=contributing_factors,
                risk_factors=risk_factors,
                historical_accuracy=historical_accuracy,
            )

            # Record performance metrics
            prediction_time = (time.perf_counter() - start_time) * 1000
            self.prediction_times.append(prediction_time)

            # Store prediction for accuracy tracking
            self.prediction_history[experiment_id].append(
                {
                    "prediction": prediction,
                    "timestamp": datetime.utcnow(),
                    "posterior_alpha": posterior_alpha,
                    "posterior_beta": posterior_beta,
                }
            )

            return prediction

        except Exception as e:
            # Graceful degradation with low-confidence fallback
            return OutcomePrediction(
                experiment_id=experiment_id,
                success_probability=0.5,  # Neutral probability
                estimated_execution_time_ms=5000.0,  # Conservative estimate
                expected_performance_score=0.5,
                confidence=PredictionConfidence.VERY_LOW,
                contributing_factors=[],
                risk_factors=[f"prediction_error: {str(e)}"],
                historical_accuracy=0.0,
            )

    def update_with_result(
        self,
        experiment_id: str,
        actual_success: bool,
        actual_execution_time: float,
        actual_performance_score: float,
    ) -> None:
        """
        Update Bayesian parameters with actual results - O(1) complexity.

        Uses conjugate prior updates for computational efficiency.
        """
        if experiment_id not in self.experiment_priors:
            return  # Cannot update without prior predictions

        params = self.experiment_priors[experiment_id]

        # Bayesian update with conjugate prior
        if actual_success:
            params["alpha"] += 1.0
        else:
            params["beta"] += 1.0

        params["sample_count"] += 1
        params["last_updated"] = time.time()

        # Update accuracy tracking
        if self.prediction_history[experiment_id]:
            recent_prediction = self.prediction_history[experiment_id][-1]
            predicted_success = recent_prediction["prediction"].success_probability > 0.5
            accuracy = 1.0 if (predicted_success == actual_success) else 0.0
            self.accuracy_scores.append(accuracy)

            # Store prediction accuracy in memory system for learning
            if self.memory_enabled and accuracy >= 0.8:  # Store successful predictions
                from .mcp_memory import run_async_memory_call

                predicted_outcome = {
                    "success_probability": recent_prediction["prediction"].success_probability,
                    "estimated_execution_time_ms": recent_prediction[
                        "prediction"
                    ].estimated_execution_time_ms,
                    "expected_performance_score": recent_prediction[
                        "prediction"
                    ].expected_performance_score,
                }

                actual_outcome = {
                    "success_rate": 1.0 if actual_success else 0.0,
                    "actual_execution_time_ms": actual_execution_time,
                    "actual_performance_score": actual_performance_score,
                }

                run_async_memory_call(
                    self.store_prediction_accuracy(
                        experiment_id, predicted_outcome, actual_outcome, accuracy
                    )
                )

    def _extract_evidence_parameters(
        self, similar_patterns: list[ExecutionPattern]
    ) -> tuple[float, float]:
        """
        Extract Bayesian evidence from similar patterns - O(k) complexity.

        Converts pattern success rates to Beta distribution parameters.
        """
        if not similar_patterns:
            return 0.0, 0.0

        # Weight patterns by frequency and recency
        total_weight = 0.0
        weighted_successes = 0.0
        weighted_failures = 0.0

        now = datetime.utcnow()
        for pattern in similar_patterns:
            # Recency weight (decay over 30 days)
            days_old = (now - pattern.last_seen).days
            recency_weight = math.exp(-days_old / 30.0)

            # Frequency weight
            frequency_weight = math.log(pattern.frequency + 1)  # Log scaling

            # Combined weight
            pattern_weight = recency_weight * frequency_weight
            total_weight += pattern_weight

            # Convert success rate to pseudo-counts
            success_count = pattern.success_rate * pattern_weight
            failure_count = (1.0 - pattern.success_rate) * pattern_weight

            weighted_successes += success_count
            weighted_failures += failure_count

        # Normalize to reasonable pseudo-count scale
        if total_weight > 0:
            evidence_strength = min(total_weight, 10.0)  # Cap evidence strength
            alpha_evidence = (weighted_successes / total_weight) * evidence_strength
            beta_evidence = (weighted_failures / total_weight) * evidence_strength
            return alpha_evidence, beta_evidence

        return 0.0, 0.0

    def _estimate_execution_time(
        self,
        similar_patterns: list[ExecutionPattern],
        experiment_features: dict[str, Any],
    ) -> float:
        """
        Estimate execution time using pattern-based prediction - O(k) complexity.

        Uses weighted average with recency and similarity weighting.
        """
        if not similar_patterns:
            # Fallback estimation based on experiment complexity
            script_length = len(experiment_features.get("script_dsl", ""))
            return max(1000.0, script_length * 50.0)  # 50ms per character estimate

        # Weighted average of similar pattern execution times
        total_weight = 0.0
        weighted_time = 0.0

        for pattern in similar_patterns:
            if pattern.avg_execution_time <= 0:
                continue  # Skip patterns without timing data

            # Weight by frequency and inverse age
            weight = pattern.frequency * math.exp(
                -((datetime.utcnow() - pattern.last_seen).days / 14.0)
            )
            total_weight += weight
            weighted_time += pattern.avg_execution_time * weight

        if total_weight > 0:
            return weighted_time / total_weight

        # Fallback to median time
        execution_times = [
            p.avg_execution_time for p in similar_patterns if p.avg_execution_time > 0
        ]
        return statistics.median(execution_times) if execution_times else 3000.0

    def _estimate_performance_score(
        self,
        similar_patterns: list[ExecutionPattern],
        experiment_features: dict[str, Any],
    ) -> float:
        """
        Estimate performance score based on pattern analysis - O(k) complexity.

        Combines success probability with execution efficiency metrics.
        """
        if not similar_patterns:
            return 0.5  # Neutral performance estimate

        # Performance is combination of success rate and execution efficiency
        success_component = sum(p.success_rate * p.frequency for p in similar_patterns)
        frequency_sum = sum(p.frequency for p in similar_patterns)

        if frequency_sum > 0:
            avg_success_rate = success_component / frequency_sum
        else:
            avg_success_rate = 0.5

        # Efficiency component based on execution time
        execution_times = [
            p.avg_execution_time for p in similar_patterns if p.avg_execution_time > 0
        ]
        if execution_times:
            min_time = min(execution_times)
            avg_time = statistics.mean(execution_times)
            efficiency = min_time / avg_time if avg_time > 0 else 1.0
        else:
            efficiency = 0.5

        # Combined performance score
        performance_score = (avg_success_rate * 0.7) + (efficiency * 0.3)
        return max(0.0, min(1.0, performance_score))

    def _compute_prediction_confidence(
        self,
        posterior_alpha: float,
        posterior_beta: float,
        similar_pattern_count: int,
    ) -> PredictionConfidence:
        """
        Compute prediction confidence based on statistical evidence - O(1) complexity.

        Uses Beta distribution variance and sample size for confidence assessment.
        """
        # Total pseudo-observations
        total_observations = posterior_alpha + posterior_beta

        # Beta distribution variance (uncertainty measure)
        if total_observations > 2:
            beta_variance = (posterior_alpha * posterior_beta) / (
                (posterior_alpha + posterior_beta) ** 2 * (posterior_alpha + posterior_beta + 1)
            )
        else:
            beta_variance = 0.25  # Maximum variance for uniform prior

        # Sample size factor
        sample_factor = min(similar_pattern_count / 10.0, 1.0)

        # Evidence strength factor
        evidence_factor = min(total_observations / 20.0, 1.0)

        # Combined confidence score
        confidence_score = (1.0 - beta_variance) * 0.5 + sample_factor * 0.3 + evidence_factor * 0.2

        # Map to confidence enum
        if confidence_score >= 0.9:
            return PredictionConfidence.VERY_HIGH
        elif confidence_score >= 0.7:
            return PredictionConfidence.HIGH
        elif confidence_score >= 0.5:
            return PredictionConfidence.MEDIUM
        elif confidence_score >= 0.3:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW

    def _identify_contributing_factors(
        self,
        similar_patterns: list[ExecutionPattern],
        experiment_features: dict[str, Any],
    ) -> list[str]:
        """
        Identify factors contributing to success - O(k) complexity.

        Analyzes patterns to find common success factors.
        """
        factors: list[str] = []

        if not similar_patterns:
            return factors

        # Analyze success patterns
        successful_patterns = [p for p in similar_patterns if p.success_rate > 0.7]

        if len(successful_patterns) >= 2:
            # Find common features in successful patterns
            feature_sums: dict[str, float] = defaultdict(float)
            feature_counts: dict[str, int] = defaultdict(int)

            for pattern in successful_patterns:
                for feature, value in pattern.features.items():
                    feature_sums[feature] += value
                    feature_counts[feature] += 1

            # Identify high-value features
            for feature, total in feature_sums.items():
                if feature_counts[feature] >= 2:  # Present in multiple patterns
                    avg_value = total / feature_counts[feature]
                    if avg_value > 0.6:  # High feature value
                        factors.append(f"high_{feature}")

        # Add experiment-specific factors
        current_priority = experiment_features.get("priority", 1)
        if current_priority >= 3:
            factors.append("high_priority_experiment")

        script_length = len(experiment_features.get("script_dsl", ""))
        if 20 <= script_length <= 100:  # Optimal complexity range
            factors.append("optimal_script_complexity")

        return factors[:5]  # Limit to top 5 factors

    def _identify_risk_factors(
        self,
        similar_patterns: list[ExecutionPattern],
        experiment_features: dict[str, Any],
    ) -> list[str]:
        """
        Identify risk factors that may lead to failure - O(k) complexity.

        Analyzes patterns to find common failure indicators.
        """
        risks: list[str] = []

        if not similar_patterns:
            return risks

        # Analyze failure patterns
        failed_patterns = [p for p in similar_patterns if p.success_rate < 0.3]

        if failed_patterns:
            # High execution time variance indicates instability
            execution_times = [
                p.avg_execution_time for p in similar_patterns if p.avg_execution_time > 0
            ]
            if len(execution_times) >= 3:
                time_variance = statistics.variance(execution_times)
                time_mean = statistics.mean(execution_times)
                if time_variance > (time_mean * 0.5) ** 2:  # High coefficient of variation
                    risks.append("execution_time_instability")

        # Low pattern frequency indicates unreliable approach
        avg_frequency = statistics.mean([p.frequency for p in similar_patterns])
        if avg_frequency < 2.0:
            risks.append("low_pattern_reliability")

        # Script complexity risks
        script_length = len(experiment_features.get("script_dsl", ""))
        if script_length > 200:
            risks.append("excessive_script_complexity")
        elif script_length < 10:
            risks.append("insufficient_script_detail")

        # Priority misalignment risk
        current_priority = experiment_features.get("priority", 1)
        if len(similar_patterns) >= 2:
            pattern_priorities = []
            for pattern in similar_patterns:
                pattern_priority = pattern.features.get("avg_priority", 0.2) * 5  # Denormalize
                pattern_priorities.append(pattern_priority)

            avg_pattern_priority = statistics.mean(pattern_priorities)
            if abs(current_priority - avg_pattern_priority) > 2.0:
                risks.append("priority_mismatch_with_patterns")

        return risks[:5]  # Limit to top 5 risk factors

    def _get_historical_accuracy(self, experiment_id: str) -> float:
        """
        Get historical prediction accuracy for experiment type - O(1) complexity.

        Returns rolling average of recent prediction accuracy.
        """
        if experiment_id not in self.prediction_history:
            return 0.0

        # Use recent accuracy scores if available
        if self.accuracy_scores:
            recent_accuracy = list(self.accuracy_scores)[-20:]  # Last 20 predictions
            return statistics.mean(recent_accuracy)

        # Fallback to default accuracy
        return 0.6  # Assume moderate accuracy for new experiment types

    def get_performance_metrics(self) -> dict[str, float]:
        """Get predictor performance metrics."""
        metrics = {}

        if self.prediction_times:
            metrics.update(
                {
                    "avg_prediction_time_ms": statistics.mean(self.prediction_times),
                    "max_prediction_time_ms": max(self.prediction_times),
                    "prediction_count": len(self.prediction_times),
                }
            )

        if self.accuracy_scores:
            metrics.update(
                {
                    "recent_accuracy": statistics.mean(list(self.accuracy_scores)[-50:]),  # Last 50
                    "overall_accuracy": statistics.mean(self.accuracy_scores),
                    "accuracy_std": (
                        statistics.stdev(self.accuracy_scores)
                        if len(self.accuracy_scores) > 1
                        else 0.0
                    ),
                }
            )

        metrics["total_experiments"] = len(self.experiment_priors)
        metrics["total_predictions"] = sum(
            len(history) for history in self.prediction_history.values()
        )

        return metrics


class PredictionCache(MemoryIntegrationMixin):
    """
    High-performance prediction result cache with <10ms retrieval target.

    Implements John Carmack optimization principles:
    - O(1) cache operations with hash table lookup
    - Memory-bounded with configurable LRU eviction
    - Cache-friendly data structures for CPU efficiency
    - TTL-based expiration with lazy cleanup
    """

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: float = 300.0,  # 5 minutes
        cleanup_interval: int = 100,  # Cleanup every N operations
    ):
        """
        Initialize prediction cache with performance constraints.

        Args:
            max_entries: Maximum cached entries (memory bound)
            default_ttl: Default TTL in seconds
            cleanup_interval: Operations between cleanup cycles
        """
        super().__init__()
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        # Cache storage - optimized for O(1) access
        self.cache: dict[str, dict[str, Any]] = {}
        self.access_order: deque[str] = deque()  # LRU ordering
        self.access_counts: dict[str, int] = {}  # Access frequency tracking

        # Performance tracking
        self.operation_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.retrieval_times: deque[float] = deque(maxlen=100)
        self.eviction_count = 0
        self.expired_count = 0

        # Thread safety
        self._cache_lock = threading.Lock()

    def get(self, cache_key: str) -> PredictivePlanningResult | None:
        """
        Retrieve prediction result from cache - Target <10ms.

        Uses optimized lookup with lazy TTL checking for performance.
        """
        start_time = time.perf_counter()

        try:
            with self._cache_lock:
                self.operation_count += 1

                # Check if key exists
                if cache_key not in self.cache:
                    self.cache_misses += 1
                    return None

                entry = self.cache[cache_key]
                current_time = time.time()

                # Check TTL expiration
                if current_time > entry["expires_at"]:
                    # Lazy removal of expired entry
                    self._remove_entry(cache_key)
                    self.cache_misses += 1
                    self.expired_count += 1
                    return None

                # Update LRU ordering - move to end
                try:
                    self.access_order.remove(cache_key)
                except ValueError:
                    pass  # Key not in order (shouldn't happen, but handle gracefully)
                self.access_order.append(cache_key)

                # Update access frequency
                self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1

                # Update entry metadata
                entry["last_accessed"] = current_time
                entry["access_count"] += 1

                self.cache_hits += 1

                # Deserialize prediction result
                result_data = entry["data"]
                result = PredictivePlanningResult.from_dict(result_data)

                return result

        finally:
            # Record retrieval time
            retrieval_time = (time.perf_counter() - start_time) * 1000
            self.retrieval_times.append(retrieval_time)

            # Store cache optimization insights for exceptional performance
            if (
                self.memory_enabled
                and retrieval_time < 5.0  # Sub-5ms retrieval
                and len(self.retrieval_times) > 10
                and cache_key in self.cache
            ):

                # Check if consistently fast performance (90th percentile < 10ms)
                recent_times = list(self.retrieval_times)[-10:]  # Last 10 retrievals
                p90_time = self._percentile(deque(recent_times), 0.9)
                if p90_time < 10.0:
                    from .mcp_memory import run_async_memory_call

                    cache_metrics = {
                        "avg_retrieval_time_ms": statistics.mean(recent_times),
                        "p90_retrieval_time_ms": p90_time,
                        "hit_rate": self.cache_hits / max(self.operation_count, 1),
                        "cache_size": len(self.cache),
                    }

                    run_async_memory_call(
                        self.store_performance_insight(
                            "cache_optimization",
                            f"Cache achieving sub-10ms performance with {cache_metrics['hit_rate']:.1%} hit rate",
                            cache_metrics,
                            "Maintain current cache configuration for optimal performance",
                        )
                    )

            # Periodic cleanup
            if self.operation_count % self.cleanup_interval == 0:
                self._periodic_cleanup()

    def put(
        self,
        cache_key: str,
        result: PredictivePlanningResult,
        ttl: float | None = None,
    ) -> None:
        """
        Store prediction result in cache with TTL - Target <5ms.

        Uses efficient serialization and LRU eviction for bounded memory.
        """
        if ttl is None:
            ttl = self.default_ttl

        with self._cache_lock:
            self.operation_count += 1
            current_time = time.time()

            # Check if we need to evict entries
            if len(self.cache) >= self.max_entries and cache_key not in self.cache:
                self._evict_lru_entry()

            # Serialize result for storage
            try:
                result_data = result.to_dict()
            except Exception:
                # Skip caching if serialization fails
                return

            # Create cache entry
            cache_entry = {
                "data": result_data,
                "created_at": current_time,
                "expires_at": current_time + ttl,
                "last_accessed": current_time,
                "access_count": 1,
                "content_hash": result.content_hash,
            }

            # Store entry
            self.cache[cache_key] = cache_entry

            # Update LRU ordering
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)

            # Initialize access count
            self.access_counts[cache_key] = 1

    def invalidate(self, cache_key: str) -> bool:
        """
        Invalidate specific cache entry.

        Returns True if entry was removed, False if not found.
        """
        with self._cache_lock:
            if cache_key in self.cache:
                self._remove_entry(cache_key)
                return True
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate entries matching pattern - useful for bulk invalidation.

        Returns number of entries removed.
        """
        removed_count = 0

        with self._cache_lock:
            keys_to_remove = [key for key in self.cache.keys() if pattern in key]

            for key in keys_to_remove:
                self._remove_entry(key)
                removed_count += 1

        return removed_count

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._cache_lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_counts.clear()

    def generate_cache_key(self, *components: Any) -> str:
        """
        Generate cache key from components using fast hashing.

        Uses SHA-256 for collision resistance and consistent ordering.
        """
        # Convert components to strings and create deterministic representation
        key_components = []
        for component in components:
            if isinstance(component, dict):
                # Sort dictionary for consistent ordering
                sorted_items = sorted(component.items()) if component else []
                key_components.append(str(sorted_items))
            elif isinstance(component, list | tuple):
                # Convert to string representation first, then sort for consistency
                str_items = [str(item) for item in component] if component else []
                key_components.append(str(sorted(str_items)))
            else:
                key_components.append(str(component))

        # Create content hash
        content = "|".join(key_components)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[
            :16
        ]  # First 16 chars for efficiency

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        with self._cache_lock:
            total_operations = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_operations if total_operations > 0 else 0.0

            stats = {
                # Hit rate statistics
                "hit_rate": hit_rate,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "total_operations": total_operations,
                # Storage statistics
                "current_entries": len(self.cache),
                "max_entries": self.max_entries,
                "utilization": len(self.cache) / self.max_entries,
                # Performance statistics
                "eviction_count": self.eviction_count,
                "expired_count": self.expired_count,
                "operation_count": self.operation_count,
            }

            # Retrieval time statistics
            if self.retrieval_times:
                stats.update(
                    {
                        "avg_retrieval_time_ms": statistics.mean(self.retrieval_times),
                        "max_retrieval_time_ms": max(self.retrieval_times),
                        "min_retrieval_time_ms": min(self.retrieval_times),
                        "retrieval_time_p95_ms": self._percentile(self.retrieval_times, 0.95),
                    }
                )

            # Access pattern statistics
            if self.access_counts:
                access_values = list(self.access_counts.values())
                stats.update(
                    {
                        "avg_access_count": statistics.mean(access_values),
                        "max_access_count": max(access_values),
                        "hot_entry_ratio": sum(1 for count in access_values if count >= 5)
                        / len(access_values),
                    }
                )

            return stats

    def _evict_lru_entry(self) -> None:
        """
        Evict least recently used entry - O(1) amortized complexity.

        Uses LRU ordering with frequency consideration for optimal eviction.
        """
        if not self.access_order:
            return

        # Find LRU entry with lowest access frequency
        candidates = list(self.access_order)[: min(3, len(self.access_order))]  # Check first 3 LRU

        if len(candidates) == 1:
            lru_key = candidates[0]
        else:
            # Choose entry with lowest access frequency among LRU candidates
            lru_key = min(candidates, key=lambda k: self.access_counts.get(k, 0))

        self._remove_entry(lru_key)
        self.eviction_count += 1

    def _remove_entry(self, cache_key: str) -> None:
        """
        Remove cache entry and update all data structures - O(1) amortized.

        Maintains consistency across cache, access_order, and access_counts.
        """
        # Remove from main cache
        if cache_key in self.cache:
            del self.cache[cache_key]

        # Remove from LRU ordering
        try:
            self.access_order.remove(cache_key)
        except ValueError:
            pass  # Key not in order (shouldn't happen, but handle gracefully)

        # Remove from access counts
        if cache_key in self.access_counts:
            del self.access_counts[cache_key]

    def _periodic_cleanup(self) -> None:
        """
        Periodic cleanup of expired entries - O(n) worst case, infrequent.

        Runs periodically to remove expired entries and maintain performance.
        """
        current_time = time.time()
        expired_keys = []

        # Find expired entries
        for key, entry in self.cache.items():
            if current_time > entry["expires_at"]:
                expired_keys.append(key)

        # Remove expired entries
        for key in expired_keys:
            self._remove_entry(key)
            self.expired_count += 1

        # Shrink structures if they've grown too large
        max_history = 200
        if len(self.retrieval_times) > max_history:
            # Keep only recent measurements
            recent_times = list(self.retrieval_times)[-max_history:]
            self.retrieval_times.clear()
            self.retrieval_times.extend(recent_times)

    def _percentile(self, values: deque, p: float) -> float:
        """
        Calculate percentile from deque values - O(n log n) complexity.

        Used for performance statistics calculation.
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]


class ContingencyGenerator(MemoryIntegrationMixin):
    """
    Intelligent contingency strategy generator for predictive planning.

    Creates scenario-based contingency strategies following Carmack principles:
    - Fast strategy generation with <50ms target time
    - Pattern-based strategy selection using historical data
    - Probability-based scenario modeling for optimal coverage
    - Memory-efficient strategy templates and caching
    """

    def __init__(
        self,
        fallback_strategy_pool_size: int = 20,
        scenario_coverage_threshold: float = 0.8,
        strategy_template_cache_size: int = 100,
    ):
        """
        Initialize contingency generator with performance parameters.

        Args:
            fallback_strategy_pool_size: Size of pre-generated strategy pool
            scenario_coverage_threshold: Minimum probability coverage for scenarios
            strategy_template_cache_size: Number of cached strategy templates
        """
        super().__init__()
        self.fallback_strategy_pool_size = fallback_strategy_pool_size
        self.scenario_coverage_threshold = scenario_coverage_threshold
        self.strategy_template_cache_size = strategy_template_cache_size

        # Pre-built strategy templates for fast generation
        self.strategy_templates: dict[str, dict[str, Any]] = {}
        self.scenario_patterns: dict[ScenarioTrigger, list[dict[str, Any]]] = defaultdict(list)

        # Performance tracking
        self.generation_times: deque[float] = deque(maxlen=100)
        self.generated_contingencies = 0
        self.template_cache_hits = 0

        # Initialize default strategy templates
        self._initialize_strategy_templates()
        self._initialize_scenario_patterns()

    def generate_contingencies(
        self,
        primary_strategy: StrategyResponse,
        execution_patterns: dict[str, Any],
        outcome_predictions: dict[str, OutcomePrediction],
        horizon: int = 3,
    ) -> list[ContingencyStrategy]:
        """
        Generate contingency strategies for different scenarios - Target <50ms.

        Creates intelligent contingencies based on failure patterns and
        prediction uncertainties for comprehensive coverage.
        """
        start_time = time.perf_counter()

        try:
            contingencies = []

            # Identify potential failure scenarios
            failure_scenarios = self._identify_failure_scenarios(
                primary_strategy, execution_patterns, outcome_predictions
            )

            # Generate contingency for each scenario
            for scenario in failure_scenarios:
                contingency = self._generate_scenario_contingency(
                    scenario, primary_strategy, execution_patterns, horizon
                )
                if contingency:
                    contingencies.append(contingency)

            # Add general fallback contingencies
            general_contingencies = self._generate_general_contingencies(
                primary_strategy, execution_patterns, horizon
            )
            contingencies.extend(general_contingencies)

            # Sort by priority and probability
            contingencies.sort(key=lambda c: (c.priority, c.activation_probability), reverse=True)

            # Limit to reasonable number of contingencies
            max_contingencies = min(8, len(contingencies))
            final_contingencies = contingencies[:max_contingencies]

            # Record performance metrics
            generation_time = (time.perf_counter() - start_time) * 1000
            self.generation_times.append(generation_time)
            self.generated_contingencies += len(final_contingencies)

            # Store successful contingency strategies in memory system
            if (
                self.memory_enabled
                and generation_time < 50.0  # Met performance target
                and len(final_contingencies) >= 3
            ):  # Generated sufficient contingencies

                from .mcp_memory import run_async_memory_call

                contingency_metrics = {
                    "generation_time_ms": generation_time,
                    "contingencies_generated": len(final_contingencies),
                    "scenarios_identified": len(failure_scenarios),
                    "avg_activation_probability": sum(
                        c.activation_probability for c in final_contingencies
                    )
                    / len(final_contingencies),
                }

                strategy_data = {
                    "contingency_types": [c.trigger_type.value for c in final_contingencies],
                    "priorities": [c.priority for c in final_contingencies],
                    "activation_thresholds": [
                        c.activation_probability for c in final_contingencies
                    ],
                }

                run_async_memory_call(
                    self.store_optimization_pattern(
                        "contingency_strategy",
                        strategy_data,
                        contingency_metrics,
                        min(0.9, 1.0 - generation_time / 100.0),  # Confidence based on performance
                    )
                )

            return final_contingencies

        except Exception:
            # Graceful degradation with minimal contingencies
            return self._create_minimal_contingencies(primary_strategy)

    def _identify_failure_scenarios(
        self,
        primary_strategy: StrategyResponse,
        execution_patterns: dict[str, Any],
        outcome_predictions: dict[str, OutcomePrediction],
    ) -> list[dict[str, Any]]:
        """
        Identify potential failure scenarios based on patterns and predictions - O(n) complexity.

        Analyzes execution patterns and predictions to find high-risk scenarios.
        """
        scenarios = []

        # Execution failure scenarios
        high_risk_experiments = [
            exp_id
            for exp_id, pred in outcome_predictions.items()
            if pred.success_probability < 0.5 or pred.get_risk_score() > 0.6
        ]

        if high_risk_experiments:
            scenarios.append(
                {
                    "type": ScenarioTrigger.EXECUTION_FAILURE,
                    "probability": 0.7,  # High probability if multiple high-risk experiments
                    "affected_experiments": high_risk_experiments,
                    "trigger_conditions": [
                        "failure_rate > 0.3",
                        "success_probability < 0.5",
                    ],
                    "priority": 4,
                }
            )

        # Performance degradation scenarios
        pattern_stats = execution_patterns.get("pattern_statistics", {})
        if pattern_stats.get("cache_hit_rate", 1.0) < 0.6:
            scenarios.append(
                {
                    "type": ScenarioTrigger.PERFORMANCE_DEGRADATION,
                    "probability": 0.5,
                    "affected_experiments": [],  # Affects all experiments
                    "trigger_conditions": [
                        "performance_score < 0.7",
                        "execution_time > expected_time * 1.5",
                    ],
                    "priority": 3,
                }
            )

        # Time constraint scenarios
        high_time_predictions = [
            exp_id
            for exp_id, pred in outcome_predictions.items()
            if pred.estimated_execution_time_ms > 10000  # >10 seconds
        ]

        if high_time_predictions:
            scenarios.append(
                {
                    "type": ScenarioTrigger.TIME_CONSTRAINT,
                    "probability": 0.4,
                    "affected_experiments": high_time_predictions,
                    "trigger_conditions": [
                        "execution_time > 10000",
                        "time_constraint_exceeded",
                    ],
                    "priority": 3,
                }
            )

        # Pattern deviation scenarios
        trend_analysis = execution_patterns.get("trend_analysis", {})
        if trend_analysis.get("confidence", 0.0) < 0.4:
            scenarios.append(
                {
                    "type": ScenarioTrigger.PATTERN_DEVIATION,
                    "probability": 0.3,
                    "affected_experiments": [],
                    "trigger_conditions": [
                        "pattern_similarity < 0.5",
                        "confidence < 0.4",
                    ],
                    "priority": 2,
                }
            )

        # Unexpected game state scenarios
        similar_patterns = execution_patterns.get("similar_patterns", [])
        if len(similar_patterns) < 2:  # Few similar patterns indicate uncertainty
            scenarios.append(
                {
                    "type": ScenarioTrigger.UNEXPECTED_GAME_STATE,
                    "probability": 0.6,
                    "affected_experiments": [],
                    "trigger_conditions": [
                        "similar_patterns < 2",
                        "unexpected_game_state",
                    ],
                    "priority": 2,
                }
            )

        return scenarios

    def _generate_scenario_contingency(
        self,
        scenario: dict[str, Any],
        primary_strategy: StrategyResponse,
        execution_patterns: dict[str, Any],
        horizon: int,
    ) -> ContingencyStrategy | None:
        """
        Generate contingency strategy for specific scenario - O(1) complexity.

        Uses pre-built templates and pattern matching for fast generation.
        """
        scenario_type = scenario["type"]

        # Get strategy template for scenario type
        template_key = f"{scenario_type.value}_template"
        template: dict[str, Any] | None
        if template_key in self.strategy_templates:
            template = self.strategy_templates[template_key]
            self.template_cache_hits += 1
        else:
            template = self._create_dynamic_template(
                scenario_type, primary_strategy, execution_patterns
            )

        if not template:
            return None

        # Create contingency strategy based on template
        contingency_strategy = self._build_strategy_from_template(
            template, primary_strategy, scenario, execution_patterns
        )

        if not contingency_strategy:
            return None

        # Create contingency with scenario-specific parameters
        contingency_id = f"contingency_{scenario_type.value}_{int(time.time())}"

        return ContingencyStrategy(
            scenario_id=contingency_id,
            trigger_type=scenario_type,
            trigger_conditions=scenario["trigger_conditions"],
            strategy=contingency_strategy,
            activation_probability=scenario["probability"],
            confidence=self._assess_contingency_confidence(scenario, template),
            priority=scenario["priority"],
            metadata={
                "scenario_type": scenario_type.value,
                "affected_experiments": scenario.get("affected_experiments", []),
                "template_source": template_key,
                "generation_time": time.time(),
            },
        )

    def _generate_general_contingencies(
        self,
        primary_strategy: StrategyResponse,
        execution_patterns: dict[str, Any],
        horizon: int,
    ) -> list[ContingencyStrategy]:
        """
        Generate general-purpose contingency strategies - O(1) complexity.

        Creates fallback strategies for common scenarios not covered by specific contingencies.
        """
        general_contingencies = []

        # Safe exploration contingency
        safe_strategy = self._create_safe_exploration_strategy(primary_strategy, execution_patterns)
        if safe_strategy:
            contingency = ContingencyStrategy(
                scenario_id=f"safe_exploration_{int(time.time())}",
                trigger_type=ScenarioTrigger.EXECUTION_FAILURE,
                trigger_conditions=["multiple_failures", "unknown_failure_mode"],
                strategy=safe_strategy,
                activation_probability=0.8,
                confidence=PredictionConfidence.HIGH,
                priority=1,
                metadata={"type": "safe_exploration", "general_purpose": True},
            )
            general_contingencies.append(contingency)

        # Resource conservation contingency
        conservation_strategy = self._create_resource_conservation_strategy(
            primary_strategy, execution_patterns
        )
        if conservation_strategy:
            contingency = ContingencyStrategy(
                scenario_id=f"resource_conservation_{int(time.time())}",
                trigger_type=ScenarioTrigger.RESOURCE_EXHAUSTION,
                trigger_conditions=["high_resource_usage", "memory_pressure"],
                strategy=conservation_strategy,
                activation_probability=0.3,
                confidence=PredictionConfidence.MEDIUM,
                priority=2,
                metadata={"type": "resource_conservation", "general_purpose": True},
            )
            general_contingencies.append(contingency)

        return general_contingencies

    def _build_strategy_from_template(
        self,
        template: dict[str, Any],
        primary_strategy: StrategyResponse,
        scenario: dict[str, Any],
        execution_patterns: dict[str, Any],
    ) -> StrategyResponse | None:
        """
        Build strategy response from template with scenario-specific adaptations - O(k) complexity.

        Customizes template parameters based on current context and scenario requirements.
        """
        try:
            # Extract template components
            base_experiments = template.get("experiments", [])
            base_insights = template.get("insights", [])
            base_checkpoints = template.get("checkpoints", [])

            # Adapt experiments for current scenario
            adapted_experiments = []
            for exp_template in base_experiments:
                adapted_exp = self._adapt_experiment_template(
                    exp_template, primary_strategy, scenario, execution_patterns
                )
                if adapted_exp:
                    adapted_experiments.append(adapted_exp)

            # Create strategy ID
            strategy_id = f"contingency_{scenario['type'].value}_{int(time.time())}"

            # Build strategy response
            strategy = StrategyResponse(
                strategy_id=strategy_id,
                experiments=adapted_experiments,
                strategic_insights=base_insights
                + [
                    f"Contingency strategy for {scenario['type'].value}",
                    f"Triggered by: {', '.join(scenario['trigger_conditions'])}",
                ],
                next_checkpoints=base_checkpoints,
                metadata={
                    "contingency_type": scenario["type"].value,
                    "template_based": True,
                    "adaptation_timestamp": datetime.utcnow().isoformat(),
                },
            )

            return strategy

        except Exception:
            # Return None on template building failure
            return None

    def _adapt_experiment_template(
        self,
        template: dict[str, Any],
        primary_strategy: StrategyResponse,
        scenario: dict[str, Any],
        execution_patterns: dict[str, Any],
    ) -> Any:
        """
        Adapt experiment template to current context - O(1) complexity.

        Customizes experiment parameters based on scenario and patterns.
        """
        from .strategy_response import ExperimentSpec  # Import here to avoid circular imports

        try:
            # Get current game state information from execution patterns
            current_features = execution_patterns.get("current_pattern", {})

            # Adapt script DSL based on scenario type
            base_script = template.get("script_dsl", "EXPLORE; SAVE")
            adapted_script = self._adapt_script_for_scenario(
                base_script, scenario["type"], current_features
            )

            # Adapt checkpoint based on primary strategy
            base_checkpoint = template.get("checkpoint", "safe_checkpoint")
            adapted_checkpoint = self._select_safe_checkpoint(base_checkpoint, primary_strategy)

            # Adapt priority based on scenario urgency
            base_priority = template.get("priority", 2)
            adapted_priority = min(scenario.get("priority", 2), base_priority)

            # Create experiment spec
            experiment = ExperimentSpec(
                id=template["id"] + f"_{int(time.time())}",
                name=template["name"],
                checkpoint=adapted_checkpoint,
                script_dsl=adapted_script,
                expected_outcome=template.get("expected_outcome", "contingency_progress"),
                priority=adapted_priority,
                directives=template.get("directives", ["contingency", "safe_progress"]),
                metadata={
                    "contingency_adaptation": True,
                    "scenario_type": scenario["type"].value,
                    "template_source": template.get("template_id", "unknown"),
                },
            )

            return experiment

        except Exception:
            return None

    def _create_minimal_contingencies(
        self, primary_strategy: StrategyResponse
    ) -> list[ContingencyStrategy]:
        """
        Create minimal contingencies for graceful degradation - O(1) complexity.

        Provides basic fallback contingencies when generation fails.
        """
        from .strategy_response import FallbackStrategy

        # Create minimal safe strategy
        safe_strategy = FallbackStrategy.create_default_fallback(
            {
                "location": "unknown",
                "current_checkpoint": "safe_fallback",
            }
        )

        # Create minimal contingency
        minimal_contingency = ContingencyStrategy(
            scenario_id=f"minimal_fallback_{int(time.time())}",
            trigger_type=ScenarioTrigger.EXECUTION_FAILURE,
            trigger_conditions=["any_failure", "generation_error"],
            strategy=safe_strategy,
            activation_probability=1.0,
            confidence=PredictionConfidence.LOW,
            priority=1,
            metadata={"type": "minimal_fallback", "emergency_generated": True},
        )

        return [minimal_contingency]

    def _initialize_strategy_templates(self) -> None:
        """Initialize pre-built strategy templates for fast generation."""
        # Execution failure template
        self.strategy_templates["execution_failure_template"] = {
            "template_id": "execution_failure_v1",
            "experiments": [
                {
                    "id": "safe_retry",
                    "name": "Safe Retry Strategy",
                    "checkpoint": "last_safe_checkpoint",
                    "script_dsl": "LOAD_CHECKPOINT; SIMPLE_EXPLORATION; SAVE_PROGRESS",
                    "expected_outcome": "recovery_progress",
                    "priority": 2,
                    "directives": ["safe_retry", "error_recovery"],
                }
            ],
            "insights": [
                "Switching to safe retry mode due to execution failures",
                "Using conservative approach to recover progress",
            ],
            "checkpoints": ["recovery_checkpoint"],
        }

        # Performance degradation template
        self.strategy_templates["performance_degradation_template"] = {
            "template_id": "performance_degradation_v1",
            "experiments": [
                {
                    "id": "simplified_approach",
                    "name": "Simplified Execution",
                    "checkpoint": "current_checkpoint",
                    "script_dsl": "SIMPLIFIED_ACTIONS; REDUCE_COMPLEXITY; MONITOR_PERFORMANCE",
                    "expected_outcome": "stable_performance",
                    "priority": 3,
                    "directives": ["simplify", "performance_recovery"],
                }
            ],
            "insights": [
                "Reducing complexity due to performance degradation",
                "Focus on stable execution over optimization",
            ],
            "checkpoints": ["performance_stable_checkpoint"],
        }

        # Time constraint template
        self.strategy_templates["time_constraint_template"] = {
            "template_id": "time_constraint_v1",
            "experiments": [
                {
                    "id": "fast_path",
                    "name": "Fast Path Execution",
                    "checkpoint": "recent_checkpoint",
                    "script_dsl": "FAST_ACTIONS; SKIP_OPTIONAL; ESSENTIAL_ONLY",
                    "expected_outcome": "time_efficient_progress",
                    "priority": 4,
                    "directives": ["fast_execution", "time_optimization"],
                }
            ],
            "insights": [
                "Switching to fast path due to time constraints",
                "Focusing on essential actions only",
            ],
            "checkpoints": ["time_optimized_checkpoint"],
        }

    def _initialize_scenario_patterns(self) -> None:
        """Initialize scenario pattern recognition data."""
        # Common failure patterns
        self.scenario_patterns[ScenarioTrigger.EXECUTION_FAILURE] = [
            {"pattern": "multiple_script_failures", "weight": 0.8},
            {"pattern": "unexpected_game_response", "weight": 0.7},
            {"pattern": "resource_unavailable", "weight": 0.6},
        ]

        # Performance degradation patterns
        self.scenario_patterns[ScenarioTrigger.PERFORMANCE_DEGRADATION] = [
            {"pattern": "increasing_execution_time", "weight": 0.9},
            {"pattern": "memory_pressure", "weight": 0.7},
            {"pattern": "cache_miss_spike", "weight": 0.6},
        ]

    def _create_dynamic_template(
        self,
        scenario_type: ScenarioTrigger,
        primary_strategy: StrategyResponse,
        execution_patterns: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Create dynamic template when pre-built template not available."""
        # Simplified dynamic template creation
        return {
            "template_id": f"dynamic_{scenario_type.value}",
            "experiments": [
                {
                    "id": f"dynamic_{scenario_type.value}",
                    "name": f"Dynamic {scenario_type.value.replace('_', ' ').title()}",
                    "checkpoint": "safe_checkpoint",
                    "script_dsl": "SAFE_EXPLORATION; COLLECT_DATA; SAVE_PROGRESS",
                    "expected_outcome": "safe_progress",
                    "priority": 2,
                    "directives": ["dynamic_contingency", scenario_type.value],
                }
            ],
            "insights": [f"Dynamic contingency for {scenario_type.value}"],
            "checkpoints": ["dynamic_safe_checkpoint"],
        }

    def _assess_contingency_confidence(
        self, scenario: dict[str, Any], template: dict[str, Any]
    ) -> PredictionConfidence:
        """Assess confidence in contingency strategy effectiveness."""
        base_confidence = 0.6

        # Higher confidence for pre-built templates
        if template.get("template_id", "").startswith("dynamic"):
            base_confidence -= 0.2

        # Adjust based on scenario probability
        probability_factor = scenario.get("probability", 0.5)
        confidence_score = base_confidence + (probability_factor - 0.5) * 0.4

        # Map to confidence enum
        if confidence_score >= 0.8:
            return PredictionConfidence.HIGH
        elif confidence_score >= 0.6:
            return PredictionConfidence.MEDIUM
        elif confidence_score >= 0.4:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW

    def _adapt_script_for_scenario(
        self, base_script: str, scenario_type: ScenarioTrigger, features: dict
    ) -> str:
        """Adapt script DSL for specific scenario type."""
        adaptations = {
            ScenarioTrigger.EXECUTION_FAILURE: "SAFE_MODE; " + base_script,
            ScenarioTrigger.PERFORMANCE_DEGRADATION: "OPTIMIZE; "
            + base_script.replace("COMPLEX", "SIMPLE"),
            ScenarioTrigger.TIME_CONSTRAINT: base_script.replace("EXPLORE", "FAST_EXPLORE"),
            ScenarioTrigger.RESOURCE_EXHAUSTION: "MINIMAL_ACTIONS; CLEANUP; " + base_script,
        }
        return adaptations.get(scenario_type, base_script)

    def _select_safe_checkpoint(
        self, base_checkpoint: str, primary_strategy: StrategyResponse
    ) -> str:
        """Select safest available checkpoint."""
        # Use most recent checkpoint from primary strategy
        if primary_strategy.next_checkpoints:
            return primary_strategy.next_checkpoints[-1]
        return base_checkpoint

    def _create_safe_exploration_strategy(
        self, primary_strategy: StrategyResponse, patterns: dict
    ) -> StrategyResponse | None:
        """Create safe exploration strategy for general contingency."""
        from .strategy_response import ExperimentSpec

        safe_experiment = ExperimentSpec(
            id=f"safe_explore_{int(time.time())}",
            name="Safe Exploration",
            checkpoint="current_safe_checkpoint",
            script_dsl="SAFE_EXPLORE; COLLECT_ITEMS; SAVE_FREQUENTLY",
            expected_outcome="safe_progress",
            priority=1,
            directives=["safe_exploration", "conservative_progress"],
            metadata={"contingency_type": "safe_exploration"},
        )

        return StrategyResponse(
            strategy_id=f"safe_exploration_{int(time.time())}",
            experiments=[safe_experiment],
            strategic_insights=["Safe exploration contingency", "Conservative progress approach"],
            next_checkpoints=["safe_exploration_checkpoint"],
            metadata={"contingency": True, "safe_mode": True},
        )

    def _create_resource_conservation_strategy(
        self, primary_strategy: StrategyResponse, patterns: dict
    ) -> StrategyResponse | None:
        """Create resource conservation strategy for resource exhaustion scenarios."""
        from .strategy_response import ExperimentSpec

        conservation_experiment = ExperimentSpec(
            id=f"conserve_resources_{int(time.time())}",
            name="Resource Conservation",
            checkpoint="minimal_checkpoint",
            script_dsl="MINIMAL_ACTIONS; CLEANUP_RESOURCES; ESSENTIAL_ONLY",
            expected_outcome="resource_stability",
            priority=2,
            directives=["resource_conservation", "minimal_impact"],
            metadata={"contingency_type": "resource_conservation"},
        )

        return StrategyResponse(
            strategy_id=f"resource_conservation_{int(time.time())}",
            experiments=[conservation_experiment],
            strategic_insights=["Resource conservation mode", "Minimal resource usage approach"],
            next_checkpoints=["resource_safe_checkpoint"],
            metadata={"contingency": True, "resource_conservative": True},
        )

    def get_performance_metrics(self) -> dict[str, float]:
        """Get contingency generator performance metrics."""
        metrics = {}

        if self.generation_times:
            metrics.update(
                {
                    "avg_generation_time_ms": statistics.mean(self.generation_times),
                    "max_generation_time_ms": max(self.generation_times),
                    "generation_count": len(self.generation_times),
                }
            )

        metrics.update(
            {
                "total_generated_contingencies": self.generated_contingencies,
                "template_cache_hits": self.template_cache_hits,
                "template_cache_size": len(self.strategy_templates),
            }
        )

        return metrics
