"""
OpusStrategist - Strategic Planning with Claude Opus

Production-ready strategic planning component that processes responses
from Claude Opus with comprehensive error handling, caching, and
circuit breaker patterns for high availability.
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .circuit_breaker import CircuitBreaker, CircuitConfig
from .claude_code_manager import ClaudeCodeManager
from .opus_strategist_exceptions import (
    DirectiveExtractionError,
    MalformedResponseError,
    OpusStrategistError,
    ResponseTimeoutError,
    StrategyValidationError,
)
from .strategy_response import FallbackStrategy, StrategyResponse
from .strategy_response_cache import ResponseCache
from .strategy_response_parser import StrategyResponseParser, ValidationRule

logger = logging.getLogger(__name__)


# Strategic Continuity Management Data Structures - Issue #113


@dataclass(frozen=True)
class StrategicContext:
    """
    Immutable strategic context for continuity management.

    Preserves strategic state across conversation cycles and system restarts.
    Following Clean Code principles with immutable data structures.
    """

    conversation_history: tuple[str, ...] = field(default_factory=tuple)
    strategic_directives: tuple[str, ...] = field(
        default_factory=tuple
    )  # JSON strings for hashability
    previous_strategies: tuple[str, ...] = field(default_factory=tuple)
    plan_version: str = "1.0"
    decision_outcomes: tuple[str, ...] = field(
        default_factory=tuple
    )  # JSON strings for hashability
    compression_metadata: dict[str, Any] | None = None
    preservation_timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Convert JSON strings back to dictionaries for strategic_directives and decision_outcomes
        strategic_directives = []
        for directive in self.strategic_directives:
            try:
                # Try to parse as JSON if it's a string
                if isinstance(directive, str):
                    strategic_directives.append(json.loads(directive))
                else:
                    strategic_directives.append(directive)
            except (json.JSONDecodeError, TypeError):
                strategic_directives.append(directive)

        decision_outcomes = []
        for outcome in self.decision_outcomes:
            try:
                # Try to parse as JSON if it's a string
                if isinstance(outcome, str):
                    decision_outcomes.append(json.loads(outcome))
                else:
                    decision_outcomes.append(outcome)
            except (json.JSONDecodeError, TypeError):
                decision_outcomes.append(outcome)

        return {
            "conversation_history": list(self.conversation_history),
            "strategic_directives": strategic_directives,
            "previous_strategies": list(self.previous_strategies),
            "plan_version": self.plan_version,
            "decision_outcomes": decision_outcomes,
            "compression_metadata": self.compression_metadata,
            "preservation_timestamp": self.preservation_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategicContext":
        """Create from dictionary after deserialization."""
        # Convert dictionaries to JSON strings for storage in tuples
        strategic_directives_json = tuple(
            json.dumps(directive, sort_keys=True) if isinstance(directive, dict) else str(directive)
            for directive in data.get("strategic_directives", [])
        )
        decision_outcomes_json = tuple(
            json.dumps(outcome, sort_keys=True) if isinstance(outcome, dict) else str(outcome)
            for outcome in data.get("decision_outcomes", [])
        )

        return cls(
            conversation_history=tuple(data.get("conversation_history", [])),
            strategic_directives=strategic_directives_json,
            previous_strategies=tuple(data.get("previous_strategies", [])),
            plan_version=data.get("plan_version", "1.0"),
            decision_outcomes=decision_outcomes_json,
            compression_metadata=data.get("compression_metadata"),
            preservation_timestamp=data.get("preservation_timestamp", time.time()),
        )


@dataclass(frozen=True)
class StrategicPlanVersion:
    """
    Immutable strategic plan version for evolution tracking.

    Tracks changes and evolution of strategic plans over time.
    """

    version_id: str
    plan_content: dict[str, Any]
    creation_timestamp: float
    previous_version: str | None = None
    evolution_reason: str = ""
    changes_summary: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version_id": self.version_id,
            "plan_content": self.plan_content,
            "creation_timestamp": self.creation_timestamp,
            "previous_version": self.previous_version,
            "evolution_reason": self.evolution_reason,
            "changes_summary": list(self.changes_summary),
        }


@dataclass(frozen=True)
class DirectiveConflict:
    """
    Immutable directive conflict representation for resolution.

    Represents conflicts between strategic directives with resolution metadata.
    """

    conflict_id: str
    conflicting_directives: tuple[dict[str, Any], ...]
    resolution_strategy: str
    resolved_directive: dict[str, Any]
    resolution_timestamp: float
    confidence_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conflict_id": self.conflict_id,
            "conflicting_directives": list(self.conflicting_directives),
            "resolution_strategy": self.resolution_strategy,
            "resolved_directive": self.resolved_directive,
            "resolution_timestamp": self.resolution_timestamp,
            "confidence_score": self.confidence_score,
        }


@dataclass(frozen=True)
class StrategicDecision:
    """
    Immutable strategic decision for history tracking.

    Records strategic decisions with context for outcome correlation.
    """

    decision_id: str
    decision_content: dict[str, Any]
    context: dict[str, Any]
    rationale: str
    timestamp: float
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision_id": self.decision_id,
            "decision_content": self.decision_content,
            "context": self.context,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class StrategicOutcome:
    """
    Immutable strategic outcome for decision correlation.

    Records outcomes of strategic decisions for learning and improvement.
    """

    outcome_id: str
    decision_id: str
    result: dict[str, Any]
    execution_metrics: dict[str, Any]
    lessons_learned: tuple[str, ...]
    timestamp: float
    success_score: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "outcome_id": self.outcome_id,
            "decision_id": self.decision_id,
            "result": self.result,
            "execution_metrics": self.execution_metrics,
            "lessons_learned": list(self.lessons_learned),
            "timestamp": self.timestamp,
            "success_score": self.success_score,
        }


class StrategyPriority(Enum):
    """Strategic request priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class StrategyRequest:
    """Structured request for strategic planning."""

    game_state: dict[str, Any]
    context: dict[str, Any]
    priority: StrategyPriority = StrategyPriority.NORMAL
    timeout_override: float | None = None
    cache_ttl_override: float | None = None


class OpusStrategist:
    """
    Strategic planning component using Claude Opus.

    Processes game state into strategic responses with experiments,
    insights, and tactical directives. Implements production patterns:
    - Response caching with TTL
    - Circuit breaker for failure protection
    - Graceful degradation with fallback strategies
    - Comprehensive error recovery
    """

    def __init__(
        self,
        claude_manager: ClaudeCodeManager,
        parser_config: ValidationRule | None = None,
        cache_size: int = 100,
        cache_ttl: float = 300.0,
        circuit_breaker_threshold: float = 0.5,
        circuit_breaker_timeout: float = 60.0,
    ):
        """
        Initialize OpusStrategist with production configuration.

        Args:
            claude_manager: ClaudeCodeManager for Opus communication
            parser_config: Custom parser validation rules
            cache_size: Maximum cached responses
            cache_ttl: Cache TTL in seconds
            circuit_breaker_threshold: Failure rate threshold (0.0-1.0)
            circuit_breaker_timeout: Circuit breaker timeout in seconds
        """
        self.claude_manager = claude_manager

        # Initialize components
        self.parser = StrategyResponseParser(
            validation_timeout=5.0,
            max_response_size=10 * 1024 * 1024,  # 10MB
            validation_rules=parser_config,
        )

        self.response_cache = ResponseCache(
            max_size=cache_size, default_ttl=cache_ttl, cleanup_interval=60.0
        )

        self.circuit_breaker = CircuitBreaker(
            config=CircuitConfig(
                failure_threshold=int(10 * circuit_breaker_threshold),  # Convert to count
                recovery_timeout=circuit_breaker_timeout,
                expected_exception_types=(OpusStrategistError,),
            )
        )

        # Performance and reliability metrics
        self.metrics = {
            "total_requests": 0,
            "successful_responses": 0,
            "cache_hits": 0,
            "fallback_responses": 0,
            "circuit_breaker_trips": 0,
            "avg_response_time_ms": 0.0,
            "max_response_time_ms": 0.0,
        }
        self._metrics_lock = threading.Lock()

        # Strategic Continuity Management - Issue #113
        self._current_strategic_context: StrategicContext | None = None
        self._plan_version_history: dict[str, StrategicPlanVersion] = {}
        self._decision_history: dict[str, StrategicDecision] = {}
        self._outcome_history: dict[str, StrategicOutcome] = {}
        self._conflict_resolutions: dict[str, DirectiveConflict] = {}
        self._continuity_lock = threading.Lock()

        # Strategic continuity metrics
        self.continuity_metrics = {
            "context_preservations": 0,
            "plan_evolutions": 0,
            "conflicts_resolved": 0,
            "decisions_tracked": 0,
            "outcomes_correlated": 0,
            "avg_context_size_bytes": 0,
            "avg_preservation_time_ms": 0.0,
        }

        logger.info(
            "OpusStrategist initialized with production configuration and strategic continuity"
        )

    def get_strategy(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any] | None = None,
        priority: StrategyPriority = StrategyPriority.NORMAL,
        use_cache: bool = True,
    ) -> StrategyResponse:
        """
        Get strategic response for current game state.

        Args:
            game_state: Current game state information
            context: Additional context for strategic planning
            priority: Request priority level
            use_cache: Whether to use cached responses

        Returns:
            StrategyResponse with experiments and directives

        Raises:
            OpusStrategistError: On systematic failures
        """
        start_time = time.time()

        try:
            self._record_metric("total_requests", 1)

            # Generate cache key
            cache_key = self._generate_cache_key(game_state, context)

            # Try cache first (if enabled and not critical priority)
            if use_cache and priority != StrategyPriority.CRITICAL:
                cached_response = self.response_cache.get(cache_key)
                if cached_response:
                    self._record_metric("cache_hits", 1)
                    logger.info(f"Cache hit for strategy request: {cache_key[:16]}...")
                    return cached_response

            # Check circuit breaker
            if not self.circuit_breaker.is_available():
                self._record_metric("circuit_breaker_trips", 1)
                logger.warning("Circuit breaker open, using fallback strategy")
                return self._create_fallback_strategy(game_state, "circuit_breaker_open")

            # Request strategy from Opus
            try:
                raw_response = self._request_opus_strategy(game_state, context, priority)
                parsed_response = self.parser.parse_response(raw_response)

                # Cache successful response
                if use_cache:
                    cache_ttl = self._get_cache_ttl_for_priority(priority)
                    self.response_cache.put(cache_key, parsed_response, cache_ttl)

                # Record success
                self.circuit_breaker.metrics.record_success()
                self._record_metric("successful_responses", 1)

                processing_time = (time.time() - start_time) * 1000
                self._update_response_time_metrics(processing_time)

                logger.info(f"Successfully processed strategy response in {processing_time:.2f}ms")
                return parsed_response

            except (StrategyValidationError, MalformedResponseError, ResponseTimeoutError) as e:
                # Record failure for circuit breaker
                self.circuit_breaker.metrics.record_failure()

                # Use fallback strategy for parsing/validation errors
                logger.warning(f"Strategy processing failed: {str(e)}")
                fallback_reason = type(e).__name__.lower()
                return self._create_fallback_strategy(game_state, fallback_reason)

        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error in get_strategy: {str(e)}")
            return self._create_fallback_strategy(game_state, "unexpected_error")

    def extract_directives(self, strategy_response: StrategyResponse) -> list[str]:
        """
        Extract actionable strategic directives from response.

        Args:
            strategy_response: Parsed strategy response

        Returns:
            List of strategic directives

        Raises:
            DirectiveExtractionError: If directive extraction fails
        """
        try:
            directives = []

            # Extract directives from experiments
            for experiment in strategy_response.experiments:
                directives.extend(experiment.directives)

            # Extract directives from insights
            for insight in strategy_response.strategic_insights:
                if insight.upper().startswith("DIRECTIVE:"):
                    directive = insight[10:].strip()  # Remove 'DIRECTIVE:' prefix
                    directives.append(directive)
                elif "DIRECTIVE" in insight.upper():
                    # Handle embedded directives
                    extracted_directive = self._extract_embedded_directive(insight)
                    if extracted_directive:
                        directives.append(extracted_directive)

            # Remove duplicates while preserving order
            unique_directives = []
            seen = set()
            for directive in directives:
                directive_lower = directive.lower()
                if directive_lower not in seen:
                    unique_directives.append(directive)
                    seen.add(directive_lower)

            logger.debug(f"Extracted {len(unique_directives)} unique directives")
            return unique_directives

        except Exception as e:
            raise DirectiveExtractionError(
                f"Failed to extract directives: {str(e)}", strategy_response.to_dict()
            ) from e

    def get_metrics(self) -> dict[str, Any]:
        """Get strategist performance metrics."""
        with self._metrics_lock:
            base_metrics = self.metrics.copy()

        # Create comprehensive metrics dictionary with mixed types
        metrics: dict[str, Any] = dict(base_metrics)

        # Add component metrics
        metrics["parser_metrics"] = self.parser.get_metrics()
        metrics["cache_metrics"] = self.response_cache.get_stats()
        metrics["circuit_breaker_state"] = self.circuit_breaker.get_state().value

        return metrics

    # Strategic Continuity Management Methods - Issue #113

    def preserve_strategic_context(
        self, strategy_response: StrategyResponse, current_context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Preserve strategic context for continuity across conversation cycles.

        Args:
            strategy_response: Current strategy response from Opus
            current_context: Current strategic context to preserve

        Returns:
            Preserved context dictionary or None on failure

        Performance target: <100ms
        """
        start_time = time.time()

        try:
            with self._continuity_lock:
                # Extract strategic information from response
                strategic_directives = []
                for experiment in strategy_response.experiments:
                    strategic_directives.extend(
                        [
                            {
                                "directive": directive,
                                "source": "experiment",
                                "priority": experiment.priority,
                            }
                            for directive in experiment.directives
                        ]
                    )

                # Extract conversation elements
                conversation_history = current_context.get("conversation_history", [])
                if isinstance(conversation_history, list):
                    # Keep last 50 messages for manageable context size
                    conversation_history = conversation_history[-50:]

                # Convert dictionaries to JSON strings for hashable tuples
                strategic_directives_json = tuple(
                    (
                        json.dumps(directive, sort_keys=True)
                        if isinstance(directive, dict)
                        else str(directive)
                    )
                    for directive in strategic_directives
                )
                decision_outcomes_json = tuple(
                    (
                        json.dumps(outcome, sort_keys=True)
                        if isinstance(outcome, dict)
                        else str(outcome)
                    )
                    for outcome in current_context.get("decision_outcomes", [])
                )

                # Create strategic context
                strategic_context = StrategicContext(
                    conversation_history=tuple(conversation_history),
                    strategic_directives=strategic_directives_json,
                    previous_strategies=tuple(current_context.get("previous_strategies", [])),
                    plan_version=current_context.get("plan_version", "1.0"),
                    decision_outcomes=decision_outcomes_json,
                    compression_metadata=None,
                    preservation_timestamp=time.time(),
                )

                # Update internal state
                self._current_strategic_context = strategic_context

                # Record metrics
                preservation_time = (time.time() - start_time) * 1000
                self._update_continuity_metric_unsafe("context_preservations", 1)
                self._update_continuity_metric_unsafe("avg_preservation_time_ms", preservation_time)

                context_size = len(json.dumps(strategic_context.to_dict()).encode("utf-8"))
                self._update_continuity_metric_unsafe("avg_context_size_bytes", context_size)

                logger.debug(
                    f"Strategic context preserved in {preservation_time:.2f}ms, size: {context_size} bytes"
                )

                return strategic_context.to_dict()

        except Exception as e:
            logger.error(f"Failed to preserve strategic context: {str(e)}")
            return None

    def restore_strategic_context(self, preserved_context: dict[str, Any]) -> bool:
        """
        Restore strategic context after conversation restart.

        Args:
            preserved_context: Previously preserved strategic context

        Returns:
            True if restoration successful, False otherwise
        """
        try:
            with self._continuity_lock:
                if preserved_context:
                    strategic_context = StrategicContext.from_dict(preserved_context)
                    self._current_strategic_context = strategic_context

                    logger.info(
                        f"Strategic context restored from {strategic_context.preservation_timestamp}"
                    )
                    return True
                else:
                    logger.warning("No preserved context provided for restoration")
                    return False

        except Exception as e:
            logger.error(f"Failed to restore strategic context: {str(e)}")
            return False

    def preserve_strategic_context_with_compression(
        self, large_context: dict[str, Any], compression_ratio: float = 0.8
    ) -> dict[str, Any] | None:
        """
        Preserve strategic context with compression for large contexts.

        Args:
            large_context: Large strategic context to compress and preserve
            compression_ratio: Target compression ratio (0.0-1.0)

        Returns:
            Compressed and preserved context or None on failure
        """
        start_time = time.time()

        try:
            with self._continuity_lock:
                # Apply strategic compression
                compressed_context = self._compress_strategic_context(
                    large_context, compression_ratio
                )

                # Convert dictionaries to JSON strings for hashable tuples
                strategic_directives_json = tuple(
                    (
                        json.dumps(directive, sort_keys=True)
                        if isinstance(directive, dict)
                        else str(directive)
                    )
                    for directive in compressed_context.get("strategic_directives", [])
                )
                decision_outcomes_json = tuple(
                    (
                        json.dumps(outcome, sort_keys=True)
                        if isinstance(outcome, dict)
                        else str(outcome)
                    )
                    for outcome in compressed_context.get("decision_outcomes", [])
                )

                # Create compressed strategic context
                strategic_context = StrategicContext(
                    conversation_history=tuple(compressed_context.get("conversation_history", [])),
                    strategic_directives=strategic_directives_json,
                    previous_strategies=tuple(compressed_context.get("previous_strategies", [])),
                    plan_version=compressed_context.get("plan_version", "1.0"),
                    decision_outcomes=decision_outcomes_json,
                    compression_metadata={
                        "compression_ratio": compression_ratio,
                        "original_size": len(json.dumps(large_context).encode("utf-8")),
                        "compressed_size": len(json.dumps(compressed_context).encode("utf-8")),
                        "compression_timestamp": time.time(),
                    },
                    preservation_timestamp=time.time(),
                )

                # Update internal state
                self._current_strategic_context = strategic_context

                # Record metrics
                preservation_time = (time.time() - start_time) * 1000
                self._update_continuity_metric_unsafe("context_preservations", 1)
                self._update_continuity_metric_unsafe("avg_preservation_time_ms", preservation_time)

                logger.debug(
                    f"Strategic context compressed and preserved in {preservation_time:.2f}ms"
                )

                return strategic_context.to_dict()

        except Exception as e:
            logger.error(f"Failed to compress and preserve strategic context: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def version_strategic_plan(self, plan_content: dict[str, Any]) -> dict[str, Any] | None:
        """
        Create version information for strategic plan.

        Args:
            plan_content: Strategic plan content to version

        Returns:
            Version information dictionary or None on failure

        Performance target: <50ms
        """
        start_time = time.time()

        try:
            with self._continuity_lock:
                # Generate version ID
                plan_hash = hashlib.md5(
                    json.dumps(plan_content, sort_keys=True).encode()
                ).hexdigest()
                version_id = f"v{len(self._plan_version_history) + 1}_{plan_hash[:8]}"

                # Create plan version
                plan_version = StrategicPlanVersion(
                    version_id=version_id,
                    plan_content=plan_content,
                    creation_timestamp=time.time(),
                    previous_version=None,  # Will be set by track_strategic_plan_evolution
                    evolution_reason="",
                    changes_summary=(),
                )

                # Store version
                self._plan_version_history[version_id] = plan_version

                # Record metrics
                versioning_time = (time.time() - start_time) * 1000
                self._update_continuity_metric_unsafe("plan_evolutions", 1)

                logger.debug(f"Strategic plan versioned in {versioning_time:.2f}ms as {version_id}")

                return plan_version.to_dict()

        except Exception as e:
            logger.error(f"Failed to version strategic plan: {str(e)}")
            return None

    def _request_opus_strategy(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any] | None,
        priority: StrategyPriority,
    ) -> str:
        """Send strategic planning request to Claude Opus."""

        # Build strategic planning prompt
        prompt = self._build_strategic_prompt(game_state, context, priority)

        # Send to Opus via ClaudeCodeManager
        try:
            strategic_process = self.claude_manager.get_strategic_process()
            if strategic_process is None:
                raise ConnectionError("No strategic process available")

            response = strategic_process.send_message(prompt)

            if not response or not response.strip():
                raise MalformedResponseError("Empty response from Opus")

            return response

        except Exception as e:
            logger.error(f"Failed to get response from Opus: {str(e)}")
            raise OpusStrategistError(f"Opus communication failed: {str(e)}") from e

    def _build_strategic_prompt(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any] | None,
        priority: StrategyPriority,
    ) -> str:
        """Build strategic planning prompt for Opus."""

        prompt_parts = [
            "STRATEGIC PLANNING REQUEST",
            "",
            "Current Game State:",
            f"- Location: {game_state.get('location', 'unknown')}",
            f"- Health: {game_state.get('health', 0)}",
            f"- Level: {game_state.get('level', 1)}",
            f"- Pokemon Count: {game_state.get('pokemon_count', 0)}",
            f"- Badges: {game_state.get('badges', 0)}",
            "",
        ]

        if context:
            prompt_parts.extend(
                [
                    "Strategic Context:",
                    *[f"- {key}: {value}" for key, value in context.items()],
                    "",
                ]
            )

        prompt_parts.extend(
            [
                f"Priority: {priority.name}",
                "",
                "Please provide strategic response in JSON format with:",
                "- strategy_id: unique identifier",
                "- experiments: array of parallel experiments with id, name, checkpoint, script_dsl, expected_outcome, priority",
                "- strategic_insights: array of strategic insights and directives",
                "- next_checkpoints: array of recommended checkpoint locations",
                "",
                "Focus on concrete, executable experiments that advance speedrun progress.",
            ]
        )

        return "\n".join(prompt_parts)

    def _generate_cache_key(
        self, game_state: dict[str, Any], context: dict[str, Any] | None
    ) -> str:
        """Generate cache key for request."""
        # Combine game state and context for cache key
        combined_state = dict(game_state)
        if context:
            combined_state.update({"context": context})

        return self.response_cache.generate_cache_key(combined_state)

    def _get_cache_ttl_for_priority(self, priority: StrategyPriority) -> float:
        """Get appropriate cache TTL based on request priority."""
        ttl_map = {
            StrategyPriority.LOW: 600.0,  # 10 minutes
            StrategyPriority.NORMAL: 300.0,  # 5 minutes
            StrategyPriority.HIGH: 120.0,  # 2 minutes
            StrategyPriority.CRITICAL: 60.0,  # 1 minute
        }
        return ttl_map.get(priority, 300.0)

    def _create_fallback_strategy(
        self, game_state: dict[str, Any], reason: str
    ) -> StrategyResponse:
        """Create fallback strategy when Opus communication fails."""
        self._record_metric("fallback_responses", 1)

        logger.info(f"Creating fallback strategy due to: {reason}")

        fallback = FallbackStrategy.create_default_fallback(game_state)

        # Add reason to metadata
        fallback_dict = fallback.to_dict()
        fallback_dict["metadata"]["fallback_reason"] = reason

        return StrategyResponse.from_dict(fallback_dict)

    def _extract_embedded_directive(self, insight: str) -> str | None:
        """Extract directive from insight text containing 'DIRECTIVE'."""
        # Simple pattern matching for embedded directives
        insight_upper = insight.upper()

        if "DIRECTIVE:" in insight_upper:
            # Find the directive part
            directive_start = insight_upper.index("DIRECTIVE:") + 10
            directive_text = insight[directive_start:].strip()

            # Find end of directive (stop at punctuation or new sentence)
            for end_char in [".", "!", "?", ";", "\n"]:
                if end_char in directive_text:
                    directive_text = directive_text[: directive_text.index(end_char)].strip()
                    break

            return directive_text if directive_text else None

        return None

    def _record_metric(self, metric_name: str, value: int) -> None:
        """Thread-safe metric recording."""
        with self._metrics_lock:
            self.metrics[metric_name] += value

    def _update_response_time_metrics(self, response_time_ms: float) -> None:
        """Update response time metrics."""
        with self._metrics_lock:
            current_avg = self.metrics["avg_response_time_ms"]
            current_count = self.metrics["successful_responses"]

            if current_count == 1:
                self.metrics["avg_response_time_ms"] = response_time_ms
            else:
                # Rolling average
                self.metrics["avg_response_time_ms"] = (
                    current_avg * (current_count - 1) + response_time_ms
                ) / current_count

            if response_time_ms > self.metrics["max_response_time_ms"]:
                self.metrics["max_response_time_ms"] = response_time_ms

    # Strategic Continuity Helper Methods

    def _compress_strategic_context(
        self, context: dict[str, Any], compression_ratio: float
    ) -> dict[str, Any]:
        """
        Apply strategic compression to large context.

        Keeps most important strategic information while reducing size.
        """
        compressed = {}

        # Compress conversation history - keep strategic messages
        if "conversation_history" in context:
            history = context["conversation_history"]
            if isinstance(history, list) and len(history) > 20:
                # Keep strategic keywords and recent messages
                strategic_messages = [
                    msg
                    for msg in history
                    if any(
                        keyword in str(msg).lower()
                        for keyword in ["strategy", "directive", "plan", "decision", "critical"]
                    )
                ]
                recent_messages = history[-10:]  # Keep last 10 messages

                # Combine and deduplicate
                seen_messages = set()
                important_messages = []
                for message in strategic_messages + recent_messages:
                    if message not in seen_messages:
                        important_messages.append(message)
                        seen_messages.add(message)

                target_size = max(5, int(len(history) * (1 - compression_ratio)))
                compressed["conversation_history"] = important_messages[:target_size]
            else:
                compressed["conversation_history"] = history

        # Compress strategic plans - keep only successful ones
        if "strategic_plans" in context:
            plans = context["strategic_plans"]
            if isinstance(plans, list):
                # Keep successful plans and recent plans
                successful_plans = [p for p in plans if p.get("status") == "completed"]
                recent_plans = plans[-5:]  # Keep last 5 plans

                # Deduplicate plans by converting to JSON strings for comparison
                seen_plan_ids = set()
                important_plans = []
                for plan in successful_plans + recent_plans:
                    plan_id = plan.get("id", json.dumps(plan, sort_keys=True))
                    if plan_id not in seen_plan_ids:
                        important_plans.append(plan)
                        seen_plan_ids.add(plan_id)

                target_size = max(3, int(len(plans) * (1 - compression_ratio)))
                compressed["strategic_plans"] = important_plans[:target_size]
            else:
                compressed["strategic_plans"] = plans

        # Compress decision outcomes - keep successful outcomes
        if "decision_outcomes" in context:
            outcomes = context["decision_outcomes"]
            if isinstance(outcomes, list):
                successful_outcomes = [
                    o
                    for o in outcomes
                    if o.get("outcome") == "success" or o.get("success_score", 0) > 0.7
                ]
                target_size = max(5, int(len(outcomes) * (1 - compression_ratio)))
                compressed["decision_outcomes"] = successful_outcomes[:target_size]
            else:
                compressed["decision_outcomes"] = outcomes

        # Preserve other important strategic information without compression
        for key in ["strategic_directives", "previous_strategies", "plan_version"]:
            if key in context:
                compressed[key] = context[key]

        return compressed

    def _update_continuity_metric(self, metric_name: str, value: float) -> None:
        """Update strategic continuity metric with thread safety."""
        with self._continuity_lock:
            self._update_continuity_metric_unsafe(metric_name, value)

    def _update_continuity_metric_unsafe(self, metric_name: str, value: float) -> None:
        """Update strategic continuity metric without acquiring lock (for internal use)."""
        if metric_name.startswith("avg_"):
            # Calculate rolling average
            current_value = self.continuity_metrics[metric_name]
            count_metric = (
                metric_name.replace("avg_", "").replace("_ms", "").replace("_bytes", "") + "s"
            )

            if count_metric in self.continuity_metrics:
                count = self.continuity_metrics[count_metric]
                if count > 0:
                    self.continuity_metrics[metric_name] = (current_value * count + value) / (
                        count + 1
                    )
                else:
                    self.continuity_metrics[metric_name] = value
            else:
                # Fallback for metrics without count
                self.continuity_metrics[metric_name] = value
        else:
            # Simple increment for count metrics
            self.continuity_metrics[metric_name] += value

    # Remaining Strategic Continuity Methods

    def track_strategic_plan_evolution(
        self, previous_plan: StrategyResponse, current_plan: StrategyResponse, evolution_reason: str
    ) -> dict[str, Any] | None:
        """
        Track evolution of strategic plans over time.

        Args:
            previous_plan: Previous strategic plan
            current_plan: Current strategic plan
            evolution_reason: Reason for plan evolution

        Returns:
            Evolution history dictionary or None on failure
        """
        try:
            with self._continuity_lock:
                # Generate evolution ID
                evolution_id = f"evo_{int(time.time())}_{len(self._plan_version_history)}"

                # Analyze changes between plans
                changes = self._analyze_plan_changes(previous_plan, current_plan)

                # Create updated plan version
                current_version = StrategicPlanVersion(
                    version_id=evolution_id,
                    plan_content=current_plan.to_dict(),
                    creation_timestamp=time.time(),
                    previous_version=(
                        previous_plan.strategy_id if hasattr(previous_plan, "strategy_id") else None
                    ),
                    evolution_reason=evolution_reason,
                    changes_summary=tuple(changes),
                )

                # Store evolution
                self._plan_version_history[evolution_id] = current_version

                # Update metrics
                self._update_continuity_metric_unsafe("plan_evolutions", 1)

                logger.info(
                    f"Strategic plan evolution tracked: {evolution_id} - {evolution_reason}"
                )

                return {
                    "evolution_id": evolution_id,
                    "changes": changes,
                    "evolution_reason": evolution_reason,
                    "timestamp": current_version.creation_timestamp,
                }

        except Exception as e:
            logger.error(f"Failed to track strategic plan evolution: {str(e)}")
            return None

    def resolve_strategic_directive_conflicts(
        self, conflicting_directives: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Resolve conflicts between strategic directives.

        Args:
            conflicting_directives: List of conflicting directives

        Returns:
            List of resolved directives

        Performance target: <200ms
        """
        start_time = time.time()

        try:
            with self._continuity_lock:
                if not conflicting_directives:
                    return []

                # Group directives by similarity
                directive_groups = self._group_similar_directives(conflicting_directives)
                resolved_directives = []

                for group in directive_groups:
                    if len(group) == 1:
                        # No conflict - add directly
                        resolved_directives.append(group[0])
                    else:
                        # Resolve conflict using priority-based resolution
                        resolved = self._resolve_directive_group_conflict(group)
                        if resolved:
                            resolved_directives.append(resolved)

                            # Record conflict resolution
                            conflict_id = (
                                f"conflict_{int(time.time())}_{len(self._conflict_resolutions)}"
                            )
                            conflict_record = DirectiveConflict(
                                conflict_id=conflict_id,
                                conflicting_directives=tuple(group),
                                resolution_strategy="priority_based",
                                resolved_directive=resolved,
                                resolution_timestamp=time.time(),
                                confidence_score=resolved.get("priority", 5) / 10.0,
                            )

                            self._conflict_resolutions[conflict_id] = conflict_record

                # Update metrics
                resolution_time = (time.time() - start_time) * 1000
                self._update_continuity_metric_unsafe("conflicts_resolved", len(directive_groups))

                logger.debug(
                    f"Resolved {len(directive_groups)} directive conflicts in {resolution_time:.2f}ms"
                )

                return resolved_directives

        except Exception as e:
            logger.error(f"Failed to resolve strategic directive conflicts: {str(e)}")
            return conflicting_directives  # Return original on failure

    def record_strategic_decision(self, decision: dict[str, Any]) -> bool:
        """
        Record a strategic decision for outcome tracking.

        Args:
            decision: Strategic decision to record

        Returns:
            True if recording successful, False otherwise
        """
        try:
            with self._continuity_lock:
                decision_id = decision.get("decision_id")
                if not decision_id:
                    logger.warning("Decision missing decision_id, generating one")
                    decision_id = f"decision_{int(time.time())}_{len(self._decision_history)}"

                strategic_decision = StrategicDecision(
                    decision_id=decision_id,
                    decision_content=decision.copy(),
                    context=decision.get("context", {}),
                    rationale=decision.get("rationale", ""),
                    timestamp=decision.get("timestamp", time.time()),
                    confidence=decision.get("confidence", 1.0),
                )

                self._decision_history[decision_id] = strategic_decision

                # Update metrics
                self._update_continuity_metric_unsafe("decisions_tracked", 1)

                logger.info(f"Strategic decision recorded: {decision_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to record strategic decision: {str(e)}")
            return False

    def record_strategic_outcome(self, outcome: dict[str, Any]) -> bool:
        """
        Record outcome of a strategic decision.

        Args:
            outcome: Strategic outcome to record

        Returns:
            True if recording successful, False otherwise
        """
        try:
            with self._continuity_lock:
                outcome_id = outcome.get("outcome_id")
                if not outcome_id:
                    outcome_id = f"outcome_{int(time.time())}_{len(self._outcome_history)}"

                decision_id = outcome.get("decision_id")
                if not decision_id:
                    logger.warning("Outcome missing decision_id - cannot correlate")
                    return False

                strategic_outcome = StrategicOutcome(
                    outcome_id=outcome_id,
                    decision_id=decision_id,
                    result=outcome.get("result", {}),
                    execution_metrics=outcome.get("execution_metrics", {}),
                    lessons_learned=tuple(outcome.get("lessons_learned", [])),
                    timestamp=outcome.get("timestamp", time.time()),
                    success_score=outcome.get("success_score", 0.5),
                )

                self._outcome_history[outcome_id] = strategic_outcome

                logger.info(f"Strategic outcome recorded: {outcome_id} for decision {decision_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to record strategic outcome: {str(e)}")
            return False

    def correlate_decision_with_outcome(self, decision_id: str) -> dict[str, Any] | None:
        """
        Correlate strategic decision with its outcome.

        Args:
            decision_id: ID of decision to correlate

        Returns:
            Correlation data or None if not found
        """
        try:
            with self._continuity_lock:
                decision = self._decision_history.get(decision_id)
                if not decision:
                    logger.warning(f"Decision {decision_id} not found for correlation")
                    return None

                # Find matching outcomes
                outcomes = [
                    outcome
                    for outcome in self._outcome_history.values()
                    if outcome.decision_id == decision_id
                ]

                if not outcomes:
                    logger.debug(f"No outcomes found for decision {decision_id}")
                    return None

                # Create correlation data
                correlation = {
                    "decision_id": decision_id,
                    "decision": decision.to_dict(),
                    "outcomes": [outcome.to_dict() for outcome in outcomes],
                    "correlation_score": sum(o.success_score for o in outcomes) / len(outcomes),
                    "lessons_learned": list(
                        {lesson for outcome in outcomes for lesson in outcome.lessons_learned}
                    ),
                    "correlation_timestamp": time.time(),
                }

                # Update metrics
                self._update_continuity_metric_unsafe("outcomes_correlated", 1)

                logger.debug(f"Correlated decision {decision_id} with {len(outcomes)} outcomes")

                return correlation

        except Exception as e:
            logger.error(f"Failed to correlate decision with outcome: {str(e)}")
            return None

    def _analyze_plan_changes(
        self, previous_plan: StrategyResponse, current_plan: StrategyResponse
    ) -> list[str]:
        """Analyze changes between strategic plans."""
        changes = []

        try:
            # Compare experiments
            prev_experiments = {exp.name for exp in previous_plan.experiments}
            curr_experiments = {exp.name for exp in current_plan.experiments}

            added_experiments = curr_experiments - prev_experiments
            removed_experiments = prev_experiments - curr_experiments

            if added_experiments:
                changes.append(f"Added experiments: {', '.join(added_experiments)}")
            if removed_experiments:
                changes.append(f"Removed experiments: {', '.join(removed_experiments)}")

            # Compare strategic insights
            if len(current_plan.strategic_insights) != len(previous_plan.strategic_insights):
                changes.append(
                    f"Strategic insights changed from {len(previous_plan.strategic_insights)} to {len(current_plan.strategic_insights)}"
                )

        except Exception as e:
            logger.warning(f"Error analyzing plan changes: {str(e)}")
            changes.append("Plan analysis failed - structural changes detected")

        return changes if changes else ["Minor strategic adjustments"]

    def _group_similar_directives(
        self, directives: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Group similar strategic directives for conflict resolution."""
        groups: list[list[dict[str, Any]]] = []

        for directive in directives:
            directive_text = directive.get("directive", "").lower()

            # Find existing group with similar directive
            found_group = None
            for group in groups:
                for existing_directive in group:
                    existing_text = existing_directive.get("directive", "").lower()

                    # Simple similarity check - could be enhanced
                    if (
                        directive_text in existing_text
                        or existing_text in directive_text
                        or any(
                            word in existing_text
                            for word in directive_text.split()
                            if len(word) > 3
                        )
                    ):
                        found_group = group
                        break

                if found_group:
                    break

            if found_group:
                found_group.append(directive)
            else:
                groups.append([directive])

        return groups

    def _resolve_directive_group_conflict(
        self, directive_group: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Resolve conflict within a group of similar directives."""
        if not directive_group:
            return None

        # Priority-based resolution - highest priority wins
        return max(directive_group, key=lambda d: d.get("priority", 0))


# Re-export classes for easier imports
__all__ = [
    "OpusStrategist",
    "StrategyPriority",
    "StrategyRequest",
    "StrategyResponse",
    "StrategyResponseParser",
    "ResponseCache",
    "FallbackStrategy",
]
