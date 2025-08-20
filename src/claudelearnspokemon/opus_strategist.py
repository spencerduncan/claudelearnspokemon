"""
OpusStrategist - Strategic Planning with Claude Opus

Production-ready strategic planning component that processes responses
from Claude Opus with comprehensive error handling, caching, and
circuit breaker patterns for high availability.
"""

import logging
import threading
import time
from dataclasses import dataclass
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

        logger.info("OpusStrategist initialized with production configuration")

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

    def request_strategy(
        self,
        game_state: dict[str, Any],
        recent_results: list[dict[str, Any]],
        priority: StrategyPriority = StrategyPriority.NORMAL,
    ) -> dict[str, Any]:
        """
        Request strategic planning from Claude Opus with game state and recent results.

        This method specifically focuses on strategic planning interface, formatting
        requests for optimal Opus consumption and parsing JSON strategy responses
        into actionable plans.

        Args:
            game_state: Current game state information
            recent_results: List of recent execution results for context
            priority: Strategic request priority level

        Returns:
            Dict containing strategic plan with experiments, insights, and checkpoints

        Raises:
            OpusStrategistError: On systematic strategic planning failures
        """
        start_time = time.time()

        try:
            self._record_metric("total_requests", 1)

            # Build strategic planning request with recent results context
            strategic_prompt = self._build_strategic_planning_request(
                game_state, recent_results, priority
            )

            # Check circuit breaker for strategic planning
            if not self.circuit_breaker.is_available():
                self._record_metric("circuit_breaker_trips", 1)
                logger.warning("Circuit breaker open for strategic planning, using fallback")
                return self._create_strategic_fallback_plan(
                    game_state, recent_results, "circuit_breaker_open"
                )

            try:
                # Request strategic plan from Opus
                raw_response = self._request_strategic_plan_from_opus(strategic_prompt)

                # Parse JSON strategy response into actionable plan
                strategic_plan = self._parse_strategic_response(raw_response)

                # Validate strategic plan structure
                validated_plan = self._validate_strategic_plan(strategic_plan)

                # Record success metrics
                self.circuit_breaker.metrics.record_success()
                self._record_metric("successful_responses", 1)

                processing_time = (time.time() - start_time) * 1000
                self._update_response_time_metrics(processing_time)

                logger.info(f"Strategic planning completed in {processing_time:.2f}ms")
                return validated_plan

            except (StrategyValidationError, MalformedResponseError, ResponseTimeoutError) as e:
                # Record failure and provide fallback
                self.circuit_breaker.metrics.record_failure()
                logger.warning(f"Strategic planning failed: {str(e)}")

                fallback_reason = type(e).__name__.lower()
                return self._create_strategic_fallback_plan(
                    game_state, recent_results, fallback_reason
                )

        except Exception as e:
            # Catch-all for unexpected strategic planning errors
            logger.error(f"Unexpected error in strategic planning: {str(e)}")
            return self._create_strategic_fallback_plan(
                game_state, recent_results, "unexpected_error"
            )

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

    def _build_strategic_planning_request(
        self,
        game_state: dict[str, Any],
        recent_results: list[dict[str, Any]],
        priority: StrategyPriority,
    ) -> str:
        """
        Build strategic planning request formatted for Claude Opus consumption.

        This method formats game state and recent results into a structured
        prompt optimized for strategic planning by Claude Opus.
        """
        prompt_parts = [
            "STRATEGIC PLANNING REQUEST",
            "=" * 50,
            "",
            "Current Game State:",
        ]

        # Format game state for strategic context
        for key, value in game_state.items():
            prompt_parts.append(f"- {key}: {value}")

        prompt_parts.extend(["", "Recent Execution Results:"])

        # Format recent results for strategic analysis
        if not recent_results:
            prompt_parts.append("- No recent results available for analysis")
        else:
            for i, result in enumerate(recent_results, 1):
                prompt_parts.extend(
                    [
                        "",
                        f"Result {i} (Worker: {result.get('worker_id', 'unknown')}):",
                        f"  - success: {result.get('success', False)}",
                        f"  - execution_time: {result.get('execution_time', 0.0)}",
                        f"  - actions_taken: {result.get('actions_taken', [])}",
                        f"  - final_state: {result.get('final_state', {})}",
                        f"  - patterns_discovered: {result.get('patterns_discovered', [])}",
                    ]
                )

        prompt_parts.extend(
            [
                "",
                f"Strategic Priority: {priority.name}",
                "",
                "STRATEGIC PLANNING OBJECTIVES:",
                "1. Analyze current game state and recent execution patterns",
                "2. Identify opportunities for strategic advancement",
                "3. Generate diverse parallel experiments for testing",
                "4. Provide strategic insights based on result patterns",
                "5. Recommend optimal checkpoints for future experiments",
                "",
                "Please provide strategic response in JSON format with:",
                "- strategy_id: unique identifier for this strategic plan",
                "- experiments: array of parallel experiments with id, name, checkpoint, script_dsl, expected_outcome, priority",
                "- strategic_insights: array of strategic insights and actionable directives",
                "- next_checkpoints: array of recommended checkpoint locations for strategic advantage",
                "",
                "Focus on actionable experiments that leverage successful patterns",
                "from recent results while avoiding identified failure patterns.",
            ]
        )

        return "\n".join(prompt_parts)

    def _request_strategic_plan_from_opus(self, strategic_prompt: str) -> str:
        """Send strategic planning request to Claude Opus and return raw response."""
        try:
            strategic_process = self.claude_manager.get_strategic_process()
            if strategic_process is None:
                raise ConnectionError("No strategic process available for planning")

            response = strategic_process.send_message(strategic_prompt)

            if not response or not response.strip():
                raise MalformedResponseError("Empty strategic planning response from Opus")

            return response

        except Exception as e:
            logger.error(f"Failed to get strategic plan from Opus: {str(e)}")
            raise OpusStrategistError(f"Strategic planning communication failed: {str(e)}") from e

    def _parse_strategic_response(self, raw_response: str) -> dict[str, Any]:
        """
        Parse JSON strategy response from Opus into structured strategic plan.

        This method handles JSON parsing with comprehensive error handling
        for malformed responses.
        """
        import json

        try:
            # Attempt to parse JSON response
            strategic_plan = json.loads(raw_response)

            # Basic structure validation
            if not isinstance(strategic_plan, dict):
                raise MalformedResponseError("Strategic response is not a valid JSON object")

            return strategic_plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strategic JSON response: {str(e)}")
            raise MalformedResponseError(f"Invalid JSON in strategic response: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error parsing strategic response: {str(e)}")
            raise MalformedResponseError(f"Strategic response parsing failed: {str(e)}") from e

    def _validate_strategic_plan(self, strategic_plan: dict[str, Any]) -> dict[str, Any]:
        """
        Validate strategic plan structure and content.

        Ensures the strategic plan contains required fields and valid data
        structures for downstream processing.
        """
        try:
            # Validate required top-level fields
            required_fields = [
                "strategy_id",
                "experiments",
                "strategic_insights",
                "next_checkpoints",
            ]
            for field in required_fields:
                if field not in strategic_plan:
                    logger.warning(f"Strategic plan missing required field: {field}")
                    # Add default empty values for missing fields
                    if field == "strategy_id":
                        strategic_plan[field] = f"strategic_{int(time.time())}"
                    else:
                        strategic_plan[field] = []

            # Validate experiments structure
            experiments = strategic_plan.get("experiments", [])
            if not isinstance(experiments, list):
                logger.warning("Experiments field is not a list, converting")
                strategic_plan["experiments"] = []

            # Validate each experiment has minimum required structure
            validated_experiments: list[dict[str, Any]] = []
            for exp in experiments:
                if isinstance(exp, dict):
                    # Ensure required experiment fields exist
                    exp_defaults = {
                        "id": f"exp_{len(validated_experiments)}",
                        "name": "Strategic experiment",
                        "checkpoint": "current_position",
                        "script_dsl": "# Strategic action",
                        "expected_outcome": "Strategic advancement",
                        "priority": "medium",
                    }

                    validated_exp = {**exp_defaults, **exp}
                    validated_experiments.append(validated_exp)
                else:
                    logger.warning(f"Invalid experiment structure: {exp}")

            strategic_plan["experiments"] = validated_experiments

            # Validate strategic insights
            insights = strategic_plan.get("strategic_insights", [])
            if not isinstance(insights, list):
                logger.warning("Strategic insights field is not a list, converting")
                strategic_plan["strategic_insights"] = [str(insights)] if insights else []

            # Validate next checkpoints
            checkpoints = strategic_plan.get("next_checkpoints", [])
            if not isinstance(checkpoints, list):
                logger.warning("Next checkpoints field is not a list, converting")
                strategic_plan["next_checkpoints"] = [str(checkpoints)] if checkpoints else []

            logger.debug(f"Strategic plan validated with {len(validated_experiments)} experiments")
            return strategic_plan

        except Exception as e:
            logger.error(f"Strategic plan validation failed: {str(e)}")
            raise StrategyValidationError(
                f"Invalid strategic plan structure: {str(e)}", None
            ) from e

    def _create_strategic_fallback_plan(
        self, game_state: dict[str, Any], recent_results: list[dict[str, Any]], reason: str
    ) -> dict[str, Any]:
        """
        Create fallback strategic plan when Opus strategic planning fails.

        This method creates a sensible fallback plan based on game state
        and recent results when strategic planning communication fails.
        """
        self._record_metric("fallback_responses", 1)

        logger.info(f"Creating strategic fallback plan due to: {reason}")

        # Analyze recent results to inform fallback strategy
        successful_patterns = []
        failed_patterns = []

        for result in recent_results:
            if result.get("success", False):
                successful_patterns.extend(result.get("patterns_discovered", []))
            else:
                failed_patterns.extend(result.get("patterns_discovered", []))

        # Create basic strategic experiments based on game state
        fallback_experiments = []

        # Basic exploration experiment
        fallback_experiments.append(
            {
                "id": "fallback_exploration",
                "name": "Safe exploration pattern",
                "checkpoint": f"fallback_{game_state.get('location', 'unknown')}",
                "script_dsl": "SAFE_MOVE; OBSERVE; SAFE_MOVE",
                "expected_outcome": "Gather information safely",
                "priority": "medium",
            }
        )

        # Pattern-based experiment if we have successful patterns
        if successful_patterns:
            most_common_pattern = max(set(successful_patterns), key=successful_patterns.count)
            fallback_experiments.append(
                {
                    "id": "fallback_pattern_repeat",
                    "name": f"Repeat successful {most_common_pattern}",
                    "checkpoint": game_state.get("location", "current_position"),
                    "script_dsl": f"# Repeat {most_common_pattern} pattern",
                    "expected_outcome": f"Leverage {most_common_pattern} success",
                    "priority": "high",
                }
            )

        # Create strategic insights from available data
        strategic_insights = [
            f"Fallback strategy activated due to: {reason}",
            f"Current location: {game_state.get('location', 'unknown')}",
        ]

        if successful_patterns:
            strategic_insights.append(
                f"Successful patterns available: {', '.join(set(successful_patterns))}"
            )
        if failed_patterns:
            strategic_insights.append(f"Avoid failed patterns: {', '.join(set(failed_patterns))}")

        # Recommend basic checkpoints
        next_checkpoints = [
            f"checkpoint_{game_state.get('location', 'fallback')}_safe",
            "exploration_checkpoint_1",
        ]

        fallback_plan = {
            "strategy_id": f"fallback_{reason}_{int(time.time())}",
            "experiments": fallback_experiments,
            "strategic_insights": strategic_insights,
            "next_checkpoints": next_checkpoints,
            "metadata": {
                "fallback_reason": reason,
                "created_at": time.time(),
                "game_state_snapshot": game_state.copy(),
                "recent_results_count": len(recent_results),
            },
        }

        return fallback_plan


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
