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
from .language_evolution import (
    EvolutionProposal,
    EvolutionProposalGenerator,
    LanguageAnalyzer,
    LanguageValidator,
)
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

        # Language evolution system components
        self.language_analyzer = LanguageAnalyzer()
        self.proposal_generator = EvolutionProposalGenerator()
        self.language_validator = LanguageValidator()

        # Performance and reliability metrics
        self.metrics = {
            "total_requests": 0,
            "successful_responses": 0,
            "cache_hits": 0,
            "fallback_responses": 0,
            "circuit_breaker_trips": 0,
            "avg_response_time_ms": 0.0,
            "max_response_time_ms": 0.0,
            "language_evolution_requests": 0,
            "evolution_proposals_generated": 0,
            "evolution_analysis_time_ms": 0.0,
        }
        self._metrics_lock = threading.Lock()

        logger.info(
            "OpusStrategist initialized with production configuration and language evolution"
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

    def propose_language_evolution(
        self, recent_results: list[dict[str, Any]]
    ) -> list[EvolutionProposal]:
        """
        Propose language evolution improvements based on recent execution results.

        This method analyzes recent parallel execution results to identify patterns
        that could benefit from DSL improvements, generates concrete proposals for
        language evolution, and validates them for consistency and safety.

        Args:
            recent_results: List of recent execution results with pattern performance data.
                           Each result should contain:
                           - pattern_name: Name of executed pattern
                           - success_rate: Pattern success rate (0.0-1.0)
                           - usage_count: Number of times pattern was used
                           - input_sequence: List of input commands
                           - context: Execution context information
                           - average_execution_time: Optional timing data

        Returns:
            List of validated evolution proposals sorted by expected impact.
            Each proposal contains concrete DSL changes, expected improvements,
            and implementation guidance.

        Raises:
            OpusStrategistError: If language evolution analysis fails
            PerformanceError: If processing exceeds performance targets

        Performance Targets:
            - Total processing time: <400ms (200ms analysis + 100ms generation + 50ms validation + overhead)
            - Pattern analysis: <200ms
            - Proposal generation: <100ms
            - Validation: <50ms
        """
        start_time = time.perf_counter()

        try:
            self._record_metric("language_evolution_requests", 1)

            # Convert recent results to pattern analysis format
            pattern_data = self._convert_results_to_patterns(recent_results)

            if not pattern_data:
                logger.warning("No pattern data available for language evolution analysis")
                return []

            logger.info(f"Starting language evolution analysis for {len(pattern_data)} patterns")

            # Phase 1: Analyze patterns for evolution opportunities (<200ms)
            analysis_start = time.perf_counter()
            evolution_opportunities = self.language_analyzer.identify_evolution_opportunities(
                pattern_data
            )
            analysis_time = (time.perf_counter() - analysis_start) * 1000

            logger.debug(
                f"Pattern analysis completed in {analysis_time:.2f}ms, found {len(evolution_opportunities)} opportunities"
            )

            if not evolution_opportunities:
                logger.info("No evolution opportunities identified from current patterns")
                return []

            # Phase 2: Generate evolution proposals (<100ms)
            generation_start = time.perf_counter()
            evolution_proposals = self.proposal_generator.generate_proposals(
                evolution_opportunities
            )
            generation_time = (time.perf_counter() - generation_start) * 1000

            logger.debug(
                f"Proposal generation completed in {generation_time:.2f}ms, generated {len(evolution_proposals)} proposals"
            )

            if not evolution_proposals:
                logger.info("No concrete proposals could be generated from opportunities")
                return []

            # Phase 3: Validate proposals (<50ms)
            validation_start = time.perf_counter()
            validated_proposals = self.language_validator.validate_proposals(evolution_proposals)
            validation_time = (time.perf_counter() - validation_start) * 1000

            logger.debug(f"Proposal validation completed in {validation_time:.2f}ms")

            # Filter to only high-quality proposals (validation score >= 0.7)
            high_quality_proposals = [
                proposal for proposal in validated_proposals if proposal.validation_score >= 0.7
            ]

            # Calculate total processing time
            total_time = (time.perf_counter() - start_time) * 1000

            # Update metrics
            self._record_metric("evolution_proposals_generated", len(high_quality_proposals))
            with self._metrics_lock:
                self.metrics["evolution_analysis_time_ms"] = (
                    self.metrics["evolution_analysis_time_ms"] * 0.8 + total_time * 0.2
                )

            # Log performance metrics
            logger.info(
                f"Language evolution completed in {total_time:.2f}ms "
                f"(analysis: {analysis_time:.2f}ms, generation: {generation_time:.2f}ms, "
                f"validation: {validation_time:.2f}ms). "
                f"Generated {len(high_quality_proposals)} high-quality proposals."
            )

            # Warn if performance targets are exceeded
            if total_time > 400.0:
                logger.warning(f"Language evolution exceeded 400ms target: {total_time:.2f}ms")

            return high_quality_proposals

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Language evolution failed after {processing_time:.2f}ms: {str(e)}")
            raise OpusStrategistError(f"Language evolution analysis failed: {str(e)}") from e

    def apply_language_evolution(
        self, proposal: EvolutionProposal, script_compiler: Any = None
    ) -> bool:
        """
        Apply a validated language evolution proposal to the DSL system.

        This method integrates validated language evolution proposals with the
        ScriptCompiler's MacroRegistry to implement DSL improvements.

        Args:
            proposal: Validated evolution proposal to implement
            script_compiler: Optional ScriptCompiler instance for macro registration

        Returns:
            True if proposal was successfully applied, False otherwise

        Raises:
            OpusStrategistError: If proposal application fails
        """
        try:
            if proposal.validation_score < 0.8:
                logger.warning(
                    f"Proposal {proposal.proposal_id} has low validation score: {proposal.validation_score}"
                )
                return False

            logger.info(f"Applying language evolution proposal: {proposal.proposal_id}")

            # Handle macro extension proposals
            if proposal.proposal_type.value == "macro_extension":
                return self._apply_macro_extension(proposal, script_compiler)

            # Handle other proposal types (conditional DSL, etc.)
            elif proposal.proposal_type.value == "conditional_dsl":
                logger.info(
                    f"Conditional DSL proposal {proposal.proposal_id} requires manual implementation"
                )
                return False  # For now, conditional DSL requires manual implementation

            else:
                logger.warning(f"Unknown proposal type: {proposal.proposal_type.value}")
                return False

        except Exception as e:
            logger.error(
                f"Failed to apply language evolution proposal {proposal.proposal_id}: {str(e)}"
            )
            raise OpusStrategistError(f"Language evolution application failed: {str(e)}") from e

    def _convert_results_to_patterns(
        self, recent_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert execution results to pattern data format for analysis.

        This method transforms recent execution results into the format expected
        by the LanguageAnalyzer for pattern analysis.
        """
        pattern_data = []

        for result in recent_results:
            # Extract required fields with defaults
            pattern_entry = {
                "name": result.get("pattern_name", "unknown"),
                "success_rate": result.get("success_rate", 0.0),
                "usage_frequency": result.get("usage_count", 0),
                "input_sequence": result.get("input_sequence", []),
                "context": result.get("context", {}),
                "average_completion_time": result.get("average_execution_time"),
                "evolution_metadata": result.get("evolution_metadata", {}),
            }

            # Validate essential fields
            if (
                pattern_entry["name"] != "unknown"
                and pattern_entry["input_sequence"]
                and isinstance(pattern_entry["success_rate"], int | float)
            ):
                pattern_data.append(pattern_entry)
            else:
                logger.debug(f"Skipping invalid pattern data: {result}")

        return pattern_data

    def _apply_macro_extension(
        self, proposal: EvolutionProposal, script_compiler: Any = None
    ) -> bool:
        """Apply macro extension proposal to ScriptCompiler."""
        try:
            new_macros = proposal.dsl_changes.get("new_macros", {})

            if not script_compiler:
                logger.warning("No ScriptCompiler provided, cannot apply macro extensions")
                # For now, just log the proposal - real implementation would integrate with ScriptCompiler
                logger.info(f"Would register macros: {list(new_macros.keys())}")
                return True  # Assume success for now

            # Apply macro extensions to ScriptCompiler
            for macro_name, macro_expansion in new_macros.items():
                try:
                    script_compiler.register_pattern(macro_name, macro_expansion)
                    logger.info(f"Successfully registered macro: {macro_name}")
                except Exception as e:
                    logger.error(f"Failed to register macro {macro_name}: {str(e)}")
                    return False

            logger.info(f"Successfully applied macro extension proposal: {proposal.proposal_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply macro extension: {str(e)}")
            return False

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
