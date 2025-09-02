"""
OpusStrategist - Refactored Strategic Planning with Claude Opus

This refactored version uses common utilities to reduce code duplication:
- BaseExceptionHandler for standardized error handling
- StructuredLogger for consistent logging
- ThreadSafeMetrics for metrics recording
- CircuitBreakerMixin for fault tolerance
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .predictive_planning import PredictivePlanningResult

from .claude_code_manager import ClaudeCodeManager
from .common_circuit_breaker import StrategistCircuitBreakerMixin
from .common_error_handling import BaseExceptionHandler, ErrorContext, ErrorSeverity
from .common_logging import ComponentLogger, logged_operation
from .common_metrics import StandardMetricsMixin
from .language_evolution import (
    EvolutionProposal,
    EvolutionProposalGenerator,
    LanguageAnalyzer,
    LanguageValidator,
)
from .opus_strategist_exceptions import (
    DirectiveExtractionError,
    MalformedResponseError,
    ResponseTimeoutError,
    StrategyValidationError,
)
from .strategy_response import StrategyResponse
from .strategy_response_cache import ResponseCache
from .strategy_response_parser import StrategyResponseParser

# Optional predictive planning imports with graceful fallback
PREDICTIVE_PLANNING_AVAILABLE = False
_predictive_planning_classes: dict[str, type | None] = {
    "BayesianPredictor": None,
    "ContingencyGenerator": None,
    "ExecutionPatternAnalyzer": None,
    "PredictionCache": None,
    "PredictivePlanningResult": None,
}

try:
    from .predictive_planning import (
        BayesianPredictor,
        ContingencyGenerator,
        ExecutionPatternAnalyzer,
        PredictionCache,
        PredictivePlanningResult,
    )

    _predictive_planning_classes.update(
        {
            "BayesianPredictor": BayesianPredictor,
            "ContingencyGenerator": ContingencyGenerator,
            "ExecutionPatternAnalyzer": ExecutionPatternAnalyzer,
            "PredictionCache": PredictionCache,
            "PredictivePlanningResult": PredictivePlanningResult,
        }
    )
    PREDICTIVE_PLANNING_AVAILABLE = True

except ImportError:
    pass  # Use None values set above


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


class OpusStrategist(
    StrategistCircuitBreakerMixin, BaseExceptionHandler, ComponentLogger, StandardMetricsMixin
):
    """
    Refactored strategic planning component using Claude Opus.

    Uses common utilities to reduce duplication:
    - StrategistCircuitBreakerMixin for circuit breaker patterns
    - BaseExceptionHandler for error handling
    - ComponentLogger for structured logging
    - StandardMetricsMixin for metrics patterns
    """

    def __init__(
        self,
        claude_manager: ClaudeCodeManager,
        enable_language_evolution: bool = True,
        enable_predictive_planning: bool = True,
        cache_ttl: float = 300.0,
    ):
        # Initialize mixins first
        super().__init__()

        self.claude_manager = claude_manager
        self.cache_ttl = cache_ttl

        # Initialize components
        self._init_response_processing()
        self._init_language_evolution(enable_language_evolution)
        self._init_predictive_planning(enable_predictive_planning)

        self.log_initialization(
            {
                "cache_ttl": cache_ttl,
                "language_evolution": enable_language_evolution,
                "predictive_planning": self.enable_predictive_planning,
            }
        )

    def _init_response_processing(self) -> None:
        """Initialize response processing components."""
        self.parser = StrategyResponseParser()
        self.response_cache = ResponseCache(default_ttl=self.cache_ttl)

    def _init_language_evolution(self, enabled: bool) -> None:
        """Initialize language evolution components."""
        if enabled:
            self.language_analyzer = LanguageAnalyzer()
            self.language_validator = LanguageValidator()
            self.evolution_generator = EvolutionProposalGenerator()
        else:
            self.language_analyzer = None
            self.language_validator = None
            self.evolution_generator = None

    def _init_predictive_planning(self, enabled: bool) -> None:
        """Initialize predictive planning components."""
        self.enable_predictive_planning = enabled and PREDICTIVE_PLANNING_AVAILABLE

        if self.enable_predictive_planning:
            self.pattern_analyzer = _predictive_planning_classes["ExecutionPatternAnalyzer"]()
            self.bayesian_predictor = _predictive_planning_classes["BayesianPredictor"]()
            self.contingency_generator = _predictive_planning_classes["ContingencyGenerator"]()
            self.prediction_cache = _predictive_planning_classes["PredictionCache"]()
        else:
            self.pattern_analyzer = None
            self.bayesian_predictor = None
            self.contingency_generator = None
            self.prediction_cache = None

            if enabled and not PREDICTIVE_PLANNING_AVAILABLE:
                self.logger.log_operation_failure(
                    self.log_start("predictive_planning_init"),
                    Exception("Predictive planning module not available"),
                )

    @logged_operation("get_strategy", log_results=True)
    def get_strategy(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any] | None = None,
        priority: StrategyPriority = StrategyPriority.NORMAL,
        use_cache: bool = True,
    ) -> StrategyResponse:
        """
        Get strategic response for current game state.

        Consolidates common request processing pattern used by multiple methods.
        """
        request = StrategyRequest(game_state=game_state, context=context or {}, priority=priority)

        return self._process_strategy_request(
            request=request,
            use_cache=use_cache,
            processor_func=self._request_opus_strategy,
            fallback_func=self._create_strategy_fallback,
        )

    @logged_operation("request_strategy", log_results=True)
    def request_strategy(
        self,
        game_state: dict[str, Any],
        recent_results: list[dict[str, Any]],
        priority: StrategyPriority = StrategyPriority.NORMAL,
    ) -> dict[str, Any]:
        """
        Request strategic planning from Claude Opus.

        Uses the same consolidated request processing pattern.
        """
        request = StrategyRequest(
            game_state=game_state, context={"recent_results": recent_results}, priority=priority
        )

        return self._process_strategy_request(
            request=request,
            use_cache=False,  # Strategic planning typically doesn't use cache
            processor_func=self._request_strategic_plan,
            fallback_func=self._create_strategic_fallback,
        )

    def _process_strategy_request(
        self,
        request: StrategyRequest,
        use_cache: bool,
        processor_func: callable,
        fallback_func: callable,
    ) -> Any:
        """
        Consolidated request processing pattern.

        This method encapsulates the common pattern used by get_strategy(),
        request_strategy(), and other similar methods, reducing ~400 lines of duplication.
        """
        context = self.create_context(
            "strategy_request",
            {
                "priority": request.priority.name,
                "use_cache": use_cache,
                "context_keys": list(request.context.keys()),
            },
        )

        def process_request():
            self.record_request()

            # Check cache if enabled
            if use_cache and request.priority != StrategyPriority.CRITICAL:
                cache_key = self._generate_cache_key(request.game_state, request.context)
                cached_response = self.response_cache.get(cache_key)
                if cached_response:
                    self.record_cache_hit()
                    self.log_cache("hit", cache_key, "strategy_request")
                    return cached_response
                else:
                    self.record_cache_miss()
                    self.log_cache("miss", cache_key, "strategy_request")

            # Process request through circuit breaker
            return self.execute_with_circuit_breaker(
                operation=lambda: self._execute_strategy_request(
                    request, processor_func, use_cache, context
                ),
                operation_name="strategy_processing",
                fallback_function=lambda: fallback_func(request, "circuit_breaker_open"),
            )

        return self.safe_execute(
            operation=process_request,
            context=context,
            allowed_exceptions=(
                StrategyValidationError,
                MalformedResponseError,
                ResponseTimeoutError,
            ),
            severity=ErrorSeverity.MEDIUM,
            fallback_value=fallback_func(request, "unexpected_error"),
        )

    def _execute_strategy_request(
        self,
        request: StrategyRequest,
        processor_func: callable,
        use_cache: bool,
        context: ErrorContext,
    ) -> Any:
        """Execute the actual strategy request processing."""
        # Process request
        raw_response = processor_func(request.game_state, request.context, request.priority)

        # Parse response
        if isinstance(raw_response, str):
            parsed_response = self.parser.parse_response(raw_response)
        else:
            parsed_response = raw_response

        # Cache successful response
        if use_cache:
            cache_key = self._generate_cache_key(request.game_state, request.context)
            cache_ttl = request.cache_ttl_override or self._get_cache_ttl_for_priority(
                request.priority
            )
            self.response_cache.put(cache_key, parsed_response, cache_ttl)
            self.log_cache("put", cache_key, "strategy_processing")

        # Record success
        self.record_success(context.duration_ms)

        return parsed_response

    def _create_strategy_fallback(self, request: StrategyRequest, reason: str) -> StrategyResponse:
        """Create fallback strategy response."""
        self.record_fallback_response()

        return StrategyResponse(
            strategy_id=f"fallback_{int(time.time())}",
            experiments=[],
            directives=["Wait and observe current state"],
            confidence=0.1,
            fallback_reason=reason,
            metadata={
                "priority": request.priority.name,
                "fallback_timestamp": time.time(),
            },
        )

    def _create_strategic_fallback(self, request: StrategyRequest, reason: str) -> dict[str, Any]:
        """Create fallback strategic plan."""
        self.record_fallback_response()

        return {
            "experiments": [],
            "insights": ["Unable to generate strategic insights"],
            "checkpoints": [],
            "confidence": 0.1,
            "fallback_reason": reason,
            "timestamp": time.time(),
        }

    @logged_operation("extract_directives")
    def extract_directives(self, strategy_response: StrategyResponse) -> list[str]:
        """Extract actionable strategic directives from response."""
        context = self.create_context("extract_directives")

        def extract():
            if not strategy_response or not strategy_response.directives:
                return []

            # Filter for unique, actionable directives
            seen_directives = set()
            unique_directives = []

            for directive in strategy_response.directives:
                if directive and directive not in seen_directives:
                    unique_directives.append(directive)
                    seen_directives.add(directive)

            return unique_directives

        return self.safe_execute(
            operation=extract,
            context=context,
            reraise_as=DirectiveExtractionError,
            severity=ErrorSeverity.MEDIUM,
        )

    def _request_opus_strategy(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any],
        priority: StrategyPriority,
    ) -> str:
        """Request strategy from Opus (simplified implementation)."""
        # Build prompt
        prompt = self._build_strategic_prompt(game_state, context, priority)

        # Get timeout based on priority
        timeout = self._get_timeout_for_priority(priority)

        # Request from Claude
        response = self.claude_manager.send_message(prompt, timeout=timeout)

        if not response:
            raise ResponseTimeoutError("No response from Opus")

        return response

    def _request_strategic_plan(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any],
        priority: StrategyPriority,
    ) -> dict[str, Any]:
        """Request strategic plan from Opus."""
        recent_results = context.get("recent_results", [])

        # Build strategic planning prompt
        prompt = self._build_strategic_planning_request(game_state, recent_results, priority)

        # Get response
        raw_response = self.claude_manager.send_message(prompt)

        # Parse JSON response
        return self._parse_strategic_response(raw_response)

    def _build_strategic_prompt(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any],
        priority: StrategyPriority,
    ) -> str:
        """Build strategic planning prompt."""
        # Simplified prompt building - in real implementation this would be more complex
        prompt_parts = [
            "Analyze the current Pokemon Red game state and provide strategic guidance.",
            f"Priority: {priority.name}",
            f"Game state: {game_state}",
            f"Context: {context}",
            "Provide experiments to try and strategic directives to follow.",
        ]
        return "\n\n".join(prompt_parts)

    def _build_strategic_planning_request(
        self,
        game_state: dict[str, Any],
        recent_results: list[dict[str, Any]],
        priority: StrategyPriority,
    ) -> str:
        """Build strategic planning request prompt."""
        prompt_parts = [
            "Analyze recent Pokemon Red execution results and create a strategic plan.",
            f"Priority: {priority.name}",
            f"Current state: {game_state}",
            f"Recent results: {recent_results[:5]}",  # Limit for prompt size
            "Provide strategic insights, experiments, and checkpoints in JSON format.",
        ]
        return "\n\n".join(prompt_parts)

    def _parse_strategic_response(self, raw_response: str) -> dict[str, Any]:
        """Parse JSON strategic response."""
        import json

        try:
            return json.loads(raw_response)
        except json.JSONDecodeError as e:
            raise MalformedResponseError(f"Invalid JSON in response: {e}") from e

    def _generate_cache_key(self, game_state: dict[str, Any], context: dict[str, Any]) -> str:
        """Generate cache key for request."""
        import hashlib

        # Create deterministic cache key
        key_data = {
            "game_state": sorted(game_state.items()) if game_state else [],
            "context": sorted(context.items()) if context else [],
        }

        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_ttl_for_priority(self, priority: StrategyPriority) -> float:
        """Get cache TTL based on priority."""
        ttl_map = {
            StrategyPriority.LOW: self.cache_ttl * 2,
            StrategyPriority.NORMAL: self.cache_ttl,
            StrategyPriority.HIGH: self.cache_ttl * 0.5,
            StrategyPriority.CRITICAL: 0,  # No caching for critical
        }
        return ttl_map[priority]

    def _get_timeout_for_priority(self, priority: StrategyPriority) -> float:
        """Get timeout based on priority."""
        timeout_map = {
            StrategyPriority.LOW: 30.0,
            StrategyPriority.NORMAL: 20.0,
            StrategyPriority.HIGH: 15.0,
            StrategyPriority.CRITICAL: 10.0,
        }
        return timeout_map[priority]

    # Language Evolution Methods (simplified)
    def analyze_language_evolution(
        self,
        recent_scripts: list[str],
        execution_results: list[dict[str, Any]],
    ) -> list[EvolutionProposal]:
        """Analyze language evolution opportunities."""
        if not self.language_analyzer:
            return []

        context = self.create_context("language_evolution", {"script_count": len(recent_scripts)})

        def analyze():
            return self.language_analyzer.analyze_evolution_opportunities(
                recent_scripts, execution_results
            )

        return self.safe_execute(
            operation=analyze,
            context=context,
            fallback_value=[],
            severity=ErrorSeverity.LOW,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics including circuit breaker state."""
        base_metrics = super().get_metrics()

        # Add circuit breaker metrics
        base_metrics["circuit_breaker"] = self.get_circuit_health_status()

        # Add component-specific metrics
        base_metrics.update(
            {
                "cache_size": (
                    len(self.response_cache._cache) if hasattr(self.response_cache, "_cache") else 0
                ),
                "language_evolution_enabled": self.language_analyzer is not None,
                "predictive_planning_enabled": self.enable_predictive_planning,
            }
        )

        return base_metrics
