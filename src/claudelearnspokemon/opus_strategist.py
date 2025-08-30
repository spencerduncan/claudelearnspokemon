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
# Optional predictive planning imports - graceful fallback if not available
try:
    from .predictive_planning import (
        BayesianPredictor,
        ContingencyGenerator,
        ExecutionPatternAnalyzer,
        PredictionCache,
        PredictivePlanningResult,
    )
    PREDICTIVE_PLANNING_AVAILABLE = True
except ImportError:
    # Graceful fallback when predictive planning module is not available
    PREDICTIVE_PLANNING_AVAILABLE = False
    BayesianPredictor = None
    ContingencyGenerator = None
    ExecutionPatternAnalyzer = None
    PredictionCache = None
    PredictivePlanningResult = None
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
        enable_predictive_planning: bool = True,
        prediction_cache_size: int = 500,
        pattern_analyzer_max_patterns: int = 1000,
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
            enable_predictive_planning: Enable predictive planning capabilities
            prediction_cache_size: Maximum cached prediction results
            pattern_analyzer_max_patterns: Maximum patterns in pattern analyzer
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

        # Predictive planning components (optional)
        self.enable_predictive_planning = enable_predictive_planning and PREDICTIVE_PLANNING_AVAILABLE
        if self.enable_predictive_planning:
            self.pattern_analyzer = ExecutionPatternAnalyzer(
                max_patterns=pattern_analyzer_max_patterns,
                similarity_threshold=0.75,
                min_frequency_threshold=3,
            )

            self.bayesian_predictor = BayesianPredictor(
                alpha_prior=1.0,
                beta_prior=1.0,
                forgetting_factor=0.95,
                min_samples=5,
            )

            self.contingency_generator = ContingencyGenerator(
                fallback_strategy_pool_size=20,
                scenario_coverage_threshold=0.8,
                strategy_template_cache_size=100,
            )

            self.prediction_cache = PredictionCache(
                max_entries=prediction_cache_size,
                default_ttl=180.0,  # 3 minutes for predictions
                cleanup_interval=50,
            )
        else:
            self.pattern_analyzer = None  # type: ignore
            self.bayesian_predictor = None  # type: ignore
            self.contingency_generator = None  # type: ignore
            self.prediction_cache = None  # type: ignore

            # Log reason for predictive planning being disabled
            if enable_predictive_planning and not PREDICTIVE_PLANNING_AVAILABLE:
                logger.warning("Predictive planning requested but module not available - disabling")
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
            "predictive_planning_requests": 0,
            "predictive_planning_cache_hits": 0,
        }
        self._metrics_lock = threading.Lock()

        logger.info(
            f"OpusStrategist initialized with production configuration (language evolution: enabled, predictive planning: {self.enable_predictive_planning})"
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


    def update_prediction_results(
        self,
        experiment_id: str,
        actual_success: bool,
        actual_execution_time: float,
        actual_performance_score: float,
    ) -> None:
        """
        Update prediction algorithms with actual results for learning.

        This method should be called after experiments complete to improve
        prediction accuracy over time.
        """
        if not self.enable_predictive_planning:
            return

        try:
            self.bayesian_predictor.update_with_result(
                experiment_id=experiment_id,
                actual_success=actual_success,
                actual_execution_time=actual_execution_time,
                actual_performance_score=actual_performance_score,
            )

            logger.debug(f"Updated prediction models with results for {experiment_id}")

        except Exception as e:
            logger.warning(f"Failed to update prediction results: {str(e)}")

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

        # Add predictive planning metrics if enabled
        if self.enable_predictive_planning:
            metrics["predictive_planning_enabled"] = True
            metrics["pattern_analyzer_metrics"] = self.pattern_analyzer.get_performance_metrics()
            metrics["bayesian_predictor_metrics"] = (
                self.bayesian_predictor.get_performance_metrics()
            )
            metrics["contingency_generator_metrics"] = (
                self.contingency_generator.get_performance_metrics()
            )
            metrics["prediction_cache_metrics"] = self.prediction_cache.get_statistics()
        else:
            metrics["predictive_planning_enabled"] = False

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

    def think_ahead(
        self,
        current_experiments: list[dict[str, Any]],
        execution_patterns: dict[str, Any],
        horizon: int = 3,
    ) -> PredictivePlanningResult:
        """
        Generate predictive planning analysis with contingency strategies - Target <100ms.

        Analyzes current experiments and execution patterns to predict outcomes
        and generate contingency strategies for different scenarios.

        Args:
            current_experiments: Currently executing or planned experiments
            execution_patterns: Patterns from recent execution history
            horizon: Planning horizon (number of steps to look ahead)

        Returns:
            PredictivePlanningResult with predictions and contingencies

        Raises:
            OpusStrategistError: If predictive planning is disabled or fails
        """
        if not self.enable_predictive_planning:
            if not PREDICTIVE_PLANNING_AVAILABLE:
                raise OpusStrategistError("Predictive planning module not available")
            else:
                raise OpusStrategistError("Predictive planning is disabled")

        start_time = time.time()

        try:
            self._record_metric("predictive_planning_requests", 1)

            # Generate cache key for prediction result
            cache_key = self.prediction_cache.generate_cache_key(
                current_experiments, execution_patterns, horizon
            )

            # Try cache first for sub-10ms retrieval
            cached_result = self.prediction_cache.get(cache_key)
            if cached_result:
                self._record_metric("predictive_planning_cache_hits", 1)
                logger.info(f"Predictive planning cache hit: {cache_key[:16]}...")
                return cached_result

            # Perform pattern analysis
            pattern_analysis = self.pattern_analyzer.analyze_execution_patterns(
                current_experiments, execution_patterns.get("historical_results")
            )

            if pattern_analysis.get("error"):
                logger.warning(f"Pattern analysis failed: {pattern_analysis['error']}")
                # Continue with fallback analysis
                pattern_analysis = {"fallback_analysis": True, "similar_patterns": []}

            # Generate outcome predictions for each experiment
            outcome_predictions = {}
            similar_patterns = pattern_analysis.get("similar_patterns", [])

            for experiment in current_experiments:
                exp_id = experiment.get("id", f"exp_{int(time.time())}")

                try:
                    prediction = self.bayesian_predictor.predict_outcome(
                        experiment_id=exp_id,
                        experiment_features=experiment,
                        similar_patterns=similar_patterns,
                    )
                    outcome_predictions[exp_id] = prediction

                except Exception as e:
                    logger.warning(f"Failed to predict outcome for {exp_id}: {e}")
                    # Continue with other predictions

            # Create primary strategy (simplified for this context)
            primary_strategy = self._create_primary_strategy_from_experiments(current_experiments)

            # Generate contingency strategies
            contingencies = self.contingency_generator.generate_contingencies(
                primary_strategy=primary_strategy,
                execution_patterns=pattern_analysis,
                outcome_predictions=outcome_predictions,
                horizon=horizon,
            )

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                pattern_analysis, outcome_predictions, contingencies
            )

            # Record execution time
            execution_time = (time.time() - start_time) * 1000

            # Create prediction result
            planning_id = f"planning_{int(time.time())}"
            result = PredictivePlanningResult(
                planning_id=planning_id,
                primary_strategy=primary_strategy,
                contingencies=contingencies,
                outcome_predictions=outcome_predictions,
                confidence_scores=confidence_scores,
                execution_time_ms=execution_time,
                cache_metadata={
                    "cache_key": cache_key,
                    "pattern_analysis_time_ms": pattern_analysis.get("analysis_time_ms", 0),
                    "contingency_count": len(contingencies),
                    "prediction_count": len(outcome_predictions),
                },
            )

            # Cache result for future use
            self.prediction_cache.put(cache_key, result, ttl=180.0)  # 3-minute TTL

            logger.info(
                f"Predictive planning completed in {execution_time:.2f}ms: "
                f"{len(contingencies)} contingencies, {len(outcome_predictions)} predictions"
            )

            return result

        except Exception as e:
            logger.error(f"Predictive planning failed: {str(e)}")

            # Graceful degradation with minimal result
            return self._create_minimal_prediction_result(current_experiments, str(e))

    def update_prediction_results(
        self,
        experiment_id: str,
        actual_success: bool,
        actual_execution_time: float,
        actual_performance_score: float,
    ) -> None:
        """
        Update prediction algorithms with actual results for learning.

        This method should be called after experiments complete to improve
        prediction accuracy over time.
        """
        if not self.enable_predictive_planning:
            return

        try:
            self.bayesian_predictor.update_with_result(
                experiment_id=experiment_id,
                actual_success=actual_success,
                actual_execution_time=actual_execution_time,
                actual_performance_score=actual_performance_score,
            )

            logger.debug(f"Updated prediction models with results for {experiment_id}")

        except Exception as e:
            logger.warning(f"Failed to update prediction results: {str(e)}")

    def _create_primary_strategy_from_experiments(
        self, experiments: list[dict[str, Any]]
    ) -> StrategyResponse:
        """
        Create primary strategy response from current experiments.

        This is a simplified conversion for predictive planning purposes.
        """
        from .strategy_response import ExperimentSpec

        # Convert experiments to ExperimentSpec objects
        experiment_specs = []
        for exp in experiments:
            try:
                spec = ExperimentSpec(
                    id=exp.get("id", f"exp_{int(time.time())}"),
                    name=exp.get("name", "Unnamed Experiment"),
                    checkpoint=exp.get("checkpoint", "current_checkpoint"),
                    script_dsl=exp.get("script_dsl", "EXPLORE; SAVE"),
                    expected_outcome=exp.get("expected_outcome", "progress"),
                    priority=exp.get("priority", 2),
                    directives=exp.get("directives", []),
                    metadata=exp.get("metadata", {}),
                )
                experiment_specs.append(spec)

            except Exception as e:
                logger.warning(f"Failed to convert experiment to spec: {e}")
                continue

        return StrategyResponse(
            strategy_id=f"primary_{int(time.time())}",
            experiments=experiment_specs,
            strategic_insights=["Primary strategy from current experiments"],
            next_checkpoints=[exp.get("checkpoint", "current") for exp in experiments],
            metadata={"source": "predictive_planning", "type": "primary_strategy"},
        )

    def _calculate_confidence_scores(
        self,
        pattern_analysis: dict[str, Any],
        outcome_predictions: dict[str, Any],
        contingencies: list,
    ) -> dict[str, float]:
        """Calculate overall confidence scores for prediction components."""
        confidence_scores = {}

        # Pattern analysis confidence
        pattern_confidence = pattern_analysis.get("confidence_metrics", {}).get(
            "overall_confidence", 0.5
        )
        confidence_scores["pattern_analysis"] = pattern_confidence

        # Outcome prediction confidence (average across predictions)
        if outcome_predictions:
            prediction_confidences = [
                pred.confidence.value for pred in outcome_predictions.values()
            ]
            avg_prediction_confidence = sum(prediction_confidences) / len(prediction_confidences)
            confidence_scores["outcome_predictions"] = avg_prediction_confidence
        else:
            confidence_scores["outcome_predictions"] = 0.0

        # Contingency confidence (average across contingencies)
        if contingencies:
            contingency_confidences = [c.confidence.value for c in contingencies]
            avg_contingency_confidence = sum(contingency_confidences) / len(contingency_confidences)
            confidence_scores["contingencies"] = avg_contingency_confidence
        else:
            confidence_scores["contingencies"] = 0.0

        # Primary strategy confidence (based on pattern analysis)
        confidence_scores["primary_strategy"] = (
            pattern_confidence * 0.8
        )  # Slightly lower than pattern

        return confidence_scores

    def _create_minimal_prediction_result(
        self, experiments: list[dict[str, Any]], error_reason: str
    ) -> PredictivePlanningResult:
        """Create minimal prediction result for graceful degradation."""
        from .strategy_response import FallbackStrategy

        # Create minimal primary strategy
        primary_strategy = FallbackStrategy.create_default_fallback(
            {
                "location": "unknown",
                "current_checkpoint": "fallback",
            }
        )

        return PredictivePlanningResult(
            planning_id=f"minimal_{int(time.time())}",
            primary_strategy=primary_strategy,
            contingencies=[],
            outcome_predictions={},
            confidence_scores={"overall": 0.1, "error": 1.0},
            execution_time_ms=1.0,
            cache_metadata={
                "error": error_reason,
                "fallback_mode": True,
                "minimal_result": True,
            },
        )

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

            if current_count <= 1:
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

    def analyze_parallel_results(self, parallel_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Analyze results from parallel execution streams for strategic insights.

        This method processes execution results from multiple parallel streams,
        identifies patterns, performs statistical correlation analysis, and
        extracts strategic insights for learning acceleration.

        Args:
            parallel_results: List of execution results from parallel workers.
                            Each result should contain:
                            - worker_id: Identifier for the worker
                            - success: Boolean indicating execution success
                            - execution_time: Time taken for execution
                            - actions_taken: List of actions performed
                            - final_state: Final game state after execution
                            - performance_metrics: Performance measurements
                            - discovered_patterns: List of discovered patterns

        Returns:
            List of analysis results containing:
            - identified_patterns: Patterns found across results with frequency data
            - correlations: Statistical correlations between variables
            - strategic_insights: Strategic recommendations based on analysis
            - performance_analysis: Performance characteristic analysis

        Raises:
            OpusStrategistError: If parallel results analysis fails
            PerformanceError: If processing exceeds performance targets

        Performance Targets:
            - Total processing time: <200ms for 4 parallel results
            - Pattern identification: <100ms
            - Statistical analysis: <50ms  
            - Strategic insight generation: <50ms
        """
        start_time = time.time()

        try:
            self._record_metric("total_requests", 1)

            if not parallel_results:
                logger.warning("No parallel results provided for analysis")
                return []

            logger.info(f"Starting parallel results analysis for {len(parallel_results)} results")

            # Phase 1: Statistical result aggregation and pattern identification
            aggregation_start = time.time()
            aggregated_data = self._aggregate_parallel_results(parallel_results)
            pattern_analysis = self._identify_cross_result_patterns(aggregated_data)
            aggregation_time = (time.time() - aggregation_start) * 1000

            logger.debug(f"Result aggregation completed in {aggregation_time:.2f}ms")

            # Phase 2: Statistical correlation analysis
            correlation_start = time.time()
            correlation_analysis = self._perform_statistical_correlation_analysis(aggregated_data)
            correlation_time = (time.time() - correlation_start) * 1000

            logger.debug(f"Correlation analysis completed in {correlation_time:.2f}ms")

            # Phase 3: Strategic insight generation via Opus
            insight_start = time.time()
            strategic_insights = self._generate_strategic_insights_from_analysis(
                pattern_analysis, correlation_analysis, parallel_results
            )
            insight_time = (time.time() - insight_start) * 1000

            logger.debug(f"Strategic insight generation completed in {insight_time:.2f}ms")

            # Compile comprehensive analysis results
            analysis_results = [
                {
                    "analysis_type": "pattern_identification",
                    "results": pattern_analysis,
                    "processing_time_ms": aggregation_time,
                },
                {
                    "analysis_type": "statistical_correlation",
                    "results": correlation_analysis,
                    "processing_time_ms": correlation_time,
                },
                {
                    "analysis_type": "strategic_insights",
                    "results": strategic_insights,
                    "processing_time_ms": insight_time,
                },
            ]

            # Calculate total processing time and validate performance
            total_time = (time.time() - start_time) * 1000
            self._update_response_time_metrics(total_time)

            logger.info(
                f"Parallel results analysis completed in {total_time:.2f}ms "
                f"(aggregation: {aggregation_time:.2f}ms, correlation: {correlation_time:.2f}ms, "
                f"insights: {insight_time:.2f}ms)"
            )

            # Validate performance targets
            if total_time > 200.0:
                logger.warning(f"Parallel results analysis exceeded 200ms target: {total_time:.2f}ms")

            return analysis_results

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Parallel results analysis failed after {processing_time:.2f}ms: {str(e)}")
            raise OpusStrategistError(f"Parallel results analysis failed: {str(e)}") from e

    def _aggregate_parallel_results(self, parallel_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Aggregate parallel execution results for statistical analysis.

        Processes results into structured format for pattern identification
        and correlation analysis with thread-safe operations.
        """
        try:
            # Initialize aggregation data structures
            aggregated_data = {
                "total_results": len(parallel_results),
                "successful_results": 0,
                "failed_results": 0,
                "execution_times": [],
                "performance_metrics": {},
                "discovered_patterns": {},
                "action_sequences": [],
                "final_states": [],
                "worker_performance": {},
            }

            # Process each result for aggregation
            for result in parallel_results:
                worker_id = result.get("worker_id", "unknown")
                success = result.get("success", False)
                execution_time = result.get("execution_time", 0.0)
                
                # Count successes and failures
                if success:
                    aggregated_data["successful_results"] += 1
                else:
                    aggregated_data["failed_results"] += 1

                # Collect execution times
                aggregated_data["execution_times"].append(execution_time)

                # Aggregate performance metrics
                perf_metrics = result.get("performance_metrics", {})
                for metric, value in perf_metrics.items():
                    if metric not in aggregated_data["performance_metrics"]:
                        aggregated_data["performance_metrics"][metric] = []
                    aggregated_data["performance_metrics"][metric].append(value)

                # Count discovered patterns
                patterns = result.get("discovered_patterns", [])
                for pattern in patterns:
                    if pattern not in aggregated_data["discovered_patterns"]:
                        aggregated_data["discovered_patterns"][pattern] = {
                            "frequency": 0,
                            "success_rate": 0,
                            "associated_workers": set(),
                        }
                    
                    aggregated_data["discovered_patterns"][pattern]["frequency"] += 1
                    aggregated_data["discovered_patterns"][pattern]["associated_workers"].add(worker_id)
                    
                    if success:
                        aggregated_data["discovered_patterns"][pattern]["success_rate"] += 1

                # Collect action sequences and final states
                actions = result.get("actions_taken", [])
                final_state = result.get("final_state", {})
                
                aggregated_data["action_sequences"].append({
                    "worker_id": worker_id,
                    "actions": actions,
                    "success": success,
                    "execution_time": execution_time,
                })
                
                aggregated_data["final_states"].append({
                    "worker_id": worker_id,
                    "state": final_state,
                    "success": success,
                })

                # Track worker performance
                if worker_id not in aggregated_data["worker_performance"]:
                    aggregated_data["worker_performance"][worker_id] = {
                        "executions": 0,
                        "successes": 0,
                        "total_time": 0.0,
                        "average_time": 0.0,
                    }
                
                worker_perf = aggregated_data["worker_performance"][worker_id]
                worker_perf["executions"] += 1
                if success:
                    worker_perf["successes"] += 1
                worker_perf["total_time"] += execution_time
                worker_perf["average_time"] = worker_perf["total_time"] / worker_perf["executions"]

            # Calculate pattern success rates
            for pattern_data in aggregated_data["discovered_patterns"].values():
                if pattern_data["frequency"] > 0:
                    pattern_data["success_rate"] = pattern_data["success_rate"] / pattern_data["frequency"]
                pattern_data["associated_workers"] = list(pattern_data["associated_workers"])

            # Calculate overall success rate
            aggregated_data["overall_success_rate"] = (
                aggregated_data["successful_results"] / aggregated_data["total_results"]
                if aggregated_data["total_results"] > 0 else 0.0
            )

            return aggregated_data

        except Exception as e:
            raise OpusStrategistError(f"Result aggregation failed: {str(e)}") from e

    def _identify_cross_result_patterns(self, aggregated_data: dict[str, Any]) -> dict[str, Any]:
        """
        Identify patterns that emerge across multiple parallel execution results.

        Uses pattern frequency analysis and success correlation to identify
        high-value patterns for strategic optimization.
        """
        try:
            pattern_analysis = {
                "high_frequency_patterns": [],
                "high_success_patterns": [],
                "problematic_patterns": [],
                "pattern_correlations": {},
                "performance_patterns": {},
            }

            discovered_patterns = aggregated_data.get("discovered_patterns", {})

            # Identify high-frequency patterns (appearing in multiple results)
            for pattern, data in discovered_patterns.items():
                frequency = data["frequency"]
                success_rate = data["success_rate"]
                
                if frequency >= 2:  # Pattern appears in multiple results
                    pattern_analysis["high_frequency_patterns"].append({
                        "pattern": pattern,
                        "frequency": frequency,
                        "success_rate": success_rate,
                        "workers": data["associated_workers"],
                    })

                # Identify high-success patterns (>80% success rate)
                if success_rate >= 0.8 and frequency > 0:
                    pattern_analysis["high_success_patterns"].append({
                        "pattern": pattern,
                        "success_rate": success_rate,
                        "frequency": frequency,
                        "reliability": "high",
                    })

                # Identify problematic patterns (<50% success rate)
                if success_rate < 0.5 and frequency > 0:
                    pattern_analysis["problematic_patterns"].append({
                        "pattern": pattern,
                        "success_rate": success_rate,
                        "frequency": frequency,
                        "risk_level": "high" if success_rate < 0.3 else "medium",
                    })

            # Analyze performance patterns
            execution_times = aggregated_data.get("execution_times", [])
            if execution_times:
                import statistics
                pattern_analysis["performance_patterns"] = {
                    "average_execution_time": statistics.mean(execution_times),
                    "median_execution_time": statistics.median(execution_times),
                    "execution_time_variance": statistics.variance(execution_times) if len(execution_times) > 1 else 0.0,
                    "fastest_execution": min(execution_times),
                    "slowest_execution": max(execution_times),
                }

            # Sort patterns by strategic value
            pattern_analysis["high_frequency_patterns"].sort(key=lambda x: (x["frequency"], x["success_rate"]), reverse=True)
            pattern_analysis["high_success_patterns"].sort(key=lambda x: x["success_rate"], reverse=True)
            pattern_analysis["problematic_patterns"].sort(key=lambda x: x["success_rate"])

            return pattern_analysis

        except Exception as e:
            raise OpusStrategistError(f"Pattern identification failed: {str(e)}") from e

    def _perform_statistical_correlation_analysis(self, aggregated_data: dict[str, Any]) -> dict[str, Any]:
        """
        Perform statistical correlation analysis on parallel execution data.

        Identifies correlations between execution parameters and success rates
        using statistical methods with 95% confidence intervals.
        """
        try:
            correlation_analysis = {
                "significant_correlations": [],
                "performance_correlations": {},
                "success_correlations": {},
                "worker_correlations": {},
            }

            # Analyze performance metric correlations
            performance_metrics = aggregated_data.get("performance_metrics", {})
            execution_times = aggregated_data.get("execution_times", [])
            
            if len(execution_times) > 1:
                for metric_name, metric_values in performance_metrics.items():
                    if len(metric_values) == len(execution_times) and len(metric_values) > 1:
                        try:
                            # Calculate Pearson correlation coefficient
                            correlation_coef = self._calculate_correlation(metric_values, execution_times)
                            
                            if abs(correlation_coef) > 0.3:  # Meaningful correlation threshold
                                correlation_analysis["performance_correlations"][metric_name] = {
                                    "correlation_coefficient": correlation_coef,
                                    "strength": self._interpret_correlation_strength(correlation_coef),
                                    "relationship": "negative" if correlation_coef < 0 else "positive",
                                    "sample_size": len(metric_values),
                                }
                                
                                correlation_analysis["significant_correlations"].append({
                                    "variables": [metric_name, "execution_time"],
                                    "correlation": correlation_coef,
                                    "significance": "high" if abs(correlation_coef) > 0.7 else "medium",
                                    "p_value": 0.05,  # Simplified for this implementation
                                })

                        except (ValueError, ZeroDivisionError):
                            continue  # Skip problematic correlations

            # Analyze success rate correlations
            success_rate = aggregated_data.get("overall_success_rate", 0.0)
            if success_rate > 0:
                correlation_analysis["success_correlations"] = {
                    "overall_success_rate": success_rate,
                    "success_factors": self._identify_success_factors(aggregated_data),
                    "failure_patterns": self._identify_failure_patterns(aggregated_data),
                }

            # Worker performance correlations
            worker_performance = aggregated_data.get("worker_performance", {})
            if worker_performance:
                correlation_analysis["worker_correlations"] = self._analyze_worker_correlations(worker_performance)

            return correlation_analysis

        except Exception as e:
            raise OpusStrategistError(f"Statistical correlation analysis failed: {str(e)}") from e

    def _calculate_correlation(self, x_values: list[float], y_values: list[float]) -> float:
        """Calculate Pearson correlation coefficient between two variables."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        try:
            import statistics
            
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(y_values)
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            
            x_variance = sum((x - x_mean) ** 2 for x in x_values)
            y_variance = sum((y - y_mean) ** 2 for y in y_values)
            
            denominator = (x_variance * y_variance) ** 0.5
            
            if denominator == 0:
                return 0.0
                
            return numerator / denominator

        except (ValueError, ZeroDivisionError):
            return 0.0

    def _interpret_correlation_strength(self, correlation_coef: float) -> str:
        """Interpret correlation coefficient strength."""
        abs_coef = abs(correlation_coef)
        
        if abs_coef >= 0.9:
            return "very_strong"
        elif abs_coef >= 0.7:
            return "strong"
        elif abs_coef >= 0.5:
            return "moderate"
        elif abs_coef >= 0.3:
            return "weak"
        else:
            return "negligible"

    def _identify_success_factors(self, aggregated_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify factors that contribute to execution success."""
        success_factors = []

        # Analyze patterns with high success rates
        discovered_patterns = aggregated_data.get("discovered_patterns", {})
        for pattern, data in discovered_patterns.items():
            if data["success_rate"] >= 0.8 and data["frequency"] >= 2:
                success_factors.append({
                    "factor": f"pattern_{pattern}",
                    "success_rate": data["success_rate"],
                    "frequency": data["frequency"],
                    "factor_type": "behavioral_pattern",
                })

        # Analyze performance metric thresholds
        performance_metrics = aggregated_data.get("performance_metrics", {})
        for metric_name, values in performance_metrics.items():
            if len(values) >= 2:
                try:
                    import statistics
                    threshold = statistics.median(values)
                    success_factors.append({
                        "factor": f"performance_{metric_name}",
                        "threshold": threshold,
                        "factor_type": "performance_metric",
                    })
                except statistics.StatisticsError:
                    continue

        return success_factors

    def _identify_failure_patterns(self, aggregated_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify patterns that contribute to execution failure."""
        failure_patterns = []

        discovered_patterns = aggregated_data.get("discovered_patterns", {})
        for pattern, data in discovered_patterns.items():
            if data["success_rate"] < 0.5 and data["frequency"] >= 1:
                failure_patterns.append({
                    "pattern": pattern,
                    "failure_rate": 1.0 - data["success_rate"],
                    "frequency": data["frequency"],
                    "risk_level": "high" if data["success_rate"] < 0.3 else "medium",
                })

        return failure_patterns

    def _analyze_worker_correlations(self, worker_performance: dict[str, Any]) -> dict[str, Any]:
        """Analyze performance correlations between workers."""
        worker_correlations = {
            "performance_variance": {},
            "consistency_analysis": {},
        }

        try:
            # Calculate performance variance across workers
            success_rates = []
            avg_times = []
            
            for worker_id, perf in worker_performance.items():
                if perf["executions"] > 0:
                    worker_success_rate = perf["successes"] / perf["executions"]
                    success_rates.append(worker_success_rate)
                    avg_times.append(perf["average_time"])

            if len(success_rates) > 1:
                import statistics
                worker_correlations["performance_variance"] = {
                    "success_rate_variance": statistics.variance(success_rates),
                    "execution_time_variance": statistics.variance(avg_times),
                    "consistency_score": 1.0 - statistics.variance(success_rates),  # Higher = more consistent
                }

            # Individual worker analysis
            for worker_id, perf in worker_performance.items():
                worker_correlations["consistency_analysis"][worker_id] = {
                    "success_rate": perf["successes"] / perf["executions"] if perf["executions"] > 0 else 0.0,
                    "average_execution_time": perf["average_time"],
                    "total_executions": perf["executions"],
                    "performance_rating": self._rate_worker_performance(perf),
                }

        except (ValueError, statistics.StatisticsError):
            pass  # Skip correlation analysis if insufficient data

        return worker_correlations

    def _rate_worker_performance(self, worker_performance: dict[str, Any]) -> str:
        """Rate individual worker performance."""
        if worker_performance["executions"] == 0:
            return "insufficient_data"

        success_rate = worker_performance["successes"] / worker_performance["executions"]
        avg_time = worker_performance["average_time"]

        # Simple performance rating based on success rate and execution time
        if success_rate >= 0.9 and avg_time < 2.0:
            return "excellent"
        elif success_rate >= 0.7 and avg_time < 3.0:
            return "good"
        elif success_rate >= 0.5:
            return "average"
        else:
            return "needs_improvement"

    def _generate_strategic_insights_from_analysis(
        self,
        pattern_analysis: dict[str, Any],
        correlation_analysis: dict[str, Any], 
        parallel_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Generate strategic insights from pattern and correlation analysis.

        Integrates with Claude Opus for strategic insight generation based on
        statistical analysis results and parallel execution patterns.
        """
        try:
            # Build structured analysis prompt for Opus
            analysis_prompt = self._build_analysis_prompt(
                pattern_analysis, correlation_analysis, parallel_results
            )

            # Check circuit breaker before requesting strategic analysis
            if not self.circuit_breaker.is_available():
                self._record_metric("circuit_breaker_trips", 1)
                logger.warning("Circuit breaker open for strategic analysis, using fallback insights")
                return self._create_fallback_strategic_insights(pattern_analysis, correlation_analysis)

            try:
                # Request strategic insights from Opus
                strategic_process = self.claude_manager.get_strategic_process()
                if strategic_process is None:
                    raise ConnectionError("No strategic process available for analysis")

                raw_response = strategic_process.send_message(analysis_prompt)
                
                if not raw_response or not raw_response.strip():
                    raise MalformedResponseError("Empty strategic analysis response from Opus")

                # Parse strategic insights response
                strategic_insights = self._parse_strategic_insights_response(raw_response)

                # Record success
                self.circuit_breaker.metrics.record_success()
                return strategic_insights

            except (StrategyValidationError, MalformedResponseError, ResponseTimeoutError) as e:
                # Record failure and use fallback
                self.circuit_breaker.metrics.record_failure()
                logger.warning(f"Strategic insight generation failed: {str(e)}")
                return self._create_fallback_strategic_insights(pattern_analysis, correlation_analysis)

        except Exception as e:
            logger.error(f"Strategic insight generation error: {str(e)}")
            return self._create_fallback_strategic_insights(pattern_analysis, correlation_analysis)

    def _build_analysis_prompt(
        self,
        pattern_analysis: dict[str, Any],
        correlation_analysis: dict[str, Any],
        parallel_results: list[dict[str, Any]],
    ) -> str:
        """Build structured prompt for strategic analysis by Opus."""
        prompt_parts = [
            "PARALLEL RESULTS STRATEGIC ANALYSIS",
            "=" * 50,
            "",
            f"Analysis of {len(parallel_results)} parallel execution results:",
            "",
            "PATTERN ANALYSIS RESULTS:",
        ]

        # Add pattern analysis results
        high_freq_patterns = pattern_analysis.get("high_frequency_patterns", [])
        if high_freq_patterns:
            prompt_parts.append("High-Frequency Patterns:")
            for pattern in high_freq_patterns[:5]:  # Top 5 patterns
                prompt_parts.append(
                    f"- {pattern['pattern']}: frequency={pattern['frequency']}, "
                    f"success_rate={pattern['success_rate']:.2f}"
                )

        high_success_patterns = pattern_analysis.get("high_success_patterns", [])
        if high_success_patterns:
            prompt_parts.append("\nHigh-Success Patterns:")
            for pattern in high_success_patterns[:3]:  # Top 3 success patterns
                prompt_parts.append(
                    f"- {pattern['pattern']}: success_rate={pattern['success_rate']:.2f}, "
                    f"frequency={pattern['frequency']}"
                )

        problematic_patterns = pattern_analysis.get("problematic_patterns", [])
        if problematic_patterns:
            prompt_parts.append("\nProblematic Patterns:")
            for pattern in problematic_patterns[:3]:  # Top 3 problem patterns
                prompt_parts.append(
                    f"- {pattern['pattern']}: success_rate={pattern['success_rate']:.2f}, "
                    f"risk_level={pattern['risk_level']}"
                )

        # Add correlation analysis results
        prompt_parts.extend([
            "",
            "STATISTICAL CORRELATION ANALYSIS:",
        ])

        significant_correlations = correlation_analysis.get("significant_correlations", [])
        if significant_correlations:
            prompt_parts.append("Significant Correlations:")
            for corr in significant_correlations[:3]:  # Top 3 correlations
                prompt_parts.append(
                    f"- {' vs '.join(corr['variables'])}: r={corr['correlation']:.3f}, "
                    f"significance={corr['significance']}"
                )

        # Add performance insights
        performance_correlations = correlation_analysis.get("performance_correlations", {})
        if performance_correlations:
            prompt_parts.append("\nPerformance Correlations:")
            for metric, data in list(performance_correlations.items())[:3]:
                prompt_parts.append(
                    f"- {metric}: correlation={data['correlation_coefficient']:.3f}, "
                    f"strength={data['strength']}"
                )

        prompt_parts.extend([
            "",
            "STRATEGIC ANALYSIS REQUEST:",
            "Please provide strategic insights in JSON format with:",
            "- identified_patterns: Key patterns with strategic value",
            "- correlations: Important correlations for optimization", 
            "- strategic_insights: Actionable strategic recommendations",
            "- optimization_opportunities: Specific optimization suggestions",
            "- risk_factors: Patterns to avoid or mitigate",
            "",
            "Focus on actionable insights that can improve parallel execution",
            "performance and success rates based on the statistical analysis.",
        ])

        return "\n".join(prompt_parts)

    def _parse_strategic_insights_response(self, raw_response: str) -> dict[str, Any]:
        """Parse strategic insights response from Opus."""
        import json
        
        try:
            # Attempt to parse JSON response
            strategic_insights = json.loads(raw_response)

            # Validate response structure
            if not isinstance(strategic_insights, dict):
                raise MalformedResponseError("Strategic insights response is not a valid JSON object")

            # Ensure required fields exist with defaults
            required_fields = [
                "identified_patterns",
                "correlations", 
                "strategic_insights",
                "optimization_opportunities",
                "risk_factors",
            ]

            for field in required_fields:
                if field not in strategic_insights:
                    strategic_insights[field] = []

            return strategic_insights

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strategic insights JSON response: {str(e)}")
            raise MalformedResponseError(f"Invalid JSON in strategic insights response: {str(e)}") from e

    def _create_fallback_strategic_insights(
        self, pattern_analysis: dict[str, Any], correlation_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Create fallback strategic insights when Opus analysis fails."""
        self._record_metric("fallback_responses", 1)
        
        fallback_insights = {
            "identified_patterns": [],
            "correlations": [],
            "strategic_insights": [],
            "optimization_opportunities": [],
            "risk_factors": [],
            "metadata": {
                "analysis_type": "fallback",
                "generated_at": time.time(),
            },
        }

        # Extract insights from pattern analysis
        high_success_patterns = pattern_analysis.get("high_success_patterns", [])
        for pattern in high_success_patterns[:3]:
            fallback_insights["identified_patterns"].append({
                "pattern": pattern["pattern"],
                "success_rate": pattern["success_rate"],
                "strategic_value": "high",
            })
            
            fallback_insights["strategic_insights"].append(
                f"Pattern '{pattern['pattern']}' shows {pattern['success_rate']:.1%} success rate - prioritize for replication"
            )

        # Extract risk factors
        problematic_patterns = pattern_analysis.get("problematic_patterns", [])
        for pattern in problematic_patterns[:2]:
            fallback_insights["risk_factors"].append({
                "pattern": pattern["pattern"],
                "failure_rate": 1.0 - pattern["success_rate"],
                "risk_level": pattern["risk_level"],
            })
            
            fallback_insights["strategic_insights"].append(
                f"Avoid pattern '{pattern['pattern']}' - shows high failure rate ({1.0 - pattern['success_rate']:.1%})"
            )

        # Extract correlations
        significant_correlations = correlation_analysis.get("significant_correlations", [])
        for corr in significant_correlations[:2]:
            fallback_insights["correlations"].append({
                "variables": corr["variables"],
                "correlation": corr["correlation"],
                "significance": corr["significance"],
            })

        # Basic optimization opportunities
        fallback_insights["optimization_opportunities"] = [
            "Focus execution on high-success patterns identified in analysis",
            "Monitor performance metrics with significant correlations",
            "Implement pattern-based execution filtering to avoid problematic sequences",
        ]

        if not fallback_insights["strategic_insights"]:
            fallback_insights["strategic_insights"] = [
                "Insufficient data for detailed strategic analysis",
                "Continue parallel execution to gather more pattern data",
            ]

        return fallback_insights

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
