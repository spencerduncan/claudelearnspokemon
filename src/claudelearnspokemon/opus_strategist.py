"""OpusStrategist: Strategic planning and experiment generation using Claude Opus."""

import json
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .claude_code_manager import ClaudeCodeManager


@dataclass
class ExperimentSpec:
    """Specification for a single experiment."""

    id: str
    name: str
    description: str
    strategy: str
    parameters: dict[str, Any]
    priority: float
    estimated_duration: float
    dependencies: list[str]
    metadata: dict[str, Any]


@dataclass
class ExperimentVariation:
    """A variation of a base experiment."""

    base_experiment_id: str
    variation_type: str
    parameters: dict[str, Any]
    description: str
    expected_outcome: str


class OpusStrategist:
    """Strategic planning and experiment generation using Claude Opus.

    This class encapsulates Claude Opus conversation for strategic planning,
    analyzing game progression, synthesizing parallel results, and generating
    diverse experiment specifications for parallel execution.
    """

    def __init__(self, claude_code_manager: ClaudeCodeManager):
        """Initialize OpusStrategist with Claude Code manager.

        Args:
            claude_code_manager: ClaudeCodeManager instance for Opus communication
        """
        self.claude_manager = claude_code_manager
        self.logger = logging.getLogger(__name__)

        # Pattern analysis cache (from Issue #100)
        self._pattern_cache: dict[str, Any] = {}
        self._correlation_cache: dict[str, float] = {}

        # Experiment generation state
        self._experiment_counter = 0
        self._experiment_history: list[dict[str, Any]] = []
        self._validation_rules = self._initialize_validation_rules()

    def request_strategy(
        self, game_state: dict[str, Any], recent_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Request strategic direction from Opus.

        Args:
            game_state: Current game state information
            recent_results: Recent execution results for context

        Returns:
            Strategic response with experiment specifications
        """
        strategy_prompt = self._build_strategy_prompt(game_state, recent_results)

        try:
            strategic_process = self.claude_manager.get_strategic_process()
            if strategic_process is None:
                raise RuntimeError("No strategic process available")
            response = strategic_process.send_message(strategy_prompt)
            if response is None:
                raise RuntimeError("No response received from strategic process")
            strategy_data = self._parse_strategic_response(response)

            self.logger.info(
                f"Received strategy with {len(strategy_data.get('experiments', []))} experiments"
            )
            return strategy_data

        except Exception as e:
            self.logger.error(f"Failed to get strategic response: {e}")
            return {"experiments": [], "analysis": "", "recommendations": []}

    def analyze_parallel_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Analyze results from parallel trials (Issue #100 functionality).

        Args:
            results: List of execution results from parallel trials

        Returns:
            List of analyzed patterns and correlations
        """
        start_time = time.time()

        if not results:
            return []

        # Extract metrics from results
        metrics = self._extract_metrics(results)

        # Calculate correlations between different approaches
        correlations = self._calculate_correlations(metrics)

        # Identify patterns across results
        patterns = self._identify_patterns(results)

        # Compress results for efficiency (95% size reduction target)
        compressed_analysis = self._compress_analysis(correlations, patterns)

        processing_time = (time.time() - start_time) * 1000
        self.logger.info(f"Parallel results analysis completed in {processing_time:.1f}ms")

        return compressed_analysis

    def extract_experiments_from_response(
        self, strategic_response: dict[str, Any]
    ) -> list[ExperimentSpec]:
        """Extract parallel experiments from Opus strategic responses.

        Args:
            strategic_response: Response from request_strategy

        Returns:
            List of experiment specifications
        """
        start_time = time.time()

        experiments = []
        raw_experiments = strategic_response.get("experiments", [])

        for i, exp_data in enumerate(raw_experiments):
            try:
                experiment = self._parse_experiment_data(exp_data, i)
                if self._validate_experiment_spec(experiment):
                    experiments.append(experiment)
                else:
                    self.logger.warning(f"Experiment {experiment.id} failed validation")
            except Exception as e:
                self.logger.error(f"Failed to parse experiment {i}: {e}")

        processing_time = (time.time() - start_time) * 1000
        self.logger.info(f"Extracted {len(experiments)} experiments in {processing_time:.1f}ms")

        return experiments

    def generate_experiment_variations(
        self, base_experiment: ExperimentSpec, count: int = 3
    ) -> list[ExperimentSpec]:
        """Generate diverse experiment variations based on strategy.

        Args:
            base_experiment: Base experiment to generate variations from
            count: Number of variations to generate

        Returns:
            List of experiment variations
        """
        start_time = time.time()

        variations = []

        # Generate parameter variations
        param_variations = self._generate_parameter_variations(base_experiment, count // 2 + 1)
        variations.extend(param_variations)

        # Generate strategy variations
        strategy_variations = self._generate_strategy_variations(
            base_experiment, count - len(param_variations)
        )
        variations.extend(strategy_variations)

        processing_time = (time.time() - start_time) * 1000
        self.logger.info(f"Generated {len(variations)} variations in {processing_time:.1f}ms")

        return variations[:count]

    def validate_experiment_executability(
        self, experiment: ExperimentSpec
    ) -> tuple[bool, list[str]]:
        """Validate experiment specifications for executability.

        Args:
            experiment: Experiment specification to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        start_time = time.time()

        issues = []

        # Apply validation rules
        for rule_name, rule_func in self._validation_rules.items():
            try:
                if not rule_func(experiment):  # type: ignore
                    issues.append(f"Failed {rule_name} validation")
            except Exception as e:
                issues.append(f"Validation error in {rule_name}: {str(e)}")

        is_valid = len(issues) == 0

        processing_time = (time.time() - start_time) * 1000
        if processing_time > 10:  # Log if slower than 10ms target
            self.logger.warning(f"Validation took {processing_time:.1f}ms (target <10ms)")

        return is_valid, issues

    def create_experiment_metadata(
        self, experiment: ExperimentSpec, additional_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create experiment metadata for tracking and analysis.

        Args:
            experiment: Experiment specification
            additional_data: Additional metadata to include

        Returns:
            Comprehensive experiment metadata
        """
        metadata = {
            "id": experiment.id,
            "name": experiment.name,
            "created_at": time.time(),
            "strategy": experiment.strategy,
            "priority": experiment.priority,
            "estimated_duration": experiment.estimated_duration,
            "parameters": experiment.parameters.copy(),
            "dependencies": experiment.dependencies.copy(),
            "validation_status": "pending",
            "execution_status": "not_started",
            "tags": self._generate_experiment_tags(experiment),
            "complexity_score": self._calculate_complexity_score(experiment),
            "diversity_score": self._calculate_diversity_score(experiment),
        }

        if additional_data:
            metadata.update(additional_data)

        # Store in experiment history
        self._experiment_history.append(metadata)

        return metadata

    # Private helper methods

    def _build_strategy_prompt(
        self, game_state: dict[str, Any], recent_results: list[dict[str, Any]]
    ) -> str:
        """Build strategic planning prompt for Opus."""
        prompt = f"""
        Strategic Planning Request:

        Game State: {json.dumps(game_state, indent=2)}
        Recent Results: {json.dumps(recent_results[-3:], indent=2)}

        Please provide strategic analysis and generate 3-5 diverse experiment specifications.

        Response format:
        {{
            "analysis": "strategic analysis of current situation",
            "experiments": [
                {{
                    "name": "experiment name",
                    "description": "detailed description",
                    "strategy": "approach/strategy type",
                    "parameters": {{"param1": "value1"}},
                    "priority": 0.8,
                    "estimated_duration": 120.0,
                    "dependencies": []
                }}
            ],
            "recommendations": ["recommendation 1", "recommendation 2"]
        }}
        """

        return prompt

    def _parse_strategic_response(self, response: str) -> dict[str, Any]:
        """Parse strategic response from Opus."""
        try:
            # Extract JSON from response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse strategic response: {e}")
            return {"experiments": [], "analysis": "", "recommendations": []}

    def _parse_experiment_data(self, exp_data: dict[str, Any], index: int) -> ExperimentSpec:
        """Parse experiment data into ExperimentSpec."""
        self._experiment_counter += 1

        return ExperimentSpec(
            id=f"exp_{self._experiment_counter:04d}",
            name=exp_data.get("name", f"Experiment {index + 1}"),
            description=exp_data.get("description", ""),
            strategy=exp_data.get("strategy", "general"),
            parameters=exp_data.get("parameters", {}),
            priority=float(exp_data.get("priority", 0.5)),
            estimated_duration=float(exp_data.get("estimated_duration", 60.0)),
            dependencies=exp_data.get("dependencies", []),
            metadata={},
        )

    def _validate_experiment_spec(self, experiment: ExperimentSpec) -> bool:
        """Basic validation of experiment specification."""
        is_valid, issues = self.validate_experiment_executability(experiment)
        return is_valid

    def _generate_parameter_variations(
        self, base_experiment: ExperimentSpec, count: int
    ) -> list[ExperimentSpec]:
        """Generate parameter-based variations of experiment."""
        variations = []

        for i in range(count):
            # Create variation with modified parameters
            new_params = base_experiment.parameters.copy()

            # Modify numeric parameters by small amounts
            for param, value in new_params.items():
                if isinstance(value, int | float):
                    variation_factor = 0.8 + (i * 0.1)  # 0.8, 0.9, 1.0, 1.1, 1.2
                    new_params[param] = value * variation_factor

            variation = ExperimentSpec(
                id=f"{base_experiment.id}_var_{i+1}",
                name=f"{base_experiment.name} (Param Var {i+1})",
                description=f"Parameter variation of {base_experiment.name}",
                strategy=base_experiment.strategy,
                parameters=new_params,
                priority=base_experiment.priority * 0.9,  # Slightly lower priority
                estimated_duration=base_experiment.estimated_duration,
                dependencies=base_experiment.dependencies.copy(),
                metadata={"variation_type": "parameter", "base_experiment": base_experiment.id},
            )

            variations.append(variation)

        return variations

    def _generate_strategy_variations(
        self, base_experiment: ExperimentSpec, count: int
    ) -> list[ExperimentSpec]:
        """Generate strategy-based variations of experiment."""
        variations = []
        strategy_types = ["aggressive", "conservative", "exploratory", "optimized", "hybrid"]

        for i in range(min(count, len(strategy_types))):
            new_strategy = strategy_types[i]
            if new_strategy == base_experiment.strategy:
                continue

            variation = ExperimentSpec(
                id=f"{base_experiment.id}_strat_{i+1}",
                name=f"{base_experiment.name} ({new_strategy.title()})",
                description=f"Strategy variation using {new_strategy} approach",
                strategy=new_strategy,
                parameters=base_experiment.parameters.copy(),
                priority=base_experiment.priority * 0.85,  # Lower priority for variations
                estimated_duration=base_experiment.estimated_duration * 1.1,  # Slightly longer
                dependencies=base_experiment.dependencies.copy(),
                metadata={"variation_type": "strategy", "base_experiment": base_experiment.id},
            )

            variations.append(variation)

        return variations

    def _initialize_validation_rules(self) -> dict[str, Callable[[ExperimentSpec], bool]]:
        """Initialize experiment validation rules."""
        return {
            "has_name": lambda exp: bool(exp.name and exp.name.strip()),
            "has_description": lambda exp: bool(exp.description and exp.description.strip()),
            "valid_priority": lambda exp: 0.0 <= exp.priority <= 1.0,
            "positive_duration": lambda exp: exp.estimated_duration > 0,
            "valid_parameters": lambda exp: isinstance(exp.parameters, dict),
            "valid_dependencies": lambda exp: isinstance(exp.dependencies, list),
            "reasonable_duration": lambda exp: exp.estimated_duration < 3600,  # Less than 1 hour
        }

    def _generate_experiment_tags(self, experiment: ExperimentSpec) -> list[str]:
        """Generate tags for experiment categorization."""
        tags = [experiment.strategy]

        if experiment.priority > 0.8:
            tags.append("high_priority")
        elif experiment.priority < 0.3:
            tags.append("low_priority")

        if experiment.estimated_duration > 300:  # 5 minutes
            tags.append("long_running")
        else:
            tags.append("quick")

        if len(experiment.dependencies) > 0:
            tags.append("has_dependencies")

        return tags

    def _calculate_complexity_score(self, experiment: ExperimentSpec) -> float:
        """Calculate complexity score for experiment."""
        complexity = 0.0

        # Parameter complexity
        complexity += len(experiment.parameters) * 0.1

        # Dependency complexity
        complexity += len(experiment.dependencies) * 0.2

        # Duration complexity
        if experiment.estimated_duration > 300:
            complexity += 0.3

        return min(complexity, 1.0)

    def _calculate_diversity_score(self, experiment: ExperimentSpec) -> float:
        """Calculate diversity score compared to experiment history."""
        if not self._experiment_history:
            return 1.0  # First experiment is maximally diverse

        # Compare strategy diversity
        strategies = [exp.get("strategy", "") for exp in self._experiment_history]
        strategy_diversity = 1.0 if experiment.strategy not in strategies else 0.5

        # Parameter diversity (simplified)
        param_diversity = len(experiment.parameters) / 10.0  # Normalize to 0-1

        return (strategy_diversity + param_diversity) / 2.0

    # Issue #100 functionality (parallel results analysis)

    def _extract_metrics(self, results: list[dict[str, Any]]) -> dict[str, list[float]]:
        """Extract numerical metrics from execution results."""
        metrics = defaultdict(list)

        for result in results:
            if "performance" in result:
                perf = result["performance"]
                for metric_name, value in perf.items():
                    if isinstance(value, int | float):
                        metrics[metric_name].append(float(value))

        return dict(metrics)

    def _calculate_correlations(self, metrics: dict[str, list[float]]) -> dict[str, float]:
        """Calculate Pearson correlations between metrics."""
        correlations = {}

        metric_names = list(metrics.keys())
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i + 1 :]:
                if len(metrics[metric1]) >= 2 and len(metrics[metric2]) >= 2:
                    try:
                        correlation = statistics.correlation(metrics[metric1], metrics[metric2])
                        correlations[f"{metric1}_vs_{metric2}"] = correlation
                    except statistics.StatisticsError:
                        pass  # Skip if correlation can't be calculated

        return correlations

    def _identify_patterns(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Identify patterns across parallel results."""
        patterns = {
            "success_rate": 0.0,
            "average_score": 0.0,
            "common_strategies": [],
            "failure_modes": [],
        }

        if not results:
            return patterns

        # Calculate success rate
        successful = [r for r in results if r.get("success", False)]
        patterns["success_rate"] = len(successful) / len(results)

        # Calculate average score
        scores = [r.get("score", 0) for r in results if "score" in r]
        if scores:
            patterns["average_score"] = statistics.mean(scores)

        # Identify common strategies
        strategies = [r.get("strategy", "unknown") for r in results]
        strategy_counts: dict[str, int] = defaultdict(int)
        for strategy in strategies:
            strategy_counts[strategy] += 1

        patterns["common_strategies"] = sorted(
            strategy_counts.items(), key=lambda x: x[1], reverse=True
        )[
            :3
        ]  # Top 3 strategies

        return patterns

    def _compress_analysis(
        self, correlations: dict[str, float], patterns: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Compress analysis results for efficient storage (95% size reduction target)."""
        compressed = []

        # Only keep significant correlations
        significant_correlations = {
            k: v
            for k, v in correlations.items()
            if abs(v) > 0.3  # Only correlations with magnitude > 0.3
        }

        if significant_correlations:
            compressed.append({"type": "correlations", "data": significant_correlations})

        # Compress patterns
        key_patterns = {
            "success_rate": patterns.get("success_rate", 0.0),
            "average_score": patterns.get("average_score", 0.0),
            "top_strategy": patterns.get("common_strategies", [("unknown", 0)])[0][0],
        }

        compressed.append({"type": "patterns", "data": key_patterns})

        return compressed
