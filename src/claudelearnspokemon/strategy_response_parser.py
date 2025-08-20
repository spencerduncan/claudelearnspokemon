"""
Strategy Response Parser

Production-grade JSON parsing and validation for Opus strategy responses.
Implements timeout protection, size limits, and comprehensive error recovery.
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Any

from .opus_strategist_exceptions import (
    MalformedResponseError,
    ResponseTimeoutError,
    StrategyValidationError,
)
from .strategy_response import ExperimentSpec, StrategyResponse

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Configuration for response validation rules."""

    max_response_size: int = 10 * 1024 * 1024  # 10MB
    max_experiments: int = 50
    max_insights: int = 100
    max_checkpoints: int = 100
    required_fields: set[str] = field(default_factory=set)
    required_experiment_fields: set[str] = field(default_factory=set)

    def __post_init__(self):
        """Set default required fields."""
        if not self.required_fields:
            self.required_fields.update(
                {
                    "strategy_id",
                    "experiments",
                    "strategic_insights",
                    "next_checkpoints",
                }
            )

        if not self.required_experiment_fields:
            self.required_experiment_fields.update(
                {
                    "id",
                    "name",
                    "checkpoint",
                    "script_dsl",
                    "expected_outcome",
                    "priority",
                }
            )


class StrategyResponseParser:
    """
    Production-ready parser for Opus strategy responses.

    Handles JSON parsing, validation, size limits, and timeout protection.
    Designed for high availability with comprehensive error recovery.
    """

    def __init__(
        self,
        validation_timeout: float = 5.0,
        max_response_size: int = 10 * 1024 * 1024,
        validation_rules: ValidationRule | None = None,
    ):
        """
        Initialize parser with production configuration.

        Args:
            validation_timeout: Maximum time for parsing and validation (seconds)
            max_response_size: Maximum response size in bytes
            validation_rules: Custom validation configuration
        """
        self.validation_timeout = validation_timeout
        self.max_response_size = max_response_size
        self.validation_rules = validation_rules or ValidationRule()
        self.parser_metrics = {
            "total_parses": 0,
            "successful_parses": 0,
            "validation_failures": 0,
            "timeout_failures": 0,
            "size_limit_failures": 0,
            "malformed_json_failures": 0,
        }
        self._metrics_lock = threading.Lock()

        logger.info(f"StrategyResponseParser initialized with {validation_timeout}s timeout")

    def parse_response(self, raw_response: str) -> StrategyResponse:
        """
        Parse and validate raw strategy response from Opus.

        Args:
            raw_response: Raw JSON string from Claude Opus

        Returns:
            Validated StrategyResponse object

        Raises:
            MalformedResponseError: Invalid JSON or size exceeded
            StrategyValidationError: Response structure invalid
            ResponseTimeoutError: Processing timeout exceeded
        """
        start_time = time.time()

        with self._metrics_lock:
            self.parser_metrics["total_parses"] += 1

        try:
            # Size validation first (fail fast)
            if len(raw_response.encode("utf-8")) > self.max_response_size:
                self._record_metric("size_limit_failures")
                raise MalformedResponseError(
                    f"Response exceeds maximum size limit ({self.max_response_size} bytes)",
                    raw_response[:1000] + "... [truncated]",
                )

            # Parse with timeout protection
            parsed_data = self._parse_json_with_timeout(raw_response)

            # Validate response structure
            self._validate_response_structure(parsed_data)

            # Convert to StrategyResponse object
            strategy_response = self._convert_to_strategy_response(parsed_data)

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Successfully parsed strategy response in {processing_time:.2f}ms")

            with self._metrics_lock:
                self.parser_metrics["successful_parses"] += 1

            return strategy_response

        except (StrategyValidationError, MalformedResponseError, ResponseTimeoutError):
            # Re-raise known exceptions
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            self._record_metric("malformed_json_failures")
            logger.error(f"Unexpected parsing error: {str(e)}")
            raise MalformedResponseError(
                f"Unexpected parsing error: {str(e)}", raw_response[:500]
            ) from e

    def _parse_json_with_timeout(self, raw_response: str) -> dict[str, Any]:
        """Parse JSON with timeout protection."""

        def parse_worker():
            """Worker function for timeout-protected JSON parsing."""
            try:
                return json.loads(raw_response)
            except json.JSONDecodeError as e:
                raise MalformedResponseError(
                    f"Invalid JSON format: {str(e)}", raw_response[:500]
                ) from e

        # Use ThreadPoolExecutor for timeout protection
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(parse_worker)

            try:
                result = future.result(timeout=self.validation_timeout)
                return result
            except TimeoutError:
                self._record_metric("timeout_failures")
                raise ResponseTimeoutError(
                    f"JSON parsing timeout after {self.validation_timeout}s",
                    self.validation_timeout,
                ) from None

    def _validate_response_structure(self, data: dict[str, Any]) -> None:
        """Validate response structure against schema."""
        validation_errors = []

        # Check required top-level fields
        for field_name in self.validation_rules.required_fields:
            if field_name not in data:
                validation_errors.append(f"Missing required field: {field_name}")

        if validation_errors:
            self._record_metric("validation_failures")
            raise StrategyValidationError(
                f"Response validation failed: {'; '.join(validation_errors)}", validation_errors
            )

        # Validate field types and constraints
        self._validate_field_types(data, validation_errors)
        self._validate_experiments(data.get("experiments", []), validation_errors)
        self._validate_size_constraints(data, validation_errors)

        if validation_errors:
            self._record_metric("validation_failures")
            raise StrategyValidationError(
                f"Response validation failed: {'; '.join(validation_errors)}", validation_errors
            )

    def _validate_field_types(self, data: dict[str, Any], validation_errors: list[str]) -> None:
        """Validate field types."""
        if not isinstance(data.get("strategy_id"), str) or not data.get("strategy_id"):
            validation_errors.append("strategy_id must be non-empty string")

        if not isinstance(data.get("experiments"), list):
            validation_errors.append("experiments must be a list")

        if not isinstance(data.get("strategic_insights"), list):
            validation_errors.append("strategic_insights must be a list")

        if not isinstance(data.get("next_checkpoints"), list):
            validation_errors.append("next_checkpoints must be a list")

    def _validate_experiments(self, experiments: list[Any], validation_errors: list[str]) -> None:
        """Validate experiment structure."""
        for i, exp in enumerate(experiments):
            if not isinstance(exp, dict):
                validation_errors.append(f"Experiment {i} must be an object")
                continue

            # Check required experiment fields
            for field_name in self.validation_rules.required_experiment_fields:
                if field_name not in exp:
                    validation_errors.append(f"Experiment {i} missing required field: {field_name}")

            # Validate priority is positive integer
            if "priority" in exp:
                if not isinstance(exp["priority"], int) or exp["priority"] < 1:
                    validation_errors.append(f"Experiment {i} priority must be positive integer")

    def _validate_size_constraints(
        self, data: dict[str, Any], validation_errors: list[str]
    ) -> None:
        """Validate size constraints."""
        experiments = data.get("experiments", [])
        if len(experiments) > self.validation_rules.max_experiments:
            validation_errors.append(
                f"Too many experiments: {len(experiments)} > {self.validation_rules.max_experiments}"
            )

        insights = data.get("strategic_insights", [])
        if len(insights) > self.validation_rules.max_insights:
            validation_errors.append(
                f"Too many insights: {len(insights)} > {self.validation_rules.max_insights}"
            )

        checkpoints = data.get("next_checkpoints", [])
        if len(checkpoints) > self.validation_rules.max_checkpoints:
            validation_errors.append(
                f"Too many checkpoints: {len(checkpoints)} > {self.validation_rules.max_checkpoints}"
            )

    def _convert_to_strategy_response(self, data: dict[str, Any]) -> StrategyResponse:
        """Convert validated dictionary to StrategyResponse object."""
        try:
            experiments = []
            for exp_data in data["experiments"]:
                experiment = ExperimentSpec(
                    id=exp_data["id"],
                    name=exp_data["name"],
                    checkpoint=exp_data["checkpoint"],
                    script_dsl=exp_data["script_dsl"],
                    expected_outcome=exp_data["expected_outcome"],
                    priority=exp_data["priority"],
                    directives=exp_data.get("directives", []),
                    metadata=exp_data.get("metadata", {}),
                )
                experiments.append(experiment)

            return StrategyResponse(
                strategy_id=data["strategy_id"],
                experiments=experiments,
                strategic_insights=data["strategic_insights"],
                next_checkpoints=data["next_checkpoints"],
                metadata=data.get("metadata", {}),
            )

        except (ValueError, KeyError) as e:
            raise StrategyValidationError(f"Failed to construct StrategyResponse: {str(e)}") from e

    def _record_metric(self, metric_name: str) -> None:
        """Thread-safe metric recording."""
        with self._metrics_lock:
            self.parser_metrics[metric_name] += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get parser performance metrics."""
        with self._metrics_lock:
            base_metrics = self.parser_metrics.copy()

        # Create new metrics dict with calculated rates
        metrics: dict[str, Any] = dict(base_metrics)

        # Calculate derived metrics
        total = base_metrics["total_parses"]
        if total > 0:
            metrics["success_rate"] = base_metrics["successful_parses"] / total
            metrics["validation_failure_rate"] = base_metrics["validation_failures"] / total
            metrics["timeout_rate"] = base_metrics["timeout_failures"] / total
        else:
            metrics["success_rate"] = 0.0
            metrics["validation_failure_rate"] = 0.0
            metrics["timeout_rate"] = 0.0

        return metrics

    def reset_metrics(self) -> None:
        """Reset parser metrics (useful for testing)."""
        with self._metrics_lock:
            for key in self.parser_metrics:
                self.parser_metrics[key] = 0
