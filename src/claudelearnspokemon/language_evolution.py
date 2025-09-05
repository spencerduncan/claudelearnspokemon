"""
Language Evolution System for OpusStrategist

Implements Clean Code architecture for DSL improvement proposals based on
empirical pattern analysis. Follows SOLID principles throughout:

- Single Responsibility: Each class has one clear purpose
- Open/Closed: Extensible for new evolution strategies
- Liskov Substitution: Implementations are substitutable
- Interface Segregation: Clean interfaces for different concerns
- Dependency Inversion: Depend on abstractions, not concretions

Performance Requirements:
- Pattern analysis: <200ms
- Proposal generation: <100ms
- Validation: <50ms

Architecture:
- LanguageAnalyzer: Analyzes patterns for evolution opportunities
- EvolutionProposalGenerator: Generates DSL improvement proposals
- LanguageValidator: Validates language consistency
- OpusStrategist integration: Main propose_language_evolution interface
"""

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict


class ContextData(TypedDict):
    """Type definition for context analysis data."""

    success_rates: list[float]
    frequencies: list[int]
    pattern_names: list[str]


class EvolutionOpportunityType(Enum):
    """Types of language evolution opportunities."""

    COMMON_SEQUENCE = "common_sequence"
    CONTEXT_DEPENDENT = "context_dependent"
    LOW_SUCCESS_PATTERN = "low_success_pattern"
    REPETITIVE_PATTERN = "repetitive_pattern"
    CONDITIONAL_ENHANCEMENT = "conditional_enhancement"


class ProposalType(Enum):
    """Types of DSL evolution proposals."""

    MACRO_EXTENSION = "macro_extension"
    PATTERN_OPTIMIZATION = "pattern_optimization"
    SYNTAX_IMPROVEMENT = "syntax_improvement"
    CONDITIONAL_DSL = "conditional_dsl"


class ImplementationComplexity(Enum):
    """Implementation complexity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class EvolutionOpportunity:
    """
    Immutable opportunity for language evolution.

    Represents a specific pattern or group of patterns that could
    benefit from language evolution based on empirical analysis.
    """

    opportunity_id: str
    opportunity_type: EvolutionOpportunityType
    pattern_names: list[str]
    common_sequence: list[str] | None = None
    frequency: int = 0
    average_success_rate: float = 0.0
    improvement_potential: float = 0.0
    context_dependencies: dict[str, Any] | None = None
    priority_score: float = 0.0

    def __post_init__(self):
        """Validate opportunity data."""
        if not self.opportunity_id:
            raise ValueError("Opportunity ID cannot be empty")
        if not self.pattern_names:
            raise ValueError("Pattern names cannot be empty")
        if not 0.0 <= self.average_success_rate <= 1.0:
            raise ValueError("Average success rate must be between 0.0 and 1.0")
        if not 0.0 <= self.improvement_potential <= 1.0:
            raise ValueError("Improvement potential must be between 0.0 and 1.0")


@dataclass(frozen=True)
class EvolutionProposal:
    """
    Immutable DSL evolution proposal.

    Contains specific proposed changes to the DSL with expected
    improvements and implementation details.
    """

    proposal_id: str
    proposal_type: ProposalType
    opportunity_basis: EvolutionOpportunity
    dsl_changes: dict[str, Any]
    expected_improvement: dict[str, float]
    validation_score: float
    implementation_complexity: ImplementationComplexity
    estimated_development_time: float = 0.0  # Hours
    risk_assessment: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate proposal data."""
        if not self.proposal_id:
            raise ValueError("Proposal ID cannot be empty")
        if not self.dsl_changes:
            raise ValueError("DSL changes cannot be empty")
        if not 0.0 <= self.validation_score <= 1.0:
            raise ValueError("Validation score must be between 0.0 and 1.0")


class PatternAnalysisStrategy(ABC):
    """
    Abstract base for pattern analysis strategies.

    Follows Strategy pattern for different analysis approaches.
    Enables extensibility for new analysis methods (Open/Closed Principle).
    """

    @abstractmethod
    def analyze_patterns(self, patterns: list[dict[str, Any]]) -> list[EvolutionOpportunity]:
        """
        Analyze patterns for evolution opportunities.

        Args:
            patterns: List of pattern data

        Returns:
            List of identified opportunities
        """
        pass


class CommonSequenceAnalysisStrategy(PatternAnalysisStrategy):
    """
    Analyzes patterns for common input sequences that could be abstracted.

    Implements Single Responsibility: Only handles common sequence detection.
    """

    def __init__(self, min_frequency: int = 3, min_sequence_length: int = 2):
        self.min_frequency = min_frequency
        self.min_sequence_length = min_sequence_length

    def analyze_patterns(self, patterns: list[dict[str, Any]]) -> list[EvolutionOpportunity]:
        """Find common sequences across patterns."""
        sequence_frequency = defaultdict(list)

        # Extract sequences from patterns
        for pattern in patterns:
            input_sequence = pattern.get("input_sequence", [])
            pattern_name = pattern.get("name", "unknown")

            # Generate all subsequences of minimum length
            for i in range(len(input_sequence)):
                for j in range(i + self.min_sequence_length, len(input_sequence) + 1):
                    subsequence = tuple(input_sequence[i:j])
                    sequence_frequency[subsequence].append(
                        {
                            "pattern_name": pattern_name,
                            "success_rate": pattern.get("success_rate", 0.0),
                            "usage_frequency": pattern.get("usage_frequency", 0),
                        }
                    )

        # Identify common sequences above frequency threshold
        opportunities = []
        for sequence, pattern_occurrences in sequence_frequency.items():
            if len(pattern_occurrences) >= self.min_frequency:
                pattern_names = [occ["pattern_name"] for occ in pattern_occurrences]
                avg_success_rate = sum(occ["success_rate"] for occ in pattern_occurrences) / len(
                    pattern_occurrences
                )
                total_frequency = sum(occ["usage_frequency"] for occ in pattern_occurrences)

                # Calculate improvement potential based on frequency and current success
                improvement_potential = min(0.3, (1.0 - avg_success_rate) * 0.5)
                priority_score = total_frequency * improvement_potential

                opportunity = EvolutionOpportunity(
                    opportunity_id=f"common_seq_{hash(sequence) % 10000}",
                    opportunity_type=EvolutionOpportunityType.COMMON_SEQUENCE,
                    pattern_names=pattern_names,
                    common_sequence=list(sequence),
                    frequency=total_frequency,
                    average_success_rate=avg_success_rate,
                    improvement_potential=improvement_potential,
                    priority_score=priority_score,
                )
                opportunities.append(opportunity)

        # Sort by priority score (highest first)
        return sorted(opportunities, key=lambda x: x.priority_score, reverse=True)


class LowSuccessPatternAnalysisStrategy(PatternAnalysisStrategy):
    """
    Identifies patterns with low success rates for optimization.

    Single Responsibility: Focus on underperforming patterns only.
    """

    def __init__(self, success_threshold: float = 0.7, min_usage: int = 10):
        self.success_threshold = success_threshold
        self.min_usage = min_usage

    def analyze_patterns(self, patterns: list[dict[str, Any]]) -> list[EvolutionOpportunity]:
        """Find patterns with low success rates that need optimization."""
        opportunities = []

        for pattern in patterns:
            success_rate = pattern.get("success_rate", 0.0)
            usage_frequency = pattern.get("usage_frequency", 0)
            pattern_name = pattern.get("name", "unknown")

            # Check if pattern meets criteria for optimization
            if success_rate < self.success_threshold and usage_frequency >= self.min_usage:

                # Higher improvement potential for lower success rates
                improvement_potential = min(0.5, (self.success_threshold - success_rate))
                priority_score = usage_frequency * improvement_potential

                opportunity = EvolutionOpportunity(
                    opportunity_id=f"low_success_{pattern_name}",
                    opportunity_type=EvolutionOpportunityType.LOW_SUCCESS_PATTERN,
                    pattern_names=[pattern_name],
                    frequency=usage_frequency,
                    average_success_rate=success_rate,
                    improvement_potential=improvement_potential,
                    priority_score=priority_score,
                    context_dependencies=pattern.get("context", {}),
                )
                opportunities.append(opportunity)

        return sorted(opportunities, key=lambda x: x.priority_score, reverse=True)


class ContextDependentAnalysisStrategy(PatternAnalysisStrategy):
    """
    Identifies context-dependent patterns for conditional DSL features.

    Single Responsibility: Focus on context analysis only.
    """

    def __init__(self, context_variance_threshold: float = 0.1):
        self.context_variance_threshold = context_variance_threshold

    def analyze_patterns(self, patterns: list[dict[str, Any]]) -> list[EvolutionOpportunity]:
        """Find patterns with context-dependent success rates."""
        # Group patterns by base sequence
        sequence_groups = defaultdict(list)

        for pattern in patterns:
            input_sequence = tuple(pattern.get("input_sequence", []))
            sequence_groups[input_sequence].append(pattern)

        opportunities = []

        for sequence, pattern_group in sequence_groups.items():
            if len(pattern_group) < 2:
                continue  # Need multiple contexts for comparison

            # Analyze context variance in success rates
            contexts: dict[str, ContextData] = {}
            for pattern in pattern_group:
                context_key = self._extract_context_key(pattern.get("context", {}))
                if context_key not in contexts:
                    contexts[context_key] = {
                        "success_rates": [],
                        "frequencies": [],
                        "pattern_names": [],
                    }
                contexts[context_key]["success_rates"].append(pattern.get("success_rate", 0.0))
                contexts[context_key]["frequencies"].append(pattern.get("usage_frequency", 0))
                contexts[context_key]["pattern_names"].append(pattern.get("name", "unknown"))

            # Calculate variance in success rates across contexts
            if len(contexts) > 1:
                context_success_rates = []
                for context_data in contexts.values():
                    avg_success = sum(context_data["success_rates"]) / len(
                        context_data["success_rates"]
                    )
                    context_success_rates.append(avg_success)

                variance = self._calculate_variance(context_success_rates)

                if variance >= self.context_variance_threshold:
                    pattern_names: list[str] = []
                    total_frequency = 0
                    for context_data in contexts.values():
                        pattern_names.extend(context_data["pattern_names"])
                        total_frequency += sum(context_data["frequencies"])

                    avg_success = sum(context_success_rates) / len(context_success_rates)
                    improvement_potential = min(
                        0.4, variance * 0.8
                    )  # Variance indicates improvement opportunity
                    priority_score = total_frequency * improvement_potential

                    opportunity = EvolutionOpportunity(
                        opportunity_id=f"context_dep_{hash(sequence) % 10000}",
                        opportunity_type=EvolutionOpportunityType.CONTEXT_DEPENDENT,
                        pattern_names=list(set(pattern_names)),
                        common_sequence=list(sequence),
                        frequency=total_frequency,
                        average_success_rate=avg_success,
                        improvement_potential=improvement_potential,
                        priority_score=priority_score,
                        context_dependencies=dict(contexts),
                    )
                    opportunities.append(opportunity)

        return sorted(opportunities, key=lambda x: x.priority_score, reverse=True)

    def _extract_context_key(self, context: dict[str, Any]) -> str:
        """Extract key context attributes for grouping."""
        key_attributes = ["location", "menu_type", "battle_state", "health_status"]
        context_parts = []

        for attr in key_attributes:
            if attr in context:
                context_parts.append(f"{attr}:{context[attr]}")

        return "|".join(context_parts) if context_parts else "default"

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)

        # For small sample sizes, also consider the range as an indicator
        value_range = max(values) - min(values)

        # Use the maximum of variance and scaled range to better detect context differences
        return max(variance, value_range * 0.5)


class LanguageAnalyzer:
    """
    Analyzes patterns for language evolution opportunities.

    Implements Single Responsibility Principle: Only handles pattern analysis.
    Uses Strategy pattern for different analysis approaches (Open/Closed).
    """

    def __init__(self, strategies: list[PatternAnalysisStrategy] | None = None):
        """Initialize with analysis strategies."""
        if strategies is None:
            # Default strategies
            self.strategies = [
                CommonSequenceAnalysisStrategy(),
                LowSuccessPatternAnalysisStrategy(),
                ContextDependentAnalysisStrategy(),
            ]
        else:
            self.strategies = strategies

    def identify_evolution_opportunities(
        self, patterns: list[dict[str, Any]]
    ) -> list[EvolutionOpportunity]:
        """
        Identify language evolution opportunities from pattern analysis.

        Performance target: <200ms for typical pattern sets.

        Args:
            patterns: List of pattern data for analysis

        Returns:
            List of evolution opportunities sorted by priority

        Raises:
            PerformanceError: If analysis exceeds 200ms target
        """
        start_time = time.perf_counter()

        try:
            all_opportunities = []

            # Apply all strategies to find different types of opportunities
            for strategy in self.strategies:
                opportunities = strategy.analyze_patterns(patterns)
                all_opportunities.extend(opportunities)

            # Remove duplicates and merge similar opportunities
            unique_opportunities = self._merge_similar_opportunities(all_opportunities)

            # Sort by priority score
            sorted_opportunities = sorted(
                unique_opportunities, key=lambda x: x.priority_score, reverse=True
            )

            # Validate performance target
            analysis_time = (time.perf_counter() - start_time) * 1000
            if analysis_time > 200.0:
                raise PerformanceError(
                    f"Pattern analysis took {analysis_time:.2f}ms, target is <200ms"
                )

            return sorted_opportunities

        except Exception as e:
            analysis_time = (time.perf_counter() - start_time) * 1000
            raise AnalysisError(f"Pattern analysis failed after {analysis_time:.2f}ms: {e}") from e

    def _merge_similar_opportunities(
        self, opportunities: list[EvolutionOpportunity]
    ) -> list[EvolutionOpportunity]:
        """Merge similar opportunities to avoid duplicates."""
        # Group by pattern overlap
        merged = []
        used_opportunities = set()

        for i, opp1 in enumerate(opportunities):
            if i in used_opportunities:
                continue

            similar_group = [opp1]
            used_opportunities.add(i)

            # Find similar opportunities (same patterns or high overlap)
            for j, opp2 in enumerate(opportunities[i + 1 :], start=i + 1):
                if j in used_opportunities:
                    continue

                pattern_overlap = len(set(opp1.pattern_names) & set(opp2.pattern_names))
                if pattern_overlap > 0 and pattern_overlap >= len(opp1.pattern_names) * 0.5:
                    similar_group.append(opp2)
                    used_opportunities.add(j)

            # If group has multiple opportunities, merge them
            if len(similar_group) > 1:
                merged_opportunity = self._merge_opportunity_group(similar_group)
                merged.append(merged_opportunity)
            else:
                merged.append(opp1)

        return merged

    def _merge_opportunity_group(
        self, opportunities: list[EvolutionOpportunity]
    ) -> EvolutionOpportunity:
        """Merge a group of similar opportunities."""
        # Combine pattern names
        all_pattern_names = []
        for opp in opportunities:
            all_pattern_names.extend(opp.pattern_names)
        unique_pattern_names = list(set(all_pattern_names))

        # Average metrics
        avg_success_rate = sum(opp.average_success_rate for opp in opportunities) / len(
            opportunities
        )
        total_frequency = sum(opp.frequency for opp in opportunities)
        avg_improvement_potential = sum(opp.improvement_potential for opp in opportunities) / len(
            opportunities
        )
        combined_priority_score = sum(opp.priority_score for opp in opportunities)

        # Use the highest priority opportunity as the base
        base_opportunity = max(opportunities, key=lambda x: x.priority_score)

        return EvolutionOpportunity(
            opportunity_id=f"merged_{base_opportunity.opportunity_id}",
            opportunity_type=base_opportunity.opportunity_type,
            pattern_names=unique_pattern_names,
            common_sequence=base_opportunity.common_sequence,
            frequency=total_frequency,
            average_success_rate=avg_success_rate,
            improvement_potential=avg_improvement_potential,
            priority_score=combined_priority_score,
            context_dependencies=base_opportunity.context_dependencies,
        )


class ProposalGenerationStrategy(ABC):
    """
    Abstract base for proposal generation strategies.

    Follows Strategy pattern for different generation approaches.
    """

    @abstractmethod
    def generate_proposals(
        self, opportunities: list[EvolutionOpportunity]
    ) -> list[EvolutionProposal]:
        """Generate evolution proposals from opportunities."""
        pass


class MacroExtensionGenerationStrategy(ProposalGenerationStrategy):
    """
    Generates macro extension proposals from common sequences.

    Single Responsibility: Only handles macro extension proposals.
    """

    def generate_proposals(
        self, opportunities: list[EvolutionOpportunity]
    ) -> list[EvolutionProposal]:
        """Generate macro extension proposals."""
        proposals = []

        for opportunity in opportunities:
            if opportunity.opportunity_type != EvolutionOpportunityType.COMMON_SEQUENCE:
                continue

            if not opportunity.common_sequence:
                continue

            # Generate macro name from sequence
            macro_name = self._generate_macro_name(opportunity.common_sequence)

            # Define DSL changes
            dsl_changes = {
                "new_macros": {macro_name: opportunity.common_sequence},
                "affected_patterns": opportunity.pattern_names,
            }

            # Calculate expected improvement
            expected_improvement = {
                "success_rate_increase": opportunity.improvement_potential,
                "code_reuse_factor": len(opportunity.pattern_names),
                "maintenance_reduction": 0.2,  # Estimated reduction in maintenance overhead
            }

            # Assess implementation complexity
            complexity = self._assess_macro_complexity(opportunity.common_sequence)

            proposal = EvolutionProposal(
                proposal_id=f"macro_ext_{opportunity.opportunity_id}",
                proposal_type=ProposalType.MACRO_EXTENSION,
                opportunity_basis=opportunity,
                dsl_changes=dsl_changes,
                expected_improvement=expected_improvement,
                validation_score=0.0,  # Will be set by validator
                implementation_complexity=complexity,
                estimated_development_time=self._estimate_development_time(complexity),
                risk_assessment={"breaking_changes": False, "compatibility_risk": "low"},
            )

            proposals.append(proposal)

        return proposals

    def _generate_macro_name(self, sequence: list[str]) -> str:
        """Generate appropriate macro name from sequence."""
        # Simple heuristic-based naming
        if "START" in sequence and "DOWN" in sequence:
            return "MENU_NAVIGATE"
        elif sequence.count("UP") > 2:
            return "MOVE_UP_MULTI"
        elif sequence.count("A") > 1:
            return "MULTI_CONFIRM"
        else:
            # Generate generic name
            return f"SEQUENCE_{len(sequence)}_STEPS"

    def _assess_macro_complexity(self, sequence: list[str]) -> ImplementationComplexity:
        """Assess implementation complexity of macro."""
        if len(sequence) <= 3:
            return ImplementationComplexity.LOW
        elif len(sequence) <= 6:
            return ImplementationComplexity.MEDIUM
        else:
            return ImplementationComplexity.HIGH

    def _estimate_development_time(self, complexity: ImplementationComplexity) -> float:
        """Estimate development time in hours."""
        time_map = {
            ImplementationComplexity.LOW: 0.5,
            ImplementationComplexity.MEDIUM: 1.5,
            ImplementationComplexity.HIGH: 4.0,
        }
        return time_map.get(complexity, 2.0)


class ConditionalDSLGenerationStrategy(ProposalGenerationStrategy):
    """
    Generates conditional DSL proposals from context-dependent patterns.

    Single Responsibility: Only handles conditional DSL proposals.
    """

    def generate_proposals(
        self, opportunities: list[EvolutionOpportunity]
    ) -> list[EvolutionProposal]:
        """Generate conditional DSL proposals."""
        proposals = []

        for opportunity in opportunities:
            if opportunity.opportunity_type != EvolutionOpportunityType.CONTEXT_DEPENDENT:
                continue

            if not opportunity.context_dependencies or not opportunity.common_sequence:
                continue

            # Generate conditional syntax
            conditional_syntax = self._generate_conditional_syntax(
                opportunity.common_sequence, opportunity.context_dependencies
            )

            dsl_changes = {
                "new_syntax": {"conditional_pattern": conditional_syntax},
                "context_variables": list(opportunity.context_dependencies.keys()),
                "affected_patterns": opportunity.pattern_names,
            }

            expected_improvement = {
                "context_awareness": opportunity.improvement_potential,
                "success_rate_variance_reduction": 0.3,
                "adaptive_behavior": True,
            }

            proposal = EvolutionProposal(
                proposal_id=f"conditional_{opportunity.opportunity_id}",
                proposal_type=ProposalType.CONDITIONAL_DSL,
                opportunity_basis=opportunity,
                dsl_changes=dsl_changes,
                expected_improvement=expected_improvement,
                validation_score=0.0,
                implementation_complexity=ImplementationComplexity.MEDIUM,
                estimated_development_time=3.0,
                risk_assessment={"syntax_complexity": "medium", "learning_curve": "moderate"},
            )

            proposals.append(proposal)

        return proposals

    def _generate_conditional_syntax(
        self, base_sequence: list[str], context_dependencies: dict[str, Any]
    ) -> str:
        """Generate conditional syntax for context-dependent patterns."""
        # Simplified conditional syntax generation
        contexts = list(context_dependencies.keys())

        if len(contexts) == 2:
            return f"if context.location == '{contexts[0]}' then {base_sequence} else optimized_{base_sequence} end"
        else:
            return f"match context.location with {contexts[0]} -> {base_sequence} | _ -> default_{base_sequence} end"


class EvolutionProposalGenerator:
    """
    Generates DSL improvement proposals from evolution opportunities.

    Single Responsibility: Only handles proposal generation.
    Uses Strategy pattern for different proposal types (Open/Closed).
    """

    def __init__(self, strategies: list[ProposalGenerationStrategy] | None = None):
        """Initialize with generation strategies."""
        if strategies is None:
            self.strategies = [
                MacroExtensionGenerationStrategy(),
                ConditionalDSLGenerationStrategy(),
            ]
        else:
            self.strategies = strategies

    def generate_proposals(
        self, opportunities: list[EvolutionOpportunity]
    ) -> list[EvolutionProposal]:
        """
        Generate evolution proposals from opportunities.

        Performance target: <100ms for typical opportunity sets.

        Args:
            opportunities: List of evolution opportunities

        Returns:
            List of evolution proposals sorted by expected impact

        Raises:
            PerformanceError: If generation exceeds 100ms target
        """
        start_time = time.perf_counter()

        try:
            all_proposals = []

            # Apply all strategies to generate different types of proposals
            for strategy in self.strategies:
                proposals = strategy.generate_proposals(opportunities)
                all_proposals.extend(proposals)

            # Sort by expected impact
            sorted_proposals = sorted(
                all_proposals, key=lambda p: sum(p.expected_improvement.values()), reverse=True
            )

            # Validate performance target
            generation_time = (time.perf_counter() - start_time) * 1000
            if generation_time > 100.0:
                raise PerformanceError(
                    f"Proposal generation took {generation_time:.2f}ms, target is <100ms"
                )

            return sorted_proposals

        except Exception as e:
            generation_time = (time.perf_counter() - start_time) * 1000
            raise GenerationError(
                f"Proposal generation failed after {generation_time:.2f}ms: {e}"
            ) from e


class ValidationRule(ABC):
    """
    Abstract base for validation rules.

    Follows Strategy pattern for different validation approaches.
    """

    @abstractmethod
    def validate(self, proposal: EvolutionProposal) -> tuple[bool, list[str]]:
        """
        Validate a proposal.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass


class MacroNameConflictRule(ValidationRule):
    """Validates that new macros don't conflict with existing names."""

    def __init__(self, existing_macros: list[str] | None = None):
        self.existing_macros = set(
            existing_macros
            or [
                "MOVE_UP",
                "MOVE_DOWN",
                "MOVE_LEFT",
                "MOVE_RIGHT",
                "MENU_OPEN",
                "MENU_CLOSE",
                "CONFIRM",
                "CANCEL",
                "SHORT_WAIT",
                "MEDIUM_WAIT",
                "LONG_WAIT",
            ]
        )

    def validate(self, proposal: EvolutionProposal) -> tuple[bool, list[str]]:
        """Check for macro name conflicts."""
        errors = []

        if proposal.proposal_type != ProposalType.MACRO_EXTENSION:
            return True, []

        new_macros = proposal.dsl_changes.get("new_macros", {})

        for macro_name in new_macros.keys():
            if macro_name in self.existing_macros:
                errors.append(f"Macro name '{macro_name}' conflicts with existing macro")

        return len(errors) == 0, errors


class SyntaxConsistencyRule(ValidationRule):
    """Validates that new syntax is consistent with existing DSL."""

    def validate(self, proposal: EvolutionProposal) -> tuple[bool, list[str]]:
        """Check syntax consistency."""
        errors = []

        if proposal.proposal_type != ProposalType.CONDITIONAL_DSL:
            return True, []

        new_syntax = proposal.dsl_changes.get("new_syntax", {})

        # Simple syntax validation rules
        for syntax_name, syntax_definition in new_syntax.items():
            if not isinstance(syntax_definition, str):
                errors.append(f"Syntax definition for '{syntax_name}' must be a string")
                continue

            # Check for balanced keywords
            if "if" in syntax_definition and "end" not in syntax_definition:
                errors.append(f"Conditional syntax '{syntax_name}' missing 'end' keyword")

            if "match" in syntax_definition and "with" not in syntax_definition:
                errors.append(f"Match syntax '{syntax_name}' missing 'with' keyword")

        return len(errors) == 0, errors


class ComplexityThresholdRule(ValidationRule):
    """Validates that proposals don't exceed complexity thresholds."""

    def __init__(self, max_development_time: float = 8.0):
        self.max_development_time = max_development_time

    def validate(self, proposal: EvolutionProposal) -> tuple[bool, list[str]]:
        """Check complexity thresholds."""
        errors = []

        if proposal.estimated_development_time > self.max_development_time:
            errors.append(
                f"Development time {proposal.estimated_development_time}h exceeds "
                f"maximum {self.max_development_time}h"
            )

        if proposal.implementation_complexity == ImplementationComplexity.HIGH:
            # High complexity proposals need stronger justification
            min_improvement = 0.3
            total_improvement = sum(proposal.expected_improvement.values())
            if total_improvement < min_improvement:
                errors.append(
                    f"High complexity proposal requires improvement >= {min_improvement}, "
                    f"got {total_improvement}"
                )

        return len(errors) == 0, errors


class LanguageValidator:
    """
    Validates language evolution proposals for consistency and safety.

    Single Responsibility: Only handles validation logic.
    Uses Strategy pattern for different validation rules (Open/Closed).
    """

    def __init__(self, rules: list[ValidationRule] | None = None):
        """Initialize with validation rules."""
        if rules is None:
            self.rules = [
                MacroNameConflictRule(),
                SyntaxConsistencyRule(),
                ComplexityThresholdRule(),
            ]
        else:
            self.rules = rules

    def validate_proposals(self, proposals: list[EvolutionProposal]) -> list[EvolutionProposal]:
        """
        Validate evolution proposals and update validation scores.

        Performance target: <50ms for typical proposal sets.

        Args:
            proposals: List of evolution proposals to validate

        Returns:
            List of validated proposals with updated validation scores

        Raises:
            PerformanceError: If validation exceeds 50ms target
        """
        start_time = time.perf_counter()

        try:
            validated_proposals = []

            for proposal in proposals:
                validation_score, errors = self._validate_proposal(proposal)

                # Update proposal with validation score
                validated_proposal = EvolutionProposal(
                    proposal_id=proposal.proposal_id,
                    proposal_type=proposal.proposal_type,
                    opportunity_basis=proposal.opportunity_basis,
                    dsl_changes=proposal.dsl_changes,
                    expected_improvement=proposal.expected_improvement,
                    validation_score=validation_score,
                    implementation_complexity=proposal.implementation_complexity,
                    estimated_development_time=proposal.estimated_development_time,
                    risk_assessment={
                        **(proposal.risk_assessment or {}),
                        "validation_errors": errors,
                    },
                )

                validated_proposals.append(validated_proposal)

            # Validate performance target
            validation_time = (time.perf_counter() - start_time) * 1000
            if validation_time > 50.0:
                raise PerformanceError(f"Validation took {validation_time:.2f}ms, target is <50ms")

            return validated_proposals

        except Exception as e:
            validation_time = (time.perf_counter() - start_time) * 1000
            raise ValidationError(f"Validation failed after {validation_time:.2f}ms: {e}") from e

    def _validate_proposal(self, proposal: EvolutionProposal) -> tuple[float, list[str]]:
        """Validate a single proposal using all rules."""
        all_errors = []
        passed_rules = 0

        for rule in self.rules:
            is_valid, errors = rule.validate(proposal)
            if is_valid:
                passed_rules += 1
            else:
                all_errors.extend(errors)

        # Calculate validation score based on passed rules
        validation_score = passed_rules / len(self.rules) if self.rules else 1.0

        return validation_score, all_errors


# Custom Exception Classes
class PerformanceError(Exception):
    """Raised when performance targets are not met."""

    pass


class AnalysisError(Exception):
    """Raised when pattern analysis fails."""

    pass


class GenerationError(Exception):
    """Raised when proposal generation fails."""

    pass


class ValidationError(Exception):
    """Raised when proposal validation fails."""

    pass


# Export main classes for clean imports
__all__ = [
    "LanguageAnalyzer",
    "EvolutionProposalGenerator",
    "LanguageValidator",
    "EvolutionOpportunity",
    "EvolutionProposal",
    "EvolutionOpportunityType",
    "ProposalType",
    "ImplementationComplexity",
    "PerformanceError",
    "AnalysisError",
    "GenerationError",
    "ValidationError",
]
