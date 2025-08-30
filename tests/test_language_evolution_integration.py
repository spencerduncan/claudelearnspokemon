"""
Integration test for Language Evolution System functionality.

Tests the actual implementation with real data to validate:
- Performance targets are met
- Clean Code architecture works correctly
- Components integrate properly
- SOLID principles are maintained
"""

import time
import unittest
from unittest.mock import Mock

import pytest

from src.claudelearnspokemon.language_evolution import (
    EvolutionOpportunityType,
    EvolutionProposalGenerator,
    ImplementationComplexity,
    LanguageAnalyzer,
    LanguageValidator,
    ProposalType,
)


@pytest.mark.integration
@pytest.mark.medium
class TestLanguageEvolutionSystemIntegration(unittest.TestCase):
    """Integration tests for the complete Language Evolution System."""

    def setUp(self):
        """Set up test environment with real components."""
        self.analyzer = LanguageAnalyzer()
        self.generator = EvolutionProposalGenerator()
        self.validator = LanguageValidator()

        # Create realistic test pattern data
        self.test_patterns = [
            {
                "name": "menu_navigation_slow",
                "success_rate": 0.45,  # Low success - improvement opportunity
                "usage_frequency": 120,
                "input_sequence": ["START", "DOWN", "DOWN", "A", "1"],
                "context": {"location": "any", "menu_type": "main"},
                "evolution_metadata": {},
            },
            {
                "name": "menu_navigation_fast",
                "success_rate": 0.85,  # Good success
                "usage_frequency": 80,
                "input_sequence": ["START", "DOWN", "DOWN", "A"],  # Similar but shorter
                "context": {"location": "any", "menu_type": "main"},
                "evolution_metadata": {},
            },
            {
                "name": "item_use_battle",
                "success_rate": 0.65,  # Medium success in battle
                "usage_frequency": 95,
                "input_sequence": ["START", "RIGHT", "DOWN", "A", "A"],
                "context": {"location": "battle", "item_type": "healing"},
                "evolution_metadata": {},
            },
            {
                "name": "item_use_field",
                "success_rate": 0.90,  # High success in field
                "usage_frequency": 75,
                "input_sequence": ["START", "RIGHT", "DOWN", "A", "A"],
                "context": {"location": "field", "item_type": "healing"},
                "evolution_metadata": {},
            },
            {
                "name": "movement_up_multi",
                "success_rate": 0.70,
                "usage_frequency": 150,
                "input_sequence": ["UP", "UP", "UP", "UP"],  # Repetitive pattern
                "context": {"location": "field"},
                "evolution_metadata": {},
            },
            {
                "name": "confirm_sequence",
                "success_rate": 0.95,
                "usage_frequency": 200,
                "input_sequence": ["A", "1", "A"],  # Common confirmation pattern
                "context": {"location": "any"},
                "evolution_metadata": {},
            },
        ]

    def test_end_to_end_language_evolution_pipeline(self):
        """Test complete language evolution pipeline with real data."""
        start_time = time.perf_counter()

        # Phase 1: Pattern Analysis
        analysis_start = time.perf_counter()
        opportunities = self.analyzer.identify_evolution_opportunities(self.test_patterns)
        analysis_time = (time.perf_counter() - analysis_start) * 1000

        # Validate analysis results
        self.assertGreater(len(opportunities), 0, "Should identify evolution opportunities")
        self.assertLess(analysis_time, 200, f"Analysis took {analysis_time:.2f}ms, target <200ms")

        # Verify types of opportunities found
        opportunity_types = [opp.opportunity_type for opp in opportunities]
        self.assertIn(EvolutionOpportunityType.LOW_SUCCESS_PATTERN, opportunity_types)
        # Should find menu_navigation_slow as low success pattern

        # Phase 2: Proposal Generation
        generation_start = time.perf_counter()
        proposals = self.generator.generate_proposals(opportunities)
        generation_time = (time.perf_counter() - generation_start) * 1000

        # Validate proposal generation
        self.assertGreater(len(proposals), 0, "Should generate evolution proposals")
        self.assertLess(
            generation_time, 100, f"Generation took {generation_time:.2f}ms, target <100ms"
        )

        # Verify proposal types
        proposal_types = [prop.proposal_type for prop in proposals]
        self.assertIn(ProposalType.MACRO_EXTENSION, proposal_types)

        # Phase 3: Validation
        validation_start = time.perf_counter()
        validated_proposals = self.validator.validate_proposals(proposals)
        validation_time = (time.perf_counter() - validation_start) * 1000

        # Validate validation results
        self.assertEqual(
            len(validated_proposals), len(proposals), "All proposals should be validated"
        )
        self.assertLess(
            validation_time, 50, f"Validation took {validation_time:.2f}ms, target <50ms"
        )

        # Check validation scores are assigned
        for proposal in validated_proposals:
            self.assertGreaterEqual(proposal.validation_score, 0.0)
            self.assertLessEqual(proposal.validation_score, 1.0)

        # Total pipeline performance
        total_time = (time.perf_counter() - start_time) * 1000
        self.assertLess(total_time, 400, f"Total pipeline took {total_time:.2f}ms, target <400ms")

        print("\nLanguage Evolution Pipeline Performance:")
        print(f"  Analysis: {analysis_time:.2f}ms (target <200ms)")
        print(f"  Generation: {generation_time:.2f}ms (target <100ms)")
        print(f"  Validation: {validation_time:.2f}ms (target <50ms)")
        print(f"  Total: {total_time:.2f}ms (target <400ms)")
        print(f"  Opportunities found: {len(opportunities)}")
        print(f"  Proposals generated: {len(proposals)}")
        print(
            f"  High-quality proposals: {len([p for p in validated_proposals if p.validation_score >= 0.7])}"
        )

    def test_common_sequence_identification_accuracy(self):
        """Test that common sequence analysis correctly identifies shared patterns."""
        # Patterns with common sequences
        common_sequence_patterns = [
            {
                "name": "pattern_a",
                "success_rate": 0.75,
                "usage_frequency": 50,
                "input_sequence": ["START", "DOWN", "A", "B"],  # Common: START, DOWN
                "context": {},
            },
            {
                "name": "pattern_b",
                "success_rate": 0.68,
                "usage_frequency": 45,
                "input_sequence": ["START", "DOWN", "RIGHT", "A"],  # Common: START, DOWN
                "context": {},
            },
            {
                "name": "pattern_c",
                "success_rate": 0.82,
                "usage_frequency": 60,
                "input_sequence": ["START", "DOWN", "DOWN", "A"],  # Common: START, DOWN
                "context": {},
            },
        ]

        opportunities = self.analyzer.identify_evolution_opportunities(common_sequence_patterns)

        # Should identify common sequence opportunity
        common_seq_opportunities = [
            opp
            for opp in opportunities
            if opp.opportunity_type == EvolutionOpportunityType.COMMON_SEQUENCE
        ]

        self.assertGreater(len(common_seq_opportunities), 0, "Should identify common sequence")

        # Verify the common sequence includes expected elements
        for opp in common_seq_opportunities:
            if opp.common_sequence:
                self.assertIn("START", opp.common_sequence)
                self.assertIn("DOWN", opp.common_sequence)

    def test_context_dependent_pattern_detection(self):
        """Test detection of context-dependent success patterns."""
        # Same base pattern with different success rates by context
        context_patterns = [
            {
                "name": "healing_center",
                "success_rate": 0.95,
                "usage_frequency": 40,
                "input_sequence": ["START", "DOWN", "A", "A"],
                "context": {"location": "pokemon_center"},
            },
            {
                "name": "healing_field",
                "success_rate": 0.45,
                "usage_frequency": 35,
                "input_sequence": ["START", "DOWN", "A", "A"],
                "context": {"location": "field"},
            },
        ]

        opportunities = self.analyzer.identify_evolution_opportunities(context_patterns)

        # Should identify context-dependent opportunity
        context_opportunities = [
            opp
            for opp in opportunities
            if opp.opportunity_type == EvolutionOpportunityType.CONTEXT_DEPENDENT
        ]

        self.assertGreater(len(context_opportunities), 0, "Should identify context dependency")

        # Verify context information is captured
        for opp in context_opportunities:
            self.assertIsNotNone(opp.context_dependencies)

    def test_low_success_pattern_prioritization(self):
        """Test that low success patterns are correctly prioritized for improvement."""
        opportunities = self.analyzer.identify_evolution_opportunities(self.test_patterns)

        # Should identify low success patterns
        low_success_opportunities = [
            opp
            for opp in opportunities
            if opp.opportunity_type == EvolutionOpportunityType.LOW_SUCCESS_PATTERN
        ]

        self.assertGreater(
            len(low_success_opportunities), 0, "Should identify low success patterns"
        )

        # Verify menu_navigation_slow is identified (success_rate = 0.45)
        pattern_names = []
        for opp in low_success_opportunities:
            pattern_names.extend(opp.pattern_names)

        self.assertIn("menu_navigation_slow", pattern_names)

    def test_macro_extension_proposal_quality(self):
        """Test that macro extension proposals are well-formed and implementable."""
        opportunities = self.analyzer.identify_evolution_opportunities(self.test_patterns)
        proposals = self.generator.generate_proposals(opportunities)

        # Find macro extension proposals
        macro_proposals = [
            prop for prop in proposals if prop.proposal_type == ProposalType.MACRO_EXTENSION
        ]

        self.assertGreater(len(macro_proposals), 0, "Should generate macro extension proposals")

        for proposal in macro_proposals:
            # Verify proposal structure
            self.assertIn("new_macros", proposal.dsl_changes)
            self.assertGreater(len(proposal.dsl_changes["new_macros"]), 0)

            # Verify macro definitions
            for macro_name, macro_expansion in proposal.dsl_changes["new_macros"].items():
                self.assertIsInstance(macro_name, str)
                self.assertGreater(len(macro_name), 0)
                self.assertIsInstance(macro_expansion, list)
                self.assertGreater(len(macro_expansion), 0)

            # Verify expected improvements are specified
            self.assertGreater(len(proposal.expected_improvement), 0)

            # Verify implementation complexity is assessed
            self.assertIsInstance(proposal.implementation_complexity, ImplementationComplexity)

    def test_validation_rules_prevent_conflicts(self):
        """Test that validation rules properly prevent problematic proposals."""
        # Create a proposal with conflicting macro name
        from src.claudelearnspokemon.language_evolution import (
            EvolutionOpportunity,
            EvolutionProposal,
        )

        # Mock opportunity
        mock_opportunity = EvolutionOpportunity(
            opportunity_id="test_conflict",
            opportunity_type=EvolutionOpportunityType.COMMON_SEQUENCE,
            pattern_names=["test_pattern"],
            common_sequence=["A", "B"],
            priority_score=1.0,
        )

        # Create proposal with conflicting macro name (MOVE_UP is builtin)
        conflicting_proposal = EvolutionProposal(
            proposal_id="test_conflict_prop",
            proposal_type=ProposalType.MACRO_EXTENSION,
            opportunity_basis=mock_opportunity,
            dsl_changes={"new_macros": {"MOVE_UP": ["UP"]}},  # Conflicts with builtin
            expected_improvement={"success_rate": 0.1},
            validation_score=0.0,
            implementation_complexity=ImplementationComplexity.LOW,
        )

        validated = self.validator.validate_proposals([conflicting_proposal])

        # Should have low validation score due to conflict
        self.assertLess(validated[0].validation_score, 0.9)

        # Should have validation errors in risk assessment
        self.assertIn("validation_errors", validated[0].risk_assessment)
        self.assertGreater(len(validated[0].risk_assessment["validation_errors"]), 0)


@pytest.mark.integration
@pytest.mark.medium
class TestOpusStrategistLanguageEvolutionIntegration(unittest.TestCase):
    """Test OpusStrategist integration with Language Evolution System."""

    def setUp(self):
        """Set up OpusStrategist with mocked dependencies."""
        # Import after ensuring module is available
        from src.claudelearnspokemon.opus_strategist import OpusStrategist

        # Mock ClaudeCodeManager
        self.mock_claude_manager = Mock()
        self.mock_strategic_process = Mock()
        self.mock_claude_manager.get_strategic_process.return_value = self.mock_strategic_process

        # Create OpusStrategist instance
        self.strategist = OpusStrategist(self.mock_claude_manager)

        # Mock recent results data
        self.mock_recent_results = [
            {
                "pattern_name": "menu_slow",
                "success_rate": 0.45,
                "usage_count": 80,
                "input_sequence": ["START", "DOWN", "DOWN", "A", "1"],
                "context": {"location": "menu"},
                "average_execution_time": 2.5,
            },
            {
                "pattern_name": "confirm_pattern",
                "success_rate": 0.85,
                "usage_count": 150,
                "input_sequence": ["A", "1", "A"],
                "context": {"location": "any"},
                "average_execution_time": 0.8,
            },
        ]

    def test_propose_language_evolution_method_integration(self):
        """Test that propose_language_evolution method works with OpusStrategist."""
        # Test the method exists and is callable
        self.assertTrue(hasattr(self.strategist, "propose_language_evolution"))
        self.assertTrue(callable(self.strategist.propose_language_evolution))

        # Test actual method execution
        proposals = self.strategist.propose_language_evolution(self.mock_recent_results)

        # Should return a list
        self.assertIsInstance(proposals, list)

        # Proposals should be high-quality (validation score >= 0.7)
        for proposal in proposals:
            self.assertGreaterEqual(proposal.validation_score, 0.7)

    def test_language_evolution_performance_in_strategist_context(self):
        """Test language evolution performance within OpusStrategist context."""
        start_time = time.perf_counter()

        _proposals = self.strategist.propose_language_evolution(self.mock_recent_results)

        processing_time = (time.perf_counter() - start_time) * 1000

        # Should meet overall performance target
        self.assertLess(
            processing_time, 400, f"Processing took {processing_time:.2f}ms, target <400ms"
        )

        # Should update strategist metrics
        metrics = self.strategist.get_metrics()
        self.assertIn("language_evolution_requests", metrics)
        self.assertGreater(metrics["language_evolution_requests"], 0)

    def test_apply_language_evolution_integration(self):
        """Test that apply_language_evolution integrates with OpusStrategist."""
        # Get some proposals first
        proposals = self.strategist.propose_language_evolution(self.mock_recent_results)

        if proposals:
            # Test applying the best proposal
            best_proposal = max(proposals, key=lambda p: p.validation_score)

            # Test the apply method exists and works
            self.assertTrue(hasattr(self.strategist, "apply_language_evolution"))
            result = self.strategist.apply_language_evolution(best_proposal)

            # Should return a boolean result
            self.assertIsInstance(result, bool)


if __name__ == "__main__":
    # Run integration tests with detailed output
    print("=== Language Evolution System Integration Tests ===")
    print("Testing Clean Code architecture with SOLID principles")
    print("Performance validation and component integration")
    print()

    unittest.main(verbosity=2)
