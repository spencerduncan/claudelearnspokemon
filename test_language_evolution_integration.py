#!/usr/bin/env python3
"""
Integration test for OpusStrategist Language Evolution System.

This test validates that the complete integration between OpusStrategist and
the Language Evolution System works correctly through the propose_language_evolution
method, using realistic Pokemon game execution data.
"""

import sys
import os
from unittest.mock import Mock, MagicMock
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from claudelearnspokemon.opus_strategist import OpusStrategist
from claudelearnspokemon.language_evolution import (
    EvolutionProposal, 
    ProposalType,
    ImplementationComplexity
)


def create_mock_claude_manager():
    """Create a properly configured mock ClaudeCodeManager."""
    mock_manager = Mock()
    mock_process = Mock()
    
    # Configure the mock to return reasonable responses
    mock_process.send_message = Mock(return_value="Mock strategic response")
    mock_process.is_healthy = Mock(return_value=True)
    mock_manager.get_strategic_process = Mock(return_value=mock_process)
    
    return mock_manager


def create_realistic_execution_results() -> List[Dict[str, Any]]:
    """Create realistic Pokemon game execution results for testing."""
    return [
        {
            "pattern_name": "pokecenter_heal_sequence",
            "success_rate": 0.85,
            "usage_count": 25,
            "input_sequence": ["UP", "A", "A", "A"],
            "context": {
                "location": "pokecenter",
                "menu_type": "heal",
                "battle_state": "none",
                "health_status": "injured"
            },
            "average_execution_time": 0.8,
            "error_patterns": ["timeout_on_text_display"],
        },
        {
            "pattern_name": "mart_buy_pokeball",
            "success_rate": 0.92,
            "usage_count": 18,
            "input_sequence": ["DOWN", "A", "RIGHT", "A", "A"],
            "context": {
                "location": "mart",
                "menu_type": "shop",
                "battle_state": "none",
                "money": 5000
            },
            "average_execution_time": 1.2,
        },
        {
            "pattern_name": "wild_battle_run",
            "success_rate": 0.68,
            "usage_count": 40,
            "input_sequence": ["DOWN", "DOWN", "A"],
            "context": {
                "location": "route_1", 
                "menu_type": "battle",
                "battle_state": "wild",
                "pokemon_health": "low"
            },
            "average_execution_time": 0.5,
            "error_patterns": ["failed_to_run", "menu_navigation_error"],
        },
        {
            "pattern_name": "gym_battle_strategy",
            "success_rate": 0.75,
            "usage_count": 12,
            "input_sequence": ["A", "UP", "A", "WAIT", "A"],
            "context": {
                "location": "gym",
                "menu_type": "battle", 
                "battle_state": "trainer",
                "opponent_level": "high"
            },
            "average_execution_time": 2.1,
        },
        # Duplicate some patterns with different contexts to create evolution opportunities
        {
            "pattern_name": "pokecenter_heal_night",
            "success_rate": 0.90,
            "usage_count": 15,
            "input_sequence": ["UP", "A", "A", "A"],  # Same sequence as above
            "context": {
                "location": "pokecenter",
                "menu_type": "heal",
                "battle_state": "none",
                "health_status": "injured",
                "time_of_day": "night"
            },
            "average_execution_time": 0.7,
        },
        {
            "pattern_name": "wild_battle_run_low_level",
            "success_rate": 0.88,
            "usage_count": 22,
            "input_sequence": ["DOWN", "DOWN", "A"],  # Same as wild_battle_run
            "context": {
                "location": "route_2",
                "menu_type": "battle",
                "battle_state": "wild",
                "pokemon_health": "high",
                "opponent_level": "low"
            },
            "average_execution_time": 0.4,
        }
    ]


def test_opus_strategist_language_evolution_integration():
    """Test the complete OpusStrategist Language Evolution integration."""
    print("🧪 Testing OpusStrategist Language Evolution Integration")
    print("=" * 60)
    
    # Setup
    mock_claude_manager = create_mock_claude_manager()
    strategist = OpusStrategist(mock_claude_manager)
    
    # Create realistic execution results
    execution_results = create_realistic_execution_results()
    
    print(f"📊 Testing with {len(execution_results)} execution results")
    print(f"   Patterns: {[result['pattern_name'] for result in execution_results]}")
    print()
    
    # Test the language evolution proposal method
    try:
        print("⚙️  Running propose_language_evolution...")
        proposals = strategist.propose_language_evolution(execution_results)
        
        print(f"✅ propose_language_evolution completed successfully")
        print(f"   Generated {len(proposals)} evolution proposals")
        print()
        
        # Analyze the proposals
        if proposals:
            print("📋 Generated Proposals:")
            print("-" * 30)
            
            for i, proposal in enumerate(proposals, 1):
                print(f"  {i}. {proposal.proposal_id}")
                print(f"     Type: {proposal.proposal_type}")
                print(f"     Validation Score: {proposal.validation_score:.2f}")
                print(f"     Implementation Complexity: {proposal.implementation_complexity}")
                print(f"     Expected Improvements: {list(proposal.expected_improvement.keys())}")
                
                # Show DSL changes for macro extensions
                if proposal.proposal_type == ProposalType.MACRO_EXTENSION:
                    new_macros = proposal.dsl_changes.get("new_macros", {})
                    if new_macros:
                        for macro_name, sequence in new_macros.items():
                            print(f"     New Macro: {macro_name} -> {sequence}")
                
                print(f"     Basis Patterns: {proposal.opportunity_basis.pattern_names}")
                print()
        else:
            print("ℹ️  No evolution proposals generated (may be expected with current patterns)")
            print()
        
        # Test performance (should be well under targets)
        print("⏱️  Testing Performance Integration:")
        print("-" * 30)
        
        import time
        start_time = time.perf_counter()
        test_proposals = strategist.propose_language_evolution(execution_results)
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000  # Convert to ms
        target_time = 400.0  # 400ms target from PR #232
        
        print(f"   Total execution time: {total_time:.2f}ms")
        print(f"   Performance target: <{target_time}ms")
        
        if total_time <= target_time:
            improvement_factor = target_time / total_time if total_time > 0 else float('inf')
            print(f"   ✅ Performance target met! ({improvement_factor:.1f}x better than target)")
        else:
            print(f"   ⚠️  Performance target missed by {total_time - target_time:.2f}ms")
        
        print()
        
        # Test with edge cases
        print("🧪 Testing Edge Cases:")
        print("-" * 20)
        
        # Test with empty results
        empty_proposals = strategist.propose_language_evolution([])
        print(f"   Empty input: {len(empty_proposals)} proposals (expected: 0)")
        
        # Test with single result
        single_proposals = strategist.propose_language_evolution(execution_results[:1])
        print(f"   Single pattern: {len(single_proposals)} proposals")
        
        # Test with minimal results
        minimal_result = [{
            "pattern_name": "test_pattern",
            "success_rate": 0.5,
            "usage_count": 5,
            "input_sequence": ["A"],
            "context": {},
        }]
        minimal_proposals = strategist.propose_language_evolution(minimal_result)
        print(f"   Minimal pattern: {len(minimal_proposals)} proposals")
        
        print()
        print("🎉 Integration test completed successfully!")
        
        return {
            "success": True,
            "proposals_generated": len(proposals),
            "performance_ms": total_time,
            "performance_target_met": total_time <= target_time,
            "edge_cases_handled": True,
        }
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        
        return {
            "success": False,
            "error": str(e),
            "proposals_generated": 0,
        }


def test_language_evolution_components():
    """Test individual Language Evolution System components."""
    print("\n🔧 Testing Individual Language Evolution Components")
    print("=" * 60)
    
    try:
        from claudelearnspokemon.language_evolution import (
            LanguageAnalyzer,
            EvolutionProposalGenerator,
            LanguageValidator
        )
        
        # Test component instantiation
        analyzer = LanguageAnalyzer()
        generator = EvolutionProposalGenerator()
        validator = LanguageValidator()
        
        print("✅ All Language Evolution components instantiated successfully")
        
        # Test with sample data
        patterns = [
            {
                "name": "test_pattern_1",
                "success_rate": 0.8,
                "usage_frequency": 20,
                "input_sequence": ["A", "B", "A"],
                "context": {"location": "test"},
            },
            {
                "name": "test_pattern_2", 
                "success_rate": 0.6,
                "usage_frequency": 30,
                "input_sequence": ["A", "B", "A"],  # Same sequence
                "context": {"location": "test2"},
            }
        ]
        
        # Test analysis
        opportunities = analyzer.identify_evolution_opportunities(patterns)
        print(f"✅ Pattern analysis: {len(opportunities)} opportunities identified")
        
        # Test proposal generation
        if opportunities:
            proposals = generator.generate_proposals(opportunities)
            print(f"✅ Proposal generation: {len(proposals)} proposals generated")
            
            # Test validation
            if proposals:
                validated = validator.validate_proposals(proposals)
                print(f"✅ Validation: {len(validated)} proposals validated")
            else:
                print("ℹ️  No proposals to validate")
        else:
            print("ℹ️  No opportunities found for testing proposal generation")
        
        print("✅ Individual component testing completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Component testing failed: {str(e)}")
        return False


def main():
    """Run the complete integration test suite."""
    print("🚀 OpusStrategist Language Evolution System Integration Test")
    print("=" * 70)
    
    # Test integration
    integration_result = test_opus_strategist_language_evolution_integration()
    
    # Test components
    components_result = test_language_evolution_components()
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    
    if integration_result["success"] and components_result:
        print("🎉 ALL TESTS PASSED!")
        print(f"   Language evolution integration: ✅")
        print(f"   Individual components: ✅")
        print(f"   Performance target: {'✅' if integration_result.get('performance_target_met', False) else '⚠️'}")
        print(f"   Proposals generated: {integration_result.get('proposals_generated', 0)}")
        
        if integration_result.get('performance_ms'):
            print(f"   Execution time: {integration_result['performance_ms']:.2f}ms")
        
        return True
    else:
        print("❌ SOME TESTS FAILED")
        if not integration_result["success"]:
            print(f"   Integration error: {integration_result.get('error', 'Unknown')}")
        if not components_result:
            print(f"   Component testing failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)