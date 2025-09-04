#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark for OpusStrategist Language Evolution System

This script validates the performance claims made in PR #232 for the Language Evolution System.
Tests all components against their specified performance targets:

- Pattern Analysis: <200ms
- Context Analysis: <150ms  
- Proposal Generation: <100ms
- Macro Proposals: <80ms
- Language Validation: <50ms
- Syntax Validation: <40ms
- End-to-End Pipeline: <400ms

Generates JSON results with detailed performance metrics including confidence intervals.
"""

import json
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from claudelearnspokemon.language_evolution import (
    LanguageAnalyzer,
    EvolutionProposalGenerator,
    LanguageValidator,
    EvolutionOpportunityType,
    ProposalType,
    ImplementationComplexity,
)


def create_test_patterns(count: int = 50) -> List[Dict[str, Any]]:
    """Create realistic test patterns for benchmarking."""
    patterns = []
    
    # Common Pokemon game sequences
    base_sequences = [
        ["START", "DOWN", "A"],  # Menu navigation
        ["A", "A", "B"],  # Double confirm + back
        ["UP", "UP", "UP", "A"],  # Multi-move + confirm
        ["LEFT", "DOWN", "RIGHT", "A"],  # Movement pattern
        ["START", "DOWN", "DOWN", "A", "B"],  # Deep menu access
        ["A", "WAIT", "A"],  # Confirm with timing
        ["UP", "A", "DOWN", "A"],  # Toggle selection
    ]
    
    contexts = [
        {"location": "pokecenter", "menu_type": "heal", "battle_state": "none"},
        {"location": "mart", "menu_type": "shop", "battle_state": "none"},
        {"location": "route_1", "menu_type": "bag", "battle_state": "wild"},
        {"location": "gym", "menu_type": "battle", "battle_state": "trainer"},
        {"location": "pokecenter", "menu_type": "pc", "battle_state": "none"},
        {"location": "route_2", "menu_type": "pokemon", "battle_state": "none"},
    ]
    
    for i in range(count):
        sequence_idx = i % len(base_sequences)
        context_idx = i % len(contexts)
        
        # Add some variation to success rates and frequencies
        base_success = 0.7 + (i % 3) * 0.1  # 0.7, 0.8, 0.9 rotation
        frequency_variation = 10 + (i % 40)  # 10-50 range
        
        pattern = {
            "name": f"pattern_{i}_{sequence_idx}",
            "success_rate": base_success + (hash(str(i)) % 100) / 1000.0,  # Add small variation
            "usage_frequency": frequency_variation + (i % 20),
            "input_sequence": base_sequences[sequence_idx].copy(),
            "context": contexts[context_idx].copy(),
            "average_execution_time": 0.5 + (i % 10) * 0.1,  # 0.5-1.4s range
        }
        patterns.append(pattern)
    
    return patterns


def benchmark_component(component_name: str, operation_func, iterations: int = 50) -> Dict[str, Any]:
    """Benchmark a component with statistical analysis."""
    print(f"  Benchmarking {component_name}...")
    
    times = []
    errors = []
    
    for i in range(iterations):
        try:
            start_time = time.perf_counter()
            result = operation_func(i)
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            times.append(execution_time)
            
        except Exception as e:
            errors.append(str(e))
    
    if not times:
        return {
            "component": component_name,
            "success": False,
            "error": f"All {iterations} iterations failed",
            "errors": errors[:5],  # First 5 errors
        }
    
    # Calculate comprehensive statistics
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
    median_time = statistics.median(times)
    
    # Calculate confidence intervals (95%)
    if len(times) > 1:
        sorted_times = sorted(times)
        confidence_95_lower = sorted_times[int(0.025 * len(times))]
        confidence_95_upper = sorted_times[int(0.975 * len(times))]
    else:
        confidence_95_lower = confidence_95_upper = avg_time
    
    return {
        "component": component_name,
        "success": True,
        "iterations": iterations,
        "successful_runs": len(times),
        "failed_runs": len(errors),
        "avg_time_ms": avg_time,
        "median_time_ms": median_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_dev_ms": std_dev,
        "confidence_95_lower_ms": confidence_95_lower,
        "confidence_95_upper_ms": confidence_95_upper,
        "sample_count": len(times),
        "errors": errors[:3] if errors else [],  # Sample of errors
    }


def benchmark_pattern_analysis():
    """Benchmark pattern analysis performance (<200ms target)."""
    analyzer = LanguageAnalyzer()
    test_patterns = create_test_patterns(100)  # Larger dataset for analysis
    
    def run_analysis(iteration):
        # Use subset to simulate real-world varying loads
        pattern_subset = test_patterns[:20 + (iteration % 30)]
        return analyzer.identify_evolution_opportunities(pattern_subset)
    
    return benchmark_component("Pattern Analysis", run_analysis, iterations=30)


def benchmark_context_analysis():
    """Benchmark context-specific analysis (<150ms target)."""
    from claudelearnspokemon.language_evolution import ContextDependentAnalysisStrategy
    
    strategy = ContextDependentAnalysisStrategy()
    test_patterns = create_test_patterns(80)
    
    def run_context_analysis(iteration):
        pattern_subset = test_patterns[:15 + (iteration % 25)]
        return strategy.analyze_patterns(pattern_subset)
    
    return benchmark_component("Context Analysis", run_context_analysis, iterations=40)


def benchmark_proposal_generation():
    """Benchmark proposal generation performance (<100ms target)."""
    generator = EvolutionProposalGenerator()
    analyzer = LanguageAnalyzer()
    test_patterns = create_test_patterns(60)
    
    # Pre-generate opportunities to isolate generation performance
    opportunities = analyzer.identify_evolution_opportunities(test_patterns)
    
    def run_generation(iteration):
        # Use subset of opportunities
        opp_subset = opportunities[:5 + (iteration % 10)]
        return generator.generate_proposals(opp_subset)
    
    return benchmark_component("Proposal Generation", run_generation, iterations=60)


def benchmark_macro_proposals():
    """Benchmark macro-specific proposal generation (<80ms target)."""
    from claudelearnspokemon.language_evolution import MacroExtensionGenerationStrategy
    
    strategy = MacroExtensionGenerationStrategy()
    analyzer = LanguageAnalyzer()
    test_patterns = create_test_patterns(40)
    
    # Generate opportunities with focus on common sequences
    opportunities = analyzer.identify_evolution_opportunities(test_patterns)
    macro_opportunities = [
        opp for opp in opportunities 
        if opp.opportunity_type == EvolutionOpportunityType.COMMON_SEQUENCE
    ]
    
    def run_macro_generation(iteration):
        opp_subset = macro_opportunities[:3 + (iteration % 5)]
        return strategy.generate_proposals(opp_subset)
    
    return benchmark_component("Macro Proposals", run_macro_generation, iterations=80)


def benchmark_language_validation():
    """Benchmark language validation performance (<50ms target)."""
    validator = LanguageValidator()
    generator = EvolutionProposalGenerator()
    analyzer = LanguageAnalyzer()
    
    # Pre-generate proposals for validation
    test_patterns = create_test_patterns(30)
    opportunities = analyzer.identify_evolution_opportunities(test_patterns)
    proposals = generator.generate_proposals(opportunities)
    
    def run_validation(iteration):
        # Validate subset of proposals
        proposal_subset = proposals[:3 + (iteration % 5)]
        return validator.validate_proposals(proposal_subset)
    
    return benchmark_component("Language Validation", run_validation, iterations=100)


def benchmark_syntax_validation():
    """Benchmark syntax validation performance (<40ms target)."""
    from claudelearnspokemon.language_evolution import SyntaxConsistencyRule
    
    rule = SyntaxConsistencyRule()
    generator = EvolutionProposalGenerator()
    analyzer = LanguageAnalyzer()
    
    # Generate conditional DSL proposals for syntax validation
    test_patterns = create_test_patterns(25)
    opportunities = analyzer.identify_evolution_opportunities(test_patterns)
    proposals = generator.generate_proposals(opportunities)
    
    conditional_proposals = [
        p for p in proposals 
        if p.proposal_type == ProposalType.CONDITIONAL_DSL
    ]
    
    def run_syntax_validation(iteration):
        if not conditional_proposals:
            # Create a mock conditional proposal for testing
            from claudelearnspokemon.language_evolution import EvolutionProposal, EvolutionOpportunity
            mock_opp = EvolutionOpportunity(
                opportunity_id="test",
                opportunity_type=EvolutionOpportunityType.CONTEXT_DEPENDENT,
                pattern_names=["test_pattern"],
            )
            mock_proposal = EvolutionProposal(
                proposal_id=f"test_{iteration}",
                proposal_type=ProposalType.CONDITIONAL_DSL,
                opportunity_basis=mock_opp,
                dsl_changes={
                    "new_syntax": {
                        "conditional_test": f"if context.location == 'test_{iteration % 3}' then [A, B] else [B, A] end"
                    }
                },
                expected_improvement={"test": 0.1},
                validation_score=0.0,
                implementation_complexity=ImplementationComplexity.LOW,
            )
            return rule.validate(mock_proposal)
        else:
            proposal = conditional_proposals[iteration % len(conditional_proposals)]
            return rule.validate(proposal)
    
    return benchmark_component("Syntax Validation", run_syntax_validation, iterations=120)


def benchmark_end_to_end_pipeline():
    """Benchmark complete pipeline performance (<400ms target)."""
    analyzer = LanguageAnalyzer()
    generator = EvolutionProposalGenerator()
    validator = LanguageValidator()
    
    def run_complete_pipeline(iteration):
        # Full pipeline with varying data sizes
        pattern_count = 20 + (iteration % 30)  # 20-50 patterns
        test_patterns = create_test_patterns(pattern_count)
        
        # Step 1: Analyze patterns
        opportunities = analyzer.identify_evolution_opportunities(test_patterns)
        
        # Step 2: Generate proposals
        proposals = generator.generate_proposals(opportunities)
        
        # Step 3: Validate proposals
        validated_proposals = validator.validate_proposals(proposals)
        
        return {
            "opportunities": len(opportunities),
            "proposals": len(proposals),
            "validated": len(validated_proposals),
            "patterns_processed": len(test_patterns),
        }
    
    return benchmark_component("End-to-End Pipeline", run_complete_pipeline, iterations=25)


def validate_performance_targets(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate results against performance targets."""
    targets = {
        "Pattern Analysis": 200.0,
        "Context Analysis": 150.0,
        "Proposal Generation": 100.0,
        "Macro Proposals": 80.0,
        "Language Validation": 50.0,
        "Syntax Validation": 40.0,
        "End-to-End Pipeline": 400.0,
    }
    
    validation_results = {
        "total_benchmarks": len(results),
        "benchmarks_passed": 0,
        "performance_claims_validated": [],
        "performance_failures": [],
    }
    
    for result in results:
        if not result.get("success", False):
            validation_results["performance_failures"].append({
                "component": result["component"],
                "reason": "benchmark_failed",
                "error": result.get("error", "Unknown error"),
            })
            continue
            
        component = result["component"]
        target_ms = targets.get(component, float('inf'))
        actual_ms = result["avg_time_ms"]
        
        if actual_ms <= target_ms:
            validation_results["benchmarks_passed"] += 1
            improvement_factor = target_ms / actual_ms if actual_ms > 0 else float('inf')
            validation_results["performance_claims_validated"].append({
                "component": component,
                "target_ms": target_ms,
                "actual_ms": actual_ms,
                "improvement_factor": improvement_factor,
                "sample_count": result["sample_count"],
                "confidence_95_lower": result.get("confidence_95_lower_ms", actual_ms),
                "confidence_95_upper": result.get("confidence_95_upper_ms", actual_ms),
            })
        else:
            validation_results["performance_failures"].append({
                "component": component,
                "target_ms": target_ms,
                "actual_ms": actual_ms,
                "performance_deficit": actual_ms - target_ms,
                "reason": "target_exceeded",
            })
    
    # Calculate average improvement
    if validation_results["performance_claims_validated"]:
        improvements = [
            claim["improvement_factor"] 
            for claim in validation_results["performance_claims_validated"]
            if claim["improvement_factor"] != float('inf')
        ]
        validation_results["average_improvement"] = statistics.mean(improvements) if improvements else 0
    else:
        validation_results["average_improvement"] = 0
    
    return validation_results


def main():
    """Run comprehensive Language Evolution System benchmark."""
    print("🚀 Language Evolution System Performance Benchmark")
    print("=" * 60)
    print(f"Testing performance targets from PR #232")
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    # Run all benchmarks
    benchmark_functions = [
        benchmark_pattern_analysis,
        benchmark_context_analysis,
        benchmark_proposal_generation,
        benchmark_macro_proposals,
        benchmark_language_validation,
        benchmark_syntax_validation,
        benchmark_end_to_end_pipeline,
    ]
    
    results = []
    
    for i, benchmark_func in enumerate(benchmark_functions, 1):
        print(f"📊 Running benchmark {i}/{len(benchmark_functions)}: {benchmark_func.__name__}")
        
        try:
            result = benchmark_func()
            results.append(result)
            
            if result.get("success", False):
                avg_time = result["avg_time_ms"]
                sample_count = result["sample_count"]
                print(f"   ✓ Completed: {avg_time:.2f}ms average ({sample_count} samples)")
            else:
                print(f"   ✗ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            error_result = {
                "component": benchmark_func.__name__,
                "success": False,
                "error": str(e),
            }
            results.append(error_result)
            print(f"   ✗ Exception: {str(e)}")
        
        print()
    
    # Validate against performance targets
    print("🎯 Validating Performance Targets")
    print("-" * 40)
    
    validation_results = validate_performance_targets(results)
    
    passed = validation_results["benchmarks_passed"]
    total = validation_results["total_benchmarks"]
    
    print(f"Benchmarks passed: {passed}/{total}")
    print(f"Average improvement factor: {validation_results.get('average_improvement', 0):.2f}x")
    print()
    
    # Show individual results
    for result in results:
        if result.get("success", False):
            component = result["component"]
            avg_time = result["avg_time_ms"]
            samples = result["sample_count"]
            
            # Find corresponding validation
            validated = None
            for claim in validation_results.get("performance_claims_validated", []):
                if claim["component"] == component:
                    validated = claim
                    break
            
            if validated:
                target = validated["target_ms"]
                improvement = validated["improvement_factor"]
                status = f"✓ {improvement:.1f}x better than {target}ms target"
            else:
                status = "✗ Target not met"
            
            print(f"  {component}: {avg_time:.2f}ms ({samples} samples) {status}")
        else:
            print(f"  {result['component']}: ✗ FAILED - {result.get('error', 'Unknown error')}")
    
    # Generate detailed JSON report
    timestamp = int(time.time())
    report = {
        "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        "validation_status": "PASSED" if passed == total else "FAILED",
        "overall_performance": validation_results,
        "detailed_results": results,
    }
    
    # Save results to file
    results_filename = f"language_evolution_benchmark_results_{timestamp}.json"
    with open(results_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Detailed results saved to: {results_filename}")
    
    # Final status
    if passed == total:
        print("\n🎉 ALL PERFORMANCE TARGETS MET!")
        print("Language Evolution System performance claims validated.")
        return True
    else:
        failed = total - passed
        print(f"\n❌ {failed}/{total} BENCHMARKS FAILED")
        print("Performance targets not met. See detailed results above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)