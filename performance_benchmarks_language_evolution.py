#!/usr/bin/env python3
"""
Performance Benchmarks for OpusStrategist Language Evolution System

Empirical validation of the claimed performance improvements:
- Evolution analysis: 0.20ms (1000x faster than 200ms target)
- Proposal generation: 0.04ms (2500x faster than 100ms target)
- Language validation: 0.02ms (2500x faster than 50ms target)
- End-to-end pipeline: 0.32ms (1200x faster than 400ms target)

Scientist approach: Measure multiple runs, statistical analysis, confidence intervals.
"""

import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Callable
from unittest.mock import Mock

# Import the language evolution components
from src.claudelearnspokemon.language_evolution import (
    LanguageAnalyzer,
    EvolutionProposalGenerator,
    LanguageValidator,
    ContextData,
    EvolutionOpportunityType,
    ProposalType,
    ImplementationComplexity
)


@dataclass
class BenchmarkResult:
    """Data structure for benchmark measurements."""
    component: str
    operation: str
    target_ms: float
    actual_ms: float
    improvement_factor: float
    sample_count: int
    min_ms: float
    max_ms: float
    std_dev_ms: float
    confidence_95_lower: float
    confidence_95_upper: float


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    timestamp: str
    results: List[BenchmarkResult]
    overall_performance: Dict[str, Any]
    validation_status: str


@contextmanager
def high_precision_timer():
    """Context manager for high-precision timing measurements."""
    start = time.perf_counter_ns()
    yield
    end = time.perf_counter_ns()
    elapsed_ms = (end - start) / 1_000_000  # Convert to milliseconds
    globals()['_last_measurement'] = elapsed_ms


def benchmark_component(func: Callable, iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark a component with multiple iterations for statistical validity.
    
    Args:
        func: Function to benchmark
        iterations: Number of measurement iterations
        
    Returns:
        Statistical measurements including mean, std dev, confidence intervals
    """
    measurements = []
    
    for _ in range(iterations):
        with high_precision_timer():
            func()
        measurements.append(globals()['_last_measurement'])
    
    mean_ms = statistics.mean(measurements)
    std_dev = statistics.stdev(measurements) if len(measurements) > 1 else 0
    
    # Calculate 95% confidence interval
    if len(measurements) > 1:
        margin_of_error = 1.96 * (std_dev / (len(measurements) ** 0.5))
        confidence_95_lower = mean_ms - margin_of_error
        confidence_95_upper = mean_ms + margin_of_error
    else:
        confidence_95_lower = confidence_95_upper = mean_ms
    
    return {
        'mean_ms': mean_ms,
        'min_ms': min(measurements),
        'max_ms': max(measurements),
        'std_dev_ms': std_dev,
        'confidence_95_lower': confidence_95_lower,
        'confidence_95_upper': confidence_95_upper,
        'measurements': measurements
    }


class LanguageEvolutionBenchmark:
    """Comprehensive benchmarking suite for Language Evolution System."""
    
    def __init__(self):
        """Initialize benchmark suite with test data."""
        self.setup_test_data()
        self.setup_components()
        
    def setup_test_data(self):
        """Create realistic test data for benchmarking."""
        # Pattern analysis test data
        self.pattern_data = {
            'battle_sequences': [
                ['move_forward', 'attack', 'move_back', 'heal'],
                ['move_forward', 'attack', 'move_forward', 'attack'],
                ['move_back', 'heal', 'move_forward', 'attack'],
                ['attack', 'move_back', 'heal', 'move_forward']
            ] * 25,  # 100 sequences total
            'success_rates': [0.8, 0.9, 0.7, 0.85] * 25,
            'execution_times': [12.5, 8.2, 15.1, 10.3] * 25
        }
        
        # Context data for testing
        self.context_data = ContextData(
            success_rates=[0.8, 0.9, 0.7, 0.85, 0.92, 0.76] * 10,
            frequencies=[15, 22, 8, 18, 25, 12] * 10,
            pattern_names=['forward_attack', 'defensive_heal', 'combo_strike', 
                          'retreat_heal', 'aggressive_rush', 'tactical_wait'] * 10
        )
        
        # Mock historical results for proposal generation
        self.mock_results = [
            Mock(success_rate=0.85, pattern=['move', 'attack'], frequency=20),
            Mock(success_rate=0.70, pattern=['heal', 'defend'], frequency=15),
            Mock(success_rate=0.92, pattern=['combo', 'strike'], frequency=8),
        ] * 20  # 60 results total
        
    def setup_components(self):
        """Initialize the language evolution components."""
        self.analyzer = LanguageAnalyzer()
        self.generator = EvolutionProposalGenerator()
        self.validator = LanguageValidator()
        
    def benchmark_pattern_analysis(self) -> Dict[str, float]:
        """Benchmark pattern analysis performance (target: <200ms)."""
        def analysis_operation():
            opportunities = self.analyzer.analyze_patterns(
                self.pattern_data['battle_sequences'],
                self.pattern_data['success_rates']
            )
            return opportunities
            
        return benchmark_component(analysis_operation, iterations=200)
    
    def benchmark_context_analysis(self) -> Dict[str, float]:
        """Benchmark context-dependent pattern analysis."""
        def context_operation():
            analysis = self.analyzer.analyze_context_patterns(self.context_data)
            return analysis
            
        return benchmark_component(context_operation, iterations=200)
    
    def benchmark_proposal_generation(self) -> Dict[str, float]:
        """Benchmark proposal generation performance (target: <100ms)."""
        def generation_operation():
            proposals = self.generator.generate_proposals(
                EvolutionOpportunityType.COMMON_SEQUENCE,
                {'common_patterns': [['move', 'attack'], ['heal', 'defend']]},
                impact_score=0.85
            )
            return proposals
            
        return benchmark_component(generation_operation, iterations=500)
    
    def benchmark_macro_proposals(self) -> Dict[str, float]:
        """Benchmark macro extension proposal generation."""
        def macro_operation():
            proposals = self.generator.generate_macro_proposals(
                [['move_forward', 'attack'], ['move_back', 'heal']] * 10,
                min_frequency=5
            )
            return proposals
            
        return benchmark_component(macro_operation, iterations=300)
    
    def benchmark_language_validation(self) -> Dict[str, float]:
        """Benchmark language validation performance (target: <50ms)."""
        def validation_operation():
            # Create test proposal for validation
            test_proposal = {
                'type': ProposalType.MACRO_EXTENSION,
                'content': 'battle_combo = move_forward -> attack -> retreat',
                'complexity': ImplementationComplexity.LOW,
                'impact': 0.85
            }
            
            result = self.validator.validate_proposal(
                test_proposal,
                existing_dsl=['move_forward', 'attack', 'retreat', 'heal']
            )
            return result
            
        return benchmark_component(validation_operation, iterations=1000)
    
    def benchmark_syntax_validation(self) -> Dict[str, float]:
        """Benchmark syntax consistency validation."""
        def syntax_operation():
            result = self.validator.validate_syntax_consistency(
                ['battle_combo', 'healing_sequence', 'attack_pattern'],
                ['move_forward', 'attack', 'retreat', 'heal', 'defend']
            )
            return result
            
        return benchmark_component(syntax_operation, iterations=800)
    
    def benchmark_end_to_end_pipeline(self) -> Dict[str, float]:
        """Benchmark complete end-to-end evolution pipeline (target: <400ms)."""
        def pipeline_operation():
            # Step 1: Pattern analysis
            opportunities = self.analyzer.analyze_patterns(
                self.pattern_data['battle_sequences'][:10],  # Smaller dataset for e2e
                self.pattern_data['success_rates'][:10]
            )
            
            # Step 2: Proposal generation
            if opportunities:
                proposals = self.generator.generate_proposals(
                    list(opportunities.keys())[0],
                    list(opportunities.values())[0],
                    impact_score=0.8
                )
            else:
                proposals = []
            
            # Step 3: Validation
            validated_proposals = []
            for proposal in proposals[:3]:  # Validate first 3 proposals
                validation_result = self.validator.validate_proposal(
                    proposal,
                    existing_dsl=['move_forward', 'attack', 'retreat', 'heal']
                )
                if validation_result.get('is_valid', False):
                    validated_proposals.append(proposal)
            
            return {
                'opportunities_found': len(opportunities),
                'proposals_generated': len(proposals),
                'proposals_validated': len(validated_proposals)
            }
            
        return benchmark_component(pipeline_operation, iterations=150)
    
    def run_comprehensive_benchmark(self) -> BenchmarkSuite:
        """Run the complete benchmark suite and generate results."""
        print("🔬 Starting comprehensive Language Evolution System benchmarks...")
        print("=" * 70)
        
        benchmarks = [
            ("Pattern Analysis", "analyze_patterns", 200.0, self.benchmark_pattern_analysis),
            ("Context Analysis", "analyze_context", 150.0, self.benchmark_context_analysis),
            ("Proposal Generation", "generate_proposals", 100.0, self.benchmark_proposal_generation),
            ("Macro Proposals", "generate_macros", 80.0, self.benchmark_macro_proposals),
            ("Language Validation", "validate_proposal", 50.0, self.benchmark_language_validation),
            ("Syntax Validation", "validate_syntax", 40.0, self.benchmark_syntax_validation),
            ("End-to-End Pipeline", "complete_pipeline", 400.0, self.benchmark_end_to_end_pipeline),
        ]
        
        results = []
        overall_performance = {
            'total_benchmarks': len(benchmarks),
            'benchmarks_passed': 0,
            'average_improvement': 0,
            'performance_claims_validated': []
        }
        
        improvement_factors = []
        
        for component, operation, target_ms, benchmark_func in benchmarks:
            print(f"\n📊 Benchmarking {component} ({operation})...")
            print(f"   Target: <{target_ms}ms")
            
            try:
                measurements = benchmark_func()
                actual_ms = measurements['mean_ms']
                improvement_factor = target_ms / actual_ms
                
                result = BenchmarkResult(
                    component=component,
                    operation=operation,
                    target_ms=target_ms,
                    actual_ms=actual_ms,
                    improvement_factor=improvement_factor,
                    sample_count=len(measurements['measurements']),
                    min_ms=measurements['min_ms'],
                    max_ms=measurements['max_ms'],
                    std_dev_ms=measurements['std_dev_ms'],
                    confidence_95_lower=measurements['confidence_95_lower'],
                    confidence_95_upper=measurements['confidence_95_upper']
                )
                
                results.append(result)
                improvement_factors.append(improvement_factor)
                
                if actual_ms <= target_ms:
                    overall_performance['benchmarks_passed'] += 1
                    status = "✅ PASSED"
                else:
                    status = "❌ FAILED"
                
                print(f"   Actual: {actual_ms:.3f}ms (±{measurements['std_dev_ms']:.3f})")
                print(f"   Improvement: {improvement_factor:.1f}x faster {status}")
                print(f"   95% CI: [{measurements['confidence_95_lower']:.3f}, {measurements['confidence_95_upper']:.3f}]ms")
                
            except Exception as e:
                print(f"   ❌ BENCHMARK FAILED: {e}")
                results.append(BenchmarkResult(
                    component=component,
                    operation=operation,
                    target_ms=target_ms,
                    actual_ms=-1,  # Error indicator
                    improvement_factor=0,
                    sample_count=0,
                    min_ms=0,
                    max_ms=0,
                    std_dev_ms=0,
                    confidence_95_lower=0,
                    confidence_95_upper=0
                ))
        
        # Calculate overall performance metrics
        if improvement_factors:
            overall_performance['average_improvement'] = statistics.mean(improvement_factors)
        
        # Validate specific performance claims from PR
        claims_validation = self.validate_performance_claims(results)
        overall_performance['performance_claims_validated'] = claims_validation
        
        validation_status = "VALIDATED" if overall_performance['benchmarks_passed'] >= 6 else "CONCERNS"
        
        benchmark_suite = BenchmarkSuite(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            results=results,
            overall_performance=overall_performance,
            validation_status=validation_status
        )
        
        self.print_summary_report(benchmark_suite)
        return benchmark_suite
    
    def validate_performance_claims(self, results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
        """Validate specific performance claims made in the PR."""
        claims = [
            {"claim": "Evolution analysis: 0.20ms (1000x faster)", "target": 0.20, "component": "Pattern Analysis"},
            {"claim": "Proposal generation: 0.04ms (2500x faster)", "target": 0.04, "component": "Proposal Generation"},
            {"claim": "Language validation: 0.02ms (2500x faster)", "target": 0.02, "component": "Language Validation"},
            {"claim": "End-to-end pipeline: 0.32ms (1200x faster)", "target": 0.32, "component": "End-to-End Pipeline"},
        ]
        
        validation_results = []
        
        for claim in claims:
            matching_result = next(
                (r for r in results if r.component == claim["component"]), 
                None
            )
            
            if matching_result and matching_result.actual_ms > 0:
                claim_validated = matching_result.actual_ms <= claim["target"] * 2  # Allow 2x margin
                validation_results.append({
                    "claim": claim["claim"],
                    "expected_ms": claim["target"],
                    "actual_ms": matching_result.actual_ms,
                    "validated": claim_validated,
                    "deviation_factor": matching_result.actual_ms / claim["target"]
                })
        
        return validation_results
    
    def print_summary_report(self, benchmark_suite: BenchmarkSuite):
        """Print a comprehensive summary report."""
        print("\n" + "=" * 70)
        print("🎯 LANGUAGE EVOLUTION SYSTEM - PERFORMANCE BENCHMARK REPORT")
        print("=" * 70)
        print(f"Timestamp: {benchmark_suite.timestamp}")
        print(f"Status: {benchmark_suite.validation_status}")
        
        passed = benchmark_suite.overall_performance['benchmarks_passed']
        total = benchmark_suite.overall_performance['total_benchmarks']
        avg_improvement = benchmark_suite.overall_performance['average_improvement']
        
        print(f"\n📈 Overall Performance:")
        print(f"   Benchmarks Passed: {passed}/{total}")
        print(f"   Average Improvement: {avg_improvement:.1f}x faster")
        
        print(f"\n🧪 Performance Claims Validation:")
        for claim in benchmark_suite.overall_performance['performance_claims_validated']:
            status = "✅" if claim['validated'] else "❌"
            print(f"   {status} {claim['claim']}")
            print(f"      Expected: {claim['expected_ms']:.3f}ms, Actual: {claim['actual_ms']:.3f}ms")
            
        print(f"\n📊 Detailed Results:")
        for result in benchmark_suite.results:
            if result.actual_ms > 0:
                print(f"   {result.component}: {result.actual_ms:.3f}ms ({result.improvement_factor:.1f}x)")
            
        print("\n" + "=" * 70)
    
    def save_results(self, benchmark_suite: BenchmarkSuite, filename: str):
        """Save benchmark results to JSON file."""
        # Convert dataclass to dict for JSON serialization
        data = {
            'timestamp': benchmark_suite.timestamp,
            'validation_status': benchmark_suite.validation_status,
            'overall_performance': benchmark_suite.overall_performance,
            'results': [asdict(result) for result in benchmark_suite.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Run the comprehensive benchmark suite."""
    print("🔬 OpusStrategist Language Evolution System - Performance Validation")
    print("Scientist Approach: Empirical measurement with statistical analysis")
    print("=" * 70)
    
    benchmark = LanguageEvolutionBenchmark()
    
    # Run comprehensive benchmarks
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results for analysis
    results_file = f"language_evolution_benchmark_results_{int(time.time())}.json"
    benchmark.save_results(results, results_file)
    
    return results


if __name__ == "__main__":
    main()