#!/usr/bin/env python3
"""
PR #232 Final Validation with Comprehensive Production Patterns

Run the staged validation harness with the enhanced production patterns
to get comprehensive validation results.
"""

import json
from pathlib import Path
from pr232_staged_validation_harness import ProductionPattern, ProductionValidationHarness


def load_comprehensive_patterns() -> list:
    """Load comprehensive production patterns from JSON file."""
    pattern_file = Path("comprehensive_production_patterns.json")
    
    if not pattern_file.exists():
        print("❌ Comprehensive patterns file not found. Run pr232_enhanced_pattern_extractor.py first.")
        return []
    
    with open(pattern_file, 'r') as f:
        pattern_data = json.load(f)
    
    production_patterns = []
    for data in pattern_data:
        pattern = ProductionPattern(
            name=data["name"],
            success_rate=data["success_rate"],
            usage_frequency=data["usage_frequency"],
            input_sequence=data["input_sequence"],
            context=data["context"],
            source=data["source"],
            evolution_metadata=data["evolution_metadata"]
        )
        production_patterns.append(pattern)
    
    return production_patterns


def main():
    """Run final comprehensive validation."""
    print("🔬 PR #232 Final Validation with Comprehensive Production Patterns")
    print("="*80)
    
    # Load comprehensive patterns
    production_patterns = load_comprehensive_patterns()
    
    if not production_patterns:
        return
    
    print(f"📥 Loaded {len(production_patterns)} comprehensive production patterns")
    
    # Run comprehensive validation
    harness = ProductionValidationHarness(production_patterns)
    report = harness.run_comprehensive_validation()
    
    # Save final results
    results_file = f"pr232_final_validation_report_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        from dataclasses import asdict
        import time
        
        report_dict = {
            'timestamp': report.timestamp,
            'production_patterns_extracted': report.production_patterns_extracted,
            'validation_results': [asdict(r) for r in report.validation_results],
            'stress_test_result': asdict(report.stress_test_result) if report.stress_test_result else None,
            'performance_claims_validated': report.performance_claims_validated,
            'overall_status': report.overall_status,
            'recommendations': report.recommendations
        }
        json.dump(report_dict, f, indent=2)
    
    print(f"\n💾 Final validation results saved to: {results_file}")
    
    # Print final summary
    print("\n" + "="*80)
    print("🎯 PR #232 FINAL VALIDATION SUMMARY")
    print("="*80)
    
    successful_validations = sum(1 for r in report.validation_results if r.success)
    total_validations = len(report.validation_results)
    
    print(f"Production Patterns: {report.production_patterns_extracted}")
    print(f"Performance Tests: {successful_validations}/{total_validations} passed")
    print(f"Overall Status: {report.overall_status}")
    
    if report.stress_test_result:
        throughput_percent = (report.stress_test_result.actual_throughput / 
                            report.stress_test_result.target_throughput) * 100
        print(f"Stress Test: {throughput_percent:.1f}% of target throughput")
    
    print("\nKey Performance Results:")
    for result in report.validation_results:
        if result.success and result.actual_ms > 0:
            print(f"  ✅ {result.component}: {result.actual_ms:.3f}ms ({result.improvement_factor:.0f}x improvement)")
        elif not result.success:
            print(f"  ❌ {result.component}: Failed validation")
    
    print("\n" + "="*80)
    
    return report


if __name__ == "__main__":
    import time
    main()