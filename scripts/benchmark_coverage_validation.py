#!/usr/bin/env python3
"""
Performance benchmark for coverage validation system.
Measures impact on development workflow.
"""

import time
import json
import tempfile
from pathlib import Path
import subprocess
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from validate_coverage import CoverageValidator


def create_large_coverage_dataset(num_files: int = 100) -> str:
    """Create large coverage dataset for performance testing."""
    coverage_data = {
        "files": {
            f"src/component_{i}.py": {
                "summary": {
                    "covered_lines": 45 + (i % 55),  # Vary coverage
                    "num_statements": 100,
                    "percent_covered": 45.0 + (i % 55)
                }
            } for i in range(num_files)
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coverage_data, f)
        return f.name


def benchmark_validation_performance():
    """Benchmark validation system performance."""
    print("🚀 Coverage Validation Performance Benchmark")
    print("=" * 50)
    
    file_counts = [10, 50, 100, 200, 500]
    results = {}
    
    for num_files in file_counts:
        print(f"📊 Testing with {num_files} files...")
        
        # Create test data
        temp_file = create_large_coverage_dataset(num_files)
        
        try:
            # Benchmark loading
            start_time = time.time()
            validator = CoverageValidator(temp_file)
            load_time = time.time() - start_time
            
            # Benchmark component coverage extraction
            start_time = time.time()
            coverage = validator.get_component_coverage("component")
            extract_time = time.time() - start_time
            
            # Benchmark validation
            claims = [
                {
                    "component": "component",
                    "claimed_coverage": 70.0,
                    "type": "combined",
                    "tolerance": 5.0
                }
            ]
            
            start_time = time.time()
            validation_results = validator.validate_coverage_claims(claims)
            validate_time = time.time() - start_time
            
            # Benchmark report generation
            start_time = time.time()
            report = validator.generate_report(validation_results)
            report_time = time.time() - start_time
            
            total_time = load_time + extract_time + validate_time + report_time
            
            results[num_files] = {
                "load_time": load_time,
                "extract_time": extract_time,
                "validate_time": validate_time,
                "report_time": report_time,
                "total_time": total_time,
                "files_found": len(coverage),
                "validation_status": validation_results[0]["status"]
            }
            
            print(f"   ✅ Total time: {total_time:.3f}s ({len(coverage)} files processed)")
            
        finally:
            # Clean up
            Path(temp_file).unlink()
    
    return results


def benchmark_ci_pipeline_impact():
    """Benchmark CI pipeline integration impact."""
    print("\n🏗️ CI Pipeline Integration Benchmark")
    print("=" * 50)
    
    # Measure baseline test execution
    print("📊 Measuring baseline test execution...")
    start_time = time.time()
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "tests/test_coverage_validation.py", "-q"],
            capture_output=True,
            text=True,
            timeout=60
        )
        baseline_time = time.time() - start_time
        test_success = result.returncode == 0
        
        print(f"   ✅ Baseline test time: {baseline_time:.2f}s")
        print(f"   ✅ Tests passed: {test_success}")
        
    except subprocess.TimeoutExpired:
        baseline_time = 60.0
        test_success = False
        print(f"   ⚠️ Tests timed out after 60s")
    
    # Measure coverage generation
    print("📊 Measuring coverage generation...")
    start_time = time.time()
    try:
        subprocess.run(
            ["coverage", "run", "-m", "pytest", "tests/test_coverage_validation.py", "-q"],
            capture_output=True,
            timeout=60
        )
        subprocess.run(
            ["coverage", "json"],
            capture_output=True,
            timeout=30
        )
        coverage_time = time.time() - start_time
        
        print(f"   ✅ Coverage generation time: {coverage_time:.2f}s")
        
    except subprocess.TimeoutExpired:
        coverage_time = 60.0
        print(f"   ⚠️ Coverage generation timed out")
    
    # Measure validation execution
    print("📊 Measuring validation execution...")
    start_time = time.time()
    try:
        result = subprocess.run(
            ["python3", "scripts/validate_coverage.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        validation_time = time.time() - start_time
        validation_success = result.returncode in [0, 1]  # 0 = pass, 1 = fail (expected)
        
        print(f"   ✅ Validation time: {validation_time:.2f}s")
        print(f"   ✅ Validation executed: {validation_success}")
        
    except subprocess.TimeoutExpired:
        validation_time = 30.0
        validation_success = False
        print(f"   ⚠️ Validation timed out")
    
    total_overhead = coverage_time + validation_time
    
    return {
        "baseline_test_time": baseline_time,
        "coverage_time": coverage_time,
        "validation_time": validation_time,
        "total_overhead": total_overhead,
        "test_success": test_success,
        "validation_success": validation_success
    }


def generate_performance_report(validation_results, ci_results):
    """Generate comprehensive performance report."""
    print("\n📈 Performance Report")
    print("=" * 50)
    
    print("🔍 Validation System Performance:")
    print("-" * 30)
    
    for num_files, metrics in validation_results.items():
        throughput = num_files / metrics["total_time"]
        print(f"  {num_files:3d} files: {metrics['total_time']:.3f}s ({throughput:.1f} files/sec)")
    
    # Find largest dataset performance
    max_files = max(validation_results.keys())
    max_performance = validation_results[max_files]
    
    print(f"\n🎯 Maximum Performance Validated:")
    print(f"  Files processed: {max_files}")
    print(f"  Total time: {max_performance['total_time']:.3f}s")
    print(f"  Throughput: {max_files / max_performance['total_time']:.1f} files/sec")
    
    print("\n🏗️ CI Pipeline Impact:")
    print("-" * 30)
    print(f"  Baseline test execution: {ci_results['baseline_test_time']:.2f}s")
    print(f"  Coverage generation: {ci_results['coverage_time']:.2f}s")  
    print(f"  Coverage validation: {ci_results['validation_time']:.2f}s")
    print(f"  Total added overhead: {ci_results['total_overhead']:.2f}s")
    
    overhead_percentage = (ci_results['total_overhead'] / ci_results['baseline_test_time']) * 100
    print(f"  Relative overhead: {overhead_percentage:.1f}%")
    
    print("\n✅ Performance Assessment:")
    print("-" * 30)
    
    # Performance thresholds
    validation_acceptable = max_performance['total_time'] < 2.0  # < 2 seconds for 500 files
    ci_acceptable = ci_results['total_overhead'] < 60.0  # < 60 seconds overhead
    
    print(f"  Validation system speed: {'✅ EXCELLENT' if validation_acceptable else '⚠️ NEEDS OPTIMIZATION'}")
    print(f"  CI pipeline impact: {'✅ ACCEPTABLE' if ci_acceptable else '⚠️ HIGH IMPACT'}")
    
    if validation_acceptable and ci_acceptable:
        print(f"  Overall assessment: ✅ PERFORMANCE TARGETS MET")
    else:
        print(f"  Overall assessment: ⚠️ OPTIMIZATION NEEDED")
    
    return {
        "validation_acceptable": validation_acceptable,
        "ci_acceptable": ci_acceptable,
        "max_throughput": max_files / max_performance['total_time'],
        "ci_overhead_seconds": ci_results['total_overhead'],
        "ci_overhead_percentage": overhead_percentage
    }


def main():
    """Main benchmark execution."""
    print("🚀 Quality Process Enhancement - Performance Benchmark")
    print("=" * 60)
    print("Testing coverage validation system performance and CI impact...")
    print()
    
    # Run validation performance benchmark
    validation_results = benchmark_validation_performance()
    
    # Run CI pipeline benchmark
    ci_results = benchmark_ci_pipeline_impact()
    
    # Generate comprehensive report
    assessment = generate_performance_report(validation_results, ci_results)
    
    print("\n📊 Benchmark Complete!")
    print("=" * 60)
    
    if assessment["validation_acceptable"] and assessment["ci_acceptable"]:
        print("🎉 All performance targets met! System ready for production.")
        return 0
    else:
        print("⚠️ Performance optimization recommended before full deployment.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)