#!/usr/bin/env python3
"""
Coverage Validation Script
Prevents inaccurate coverage claims by validating actual coverage metrics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


class CoverageValidator:
    """Validates coverage metrics and prevents false quality claims."""
    
    def __init__(self, coverage_file: str = "coverage.json"):
        self.coverage_file = Path(coverage_file)
        self.coverage_data = self._load_coverage_data()
    
    def _load_coverage_data(self) -> Dict:
        """Load coverage data from JSON file."""
        if not self.coverage_file.exists():
            raise FileNotFoundError(f"Coverage file not found: {self.coverage_file}")
        
        with open(self.coverage_file) as f:
            return json.load(f)
    
    def get_component_coverage(self, component_pattern: str) -> Dict[str, float]:
        """Get coverage metrics for components matching pattern."""
        files = self.coverage_data.get("files", {})
        component_files = {
            path: data for path, data in files.items() 
            if component_pattern in path
        }
        
        if not component_files:
            raise ValueError(f"No files found matching pattern: {component_pattern}")
        
        coverage_results = {}
        for file_path, file_data in component_files.items():
            summary = file_data.get("summary", {})
            coverage_results[file_path] = summary.get("percent_covered", 0.0)
        
        return coverage_results
    
    def calculate_combined_coverage(self, component_pattern: str) -> Dict[str, float]:
        """Calculate combined coverage metrics for component group."""
        files = self.coverage_data.get("files", {})
        component_files = {
            path: data for path, data in files.items() 
            if component_pattern in path
        }
        
        if not component_files:
            raise ValueError(f"No files found matching pattern: {component_pattern}")
        
        total_statements = 0
        total_covered = 0
        
        for file_path, file_data in component_files.items():
            summary = file_data.get("summary", {})
            statements = summary.get("num_statements", 0)
            covered = summary.get("covered_lines", 0)
            
            total_statements += statements
            total_covered += covered
        
        combined_coverage = (total_covered / total_statements * 100) if total_statements > 0 else 0.0
        
        return {
            "total_statements": total_statements,
            "total_covered": total_covered,
            "combined_coverage": combined_coverage,
            "component_count": len(component_files)
        }
    
    def validate_coverage_claims(self, claims: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Validate coverage claims against actual metrics."""
        validation_results = []
        
        for claim in claims:
            component = claim.get("component")
            claimed_coverage = claim.get("claimed_coverage")
            tolerance = claim.get("tolerance", 5.0)  # 5% tolerance by default
            
            try:
                if claim.get("type") == "combined":
                    metrics = self.calculate_combined_coverage(component)
                    actual_coverage = metrics["combined_coverage"]
                else:
                    coverage_results = self.get_component_coverage(component)
                    actual_coverage = sum(coverage_results.values()) / len(coverage_results)
                
                difference = abs(claimed_coverage - actual_coverage)
                is_valid = difference <= tolerance
                
                validation_results.append({
                    "component": component,
                    "claimed_coverage": claimed_coverage,
                    "actual_coverage": actual_coverage,
                    "difference": difference,
                    "tolerance": tolerance,
                    "is_valid": is_valid,
                    "status": "PASS" if is_valid else "FAIL"
                })
                
            except Exception as e:
                validation_results.append({
                    "component": component,
                    "claimed_coverage": claimed_coverage,
                    "actual_coverage": None,
                    "difference": None,
                    "tolerance": tolerance,
                    "is_valid": False,
                    "status": "ERROR",
                    "error": str(e)
                })
        
        return validation_results
    
    def generate_report(self, validation_results: List[Dict[str, any]]) -> str:
        """Generate a validation report."""
        report = []
        report.append("Coverage Validation Report")
        report.append("=" * 50)
        report.append("")
        
        passed = sum(1 for r in validation_results if r["status"] == "PASS")
        failed = sum(1 for r in validation_results if r["status"] == "FAIL")
        errors = sum(1 for r in validation_results if r["status"] == "ERROR")
        
        report.append(f"Summary: {passed} PASS, {failed} FAIL, {errors} ERROR")
        report.append("")
        
        for result in validation_results:
            status = result["status"]
            component = result["component"]
            
            if status == "PASS":
                report.append(f"✅ {component}: {result['actual_coverage']:.2f}% (claimed: {result['claimed_coverage']:.2f}%)")
            elif status == "FAIL":
                report.append(f"❌ {component}: {result['actual_coverage']:.2f}% (claimed: {result['claimed_coverage']:.2f}%, diff: {result['difference']:.2f}%)")
            else:
                report.append(f"⚠️  {component}: ERROR - {result['error']}")
        
        return "\n".join(report)


def main():
    """Main validation function."""
    validator = CoverageValidator()
    
    # Define coverage claims to validate
    claims = [
        {
            "component": "pokemon_gym_adapter",
            "claimed_coverage": 95.0,
            "type": "combined",
            "tolerance": 5.0
        }
    ]
    
    # Validate claims
    results = validator.validate_coverage_claims(claims)
    
    # Generate and print report
    report = validator.generate_report(results)
    print(report)
    
    # Exit with error code if any validations failed
    failed_count = sum(1 for r in results if r["status"] in ["FAIL", "ERROR"])
    if failed_count > 0:
        print(f"\n❌ Coverage validation failed: {failed_count} issues found")
        sys.exit(1)
    else:
        print(f"\n✅ All coverage claims validated successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()