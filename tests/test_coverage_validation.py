#!/usr/bin/env python3
"""
Comprehensive tests for the Coverage Validation System.
Ensures reliable quality process enforcement.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

# Import our validation system
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from validate_coverage import CoverageValidator


class TestCoverageValidator:
    """Test suite for coverage validation system."""
    
    @pytest.fixture
    def sample_coverage_data(self):
        """Sample coverage data for testing."""
        return {
            "files": {
                "src/pokemon_gym_adapter.py": {
                    "summary": {
                        "covered_lines": 298,
                        "num_statements": 447,
                        "percent_covered": 66.85
                    }
                },
                "src/pokemon_gym_adapter_exceptions.py": {
                    "summary": {
                        "covered_lines": 49,
                        "num_statements": 146,
                        "percent_covered": 33.52
                    }
                },
                "src/pokemon_gym_adapter_types.py": {
                    "summary": {
                        "covered_lines": 186,
                        "num_statements": 186,
                        "percent_covered": 100.0
                    }
                },
                "src/pokemon_gym_factory.py": {
                    "summary": {
                        "covered_lines": 129,
                        "num_statements": 140,
                        "percent_covered": 13.37
                    }
                },
                "src/other_component.py": {
                    "summary": {
                        "covered_lines": 95,
                        "num_statements": 100,
                        "percent_covered": 95.0
                    }
                }
            }
        }
    
    @pytest.fixture
    def temp_coverage_file(self, sample_coverage_data):
        """Create temporary coverage file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_coverage_data, f)
            return f.name
    
    def test_load_coverage_data_success(self, temp_coverage_file):
        """Test successful loading of coverage data."""
        validator = CoverageValidator(temp_coverage_file)
        assert validator.coverage_data is not None
        assert "files" in validator.coverage_data
        
        # Clean up
        Path(temp_coverage_file).unlink()
    
    def test_load_coverage_data_file_not_found(self):
        """Test error handling when coverage file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            CoverageValidator("nonexistent_file.json")
    
    def test_get_component_coverage_pokemon_gym(self, temp_coverage_file):
        """Test getting coverage for pokemon_gym components."""
        validator = CoverageValidator(temp_coverage_file)
        coverage = validator.get_component_coverage("pokemon_gym")
        
        # Should find 4 pokemon_gym files
        assert len(coverage) == 4
        assert "src/pokemon_gym_adapter.py" in coverage
        assert "src/pokemon_gym_adapter_exceptions.py" in coverage
        assert "src/pokemon_gym_adapter_types.py" in coverage
        assert "src/pokemon_gym_factory.py" in coverage
        
        # Check specific values
        assert coverage["src/pokemon_gym_adapter.py"] == 66.85
        assert coverage["src/pokemon_gym_adapter_types.py"] == 100.0
        
        # Clean up
        Path(temp_coverage_file).unlink()
    
    def test_get_component_coverage_no_match(self, temp_coverage_file):
        """Test error when no components match pattern."""
        validator = CoverageValidator(temp_coverage_file)
        
        with pytest.raises(ValueError, match="No files found matching pattern"):
            validator.get_component_coverage("nonexistent_component")
        
        # Clean up
        Path(temp_coverage_file).unlink()
    
    def test_calculate_combined_coverage(self, temp_coverage_file):
        """Test combined coverage calculation for pokemon_gym components."""
        validator = CoverageValidator(temp_coverage_file)
        result = validator.calculate_combined_coverage("pokemon_gym")
        
        # Expected totals: 662 covered out of 919 statements = 72.03%
        assert result["total_statements"] == 919
        assert result["total_covered"] == 662
        assert abs(result["combined_coverage"] - 72.03) < 0.05
        assert result["component_count"] == 4
        
        # Clean up
        Path(temp_coverage_file).unlink()
    
    def test_validate_coverage_claims_pass(self, temp_coverage_file):
        """Test validation that passes within tolerance."""
        validator = CoverageValidator(temp_coverage_file)
        
        claims = [
            {
                "component": "pokemon_gym",
                "claimed_coverage": 70.0,
                "type": "combined",
                "tolerance": 5.0
            }
        ]
        
        results = validator.validate_coverage_claims(claims)
        assert len(results) == 1
        assert results[0]["status"] == "PASS"
        assert results[0]["is_valid"] is True
        assert abs(results[0]["actual_coverage"] - 72.03) < 0.05
        
        # Clean up
        Path(temp_coverage_file).unlink()
    
    def test_validate_coverage_claims_fail(self, temp_coverage_file):
        """Test validation that fails outside tolerance."""
        validator = CoverageValidator(temp_coverage_file)
        
        claims = [
            {
                "component": "pokemon_gym",
                "claimed_coverage": 95.0,
                "type": "combined",
                "tolerance": 5.0
            }
        ]
        
        results = validator.validate_coverage_claims(claims)
        assert len(results) == 1
        assert results[0]["status"] == "FAIL"
        assert results[0]["is_valid"] is False
        assert abs(results[0]["difference"] - 22.97) < 0.05
        
        # Clean up
        Path(temp_coverage_file).unlink()
    
    def test_validate_coverage_claims_error(self, temp_coverage_file):
        """Test validation error handling."""
        validator = CoverageValidator(temp_coverage_file)
        
        claims = [
            {
                "component": "nonexistent_component",
                "claimed_coverage": 95.0,
                "type": "combined",
                "tolerance": 5.0
            }
        ]
        
        results = validator.validate_coverage_claims(claims)
        assert len(results) == 1
        assert results[0]["status"] == "ERROR"
        assert results[0]["is_valid"] is False
        assert "error" in results[0]
        
        # Clean up
        Path(temp_coverage_file).unlink()
    
    def test_validate_individual_component_coverage(self, temp_coverage_file):
        """Test validation of individual component (not combined)."""
        validator = CoverageValidator(temp_coverage_file)
        
        claims = [
            {
                "component": "other_component",
                "claimed_coverage": 95.0,
                "tolerance": 1.0
            }
        ]
        
        results = validator.validate_coverage_claims(claims)
        assert len(results) == 1
        assert results[0]["status"] == "PASS"
        assert abs(results[0]["actual_coverage"] - 95.0) < 0.01
        
        # Clean up
        Path(temp_coverage_file).unlink()
    
    def test_generate_report_mixed_results(self, temp_coverage_file):
        """Test report generation with mixed pass/fail results."""
        validator = CoverageValidator(temp_coverage_file)
        
        claims = [
            {
                "component": "other_component",
                "claimed_coverage": 95.0,
                "tolerance": 1.0
            },
            {
                "component": "pokemon_gym",
                "claimed_coverage": 95.0,
                "type": "combined",
                "tolerance": 5.0
            }
        ]
        
        results = validator.validate_coverage_claims(claims)
        report = validator.generate_report(results)
        
        assert "1 PASS, 1 FAIL, 0 ERROR" in report
        assert "✅" in report
        assert "❌" in report
        assert "other_component: 95.00%" in report
        assert "pokemon_gym: 72.03%" in report
        
        # Clean up
        Path(temp_coverage_file).unlink()
    
    def test_custom_tolerance_levels(self, temp_coverage_file):
        """Test custom tolerance levels."""
        validator = CoverageValidator(temp_coverage_file)
        
        # Test with very strict tolerance (should fail)
        claims = [
            {
                "component": "pokemon_gym",
                "claimed_coverage": 72.0,
                "type": "combined",
                "tolerance": 0.01  # Very strict
            }
        ]
        
        results = validator.validate_coverage_claims(claims)
        assert results[0]["status"] == "FAIL"  # 72.02% vs 72.00% with 0.01% tolerance
        
        # Test with lenient tolerance (should pass)
        claims[0]["tolerance"] = 1.0  # Lenient
        results = validator.validate_coverage_claims(claims)
        assert results[0]["status"] == "PASS"
        
        # Clean up
        Path(temp_coverage_file).unlink()


class TestCoverageValidationIntegration:
    """Integration tests for the complete validation system."""
    
    @pytest.mark.integration
    def test_end_to_end_validation_workflow(self):
        """Test the complete validation workflow."""
        # This test would require actual coverage data
        # For now, we test the integration points
        
        # Test that the script can be imported and run
        from validate_coverage import main
        
        # Test would run: coverage -> json -> validation -> report
        # In actual implementation, this would use real coverage data
        assert callable(main)
    
    @pytest.mark.integration
    def test_ci_cd_integration_points(self):
        """Test CI/CD integration points."""
        # Verify that validation script returns correct exit codes
        # This would be tested in actual CI/CD environment
        pass
    
    @pytest.mark.performance
    def test_validation_performance(self):
        """Test performance of validation system."""
        # Create large sample data
        large_coverage_data = {
            "files": {
                f"src/component_{i}.py": {
                    "summary": {
                        "covered_lines": 50,
                        "num_statements": 100,
                        "percent_covered": 50.0
                    }
                } for i in range(100)
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(large_coverage_data, f)
            temp_file = f.name
        
        import time
        start_time = time.time()
        
        validator = CoverageValidator(temp_file)
        coverage = validator.get_component_coverage("component")
        
        end_time = time.time()
        performance_time = end_time - start_time
        
        # Should process 100 files in reasonable time (< 1 second)
        assert len(coverage) == 100
        assert performance_time < 1.0
        
        # Clean up
        Path(temp_file).unlink()


@pytest.mark.integration
def test_main_function_behavior():
    """Test the main function behavior with mocked coverage data."""
    # Test that main function correctly processes claims and exits
    sample_data = {
        "files": {
            "src/pokemon_gym_adapter.py": {
                "summary": {
                    "covered_lines": 298,
                    "num_statements": 447,
                    "percent_covered": 66.85
                }
            }
        }
    }
    
    with patch('builtins.open', mock_open(read_data=json.dumps(sample_data))):
        with patch('pathlib.Path.exists', return_value=True):
            with patch('sys.exit') as mock_exit:
                try:
                    from validate_coverage import main
                    main()
                    # Should exit with code 1 due to coverage mismatch
                    mock_exit.assert_called_with(1)
                except SystemExit:
                    pass