#!/usr/bin/env python3
"""
Validation script for the newly implemented analyze_parallel_results method.

Tests the core functionality without requiring full integration setup.
This validates Phase 1 & 2 implementation with statistical rigor.
"""

import sys
import time
from unittest.mock import Mock

# Add the source directory to the Python path
sys.path.insert(0, '/workspace/repo/src')

def create_test_parallel_results():
    """Create realistic test data for parallel execution results."""
    return [
        {
            "worker_id": "worker_1",
            "success": True,
            "execution_time": 1.23,
            "actions_taken": ["A", "B", "START", "A", "RIGHT"],
            "final_state": {"x": 10, "y": 5, "level": 3},
            "performance_metrics": {"frame_rate": 60, "input_lag": 16.7},
            "discovered_patterns": ["menu_optimization", "movement_sequence"],
        },
        {
            "worker_id": "worker_2",
            "success": True,
            "execution_time": 1.45,
            "actions_taken": ["A", "B", "START", "B", "RIGHT"],
            "final_state": {"x": 12, "y": 5, "level": 3},
            "performance_metrics": {"frame_rate": 59, "input_lag": 18.2},
            "discovered_patterns": ["menu_optimization", "alternate_sequence"],
        },
        {
            "worker_id": "worker_3",
            "success": False,
            "execution_time": 2.1,
            "actions_taken": ["A", "START", "LEFT", "A"],
            "final_state": {"x": 8, "y": 4, "level": 2},
            "performance_metrics": {"frame_rate": 45, "input_lag": 22.1},
            "discovered_patterns": ["failed_sequence"],
        },
        {
            "worker_id": "worker_4",
            "success": True,
            "execution_time": 1.18,
            "actions_taken": ["B", "A", "START", "A", "RIGHT"],
            "final_state": {"x": 11, "y": 5, "level": 3},
            "performance_metrics": {"frame_rate": 61, "input_lag": 15.9},
            "discovered_patterns": ["menu_optimization", "speed_optimization"],
        },
    ]

def test_analyze_parallel_results():
    """Test the analyze_parallel_results method implementation."""
    print("Testing analyze_parallel_results implementation...")
    
    # Mock the Claude Code Manager
    mock_manager = Mock()
    mock_strategic_process = Mock()
    mock_manager.get_strategic_process.return_value = mock_strategic_process
    
    # Mock Opus response
    mock_response = """{
        "identified_patterns": [
            {
                "pattern": "menu_optimization",
                "frequency": 3,
                "success_correlation": 1.0,
                "performance_impact": "reduces execution time by 15%"
            }
        ],
        "correlations": [
            {
                "variables": ["frame_rate", "execution_time"],
                "correlation": -0.73,
                "significance": "strong negative correlation"
            }
        ],
        "strategic_insights": [
            "Menu optimization pattern shows consistent 100% success rate",
            "Performance correlation indicates frame_rate > 58 improves success"
        ],
        "optimization_opportunities": [
            "Focus on menu optimization patterns for consistent performance",
            "Monitor frame rate to predict execution success"
        ],
        "risk_factors": [
            "Left movement patterns correlate with failures - avoid in speedruns"
        ]
    }"""
    
    mock_strategic_process.send_message.return_value = mock_response
    
    # Import and initialize OpusStrategist
    from claudelearnspokemon.opus_strategist import OpusStrategist
    
    strategist = OpusStrategist(mock_manager)
    
    # Create test data
    parallel_results = create_test_parallel_results()
    
    print(f"Testing with {len(parallel_results)} parallel results...")
    
    # Measure performance
    start_time = time.time()
    
    try:
        # Execute analyze_parallel_results
        analysis_results = strategist.analyze_parallel_results(parallel_results)
        
        execution_time = (time.time() - start_time) * 1000
        
        print(f"✓ Analysis completed in {execution_time:.2f}ms")
        
        # Validate results structure
        assert isinstance(analysis_results, list), "Results should be a list"
        assert len(analysis_results) >= 3, "Should have at least 3 analysis types"
        
        print(f"✓ Returned {len(analysis_results)} analysis types")
        
        # Check analysis types
        analysis_types = [result["analysis_type"] for result in analysis_results]
        expected_types = ["pattern_identification", "statistical_correlation", "strategic_insights"]
        
        for expected_type in expected_types:
            assert expected_type in analysis_types, f"Missing analysis type: {expected_type}"
        
        print("✓ All expected analysis types present")
        
        # Validate pattern identification results
        pattern_result = next(r for r in analysis_results if r["analysis_type"] == "pattern_identification")
        pattern_data = pattern_result["results"]
        
        assert "high_frequency_patterns" in pattern_data
        assert "high_success_patterns" in pattern_data
        assert "problematic_patterns" in pattern_data
        assert "performance_patterns" in pattern_data
        
        print("✓ Pattern identification structure validated")
        
        # Validate statistical correlation results
        correlation_result = next(r for r in analysis_results if r["analysis_type"] == "statistical_correlation")
        correlation_data = correlation_result["results"]
        
        assert "significant_correlations" in correlation_data
        assert "performance_correlations" in correlation_data
        assert "success_correlations" in correlation_data
        assert "worker_correlations" in correlation_data
        
        print("✓ Statistical correlation structure validated")
        
        # Validate strategic insights
        insights_result = next(r for r in analysis_results if r["analysis_type"] == "strategic_insights")
        insights_data = insights_result["results"]
        
        assert "identified_patterns" in insights_data
        assert "correlations" in insights_data
        assert "strategic_insights" in insights_data
        assert "optimization_opportunities" in insights_data
        assert "risk_factors" in insights_data
        
        print("✓ Strategic insights structure validated")
        
        # Performance validation
        if execution_time > 200.0:
            print(f"⚠ Performance target missed: {execution_time:.2f}ms > 200ms target")
        else:
            print(f"✓ Performance target met: {execution_time:.2f}ms < 200ms target")
        
        # Validate Claude integration
        mock_strategic_process.send_message.assert_called_once()
        call_args = mock_strategic_process.send_message.call_args[0][0]
        
        assert "PARALLEL RESULTS STRATEGIC ANALYSIS" in call_args
        assert "PATTERN ANALYSIS RESULTS" in call_args
        assert "STATISTICAL CORRELATION ANALYSIS" in call_args
        
        print("✓ Claude Opus integration validated")
        
        # Print summary statistics
        print("\n" + "="*50)
        print("ANALYSIS RESULTS SUMMARY")
        print("="*50)
        
        for result in analysis_results:
            analysis_type = result["analysis_type"]
            processing_time = result["processing_time_ms"]
            
            print(f"{analysis_type}: {processing_time:.2f}ms")
            
            if analysis_type == "pattern_identification":
                data = result["results"]
                high_freq = len(data.get("high_frequency_patterns", []))
                high_success = len(data.get("high_success_patterns", []))
                problematic = len(data.get("problematic_patterns", []))
                
                print(f"  - High-frequency patterns: {high_freq}")
                print(f"  - High-success patterns: {high_success}")
                print(f"  - Problematic patterns: {problematic}")
                
            elif analysis_type == "statistical_correlation":
                data = result["results"]
                correlations = len(data.get("significant_correlations", []))
                print(f"  - Significant correlations: {correlations}")
                
            elif analysis_type == "strategic_insights":
                data = result["results"]
                insights = len(data.get("strategic_insights", []))
                opportunities = len(data.get("optimization_opportunities", []))
                risks = len(data.get("risk_factors", []))
                
                print(f"  - Strategic insights: {insights}")
                print(f"  - Optimization opportunities: {opportunities}")
                print(f"  - Risk factors: {risks}")
        
        print("\n✓ ALL TESTS PASSED - analyze_parallel_results implementation validated!")
        return True
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        print(f"✗ Test failed after {execution_time:.2f}ms: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_behavior():
    """Test fallback behavior when Claude Opus is unavailable."""
    print("\nTesting fallback behavior...")
    
    # Mock the Claude Code Manager to simulate failure
    mock_manager = Mock()
    mock_manager.get_strategic_process.return_value = None  # Simulate no strategic process
    
    from claudelearnspokemon.opus_strategist import OpusStrategist
    
    strategist = OpusStrategist(mock_manager)
    
    # Create test data
    parallel_results = create_test_parallel_results()
    
    try:
        # Execute analyze_parallel_results with fallback
        analysis_results = strategist.analyze_parallel_results(parallel_results)
        
        # Validate fallback results
        assert isinstance(analysis_results, list), "Fallback should still return results list"
        assert len(analysis_results) >= 3, "Fallback should have at least 3 analysis types"
        
        # Check strategic insights has fallback metadata
        insights_result = next(r for r in analysis_results if r["analysis_type"] == "strategic_insights")
        insights_data = insights_result["results"]
        
        assert "metadata" in insights_data
        assert insights_data["metadata"].get("analysis_type") == "fallback"
        
        print("✓ Fallback behavior validated")
        return True
        
    except Exception as e:
        print(f"✗ Fallback test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("analyze_parallel_results Implementation Validation")
    print("=" * 60)
    
    success = test_analyze_parallel_results()
    if success:
        success = test_fallback_behavior()
    
    if success:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("The analyze_parallel_results implementation is working correctly.")
    else:
        print("\n❌ VALIDATION TESTS FAILED!")
        print("Implementation needs fixes before proceeding.")
        sys.exit(1)