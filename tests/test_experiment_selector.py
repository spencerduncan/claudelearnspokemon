"""
Comprehensive test suite for ExperimentSelector variation generation

Tests designed by worker1 (Rex) to validate:
1. Basic functionality works as expected
2. Edge cases are handled gracefully  
3. Error conditions don't crash the system
4. Performance requirements are met
5. Validation catches invalid experiments

Author: worker1 (Rex) - Skeptical testing approach
"""

import pytest
import time
from unittest.mock import Mock, patch
from collections import defaultdict

from src.claudelearnspokemon.experiment_selector import (
    ExperimentSelector,
    VariationConfig,
    VariationStrategy, 
    ConstraintViolationType,
    ValidationResult,
    ExperimentSpec,
    ParameterBounds,
    VariationValidator,
    ConservativeVariationGenerator,
    create_experiment_selector
)


class TestExperimentSelectorBasics:
    """Test basic ExperimentSelector functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = VariationConfig(
            max_variations_per_batch=10,
            max_generation_time_ms=50.0,
            min_variation_distance=0.1
        )
        self.selector = ExperimentSelector(self.config)
    
    def test_initialization(self):
        """Test ExperimentSelector initializes correctly"""
        assert self.selector is not None
        assert self.selector.config == self.config
        assert len(self.selector._experiment_queue) == 0
        assert len(self.selector._experiment_history) == 0
        assert isinstance(self.selector.validator, VariationValidator)
    
    def test_factory_function(self):
        """Test factory function creates valid selector"""
        selector = create_experiment_selector()
        assert isinstance(selector, ExperimentSelector)
        assert selector.config is not None
    
    def test_add_valid_experiment(self):
        """Test adding valid experiment to queue"""
        experiment = {
            'experiment_id': 'test_exp_1',
            'parameters': {'movement_sequence_length': 10},
            'strategy_type': 'navigation',
            'dsl_script': 'A B START',
            'estimated_duration': 60.0
        }
        
        self.selector.add_experiment(experiment, priority=1.0)
        assert len(self.selector._experiment_queue) == 1
        assert 'test_exp_1' in self.selector._experiment_history
    
    def test_add_duplicate_experiment_rejected(self):
        """Test duplicate experiments are rejected"""
        experiment = {
            'experiment_id': 'duplicate_test',
            'parameters': {},
            'strategy_type': 'navigation', 
            'dsl_script': 'A B',
            'estimated_duration': 30.0
        }
        
        self.selector.add_experiment(experiment, priority=1.0)
        self.selector.add_experiment(experiment, priority=2.0)  # Duplicate
        
        assert len(self.selector._experiment_queue) == 1  # Only one added


class TestVariationGeneration:
    """Test variation generation functionality"""
    
    def setup_method(self):
        self.selector = ExperimentSelector()
        self.base_experiment = {
            'experiment_id': 'base_exp',
            'parameters': {
                'movement_sequence_length': 20,
                'pathfind_distance': 50,
                'interaction_attempts': 3
            },
            'strategy_type': 'navigation',
            'dsl_script': 'UP DOWN LEFT RIGHT PATHFIND_TO 10 15',
            'estimated_duration': 120.0
        }
    
    def test_generate_variations_basic(self):
        """Test basic variation generation works"""
        variations = self.selector.generate_variations(self.base_experiment)
        
        assert isinstance(variations, list)
        assert len(variations) > 0
        assert len(variations) <= self.selector.config.max_variations_per_batch
        
        # All variations should have different IDs
        ids = [v['experiment_id'] for v in variations]
        assert len(set(ids)) == len(ids)
    
    def test_generate_variations_respects_time_limit(self):
        """Test variation generation respects time limits"""
        config = VariationConfig(max_generation_time_ms=1.0)  # Very short limit
        selector = ExperimentSelector(config)
        
        start_time = time.perf_counter()
        variations = selector.generate_variations(self.base_experiment)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Should respect time limit (with some tolerance)
        assert duration_ms < 50.0  # Generous tolerance for test environment
    
    def test_generate_variations_with_invalid_base(self):
        """Test variation generation handles invalid base experiment gracefully"""
        invalid_experiment = {
            'experiment_id': '',  # Invalid empty ID
            'parameters': 'not_a_dict',  # Invalid type
            'strategy_type': 'navigation',
            'dsl_script': '',  # Empty script
            'estimated_duration': -1.0  # Invalid duration
        }
        
        variations = self.selector.generate_variations(invalid_experiment)
        
        # Should handle gracefully, possibly returning fallback variations
        assert isinstance(variations, list)
        # Rex: System should not crash, even with garbage input
    
    def test_generate_variations_performance_tracking(self):
        """Test variation generation tracks performance statistics"""
        initial_stats = self.selector.get_generation_stats()
        initial_calls = initial_stats.get('total_calls', 0)
        
        self.selector.generate_variations(self.base_experiment)
        
        final_stats = self.selector.get_generation_stats()
        assert final_stats['total_calls'] == initial_calls + 1
        assert final_stats['total_variations'] >= 0
        assert final_stats['total_time_ms'] > 0


class TestValidationSystem:
    """Test the comprehensive validation system"""
    
    def setup_method(self):
        self.validator = VariationValidator()
    
    def test_valid_experiment_passes_validation(self):
        """Test valid experiment passes all validation layers"""
        experiment = ExperimentSpec(
            experiment_id='valid_exp',
            parameters={'movement_sequence_length': 15},
            strategy_type='navigation',
            dsl_script='A B UP DOWN',
            estimated_duration=60.0
        )
        
        result = self.validator.validate_variation(experiment)
        assert result.is_valid is True
        assert len(result.violations) == 0
    
    def test_invalid_structure_detected(self):
        """Test invalid experiment structure is detected"""
        experiment = ExperimentSpec(
            experiment_id='',  # Invalid empty ID
            parameters='not_dict',  # Invalid type
            strategy_type='navigation',
            dsl_script='',  # Empty script
            estimated_duration=60.0
        )
        
        result = self.validator.validate_variation(experiment)
        assert result.is_valid is False
        assert ConstraintViolationType.INVALID_COMMAND in result.violations
    
    def test_invalid_dsl_commands_detected(self):
        """Test invalid DSL commands are detected"""
        experiment = ExperimentSpec(
            experiment_id='dsl_test',
            parameters={},
            strategy_type='navigation',
            dsl_script='INVALID_COMMAND DANGEROUS_OPERATION',
            estimated_duration=60.0
        )
        
        result = self.validator.validate_variation(experiment)
        assert result.is_valid is False
        assert ConstraintViolationType.INVALID_COMMAND in result.violations
    
    def test_parameter_bounds_validation(self):
        """Test parameter bounds are enforced"""
        experiment = ExperimentSpec(
            experiment_id='bounds_test',
            parameters={'movement_sequence_length': 1000},  # Exceeds max bound of 50
            strategy_type='navigation',
            dsl_script='A B',
            estimated_duration=60.0
        )
        
        result = self.validator.validate_variation(experiment)
        assert result.is_valid is False
        assert ConstraintViolationType.PARAMETER_OUT_OF_BOUNDS in result.violations
    
    def test_performance_violations_detected(self):
        """Test performance violations are caught"""
        experiment = ExperimentSpec(
            experiment_id='perf_test',
            parameters={},
            strategy_type='navigation',
            dsl_script='A B',
            estimated_duration=500.0  # Exceeds 300s limit
        )
        
        result = self.validator.validate_variation(experiment)
        assert result.is_valid is False
        assert ConstraintViolationType.PERFORMANCE_VIOLATION in result.violations
    
    def test_validation_caching(self):
        """Test validation results are cached for performance"""
        experiment = ExperimentSpec(
            experiment_id='cache_test',
            parameters={},
            strategy_type='navigation',
            dsl_script='A B',
            estimated_duration=60.0
        )
        
        # First validation
        result1 = self.validator.validate_variation(experiment)
        cache_size_1 = len(self.validator.validation_cache)
        
        # Second validation should use cache
        result2 = self.validator.validate_variation(experiment)
        cache_size_2 = len(self.validator.validation_cache)
        
        assert result1.is_valid == result2.is_valid
        assert cache_size_1 == cache_size_2  # No new cache entry


class TestConservativeVariationGenerator:
    """Test conservative variation generation strategy"""
    
    def setup_method(self):
        self.generator = ConservativeVariationGenerator()
        self.config = VariationConfig()
        self.base_experiment = ExperimentSpec(
            experiment_id='base',
            parameters={'movement_sequence_length': 20, 'pathfind_distance': 30},
            strategy_type='navigation',
            dsl_script='A B UP DOWN',
            estimated_duration=60.0
        )
    
    def test_conservative_generates_variations(self):
        """Test conservative generator creates variations"""
        variations = self.generator.generate_variations(self.base_experiment, 5, self.config)
        
        assert len(variations) > 0
        assert len(variations) <= 5
        
        # All should be valid ExperimentSpec objects
        for variation in variations:
            assert isinstance(variation, ExperimentSpec)
            assert variation.experiment_id != self.base_experiment.experiment_id
            assert variation.source_experiment_id == self.base_experiment.experiment_id
    
    def test_conservative_makes_small_changes(self):
        """Test conservative generator makes only small parameter changes"""
        variations = self.generator.generate_variations(self.base_experiment, 3, self.config)
        
        for variation in variations:
            for param_name, new_value in variation.parameters.items():
                if param_name in self.base_experiment.parameters:
                    original_value = self.base_experiment.parameters[param_name]
                    if isinstance(original_value, (int, float)):
                        # Changes should be small (within ~20% for conservative approach)
                        change_ratio = abs(new_value - original_value) / max(1, abs(original_value))
                        assert change_ratio < 0.3  # Conservative changes
    
    def test_conservative_handles_errors_gracefully(self):
        """Test conservative generator handles errors without crashing"""
        # Create a problematic base experiment
        bad_experiment = ExperimentSpec(
            experiment_id='bad',
            parameters={'invalid_param': 'not_a_number'},
            strategy_type='unknown',
            dsl_script='INVALID COMMANDS',
            estimated_duration=60.0
        )
        
        variations = self.generator.generate_variations(bad_experiment, 3, self.config)
        
        # Should not crash, but may return fewer variations
        assert isinstance(variations, list)


class TestPriorityQueue:
    """Test experiment priority queue functionality"""
    
    def setup_method(self):
        self.selector = ExperimentSelector()
    
    def test_priority_ordering(self):
        """Test experiments are ordered by priority correctly"""
        experiments = [
            ({'experiment_id': 'low', 'parameters': {}, 'strategy_type': 'nav', 'dsl_script': 'A', 'estimated_duration': 30}, 0.5),
            ({'experiment_id': 'high', 'parameters': {}, 'strategy_type': 'nav', 'dsl_script': 'B', 'estimated_duration': 30}, 2.0),
            ({'experiment_id': 'medium', 'parameters': {}, 'strategy_type': 'nav', 'dsl_script': 'UP', 'estimated_duration': 30}, 1.0),
        ]
        
        for exp, priority in experiments:
            self.selector.add_experiment(exp, priority)
        
        selected = self.selector.select_next_experiments(3)
        
        # Should be ordered by priority (highest first)
        assert len(selected) == 3
        assert selected[0]['experiment_id'] == 'high'
        assert selected[1]['experiment_id'] == 'medium'
        assert selected[2]['experiment_id'] == 'low'
    
    def test_select_more_than_available(self):
        """Test selecting more experiments than available"""
        self.selector.add_experiment({
            'experiment_id': 'only_one',
            'parameters': {},
            'strategy_type': 'navigation',
            'dsl_script': 'A B',
            'estimated_duration': 30
        }, priority=1.0)
        
        selected = self.selector.select_next_experiments(5)  # Request more than available
        
        assert len(selected) == 1  # Should only return what's available
        assert selected[0]['experiment_id'] == 'only_one'


class TestCompletionTracking:
    """Test experiment completion tracking"""
    
    def setup_method(self):
        self.selector = ExperimentSelector()
    
    def test_mark_completed_basic(self):
        """Test marking experiment as completed"""
        result = {'success': True, 'duration': 45.0, 'score': 100}
        
        self.selector.mark_completed('test_exp', result)
        
        history = self.selector.get_experiment_history()
        assert len(history) == 1
        assert history[0]['experiment_id'] == 'test_exp'
        assert history[0]['result'] == result
        assert 'completed_at' in history[0]
    
    def test_completion_history_bounded(self):
        """Test completion history is bounded to prevent memory growth"""
        # Add more completions than the max history size (1000)
        for i in range(1200):
            self.selector.mark_completed(f'exp_{i}', {'score': i})
        
        history = self.selector.get_experiment_history()
        assert len(history) <= 1000  # Should be bounded


class TestPriorityCalculation:
    """Test experiment priority calculation"""
    
    def setup_method(self):
        self.selector = ExperimentSelector()
    
    def test_priority_calculation_basic(self):
        """Test basic priority calculation"""
        experiment = {
            'experiment_id': 'test',
            'estimated_duration': 60.0,
            'strategy_type': 'navigation'
        }
        
        priority = self.selector.calculate_priority(experiment)
        
        assert isinstance(priority, float)
        assert 0.1 <= priority <= 10.0  # Should be within bounds
    
    def test_priority_prefers_shorter_experiments(self):
        """Test priority calculation prefers shorter experiments"""
        short_exp = {'estimated_duration': 30.0, 'strategy_type': 'navigation'}
        long_exp = {'estimated_duration': 180.0, 'strategy_type': 'navigation'}
        
        short_priority = self.selector.calculate_priority(short_exp)
        long_priority = self.selector.calculate_priority(long_exp)
        
        assert short_priority > long_priority
    
    def test_priority_handles_invalid_input(self):
        """Test priority calculation handles invalid input gracefully"""
        invalid_exp = {'invalid_field': 'invalid_value'}
        
        priority = self.selector.calculate_priority(invalid_exp)
        
        assert priority == 1.0  # Should return safe default


class TestErrorHandling:
    """Test comprehensive error handling"""
    
    def setup_method(self):
        self.selector = ExperimentSelector()
    
    def test_add_experiment_with_invalid_data(self):
        """Test adding experiment with invalid data doesn't crash"""
        invalid_experiments = [
            None,
            "not_a_dict",
            {},  # Missing required fields
            {'experiment_id': None},  # Invalid ID type
        ]
        
        for invalid_exp in invalid_experiments:
            # Should not raise exceptions
            try:
                self.selector.add_experiment(invalid_exp, 1.0)
            except Exception as e:
                pytest.fail(f"add_experiment should not raise exception for invalid input: {e}")
    
    @patch('src.claudelearnspokemon.experiment_selector.ConservativeVariationGenerator.generate_variations')
    def test_variation_generation_handles_generator_failure(self, mock_generate):
        """Test system handles variation generator failures"""
        mock_generate.side_effect = Exception("Generator failed")
        
        base_experiment = {
            'experiment_id': 'base',
            'parameters': {},
            'strategy_type': 'navigation',
            'dsl_script': 'A B',
            'estimated_duration': 60.0
        }
        
        variations = self.selector.generate_variations(base_experiment)
        
        # Should return fallback variations instead of crashing
        assert isinstance(variations, list)
        # May be empty or contain fallback variations


class TestParameterBounds:
    """Test parameter bounds validation"""
    
    def test_command_bounds_defined(self):
        """Test all expected command bounds are defined"""
        expected_bounds = [
            'movement_sequence_length',
            'pathfind_distance', 
            'interaction_attempts',
            'wait_time_ms',
            'script_complexity'
        ]
        
        for bound_name in expected_bounds:
            assert bound_name in ParameterBounds.COMMAND_BOUNDS
            min_val, max_val = ParameterBounds.COMMAND_BOUNDS[bound_name]
            assert isinstance(min_val, (int, float))
            assert isinstance(max_val, (int, float))
            assert min_val < max_val
    
    def test_pokemon_constraints_defined(self):
        """Test Pokemon-specific constraints are defined"""
        expected_constraints = [
            'max_hp',
            'level_range',
            'item_quantity',
            'move_pp'
        ]
        
        for constraint_name in expected_constraints:
            assert constraint_name in ParameterBounds.POKEMON_CONSTRAINTS
            min_val, max_val = ParameterBounds.POKEMON_CONSTRAINTS[constraint_name]
            assert isinstance(min_val, (int, float))
            assert isinstance(max_val, (int, float))
            assert min_val <= max_val  # Equal is OK (e.g., for boolean constraints)


class TestPerformanceRequirements:
    """Test performance requirements are met"""
    
    def setup_method(self):
        self.selector = ExperimentSelector()
        self.base_experiment = {
            'experiment_id': 'perf_test',
            'parameters': {'movement_sequence_length': 10},
            'strategy_type': 'navigation',
            'dsl_script': 'A B UP DOWN',
            'estimated_duration': 60.0
        }
    
    def test_variation_generation_meets_timing_requirements(self):
        """Test variation generation completes within timing requirements"""
        config = VariationConfig(
            max_variations_per_batch=20,
            max_generation_time_ms=100.0  # 100ms limit
        )
        selector = ExperimentSelector(config)
        
        start_time = time.perf_counter()
        variations = selector.generate_variations(self.base_experiment)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Should complete within reasonable time (with test environment tolerance)
        assert duration_ms < 200.0  # Allow 2x tolerance for test environment
        assert len(variations) > 0  # Should still produce useful results
    
    def test_validation_performance(self):
        """Test validation performance is acceptable"""
        validator = VariationValidator()
        experiment = ExperimentSpec(
            experiment_id='perf_validation_test',
            parameters={'movement_sequence_length': 15},
            strategy_type='navigation',
            dsl_script='A B UP DOWN LEFT RIGHT',
            estimated_duration=60.0
        )
        
        start_time = time.perf_counter()
        result = validator.validate_variation(experiment)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        assert duration_ms < 50.0  # Should validate quickly
        assert result.performance_cost < 50.0  # Internal timing should also be reasonable


# Integration tests
class TestIntegration:
    """Test integration between different components"""
    
    def setup_method(self):
        self.selector = ExperimentSelector()
    
    def test_full_workflow_basic(self):
        """Test complete workflow from adding to selecting to generating variations"""
        # Add initial experiment
        base_experiment = {
            'experiment_id': 'workflow_test',
            'parameters': {'movement_sequence_length': 15},
            'strategy_type': 'navigation',
            'dsl_script': 'A B UP DOWN',
            'estimated_duration': 60.0
        }
        
        self.selector.add_experiment(base_experiment, priority=1.5)
        
        # Select experiment
        selected = self.selector.select_next_experiments(1)
        assert len(selected) == 1
        assert selected[0]['experiment_id'] == 'workflow_test'
        
        # Generate variations
        variations = self.selector.generate_variations(selected[0])
        assert len(variations) > 0
        
        # Mark as completed
        result = {'success': True, 'score': 95}
        self.selector.mark_completed('workflow_test', result)
        
        # Check completion tracking
        history = self.selector.get_experiment_history()
        assert len(history) == 1
        assert history[0]['experiment_id'] == 'workflow_test'
    
    def test_system_stats_tracking(self):
        """Test system properly tracks statistics"""
        base_experiment = {
            'experiment_id': 'stats_test',
            'parameters': {},
            'strategy_type': 'navigation',
            'dsl_script': 'A B',
            'estimated_duration': 30.0
        }
        
        # Generate variations multiple times
        for i in range(3):
            self.selector.generate_variations(base_experiment)
        
        stats = self.selector.get_generation_stats()
        
        assert stats['total_calls'] == 3
        assert stats['total_variations'] >= 0
        assert stats['total_time_ms'] > 0
        assert 'average_variations_per_call' in stats
        assert 'average_time_per_call_ms' in stats


if __name__ == "__main__":
    # Rex: Run basic verification
    pytest.main([__file__, "-v"])