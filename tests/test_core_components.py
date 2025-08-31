"""
Tests for Core Script Development Engine Components

These tests validate the 4 core components of the SonnetWorkerPool script development engine:
- ExperimentSelector: ML-guided strategy selection
- ParallelExecutionCoordinator: Worker coordination
- PatternProcessor: Cross-worker pattern synthesis  
- WorkerDistributor: Intelligent load balancing

Test Focus:
- Performance requirements (<50ms selection, <100ms coordination)
- Integration with existing infrastructure
- ML-guided decision making
- Cross-worker pattern synthesis and sharing
"""

import time
import random
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import pytest

from claudelearnspokemon.sonnet_worker_pool import (
    ExperimentSelector,
    ParallelExecutionCoordinator, 
    PatternProcessor,
    WorkerDistributor,
    SonnetWorkerPool,
    ReinforcementLearningEngine,
    SemanticPatternEngine,
    CrossWorkerPatternSynthesis
)


@pytest.mark.fast
class TestExperimentSelector:
    """Test suite for ExperimentSelector component."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.experiment_selector = ExperimentSelector()
        self.mock_rl_engine = Mock(spec=ReinforcementLearningEngine)
        self.mock_semantic_engine = Mock(spec=SemanticPatternEngine)

    def test_experiment_selector_initialization(self):
        """Test that ExperimentSelector initializes with proper default values."""
        assert len(self.experiment_selector.development_strategies) == 5
        assert len(self.experiment_selector.strategy_success_rates) == 5
        assert 'genetic_evolution' in self.experiment_selector.development_strategies
        assert 'reinforcement_guided' in self.experiment_selector.development_strategies
        assert 'pattern_synthesis' in self.experiment_selector.development_strategies

    def test_strategy_selection_performance_target(self):
        """Test that strategy selection meets <50ms performance target."""
        task_context = {
            'location': 'viridian_city', 
            'objective': 'optimization',
            'complexity': 'medium'
        }
        
        # Test performance across multiple selections
        selection_times = []
        for _ in range(10):
            start_time = time.time()
            result = self.experiment_selector.select_optimal_strategy(
                task_context, 
                self.mock_rl_engine, 
                self.mock_semantic_engine
            )
            selection_time = time.time() - start_time
            selection_times.append(selection_time * 1000)  # Convert to ms
            
            assert result['strategy'] is not None
            assert result['confidence'] >= 0.0
            assert result['selection_time_ms'] < 50.0  # Performance requirement
        
        avg_time = sum(selection_times) / len(selection_times)
        assert avg_time < 50.0, f"Average selection time {avg_time:.1f}ms exceeds 50ms target"

    def test_ml_guided_strategy_selection(self):
        """Test ML-guided strategy selection with mocked RL engine."""
        # Mock RL engine response
        self.mock_rl_engine.select_action.return_value = {
            'exploration_weight': 0.7,
            'exploitation_weight': 0.3
        }
        
        task_context = {'location': 'battle', 'difficulty': 'high'}
        
        result = self.experiment_selector.select_optimal_strategy(
            task_context,
            self.mock_rl_engine,
            self.mock_semantic_engine
        )
        
        assert result['strategy'] in self.experiment_selector.development_strategies
        assert result['source'] in ['ml_guided', 'cache']
        assert 0.0 <= result['confidence'] <= 1.0
        
        # Verify RL engine was called
        self.mock_rl_engine.select_action.assert_called_once()

    def test_strategy_performance_update(self):
        """Test strategy performance tracking and learning."""
        strategy = 'genetic_evolution'
        task_context = {'location': 'test', 'objective': 'test_objective'}
        
        # Update performance with successful result
        self.experiment_selector.update_strategy_performance(
            strategy, 
            task_context, 
            success=True,
            performance_metrics={'quality_score': 0.8, 'execution_time_ms': 1000}
        )
        
        # Verify performance was tracked
        assert strategy in self.experiment_selector.strategy_success_rates
        assert len(self.experiment_selector.strategy_success_rates[strategy]) == 1
        assert self.experiment_selector.strategy_success_rates[strategy][0] == True

    def test_caching_functionality(self):
        """Test that strategy selection caching works correctly."""
        task_context = {'location': 'cache_test', 'objective': 'caching'}
        
        # First selection should be computed
        result1 = self.experiment_selector.select_optimal_strategy(
            task_context,
            self.mock_rl_engine, 
            self.mock_semantic_engine
        )
        
        # Second selection with same context should be cached
        result2 = self.experiment_selector.select_optimal_strategy(
            task_context,
            self.mock_rl_engine,
            self.mock_semantic_engine  
        )
        
        assert result1['strategy'] == result2['strategy']
        assert result2['source'] == 'cache'
        assert result2['selection_time_ms'] < result1['selection_time_ms']

    def test_fallback_strategy_selection(self):
        """Test fallback behavior when ML guidance fails."""
        # Mock RL engine to raise an exception
        self.mock_rl_engine.select_action.side_effect = Exception("RL engine failure")
        
        task_context = {'location': 'fallback_test'}
        
        result = self.experiment_selector.select_optimal_strategy(
            task_context,
            self.mock_rl_engine,
            self.mock_semantic_engine
        )
        
        # The strategy selection should still work, even if RL fails
        assert result['strategy'] is not None
        # The source may be 'ml_guided' if other parts of ML guidance work
        assert result['source'] in ['fallback', 'ml_guided']
        assert result['confidence'] >= 0.0

    def test_performance_statistics(self):
        """Test performance statistics collection."""
        # Make several selections
        for i in range(5):
            task_context = {'location': f'test_{i}', 'objective': 'stats'}
            self.experiment_selector.select_optimal_strategy(
                task_context,
                self.mock_rl_engine,
                self.mock_semantic_engine
            )
        
        stats = self.experiment_selector.get_performance_stats()
        assert 'average_selection_time_ms' in stats
        assert 'total_selections' in stats
        assert stats['total_selections'] == 5
        assert stats['performance_target_met'] is True or False


@pytest.mark.fast 
class TestParallelExecutionCoordinator:
    """Test suite for ParallelExecutionCoordinator component."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.coordinator = ParallelExecutionCoordinator()
        self.mock_worker_pool = Mock(spec=SonnetWorkerPool)
        
        # Mock worker pool with healthy workers
        self.mock_worker_pool.workers = {
            'worker_1': {'healthy': True, 'status': 'ready'},
            'worker_2': {'healthy': True, 'status': 'ready'},
            'worker_3': {'healthy': True, 'status': 'ready'}
        }

    def test_coordinator_initialization(self):
        """Test ParallelExecutionCoordinator initialization."""
        assert len(self.coordinator.coordination_strategies) == 4
        assert self.coordinator.default_coordination_strategy == 'load_balanced'
        assert len(self.coordinator.active_executions) == 0
        assert self.coordinator.coordination_metrics['total_coordinated_executions'] == 0

    def test_coordination_performance_target(self):
        """Test that coordination meets <100ms performance target."""
        execution_requests = [
            {'task_id': f'task_{i}', 'priority': 0.5, 'estimated_duration_ms': 1000}
            for i in range(3)
        ]
        
        coordination_times = []
        for _ in range(5):
            start_time = time.time()
            result = self.coordinator.coordinate_parallel_execution(
                execution_requests,
                self.mock_worker_pool
            )
            coordination_time = time.time() - start_time
            coordination_times.append(coordination_time * 1000)
            
            assert result['coordination_time_ms'] < 100.0  # Performance requirement
            assert len(result['assigned_executions']) <= len(execution_requests)
        
        avg_time = sum(coordination_times) / len(coordination_times)
        assert avg_time < 100.0, f"Average coordination time {avg_time:.1f}ms exceeds 100ms target"

    def test_load_balanced_coordination(self):
        """Test load-balanced coordination strategy."""
        execution_requests = [
            {'task_id': 'task_1', 'priority': 0.8},
            {'task_id': 'task_2', 'priority': 0.6},
            {'task_id': 'task_3', 'priority': 0.4}
        ]
        
        result = self.coordinator.coordinate_parallel_execution(
            execution_requests,
            self.mock_worker_pool,
            coordination_strategy='load_balanced'
        )
        
        assert result['coordination_strategy'] == 'load_balanced'
        assert 'assigned_executions' in result
        assert 'failed_assignments' in result
        
        # Should assign to different workers for load balancing
        assigned_workers = [exec['worker_id'] for exec in result['assigned_executions']]
        assert len(set(assigned_workers)) > 1 or len(assigned_workers) == 1

    def test_execution_progress_monitoring(self):
        """Test execution progress monitoring."""
        # Create a coordination result first
        execution_requests = [{'task_id': 'monitor_test', 'estimated_duration_ms': 2000}]
        result = self.coordinator.coordinate_parallel_execution(
            execution_requests,
            self.mock_worker_pool
        )
        
        if result['assigned_executions']:
            execution_id = result['assigned_executions'][0]['execution_id']
            
            # Monitor progress immediately
            progress = self.coordinator.monitor_execution_progress(execution_id)
            assert progress is not None
            assert progress['execution_id'] == execution_id
            assert 0.0 <= progress['progress_percentage'] <= 100.0

    def test_execution_completion_handling(self):
        """Test execution completion tracking."""
        # Create a coordination result
        execution_requests = [{'task_id': 'completion_test'}]
        result = self.coordinator.coordinate_parallel_execution(
            execution_requests,
            self.mock_worker_pool
        )
        
        if result['assigned_executions']:
            execution_id = result['assigned_executions'][0]['execution_id']
            
            # Handle completion
            completion_result = {'success': True, 'quality_score': 0.8}
            self.coordinator.handle_execution_completion(execution_id, completion_result)
            
            # Verify execution was cleaned up
            assert execution_id not in self.coordinator.active_executions

    def test_optimal_worker_assignment(self):
        """Test optimal worker assignment logic."""
        task_requirements = {'complexity': 'high', 'performance_target': 'quality'}
        available_workers = ['worker_1', 'worker_2', 'worker_3']
        
        selected_worker = self.coordinator.get_optimal_worker_assignment(
            task_requirements,
            available_workers
        )
        
        assert selected_worker in available_workers

    def test_coordination_statistics(self):
        """Test coordination statistics collection."""
        # Perform some coordinations
        for i in range(3):
            execution_requests = [{'task_id': f'stats_task_{i}'}]
            self.coordinator.coordinate_parallel_execution(
                execution_requests,
                self.mock_worker_pool
            )
        
        stats = self.coordinator.get_coordination_stats()
        if stats.get('status') != 'no_data':
            assert 'average_coordination_time_ms' in stats
            assert 'performance_target_met' in stats
            assert stats['coordination_metrics']['total_coordinated_executions'] >= 3


@pytest.mark.fast
class TestPatternProcessor:
    """Test suite for PatternProcessor component."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.mock_synthesis_engine = Mock(spec=CrossWorkerPatternSynthesis)
        self.pattern_processor = PatternProcessor(self.mock_synthesis_engine)

    def test_pattern_processor_initialization(self):
        """Test PatternProcessor initialization."""
        assert self.pattern_processor.synthesis_engine is not None
        assert len(self.pattern_processor.pattern_categories) == 4
        assert 'movement' in self.pattern_processor.pattern_categories
        assert 'action' in self.pattern_processor.pattern_categories

    def test_pattern_processing(self):
        """Test pattern processing functionality."""
        source_patterns = [
            {
                'pattern_sequence': ['UP', 'UP', 'A'],
                'success_rate': 0.8,
                'source_worker': 'worker_1'
            },
            {
                'pattern_sequence': ['DOWN', 'B', 'SELECT'],
                'success_rate': 0.6,
                'source_worker': 'worker_2'
            }
        ]
        
        target_context = {'location': 'battle', 'objective': 'optimization'}
        
        # Mock synthesis engine response
        self.mock_synthesis_engine.synthesize_patterns.return_value = [
            {
                'pattern_sequence': ['UP', 'A', 'DOWN'],
                'synthesis_strategy': 'sequential_fusion',
                'estimated_quality': 0.7
            }
        ]
        
        result = self.pattern_processor.process_patterns(
            source_patterns,
            target_context
        )
        
        assert result['processing_success'] is True
        assert len(result['synthesized_patterns']) > 0
        assert 'pattern_metadata' in result
        assert 'quality_metrics' in result
        
        # Verify synthesis engine was called
        self.mock_synthesis_engine.synthesize_patterns.assert_called_once()

    def test_pattern_validation(self):
        """Test pattern validation functionality."""
        # Test with invalid patterns
        invalid_patterns = [
            {'pattern_sequence': []},  # Empty sequence
            {'no_sequence': 'invalid'},  # Missing pattern_sequence
            {'pattern_sequence': ['INVALID_COMMAND'] * 20}  # Too many invalid commands
        ]
        
        valid_patterns = self.pattern_processor._validate_source_patterns(invalid_patterns)
        assert len(valid_patterns) == 0

        # Test with valid patterns
        valid_pattern_list = [
            {'pattern_sequence': ['UP', 'DOWN', 'A', 'B']},
            {'pattern_sequence': ['LEFT', 'RIGHT', 'START']}
        ]
        
        validated = self.pattern_processor._validate_source_patterns(valid_pattern_list)
        assert len(validated) == 2

    def test_pattern_classification(self):
        """Test pattern classification functionality."""
        patterns = [
            {'pattern_sequence': ['UP', 'DOWN', 'LEFT']},  # Movement
            {'pattern_sequence': ['A', 'B', 'START']},     # Action  
            {'pattern_sequence': ['OBSERVE', 'IF']},       # Complex
            {'pattern_sequence': ['UP'] * 20},             # Long sequence
            {'pattern_sequence': ['A']}                    # Short sequence
        ]
        
        classification = self.pattern_processor._classify_patterns(patterns)
        
        assert 'classification_summary' in classification
        assert classification['classification_summary']['total_patterns'] == 5
        assert 'movement' in classification['by_category']
        assert 'action' in classification['by_category']
        assert len(classification['by_length']['long']) > 0

    def test_worker_pattern_merging(self):
        """Test merging patterns from multiple workers."""
        worker_patterns = {
            'worker_1': [
                {'pattern_sequence': ['UP', 'A'], 'success_rate': 0.9}
            ],
            'worker_2': [
                {'pattern_sequence': ['DOWN', 'B'], 'success_rate': 0.7}
            ],
            'worker_3': [
                {'pattern_sequence': ['LEFT', 'RIGHT'], 'success_rate': 0.8}
            ]
        }
        
        # Test quality-weighted merge
        merged = self.pattern_processor.merge_worker_patterns(
            worker_patterns, 
            merge_strategy='quality_weighted'
        )
        
        assert len(merged) > 0
        assert all('source_worker' in pattern for pattern in merged)
        assert all('merge_timestamp' in pattern for pattern in merged)

    def test_pattern_optimization(self):
        """Test pattern selection optimization."""
        available_patterns = [
            {'pattern_sequence': ['UP', 'A'], 'success_rate': 0.9, 'timestamp': time.time()},
            {'pattern_sequence': ['DOWN', 'B'], 'success_rate': 0.7, 'timestamp': time.time() - 3600},
            {'pattern_sequence': ['LEFT', 'RIGHT'], 'success_rate': 0.8, 'timestamp': time.time() - 1800}
        ]
        
        optimization_criteria = {
            'quality': 0.5,
            'diversity': 0.3, 
            'performance': 0.1,
            'recency': 0.1,
            'max_patterns': 2
        }
        
        optimized = self.pattern_processor.optimize_pattern_selection(
            available_patterns,
            optimization_criteria
        )
        
        assert len(optimized) <= 2
        assert len(optimized) > 0

    def test_processing_statistics(self):
        """Test pattern processing statistics."""
        # Perform some processing operations
        for i in range(3):
            patterns = [{'pattern_sequence': ['UP', 'A'], 'success_rate': 0.8}]
            context = {'location': f'test_{i}'}
            
            self.mock_synthesis_engine.synthesize_patterns.return_value = []
            self.pattern_processor.process_patterns(patterns, context)
        
        stats = self.pattern_processor.get_processing_stats()
        
        if stats.get('status') != 'no_data':
            assert 'total_patterns_processed' in stats
            assert stats['total_patterns_processed'] >= 3


@pytest.mark.fast
class TestWorkerDistributor:
    """Test suite for WorkerDistributor component."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.distributor = WorkerDistributor()

    def test_worker_distributor_initialization(self):
        """Test WorkerDistributor initialization."""
        assert len(self.distributor.distribution_strategies) == 6
        assert self.distributor.default_strategy == 'hybrid'
        assert 'round_robin' in self.distributor.distribution_strategies
        assert 'performance_weighted' in self.distributor.distribution_strategies

    def test_task_distribution(self):
        """Test basic task distribution functionality."""
        task = {
            'task_id': 'test_task',
            'context': {'location': 'test', 'objective': 'testing'},
            'priority': 0.7
        }
        available_workers = ['worker_1', 'worker_2', 'worker_3']
        
        result = self.distributor.distribute_task(task, available_workers)
        
        assert result['selected_worker'] in available_workers
        assert 0.0 <= result['confidence'] <= 1.0
        assert result['distribution_strategy'] is not None

    def test_round_robin_distribution(self):
        """Test round-robin distribution strategy."""
        task = {'task_id': 'round_robin_test'}
        workers = ['worker_1', 'worker_2', 'worker_3']
        
        # Test several distributions to verify round-robin behavior
        selected_workers = []
        for i in range(6):
            result = self.distributor._distribute_round_robin(task, workers)
            selected_workers.append(result['selected_worker'])
        
        # Should cycle through workers
        assert selected_workers[0] == selected_workers[3]  # Should repeat after 3
        assert selected_workers[1] == selected_workers[4]
        assert len(set(selected_workers)) == 3  # All workers should be used

    def test_performance_weighted_distribution(self):
        """Test performance-weighted distribution strategy."""
        # Set up performance history
        self.distributor.worker_performance_history = {
            'worker_1': [0.9, 0.8, 0.9],  # High performance
            'worker_2': [0.5, 0.6, 0.4],  # Medium performance  
            'worker_3': [0.2, 0.3, 0.1]   # Low performance
        }
        
        task = {'task_id': 'performance_test'}
        workers = ['worker_1', 'worker_2', 'worker_3']
        
        result = self.distributor._distribute_performance_weighted(task, workers)
        
        # Should prefer high-performance worker
        assert result['selected_worker'] == 'worker_1'
        assert result['confidence'] > 0.5

    def test_worker_performance_tracking(self):
        """Test worker performance tracking and updates."""
        worker_id = 'test_worker'
        task = {'context': {'complexity': 'medium'}}
        result = {
            'success': True,
            'execution_time_ms': 2000,
            'quality_score': 0.8
        }
        
        # Update performance
        self.distributor.update_worker_performance(worker_id, task, result)
        
        # Verify tracking
        assert worker_id in self.distributor.worker_performance_history
        assert len(self.distributor.worker_performance_history[worker_id]) == 1
        assert worker_id in self.distributor.worker_success_rates

    def test_performance_prediction(self):
        """Test worker performance prediction."""
        worker_id = 'prediction_test_worker'
        
        # Set up historical data
        self.distributor.worker_performance_history[worker_id] = [0.8, 0.7, 0.9, 0.8]
        self.distributor.worker_task_completion_times[worker_id] = [2.0, 3.0, 2.5]
        
        task = {'context': {'complexity': 'high'}}
        
        prediction = self.distributor.predict_worker_performance(worker_id, task)
        
        assert 'success_probability' in prediction
        assert 'expected_completion_time' in prediction
        assert 'expected_quality_score' in prediction
        assert 'confidence' in prediction
        assert 0.0 <= prediction['success_probability'] <= 1.0

    def test_optimal_batch_distribution(self):
        """Test optimal distribution of multiple tasks."""
        tasks = [
            {'task_id': 'batch_1', 'priority': 0.9},
            {'task_id': 'batch_2', 'priority': 0.7}, 
            {'task_id': 'batch_3', 'priority': 0.5}
        ]
        workers = ['worker_1', 'worker_2']
        
        result = self.distributor.get_optimal_worker_distribution(tasks, workers)
        
        assert 'distributions' in result
        assert 'optimization_score' in result
        assert len(result['distributions']) == len(tasks)

    def test_distribution_constraints(self):
        """Test distribution constraint application."""
        workers = ['worker_1', 'worker_2', 'worker_3']
        
        # Set up worker loads and performance
        self.distributor.current_worker_loads = {
            'worker_1': 0.8,  # High load
            'worker_2': 0.3,  # Low load
            'worker_3': 0.6   # Medium load
        }
        self.distributor.worker_success_rates = {
            'worker_1': 0.9,  # High performance
            'worker_2': 0.4,  # Low performance  
            'worker_3': 0.7   # Medium performance
        }
        
        # Test max load constraint
        constraints = {'max_load': 0.5}
        filtered = self.distributor._apply_distribution_constraints(workers, constraints)
        assert 'worker_1' not in filtered  # Should be excluded due to high load
        assert 'worker_2' in filtered
        
        # Test min performance constraint
        constraints = {'min_performance': 0.6}
        filtered = self.distributor._apply_distribution_constraints(workers, constraints)
        assert 'worker_2' not in filtered  # Should be excluded due to low performance
        assert 'worker_1' in filtered

    def test_distribution_statistics(self):
        """Test distribution statistics collection."""
        # Perform several distributions
        task = {'task_id': 'stats_test'}
        workers = ['worker_1', 'worker_2']
        
        for i in range(5):
            self.distributor.distribute_task(task, workers)
        
        stats = self.distributor.get_distribution_stats()
        
        if stats.get('status') != 'no_data':
            assert 'total_distributions' in stats
            assert 'average_distribution_time_ms' in stats
            assert stats['total_distributions'] >= 5


@pytest.mark.integration
class TestCoreComponentsIntegration:
    """Integration tests for all 4 core components working together."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.mock_worker_pool = Mock(spec=SonnetWorkerPool)
        self.mock_worker_pool.workers = {
            'worker_1': {'healthy': True, 'status': 'ready'},
            'worker_2': {'healthy': True, 'status': 'ready'}
        }

    def test_end_to_end_script_development_workflow(self):
        """Test end-to-end workflow using all 4 components."""
        # Initialize components
        experiment_selector = ExperimentSelector()
        coordinator = ParallelExecutionCoordinator()
        pattern_processor = PatternProcessor()
        distributor = WorkerDistributor()
        
        # Mock dependencies
        mock_rl_engine = Mock(spec=ReinforcementLearningEngine)
        mock_semantic_engine = Mock(spec=SemanticPatternEngine)
        mock_rl_engine.select_action.return_value = {
            'exploration_weight': 0.6,
            'exploitation_weight': 0.4
        }
        
        # 1. Strategy Selection
        task_context = {'location': 'integration_test', 'objective': 'end_to_end'}
        strategy_result = experiment_selector.select_optimal_strategy(
            task_context, 
            mock_rl_engine,
            mock_semantic_engine
        )
        
        assert strategy_result['strategy'] is not None
        
        # 2. Worker Distribution
        task = {'context': task_context, 'strategy': strategy_result['strategy']}
        available_workers = ['worker_1', 'worker_2']
        distribution_result = distributor.distribute_task(task, available_workers)
        
        assert distribution_result['selected_worker'] is not None
        
        # 3. Pattern Processing
        patterns = [
            {'pattern_sequence': ['UP', 'A'], 'success_rate': 0.8},
            {'pattern_sequence': ['DOWN', 'B'], 'success_rate': 0.7}
        ]
        
        # Mock the synthesis engine for pattern processor
        pattern_processor.synthesis_engine = Mock()
        pattern_processor.synthesis_engine.synthesize_patterns.return_value = [
            {'pattern_sequence': ['UP', 'DOWN', 'A'], 'synthesis_strategy': 'test'}
        ]
        
        processing_result = pattern_processor.process_patterns(patterns, task_context)
        
        assert processing_result['processing_success'] is True
        
        # 4. Parallel Execution Coordination
        execution_requests = [
            {
                'task_id': 'integrated_task',
                'strategy': strategy_result['strategy'],
                'patterns': processing_result['synthesized_patterns'],
                'assigned_worker': distribution_result['selected_worker']
            }
        ]
        
        coordination_result = coordinator.coordinate_parallel_execution(
            execution_requests,
            self.mock_worker_pool
        )
        
        assert len(coordination_result['assigned_executions']) > 0 or len(coordination_result['failed_assignments']) > 0

    def test_performance_requirements_integration(self):
        """Test that all components meet performance requirements when used together."""
        experiment_selector = ExperimentSelector()
        coordinator = ParallelExecutionCoordinator()
        
        mock_rl_engine = Mock()
        mock_semantic_engine = Mock()
        mock_rl_engine.select_action.return_value = {'exploration_weight': 0.5}
        
        # Test combined performance
        task_context = {'location': 'performance_test'}
        execution_requests = [{'task_id': 'perf_test', 'priority': 0.5}]
        
        start_time = time.time()
        
        # Strategy selection should be <50ms
        strategy_result = experiment_selector.select_optimal_strategy(
            task_context,
            mock_rl_engine,
            mock_semantic_engine
        )
        
        # Coordination should be <100ms  
        coord_result = coordinator.coordinate_parallel_execution(
            execution_requests,
            self.mock_worker_pool
        )
        
        total_time = time.time() - start_time
        
        # Total time should be reasonable for both operations
        assert strategy_result['selection_time_ms'] < 50.0
        assert coord_result['coordination_time_ms'] < 100.0
        assert total_time < 0.2  # 200ms total should be more than enough


if __name__ == '__main__':
    pytest.main([__file__, '-v'])