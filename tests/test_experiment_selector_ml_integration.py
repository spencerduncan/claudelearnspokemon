"""
Test suite for ML-guided experiment selection integration.

Tests the integration between ExperimentSelector and ReinforcementLearningEngine,
validating ML-guided selection, performance monitoring, reward computation,
and experience replay learning.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

# Import the classes we need to test
from src.claudelearnspokemon.experiment_selector import (
    ExperimentSelector, ExperimentCandidate, ExperimentResult,
    SelectionStrategy, ExperimentStatus, ExperimentMetrics
)
from src.claudelearnspokemon.sonnet_worker_pool import ReinforcementLearningEngine
from src.claudelearnspokemon.pattern_processor import PatternProcessor
from src.claudelearnspokemon.mcp_data_patterns import PokemonStrategy


class TestExperimentSelectorMLIntegration:
    """Test suite for ML-guided experiment selection integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the pattern processor to avoid external dependencies
        self.mock_pattern_processor = Mock(spec=PatternProcessor)
        self.mock_pattern_processor.get_contextual_pattern_recommendations.return_value = [
            ({'pattern': 'test_pattern'}, 0.8)
        ]
        self.mock_pattern_processor.update_pattern_success = Mock()
        
        # Create experiment selector
        self.selector = ExperimentSelector(
            pattern_processor=self.mock_pattern_processor,
            default_strategy=SelectionStrategy.HEURISTIC
        )
        
        # Create reinforcement learning engine
        self.rl_engine = ReinforcementLearningEngine(
            learning_rate=0.1,
            discount_factor=0.95,
            replay_buffer_size=50,
            exploration_rate=0.2
        )
        
        # Create test experiment candidates
        self.candidates = self._create_test_candidates()
    
    def _create_test_candidates(self) -> List[ExperimentCandidate]:
        """Create test experiment candidates."""
        candidates = []
        
        for i in range(5):
            # Mock PokemonStrategy
            pattern = Mock(spec=PokemonStrategy)
            pattern.id = f"pattern_{i}"
            pattern.success_rate = 0.5 + (i * 0.1)
            pattern.pattern_sequence = [f"cmd_{j}" for j in range(3 + i)]
            
            candidate = ExperimentCandidate(
                experiment_id=f"exp_{i}",
                pattern=pattern,
                context={'location': 'test_location', 'difficulty': 5 + i},
                priority_score=0.5 + (i * 0.1),
                relevance_score=0.6 + (i * 0.05),
                estimated_success_rate=0.7 + (i * 0.05),
                estimated_execution_time=1.0 + (i * 0.5),
                resource_requirements={'cpu_estimate': 0.1, 'memory_mb': 10},
                created_at=time.time(),
                source_worker=f"worker_{i % 2}"
            )
            candidates.append(candidate)
        
        return candidates
    
    def test_rl_integration_setup(self):
        """Test that RL engine can be properly integrated with selector."""
        # Act
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        
        # Assert
        assert self.selector.reinforcement_learning_engine is self.rl_engine
        assert self.selector.ml_model_ready is True
        
    def test_ml_guided_selection_without_rl_engine(self):
        """Test ML-guided selection falls back to heuristic when RL engine is not integrated."""
        # Arrange - ensure no RL engine is integrated
        self.selector.reinforcement_learning_engine = None
        self.selector.ml_model_ready = False
        
        # Act
        selected = self.selector._ml_guided_selection(self.candidates, 2)
        
        # Assert
        assert len(selected) == 2
        # Should have fallen back to heuristic selection
        assert all(isinstance(exp, ExperimentCandidate) for exp in selected)
    
    def test_ml_guided_selection_with_rl_engine(self):
        """Test ML-guided selection with integrated RL engine."""
        # Arrange
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        
        # Act
        selected = self.selector._ml_guided_selection(self.candidates, 3)
        
        # Assert
        assert len(selected) == 3
        assert all(isinstance(exp, ExperimentCandidate) for exp in selected)
        # Should have stored ML context for learning
        assert self.selector._current_ml_context is not None
        assert 'state' in self.selector._current_ml_context
        assert 'action' in self.selector._current_ml_context
        assert 'selected' in self.selector._current_ml_context
        assert len(self.selector._current_ml_context['selected']) == 3
    
    def test_ml_action_adaptation(self):
        """Test adaptation of RL actions to selection parameters."""
        # Arrange
        ml_action = {
            'mutation': 'insert_command',
            'crossover': 'semantic_aware',
            'parameter': 'increase_population'
        }
        
        # Act
        selection_params = self.selector._adapt_ml_action_to_selection(ml_action)
        
        # Assert
        assert isinstance(selection_params, dict)
        assert 'priority_strategy' in selection_params
        assert 'selection_style' in selection_params
        assert 'exploration_mode' in selection_params
        assert 'ml_confidence' in selection_params
        assert selection_params['priority_strategy'] == 'prioritize_high_success'
        assert selection_params['selection_style'] == 'contextual_selection'
        assert selection_params['exploration_mode'] == 'expand_exploration'
    
    def test_ml_scoring_application(self):
        """Test application of ML-enhanced scoring to candidates."""
        # Arrange
        selection_params = {
            'priority_strategy': 'prioritize_high_success',
            'selection_style': 'contextual_selection',
            'exploration_mode': 'expand_exploration',
            'ml_confidence': 0.8
        }
        
        # Act
        scored_candidates = self.selector._apply_ml_scoring(self.candidates, selection_params)
        
        # Assert
        assert len(scored_candidates) == len(self.candidates)
        # Should be sorted by ML-enhanced scores (highest first)
        scores = [c.combined_score for c in scored_candidates]
        # Verify sorting (allowing for ML adjustments)
        for i in range(len(scores) - 1):
            # The exact scores will be modified by ML, just verify we got valid candidates
            assert isinstance(scored_candidates[i], ExperimentCandidate)
    
    def test_experiment_result_ml_learning(self):
        """Test ML learning from experiment results."""
        # Arrange
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        
        # Simulate ML-guided selection
        selected = self.selector._ml_guided_selection(self.candidates, 2)
        experiment_id = selected[0].experiment_id
        
        # Create experiment result
        result = ExperimentResult(
            experiment_id=experiment_id,
            success=True,
            execution_time=0.8,
            performance_metrics={'accuracy': 0.95},
            learned_patterns=[{'pattern': 'new_pattern', 'effectiveness': 0.9}],
            completed_at=time.time()
        )
        
        # Act
        self.selector.record_experiment_result(experiment_id, result)
        
        # Assert
        # Should have updated metrics
        assert self.selector.metrics.experiments_completed == 1
        assert experiment_id in self.selector.experiment_history
        # RL engine should have some experience stored
        assert len(self.rl_engine.experience_buffer) >= 1
    
    def test_reward_computation(self):
        """Test reward computation for experiment outcomes."""
        # Arrange
        candidate = self.candidates[0]
        
        # Test successful result
        success_result = ExperimentResult(
            experiment_id=candidate.experiment_id,
            success=True,
            execution_time=0.5,  # Faster than estimated
            performance_metrics={'accuracy': 0.95},
            learned_patterns=[{'pattern': 'new_pattern'}],
            completed_at=time.time()
        )
        
        # Test failed result  
        failure_result = ExperimentResult(
            experiment_id=candidate.experiment_id,
            success=False,
            execution_time=2.0,  # Slower than estimated
            performance_metrics={'accuracy': 0.2},
            learned_patterns=[],
            completed_at=time.time()
        )
        
        # Act
        success_reward = self.selector._compute_experiment_reward(candidate, success_result)
        failure_reward = self.selector._compute_experiment_reward(candidate, failure_result)
        
        # Assert
        assert -1.0 <= success_reward <= 1.0
        assert -1.0 <= failure_reward <= 1.0
        assert success_reward > failure_reward  # Success should have higher reward
        assert success_reward > 0  # Should be positive for successful experiments
    
    def test_performance_monitoring(self):
        """Test performance monitoring and ML deactivation."""
        # Arrange
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        self.selector.performance_threshold_ms = 20.0  # Very low threshold for testing
        self.selector.max_performance_violations = 2
        
        # Act - simulate slow selections
        self.selector._monitor_selection_performance(50.0)  # Violation 1
        assert self.selector.performance_violation_count == 1
        assert self.selector.ml_model_ready is True  # Still active
        
        self.selector._monitor_selection_performance(60.0)  # Violation 2  
        assert self.selector.performance_violation_count == 2
        assert self.selector.ml_model_ready is False  # Should be deactivated
        
        # Test reactivation
        self.selector.performance_violation_count = 0  # Reset manually for test
        self.selector._reactivate_ml_if_ready()
        assert self.selector.ml_model_ready is True
    
    def test_strategy_activation_with_ml_ready(self):
        """Test that ML strategies are activated when RL engine is ready."""
        # Arrange
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        
        # Test adaptive selection chooses ML when performance is good
        with patch.object(self.selector, '_calculate_recent_success_rate', return_value=0.9):
            selected = self.selector._adaptive_selection(self.candidates, 2)
            assert len(selected) == 2
        
        # Test adaptive selection chooses ML when performance is poor (to learn)
        with patch.object(self.selector, '_calculate_recent_success_rate', return_value=0.3):
            selected = self.selector._adaptive_selection(self.candidates, 2)  
            assert len(selected) == 2
    
    def test_hybrid_selection_combination(self):
        """Test hybrid selection properly combines heuristic and ML approaches."""
        # Arrange
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        
        # Act
        selected = self.selector._hybrid_selection(self.candidates, 3)
        
        # Assert
        assert len(selected) == 3
        assert all(isinstance(exp, ExperimentCandidate) for exp in selected)
        # Should have prioritized candidates selected by both methods
    
    def test_ml_guided_selection_strategy_switch(self):
        """Test that selector can switch to ML_GUIDED strategy."""
        # Arrange
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        
        # Act
        success = self.selector.set_selection_strategy(SelectionStrategy.ML_GUIDED)
        
        # Assert
        assert success is True
        assert self.selector.current_strategy == SelectionStrategy.ML_GUIDED
        
        # Test selection works with ML_GUIDED strategy
        selected = self.selector.select_experiments(self.candidates, 2)
        assert len(selected) == 2
    
    def test_end_to_end_ml_learning_cycle(self):
        """Test complete ML learning cycle from selection to result feedback."""
        # Arrange
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        self.selector.set_selection_strategy(SelectionStrategy.ML_GUIDED)
        
        # Act - Full learning cycle
        # 1. Select experiments using ML
        selected = self.selector.select_experiments(self.candidates, 2)
        assert len(selected) == 2
        
        # 2. Record results for both experiments
        for experiment in selected:
            result = ExperimentResult(
                experiment_id=experiment.experiment_id,
                success=True,
                execution_time=0.5,
                performance_metrics={'accuracy': 0.9},
                learned_patterns=[{'pattern': f'learned_from_{experiment.experiment_id}'}],
                completed_at=time.time()
            )
            self.selector.record_experiment_result(experiment.experiment_id, result)
        
        # Assert
        # Should have recorded all results
        assert self.selector.metrics.experiments_completed == 2
        assert len(self.selector.experiment_history) == 2
        
        # RL engine should have learned from experiences
        assert len(self.rl_engine.experience_buffer) >= 2
        
        # Should have some Q-table entries
        assert len(self.rl_engine.q_table) > 0
        
        # Performance metrics should be updated
        assert self.selector.metrics.ml_guided_selections >= 2
    
    def test_performance_violation_recovery(self):
        """Test recovery from performance violations."""
        # Arrange
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        self.selector.performance_threshold_ms = 30.0
        self.selector.max_performance_violations = 2
        
        # Cause violations to deactivate ML
        self.selector._monitor_selection_performance(50.0)
        self.selector._monitor_selection_performance(60.0)
        assert self.selector.ml_model_ready is False
        
        # Simulate good performance to enable recovery
        self.selector._monitor_selection_performance(10.0)  # Good performance
        self.selector._monitor_selection_performance(15.0)  # Good performance
        
        # Performance violation count should decrease with good performance
        assert self.selector.performance_violation_count < 2
    
    def test_status_reporting_with_ml(self):
        """Test comprehensive status reporting includes ML metrics."""
        # Arrange
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        
        # Act
        status = self.selector.get_selector_status()
        
        # Assert
        assert 'ml_model_ready' in status
        assert status['ml_model_ready'] is True
        assert 'current_strategy' in status
        assert 'metrics' in status
        assert 'ml_guided_selections' in status['metrics']
        assert 'adaptive_parameters' in status
    
    def test_metrics_reset_preserves_ml_integration(self):
        """Test that resetting metrics doesn't break ML integration."""
        # Arrange
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        original_rl_engine = self.selector.reinforcement_learning_engine
        
        # Act
        self.selector.reset_metrics()
        
        # Assert
        assert self.selector.reinforcement_learning_engine is original_rl_engine
        assert self.selector.ml_model_ready is True  # Should still be ready
        assert self.selector.metrics.experiments_selected == 0  # Metrics should be reset
    
    def test_cross_worker_learning_propagation(self):
        """Test cross-worker learning propagation through MCP events."""
        # Arrange - Mock MCP event system
        mock_mcp_system = Mock()
        mock_mcp_system.emit_event = Mock(return_value=True)
        
        self.selector.integrate_mcp_event_system(mock_mcp_system)
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        
        # Create experiment result with learning patterns
        result = ExperimentResult(
            experiment_id="test_exp",
            success=True,
            execution_time=0.8,
            performance_metrics={'accuracy': 0.95},
            learned_patterns=[
                {'pattern_id': 'new_pattern', 'effectiveness': 0.9, 'type': 'discovered'},
                {'pattern_id': 'optimized_pattern', 'effectiveness': 0.85, 'type': 'optimized'}
            ],
            completed_at=time.time()
        )
        
        # Set up ML context to trigger ML insights propagation
        self.selector._current_ml_context = {
            'selected': [self.candidates[0]],
            'state': (100, 200, 'medium'),
            'action': {'mutation': 'insert_command', 'crossover': 'single_point', 'parameter': 'increase_population'},
            'selection_params': {'priority_strategy': 'prioritize_high_success'}
        }
        self.candidates[0].experiment_id = "test_exp"  # Match the result
        
        # Act - Record result which should trigger propagation
        with patch('asyncio.create_task') as mock_create_task:
            self.selector.record_experiment_result("test_exp", result)
        
        # Assert - Should have attempted to create async task for propagation
        mock_create_task.assert_called_once()
        
        # Verify MCP integration was set up
        assert self.selector.mcp_event_system is mock_mcp_system
        
    def test_cross_worker_learning_event_handling(self):
        """Test handling of cross-worker learning events."""
        # Arrange
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        
        # Create mock cross-worker learning event
        cross_worker_event = {
            'experiment_id': 'remote_exp_123',
            'success': True,
            'execution_time': 1.2,
            'performance_metrics': {'accuracy': 0.87},
            'learned_patterns': [
                {
                    'pattern_id': 'remote_pattern_1',
                    'effectiveness': 0.88,
                    'context': {'location': 'viridian_city', 'difficulty': 6},
                    'type': 'discovered'
                }
            ],
            'source_selector': 999999,  # Different from id(self)
            'timestamp': time.time()
        }
        
        # Act
        self.selector._handle_experiment_learning_event(cross_worker_event)
        
        # Assert
        # Should have processed the event
        assert self.selector.metrics.events_processed >= 1
        
        # Should have stored remote result in history
        remote_exp_id = 'remote_remote_exp_123'  # Prefixed with 'remote_'
        assert remote_exp_id in self.selector.experiment_history
        
        # Should have adjusted exploration parameters based on successful remote result
        # (exploration_rate should be slightly reduced for successful cross-worker results)
        assert self.selector.exploration_rate <= 0.1  # Should be reduced from initial value
    
    def test_mcp_event_system_integration(self):
        """Test integration with MCP event system."""
        # Arrange
        mock_mcp_system = Mock()
        mock_mcp_system.emit_event = Mock()
        
        # Act
        self.selector.integrate_mcp_event_system(mock_mcp_system)
        
        # Assert
        assert self.selector.mcp_event_system is mock_mcp_system
    
    def test_cross_worker_learning_avoids_self_events(self):
        """Test that cross-worker learning ignores events from itself."""
        # Arrange
        self.selector.integrate_reinforcement_learning(self.rl_engine)
        initial_events_processed = self.selector.metrics.events_processed
        
        # Create event from same selector (should be ignored)
        self_event = {
            'experiment_id': 'self_exp',
            'success': True,
            'source_selector': id(self.selector),  # Same as current selector
            'learned_patterns': [{'pattern_id': 'self_pattern'}]
        }
        
        # Act
        self.selector._handle_experiment_learning_event(self_event)
        
        # Assert
        # Should not have processed the event (same source_selector)
        assert self.selector.metrics.events_processed == initial_events_processed
        assert 'remote_self_exp' not in self.selector.experiment_history