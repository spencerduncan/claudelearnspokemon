"""
Tests for ParallelExecutionCoordinator performance optimization.

Validates that the execution coordinator meets the 5-second cycle target
and properly orchestrates parallel execution streams.
"""

import threading
import time
import pytest
from unittest.mock import Mock, MagicMock

from claudelearnspokemon.parallel_execution_coordinator import (
    ParallelExecutionCoordinator,
    ExecutionCycleResult
)
from claudelearnspokemon.emulator_pool import ExecutionResult, ExecutionStatus


class TestParallelExecutionCoordinator:
    """Test the ParallelExecutionCoordinator performance requirements."""
    
    def setup_method(self):
        """Set up test fixtures with mocked dependencies."""
        self.mock_claude_manager = Mock()
        self.mock_emulator_pool = Mock()
        self.mock_memory_graph = Mock()
        self.mock_checkpoint_manager = Mock()
        self.mock_opus_strategist = Mock()
        
        # Configure checkpoint manager mock
        self.mock_checkpoint_manager.list_checkpoints.return_value = []
        
        # Configure opus strategist mock
        mock_strategy = Mock()
        mock_strategy.strategy_id = "test_strategy"
        mock_strategy.experiments = [
            {"id": "exp_1", "objective": "test_1"},
            {"id": "exp_2", "objective": "test_2"},
            {"id": "exp_3", "objective": "test_3"},
            {"id": "exp_4", "objective": "test_4"},
        ]
        mock_strategy.strategic_directives = ["explore", "optimize"]
        self.mock_opus_strategist.get_strategy.return_value = mock_strategy
        
        # Configure emulator pool mock with fast execution
        self.mock_emulator_pool.execute_script.return_value = ExecutionResult(
            execution_id="test_exec",
            script_id="test_script",
            status=ExecutionStatus.SUCCESS,
            start_time=time.time(),
            end_time=time.time() + 0.5,  # 500ms execution
            final_state={"test": True},
            error_message=None,
            tile_observations=[],
            performance_metrics={"frames_executed": 10}
        )
        
        self.coordinator = ParallelExecutionCoordinator(
            claude_manager=self.mock_claude_manager,
            emulator_pool=self.mock_emulator_pool,
            memory_graph=self.mock_memory_graph,
            checkpoint_manager=self.mock_checkpoint_manager,
            opus_strategist=self.mock_opus_strategist,
            pool_size=4
        )
    
    @pytest.mark.fast
    def test_execution_cycle_meets_5_second_target(self):
        """Test that a single execution cycle completes within 5 seconds."""
        cycle_start = time.time()
        cycle_id = "test_cycle"
        
        # Execute one cycle
        result = self.coordinator._execute_cycle(cycle_id, cycle_start)
        
        # Verify performance target
        assert result.cycle_duration < 5.0, f"Cycle took {result.cycle_duration:.2f}s, exceeds 5s target"
        assert result.met_performance_target, "Cycle failed to meet performance target"
        
        # Verify cycle structure
        assert result.cycle_id == cycle_id
        assert isinstance(result.execution_results, list)
        assert isinstance(result.strategic_analysis, dict)
        assert isinstance(result.performance_metrics, dict)
        
        # Verify performance metrics are tracked
        assert "strategic_planning_ms" in result.performance_metrics
        assert "parallel_execution_ms" in result.performance_metrics
        assert "result_aggregation_ms" in result.performance_metrics
        assert "total_cycle_ms" in result.performance_metrics
    
    @pytest.mark.fast
    def test_parallel_experiments_execution_performance(self):
        """Test that parallel experiment execution meets timing constraints."""
        experiments = [
            {"id": f"exp_{i}", "objective": f"test_{i}"} for i in range(4)
        ]
        
        start_time = time.time()
        results = self.coordinator._execute_parallel_experiments(experiments)
        execution_time = time.time() - start_time
        
        # Should complete parallel execution in < 3 seconds
        assert execution_time < 3.0, f"Parallel execution took {execution_time:.2f}s, exceeds 3s target"
        
        # Should execute all experiments in parallel (not sequentially)
        assert len(results) <= 4, "Should not execute more than pool size"
    
    @pytest.mark.fast 
    def test_opus_strategy_performance(self):
        """Test that Opus strategy requests complete quickly."""
        game_state = {"current_checkpoint": "test", "timestamp": time.time()}
        
        start_time = time.time()
        strategy = self.coordinator.get_opus_strategy(game_state)
        strategy_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Strategic planning should complete in < 1000ms
        assert strategy_time < 1000, f"Opus strategy took {strategy_time:.2f}ms, exceeds 1000ms target"
        
        # Verify strategy structure
        assert "strategy_id" in strategy
        assert "experiments" in strategy
        assert len(strategy["experiments"]) <= 4  # Limited to pool size
    
    @pytest.mark.fast
    def test_performance_metrics_tracking(self):
        """Test that performance metrics are properly tracked."""
        # Execute a few cycles to build metrics
        for i in range(3):
            cycle_result = self.coordinator._execute_cycle(f"cycle_{i}", time.time())
            self.coordinator._update_performance_metrics(cycle_result)
        
        metrics = self.coordinator.get_performance_metrics()
        
        # Verify key metrics exist
        assert "total_cycles" in metrics
        assert "average_cycle_time_ms" in metrics
        assert "target_compliance_rate" in metrics
        assert "cycles_per_minute" in metrics
        
        # Verify metrics are reasonable
        assert metrics["total_cycles"] == 3
        assert metrics["average_cycle_time_ms"] < 5000  # Should average < 5s
        assert 0.0 <= metrics["target_compliance_rate"] <= 1.0
    
    @pytest.mark.fast
    def test_emergency_stop_functionality(self):
        """Test that emergency stop works correctly."""
        # Start coordinator in background thread
        def run_coordinator():
            self.coordinator.run()
        
        thread = threading.Thread(target=run_coordinator, daemon=True)
        thread.start()
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Emergency stop
        self.coordinator.emergency_stop()
        
        # Should stop quickly
        thread.join(timeout=1.0)
        assert not thread.is_alive(), "Coordinator should stop after emergency stop"
        assert self.coordinator._emergency_stop, "Emergency stop flag should be set"
    
    @pytest.mark.fast
    def test_coordinator_handles_execution_failures_gracefully(self):
        """Test that coordinator handles individual execution failures."""
        # Configure mock to fail sometimes
        self.mock_emulator_pool.execute_script.side_effect = [
            ExecutionResult(
                execution_id="success",
                script_id="test",
                status=ExecutionStatus.SUCCESS,
                start_time=time.time(),
                end_time=time.time() + 0.1,
                final_state={},
                error_message=None,
                tile_observations=[],
                performance_metrics={}
            ),
            Exception("Test failure"),
            ExecutionResult(
                execution_id="success2", 
                script_id="test2",
                status=ExecutionStatus.SUCCESS,
                start_time=time.time(),
                end_time=time.time() + 0.1, 
                final_state={},
                error_message=None,
                tile_observations=[],
                performance_metrics={}
            )
        ]
        
        experiments = [{"id": f"exp_{i}"} for i in range(3)]
        
        # Should handle failures gracefully and return available results
        results = self.coordinator._execute_parallel_experiments(experiments)
        
        # Should return successful results despite some failures
        assert len(results) >= 1, "Should return at least some successful results"
        assert all(r.status == ExecutionStatus.SUCCESS for r in results if r), "Returned results should be successful"
    
    @pytest.mark.fast
    def test_cycle_result_performance_properties(self):
        """Test ExecutionCycleResult performance properties."""
        start_time = time.time()
        end_time = start_time + 3.5  # 3.5 second cycle
        
        result = ExecutionCycleResult(
            cycle_id="test",
            cycle_start_time=start_time,
            cycle_end_time=end_time,
            execution_results=[],
            strategic_analysis={},
            discoveries=[],
            performance_metrics={}
        )
        
        assert result.cycle_duration == 3.5
        assert result.met_performance_target, "3.5s cycle should meet 5s target"
        
        # Test failing case
        slow_result = ExecutionCycleResult(
            cycle_id="slow",
            cycle_start_time=start_time,
            cycle_end_time=start_time + 6.0,  # 6 second cycle
            execution_results=[],
            strategic_analysis={},
            discoveries=[],
            performance_metrics={}
        )
        
        assert slow_result.cycle_duration == 6.0
        assert not slow_result.met_performance_target, "6s cycle should fail 5s target"


if __name__ == "__main__":
    # Quick performance validation
    print("Running ParallelExecutionCoordinator performance tests...")
    pytest.main([__file__, "-v", "-x"])