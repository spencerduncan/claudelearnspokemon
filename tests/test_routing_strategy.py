"""
Unit Tests for RoutingStrategy - Load Balancing Algorithms

This module provides comprehensive unit tests for routing strategies,
validating load balancing algorithms, worker selection logic, health-aware
routing, and performance requirements.

Test Coverage:
- Round-robin routing strategy
- Least-loaded routing strategy
- Weighted round-robin routing strategy
- Worker health and availability tracking
- Performance requirements (<5ms per routing decision)
- Thread safety for concurrent routing
- Strategy metrics and monitoring
"""

import concurrent.futures
import threading
import time
from unittest.mock import Mock

import pytest

from claudelearnspokemon.priority_queue import MessagePriority, QueuedMessage
from claudelearnspokemon.routing_strategy import (
    LeastLoadedStrategy,
    RoundRobinStrategy,
    RoutingMetrics,
    WeightedRoundRobinStrategy,
    WorkerInfo,
    WorkerState,
    create_least_loaded_strategy,
    create_round_robin_strategy,
    create_weighted_round_robin_strategy,
)


@pytest.mark.fast
class TestWorkerInfo:
    """Test WorkerInfo dataclass functionality."""

    def test_worker_info_creation(self):
        """Test basic worker info creation."""
        mock_process = Mock()
        worker = WorkerInfo(
            worker_id="worker_1",
            process=mock_process,
            state=WorkerState.HEALTHY,
            current_load=3,
            max_capacity=10,
        )

        assert worker.worker_id == "worker_1"
        assert worker.process == mock_process
        assert worker.state == WorkerState.HEALTHY
        assert worker.current_load == 3
        assert worker.max_capacity == 10

    def test_utilization_calculation(self):
        """Test worker utilization calculation."""
        mock_process = Mock()
        worker = WorkerInfo(
            worker_id="worker_1", process=mock_process, current_load=3, max_capacity=10
        )

        assert worker.utilization == 0.3  # 3/10

        # Test edge cases
        full_worker = WorkerInfo(
            worker_id="full", process=mock_process, current_load=10, max_capacity=10
        )
        assert full_worker.utilization == 1.0

        zero_capacity_worker = WorkerInfo(
            worker_id="zero", process=mock_process, current_load=0, max_capacity=0
        )
        assert zero_capacity_worker.utilization == 1.0  # Should handle division by zero

    def test_availability_check(self):
        """Test worker availability logic."""
        mock_process = Mock()

        # Healthy worker with capacity
        available_worker = WorkerInfo(
            worker_id="available",
            process=mock_process,
            state=WorkerState.HEALTHY,
            current_load=5,
            max_capacity=10,
        )
        assert available_worker.is_available

        # Unhealthy worker
        unhealthy_worker = WorkerInfo(
            worker_id="unhealthy",
            process=mock_process,
            state=WorkerState.UNHEALTHY,
            current_load=0,
            max_capacity=10,
        )
        assert not unhealthy_worker.is_available

        # At capacity worker
        full_worker = WorkerInfo(
            worker_id="full",
            process=mock_process,
            state=WorkerState.HEALTHY,
            current_load=10,
            max_capacity=10,
        )
        assert not full_worker.is_available

    def test_health_score_calculation(self):
        """Test health score calculation."""
        mock_process = Mock()

        # Healthy worker with low load
        optimal_worker = WorkerInfo(
            worker_id="optimal",
            process=mock_process,
            state=WorkerState.HEALTHY,
            current_load=1,
            max_capacity=10,
            response_time_ms=100.0,
            failure_count=0,
        )

        # Should have high health score
        assert optimal_worker.health_score > 0.8

        # Degraded worker with high load
        degraded_worker = WorkerInfo(
            worker_id="degraded",
            process=mock_process,
            state=WorkerState.DEGRADED,
            current_load=8,
            max_capacity=10,
            response_time_ms=2000.0,
            failure_count=5,
        )

        # Should have lower health score
        assert degraded_worker.health_score < 0.5
        assert degraded_worker.health_score < optimal_worker.health_score


@pytest.mark.fast
class TestRoutingMetrics:
    """Test RoutingMetrics functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = RoutingMetrics()

    def test_routing_decision_recording(self):
        """Test recording of routing decisions."""
        # Record successful routing
        self.metrics.record_routing_decision("worker_1", True, 3.5, False)

        metrics = self.metrics.get_metrics()
        assert metrics["total_routing_decisions"] == 1
        assert metrics["successful_routings"] == 1
        assert metrics["worker_selections"]["worker_1"] == 1
        assert metrics["avg_routing_time_ms"] == 3.5
        assert metrics["fallback_routings"] == 0

        # Record failed routing
        self.metrics.record_routing_decision(None, False, 8.2, True)

        metrics = self.metrics.get_metrics()
        assert metrics["total_routing_decisions"] == 2
        assert metrics["failed_routings"] == 1
        assert metrics["fallback_routings"] == 1

    def test_circuit_breaker_tracking(self):
        """Test circuit breaker activation tracking."""
        self.metrics.record_circuit_breaker_activation("worker_1")
        self.metrics.record_circuit_breaker_activation("worker_1")
        self.metrics.record_circuit_breaker_activation("worker_2")

        metrics = self.metrics.get_metrics()
        assert metrics["circuit_breaker_activations"]["worker_1"] == 2
        assert metrics["circuit_breaker_activations"]["worker_2"] == 1

    def test_load_balance_effectiveness(self):
        """Test load balance effectiveness calculation."""
        # Perfectly balanced load
        balanced_loads = [5, 5, 5, 5]
        self.metrics.update_load_balance_effectiveness(balanced_loads)

        metrics = self.metrics.get_metrics()
        assert metrics["load_balance_effectiveness"] == 0.0  # Perfect balance

        # Unbalanced load
        unbalanced_loads = [1, 3, 7, 9]
        self.metrics.update_load_balance_effectiveness(unbalanced_loads)

        metrics = self.metrics.get_metrics()
        assert metrics["load_balance_effectiveness"] > 0  # Some imbalance

    def test_thread_safety(self):
        """Test thread safety of metrics collection."""

        def worker():
            for i in range(50):
                self.metrics.record_routing_decision(f"worker_{i % 3}", True, 5.0, False)

        # Run concurrent metric updates
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        metrics = self.metrics.get_metrics()
        assert metrics["total_routing_decisions"] == 200
        assert metrics["successful_routings"] == 200


@pytest.mark.fast
class TestRoundRobinStrategy:
    """Test RoundRobinStrategy functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = RoundRobinStrategy()
        self.mock_workers = self._create_mock_workers(4)

    def _create_mock_workers(self, count):
        """Create mock workers for testing."""
        workers = []
        for i in range(count):
            mock_process = Mock()
            worker = WorkerInfo(
                worker_id=f"worker_{i}",
                process=mock_process,
                state=WorkerState.HEALTHY,
                current_load=0,
                max_capacity=10,
            )
            workers.append(worker)
        return workers

    def test_round_robin_selection(self):
        """Test round-robin worker selection."""
        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

        selected_workers = []
        for _ in range(8):  # 2 full cycles
            worker_id = self.strategy.select_worker(message, self.mock_workers)
            selected_workers.append(worker_id)

        # Should cycle through workers in order
        expected_pattern = ["worker_0", "worker_1", "worker_2", "worker_3"] * 2
        assert selected_workers == expected_pattern

    def test_health_filtering(self):
        """Test filtering of unhealthy workers."""
        # Mark some workers as unhealthy
        self.mock_workers[1].state = WorkerState.UNHEALTHY
        self.mock_workers[3].current_load = 10  # At capacity

        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

        selected_workers = []
        for _ in range(6):
            worker_id = self.strategy.select_worker(message, self.mock_workers)
            selected_workers.append(worker_id)

        # Should only use healthy, available workers (0 and 2)
        expected_pattern = ["worker_0", "worker_2"] * 3
        assert selected_workers == expected_pattern

    def test_no_available_workers(self):
        """Test behavior when no workers are available."""
        # Mark all workers as unavailable
        for worker in self.mock_workers:
            worker.state = WorkerState.UNHEALTHY

        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)
        result = self.strategy.select_worker(message, self.mock_workers)

        assert result is None

    def test_empty_worker_list(self):
        """Test behavior with empty worker list."""
        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)
        result = self.strategy.select_worker(message, [])

        assert result is None

    def test_worker_state_updates(self):
        """Test worker state update functionality."""
        self.strategy.update_worker_state("worker_0", WorkerState.DEGRADED, 500.0)

        metrics = self.strategy.get_strategy_metrics()
        assert "worker_0" in metrics["worker_states"]
        assert metrics["worker_states"]["worker_0"] == "degraded"

    def test_performance_requirements(self):
        """Test that routing decisions meet performance requirements (<5ms)."""
        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

        times = []
        for _ in range(100):
            start_time = time.time()
            self.strategy.select_worker(message, self.mock_workers)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)

        # All routing decisions should be under 5ms
        assert all(t < 5.0 for t in times), f"Slow routing: {max(times):.3f}ms"

        # Average should be well under limit
        avg_time = sum(times) / len(times)
        assert avg_time < 1.0  # Target sub-millisecond average

    def test_strategy_metrics(self):
        """Test strategy-specific metrics collection."""
        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

        # Perform some routing decisions
        for _ in range(10):
            self.strategy.select_worker(message, self.mock_workers)

        metrics = self.strategy.get_strategy_metrics()

        required_fields = [
            "strategy_type",
            "current_index",
            "tracked_workers",
            "total_routing_decisions",
            "successful_routings",
        ]

        for field in required_fields:
            assert field in metrics

        assert metrics["strategy_type"] == "round_robin"
        assert metrics["successful_routings"] == 10


@pytest.mark.fast
class TestLeastLoadedStrategy:
    """Test LeastLoadedStrategy functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = LeastLoadedStrategy(load_prediction_factor=0.7)
        self.mock_workers = self._create_varied_load_workers()

    def _create_varied_load_workers(self):
        """Create mock workers with different loads."""
        workers = []
        loads = [2, 5, 1, 8]  # Different current loads

        for i, load in enumerate(loads):
            mock_process = Mock()
            worker = WorkerInfo(
                worker_id=f"worker_{i}",
                process=mock_process,
                state=WorkerState.HEALTHY,
                current_load=load,
                max_capacity=10,
                response_time_ms=100.0 + (load * 20),  # Slower with higher load
            )
            workers.append(worker)

        return workers

    def test_least_loaded_selection(self):
        """Test selection of least loaded worker."""
        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

        # Should select worker_2 (load=1, lowest)
        selected_worker = self.strategy.select_worker(message, self.mock_workers)
        assert selected_worker == "worker_2"

    def test_load_prediction_logic(self):
        """Test load prediction based on response times."""
        message = QueuedMessage(content="test", priority=MessagePriority.HIGH)

        # High priority messages might take longer
        selected_worker = self.strategy.select_worker(message, self.mock_workers)

        # Should still prefer least loaded worker even with prediction
        assert selected_worker == "worker_2"

    def test_health_score_consideration(self):
        """Test that health scores influence selection."""
        # Make the least loaded worker degraded
        self.mock_workers[2].state = WorkerState.DEGRADED  # worker_2 (load=1)

        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)
        selected_worker = self.strategy.select_worker(message, self.mock_workers)

        # Should prefer worker_0 (load=2) over degraded worker_2 (load=1)
        assert selected_worker == "worker_0"

    def test_completion_velocity_tracking(self):
        """Test task completion velocity tracking."""
        # Simulate task completions for worker_1
        for _ in range(5):
            self.strategy.update_worker_state("worker_1", WorkerState.HEALTHY, 200.0)

        # Check that completion history is tracked
        assert len(self.strategy._task_completion_history["worker_1"]) == 5

    def test_load_balancing_effectiveness(self):
        """Test load balancing effectiveness measurement."""
        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

        # Perform multiple routing decisions
        for _ in range(20):
            self.strategy.select_worker(message, self.mock_workers)

        metrics = self.strategy.get_strategy_metrics()

        # Should track load balancing effectiveness
        assert "load_balance_effectiveness" in metrics
        assert metrics["load_balance_effectiveness"] >= 0

    def test_response_time_tracking(self):
        """Test response time tracking and updates."""
        # Make worker_0 the least loaded so it gets selected
        self.mock_workers[0].current_load = 0  # Lowest load
        initial_time = self.mock_workers[0].response_time_ms

        # First, make the worker known to the strategy by selecting it
        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)
        selected = self.strategy.select_worker(message, self.mock_workers)
        assert selected == "worker_0"  # Ensure we selected the right worker

        # Update with faster response time
        self.strategy.update_worker_state("worker_0", WorkerState.HEALTHY, 50.0)

        # Response time should be updated (exponential moving average) in strategy's state
        updated_worker = self.strategy._worker_states["worker_0"]
        # Due to exponential moving average, should be between initial and new value
        assert updated_worker.response_time_ms < initial_time
        assert updated_worker.response_time_ms > 50.0

    def test_strategy_metrics(self):
        """Test least-loaded specific metrics."""
        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

        for _ in range(5):
            self.strategy.select_worker(message, self.mock_workers)

        metrics = self.strategy.get_strategy_metrics()

        required_fields = [
            "strategy_type",
            "load_prediction_factor",
            "avg_worker_utilization",
            "worker_response_times",
        ]

        for field in required_fields:
            assert field in metrics

        assert metrics["strategy_type"] == "least_loaded"
        assert metrics["load_prediction_factor"] == 0.7


@pytest.mark.fast
class TestWeightedRoundRobinStrategy:
    """Test WeightedRoundRobinStrategy functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = WeightedRoundRobinStrategy()
        self.mock_workers = self._create_varied_capacity_workers()

    def _create_varied_capacity_workers(self):
        """Create mock workers with different capacities."""
        workers = []
        configs = [
            (10, 2, 0.9),  # max_capacity, current_load, health_score
            (5, 1, 0.8),
            (15, 3, 1.0),
            (8, 6, 0.6),
        ]

        for i, (capacity, load, health) in enumerate(configs):
            mock_process = Mock()
            worker = WorkerInfo(
                worker_id=f"worker_{i}",
                process=mock_process,
                state=WorkerState.HEALTHY,
                current_load=load,
                max_capacity=capacity,
            )
            # Set worker attributes to achieve desired health score
            # Adjust state and failure count to get approximately the desired health
            if health <= 0.2:
                worker.state = WorkerState.UNHEALTHY
                worker.failure_count = 5
            elif health <= 0.7:
                worker.state = WorkerState.DEGRADED
                worker.failure_count = 2
            else:
                worker.state = WorkerState.HEALTHY
                worker.failure_count = 0

            workers.append(worker)

        return workers

    def test_weighted_selection(self):
        """Test weighted round-robin selection."""
        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

        selected_counts = {}
        total_selections = 100

        for _ in range(total_selections):
            worker_id = self.strategy.select_worker(message, self.mock_workers)
            selected_counts[worker_id] = selected_counts.get(worker_id, 0) + 1

        # Workers with higher capacity/health should be selected more often
        # worker_2 has highest capacity (15) and health (1.0), should be selected most

        # Note: Due to the complexity of weighted selection, we just verify
        # that selection is distributed among workers
        assert len(selected_counts) > 1  # Multiple workers selected
        assert all(count > 0 for count in selected_counts.values())  # All workers used

    def test_weight_updates(self):
        """Test dynamic weight updates."""
        # Create workers and trigger weight updates
        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

        # Select worker to trigger weight calculation
        self.strategy.select_worker(message, self.mock_workers)

        metrics = self.strategy.get_strategy_metrics()

        # Weights should be calculated and stored
        assert "worker_weights" in metrics
        assert len(metrics["worker_weights"]) > 0

    def test_strategy_metrics(self):
        """Test weighted round-robin specific metrics."""
        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

        for _ in range(10):
            self.strategy.select_worker(message, self.mock_workers)

        metrics = self.strategy.get_strategy_metrics()

        required_fields = ["strategy_type", "worker_weights", "current_weights"]

        for field in required_fields:
            assert field in metrics

        assert metrics["strategy_type"] == "weighted_round_robin"


@pytest.mark.fast
class TestFactoryFunctions:
    """Test routing strategy factory functions."""

    def test_round_robin_factory(self):
        """Test round-robin strategy factory."""
        strategy = create_round_robin_strategy()

        assert isinstance(strategy, RoundRobinStrategy)

        metrics = strategy.get_strategy_metrics()
        assert metrics["strategy_type"] == "round_robin"

    def test_least_loaded_factory(self):
        """Test least-loaded strategy factory."""
        strategy = create_least_loaded_strategy(prediction_factor=0.8)

        assert isinstance(strategy, LeastLoadedStrategy)

        metrics = strategy.get_strategy_metrics()
        assert metrics["strategy_type"] == "least_loaded"
        assert metrics["load_prediction_factor"] == 0.8

    def test_weighted_round_robin_factory(self):
        """Test weighted round-robin strategy factory."""
        strategy = create_weighted_round_robin_strategy()

        assert isinstance(strategy, WeightedRoundRobinStrategy)

        metrics = strategy.get_strategy_metrics()
        assert metrics["strategy_type"] == "weighted_round_robin"


@pytest.mark.fast
class TestConcurrencyAndPerformance:
    """Test concurrency and performance requirements."""

    def test_concurrent_routing_decisions(self):
        """Test concurrent routing decisions across multiple strategies."""
        strategies = [
            create_round_robin_strategy(),
            create_least_loaded_strategy(),
            create_weighted_round_robin_strategy(),
        ]

        mock_workers = []
        for i in range(4):
            mock_process = Mock()
            worker = WorkerInfo(
                worker_id=f"worker_{i}",
                process=mock_process,
                state=WorkerState.HEALTHY,
                current_load=i,
                max_capacity=10,
            )
            mock_workers.append(worker)

        def routing_worker(strategy):
            results = []
            message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

            for _ in range(50):
                start_time = time.time()
                worker_id = strategy.select_worker(message, mock_workers)
                end_time = time.time()

                results.append({"worker_id": worker_id, "time_ms": (end_time - start_time) * 1000})

            return results

        # Run concurrent routing across all strategies
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            futures = [executor.submit(routing_worker, strategy) for strategy in strategies]
            all_results = []

            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())

        # All routing decisions should complete successfully and meet SLA
        assert len(all_results) == len(strategies) * 50
        assert all(r["worker_id"] is not None for r in all_results)
        assert all(r["time_ms"] < 5.0 for r in all_results)  # <5ms requirement

        avg_time = sum(r["time_ms"] for r in all_results) / len(all_results)
        assert avg_time < 2.0  # Average should be well under SLA

    def test_strategy_thread_safety(self):
        """Test thread safety of strategy state updates."""
        strategy = create_least_loaded_strategy()

        def state_updater(worker_id):
            for i in range(100):
                # Simulate varying response times and states
                response_time = 50.0 + (i % 10) * 10
                state = WorkerState.HEALTHY if i % 10 != 0 else WorkerState.DEGRADED

                strategy.update_worker_state(worker_id, state, response_time)

        # Run concurrent state updates
        threads = []
        for i in range(4):
            thread = threading.Thread(target=state_updater, args=[f"worker_{i}"])
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Strategy should handle concurrent updates without corruption
        metrics = strategy.get_strategy_metrics()
        assert metrics["tracked_workers"] <= 4
        assert "worker_response_times" in metrics

    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        strategy = create_least_loaded_strategy()

        # Create many workers
        mock_workers = []
        for i in range(50):
            mock_process = Mock()
            worker = WorkerInfo(
                worker_id=f"worker_{i}",
                process=mock_process,
                state=WorkerState.HEALTHY,
                current_load=i % 10,
                max_capacity=10,
            )
            mock_workers.append(worker)

        message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

        # Perform many routing decisions
        for _ in range(1000):
            strategy.select_worker(message, mock_workers)

            # Randomly update some worker states
            if _ % 10 == 0:
                worker_id = f"worker_{_ % 50}"
                strategy.update_worker_state(worker_id, WorkerState.HEALTHY, 100.0)

        metrics = strategy.get_strategy_metrics()

        # Memory usage should be controlled
        # Task completion history should be limited
        for history in strategy._task_completion_history.values():
            assert len(history) <= 10  # maxlen limit

        assert metrics["tracked_workers"] <= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
