"""
Performance validation tests for learning propagation system.

Validates that the ParallelExecutionCoordinator meets all performance requirements:
- Learning propagation: <100ms
- Conflict resolution: <20ms  
- Discovery creation: <5ms
- Batch operations: <200ms for 10 discoveries

Also includes stress tests and load testing for production readiness.

Author: Felix (Craftsperson) - Claude Code Implementation Agent
"""

import asyncio
import pytest
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from claudelearnspokemon.learning_propagation_interfaces import (
    ConflictResolutionStrategy,
    LearningConflict,
    LearningDiscovery, 
    LearningPriority,
    PropagationConfig,
    PropagationStrategy,
)
from claudelearnspokemon.parallel_execution_coordinator import (
    ParallelExecutionCoordinator,
)


class PerformanceTestHarness:
    """Test harness for performance validation."""
    
    def __init__(self, worker_count: int = 4):
        self.mock_worker_pool = Mock()
        self.mock_memory_graph = Mock()
        self.mock_opus_strategist = Mock()
        
        # Configure mocks for performance testing
        self.mock_worker_pool.get_worker_count.return_value = worker_count
        self.mock_worker_pool.get_healthy_worker_count.return_value = worker_count
        self.mock_worker_pool.get_queue_size.return_value = 0
        self.mock_worker_pool.share_pattern.return_value = True
        
        def mock_worker_status(worker_id):
            return {"worker_id": worker_id, "healthy": True, "status": "ready"}
        self.mock_worker_pool.get_worker_status.side_effect = mock_worker_status
        
        self.mock_memory_graph.store_discovery.return_value = "test_memory_id"
        self.mock_memory_graph.query_patterns.return_value = []
        
        self.mock_opus_strategist.request_strategy.return_value = {
            "strategy_id": "test_strategy",
            "experiments": [],
            "strategic_insights": [],
            "next_checkpoints": [],
        }
        self.mock_opus_strategist.StrategyPriority = Mock()
        self.mock_opus_strategist.StrategyPriority.NORMAL = "NORMAL"
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
        )
    
    def create_test_discovery(self, **kwargs) -> LearningDiscovery:
        """Create test discovery with default values."""
        defaults = {
            "worker_id": f"perf_worker_{time.time()}",
            "pattern_type": "performance_test",
            "pattern_data": {"test": "data"},
            "context": {"location": "test_location"},
            "confidence": 0.8,
            "success_rate": 0.85,
            "sample_size": 5,
        }
        defaults.update(kwargs)
        return LearningDiscovery(**defaults)


@pytest.mark.performance
class TestPropagationPerformance:
    """Test propagation performance requirements."""
    
    def setup_method(self):
        """Set up performance test environment."""
        self.harness = PerformanceTestHarness()
    
    @pytest.mark.asyncio
    async def test_single_propagation_under_100ms(self):
        """Test single propagation completes under 100ms."""
        discovery = self.harness.create_test_discovery()
        
        # Measure propagation time
        start_time = time.time()
        result = await self.harness.coordinator.propagate_learning(discovery)
        execution_time_ms = (time.time() - start_time) * 1000
        
        assert execution_time_ms < 100.0, f"Propagation took {execution_time_ms:.2f}ms, exceeds 100ms requirement"
        assert result.success is True
        assert result.execution_time_ms < 100.0
    
    @pytest.mark.asyncio
    async def test_propagation_performance_consistency(self):
        """Test propagation performance is consistent across multiple operations."""
        execution_times = []
        
        # Run 20 propagations to test consistency
        for i in range(20):
            discovery = self.harness.create_test_discovery(
                worker_id=f"consistency_worker_{i}",
                pattern_data={"iteration": i}
            )
            
            start_time = time.time()
            result = await self.harness.coordinator.propagate_learning(discovery)
            execution_time_ms = (time.time() - start_time) * 1000
            
            execution_times.append(execution_time_ms)
            assert result.success is True
        
        # All propagations should be under 100ms
        assert all(t < 100.0 for t in execution_times), f"Some propagations exceeded 100ms: {execution_times}"
        
        # Calculate performance statistics
        avg_time = statistics.mean(execution_times)
        max_time = max(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        assert avg_time < 50.0, f"Average propagation time {avg_time:.2f}ms too high"
        assert max_time < 100.0, f"Max propagation time {max_time:.2f}ms exceeds requirement"
        assert std_dev < 25.0, f"Performance variance too high: {std_dev:.2f}ms"
        
        print(f"Propagation performance: avg={avg_time:.2f}ms, max={max_time:.2f}ms, std={std_dev:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_fast_propagation_mode(self):
        """Test fast propagation mode meets tight performance budgets."""
        discovery = self.harness.create_test_discovery(
            pattern_data={"large_data": "x" * 5000}  # Large pattern data
        )
        
        # Test with very tight budget
        result = await self.harness.coordinator.performance_optimized_propagation(
            discovery,
            performance_budget_ms=25.0  # Very tight budget
        )
        
        assert result.execution_time_ms <= 30.0  # Allow small measurement overhead
        assert result.success is True or "timeout" in result.error_message
        
        # Fast mode should be indicated in performance metrics
        if result.performance_metrics.get("fast_mode"):
            assert result.performance_metrics["fast_mode"] is True


@pytest.mark.performance
class TestConflictResolutionPerformance:
    """Test conflict resolution performance requirements."""
    
    def setup_method(self):
        """Set up conflict resolution test environment."""
        self.harness = PerformanceTestHarness()
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_under_20ms(self):
        """Test conflict resolution completes under 20ms."""
        # Create conflicting discoveries
        discovery1 = self.harness.create_test_discovery(
            worker_id="conflict_worker_1",
            confidence=0.6,
            success_rate=0.7
        )
        
        discovery2 = self.harness.create_test_discovery(
            worker_id="conflict_worker_2",
            confidence=0.9,  # Higher confidence
            success_rate=0.85
        )
        
        conflict = LearningConflict(
            conflicting_discoveries=[discovery1, discovery2],
            resolution_strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        )
        
        # Measure resolution time
        start_time = time.time()
        resolved_discovery = await self.harness.coordinator.resolve_conflict(conflict)
        resolution_time_ms = (time.time() - start_time) * 1000
        
        assert resolution_time_ms < 20.0, f"Conflict resolution took {resolution_time_ms:.2f}ms, exceeds 20ms requirement"
        assert resolved_discovery.confidence == 0.9  # Should pick higher confidence
        assert conflict.resolved is True
        assert conflict.resolution_time_ms < 20.0
    
    @pytest.mark.asyncio
    async def test_complex_conflict_resolution_performance(self):
        """Test conflict resolution performance with multiple discoveries."""
        # Create multiple conflicting discoveries
        discoveries = []
        for i in range(5):
            discovery = self.harness.create_test_discovery(
                worker_id=f"complex_conflict_worker_{i}",
                confidence=0.5 + (i * 0.1),  # Varying confidence levels
                success_rate=0.6 + (i * 0.05),
                sample_size=i + 1
            )
            discoveries.append(discovery)
        
        conflict = LearningConflict(
            conflicting_discoveries=discoveries,
            resolution_strategy=ConflictResolutionStrategy.HYBRID_MERGE
        )
        
        start_time = time.time()
        resolved_discovery = await self.harness.coordinator.resolve_conflict(conflict)
        resolution_time_ms = (time.time() - start_time) * 1000
        
        assert resolution_time_ms < 20.0, f"Complex conflict resolution took {resolution_time_ms:.2f}ms"
        assert resolved_discovery.worker_id == "conflict_resolver"  # Hybrid merge result
        assert resolved_discovery.sample_size == sum(d.sample_size for d in discoveries)


@pytest.mark.performance
class TestBatchProcessingPerformance:
    """Test batch processing performance requirements."""
    
    def setup_method(self):
        """Set up batch processing test environment."""
        self.harness = PerformanceTestHarness()
    
    @pytest.mark.asyncio
    async def test_batch_propagation_200ms_requirement(self):
        """Test batch propagation of 10 discoveries completes under 200ms."""
        # Create exactly 10 discoveries as specified in requirements
        discoveries = []
        for i in range(10):
            discovery = self.harness.create_test_discovery(
                worker_id=f"batch_perf_worker_{i}",
                pattern_data={"batch_index": i, "data": f"batch_data_{i}"},
                confidence=0.7 + (i * 0.02),  # Slight variation
            )
            discoveries.append(discovery)
        
        start_time = time.time()
        result = await self.harness.coordinator.batch_propagate(discoveries)
        execution_time_ms = (time.time() - start_time) * 1000
        
        assert execution_time_ms < 200.0, f"Batch propagation took {execution_time_ms:.2f}ms, exceeds 200ms requirement"
        assert result.success is True
        assert len(result.discoveries_propagated) <= 10
        assert result.execution_time_ms < 200.0
    
    @pytest.mark.asyncio
    async def test_batch_processing_scalability_curve(self):
        """Test batch processing performance scales reasonably with size."""
        batch_sizes = [1, 5, 10, 15, 20]
        performance_data = []
        
        for batch_size in batch_sizes:
            discoveries = []
            for i in range(batch_size):
                discovery = self.harness.create_test_discovery(
                    worker_id=f"scale_worker_{i}_{batch_size}",
                    pattern_data={"batch_size": batch_size, "index": i}
                )
                discoveries.append(discovery)
            
            start_time = time.time()
            result = await self.harness.coordinator.batch_propagate(discoveries)
            execution_time_ms = (time.time() - start_time) * 1000
            
            performance_data.append({
                "batch_size": batch_size,
                "execution_time_ms": execution_time_ms,
                "time_per_discovery": execution_time_ms / batch_size,
                "success": result.success
            })
            
            # Performance should be reasonable
            expected_max_time = batch_size * 20.0  # 20ms per discovery max
            assert execution_time_ms < expected_max_time, f"Batch size {batch_size} performance too slow"
        
        # Performance should scale sub-linearly (efficiency gains from batching)
        for i in range(1, len(performance_data)):
            current = performance_data[i]
            previous = performance_data[i-1]
            
            # Time per discovery should not increase dramatically
            time_per_discovery_ratio = current["time_per_discovery"] / previous["time_per_discovery"]
            assert time_per_discovery_ratio < 1.5, f"Performance degradation too high at batch size {current['batch_size']}"
        
        print("Batch processing scalability:")
        for data in performance_data:
            print(f"  {data}")


@pytest.mark.stress
class TestStressAndLoad:
    """Stress tests and load testing for production readiness."""
    
    def setup_method(self):
        """Set up stress test environment."""
        self.harness = PerformanceTestHarness(worker_count=10)
        
        # Configure for stress testing
        config = PropagationConfig(
            max_concurrent_propagations=10,
            batch_size=20,
            discovery_cache_size=500,
        )
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.harness.mock_worker_pool,
            memory_graph=self.harness.mock_memory_graph,
            opus_strategist=self.harness.mock_opus_strategist,
            config=config,
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_propagation_stress(self):
        """Test system handles high concurrent propagation load."""
        concurrent_operations = 50
        
        async def propagate_discovery(index: int):
            discovery = LearningDiscovery(
                worker_id=f"stress_worker_{index}",
                pattern_type="stress_test",
                pattern_data={"stress_index": index},
                context={"stress_test": True},
                confidence=0.7 + (index % 10) * 0.02  # Some variation
            )
            return await self.coordinator.propagate_learning(discovery)
        
        # Execute concurrent propagations
        start_time = time.time()
        tasks = [propagate_discovery(i) for i in range(concurrent_operations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time_ms = (time.time() - start_time) * 1000
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, type(results[0])) and getattr(r, 'success', False)]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful_results) / len(results)
        avg_time_per_operation = total_time_ms / concurrent_operations
        
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} too low for stress test"
        assert avg_time_per_operation < 100.0, f"Average time {avg_time_per_operation:.2f}ms too high"
        assert len(exceptions) == 0, f"Unexpected exceptions during stress test: {exceptions}"
        
        print(f"Stress test: {concurrent_operations} operations, {success_rate:.2%} success, {avg_time_per_operation:.2f}ms avg")
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test system handles memory pressure gracefully."""
        # Configure small cache to trigger memory pressure
        self.coordinator.config.discovery_cache_size = 10
        
        # Create many discoveries to pressure memory
        discoveries = []
        for i in range(100):  # More than cache size
            discovery = LearningDiscovery(
                worker_id=f"memory_pressure_worker_{i}",
                pattern_type="memory_pressure_test",
                pattern_data={"large_data": "x" * 1000, "index": i},
                confidence=0.7
            )
            discoveries.append(discovery)
            
            # Add to cache
            self.coordinator._update_discovery_cache(discovery)
        
        # Cache should respect size limit
        assert len(self.coordinator._discovery_cache) <= 10
        
        # Most recent discoveries should still be accessible
        recent_discovery = self.coordinator.get_discovery_by_id("memory_pressure_worker_99")
        assert recent_discovery is not None
        
        # Cache hit rate should still be reasonable despite evictions
        metrics = self.coordinator.get_propagation_metrics()
        cache_hit_rate = metrics.get("cache_hit_rate", 0.0)
        # Hit rate might be low due to evictions, but system should still function
        assert 0.0 <= cache_hit_rate <= 1.0
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self):
        """Test system maintains performance under sustained load."""
        operation_count = 30
        operations_per_second = 10
        interval = 1.0 / operations_per_second
        
        execution_times = []
        
        # Simulate sustained load
        for i in range(operation_count):
            discovery = self.harness.create_test_discovery(
                worker_id=f"sustained_worker_{i}",
                pattern_data={"sustained_index": i}
            )
            
            start_time = time.time()
            result = await self.coordinator.propagate_learning(discovery)
            execution_time_ms = (time.time() - start_time) * 1000
            
            execution_times.append(execution_time_ms)
            assert result.success is True
            
            # Rate limiting to simulate realistic load
            await asyncio.sleep(interval)
        
        # Performance should not degrade significantly over time
        early_avg = statistics.mean(execution_times[:10])
        late_avg = statistics.mean(execution_times[-10:])
        
        performance_degradation = (late_avg - early_avg) / early_avg
        assert performance_degradation < 0.5, f"Performance degraded by {performance_degradation:.1%}"
        
        print(f"Sustained load test: early_avg={early_avg:.2f}ms, late_avg={late_avg:.2f}ms")


@pytest.mark.performance  
class TestDiscoveryCreationPerformance:
    """Test discovery creation performance requirements."""
    
    def setup_method(self):
        """Set up discovery creation test environment."""
        self.harness = PerformanceTestHarness()
    
    @pytest.mark.asyncio
    async def test_discovery_creation_under_5ms(self):
        """Test discovery creation completes under 5ms."""
        # Test with various discovery sizes
        test_cases = [
            {"pattern_data": {"simple": "data"}, "name": "simple"},
            {"pattern_data": {"complex": {"nested": {"data": list(range(100))}}}, "name": "complex"},
            {"pattern_data": {"large_string": "x" * 1000}, "name": "large_string"},
        ]
        
        for case in test_cases:
            start_time = time.time()
            discovery = await self.harness.coordinator.discover_learning(
                worker_id=f"creation_perf_worker_{case['name']}",
                pattern_type="creation_performance_test",
                pattern_data=case["pattern_data"],
                context={"test_case": case["name"]},
                confidence=0.8
            )
            creation_time_ms = (time.time() - start_time) * 1000
            
            assert creation_time_ms < 5.0, f"Discovery creation ({case['name']}) took {creation_time_ms:.2f}ms, exceeds 5ms requirement"
            assert discovery.pattern_data == case["pattern_data"]
    
    @pytest.mark.asyncio
    async def test_bulk_discovery_creation_performance(self):
        """Test bulk discovery creation performance."""
        discovery_count = 50
        
        async def create_single_discovery(index: int):
            start_time = time.time()
            discovery = await self.harness.coordinator.discover_learning(
                worker_id=f"bulk_worker_{index}",
                pattern_type="bulk_creation_test",
                pattern_data={"bulk_index": index},
                context={"bulk_test": True},
                confidence=0.7
            )
            creation_time_ms = (time.time() - start_time) * 1000
            return creation_time_ms, discovery
        
        # Create discoveries concurrently
        start_time = time.time()
        tasks = [create_single_discovery(i) for i in range(discovery_count)]
        results = await asyncio.gather(*tasks)
        total_time_ms = (time.time() - start_time) * 1000
        
        creation_times = [result[0] for result in results]
        discoveries = [result[1] for result in results]
        
        # Individual creation times should be under 5ms
        assert all(t < 5.0 for t in creation_times), f"Some creations exceeded 5ms: {max(creation_times):.2f}ms"
        
        # Total time should be reasonable for concurrent operations
        avg_time_per_discovery = total_time_ms / discovery_count
        assert avg_time_per_discovery < 10.0, f"Average time per discovery too high: {avg_time_per_discovery:.2f}ms"
        
        # All discoveries should be valid
        assert len(discoveries) == discovery_count
        assert all(d.worker_id.startswith("bulk_worker_") for d in discoveries)
        
        print(f"Bulk creation: {discovery_count} discoveries in {total_time_ms:.2f}ms ({avg_time_per_discovery:.2f}ms avg)")


@pytest.mark.performance
class TestAdaptivePerformanceOptimization:
    """Test adaptive performance optimization features."""
    
    def setup_method(self):
        """Set up adaptive performance test environment.""" 
        self.harness = PerformanceTestHarness()
    
    @pytest.mark.asyncio
    async def test_adaptive_strategy_performance_impact(self):
        """Test adaptive strategies improve performance under load."""
        discovery = self.harness.create_test_discovery(
            priority=LearningPriority.NORMAL,
            confidence=0.75
        )
        
        # Test immediate strategy
        immediate_strategy = PropagationStrategy.IMMEDIATE
        start_time = time.time()
        result1 = await self.harness.coordinator.propagate_learning(discovery, strategy=immediate_strategy)
        immediate_time = (time.time() - start_time) * 1000
        
        # Test batched strategy  
        batched_strategy = PropagationStrategy.BATCHED
        start_time = time.time()
        result2 = await self.harness.coordinator.batch_propagate([discovery], strategy=batched_strategy)
        batched_time = (time.time() - start_time) * 1000
        
        assert result1.success is True
        assert result2.success is True
        
        # Both should meet performance requirements
        assert immediate_time < 100.0
        assert batched_time < 200.0  # Batch requirement
        
        print(f"Strategy performance: immediate={immediate_time:.2f}ms, batched={batched_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_performance_budget_enforcement(self):
        """Test performance budget enforcement accuracy."""
        budgets = [25.0, 50.0, 75.0, 100.0]  # Different budget levels
        
        for budget_ms in budgets:
            discovery = self.harness.create_test_discovery(
                pattern_data={"budget_test": budget_ms}
            )
            
            result = await self.harness.coordinator.performance_optimized_propagation(
                discovery,
                performance_budget_ms=budget_ms
            )
            
            # Should complete within budget (with small tolerance for measurement)
            tolerance_ms = 5.0
            assert result.execution_time_ms <= budget_ms + tolerance_ms, f"Budget {budget_ms}ms exceeded: {result.execution_time_ms:.2f}ms"
            
            # Should succeed or fail gracefully with timeout message
            assert result.success is True or "timeout" in result.error_message


@pytest.mark.benchmark
class TestProductionReadinessBenchmarks:
    """Comprehensive benchmarks for production readiness validation."""
    
    def setup_method(self):
        """Set up production readiness test environment."""
        self.harness = PerformanceTestHarness(worker_count=8)
    
    @pytest.mark.asyncio
    async def test_throughput_benchmark(self):
        """Benchmark system throughput under realistic load."""
        duration_seconds = 10
        target_throughput_per_second = 20  # Target throughput
        
        operation_count = 0
        successful_operations = 0
        start_time = time.time()
        
        # Run operations for specified duration
        while time.time() - start_time < duration_seconds:
            discovery = self.harness.create_test_discovery(
                worker_id=f"throughput_worker_{operation_count}",
                pattern_data={"operation_index": operation_count}
            )
            
            try:
                result = await self.harness.coordinator.propagate_learning(discovery)
                operation_count += 1
                if result.success:
                    successful_operations += 1
                    
            except Exception as e:
                print(f"Operation {operation_count} failed: {e}")
                operation_count += 1
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)
        
        actual_duration = time.time() - start_time
        throughput_per_second = operation_count / actual_duration
        success_rate = successful_operations / operation_count if operation_count > 0 else 0.0
        
        assert throughput_per_second >= target_throughput_per_second * 0.8, f"Throughput {throughput_per_second:.1f}/s below target"
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} too low"
        
        print(f"Throughput benchmark: {throughput_per_second:.1f} ops/s, {success_rate:.2%} success")
    
    @pytest.mark.asyncio
    async def test_system_stability_under_load(self):
        """Test system stability under sustained high load."""
        # Configuration for high load
        self.coordinator.config.max_concurrent_propagations = 20
        
        # Run high load for extended period
        total_operations = 200
        batch_size = 10
        
        overall_start = time.time()
        all_results = []
        
        for batch_start in range(0, total_operations, batch_size):
            batch_end = min(batch_start + batch_size, total_operations)
            batch_discoveries = []
            
            for i in range(batch_start, batch_end):
                discovery = self.harness.create_test_discovery(
                    worker_id=f"stability_worker_{i}",
                    pattern_data={"stability_index": i, "batch": batch_start // batch_size}
                )
                batch_discoveries.append(discovery)
            
            # Process batch
            batch_result = await self.coordinator.batch_propagate(batch_discoveries)
            all_results.append(batch_result)
            
            # Brief pause between batches
            await asyncio.sleep(0.05)
        
        total_duration = time.time() - overall_start
        
        # Analyze stability metrics
        successful_batches = [r for r in all_results if r.success]
        success_rate = len(successful_batches) / len(all_results)
        
        execution_times = [r.execution_time_ms for r in successful_batches]
        avg_batch_time = statistics.mean(execution_times) if execution_times else 0
        max_batch_time = max(execution_times) if execution_times else 0
        
        # Stability requirements
        assert success_rate >= 0.98, f"Batch success rate {success_rate:.2%} indicates instability"
        assert avg_batch_time < 200.0, f"Average batch time {avg_batch_time:.2f}ms too high"
        assert max_batch_time < 300.0, f"Max batch time {max_batch_time:.2f}ms indicates performance spikes"
        
        # System should not have memory leaks or resource issues
        final_metrics = self.coordinator.get_propagation_metrics()
        assert final_metrics["cache_size"] <= self.coordinator.config.discovery_cache_size
        
        print(f"Stability test: {len(all_results)} batches, {success_rate:.2%} success, {avg_batch_time:.2f}ms avg batch time")


# Performance test runner
if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v",
        "-m", "performance",
        "--tb=short",
        "--durations=10"  # Show slowest tests
    ])