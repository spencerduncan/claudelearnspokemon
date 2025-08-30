"""
Comprehensive test suite for ParallelExecutionCoordinator learning propagation system.

Test Categories:
- TestLearningPropagationBasics: Core functionality and initialization
- TestDiscoveryManagement: Discovery creation, validation, and caching  
- TestConflictResolution: Conflict detection and resolution strategies
- TestSystemIntegration: Integration with SonnetWorkerPool, MemoryGraph, OpusStrategist
- TestPerformanceOptimization: Performance benchmarks and optimization features
- TestAdvancedFeatures: Adaptive strategies, circuit breaker, batch processing
- TestMCPIntegration: MCP memory integration and storage
- TestObserverPattern: Event notification and observer management

Performance Requirements:
- Learning propagation: <100ms
- Conflict resolution: <20ms
- Discovery creation: <5ms
- Batch operations: <200ms for 10 discoveries

Author: Felix (Craftsperson) - Claude Code Implementation Agent
"""

import asyncio
import pytest
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock, patch

from claudelearnspokemon.learning_propagation_interfaces import (
    ConflictResolutionStrategy,
    LearningConflict,
    LearningDiscovery,
    LearningPriority,
    LearningPropagationError,
    PropagationConfig,
    PropagationResult,
    PropagationStrategy,
    PropagationTimeoutError,
)
from claudelearnspokemon.parallel_execution_coordinator import (
    LearningPropagationObserver,
    ParallelExecutionCoordinator,
)


class MockSonnetWorkerPool:
    """Mock SonnetWorkerPool for testing."""
    
    def __init__(self, worker_count: int = 4):
        self.worker_count = worker_count
        self.healthy_count = worker_count
        self.queue_size = 0
        self.share_pattern_success = True
        self.share_pattern_calls = []
    
    def get_worker_count(self) -> int:
        return self.worker_count
    
    def get_healthy_worker_count(self) -> int:
        return self.healthy_count
    
    def get_queue_size(self) -> int:
        return self.queue_size
    
    def get_worker_status(self, worker_id: str) -> dict:
        return {
            "worker_id": worker_id,
            "healthy": True,
            "status": "ready",
        }
    
    def share_pattern(self, pattern_data: dict, discovered_by: str = None) -> bool:
        self.share_pattern_calls.append({
            "pattern_data": pattern_data,
            "discovered_by": discovered_by,
            "timestamp": time.time()
        })
        return self.share_pattern_success


class MockMemoryGraph:
    """Mock MemoryGraph for testing."""
    
    def __init__(self):
        self.stored_discoveries = []
        self.query_results = []
        self.store_success = True
    
    def store_discovery(self, discovery_dict: dict) -> str:
        if not self.store_success:
            raise Exception("Mock storage failure")
        
        discovery_id = discovery_dict.get("discovery_id", f"stored_{uuid.uuid4().hex[:8]}")
        self.stored_discoveries.append(discovery_dict)
        return discovery_id
    
    def query_patterns(self, criteria: dict) -> list:
        return self.query_results


class MockOpusStrategist:
    """Mock OpusStrategist for testing."""
    
    def __init__(self):
        self.request_strategy_calls = []
        self.StrategyPriority = Mock()
        self.StrategyPriority.NORMAL = "NORMAL"
    
    def request_strategy(self, game_state: dict, recent_results: list, priority) -> dict:
        self.request_strategy_calls.append({
            "game_state": game_state,
            "recent_results": recent_results,
            "priority": priority,
            "timestamp": time.time()
        })
        
        return {
            "strategy_id": f"strategic_{int(time.time())}",
            "experiments": [],
            "strategic_insights": ["Mock strategic insight"],
            "next_checkpoints": ["mock_checkpoint"],
        }


@pytest.mark.fast
class TestLearningPropagationBasics:
    """Test basic learning propagation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_worker_pool = MockSonnetWorkerPool()
        self.mock_memory_graph = MockMemoryGraph()
        self.mock_opus_strategist = MockOpusStrategist()
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
        )
    
    def test_initialization(self):
        """Test coordinator initializes correctly."""
        assert isinstance(self.coordinator, ParallelExecutionCoordinator)
        assert self.coordinator.worker_pool is self.mock_worker_pool
        assert self.coordinator.memory_graph is self.mock_memory_graph
        assert self.coordinator.opus_strategist is self.mock_opus_strategist
        assert isinstance(self.coordinator.config, PropagationConfig)
        assert isinstance(self.coordinator.metrics, type(self.coordinator.metrics))
    
    def test_invalid_configuration(self):
        """Test coordinator rejects invalid configuration."""
        invalid_config = PropagationConfig(
            max_propagation_time_ms=-1,  # Invalid negative time
            min_confidence_threshold=1.5,  # Invalid confidence > 1.0
        )
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            ParallelExecutionCoordinator(
                worker_pool=self.mock_worker_pool,
                memory_graph=self.mock_memory_graph,
                opus_strategist=self.mock_opus_strategist,
                config=invalid_config,
            )
    
    @pytest.mark.asyncio
    async def test_discovery_creation_performance(self):
        """Test discovery creation meets <5ms performance requirement."""
        start_time = time.time()
        
        discovery = await self.coordinator.discover_learning(
            worker_id="test_worker",
            pattern_type="movement",
            pattern_data={"action": "move_right", "frames": 10},
            context={"location": "pallet_town"},
            confidence=0.8,
            success_rate=0.9,
            sample_size=5
        )
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        assert execution_time_ms < 5.0, f"Discovery creation took {execution_time_ms:.2f}ms, exceeds 5ms target"
        assert discovery.worker_id == "test_worker"
        assert discovery.pattern_type == "movement"
        assert discovery.confidence == 0.8
        assert discovery.success_rate == 0.9
        assert discovery.sample_size == 5


@pytest.mark.medium
class TestConflictResolution:
    """Test conflict detection and resolution functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_worker_pool = MockSonnetWorkerPool()
        self.mock_memory_graph = MockMemoryGraph()
        self.mock_opus_strategist = MockOpusStrategist()
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
        )
    
    @pytest.mark.asyncio
    async def test_conflict_detection(self):
        """Test conflict detection between discoveries."""
        # Set up conflicting patterns in memory graph
        self.mock_memory_graph.query_results = [
            {
                "discovery_id": "existing_discovery",
                "pattern_type": "movement",
                "location": "pallet_town",
                "confidence": 0.3,  # Conflicts with high confidence discovery
                "success_rate": 0.3,
            }
        ]
        
        # Create high confidence discovery that should conflict
        discovery = LearningDiscovery(
            worker_id="test_worker",
            pattern_type="movement",
            context={"location": "pallet_town"},
            confidence=0.9,  # Significant difference from existing
            success_rate=0.9,
            sample_size=5
        )
        
        conflict = await self.coordinator._detect_conflicts(discovery)
        
        assert conflict is not None
        assert len(conflict.conflicting_discoveries) >= 2
        assert conflict.resolution_strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_highest_confidence(self):
        """Test conflict resolution using highest confidence strategy."""
        # Create conflicting discoveries
        discovery1 = LearningDiscovery(
            worker_id="worker1",
            pattern_type="movement",
            context={"location": "test_location"},
            confidence=0.6,
            success_rate=0.7
        )
        
        discovery2 = LearningDiscovery(
            worker_id="worker2", 
            pattern_type="movement",
            context={"location": "test_location"},
            confidence=0.9,  # Higher confidence
            success_rate=0.8
        )
        
        conflict = LearningConflict(
            conflicting_discoveries=[discovery1, discovery2],
            resolution_strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        )
        
        start_time = time.time()
        resolved_discovery = await self.coordinator.resolve_conflict(conflict)
        resolution_time_ms = (time.time() - start_time) * 1000
        
        assert resolution_time_ms < 20.0, f"Conflict resolution took {resolution_time_ms:.2f}ms, exceeds 20ms target"
        assert resolved_discovery.confidence == 0.9  # Should pick higher confidence discovery
        assert resolved_discovery.worker_id == "worker2"
        assert conflict.resolved is True
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_merge(self):
        """Test conflict resolution using hybrid merge strategy."""
        discovery1 = LearningDiscovery(
            worker_id="worker1",
            pattern_type="movement",
            confidence=0.7,
            success_rate=0.8,
            sample_size=10
        )
        
        discovery2 = LearningDiscovery(
            worker_id="worker2",
            pattern_type="movement", 
            confidence=0.8,
            success_rate=0.6,
            sample_size=5
        )
        
        conflict = LearningConflict(
            conflicting_discoveries=[discovery1, discovery2],
            resolution_strategy=ConflictResolutionStrategy.HYBRID_MERGE
        )
        
        resolved_discovery = await self.coordinator.resolve_conflict(conflict)
        
        assert resolved_discovery.worker_id == "conflict_resolver"
        assert resolved_discovery.sample_size == 15  # Combined sample sizes
        assert 0.7 <= resolved_discovery.confidence <= 0.8  # Weighted average
        assert "merged_from" in resolved_discovery.pattern_data


@pytest.mark.medium  
class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_worker_pool = MockSonnetWorkerPool()
        self.mock_memory_graph = MockMemoryGraph()
        self.mock_opus_strategist = MockOpusStrategist()
        
        # Use performance-focused configuration
        config = PropagationConfig(
            max_propagation_time_ms=50.0,  # Stricter timing
            batch_size=5,
            max_concurrent_propagations=3,
        )
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
            config=config,
        )
    
    @pytest.mark.asyncio
    async def test_propagation_performance_requirement(self):
        """Test propagation meets <100ms performance requirement."""
        discovery = LearningDiscovery(
            worker_id="perf_test_worker",
            pattern_type="battle",
            pattern_data={"move": "tackle", "effectiveness": 0.8},
            context={"location": "route_1"},
            confidence=0.8,
            success_rate=0.85,
            sample_size=10
        )
        
        start_time = time.time()
        result = await self.coordinator.propagate_learning(discovery)
        execution_time_ms = (time.time() - start_time) * 1000
        
        assert execution_time_ms < 100.0, f"Propagation took {execution_time_ms:.2f}ms, exceeds 100ms requirement"
        assert result.success is True
        assert result.execution_time_ms < 100.0
    
    @pytest.mark.asyncio
    async def test_performance_optimized_propagation(self):
        """Test performance-optimized propagation with budget enforcement."""
        discovery = LearningDiscovery(
            worker_id="budget_test_worker",
            pattern_type="exploration",
            pattern_data={"strategy": "systematic_exploration"},
            context={"location": "viridian_forest"},
            confidence=0.75,
            success_rate=0.8
        )
        
        # Test with tight budget
        result = await self.coordinator.performance_optimized_propagation(
            discovery, 
            performance_budget_ms=25.0
        )
        
        assert result.execution_time_ms <= 30.0  # Allow small buffer for measurement overhead
        # Should succeed even with tight budget using fast propagation
        assert result.success is True or "timeout" in result.error_message
    
    @pytest.mark.asyncio
    async def test_adaptive_strategy_selection(self):
        """Test adaptive propagation strategy selection."""
        discovery = LearningDiscovery(
            worker_id="adaptive_test_worker",
            pattern_type="navigation", 
            confidence=0.95,  # High confidence
            success_rate=0.9,
            priority=LearningPriority.HIGH
        )
        
        # Test under low load
        self.mock_worker_pool.queue_size = 0
        strategy = await self.coordinator.adaptive_propagation_strategy(
            discovery, 
            system_load={"cpu": 0.3, "memory": 0.4}
        )
        assert strategy == PropagationStrategy.IMMEDIATE  # High confidence -> immediate
        
        # Test under high load
        self.mock_worker_pool.queue_size = 15
        self.mock_worker_pool.healthy_count = 2
        strategy = await self.coordinator.adaptive_propagation_strategy(
            discovery,
            system_load={"cpu": 0.9, "memory": 0.8}
        )
        assert strategy == PropagationStrategy.BATCHED  # High load -> batched
    
    def test_propagation_time_estimation(self):
        """Test propagation time estimation accuracy."""
        simple_discovery = LearningDiscovery(
            pattern_type="simple_move",
            pattern_data={"move": "up"},
            worker_id="test_worker"
        )
        
        complex_discovery = LearningDiscovery(
            pattern_type="complex_battle",
            pattern_data={"battle_strategy": "x" * 1000},  # Large pattern data
            worker_id="test_worker"
        )
        
        simple_estimate = self.coordinator._estimate_propagation_time(simple_discovery)
        complex_estimate = self.coordinator._estimate_propagation_time(complex_discovery)
        
        assert complex_estimate > simple_estimate
        assert 10.0 <= simple_estimate <= 100.0  # Reasonable range
        assert 20.0 <= complex_estimate <= 200.0  # Reasonable range for complex


@pytest.mark.medium
class TestBatchProcessing:
    """Test batch processing functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_worker_pool = MockSonnetWorkerPool()
        self.mock_memory_graph = MockMemoryGraph()
        self.mock_opus_strategist = MockOpusStrategist()
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
        )
    
    @pytest.mark.asyncio
    async def test_batch_propagation_performance(self):
        """Test batch propagation meets <200ms requirement for 10 discoveries."""
        # Create 10 test discoveries
        discoveries = []
        for i in range(10):
            discovery = LearningDiscovery(
                worker_id=f"worker_{i}",
                pattern_type="test_pattern",
                pattern_data={"test_data": f"value_{i}"},
                context={"location": f"location_{i}"},
                confidence=0.7 + (i * 0.02),  # Vary confidence
                success_rate=0.8,
                sample_size=3
            )
            discoveries.append(discovery)
        
        start_time = time.time()
        result = await self.coordinator.batch_propagate(discoveries)
        execution_time_ms = (time.time() - start_time) * 1000
        
        assert execution_time_ms < 200.0, f"Batch propagation took {execution_time_ms:.2f}ms, exceeds 200ms requirement"
        assert result.success is True
        assert len(result.discoveries_propagated) <= 10
        assert result.execution_time_ms < 200.0
    
    @pytest.mark.asyncio
    async def test_batch_processing_empty_input(self):
        """Test batch processing handles empty input gracefully."""
        result = await self.coordinator.batch_propagate([])
        
        assert result.success is True
        assert len(result.discoveries_propagated) == 0
        assert result.execution_time_ms < 10.0  # Should be very fast
    
    @pytest.mark.asyncio
    async def test_batch_size_limit_enforcement(self):
        """Test batch size limit is enforced."""
        # Create more discoveries than batch limit
        discoveries = []
        batch_size = self.coordinator.config.batch_size
        
        for i in range(batch_size + 5):  # Exceed batch size
            discovery = LearningDiscovery(
                worker_id=f"batch_worker_{i}",
                pattern_type="batch_test",
                pattern_data={"index": i},
                context={"batch_test": True},
                confidence=0.7
            )
            discoveries.append(discovery)
        
        result = await self.coordinator.batch_propagate(discoveries)
        
        # Should only process up to batch_size discoveries
        assert len(result.discoveries_propagated) <= batch_size
        assert result.success is True


@pytest.mark.medium
class TestSystemIntegration:
    """Test integration with external systems."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_worker_pool = MockSonnetWorkerPool()
        self.mock_memory_graph = MockMemoryGraph()
        self.mock_opus_strategist = MockOpusStrategist()
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
        )
    
    @pytest.mark.asyncio
    async def test_worker_pool_integration(self):
        """Test integration with SonnetWorkerPool."""
        discovery = LearningDiscovery(
            worker_id="integration_worker",
            pattern_type="integration_test",
            pattern_data={"test": "integration"},
            context={"location": "test_location"},
            confidence=0.8
        )
        
        result = await self.coordinator.propagate_learning(discovery)
        
        assert result.success is True
        assert len(self.mock_worker_pool.share_pattern_calls) > 0
        
        # Verify pattern was shared correctly
        shared_pattern = self.mock_worker_pool.share_pattern_calls[0]
        assert shared_pattern["pattern_data"]["strategy_id"] == discovery.discovery_id
        assert shared_pattern["discovered_by"] == discovery.worker_id
    
    @pytest.mark.asyncio
    async def test_memory_graph_integration(self):
        """Test integration with MemoryGraph."""
        discovery = LearningDiscovery(
            worker_id="memory_test_worker",
            pattern_type="memory_test",
            pattern_data={"memory": "test"},
            context={"location": "memory_location"},
            confidence=0.85,
            success_rate=0.9
        )
        
        await self.coordinator.propagate_learning(discovery)
        
        # Verify discovery was stored in memory graph
        assert len(self.mock_memory_graph.stored_discoveries) > 0
        stored_discovery = self.mock_memory_graph.stored_discoveries[0]
        assert stored_discovery["discovery_id"] == discovery.discovery_id
        assert stored_discovery["pattern_type"] == discovery.pattern_type
    
    @pytest.mark.asyncio
    async def test_opus_strategist_integration(self):
        """Test integration with OpusStrategist."""
        discoveries = [
            LearningDiscovery(
                worker_id="opus_worker_1",
                pattern_type="strategic_pattern",
                confidence=0.8,
                success_rate=0.85
            ),
            LearningDiscovery(
                worker_id="opus_worker_2", 
                pattern_type="tactical_pattern",
                confidence=0.7,
                success_rate=0.75
            )
        ]
        
        game_state = {
            "location": "strategic_location",
            "health": 100,
            "level": 15
        }
        
        strategic_plan = await self.coordinator.integrate_with_opus_strategist(
            discoveries, 
            game_state
        )
        
        assert "strategy_id" in strategic_plan
        assert "experiments" in strategic_plan
        assert "strategic_insights" in strategic_plan
        assert len(self.mock_opus_strategist.request_strategy_calls) > 0
        
        # Verify correct data was passed to OpusStrategist
        strategy_call = self.mock_opus_strategist.request_strategy_calls[0]
        assert strategy_call["game_state"] == game_state
        assert len(strategy_call["recent_results"]) == 2


@pytest.mark.slow
class TestCircuitBreakerProtection:
    """Test circuit breaker protection and failure handling."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_worker_pool = MockSonnetWorkerPool()
        self.mock_memory_graph = MockMemoryGraph()
        self.mock_opus_strategist = MockOpusStrategist()
        
        # Enable circuit breaker with low threshold for testing
        config = PropagationConfig(
            circuit_breaker_enabled=True,
            failure_threshold=2,  # Low threshold for testing
            recovery_timeout_ms=1000.0  # Short recovery for testing
        )
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
            config=config,
        )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after failure threshold."""
        # Simulate failures by making worker pool fail
        self.mock_worker_pool.share_pattern_success = False
        
        discovery = LearningDiscovery(
            worker_id="circuit_test_worker",
            pattern_type="circuit_test",
            confidence=0.8
        )
        
        # Trigger failures to open circuit breaker
        for _ in range(3):  # Exceed failure threshold
            result = await self.coordinator.propagate_learning(discovery)
            assert result.success is False
        
        # Circuit breaker should now be open
        assert self.coordinator.circuit_breaker.get_state() in [
            self.coordinator.circuit_breaker.CircuitState.OPEN,
            self.coordinator.circuit_breaker.CircuitState.HALF_OPEN
        ]
        
        # Next propagation should fail due to circuit breaker
        result = await self.coordinator.propagate_learning(discovery)
        assert result.success is False
        assert "Circuit breaker is open" in result.error_message
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        # Force circuit breaker open
        self.mock_worker_pool.share_pattern_success = False
        
        discovery = LearningDiscovery(
            worker_id="recovery_test_worker",
            pattern_type="recovery_test",
            confidence=0.8
        )
        
        # Trigger failures
        for _ in range(3):
            await self.coordinator.propagate_learning(discovery)
        
        # Wait for recovery timeout (short for testing)
        await asyncio.sleep(1.1)  # Slightly longer than recovery timeout
        
        # Restore worker pool functionality
        self.mock_worker_pool.share_pattern_success = True
        
        # Circuit breaker should allow requests again
        result = await self.coordinator.propagate_learning(discovery)
        assert result.success is True


@pytest.mark.fast
class TestObserverPattern:
    """Test observer pattern implementation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_worker_pool = MockSonnetWorkerPool()
        self.mock_memory_graph = MockMemoryGraph()
        self.mock_opus_strategist = MockOpusStrategist()
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
        )
        
        self.test_observer = LearningPropagationObserver("test_observer")
    
    def test_observer_registration(self):
        """Test observer can be added and removed."""
        # Add observer
        self.coordinator.add_observer(self.test_observer)
        assert self.test_observer in self.coordinator._observers
        
        # Remove observer
        self.coordinator.remove_observer(self.test_observer)
        assert self.test_observer not in self.coordinator._observers
    
    @pytest.mark.asyncio
    async def test_observer_notifications(self):
        """Test observers receive event notifications."""
        self.coordinator.add_observer(self.test_observer)
        
        discovery = LearningDiscovery(
            worker_id="observer_test_worker",
            pattern_type="observer_test",
            confidence=0.8
        )
        
        initial_events = self.test_observer.events_received
        
        # Trigger propagation to generate events
        await self.coordinator.propagate_learning(discovery)
        
        # Observer should have received events
        assert self.test_observer.events_received > initial_events
        assert self.test_observer.last_event_time > 0
    
    def test_observer_error_handling(self):
        """Test observer error handling doesn't break propagation."""
        # Create observer that always raises exceptions
        failing_observer = Mock()
        failing_observer.on_learning_discovered.side_effect = Exception("Observer failure")
        
        self.coordinator.add_observer(failing_observer)
        
        # Propagation should still work despite observer failure
        discovery = LearningDiscovery(
            worker_id="error_test_worker",
            pattern_type="error_test",
            confidence=0.8
        )
        
        # This should not raise an exception
        self.coordinator._notify_observers("on_learning_discovered", discovery)


@pytest.mark.medium
class TestDiscoveryManagement:
    """Test discovery lifecycle management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_worker_pool = MockSonnetWorkerPool()
        self.mock_memory_graph = MockMemoryGraph()
        self.mock_opus_strategist = MockOpusStrategist()
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
        )
    
    def test_discovery_caching(self):
        """Test discovery caching functionality."""
        discovery = LearningDiscovery(
            discovery_id="cache_test_discovery",
            worker_id="cache_test_worker",
            pattern_type="cache_test",
            confidence=0.8
        )
        
        # Store in cache
        self.coordinator._update_discovery_cache(discovery)
        
        # Retrieve from cache
        cached_discovery = self.coordinator.get_discovery_by_id("cache_test_discovery")
        
        assert cached_discovery is not None
        assert cached_discovery.discovery_id == discovery.discovery_id
        assert cached_discovery.worker_id == discovery.worker_id
    
    def test_discovery_cache_size_limit(self):
        """Test discovery cache respects size limits."""
        # Set small cache size for testing
        self.coordinator.config.discovery_cache_size = 5
        
        # Add more discoveries than cache limit
        for i in range(10):
            discovery = LearningDiscovery(
                discovery_id=f"cache_limit_test_{i}",
                worker_id=f"worker_{i}",
                pattern_type="cache_limit_test",
                confidence=0.7
            )
            self.coordinator._update_discovery_cache(discovery)
        
        # Cache should not exceed limit
        assert len(self.coordinator._discovery_cache) <= 5
        
        # Most recent discoveries should still be in cache
        recent_discovery = self.coordinator.get_discovery_by_id("cache_limit_test_9")
        assert recent_discovery is not None
    
    def test_worker_learning_summary(self):
        """Test worker learning summary generation."""
        worker_id = "summary_test_worker"
        
        # Create several discoveries for the worker
        discoveries = []
        for i in range(3):
            discovery = LearningDiscovery(
                worker_id=worker_id,
                pattern_type=f"pattern_{i}",
                confidence=0.7 + (i * 0.1),
                success_rate=0.8,
                priority=LearningPriority.NORMAL if i < 2 else LearningPriority.HIGH
            )
            discoveries.append(discovery)
            self.coordinator._update_discovery_cache(discovery)
        
        summary = self.coordinator.get_worker_learning_summary(worker_id)
        
        assert summary["worker_id"] == worker_id
        assert summary["total_discoveries"] == 3
        assert 0.7 <= summary["avg_confidence"] <= 1.0
        assert summary["avg_success_rate"] == 0.8
        assert len(summary["pattern_types"]) == 3
        assert summary["priority_distribution"]["NORMAL"] == 2
        assert summary["priority_distribution"]["HIGH"] == 1


@pytest.mark.slow
class TestConcurrentOperations:
    """Test concurrent propagation operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_worker_pool = MockSonnetWorkerPool()
        self.mock_memory_graph = MockMemoryGraph()
        self.mock_opus_strategist = MockOpusStrategist()
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_propagation_safety(self):
        """Test concurrent propagation operations are thread-safe."""
        discoveries = []
        
        # Create multiple discoveries
        for i in range(10):
            discovery = LearningDiscovery(
                worker_id=f"concurrent_worker_{i}",
                pattern_type="concurrent_test",
                pattern_data={"index": i},
                context={"concurrent_test": True},
                confidence=0.8
            )
            discoveries.append(discovery)
        
        # Execute propagations concurrently
        tasks = []
        for discovery in discoveries:
            task = asyncio.create_task(self.coordinator.propagate_learning(discovery))
            tasks.append(task)
        
        # Wait for all propagations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All propagations should succeed (or at least not raise exceptions)
        successful_results = [r for r in results if isinstance(r, PropagationResult) and r.success]
        assert len(successful_results) > 0
        
        # No exceptions should be raised
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_discovery_creation(self):
        """Test concurrent discovery creation is thread-safe."""
        async def create_discovery(worker_index: int):
            return await self.coordinator.discover_learning(
                worker_id=f"concurrent_creation_worker_{worker_index}",
                pattern_type="concurrent_creation",
                pattern_data={"worker_index": worker_index},
                context={"test": "concurrent_creation"},
                confidence=0.7 + (worker_index * 0.05)
            )
        
        # Create discoveries concurrently
        tasks = [create_discovery(i) for i in range(5)]
        discoveries = await asyncio.gather(*tasks)
        
        # All discoveries should be created successfully
        assert len(discoveries) == 5
        assert all(isinstance(d, LearningDiscovery) for d in discoveries)
        
        # Each discovery should have unique ID
        discovery_ids = [d.discovery_id for d in discoveries]
        assert len(set(discovery_ids)) == 5  # All unique


@pytest.mark.fast
class TestDataModelValidation:
    """Test data model validation and edge cases."""
    
    def test_learning_discovery_validation(self):
        """Test LearningDiscovery validation and normalization."""
        # Test with valid data
        discovery = LearningDiscovery(
            worker_id="validation_worker",
            pattern_type="validation_test",
            confidence=0.8,
            success_rate=0.9,
            sample_size=5
        )
        
        assert discovery.is_propagation_ready(0.6)
        assert 0.0 <= discovery.get_quality_score() <= 1.0
        
        # Test edge cases
        edge_discovery = LearningDiscovery(
            confidence=1.5,  # Should be clamped to 1.0
            success_rate=-0.1,  # Should be clamped to 0.0
            sample_size=-5  # Should be clamped to 0
        )
        
        assert edge_discovery.confidence == 1.0
        assert edge_discovery.success_rate == 0.0
        assert edge_discovery.sample_size == 0
    
    def test_discovery_serialization(self):
        """Test discovery serialization and deserialization."""
        original_discovery = LearningDiscovery(
            worker_id="serialization_worker",
            pattern_type="serialization_test",
            pattern_data={"test": "data", "nested": {"value": 42}},
            context={"location": "test_location"},
            confidence=0.85,
            success_rate=0.9,
            sample_size=10,
            priority=LearningPriority.HIGH
        )
        
        # Serialize to dict
        discovery_dict = original_discovery.to_dict()
        
        # Deserialize from dict
        reconstructed_discovery = LearningDiscovery.from_dict(discovery_dict)
        
        # Verify reconstruction
        assert reconstructed_discovery.worker_id == original_discovery.worker_id
        assert reconstructed_discovery.pattern_type == original_discovery.pattern_type
        assert reconstructed_discovery.pattern_data == original_discovery.pattern_data
        assert reconstructed_discovery.confidence == original_discovery.confidence
        assert reconstructed_discovery.priority == original_discovery.priority
    
    def test_propagation_config_validation(self):
        """Test propagation configuration validation."""
        # Valid configuration
        valid_config = PropagationConfig()
        issues = valid_config.validate()
        assert len(issues) == 0
        
        # Invalid configuration
        invalid_config = PropagationConfig(
            max_propagation_time_ms=-10,
            min_confidence_threshold=2.0,
            batch_size=0
        )
        issues = invalid_config.validate()
        assert len(issues) > 0
        assert any("max_propagation_time_ms must be positive" in issue for issue in issues)
        assert any("min_confidence_threshold must be between 0.0 and 1.0" in issue for issue in issues)
        assert any("batch_size must be positive" in issue for issue in issues)


@pytest.mark.medium
class TestMCPIntegration:
    """Test MCP memory system integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_worker_pool = MockSonnetWorkerPool()
        self.mock_memory_graph = MockMemoryGraph()
        self.mock_opus_strategist = MockOpusStrategist()
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
        )
    
    @pytest.mark.asyncio
    async def test_mcp_storage_integration(self):
        """Test MCP storage integration."""
        discovery = LearningDiscovery(
            worker_id="mcp_test_worker",
            pattern_type="mcp_test",
            pattern_data={"mcp": "integration_test"},
            context={"location": "mcp_location"},
            confidence=0.85
        )
        
        # Mock MCP functions
        with patch('__main__.mcp__memgraph_memory__store_memory') as mock_store:
            mock_store.return_value = "mcp_memory_id_123"
            
            memory_id = await self.coordinator._store_discovery_mcp(discovery)
            
            assert memory_id == "mcp_memory_id_123"
            assert mock_store.called
            
            # Verify correct parameters were passed
            call_args = mock_store.call_args
            assert call_args[1]["node_type"] == "concept"
            assert call_args[1]["confidence"] == discovery.confidence
            assert "learning_propagation" in call_args[1]["tags"]
    
    @pytest.mark.asyncio
    async def test_memory_graph_sync(self):
        """Test synchronization with memory graph."""
        # Add pending discoveries
        for i in range(3):
            discovery = LearningDiscovery(
                discovery_id=f"sync_test_{i}",
                worker_id=f"sync_worker_{i}",
                pattern_type="sync_test",
                confidence=0.8
            )
            self.coordinator._pending_discoveries[discovery.discovery_id] = discovery
        
        initial_pending_count = len(self.coordinator._pending_discoveries)
        
        # Execute sync
        sync_result = await self.coordinator.sync_with_memory_graph()
        
        assert sync_result["discoveries_synced"] == initial_pending_count
        assert sync_result["errors"] == 0
        assert len(self.coordinator._pending_discoveries) == 0  # Should be cleared after sync
        assert len(self.mock_memory_graph.stored_discoveries) == initial_pending_count


@pytest.mark.slow
class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_worker_pool = MockSonnetWorkerPool(worker_count=6)
        self.mock_memory_graph = MockMemoryGraph()
        self.mock_opus_strategist = MockOpusStrategist()
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
        )
        
        # Add observer for monitoring
        self.observer = LearningPropagationObserver("e2e_test")
        self.coordinator.add_observer(self.observer)
    
    @pytest.mark.asyncio
    async def test_complete_learning_propagation_workflow(self):
        """Test complete workflow from discovery to propagation."""
        # Step 1: Create discovery
        discovery = await self.coordinator.discover_learning(
            worker_id="e2e_worker_1",
            pattern_type="speedrun_route", 
            pattern_data={
                "route": ["pallet_town", "viridian_city", "pewter_city"],
                "estimated_time": 180.5,
                "key_strategies": ["avoid_trainers", "optimal_items"]
            },
            context={
                "location": "kanto_region",
                "game_version": "red",
                "difficulty": "casual"
            },
            confidence=0.88,
            success_rate=0.92,
            sample_size=15
        )
        
        assert discovery.worker_id == "e2e_worker_1"
        assert discovery.pattern_type == "speedrun_route"
        assert discovery.priority in [LearningPriority.HIGH, LearningPriority.CRITICAL]  # High quality discovery
        
        # Step 2: Propagate discovery
        propagation_result = await self.coordinator.propagate_learning(discovery)
        
        assert propagation_result.success is True
        assert discovery.discovery_id in propagation_result.discoveries_propagated
        assert len(propagation_result.workers_updated) > 0
        assert propagation_result.execution_time_ms < 100.0  # Performance requirement
        
        # Step 3: Verify integration
        assert len(self.mock_worker_pool.share_pattern_calls) > 0
        assert len(self.mock_memory_graph.stored_discoveries) > 0
        
        # Step 4: Verify observer notifications
        assert self.observer.events_received > 0
        
        # Step 5: Check metrics
        metrics = self.coordinator.get_propagation_metrics()
        assert metrics["total_operations"] > 0
        assert metrics["success_rate"] > 0.0
    
    @pytest.mark.asyncio 
    async def test_workflow_with_conflict_resolution(self):
        """Test workflow with conflict detection and resolution."""
        # Set up conflicting patterns in memory graph
        self.mock_memory_graph.query_results = [
            {
                "discovery_id": "conflicting_discovery",
                "pattern_type": "movement",
                "location": "conflict_location", 
                "confidence": 0.3,  # Low confidence
                "success_rate": 0.4,
            }
        ]
        
        # Create high confidence discovery that conflicts
        high_confidence_discovery = await self.coordinator.discover_learning(
            worker_id="conflict_worker",
            pattern_type="movement",
            pattern_data={"direction": "right", "frames": 5},
            context={"location": "conflict_location"},
            confidence=0.95,  # High confidence - should win conflict
            success_rate=0.92,
            sample_size=20
        )
        
        # Propagate - should trigger conflict resolution
        result = await self.coordinator.propagate_learning(high_confidence_discovery)
        
        assert result.success is True
        # Should have resolved conflict in favor of higher confidence discovery
        assert result.execution_time_ms < 100.0
    
    def test_metrics_collection_accuracy(self):
        """Test metrics collection provides accurate performance data."""
        # Start with clean metrics
        initial_metrics = self.coordinator.get_propagation_metrics()
        
        # Create discovery and verify metrics update
        discovery = LearningDiscovery(
            worker_id="metrics_worker",
            pattern_type="metrics_test",
            confidence=0.85
        )
        
        self.coordinator._update_discovery_cache(discovery)
        
        updated_metrics = self.coordinator.get_propagation_metrics()
        
        # Cache metrics should have updated
        assert updated_metrics["cache_size"] > initial_metrics["cache_size"]


# Integration test with real components (when available)
@pytest.mark.integration
class TestRealSystemIntegration:
    """Integration tests with real system components."""
    
    @pytest.mark.skip(reason="Requires real SonnetWorkerPool instance")
    async def test_real_worker_pool_integration(self):
        """Test integration with real SonnetWorkerPool."""
        # This test would use actual SonnetWorkerPool, MemoryGraph, etc.
        # Skipped for now as it requires full system setup
        pass
    
    @pytest.mark.skip(reason="Requires real MCP memory system")
    async def test_real_mcp_integration(self):
        """Test integration with real MCP memory system."""
        # This test would use actual MCP memory functions
        # Skipped for now as it requires MCP system setup
        pass


# Performance benchmark tests
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests for validation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_worker_pool = MockSonnetWorkerPool(worker_count=10)
        self.mock_memory_graph = MockMemoryGraph()
        self.mock_opus_strategist = MockOpusStrategist()
        
        self.coordinator = ParallelExecutionCoordinator(
            worker_pool=self.mock_worker_pool,
            memory_graph=self.mock_memory_graph,
            opus_strategist=self.mock_opus_strategist,
        )
    
    @pytest.mark.asyncio
    async def test_propagation_performance_benchmark(self):
        """Benchmark propagation performance under various conditions."""
        benchmark_results = []
        
        # Test different discovery sizes and complexities
        test_cases = [
            {"pattern_data_size": "small", "worker_count": 2},
            {"pattern_data_size": "medium", "worker_count": 4},
            {"pattern_data_size": "large", "worker_count": 6},
        ]
        
        for case in test_cases:
            # Create discovery with specified complexity
            pattern_data = {"data": "x" * (100 if case["pattern_data_size"] == "small" else 
                                         1000 if case["pattern_data_size"] == "medium" else 5000)}
            
            discovery = LearningDiscovery(
                worker_id="benchmark_worker",
                pattern_type="benchmark_test",
                pattern_data=pattern_data,
                confidence=0.8
            )
            
            # Set worker count
            self.mock_worker_pool.worker_count = case["worker_count"]
            self.mock_worker_pool.healthy_count = case["worker_count"]
            
            # Benchmark propagation
            start_time = time.time()
            result = await self.coordinator.propagate_learning(discovery)
            execution_time_ms = (time.time() - start_time) * 1000
            
            benchmark_results.append({
                "case": case,
                "execution_time_ms": execution_time_ms,
                "success": result.success,
                "workers_updated": len(result.workers_updated)
            })
            
            # Verify performance requirement
            assert execution_time_ms < 100.0, f"Benchmark case {case} took {execution_time_ms:.2f}ms"
        
        # Log benchmark results for analysis
        for result in benchmark_results:
            print(f"Benchmark: {result}")
    
    @pytest.mark.asyncio
    async def test_batch_processing_scalability(self):
        """Test batch processing scalability with increasing batch sizes."""
        batch_sizes = [1, 5, 10, 20]
        
        for batch_size in batch_sizes:
            discoveries = []
            for i in range(batch_size):
                discovery = LearningDiscovery(
                    worker_id=f"scale_worker_{i}",
                    pattern_type="scalability_test",
                    pattern_data={"index": i},
                    confidence=0.7
                )
                discoveries.append(discovery)
            
            start_time = time.time()
            result = await self.coordinator.batch_propagate(discoveries)
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Performance should scale reasonably
            max_expected_time = min(200.0, batch_size * 15.0)  # 15ms per discovery max
            assert execution_time_ms < max_expected_time, f"Batch size {batch_size} took {execution_time_ms:.2f}ms"
            
            print(f"Batch size {batch_size}: {execution_time_ms:.2f}ms")


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=claudelearnspokemon.parallel_execution_coordinator",
        "--cov=claudelearnspokemon.learning_propagation_interfaces", 
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=95"  # 95% coverage requirement
    ])