"""
ParallelExecutionCoordinator - Learning Propagation Implementation

Implements the learning propagation system for coordinating learning discoveries
across parallel worker pools. Provides real-time learning sharing, conflict resolution,
and performance optimization for distributed Pokemon speedrun learning.

Key Features:
- Real-time learning propagation with <100ms target performance
- Conflict detection and resolution between worker discoveries
- Integration with SonnetWorkerPool, MemoryGraph, OpusStrategist, and MCP systems
- Circuit breaker pattern for reliability
- Comprehensive metrics and monitoring
- Observer pattern for event notifications

Architecture:
- Strategy Pattern: Pluggable propagation strategies
- Observer Pattern: Event notifications for monitoring
- Command Pattern: Propagation operations as commands
- Circuit Breaker: Reliability and fault tolerance

Author: Felix (Craftsperson) - Claude Code Implementation Agent
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from .circuit_breaker import CircuitBreaker, CircuitConfig, CircuitState
from .learning_propagation_interfaces import (
    ConflictResolutionStrategy,
    ILearningObserver,
    ILearningPropagator,
    LearningConflict,
    LearningDiscovery,
    LearningPriority,
    LearningPropagationError,
    PropagationConfig,
    PropagationMetrics,
    PropagationResult,
    PropagationStrategy,
    PropagationTimeoutError,
    ConflictResolutionError,
    WorkerUnavailableError,
)
from .memory_graph import MemoryGraph
from .opus_strategist import OpusStrategist
from .sonnet_worker_pool import SonnetWorkerPool

logger = logging.getLogger(__name__)


class ParallelExecutionCoordinator(ILearningPropagator):
    """
    Coordinates learning propagation across parallel worker pools.
    
    This class implements the learning propagation system that enables
    real-time sharing of discovered patterns across Sonnet workers,
    with conflict resolution and performance optimization.
    """
    
    def __init__(
        self,
        worker_pool: SonnetWorkerPool,
        memory_graph: MemoryGraph,
        opus_strategist: OpusStrategist,
        config: Optional[PropagationConfig] = None
    ):
        """
        Initialize ParallelExecutionCoordinator.
        
        Args:
            worker_pool: SonnetWorkerPool for worker management
            memory_graph: MemoryGraph for persistent learning storage
            opus_strategist: OpusStrategist for strategic conflict resolution
            config: Optional configuration override
        """
        self.worker_pool = worker_pool
        self.memory_graph = memory_graph
        self.opus_strategist = opus_strategist
        self.config = config or PropagationConfig()
        
        # Validate configuration
        config_issues = self.config.validate()
        if config_issues:
            raise ValueError(f"Invalid configuration: {', '.join(config_issues)}")
        
        # Core data structures
        self._pending_discoveries: Dict[str, LearningDiscovery] = {}
        self._active_conflicts: Dict[str, LearningConflict] = {}
        self._propagation_history: List[PropagationResult] = []
        
        # Thread safety
        self._lock = threading.RLock()
        self._propagation_semaphore = threading.Semaphore(self.config.max_concurrent_propagations)
        
        # Metrics and monitoring
        self.metrics = PropagationMetrics()
        self._observers: List[ILearningObserver] = []
        
        # Circuit breaker for reliability
        self.circuit_breaker = CircuitBreaker(
            config=CircuitConfig(
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout_ms / 1000.0,  # Convert to seconds
                expected_exception_types=(LearningPropagationError,),
            )
        ) if self.config.circuit_breaker_enabled else None
        
        # Discovery cache for performance
        self._discovery_cache: Dict[str, LearningDiscovery] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_lock = threading.Lock()
        
        # Background processing
        self._batch_processor_active = False
        self._batch_processing_thread: Optional[threading.Thread] = None
        
        logger.info(
            "ParallelExecutionCoordinator initialized",
            max_propagation_time_ms=self.config.max_propagation_time_ms,
            batch_size=self.config.batch_size,
            circuit_breaker_enabled=self.config.circuit_breaker_enabled,
        )
    
    def add_observer(self, observer: ILearningObserver) -> None:
        """Add observer for learning propagation events."""
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)
                logger.debug(f"Added learning observer: {type(observer).__name__}")
    
    def remove_observer(self, observer: ILearningObserver) -> None:
        """Remove observer from event notifications."""
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)
                logger.debug(f"Removed learning observer: {type(observer).__name__}")
    
    async def propagate_learning(
        self,
        discovery: LearningDiscovery,
        target_workers: Optional[List[str]] = None,
        strategy: Optional[PropagationStrategy] = None
    ) -> PropagationResult:
        """
        Propagate a learning discovery to target workers.
        
        Implements the core learning propagation with performance monitoring,
        conflict detection, and circuit breaker protection.
        """
        start_time = time.time()
        propagation_strategy = strategy or self.config.default_propagation_strategy
        
        try:
            # Check circuit breaker
            if self.circuit_breaker and not self.circuit_breaker.is_available():
                self.metrics.circuit_breaker_trips += 1
                raise LearningPropagationError("Circuit breaker is open")
            
            # Validate discovery quality
            if not discovery.is_propagation_ready(self.config.min_confidence_threshold):
                error_msg = f"Discovery {discovery.discovery_id} not ready for propagation"
                logger.warning(f"{error_msg} (confidence={discovery.confidence}, sample_size={discovery.sample_size})")
                self.metrics.record_quality_issue(discovery, "below_threshold")
                return PropagationResult(
                    success=False,
                    error_message=error_msg,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Acquire propagation semaphore
            with self._propagation_semaphore:
                # Notify observers
                self._notify_observers("on_learning_discovered", discovery)
                
                # Check for conflicts with existing discoveries
                conflict = await self._detect_conflicts(discovery)
                if conflict:
                    self._notify_observers("on_conflict_detected", conflict)
                    resolved_discovery = await self.resolve_conflict(conflict)
                    discovery = resolved_discovery
                
                # Execute propagation based on strategy
                result = await self._execute_propagation(discovery, target_workers, propagation_strategy)
                
                # Store in memory graph for persistence
                await self._store_discovery_persistent(discovery)
                
                # Update cache
                self._update_discovery_cache(discovery)
                
                # Record metrics
                self.metrics.record_propagation(result, discovery)
                
                # Notify observers of successful propagation
                if result.success:
                    self._notify_observers("on_learning_propagated", discovery, result)
                    if self.circuit_breaker:
                        self.circuit_breaker.metrics.record_success()
                else:
                    if self.circuit_breaker:
                        self.circuit_breaker.metrics.record_failure()
                
                # Check performance requirement
                if result.execution_time_ms > self.config.max_propagation_time_ms:
                    logger.warning(
                        f"Propagation exceeded time threshold: {result.execution_time_ms:.1f}ms > {self.config.max_propagation_time_ms}ms"
                    )
                
                return result
        
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Learning propagation failed: {e} (discovery_id={discovery.discovery_id})")
            
            if self.circuit_breaker:
                self.circuit_breaker.metrics.record_failure()
            
            return PropagationResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )
    
    async def batch_propagate(
        self,
        discoveries: List[LearningDiscovery],
        strategy: Optional[PropagationStrategy] = None
    ) -> PropagationResult:
        """
        Propagate multiple discoveries in a batch operation for efficiency.
        """
        start_time = time.time()
        batch_strategy = strategy or PropagationStrategy.BATCHED
        
        if not discoveries:
            return PropagationResult(
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Enforce batch size limit
        if len(discoveries) > self.config.batch_size:
            logger.warning(f"Batch size {len(discoveries)} exceeds limit {self.config.batch_size}")
            discoveries = discoveries[:self.config.batch_size]
        
        try:
            # Check circuit breaker
            if self.circuit_breaker and not self.circuit_breaker.is_available():
                self.metrics.circuit_breaker_trips += 1
                raise LearningPropagationError("Circuit breaker is open for batch operation")
            
            # Filter discoveries by quality threshold
            valid_discoveries = [
                d for d in discoveries 
                if d.is_propagation_ready(self.config.min_confidence_threshold)
            ]
            
            if not valid_discoveries:
                return PropagationResult(
                    success=False,
                    error_message="No discoveries meet quality threshold for batch propagation",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Process discoveries in batch with conflict detection
            resolved_discoveries = []
            total_conflicts = 0
            
            for discovery in valid_discoveries:
                conflict = await self._detect_conflicts(discovery)
                if conflict:
                    total_conflicts += 1
                    self._notify_observers("on_conflict_detected", conflict)
                    resolved_discovery = await self.resolve_conflict(conflict)
                    resolved_discoveries.append(resolved_discovery)
                else:
                    resolved_discoveries.append(discovery)
            
            # Execute batch propagation
            batch_result = await self._execute_batch_propagation(resolved_discoveries, batch_strategy)
            
            # Store all discoveries in memory graph
            for discovery in resolved_discoveries:
                await self._store_discovery_persistent(discovery)
                self._update_discovery_cache(discovery)
            
            # Update batch result with conflict information
            batch_result.conflicts_resolved = [f"conflict_{i}" for i in range(total_conflicts)]
            
            # Record metrics for all discoveries
            for discovery in resolved_discoveries:
                self.metrics.record_propagation(batch_result, discovery)
            
            logger.info(
                f"Batch propagation completed",
                discoveries=len(resolved_discoveries),
                conflicts=total_conflicts,
                execution_time_ms=batch_result.execution_time_ms
            )
            
            return batch_result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Batch propagation failed: {e} (batch_size={len(discoveries)})")
            
            return PropagationResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )
    
    async def resolve_conflict(
        self,
        conflict: LearningConflict,
        resolution_strategy: Optional[ConflictResolutionStrategy] = None
    ) -> LearningDiscovery:
        """
        Resolve conflicts between contradictory discoveries.
        
        Uses the specified resolution strategy to merge or select
        from conflicting discoveries.
        """
        start_time = time.time()
        strategy = resolution_strategy or self.config.default_conflict_resolution
        
        try:
            if not conflict.conflicting_discoveries:
                raise ConflictResolutionError("No discoveries to resolve in conflict")
            
            with self._lock:
                self._active_conflicts[conflict.conflict_id] = conflict
            
            # Apply resolution strategy
            resolved_discovery = await self._apply_conflict_resolution(conflict, strategy)
            
            # Update conflict record
            conflict.resolved = True
            conflict.resolution_result = resolved_discovery
            conflict.resolution_time_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            self.metrics.record_conflict(conflict, resolved=True)
            
            # Notify observers
            self._notify_observers("on_conflict_resolved", conflict, resolved_discovery)
            
            logger.info(
                f"Conflict resolved",
                conflict_id=conflict.conflict_id,
                strategy=strategy.value,
                resolution_time_ms=conflict.resolution_time_ms
            )
            
            return resolved_discovery
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Conflict resolution failed: {e} (conflict_id={conflict.conflict_id})")
            
            # Record failed conflict resolution
            self.metrics.record_conflict(conflict, resolved=False)
            
            raise ConflictResolutionError(f"Failed to resolve conflict: {e}") from e
    
    async def discover_learning(
        self,
        worker_id: str,
        pattern_type: str,
        pattern_data: Dict[str, Any],
        context: Dict[str, Any],
        confidence: float,
        success_rate: float = 0.0,
        sample_size: int = 1
    ) -> LearningDiscovery:
        """
        Create and process a new learning discovery from a worker.
        
        This is the main entry point for workers to report discovered patterns.
        """
        try:
            # Create discovery object
            discovery = LearningDiscovery(
                worker_id=worker_id,
                pattern_type=pattern_type,
                pattern_data=pattern_data,
                context=context,
                confidence=confidence,
                success_rate=success_rate,
                sample_size=sample_size,
                priority=self._determine_priority(confidence, success_rate),
                propagation_strategy=self._select_propagation_strategy(pattern_type, confidence)
            )
            
            # Store in pending discoveries
            with self._lock:
                self._pending_discoveries[discovery.discovery_id] = discovery
            
            logger.info(
                f"Learning discovery created",
                discovery_id=discovery.discovery_id,
                worker_id=worker_id,
                pattern_type=pattern_type,
                confidence=confidence
            )
            
            # Trigger immediate or batched propagation based on strategy
            if discovery.propagation_strategy == PropagationStrategy.IMMEDIATE:
                # Propagate immediately in background
                asyncio.create_task(self._propagate_immediate(discovery))
            else:
                # Add to batch processing queue
                self._queue_for_batch_processing(discovery)
            
            return discovery
            
        except Exception as e:
            logger.error(f"Failed to create learning discovery: {e} (worker_id={worker_id})")
            raise LearningPropagationError(f"Discovery creation failed: {e}") from e
    
    def start_batch_processor(self) -> None:
        """Start background batch processing thread."""
        if self._batch_processor_active:
            logger.warning("Batch processor already active")
            return
        
        self._batch_processor_active = True
        self._batch_processing_thread = threading.Thread(
            target=self._batch_processing_loop,
            daemon=True,
            name="LearningPropagationBatchProcessor"
        )
        self._batch_processing_thread.start()
        logger.info("Learning propagation batch processor started")
    
    def stop_batch_processor(self) -> None:
        """Stop background batch processing thread."""
        self._batch_processor_active = False
        if self._batch_processing_thread:
            self._batch_processing_thread.join(timeout=5.0)
            self._batch_processing_thread = None
        logger.info("Learning propagation batch processor stopped")
    
    def get_propagation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive propagation metrics."""
        with self._lock:
            metrics_summary = self.metrics.get_performance_summary()
            
            # Add coordinator-specific metrics
            metrics_summary.update({
                "pending_discoveries": len(self._pending_discoveries),
                "active_conflicts": len(self._active_conflicts),
                "propagation_history_size": len(self._propagation_history),
                "circuit_breaker_state": (
                    self.circuit_breaker.get_state().value 
                    if self.circuit_breaker else "disabled"
                ),
                "batch_processor_active": self._batch_processor_active,
                "cache_size": len(self._discovery_cache),
            })
            
            return metrics_summary
    
    def get_discovery_by_id(self, discovery_id: str) -> Optional[LearningDiscovery]:
        """Retrieve discovery by ID from cache or pending discoveries."""
        with self._lock:
            # Check pending discoveries first
            if discovery_id in self._pending_discoveries:
                return self._pending_discoveries[discovery_id]
            
            # Check cache
            with self._cache_lock:
                if discovery_id in self._discovery_cache:
                    # Check cache TTL
                    cache_time = self._cache_timestamps.get(discovery_id, 0)
                    if time.time() * 1000 - cache_time < self.config.discovery_cache_ttl_ms:
                        self.metrics.cache_hits += 1
                        return self._discovery_cache[discovery_id]
                    else:
                        # Expired cache entry
                        self._discovery_cache.pop(discovery_id, None)
                        self._cache_timestamps.pop(discovery_id, None)
        
        self.metrics.cache_misses += 1
        return None
    
    # Private implementation methods
    
    async def _propagate_immediate(self, discovery: LearningDiscovery) -> None:
        """Handle immediate propagation in background."""
        try:
            result = await self.propagate_learning(discovery)
            logger.debug(f"Immediate propagation completed (discovery_id={discovery.discovery_id}, success={result.success})")
        except Exception as e:
            logger.error(f"Immediate propagation failed: {e} (discovery_id={discovery.discovery_id})")
    
    def _queue_for_batch_processing(self, discovery: LearningDiscovery) -> None:
        """Queue discovery for batch processing."""
        # For now, keep in pending discoveries - batch processing will handle it
        logger.debug(f"Queued discovery for batch processing (discovery_id={discovery.discovery_id})")
    
    async def _execute_propagation(
        self,
        discovery: LearningDiscovery,
        target_workers: Optional[List[str]],
        strategy: PropagationStrategy
    ) -> PropagationResult:
        """Execute the actual propagation operation."""
        start_time = time.time()
        
        try:
            # Determine target workers
            if target_workers is None:
                # Get all healthy workers from pool
                all_workers = []
                for worker_id in range(self.worker_pool.get_worker_count()):
                    worker_status = self.worker_pool.get_worker_status(f"sonnet_worker_{worker_id}")
                    if worker_status and worker_status.get("healthy"):
                        all_workers.append(worker_status["worker_id"])
                target_workers = all_workers
            
            if not target_workers:
                raise WorkerUnavailableError("No healthy workers available for propagation")
            
            # Propagate to each target worker
            successful_workers = []
            failed_workers = []
            
            for worker_id in target_workers:
                try:
                    success = await self._propagate_to_worker(discovery, worker_id)
                    if success:
                        successful_workers.append(worker_id)
                        if worker_id not in discovery.propagated_to:
                            discovery.propagated_to.append(worker_id)
                    else:
                        failed_workers.append(worker_id)
                except Exception as e:
                    logger.warning(f"Failed to propagate to worker {worker_id}: {e}")
                    failed_workers.append(worker_id)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = PropagationResult(
                success=len(successful_workers) > 0,
                discoveries_propagated=[discovery.discovery_id],
                workers_updated=successful_workers,
                execution_time_ms=execution_time_ms,
                error_message=f"Failed workers: {failed_workers}" if failed_workers else None,
                performance_metrics={
                    "target_workers": len(target_workers),
                    "successful_workers": len(successful_workers),
                    "failed_workers": len(failed_workers),
                    "propagation_strategy": strategy.value,
                }
            )
            
            return result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Propagation execution failed: {e}")
            
            return PropagationResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )
    
    async def _execute_batch_propagation(
        self,
        discoveries: List[LearningDiscovery],
        strategy: PropagationStrategy
    ) -> PropagationResult:
        """Execute batch propagation of multiple discoveries."""
        start_time = time.time()
        
        try:
            # Get all healthy workers
            target_workers = []
            for worker_id in range(self.worker_pool.get_worker_count()):
                worker_status = self.worker_pool.get_worker_status(f"sonnet_worker_{worker_id}")
                if worker_status and worker_status.get("healthy"):
                    target_workers.append(worker_status["worker_id"])
            
            if not target_workers:
                raise WorkerUnavailableError("No healthy workers available for batch propagation")
            
            # Propagate all discoveries to all workers
            all_successful_workers = set()
            all_failed_workers = set()
            successful_discovery_ids = []
            
            for discovery in discoveries:
                successful_workers = []
                for worker_id in target_workers:
                    try:
                        success = await self._propagate_to_worker(discovery, worker_id)
                        if success:
                            successful_workers.append(worker_id)
                            all_successful_workers.add(worker_id)
                            if worker_id not in discovery.propagated_to:
                                discovery.propagated_to.append(worker_id)
                        else:
                            all_failed_workers.add(worker_id)
                    except Exception as e:
                        logger.warning(f"Batch propagation to worker {worker_id} failed: {e}")
                        all_failed_workers.add(worker_id)
                
                if successful_workers:
                    successful_discovery_ids.append(discovery.discovery_id)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            result = PropagationResult(
                success=len(successful_discovery_ids) > 0,
                discoveries_propagated=successful_discovery_ids,
                workers_updated=list(all_successful_workers),
                execution_time_ms=execution_time_ms,
                error_message=f"Failed workers: {list(all_failed_workers)}" if all_failed_workers else None,
                performance_metrics={
                    "batch_size": len(discoveries),
                    "successful_discoveries": len(successful_discovery_ids),
                    "target_workers": len(target_workers),
                    "successful_workers": len(all_successful_workers),
                    "propagation_strategy": strategy.value,
                }
            )
            
            return result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Batch propagation execution failed: {e}")
            
            return PropagationResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )
    
    async def _propagate_to_worker(self, discovery: LearningDiscovery, worker_id: str) -> bool:
        """Propagate discovery to a specific worker."""
        try:
            # Create pattern data for SonnetWorkerPool.share_pattern()
            pattern_data = {
                "strategy_id": discovery.discovery_id,
                "name": f"{discovery.pattern_type} Discovery",
                "pattern_sequence": [discovery.pattern_type],
                "success_rate": discovery.success_rate,
                "estimated_time": discovery.execution_time_ms / 1000.0,
                "resource_requirements": {"confidence": discovery.confidence},
                "risk_assessment": {"sample_size": discovery.sample_size},
                "alternatives": [],
                "optimization_history": [
                    {
                        "worker_id": discovery.worker_id,
                        "discovered_at": discovery.discovered_at.isoformat(),
                        "context": discovery.context,
                    }
                ],
            }
            
            # Use SonnetWorkerPool's share_pattern method
            success = self.worker_pool.share_pattern(pattern_data, discovered_by=discovery.worker_id)
            
            if success:
                logger.debug(f"Successfully propagated discovery to worker {worker_id}")
            else:
                logger.warning(f"Failed to propagate discovery to worker {worker_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error propagating to worker {worker_id}: {e}")
            return False
    
    async def _detect_conflicts(self, discovery: LearningDiscovery) -> Optional[LearningConflict]:
        """Detect conflicts with existing discoveries."""
        if not self.config.conflict_detection_enabled:
            return None
        
        try:
            # Query existing discoveries of same pattern type and context
            criteria = {
                "pattern_type": discovery.pattern_type,
                "location": discovery.context.get("location"),
                "limit": 10
            }
            
            existing_patterns = self.memory_graph.query_patterns(criteria)
            
            # Check for conflicts (contradictory patterns)
            conflicting_discoveries = []
            
            for pattern_data in existing_patterns:
                # Simple conflict detection - different pattern data for same context
                if (pattern_data.get("pattern_type") == discovery.pattern_type and
                    pattern_data.get("location") == discovery.context.get("location")):
                    
                    # Check if pattern data conflicts (simplified heuristic)
                    existing_success_rate = pattern_data.get("confidence", 0.0)
                    if abs(existing_success_rate - discovery.confidence) > 0.3:  # Significant difference
                        # Create a mock discovery for the conflict
                        conflicting_discovery = LearningDiscovery(
                            discovery_id=pattern_data.get("discovery_id", f"existing_{time.time()}"),
                            worker_id="existing_pattern",
                            pattern_type=discovery.pattern_type,
                            pattern_data=pattern_data,
                            context=discovery.context,
                            confidence=existing_success_rate,
                            success_rate=existing_success_rate,
                        )
                        conflicting_discoveries.append(conflicting_discovery)
            
            if conflicting_discoveries:
                # Add current discovery to conflict
                conflicting_discoveries.append(discovery)
                
                conflict = LearningConflict(
                    conflicting_discoveries=conflicting_discoveries,
                    resolution_strategy=self.config.default_conflict_resolution
                )
                
                return conflict
            
            return None
            
        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")
            return None
    
    async def _apply_conflict_resolution(
        self, 
        conflict: LearningConflict, 
        strategy: ConflictResolutionStrategy
    ) -> LearningDiscovery:
        """Apply conflict resolution strategy to resolve conflicts."""
        discoveries = conflict.conflicting_discoveries
        
        if not discoveries:
            raise ConflictResolutionError("No discoveries to resolve")
        
        if len(discoveries) == 1:
            return discoveries[0]
        
        try:
            if strategy == ConflictResolutionStrategy.LATEST_WINS:
                # Return most recent discovery
                return max(discoveries, key=lambda d: d.discovered_at)
            
            elif strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
                # Return discovery with highest confidence
                return max(discoveries, key=lambda d: d.confidence)
            
            elif strategy == ConflictResolutionStrategy.CONSENSUS_BASED:
                # Use majority consensus or average
                return await self._resolve_by_consensus(discoveries)
            
            elif strategy == ConflictResolutionStrategy.HYBRID_MERGE:
                # Attempt to merge discoveries
                return await self._merge_discoveries(discoveries)
            
            else:
                # Default to highest confidence
                return max(discoveries, key=lambda d: d.confidence)
                
        except Exception as e:
            logger.error(f"Conflict resolution strategy {strategy.value} failed: {e}")
            # Fallback to highest confidence
            return max(discoveries, key=lambda d: d.confidence)
    
    async def _resolve_by_consensus(self, discoveries: List[LearningDiscovery]) -> LearningDiscovery:
        """Resolve conflict using consensus approach."""
        # Simple consensus: weight by confidence and sample size
        best_discovery = discoveries[0]
        best_score = 0.0
        
        for discovery in discoveries:
            # Score based on confidence * sample_size
            score = discovery.confidence * min(1.0, discovery.sample_size / 10.0)
            if score > best_score:
                best_score = score
                best_discovery = discovery
        
        return best_discovery
    
    async def _merge_discoveries(self, discoveries: List[LearningDiscovery]) -> LearningDiscovery:
        """Merge conflicting discoveries into a hybrid discovery."""
        if not discoveries:
            raise ConflictResolutionError("No discoveries to merge")
        
        # Use first discovery as base
        base_discovery = discoveries[0]
        
        # Calculate weighted averages
        total_weight = sum(d.confidence * d.sample_size for d in discoveries)
        if total_weight == 0:
            return base_discovery
        
        weighted_confidence = sum(d.confidence * d.confidence * d.sample_size for d in discoveries) / total_weight
        weighted_success_rate = sum(d.success_rate * d.confidence * d.sample_size for d in discoveries) / total_weight
        total_sample_size = sum(d.sample_size for d in discoveries)
        
        # Create merged discovery
        merged_discovery = LearningDiscovery(
            worker_id="conflict_resolver",
            pattern_type=base_discovery.pattern_type,
            pattern_data=base_discovery.pattern_data.copy(),
            context=base_discovery.context.copy(),
            confidence=weighted_confidence,
            success_rate=weighted_success_rate,
            sample_size=total_sample_size,
            priority=max(d.priority for d in discoveries),
            propagation_strategy=base_discovery.propagation_strategy,
        )
        
        # Merge pattern data (simple approach - could be more sophisticated)
        merged_discovery.pattern_data["merged_from"] = [d.discovery_id for d in discoveries]
        merged_discovery.pattern_data["resolution_method"] = "hybrid_merge"
        
        return merged_discovery
    
    async def _store_discovery_persistent(self, discovery: LearningDiscovery) -> None:
        """Store discovery in persistent memory graph."""
        try:
            discovery_dict = {
                "discovery_id": discovery.discovery_id,
                "pattern_type": discovery.pattern_type,
                "pattern_data": discovery.pattern_data,
                "location": discovery.context.get("location", "unknown"),
                "success_rate": discovery.success_rate,
                "confidence": discovery.confidence,
                "checkpoint_context": discovery.checkpoint_context,
                "execution_time_ms": discovery.execution_time_ms,
                "success_count": max(1, int(discovery.success_rate * discovery.sample_size)),
                "failure_count": max(0, discovery.sample_size - int(discovery.success_rate * discovery.sample_size)),
                "related_patterns": [],  # Could be enhanced to detect relationships
            }
            
            discovery_id = self.memory_graph.store_discovery(discovery_dict)
            logger.debug(f"Stored discovery in memory graph (discovery_id={discovery_id})")
            
        except Exception as e:
            logger.error(f"Failed to store discovery persistently: {e}")
    
    def _update_discovery_cache(self, discovery: LearningDiscovery) -> None:
        """Update discovery cache with new discovery."""
        with self._cache_lock:
            # Check cache size limit
            if len(self._discovery_cache) >= self.config.discovery_cache_size:
                # Remove oldest entries
                oldest_entries = sorted(
                    self._cache_timestamps.items(),
                    key=lambda x: x[1]
                )[:len(self._discovery_cache) // 4]  # Remove 25% of oldest
                
                for discovery_id, _ in oldest_entries:
                    self._discovery_cache.pop(discovery_id, None)
                    self._cache_timestamps.pop(discovery_id, None)
            
            # Add new discovery to cache
            self._discovery_cache[discovery.discovery_id] = discovery
            self._cache_timestamps[discovery.discovery_id] = time.time() * 1000
    
    def _determine_priority(self, confidence: float, success_rate: float) -> LearningPriority:
        """Determine learning priority based on quality metrics."""
        quality_score = (confidence + success_rate) / 2.0
        
        if quality_score >= 0.9:
            return LearningPriority.CRITICAL
        elif quality_score >= 0.7:
            return LearningPriority.HIGH
        elif quality_score >= 0.5:
            return LearningPriority.NORMAL
        else:
            return LearningPriority.LOW
    
    def _select_propagation_strategy(self, pattern_type: str, confidence: float) -> PropagationStrategy:
        """Select optimal propagation strategy based on pattern characteristics."""
        # High confidence patterns get immediate propagation
        if confidence >= 0.8:
            return PropagationStrategy.IMMEDIATE
        
        # Critical pattern types get priority-based propagation
        critical_patterns = ["battle_strategy", "speedrun_route", "critical_glitch"]
        if pattern_type in critical_patterns:
            return PropagationStrategy.PRIORITY_BASED
        
        # Default to threshold-based for efficiency
        return PropagationStrategy.THRESHOLD_BASED
    
    def _notify_observers(self, method_name: str, *args, **kwargs) -> None:
        """Notify all registered observers of an event."""
        for observer in self._observers:
            try:
                method = getattr(observer, method_name)
                method(*args, **kwargs)
            except Exception as e:
                logger.error(f"Observer notification failed: {e} (observer={type(observer).__name__})")
    
    def _batch_processing_loop(self) -> None:
        """Background thread loop for batch processing."""
        logger.info("Batch processing loop started")
        
        while self._batch_processor_active:
            try:
                # Collect pending discoveries for batch processing
                batch_discoveries = []
                
                with self._lock:
                    # Get discoveries that are ready for batch processing
                    for discovery_id, discovery in list(self._pending_discoveries.items()):
                        if (discovery.propagation_strategy in [PropagationStrategy.BATCHED, PropagationStrategy.THRESHOLD_BASED] and
                            discovery.is_propagation_ready(self.config.min_confidence_threshold)):
                            batch_discoveries.append(discovery)
                            # Remove from pending
                            self._pending_discoveries.pop(discovery_id, None)
                        
                        if len(batch_discoveries) >= self.config.batch_size:
                            break
                
                # Process batch if we have discoveries
                if batch_discoveries:
                    logger.debug(f"Processing batch of {len(batch_discoveries)} discoveries")
                    
                    # Create task for batch processing
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(self.batch_propagate(batch_discoveries))
                        logger.debug(f"Batch processing completed (success={result.success})")
                    finally:
                        loop.close()
                
                # Sleep before next batch processing cycle
                time.sleep(1.0)  # 1 second batch processing interval
                
            except Exception as e:
                logger.error(f"Batch processing loop error: {e}")
                time.sleep(5.0)  # Longer sleep on error
        
        logger.info("Batch processing loop stopped")
    
    # Phase 3: Enhanced System Integration Methods
    
    async def _store_discovery_mcp(self, discovery: LearningDiscovery) -> Optional[str]:
        """Store discovery directly via MCP with enhanced metadata."""
        try:
            # Create comprehensive content for MCP storage
            content = f"""Learning Discovery: {discovery.pattern_type}
Worker: {discovery.worker_id}
Location: {discovery.context.get('location', 'unknown')}
Confidence: {discovery.confidence:.3f}
Success Rate: {discovery.success_rate:.3f}
Sample Size: {discovery.sample_size}
Priority: {discovery.priority.name}
Pattern Data: {json.dumps(discovery.pattern_data, indent=2)}
Context: {json.dumps(discovery.context, indent=2)}
Execution Time: {discovery.execution_time_ms:.1f}ms"""
            
            # Generate comprehensive tags
            tags = [
                f"discovery_id_{discovery.discovery_id}",
                f"worker_id_{discovery.worker_id}",
                f"pattern_type_{discovery.pattern_type}",
                f"priority_{discovery.priority.name.lower()}",
                "learning_propagation",
                "parallel_execution_discovery",
            ]
            
            # Add location-specific tags
            if discovery.context.get("location"):
                tags.append(f"location_{discovery.context['location']}")
            
            # Add quality-based tags
            if discovery.confidence >= 0.8:
                tags.append("high_confidence")
            elif discovery.confidence >= 0.6:
                tags.append("medium_confidence")
            else:
                tags.append("low_confidence")
            
            # Store via MCP using global functions if available
            try:
                import __main__
                if hasattr(__main__, 'mcp__memgraph_memory__store_memory'):
                    store_memory = __main__.mcp__memgraph_memory__store_memory
                    memory_id = store_memory(
                        node_type="concept",
                        content=content,
                        confidence=discovery.confidence,
                        source="parallel_execution_coordinator", 
                        tags=tags
                    )
                    logger.debug(f"Stored discovery via MCP (memory_id={memory_id})")
                    return memory_id
            except Exception as mcp_error:
                logger.warning(f"MCP storage failed: {mcp_error}")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to store discovery via MCP: {e}")
            return None
    
    async def integrate_with_opus_strategist(
        self, 
        discoveries: List[LearningDiscovery],
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate discoveries with OpusStrategist for strategic insights.
        
        Args:
            discoveries: Recent learning discoveries
            game_state: Current game state
            
        Returns:
            Strategic insights and recommendations
        """
        try:
            # Convert discoveries to execution results format for OpusStrategist
            recent_results = []
            for discovery in discoveries:
                execution_result = {
                    "worker_id": discovery.worker_id,
                    "success": discovery.success_rate > 0.6,
                    "execution_time": discovery.execution_time_ms / 1000.0,
                    "actions_taken": discovery.pattern_data.get("actions", []),
                    "final_state": discovery.context,
                    "patterns_discovered": [discovery.pattern_type],
                }
                recent_results.append(execution_result)
            
            # Request strategic planning from Opus
            strategic_plan = self.opus_strategist.request_strategy(
                game_state=game_state,
                recent_results=recent_results,
                priority=self.opus_strategist.StrategyPriority.NORMAL
            )
            
            logger.info(
                f"Integrated {len(discoveries)} discoveries with OpusStrategist",
                strategy_id=strategic_plan.get("strategy_id")
            )
            
            return strategic_plan
            
        except Exception as e:
            logger.error(f"Opus strategist integration failed: {e}")
            return {
                "strategy_id": f"fallback_{int(time.time())}",
                "experiments": [],
                "strategic_insights": [f"Integration failed: {str(e)}"],
                "next_checkpoints": [],
            }
    
    def get_worker_learning_summary(self, worker_id: str) -> Dict[str, Any]:
        """
        Get learning summary for a specific worker.
        
        Args:
            worker_id: Worker ID to get summary for
            
        Returns:
            Dictionary with worker's learning statistics
        """
        with self._lock:
            worker_discoveries = [
                d for d in self._pending_discoveries.values() 
                if d.worker_id == worker_id
            ]
            
            # Add discoveries from cache
            with self._cache_lock:
                worker_discoveries.extend([
                    d for d in self._discovery_cache.values() 
                    if d.worker_id == worker_id
                ])
            
            if not worker_discoveries:
                return {"worker_id": worker_id, "total_discoveries": 0}
            
            # Calculate statistics
            total_discoveries = len(worker_discoveries)
            avg_confidence = sum(d.confidence for d in worker_discoveries) / total_discoveries
            avg_success_rate = sum(d.success_rate for d in worker_discoveries) / total_discoveries
            pattern_types = list(set(d.pattern_type for d in worker_discoveries))
            
            # Count by priority
            priority_counts = {}
            for priority in LearningPriority:
                priority_counts[priority.name] = sum(
                    1 for d in worker_discoveries if d.priority == priority
                )
            
            return {
                "worker_id": worker_id,
                "total_discoveries": total_discoveries,
                "avg_confidence": avg_confidence,
                "avg_success_rate": avg_success_rate,
                "pattern_types": pattern_types,
                "priority_distribution": priority_counts,
                "latest_discovery": max(worker_discoveries, key=lambda d: d.discovered_at).discovery_id,
            }
    
    # Phase 4: Advanced Features
    
    async def adaptive_propagation_strategy(
        self, 
        discovery: LearningDiscovery,
        system_load: Dict[str, Any]
    ) -> PropagationStrategy:
        """
        Dynamically select propagation strategy based on system conditions.
        
        Args:
            discovery: Discovery to propagate
            system_load: Current system load metrics
            
        Returns:
            Optimal propagation strategy for current conditions
        """
        try:
            # Get current worker pool status
            healthy_workers = self.worker_pool.get_healthy_worker_count()
            total_workers = self.worker_pool.get_worker_count()
            queue_size = self.worker_pool.get_queue_size()
            
            # Calculate system load score
            worker_availability = healthy_workers / max(1, total_workers)
            queue_pressure = min(1.0, queue_size / 10.0)  # Normalize queue pressure
            
            overall_load = 1.0 - (worker_availability * (1.0 - queue_pressure))
            
            # Select strategy based on conditions
            if discovery.priority == LearningPriority.CRITICAL:
                return PropagationStrategy.IMMEDIATE
            
            if overall_load > 0.8:  # High system load
                return PropagationStrategy.BATCHED
            elif overall_load > 0.6:  # Medium system load
                return PropagationStrategy.THRESHOLD_BASED
            elif discovery.confidence > 0.9:  # High confidence discovery
                return PropagationStrategy.IMMEDIATE
            else:
                return PropagationStrategy.ADAPTIVE
        
        except Exception as e:
            logger.error(f"Adaptive strategy selection failed: {e}")
            return self.config.default_propagation_strategy
    
    async def performance_optimized_propagation(
        self,
        discovery: LearningDiscovery,
        performance_budget_ms: float = 50.0
    ) -> PropagationResult:
        """
        Execute propagation with performance optimization and budget enforcement.
        
        Args:
            discovery: Discovery to propagate
            performance_budget_ms: Maximum time budget for operation
            
        Returns:
            PropagationResult with performance metrics
        """
        start_time = time.time()
        
        try:
            # Check if we can meet performance budget
            estimated_time = self._estimate_propagation_time(discovery)
            if estimated_time > performance_budget_ms:
                logger.warning(
                    f"Estimated propagation time {estimated_time:.1f}ms exceeds budget {performance_budget_ms}ms"
                )
                # Use faster batched approach
                return await self._fast_propagation(discovery, performance_budget_ms)
            
            # Use timeout for performance enforcement
            try:
                result = await asyncio.wait_for(
                    self.propagate_learning(discovery),
                    timeout=performance_budget_ms / 1000.0
                )
                return result
                
            except asyncio.TimeoutError:
                execution_time_ms = (time.time() - start_time) * 1000
                logger.warning(f"Propagation timed out after {execution_time_ms:.1f}ms")
                
                return PropagationResult(
                    success=False,
                    error_message=f"Propagation timeout after {execution_time_ms:.1f}ms",
                    execution_time_ms=execution_time_ms,
                    performance_metrics={"timeout": True, "budget_ms": performance_budget_ms}
                )
        
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Performance optimized propagation failed: {e}")
            
            return PropagationResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )
    
    def _estimate_propagation_time(self, discovery: LearningDiscovery) -> float:
        """Estimate propagation time based on historical data and discovery characteristics."""
        base_time = 20.0  # Base propagation time in ms
        
        # Factor in discovery complexity
        pattern_complexity = len(str(discovery.pattern_data)) / 1000.0  # Rough complexity measure
        complexity_factor = 1.0 + (pattern_complexity * 0.1)
        
        # Factor in worker count
        worker_count = self.worker_pool.get_healthy_worker_count()
        worker_factor = 1.0 + (worker_count * 0.05)
        
        # Factor in conflict probability
        conflict_factor = 1.0
        if discovery.pattern_type in ["movement", "battle"]:  # Common conflict-prone patterns
            conflict_factor = 1.3
        
        estimated_time = base_time * complexity_factor * worker_factor * conflict_factor
        
        return estimated_time
    
    async def _fast_propagation(
        self, 
        discovery: LearningDiscovery, 
        budget_ms: float
    ) -> PropagationResult:
        """Execute fast propagation with reduced feature set to meet performance budget."""
        start_time = time.time()
        
        try:
            # Skip conflict detection for speed
            # Skip MCP storage for speed
            # Focus only on core worker propagation
            
            # Get subset of workers for faster propagation
            max_workers = min(3, self.worker_pool.get_healthy_worker_count())  # Limit workers for speed
            target_workers = []
            
            for i in range(max_workers):
                worker_status = self.worker_pool.get_worker_status(f"sonnet_worker_{i}")
                if worker_status and worker_status.get("healthy"):
                    target_workers.append(worker_status["worker_id"])
            
            # Fast propagation to limited workers
            successful_workers = []
            for worker_id in target_workers:
                try:
                    # Use simplified pattern data for speed
                    simple_pattern = {
                        "strategy_id": discovery.discovery_id,
                        "name": f"Fast {discovery.pattern_type}",
                        "pattern_sequence": [discovery.pattern_type],
                        "success_rate": discovery.success_rate,
                    }
                    
                    success = self.worker_pool.share_pattern(simple_pattern, discovery.worker_id)
                    if success:
                        successful_workers.append(worker_id)
                        
                except Exception as e:
                    logger.debug(f"Fast propagation to {worker_id} failed: {e}")
                    continue
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            result = PropagationResult(
                success=len(successful_workers) > 0,
                discoveries_propagated=[discovery.discovery_id],
                workers_updated=successful_workers,
                execution_time_ms=execution_time_ms,
                performance_metrics={
                    "fast_mode": True,
                    "budget_ms": budget_ms,
                    "workers_targeted": len(target_workers),
                }
            )
            
            logger.debug(
                f"Fast propagation completed",
                execution_time_ms=execution_time_ms,
                workers_updated=len(successful_workers)
            )
            
            return result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return PropagationResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
                performance_metrics={"fast_mode": True, "error": True}
            )
    
    # Enhanced MCP Integration Methods
    
    async def sync_with_memory_graph(self) -> Dict[str, Any]:
        """
        Synchronize pending discoveries with persistent memory graph.
        
        Returns:
            Synchronization result with statistics
        """
        start_time = time.time()
        sync_stats = {
            "discoveries_synced": 0,
            "conflicts_detected": 0,
            "errors": 0,
            "execution_time_ms": 0.0
        }
        
        try:
            with self._lock:
                pending_discoveries = list(self._pending_discoveries.values())
            
            for discovery in pending_discoveries:
                try:
                    await self._store_discovery_persistent(discovery)
                    sync_stats["discoveries_synced"] += 1
                    
                    # Remove from pending after successful sync
                    with self._lock:
                        self._pending_discoveries.pop(discovery.discovery_id, None)
                        
                except Exception as e:
                    logger.error(f"Failed to sync discovery {discovery.discovery_id}: {e}")
                    sync_stats["errors"] += 1
            
            sync_stats["execution_time_ms"] = (time.time() - start_time) * 1000
            
            logger.info(
                f"Memory graph sync completed",
                **sync_stats
            )
            
            return sync_stats
            
        except Exception as e:
            sync_stats["execution_time_ms"] = (time.time() - start_time) * 1000
            sync_stats["errors"] += 1
            logger.error(f"Memory graph sync failed: {e}")
            return sync_stats
    
    async def query_related_discoveries(
        self,
        discovery: LearningDiscovery,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Query for related discoveries to enhance propagation context.
        
        Args:
            discovery: Discovery to find related patterns for
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of related discoveries
        """
        try:
            criteria = {
                "pattern_type": discovery.pattern_type,
                "location": discovery.context.get("location"),
                "success_rate_min": max(0.0, discovery.success_rate - 0.2),
                "limit": 10
            }
            
            related_patterns = self.memory_graph.query_patterns(criteria)
            
            # Filter by similarity (simplified heuristic)
            similar_discoveries = []
            for pattern in related_patterns:
                similarity_score = self._calculate_discovery_similarity(discovery, pattern)
                if similarity_score >= similarity_threshold:
                    pattern["similarity_score"] = similarity_score
                    similar_discoveries.append(pattern)
            
            # Sort by similarity score
            similar_discoveries.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            logger.debug(
                f"Found {len(similar_discoveries)} related discoveries",
                discovery_id=discovery.discovery_id,
                similarity_threshold=similarity_threshold
            )
            
            return similar_discoveries
            
        except Exception as e:
            logger.error(f"Failed to query related discoveries: {e}")
            return []
    
    def _calculate_discovery_similarity(
        self, 
        discovery1: LearningDiscovery, 
        discovery2_data: Dict[str, Any]
    ) -> float:
        """Calculate similarity score between discoveries."""
        try:
            # Pattern type similarity
            type_match = 1.0 if discovery1.pattern_type == discovery2_data.get("pattern_type") else 0.0
            
            # Location similarity  
            location1 = discovery1.context.get("location", "")
            location2 = discovery2_data.get("location", "")
            location_match = 1.0 if location1 == location2 else 0.0
            
            # Confidence similarity (how close the confidence values are)
            conf1 = discovery1.confidence
            conf2 = discovery2_data.get("confidence", 0.0)
            confidence_similarity = 1.0 - abs(conf1 - conf2)
            
            # Success rate similarity
            sr1 = discovery1.success_rate
            sr2 = discovery2_data.get("success_rate", 0.0)  
            success_rate_similarity = 1.0 - abs(sr1 - sr2)
            
            # Weighted average
            similarity = (
                type_match * 0.4 +
                location_match * 0.3 +
                confidence_similarity * 0.2 +
                success_rate_similarity * 0.1
            )
            
            return similarity
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0


class LearningPropagationObserver(ILearningObserver):
    """
    Default observer implementation for learning propagation events.
    
    Provides logging and basic metrics collection for propagation events.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.events_received = 0
        self.last_event_time = 0.0
    
    def on_learning_discovered(self, discovery: LearningDiscovery) -> None:
        """Log learning discovery events."""
        self.events_received += 1
        self.last_event_time = time.time()
        
        logger.info(
            f"[{self.name}] Learning discovered",
            discovery_id=discovery.discovery_id,
            worker_id=discovery.worker_id,
            pattern_type=discovery.pattern_type,
            confidence=discovery.confidence
        )
    
    def on_learning_propagated(self, discovery: LearningDiscovery, result: PropagationResult) -> None:
        """Log learning propagation events."""
        self.events_received += 1
        self.last_event_time = time.time()
        
        logger.info(
            f"[{self.name}] Learning propagated",
            discovery_id=discovery.discovery_id,
            workers_updated=len(result.workers_updated),
            execution_time_ms=result.execution_time_ms,
            success=result.success
        )
    
    def on_conflict_detected(self, conflict: LearningConflict) -> None:
        """Log conflict detection events."""
        self.events_received += 1
        self.last_event_time = time.time()
        
        logger.warning(
            f"[{self.name}] Learning conflict detected",
            conflict_id=conflict.conflict_id,
            conflicting_discoveries=len(conflict.conflicting_discoveries),
            severity=conflict.get_conflict_severity()
        )
    
    def on_conflict_resolved(self, conflict: LearningConflict, resolution: LearningDiscovery) -> None:
        """Log conflict resolution events."""
        self.events_received += 1
        self.last_event_time = time.time()
        
        logger.info(
            f"[{self.name}] Learning conflict resolved",
            conflict_id=conflict.conflict_id,
            resolution_discovery_id=resolution.discovery_id,
            resolution_time_ms=conflict.resolution_time_ms
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get observer statistics."""
        return {
            "name": self.name,
            "events_received": self.events_received,
            "last_event_time": self.last_event_time,
        }