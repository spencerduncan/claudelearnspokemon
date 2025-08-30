"""
Learning Propagation Interfaces and Data Models

Provides interfaces, enums, and data classes for the learning propagation system
in the ParallelExecutionCoordinator. Follows SOLID principles and Strategy pattern.

Key Components:
- ILearningPropagator: Core interface for learning propagation
- LearningDiscovery: Data class for discovered patterns
- PropagationStrategy: Strategy enum for different propagation approaches
- LearningConflict: Data class for handling learning conflicts

Performance Requirements:
- Learning discovery creation: <5ms
- Conflict resolution: <20ms
- Strategy selection: <10ms

Author: Felix (Craftsperson) - Claude Code Implementation Agent
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol


class PropagationStrategy(Enum):
    """Strategy enum for learning propagation approaches."""
    
    IMMEDIATE = "immediate"          # Propagate immediately upon discovery
    BATCHED = "batched"             # Batch multiple discoveries for efficiency
    THRESHOLD_BASED = "threshold"   # Propagate when confidence threshold met
    PRIORITY_BASED = "priority"     # Propagate based on discovery priority
    ADAPTIVE = "adaptive"           # Adapt strategy based on system load


class LearningPriority(Enum):
    """Priority levels for learning discoveries."""
    
    LOW = 1
    NORMAL = 2 
    HIGH = 3
    CRITICAL = 4


class ConflictResolutionStrategy(Enum):
    """Strategy for resolving learning conflicts."""
    
    LATEST_WINS = "latest_wins"               # Most recent discovery takes precedence
    HIGHEST_CONFIDENCE = "highest_confidence" # Discovery with highest confidence wins
    CONSENSUS_BASED = "consensus"             # Resolve based on worker consensus
    HYBRID_MERGE = "hybrid_merge"             # Attempt to merge conflicting discoveries


@dataclass
class LearningDiscovery:
    """
    Data class representing a discovered learning pattern.
    
    Encapsulates all information about a pattern discovered by a worker,
    including metadata for propagation and conflict resolution.
    """
    
    # Core identification
    discovery_id: str = field(default_factory=lambda: f"discovery_{uuid.uuid4().hex[:12]}")
    worker_id: str = ""
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Learning content
    pattern_type: str = ""                      # e.g., "movement", "battle", "tile_interaction"
    pattern_data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    confidence: float = 0.0                     # 0.0 - 1.0
    success_rate: float = 0.0                   # 0.0 - 1.0
    sample_size: int = 0                        # Number of observations
    
    # Propagation metadata
    priority: LearningPriority = LearningPriority.NORMAL
    propagated_to: List[str] = field(default_factory=list)  # List of worker IDs
    propagation_strategy: PropagationStrategy = PropagationStrategy.IMMEDIATE
    
    # Performance tracking
    execution_time_ms: float = 0.0
    memory_usage_bytes: int = 0
    
    # Integration metadata
    checkpoint_context: str = ""
    game_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate and normalize discovery data after initialization."""
        # Ensure confidence is in valid range
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Ensure success rate is in valid range
        self.success_rate = max(0.0, min(1.0, self.success_rate))
        
        # Ensure sample size is non-negative
        self.sample_size = max(0, self.sample_size)
        
        # Initialize propagated_to list if None
        if self.propagated_to is None:
            self.propagated_to = []
    
    def get_quality_score(self) -> float:
        """Calculate composite quality score for prioritization."""
        if self.sample_size == 0:
            return self.confidence * 0.5  # Lower weight for unvalidated discoveries
        
        # Combine confidence, success rate, and sample size reliability
        sample_weight = min(1.0, self.sample_size / 10.0)  # Max weight at 10+ samples
        return (self.confidence * 0.4 + self.success_rate * 0.4 + sample_weight * 0.2)
    
    def is_propagation_ready(self, min_confidence: float = 0.6) -> bool:
        """Check if discovery is ready for propagation based on quality criteria."""
        return (
            self.confidence >= min_confidence and 
            self.pattern_data and 
            self.pattern_type and
            self.sample_size > 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert discovery to dictionary for serialization."""
        return {
            "discovery_id": self.discovery_id,
            "worker_id": self.worker_id,
            "discovered_at": self.discovered_at.isoformat(),
            "pattern_type": self.pattern_type,
            "pattern_data": self.pattern_data,
            "context": self.context,
            "confidence": self.confidence,
            "success_rate": self.success_rate,
            "sample_size": self.sample_size,
            "priority": self.priority.value,
            "propagated_to": self.propagated_to,
            "propagation_strategy": self.propagation_strategy.value,
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_bytes": self.memory_usage_bytes,
            "checkpoint_context": self.checkpoint_context,
            "game_state_snapshot": self.game_state_snapshot,
            "quality_score": self.get_quality_score(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningDiscovery':
        """Create discovery from dictionary representation."""
        # Handle enum fields
        priority = LearningPriority(data.get("priority", LearningPriority.NORMAL.value))
        propagation_strategy = PropagationStrategy(
            data.get("propagation_strategy", PropagationStrategy.IMMEDIATE.value)
        )
        
        # Handle datetime field
        discovered_at = datetime.fromisoformat(
            data.get("discovered_at", datetime.now(timezone.utc).isoformat())
        )
        
        return cls(
            discovery_id=data.get("discovery_id", f"discovery_{uuid.uuid4().hex[:12]}"),
            worker_id=data.get("worker_id", ""),
            discovered_at=discovered_at,
            pattern_type=data.get("pattern_type", ""),
            pattern_data=data.get("pattern_data", {}),
            context=data.get("context", {}),
            confidence=data.get("confidence", 0.0),
            success_rate=data.get("success_rate", 0.0),
            sample_size=data.get("sample_size", 0),
            priority=priority,
            propagated_to=data.get("propagated_to", []),
            propagation_strategy=propagation_strategy,
            execution_time_ms=data.get("execution_time_ms", 0.0),
            memory_usage_bytes=data.get("memory_usage_bytes", 0),
            checkpoint_context=data.get("checkpoint_context", ""),
            game_state_snapshot=data.get("game_state_snapshot", {}),
        )


@dataclass
class LearningConflict:
    """
    Data class representing a conflict between learning discoveries.
    
    Used when multiple workers discover contradictory patterns
    that need resolution before propagation.
    """
    
    conflict_id: str = field(default_factory=lambda: f"conflict_{uuid.uuid4().hex[:8]}")
    conflicting_discoveries: List[LearningDiscovery] = field(default_factory=list)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_CONFIDENCE
    resolved: bool = False
    resolution_result: Optional[LearningDiscovery] = None
    resolution_time_ms: float = 0.0
    
    def get_conflict_severity(self) -> float:
        """Calculate severity score for conflict prioritization."""
        if len(self.conflicting_discoveries) < 2:
            return 0.0
        
        # Calculate confidence variance as conflict severity measure
        confidences = [d.confidence for d in self.conflicting_discoveries]
        confidence_variance = sum((c - sum(confidences) / len(confidences)) ** 2 for c in confidences) / len(confidences)
        
        # Higher variance = higher conflict severity
        return min(1.0, confidence_variance * 4.0)  # Scale to 0-1 range
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conflict to dictionary for serialization."""
        return {
            "conflict_id": self.conflict_id,
            "conflicting_discoveries": [d.to_dict() for d in self.conflicting_discoveries],
            "detected_at": self.detected_at.isoformat(),
            "resolution_strategy": self.resolution_strategy.value,
            "resolved": self.resolved,
            "resolution_result": self.resolution_result.to_dict() if self.resolution_result else None,
            "resolution_time_ms": self.resolution_time_ms,
            "conflict_severity": self.get_conflict_severity(),
        }


@dataclass
class PropagationResult:
    """Result of a learning propagation operation."""
    
    success: bool = False
    discoveries_propagated: List[str] = field(default_factory=list)  # Discovery IDs
    workers_updated: List[str] = field(default_factory=list)         # Worker IDs
    conflicts_resolved: List[str] = field(default_factory=list)      # Conflict IDs
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "discoveries_propagated": self.discoveries_propagated,
            "workers_updated": self.workers_updated,
            "conflicts_resolved": self.conflicts_resolved,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "performance_metrics": self.performance_metrics,
        }


class ILearningPropagator(Protocol):
    """
    Interface for learning propagation implementations.
    
    Defines the contract for propagating learning discoveries
    across worker pools with conflict resolution and performance optimization.
    """
    
    async def propagate_learning(
        self,
        discovery: LearningDiscovery,
        target_workers: Optional[List[str]] = None,
        strategy: Optional[PropagationStrategy] = None
    ) -> PropagationResult:
        """
        Propagate a learning discovery to workers.
        
        Args:
            discovery: Learning discovery to propagate
            target_workers: Optional list of worker IDs to target (None = all workers)
            strategy: Optional propagation strategy override
            
        Returns:
            PropagationResult with operation details
        """
        ...
    
    async def batch_propagate(
        self,
        discoveries: List[LearningDiscovery],
        strategy: Optional[PropagationStrategy] = None
    ) -> PropagationResult:
        """
        Propagate multiple discoveries in a batch operation.
        
        Args:
            discoveries: List of discoveries to propagate
            strategy: Optional propagation strategy override
            
        Returns:
            PropagationResult with batch operation details
        """
        ...
    
    async def resolve_conflict(
        self,
        conflict: LearningConflict,
        resolution_strategy: Optional[ConflictResolutionStrategy] = None
    ) -> LearningDiscovery:
        """
        Resolve conflicts between contradictory discoveries.
        
        Args:
            conflict: Conflict requiring resolution
            resolution_strategy: Optional strategy override for resolution
            
        Returns:
            Resolved learning discovery
        """
        ...


class ILearningObserver(Protocol):
    """
    Observer interface for learning propagation events.
    
    Allows components to be notified of learning propagation events
    for monitoring, logging, and analytics.
    """
    
    def on_learning_discovered(self, discovery: LearningDiscovery) -> None:
        """Called when a new learning is discovered."""
        ...
    
    def on_learning_propagated(self, discovery: LearningDiscovery, result: PropagationResult) -> None:
        """Called when a learning is successfully propagated."""
        ...
    
    def on_conflict_detected(self, conflict: LearningConflict) -> None:
        """Called when a learning conflict is detected."""
        ...
    
    def on_conflict_resolved(self, conflict: LearningConflict, resolution: LearningDiscovery) -> None:
        """Called when a learning conflict is resolved."""
        ...


@dataclass
class PropagationConfig:
    """Configuration for learning propagation system."""
    
    # Performance settings
    max_propagation_time_ms: float = 100.0      # Maximum time for propagation operation
    batch_size: int = 10                        # Maximum discoveries per batch
    max_concurrent_propagations: int = 5         # Maximum concurrent propagation operations
    
    # Quality thresholds
    min_confidence_threshold: float = 0.6       # Minimum confidence for propagation
    min_sample_size: int = 1                    # Minimum sample size for propagation
    conflict_detection_enabled: bool = True     # Enable automatic conflict detection
    
    # Strategy settings
    default_propagation_strategy: PropagationStrategy = PropagationStrategy.IMMEDIATE
    default_conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_CONFIDENCE
    
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5                  # Failures before opening circuit
    recovery_timeout_ms: float = 30000.0        # Time before attempting recovery
    
    # Cache settings
    discovery_cache_size: int = 1000            # Maximum cached discoveries
    discovery_cache_ttl_ms: float = 300000.0    # Cache TTL (5 minutes)
    
    def validate(self) -> List[str]:
        """Validate configuration parameters and return any issues."""
        issues = []
        
        if self.max_propagation_time_ms <= 0:
            issues.append("max_propagation_time_ms must be positive")
        
        if self.batch_size <= 0:
            issues.append("batch_size must be positive")
            
        if self.max_concurrent_propagations <= 0:
            issues.append("max_concurrent_propagations must be positive")
        
        if not (0.0 <= self.min_confidence_threshold <= 1.0):
            issues.append("min_confidence_threshold must be between 0.0 and 1.0")
        
        if self.min_sample_size < 0:
            issues.append("min_sample_size must be non-negative")
            
        if self.failure_threshold <= 0:
            issues.append("failure_threshold must be positive")
        
        if self.recovery_timeout_ms <= 0:
            issues.append("recovery_timeout_ms must be positive")
            
        if self.discovery_cache_size <= 0:
            issues.append("discovery_cache_size must be positive")
            
        if self.discovery_cache_ttl_ms <= 0:
            issues.append("discovery_cache_ttl_ms must be positive")
        
        return issues


@dataclass 
class PropagationMetrics:
    """Metrics for monitoring propagation performance."""
    
    # Operation counts
    total_discoveries: int = 0
    successful_propagations: int = 0
    failed_propagations: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    
    # Performance metrics
    avg_propagation_time_ms: float = 0.0
    max_propagation_time_ms: float = 0.0
    propagations_over_threshold: int = 0        # Propagations exceeding time threshold
    
    # Quality metrics
    avg_confidence_propagated: float = 0.0
    discoveries_below_threshold: int = 0        # Discoveries below confidence threshold
    
    # System health
    circuit_breaker_trips: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Worker distribution
    worker_propagation_counts: Dict[str, int] = field(default_factory=dict)
    
    def record_propagation(self, result: PropagationResult, discovery: LearningDiscovery) -> None:
        """Record metrics from a propagation operation."""
        self.total_discoveries += 1
        
        if result.success:
            self.successful_propagations += 1
            self.avg_propagation_time_ms = self._update_average(
                self.avg_propagation_time_ms, 
                result.execution_time_ms,
                self.successful_propagations
            )
            self.avg_confidence_propagated = self._update_average(
                self.avg_confidence_propagated,
                discovery.confidence,
                self.successful_propagations
            )
        else:
            self.failed_propagations += 1
        
        # Update max time
        self.max_propagation_time_ms = max(self.max_propagation_time_ms, result.execution_time_ms)
        
        # Check threshold breach
        if result.execution_time_ms > 100.0:  # 100ms threshold
            self.propagations_over_threshold += 1
        
        # Update worker counts
        for worker_id in result.workers_updated:
            self.worker_propagation_counts[worker_id] = self.worker_propagation_counts.get(worker_id, 0) + 1
    
    def record_conflict(self, conflict: LearningConflict, resolved: bool = False) -> None:
        """Record metrics from conflict detection/resolution."""
        self.conflicts_detected += 1
        if resolved:
            self.conflicts_resolved += 1
    
    def record_quality_issue(self, discovery: LearningDiscovery, issue_type: str) -> None:
        """Record quality issues with discoveries."""
        if issue_type == "below_threshold":
            self.discoveries_below_threshold += 1
    
    def _update_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update running average with new value."""
        if count <= 1:
            return new_value
        return (current_avg * (count - 1) + new_value) / count
    
    def get_success_rate(self) -> float:
        """Calculate overall propagation success rate."""
        total = self.successful_propagations + self.failed_propagations
        return self.successful_propagations / total if total > 0 else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "success_rate": self.get_success_rate(),
            "total_operations": self.total_discoveries,
            "avg_propagation_time_ms": self.avg_propagation_time_ms,
            "max_propagation_time_ms": self.max_propagation_time_ms,
            "performance_threshold_breaches": self.propagations_over_threshold,
            "conflicts_resolved_rate": (
                self.conflicts_resolved / self.conflicts_detected 
                if self.conflicts_detected > 0 else 0.0
            ),
            "avg_confidence": self.avg_confidence_propagated,
            "active_workers": len(self.worker_propagation_counts),
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            ),
        }


# Type aliases for better code readability
WorkerID = str
DiscoveryID = str
ConflictID = str
PropagationCallback = Callable[[LearningDiscovery, PropagationResult], None]


class LearningPropagationError(Exception):
    """Base exception for learning propagation operations."""
    pass


class PropagationTimeoutError(LearningPropagationError):
    """Raised when propagation exceeds time threshold."""
    pass


class ConflictResolutionError(LearningPropagationError):
    """Raised when conflict resolution fails."""
    pass


class WorkerUnavailableError(LearningPropagationError):
    """Raised when target workers are unavailable for propagation."""
    pass