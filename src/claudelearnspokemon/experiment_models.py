"""
Core data models for ML-guided experiment selection system.

This module provides data structures for experiment candidates, results, metrics,
and status tracking used throughout the experiment selection and reinforcement
learning components.

Design follows Clean Code principles:
- Single Responsibility: Each class has one clear purpose
- Immutability: Results and candidates are immutable after creation
- Type Safety: Comprehensive type hints for all interfaces
- Validation: Built-in validation for critical fields
"""

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ExperimentStatus(Enum):
    """Status enumeration for experiment lifecycle tracking."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class SelectionStrategy(Enum):
    """Strategy enumeration for experiment selection approaches."""
    
    HEURISTIC = "heuristic"
    ML_GUIDED = "ml_guided"  
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class ExperimentCandidate:
    """
    Represents a candidate experiment for execution.
    
    Immutable data structure containing all information needed to evaluate
    and execute an experiment, including pattern details, context, scoring,
    and resource requirements.
    """
    
    experiment_id: str
    pattern: Any  # PokemonStrategy type - avoiding import for compatibility
    context: Dict[str, Any]
    priority_score: float
    relevance_score: float
    estimated_success_rate: float
    estimated_execution_time: float
    resource_requirements: Dict[str, Union[float, int]]
    created_at: float
    source_worker: str
    
    # Optional fields with defaults
    ml_confidence: Optional[float] = None
    complexity_score: Optional[float] = None
    historical_performance: Optional[Dict[str, float]] = None
    dependencies: Optional[List[str]] = field(default_factory=list)
    tags: Optional[List[str]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate candidate fields after initialization."""
        if not self.experiment_id:
            raise ValueError("Experiment ID cannot be empty")
            
        if not (0.0 <= self.priority_score <= 1.0):
            raise ValueError("Priority score must be between 0.0 and 1.0")
            
        if not (0.0 <= self.relevance_score <= 1.0):
            raise ValueError("Relevance score must be between 0.0 and 1.0")
            
        if not (0.0 <= self.estimated_success_rate <= 1.0):
            raise ValueError("Estimated success rate must be between 0.0 and 1.0")
            
        if self.estimated_execution_time <= 0:
            raise ValueError("Estimated execution time must be positive")
            
        if self.created_at <= 0:
            raise ValueError("Created timestamp must be positive")
    
    @property
    def combined_score(self) -> float:
        """
        Calculate combined score for candidate ranking.
        
        Returns:
            Weighted combination of priority, relevance, and success rate
        """
        # Default weights - can be overridden by ML scoring
        weight_priority = 0.4
        weight_relevance = 0.3
        weight_success = 0.3
        
        base_score = (
            self.priority_score * weight_priority +
            self.relevance_score * weight_relevance +
            self.estimated_success_rate * weight_success
        )
        
        # Apply ML confidence boost if available
        if self.ml_confidence is not None:
            ml_boost = (self.ml_confidence - 0.5) * 0.2  # -0.1 to +0.1 adjustment
            base_score = min(1.0, max(0.0, base_score + ml_boost))
        
        return base_score
    
    @property
    def age_seconds(self) -> float:
        """Get age of this candidate in seconds."""
        return time.time() - self.created_at
    
    def with_ml_confidence(self, confidence: float) -> 'ExperimentCandidate':
        """
        Create new candidate with ML confidence score.
        
        Args:
            confidence: ML confidence score (0.0 to 1.0)
            
        Returns:
            New ExperimentCandidate with ML confidence applied
        """
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("ML confidence must be between 0.0 and 1.0")
            
        # Create new instance with updated ML confidence
        return ExperimentCandidate(
            experiment_id=self.experiment_id,
            pattern=self.pattern,
            context=self.context,
            priority_score=self.priority_score,
            relevance_score=self.relevance_score,
            estimated_success_rate=self.estimated_success_rate,
            estimated_execution_time=self.estimated_execution_time,
            resource_requirements=self.resource_requirements,
            created_at=self.created_at,
            source_worker=self.source_worker,
            ml_confidence=confidence,
            complexity_score=self.complexity_score,
            historical_performance=self.historical_performance,
            dependencies=self.dependencies,
            tags=self.tags
        )


@dataclass(frozen=True)
class ExperimentResult:
    """
    Captures the outcome of an executed experiment.
    
    Immutable data structure recording all experiment results including
    success status, timing, performance metrics, and learned patterns.
    """
    
    experiment_id: str
    success: bool
    execution_time: float
    performance_metrics: Dict[str, Union[float, int, str]]
    learned_patterns: List[Dict[str, Any]]
    completed_at: float
    
    # Optional fields
    status: ExperimentStatus = ExperimentStatus.COMPLETED
    error_message: Optional[str] = None
    resource_usage: Optional[Dict[str, Union[float, int]]] = None
    validation_results: Optional[Dict[str, Any]] = None
    improvement_suggestions: Optional[List[str]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate result fields after initialization."""
        if not self.experiment_id:
            raise ValueError("Experiment ID cannot be empty")
            
        if self.execution_time < 0:
            raise ValueError("Execution time cannot be negative")
            
        if self.completed_at <= 0:
            raise ValueError("Completion timestamp must be positive")
            
        if not self.success and not self.error_message:
            # This is a warning, not an error - allow failed experiments without error messages
            pass
    
    @property
    def efficiency_score(self) -> float:
        """
        Calculate efficiency score based on success and execution time.
        
        Returns:
            Efficiency score (0.0 to 1.0), higher is better
        """
        if not self.success:
            return 0.0
            
        # Normalize execution time (assuming 2.0s is "average")
        time_efficiency = max(0.1, min(1.0, 2.0 / max(0.1, self.execution_time)))
        
        # Weight success heavily but consider timing
        return 0.8 + (0.2 * time_efficiency)
    
    @property
    def learning_value(self) -> float:
        """
        Calculate learning value from this experiment.
        
        Returns:
            Learning value score (0.0 to 1.0)
        """
        base_learning = 0.5 if self.success else 0.3  # Failed experiments still teach
        
        # Bonus for learned patterns
        pattern_bonus = min(0.3, len(self.learned_patterns) * 0.1)
        
        # Bonus for performance insights
        metrics_bonus = min(0.2, len(self.performance_metrics) * 0.05)
        
        return min(1.0, base_learning + pattern_bonus + metrics_bonus)


class ExperimentMetrics:
    """
    Thread-safe metrics tracking for experiment selection system.
    
    Tracks performance statistics, ML usage, and system health metrics
    with atomic operations for concurrent access.
    """
    
    def __init__(self):
        """Initialize metrics with thread-safe counters."""
        self._lock = threading.Lock()
        
        # Experiment tracking
        self.experiments_completed: int = 0
        self.experiments_failed: int = 0
        self.experiments_cancelled: int = 0
        
        # Selection strategy usage
        self.heuristic_selections: int = 0
        self.ml_guided_selections: int = 0
        self.adaptive_selections: int = 0
        self.hybrid_selections: int = 0
        
        # Performance tracking
        self.total_selection_time: float = 0.0
        self.ml_prediction_time: float = 0.0
        self.performance_violations: int = 0
        
        # Success tracking
        self.successful_experiments: int = 0
        self.total_execution_time: float = 0.0
        self.learned_patterns_count: int = 0
        
        # ML model tracking
        self.ml_model_updates: int = 0
        self.experience_buffer_size: int = 0
        self.q_table_entries: int = 0
        
        # System health
        self.memory_usage_mb: float = 0.0
        self.cpu_usage_percent: float = 0.0
        self.thread_pool_utilization: float = 0.0
        
        # Timestamps
        self.last_reset_time: float = time.time()
        self.last_update_time: float = time.time()
    
    def record_experiment_completion(self, result: ExperimentResult) -> None:
        """
        Record completion of an experiment.
        
        Args:
            result: The experiment result to record
        """
        with self._lock:
            self.experiments_completed += 1
            self.total_execution_time += result.execution_time
            self.learned_patterns_count += len(result.learned_patterns)
            
            if result.success:
                self.successful_experiments += 1
            elif result.status == ExperimentStatus.FAILED:
                self.experiments_failed += 1
            elif result.status == ExperimentStatus.CANCELLED:
                self.experiments_cancelled += 1
                
            self.last_update_time = time.time()
    
    def record_selection_usage(self, strategy: SelectionStrategy, duration_ms: float) -> None:
        """
        Record usage of a selection strategy.
        
        Args:
            strategy: The selection strategy used
            duration_ms: Time taken for selection in milliseconds
        """
        with self._lock:
            if strategy == SelectionStrategy.HEURISTIC:
                self.heuristic_selections += 1
            elif strategy == SelectionStrategy.ML_GUIDED:
                self.ml_guided_selections += 1
            elif strategy == SelectionStrategy.ADAPTIVE:
                self.adaptive_selections += 1
            elif strategy == SelectionStrategy.HYBRID:
                self.hybrid_selections += 1
                
            self.total_selection_time += duration_ms
            self.last_update_time = time.time()
    
    def record_ml_prediction_time(self, duration_ms: float) -> None:
        """Record ML prediction timing."""
        with self._lock:
            self.ml_prediction_time += duration_ms
    
    def record_performance_violation(self) -> None:
        """Record a performance violation."""
        with self._lock:
            self.performance_violations += 1
    
    def update_ml_model_stats(self, buffer_size: int, q_table_size: int) -> None:
        """
        Update ML model statistics.
        
        Args:
            buffer_size: Current experience buffer size
            q_table_size: Current Q-table size
        """
        with self._lock:
            self.experience_buffer_size = buffer_size
            self.q_table_entries = q_table_size
            self.ml_model_updates += 1
    
    def update_system_health(self, memory_mb: float, cpu_percent: float, thread_util: float) -> None:
        """
        Update system health metrics.
        
        Args:
            memory_mb: Current memory usage in MB
            cpu_percent: Current CPU usage percentage
            thread_util: Thread pool utilization (0.0 to 1.0)
        """
        with self._lock:
            self.memory_usage_mb = memory_mb
            self.cpu_usage_percent = cpu_percent
            self.thread_pool_utilization = thread_util
    
    @property
    def success_rate(self) -> float:
        """Calculate overall experiment success rate."""
        with self._lock:
            if self.experiments_completed == 0:
                return 0.0
            return self.successful_experiments / self.experiments_completed
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average experiment execution time."""
        with self._lock:
            if self.experiments_completed == 0:
                return 0.0
            return self.total_execution_time / self.experiments_completed
    
    @property
    def average_selection_time(self) -> float:
        """Calculate average selection time in milliseconds."""
        with self._lock:
            total_selections = (self.heuristic_selections + self.ml_guided_selections +
                              self.adaptive_selections + self.hybrid_selections)
            if total_selections == 0:
                return 0.0
            return self.total_selection_time / total_selections
    
    @property
    def ml_usage_ratio(self) -> float:
        """Calculate ratio of ML-guided selections to total selections."""
        with self._lock:
            total_selections = (self.heuristic_selections + self.ml_guided_selections +
                              self.adaptive_selections + self.hybrid_selections)
            if total_selections == 0:
                return 0.0
            ml_selections = self.ml_guided_selections + self.adaptive_selections + self.hybrid_selections
            return ml_selections / total_selections
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary containing all current metrics
        """
        with self._lock:
            return {
                'experiments': {
                    'completed': self.experiments_completed,
                    'successful': self.successful_experiments,
                    'failed': self.experiments_failed,
                    'cancelled': self.experiments_cancelled,
                    'success_rate': self.success_rate,
                    'average_execution_time': self.average_execution_time,
                    'learned_patterns': self.learned_patterns_count
                },
                'selection_strategies': {
                    'heuristic': self.heuristic_selections,
                    'ml_guided': self.ml_guided_selections,
                    'adaptive': self.adaptive_selections,
                    'hybrid': self.hybrid_selections,
                    'average_time_ms': self.average_selection_time,
                    'ml_usage_ratio': self.ml_usage_ratio
                },
                'performance': {
                    'violations': self.performance_violations,
                    'total_selection_time': self.total_selection_time,
                    'ml_prediction_time': self.ml_prediction_time
                },
                'ml_model': {
                    'updates': self.ml_model_updates,
                    'experience_buffer_size': self.experience_buffer_size,
                    'q_table_entries': self.q_table_entries
                },
                'system_health': {
                    'memory_usage_mb': self.memory_usage_mb,
                    'cpu_usage_percent': self.cpu_usage_percent,
                    'thread_pool_utilization': self.thread_pool_utilization
                },
                'timestamps': {
                    'last_reset': self.last_reset_time,
                    'last_update': self.last_update_time,
                    'uptime_seconds': time.time() - self.last_reset_time
                }
            }
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self.__init__()


__all__ = [
    "ExperimentStatus",
    "SelectionStrategy", 
    "ExperimentCandidate",
    "ExperimentResult",
    "ExperimentMetrics"
]