"""
Application-specific metrics for Pokemon speedrun learning agent.

Tracks domain-specific performance indicators including experiment success rates,
pattern discovery rates, script compilation performance, and AI strategy effectiveness.

Performance requirements:
- Metric collection: <2ms per measurement  
- Event-driven updates for real-time monitoring
- Memory efficient: <20MB for metric storage
- Integration with existing performance monitoring
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of Pokemon speedrun experiments."""
    
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"


class PatternType(Enum):
    """Types of discovered patterns."""
    
    MOVEMENT = "movement"
    BATTLE = "battle"
    MENU_NAVIGATION = "menu_navigation"
    ITEM_USAGE = "item_usage"
    DIALOGUE = "dialogue"
    OPTIMIZATION = "optimization"


@dataclass
class ExperimentResult:
    """Result of a single Pokemon speedrun experiment."""
    
    experiment_id: str
    status: ExperimentStatus
    duration_seconds: float
    script_length: int = 0
    script_compilation_time_ms: float = 0.0
    checkpoint_loading_time_ms: float = 0.0
    ai_strategy: str = "unknown"
    pattern_discoveries: int = 0
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    performance_score: float = 0.0  # 0.0-1.0 quality score


@dataclass
class PatternDiscovery:
    """Record of a new pattern discovered by the AI system."""
    
    pattern_id: str
    pattern_type: PatternType
    quality_score: float  # 0.0-1.0
    discovery_time_seconds: float
    experiment_id: str
    ai_worker: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    reuse_count: int = 0  # How many times pattern has been reused


@dataclass
class SpeedrunMetricsSnapshot:
    """Snapshot of current speedrun performance metrics."""
    
    # Experiment metrics
    total_experiments: int = 0
    successful_experiments: int = 0
    failed_experiments: int = 0
    average_experiment_duration: float = 0.0
    experiment_success_rate: float = 0.0
    
    # Performance metrics
    average_script_compilation_ms: float = 0.0
    average_checkpoint_loading_ms: float = 0.0
    compilation_target_compliance: float = 0.0  # % under 100ms target
    checkpoint_target_compliance: float = 0.0   # % under 500ms target
    
    # Pattern discovery metrics
    total_patterns_discovered: int = 0
    pattern_discovery_rate_per_hour: float = 0.0
    pattern_reuse_rate: float = 0.0
    average_pattern_quality: float = 0.0
    
    # AI strategy metrics
    strategy_effectiveness: Dict[str, float] = field(default_factory=dict)
    most_effective_strategy: str = "unknown"
    
    # System performance
    emulator_utilization: float = 0.0  # % of time emulators are active
    
    timestamp: float = field(default_factory=time.time)


class SpeedrunMetricsCollector:
    """
    Collector for Pokemon speedrun application-specific metrics.
    
    Tracks domain-specific performance indicators and provides
    comprehensive monitoring of AI system effectiveness.
    """
    
    def __init__(self, max_stored_experiments: int = 1000, max_stored_patterns: int = 500):
        """
        Initialize speedrun metrics collector.

        Args:
            max_stored_experiments: Maximum experiments to keep in memory
            max_stored_patterns: Maximum pattern discoveries to keep in memory
        """
        self.max_stored_experiments = max_stored_experiments
        self.max_stored_patterns = max_stored_patterns
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._experiments: List[ExperimentResult] = []
        self._pattern_discoveries: List[PatternDiscovery] = []
        
        # Performance tracking
        self._measurement_times: List[float] = []
        self._max_timing_samples = 100
        
        # Cached metrics for performance
        self._cached_snapshot: Optional[SpeedrunMetricsSnapshot] = None
        self._last_snapshot_time: float = 0.0
        self._snapshot_cache_duration: float = 5.0  # 5 second cache
        
        logger.info(f"SpeedrunMetricsCollector initialized (max_experiments={max_stored_experiments})")

    def record_experiment(self, result: ExperimentResult):
        """
        Record a completed experiment with performance measurement.
        
        Args:
            result: ExperimentResult containing experiment details
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Add to storage with size limit
                self._experiments.append(result)
                if len(self._experiments) > self.max_stored_experiments:
                    self._experiments.pop(0)  # Remove oldest
                
                # Invalidate cached snapshot
                self._cached_snapshot = None
                
                logger.debug(
                    f"Recorded experiment {result.experiment_id}: {result.status.value} "
                    f"({result.duration_seconds:.2f}s)"
                )
                
        except Exception as e:
            logger.error(f"Failed to record experiment: {e}")
        finally:
            # Track measurement performance
            measurement_time = time.time() - start_time
            self._record_measurement_time(measurement_time)

    def record_pattern_discovery(self, pattern: PatternDiscovery):
        """
        Record a new pattern discovery.
        
        Args:
            pattern: PatternDiscovery containing pattern details
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Add to storage with size limit
                self._pattern_discoveries.append(pattern)
                if len(self._pattern_discoveries) > self.max_stored_patterns:
                    self._pattern_discoveries.pop(0)  # Remove oldest
                
                # Invalidate cached snapshot
                self._cached_snapshot = None
                
                logger.info(
                    f"Pattern discovered: {pattern.pattern_type.value} "
                    f"(quality: {pattern.quality_score:.2f})"
                )
                
        except Exception as e:
            logger.error(f"Failed to record pattern discovery: {e}")
        finally:
            # Track measurement performance
            measurement_time = time.time() - start_time
            self._record_measurement_time(measurement_time)

    def record_pattern_reuse(self, pattern_id: str):
        """
        Record when a pattern is reused in another experiment.
        
        Args:
            pattern_id: ID of the pattern being reused
        """
        with self._lock:
            for pattern in self._pattern_discoveries:
                if pattern.pattern_id == pattern_id:
                    pattern.reuse_count += 1
                    logger.debug(f"Pattern {pattern_id} reuse count: {pattern.reuse_count}")
                    break

    def get_metrics_snapshot(self) -> SpeedrunMetricsSnapshot:
        """
        Get comprehensive metrics snapshot with caching.
        
        Returns:
            SpeedrunMetricsSnapshot with current system performance
        """
        current_time = time.time()
        
        # Check cache validity
        with self._lock:
            if (self._cached_snapshot and 
                current_time - self._last_snapshot_time < self._snapshot_cache_duration):
                return self._cached_snapshot
        
        # Generate fresh snapshot
        snapshot = self._generate_metrics_snapshot()
        
        with self._lock:
            self._cached_snapshot = snapshot
            self._last_snapshot_time = current_time
            
        return snapshot

    def _generate_metrics_snapshot(self) -> SpeedrunMetricsSnapshot:
        """Generate fresh metrics snapshot from stored data."""
        with self._lock:
            experiments = self._experiments[:]
            patterns = self._pattern_discoveries[:]
        
        if not experiments:
            return SpeedrunMetricsSnapshot()
        
        # Experiment statistics
        total_experiments = len(experiments)
        successful_experiments = sum(
            1 for exp in experiments if exp.status == ExperimentStatus.SUCCESS
        )
        failed_experiments = total_experiments - successful_experiments
        
        experiment_success_rate = (
            successful_experiments / total_experiments * 100.0 if total_experiments > 0 else 0.0
        )
        
        # Duration statistics
        durations = [exp.duration_seconds for exp in experiments]
        average_experiment_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Performance statistics
        compilation_times = [exp.script_compilation_time_ms for exp in experiments if exp.script_compilation_time_ms > 0]
        checkpoint_times = [exp.checkpoint_loading_time_ms for exp in experiments if exp.checkpoint_loading_time_ms > 0]
        
        average_script_compilation_ms = sum(compilation_times) / len(compilation_times) if compilation_times else 0.0
        average_checkpoint_loading_ms = sum(checkpoint_times) / len(checkpoint_times) if checkpoint_times else 0.0
        
        # Target compliance (performance requirements)
        compilation_target_compliance = (
            sum(1 for t in compilation_times if t < 100.0) / len(compilation_times) * 100.0
            if compilation_times else 0.0
        )
        checkpoint_target_compliance = (
            sum(1 for t in checkpoint_times if t < 500.0) / len(checkpoint_times) * 100.0  
            if checkpoint_times else 0.0
        )
        
        # Pattern discovery statistics
        total_patterns = len(patterns)
        
        # Calculate discovery rate (patterns per hour)
        if patterns and len(patterns) > 1:
            time_span_hours = (patterns[-1].timestamp - patterns[0].timestamp) / 3600.0
            pattern_discovery_rate_per_hour = total_patterns / time_span_hours if time_span_hours > 0 else 0.0
        else:
            pattern_discovery_rate_per_hour = 0.0
        
        # Pattern reuse rate
        total_reuses = sum(pattern.reuse_count for pattern in patterns)
        pattern_reuse_rate = total_reuses / total_patterns if total_patterns > 0 else 0.0
        
        # Average pattern quality
        pattern_qualities = [pattern.quality_score for pattern in patterns]
        average_pattern_quality = sum(pattern_qualities) / len(pattern_qualities) if pattern_qualities else 0.0
        
        # AI strategy effectiveness
        strategy_stats = {}
        for exp in experiments:
            if exp.ai_strategy not in strategy_stats:
                strategy_stats[exp.ai_strategy] = {"count": 0, "success": 0}
            strategy_stats[exp.ai_strategy]["count"] += 1
            if exp.status == ExperimentStatus.SUCCESS:
                strategy_stats[exp.ai_strategy]["success"] += 1
        
        strategy_effectiveness = {
            strategy: stats["success"] / stats["count"] * 100.0
            for strategy, stats in strategy_stats.items()
        }
        
        most_effective_strategy = (
            max(strategy_effectiveness.items(), key=lambda x: x[1])[0]
            if strategy_effectiveness else "unknown"
        )
        
        return SpeedrunMetricsSnapshot(
            # Experiment metrics
            total_experiments=total_experiments,
            successful_experiments=successful_experiments,
            failed_experiments=failed_experiments,
            average_experiment_duration=average_experiment_duration,
            experiment_success_rate=experiment_success_rate,
            
            # Performance metrics
            average_script_compilation_ms=average_script_compilation_ms,
            average_checkpoint_loading_ms=average_checkpoint_loading_ms,
            compilation_target_compliance=compilation_target_compliance,
            checkpoint_target_compliance=checkpoint_target_compliance,
            
            # Pattern metrics
            total_patterns_discovered=total_patterns,
            pattern_discovery_rate_per_hour=pattern_discovery_rate_per_hour,
            pattern_reuse_rate=pattern_reuse_rate,
            average_pattern_quality=average_pattern_quality,
            
            # Strategy metrics
            strategy_effectiveness=strategy_effectiveness,
            most_effective_strategy=most_effective_strategy,
        )

    def get_sla_compliance(self) -> Dict[str, Any]:
        """
        Calculate SLA compliance metrics.
        
        Returns:
            Dictionary with SLA compliance percentages and status
        """
        snapshot = self.get_metrics_snapshot()
        
        # SLA thresholds (configurable)
        sla_thresholds = {
            "experiment_success_rate_min": 95.0,  # 95% success rate
            "compilation_time_max_ms": 100.0,     # <100ms compilation
            "checkpoint_time_max_ms": 500.0,      # <500ms checkpoint loading
            "pattern_quality_min": 0.7,           # 70% pattern quality
        }
        
        # Calculate compliance
        compliance = {
            "experiment_success_sla": snapshot.experiment_success_rate >= sla_thresholds["experiment_success_rate_min"],
            "compilation_performance_sla": snapshot.compilation_target_compliance >= 95.0,
            "checkpoint_performance_sla": snapshot.checkpoint_target_compliance >= 95.0,
            "pattern_quality_sla": snapshot.average_pattern_quality >= sla_thresholds["pattern_quality_min"],
        }
        
        overall_compliance = all(compliance.values())
        
        return {
            "overall_sla_compliant": overall_compliance,
            "individual_compliance": compliance,
            "thresholds": sla_thresholds,
            "current_values": {
                "experiment_success_rate": snapshot.experiment_success_rate,
                "compilation_target_compliance": snapshot.compilation_target_compliance,
                "checkpoint_target_compliance": snapshot.checkpoint_target_compliance,
                "average_pattern_quality": snapshot.average_pattern_quality,
            },
            "timestamp": time.time(),
        }

    def _record_measurement_time(self, measurement_time: float):
        """Record measurement timing for performance monitoring."""
        with self._lock:
            self._measurement_times.append(measurement_time)
            if len(self._measurement_times) > self._max_timing_samples:
                self._measurement_times.pop(0)
                
            # Alert if measurement exceeds target
            if measurement_time > 0.002:  # 2ms threshold
                logger.warning(
                    f"Metrics measurement took {measurement_time:.4f}s (>0.002s target)"
                )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get collector performance metrics.
        
        Returns:
            Dictionary with timing and efficiency statistics
        """
        with self._lock:
            if not self._measurement_times:
                return {"error": "No timing measurements available"}
                
            timing_values = self._measurement_times[:]
            
            return {
                "measurement_performance": {
                    "average_ms": round(sum(timing_values) / len(timing_values) * 1000, 3),
                    "max_ms": round(max(timing_values) * 1000, 3),
                    "min_ms": round(min(timing_values) * 1000, 3),
                    "target_ms": 2.0,
                },
                "storage_efficiency": {
                    "experiments_stored": len(self._experiments),
                    "patterns_stored": len(self._pattern_discoveries),
                    "max_experiments": self.max_stored_experiments,
                    "max_patterns": self.max_stored_patterns,
                },
                "cache_performance": {
                    "snapshot_cached": self._cached_snapshot is not None,
                    "cache_age_seconds": time.time() - self._last_snapshot_time,
                    "cache_duration": self._snapshot_cache_duration,
                }
            }

    def reset_metrics(self):
        """Reset all collected metrics and statistics."""
        with self._lock:
            self._experiments.clear()
            self._pattern_discoveries.clear()
            self._measurement_times.clear()
            self._cached_snapshot = None
            self._last_snapshot_time = 0.0
            logger.info("Speedrun metrics reset")