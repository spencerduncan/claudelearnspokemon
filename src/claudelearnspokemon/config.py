"""
Configuration management for ML-guided experiment selection system.

This module provides centralized configuration for ML parameters, performance thresholds,
and system settings used throughout the experiment selection and reinforcement learning
components.

Design follows Clean Code principles:
- Single Responsibility: Centralized configuration management
- Open/Closed: Extensible for new configuration categories
- Interface Segregation: Organized by functional areas
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional


@dataclass
class ScriptDevelopmentConfig:
    """Configuration for script development and performance monitoring."""
    
    performance_threshold_ms: int = 100
    ml_model_fallback_threshold: int = 50
    max_performance_violations: int = 3
    exploration_rate_bounds: Tuple[float, float] = (0.05, 0.3)
    pattern_cache_size: int = 1000
    experience_buffer_limit: int = 1000
    thread_pool_size: int = 4
    ml_timeout_ms: int = 80
    
    # Genetic Algorithm Configuration
    GENETIC_POPULATION_SIZE: int = 8
    GENETIC_ELITE_SIZE: int = 2
    DIVERSITY_THRESHOLD: float = 0.3
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    max_generations: int = 10
    
    # Quality Thresholds
    SUCCESS_QUALITY_THRESHOLD: float = 0.8
    ACCEPTABLE_QUALITY_THRESHOLD: float = 0.6
    
    # Script Development Parameters
    DEFAULT_MAX_ITERATIONS: int = 5
    MAX_PATTERNS_IN_PROMPT: int = 10
    METRICS_LEARNING_RATE: float = 0.1
    SUCCESS_RATE_INIT: float = 0.7
    FAILED_RATE_INIT: float = 0.3


@dataclass
class ReinforcementLearningConfig:
    """Configuration for reinforcement learning parameters."""
    
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 0.2
    replay_buffer_size: int = 1000
    batch_size: int = 32
    target_update_frequency: int = 100
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01


@dataclass
class ExperimentSelectionConfig:
    """Configuration for experiment selection strategies."""
    
    max_candidates_per_selection: int = 10
    selection_timeout_ms: int = 200
    heuristic_weight_priority: float = 0.4
    heuristic_weight_relevance: float = 0.3
    heuristic_weight_success_rate: float = 0.3
    ml_confidence_threshold: float = 0.7
    adaptive_learning_window: int = 20
    hybrid_strategy_ratio: float = 0.6


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and optimization."""
    
    metrics_collection_interval_ms: int = 1000
    performance_degradation_threshold: float = 0.2
    memory_usage_limit_mb: int = 512
    cpu_usage_limit_percent: float = 80.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_ms: int = 30000
    health_check_interval_ms: int = 5000
    
    # Worker Pool Performance Configuration
    WORKER_HEALTH_CHECK_TIMEOUT: float = 0.01  # 10ms
    WORKER_POOL_INIT_TIMEOUT: float = 0.5      # 500ms
    TASK_ASSIGNMENT_TIMEOUT: float = 0.05      # 50ms
    UUID_SHORT_LENGTH: int = 8
    PATTERN_DISTRIBUTION_TIMEOUT: float = 0.1  # 100ms
    RESULT_ANALYSIS_TIMEOUT: float = 0.1       # 100ms


@dataclass
class WorkerConfig:
    """Configuration for worker pool management."""
    
    DEFAULT_WORKER_COUNT: int = 4
    WORKER_ID_PREFIX: str = "sonnet_worker"
    TASK_ID_PREFIX: str = "task"


class ConfigManager:
    """
    Central configuration manager for the ML-guided experiment selection system.
    
    Provides access to all configuration categories and handles configuration
    validation and runtime updates.
    """
    
    def __init__(self):
        """Initialize configuration manager with default values."""
        self.script_development = ScriptDevelopmentConfig()
        self.reinforcement_learning = ReinforcementLearningConfig()
        self.experiment_selection = ExperimentSelectionConfig()
        self.performance = PerformanceConfig()
        self.worker = WorkerConfig()
        
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get all configuration as a dictionary.
        
        Returns:
            Dictionary containing all configuration sections
        """
        return {
            'script_development': self.script_development.__dict__,
            'reinforcement_learning': self.reinforcement_learning.__dict__,
            'experiment_selection': self.experiment_selection.__dict__,
            'performance': self.performance.__dict__,
            'worker': self.worker.__dict__
        }
    
    def validate_config(self) -> bool:
        """
        Validate configuration parameters for consistency.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Validate performance threshold consistency
        if self.script_development.ml_timeout_ms >= self.script_development.performance_threshold_ms:
            return False
            
        # Validate exploration rate bounds
        min_rate, max_rate = self.script_development.exploration_rate_bounds
        if min_rate >= max_rate or min_rate < 0 or max_rate > 1:
            return False
            
        # Validate RL parameters
        if not (0 < self.reinforcement_learning.learning_rate <= 1):
            return False
            
        if not (0 < self.reinforcement_learning.discount_factor <= 1):
            return False
            
        # Validate experiment selection weights sum to 1
        weight_sum = (
            self.experiment_selection.heuristic_weight_priority +
            self.experiment_selection.heuristic_weight_relevance +
            self.experiment_selection.heuristic_weight_success_rate
        )
        if abs(weight_sum - 1.0) > 0.01:
            return False
            
        return True
    
    def update_exploration_rate(self, new_rate: float) -> bool:
        """
        Update the exploration rate within configured bounds.
        
        Args:
            new_rate: New exploration rate value
            
        Returns:
            True if update was successful, False if out of bounds
        """
        min_rate, max_rate = self.script_development.exploration_rate_bounds
        if min_rate <= new_rate <= max_rate:
            self.reinforcement_learning.exploration_rate = new_rate
            return True
        return False


# Global configuration instance
CONFIG = ConfigManager()

# For backward compatibility with the specification
CONFIG.script_development = {
    'performance_threshold_ms': 100,
    'ml_model_fallback_threshold': 50,
    'max_performance_violations': 3,
    'exploration_rate_bounds': [0.05, 0.3],
    'pattern_cache_size': 1000,
    'experience_buffer_limit': 1000,
    'thread_pool_size': 4,
    'ml_timeout_ms': 80,
    'GENETIC_POPULATION_SIZE': 8,
    'GENETIC_ELITE_SIZE': 2,
    'DIVERSITY_THRESHOLD': 0.3,
    'mutation_rate': 0.1,
    'crossover_rate': 0.7,
    'max_generations': 10,
    'SUCCESS_QUALITY_THRESHOLD': 0.8,
    'ACCEPTABLE_QUALITY_THRESHOLD': 0.6,
    'DEFAULT_MAX_ITERATIONS': 5,
    'MAX_PATTERNS_IN_PROMPT': 10,
    'METRICS_LEARNING_RATE': 0.1,
    'SUCCESS_RATE_INIT': 0.7,
    'FAILED_RATE_INIT': 0.3
}

__all__ = [
    "ScriptDevelopmentConfig",
    "ReinforcementLearningConfig", 
    "ExperimentSelectionConfig",
    "PerformanceConfig",
    "WorkerConfig",
    "ConfigManager",
    "CONFIG"
]