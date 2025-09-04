"""
ExperimentSelector: Variation Generation Implementation

This module implements sophisticated variation generation for Pokemon speedrun experiments,
addressing common failure modes and edge cases through conservative design principles.

Author: worker1 (Rex) - Skeptical engineering approach
Focus: Robust variation generation with comprehensive error handling
"""

import hashlib
import json
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import heapq
import itertools
from abc import ABC, abstractmethod


class VariationStrategy(Enum):
    """Supported variation generation strategies with different risk profiles"""
    CONSERVATIVE = "conservative"  # Safe, small parameter changes
    GRID_SEARCH = "grid_search"    # Systematic parameter space exploration  
    RANDOM_SAMPLING = "random_sampling"    # Random variation within bounds
    ADAPTIVE = "adaptive"          # Pattern-guided intelligent variation
    LATIN_HYPERCUBE = "latin_hypercube"    # Efficient parameter space coverage


class ConstraintViolationType(Enum):
    """Types of constraint violations that can occur during variation generation"""
    INVALID_COMMAND = "invalid_command"
    PARAMETER_OUT_OF_BOUNDS = "parameter_out_of_bounds"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PERFORMANCE_VIOLATION = "performance_violation"


@dataclass
class ValidationResult:
    """Result of experiment variation validation"""
    is_valid: bool
    violations: List[ConstraintViolationType] = field(default_factory=list)
    error_details: Dict[str, str] = field(default_factory=dict)
    performance_cost: float = 0.0  # Estimated execution cost


@dataclass 
class VariationConfig:
    """Configuration for variation generation with conservative defaults"""
    max_variations_per_batch: int = 20  # Rex: Prevent parameter explosion
    max_generation_time_ms: float = 100.0  # Rex: Performance safeguard
    min_variation_distance: float = 0.1  # Rex: Ensure meaningful differences
    constraint_validation_enabled: bool = True  # Rex: Always validate
    fallback_strategy: VariationStrategy = VariationStrategy.CONSERVATIVE
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class ExperimentSpec:
    """Represents a single experiment with metadata"""
    experiment_id: str
    parameters: Dict[str, Any]
    strategy_type: str  # e.g., "navigation", "combat", "item_usage"
    dsl_script: str
    estimated_duration: float
    created_timestamp: float = field(default_factory=time.time)
    source_experiment_id: Optional[str] = None  # For tracking variations
    

class ParameterBounds:
    """Rex-designed conservative parameter bounds for Pokemon DSL commands"""
    
    # DSL Command parameter ranges (conservative to avoid breaking game)
    COMMAND_BOUNDS = {
        'movement_sequence_length': (1, 50),    # Rex: Limit to prevent infinite loops
        'pathfind_distance': (1, 100),         # Rex: Reasonable navigation bounds
        'interaction_attempts': (1, 5),        # Rex: Prevent excessive retries
        'wait_time_ms': (100, 5000),          # Rex: Sensible timing bounds
        'script_complexity': (1, 20),         # Rex: Limit nested operations
    }
    
    # Pokemon-specific constraints  
    POKEMON_CONSTRAINTS = {
        'max_hp': (1, 999),
        'level_range': (1, 100),
        'item_quantity': (0, 99),
        'move_pp': (0, 35),
    }


class VariationValidator:
    """Rex-designed comprehensive validation for experiment variations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_cache: Dict[str, ValidationResult] = {}
        
    def validate_variation(self, experiment: ExperimentSpec) -> ValidationResult:
        """
        Comprehensive validation with multiple failure detection layers
        Rex principle: Better to reject valid variations than accept invalid ones
        """
        start_time = time.perf_counter()
        
        # Check cache first for performance
        cache_key = self._generate_cache_key(experiment)
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
            
        violations = []
        error_details = {}
        
        try:
            # Layer 1: Basic structure validation
            structure_violations = self._validate_structure(experiment)
            violations.extend(structure_violations)
            
            # Layer 2: DSL command validation  
            dsl_violations = self._validate_dsl_commands(experiment.dsl_script)
            violations.extend(dsl_violations)
            
            # Layer 3: Parameter bounds validation
            bounds_violations = self._validate_parameter_bounds(experiment.parameters)
            violations.extend(bounds_violations)
            
            # Layer 4: Logical consistency validation
            logic_violations = self._validate_logical_consistency(experiment)
            violations.extend(logic_violations)
            
            # Layer 5: Performance impact validation
            performance_violations = self._validate_performance_impact(experiment)
            violations.extend(performance_violations)
            
        except Exception as e:
            self.logger.error(f"Validation failed with exception: {e}")
            violations.append(ConstraintViolationType.RESOURCE_EXHAUSTION)
            error_details['validation_exception'] = str(e)
        
        performance_cost = time.perf_counter() - start_time
        result = ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            error_details=error_details,
            performance_cost=performance_cost * 1000  # Convert to ms
        )
        
        # Cache result for future use
        self.validation_cache[cache_key] = result
        return result
    
    def _generate_cache_key(self, experiment: ExperimentSpec) -> str:
        """Generate cache key for validation results"""
        content = f"{experiment.parameters}|{experiment.dsl_script}|{experiment.strategy_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
        
    def _validate_structure(self, experiment: ExperimentSpec) -> List[ConstraintViolationType]:
        """Validate basic experiment structure"""
        violations = []
        
        if not experiment.experiment_id:
            violations.append(ConstraintViolationType.INVALID_COMMAND)
        if not isinstance(experiment.parameters, dict):
            violations.append(ConstraintViolationType.INVALID_COMMAND)
        if not experiment.dsl_script or len(experiment.dsl_script.strip()) == 0:
            violations.append(ConstraintViolationType.INVALID_COMMAND)
            
        return violations
    
    def _validate_dsl_commands(self, dsl_script: str) -> List[ConstraintViolationType]:
        """Validate DSL command syntax and semantics"""
        violations = []
        
        # Rex: Conservative validation - only allow known safe commands
        safe_commands = {'A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'PATHFIND_TO'}
        
        try:
            lines = dsl_script.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                command_parts = line.split()
                if command_parts and command_parts[0] not in safe_commands:
                    violations.append(ConstraintViolationType.INVALID_COMMAND)
                    
        except Exception:
            violations.append(ConstraintViolationType.INVALID_COMMAND)
            
        return violations
    
    def _validate_parameter_bounds(self, parameters: Dict[str, Any]) -> List[ConstraintViolationType]:
        """Validate parameters are within acceptable bounds"""
        violations = []
        
        for param_name, value in parameters.items():
            if param_name in ParameterBounds.COMMAND_BOUNDS:
                min_val, max_val = ParameterBounds.COMMAND_BOUNDS[param_name]
                if isinstance(value, (int, float)):
                    if not (min_val <= value <= max_val):
                        violations.append(ConstraintViolationType.PARAMETER_OUT_OF_BOUNDS)
                        
        return violations
    
    def _validate_logical_consistency(self, experiment: ExperimentSpec) -> List[ConstraintViolationType]:
        """Validate logical consistency of experiment configuration"""
        violations = []
        
        # Rex: Check for obvious logical inconsistencies
        if experiment.estimated_duration < 0:
            violations.append(ConstraintViolationType.LOGICAL_INCONSISTENCY)
            
        # Check for parameter conflicts
        params = experiment.parameters
        if 'pathfind_distance' in params and 'movement_sequence_length' in params:
            if params['pathfind_distance'] > params['movement_sequence_length'] * 10:
                violations.append(ConstraintViolationType.LOGICAL_INCONSISTENCY)
                
        return violations
    
    def _validate_performance_impact(self, experiment: ExperimentSpec) -> List[ConstraintViolationType]:
        """Validate experiment won't cause performance issues"""
        violations = []
        
        # Rex: Prevent resource exhaustion
        if experiment.estimated_duration > 300:  # 5 minute limit
            violations.append(ConstraintViolationType.PERFORMANCE_VIOLATION)
            
        # Check script complexity
        script_lines = len(experiment.dsl_script.split('\n'))
        if script_lines > 100:  # Conservative line limit
            violations.append(ConstraintViolationType.PERFORMANCE_VIOLATION)
            
        return violations


class VariationGenerator(ABC):
    """Abstract base class for variation generation strategies"""
    
    @abstractmethod
    def generate_variations(self, base_experiment: ExperimentSpec, 
                          count: int, config: VariationConfig) -> List[ExperimentSpec]:
        """Generate variations of the base experiment"""
        pass


class ConservativeVariationGenerator(VariationGenerator):
    """
    Rex-designed conservative variation generator
    Principle: Small, safe changes that are unlikely to break
    """
    
    def generate_variations(self, base_experiment: ExperimentSpec, 
                          count: int, config: VariationConfig) -> List[ExperimentSpec]:
        variations = []
        
        for i in range(min(count, config.max_variations_per_batch)):
            try:
                variation = self._create_safe_variation(base_experiment, i)
                variations.append(variation)
            except Exception as e:
                logging.error(f"Conservative variation generation failed: {e}")
                # Rex: Continue with other variations instead of failing completely
                continue
                
        return variations
    
    def _create_safe_variation(self, base: ExperimentSpec, variation_id: int) -> ExperimentSpec:
        """Create a safe variation with minimal changes"""
        new_params = base.parameters.copy()
        
        # Make small parameter adjustments
        for param_name, value in new_params.items():
            if isinstance(value, (int, float)) and param_name in ParameterBounds.COMMAND_BOUNDS:
                min_val, max_val = ParameterBounds.COMMAND_BOUNDS[param_name]
                # Rex: Only 10% adjustments to minimize risk
                adjustment = (max_val - min_val) * 0.1 * (random.random() - 0.5)
                new_value = max(min_val, min(max_val, value + adjustment))
                new_params[param_name] = new_value
        
        return ExperimentSpec(
            experiment_id=f"{base.experiment_id}_conservative_var_{variation_id}",
            parameters=new_params,
            strategy_type=base.strategy_type,
            dsl_script=base.dsl_script,  # Rex: Don't modify DSL in conservative mode
            estimated_duration=base.estimated_duration * 1.1,  # Slight duration increase
            source_experiment_id=base.experiment_id
        )


class ExperimentSelector:
    """
    Rex-engineered ExperimentSelector with robust variation generation
    
    Key Design Principles:
    1. Conservative bounds to prevent parameter explosion
    2. Multi-layer validation to catch invalid variations  
    3. Graceful error handling with fallback strategies
    4. Performance safeguards to prevent blocking
    5. Comprehensive logging for debugging
    """
    
    def __init__(self, config: Optional[VariationConfig] = None):
        self.config = config or VariationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Priority queue for experiment selection (min-heap, so we negate priorities)
        self._experiment_queue: List[Tuple[float, ExperimentSpec]] = []
        
        # History tracking to avoid duplicates
        self._experiment_history: Set[str] = set()
        self._completed_experiments: deque = deque(maxlen=1000)  # Rex: Bounded history
        
        # Performance tracking
        self._generation_stats: Dict[str, Union[int, float]] = defaultdict(int)
        
        # Validation system
        self.validator = VariationValidator()
        
        # Variation generators
        self.generators = {
            VariationStrategy.CONSERVATIVE: ConservativeVariationGenerator(),
            # Rex: Start with just conservative, add others incrementally
        }
        
        self.logger.info(f"ExperimentSelector initialized with {len(self.generators)} generators")
    
    def add_experiment(self, experiment_dict: Dict[str, Any], priority: float) -> None:
        """Add experiment to priority queue with validation"""
        try:
            # Convert dict to ExperimentSpec
            experiment = self._dict_to_experiment_spec(experiment_dict)
            
            # Validate experiment before adding
            validation = self.validator.validate_variation(experiment)
            if not validation.is_valid:
                self.logger.warning(f"Invalid experiment rejected: {validation.violations}")
                return
            
            # Check for duplicates
            if experiment.experiment_id in self._experiment_history:
                self.logger.debug(f"Duplicate experiment skipped: {experiment.experiment_id}")
                return
                
            # Add to priority queue (negate priority for min-heap)
            heapq.heappush(self._experiment_queue, (-priority, experiment))
            self._experiment_history.add(experiment.experiment_id)
            
            self.logger.debug(f"Added experiment {experiment.experiment_id} with priority {priority}")
            
        except Exception as e:
            self.logger.error(f"Failed to add experiment: {e}")
            # Rex: Don't crash the system, just log and continue
    
    def select_next_experiments(self, count: int) -> List[Dict[str, Any]]:
        """Select next experiments for parallel execution"""
        selected = []
        
        try:
            actual_count = min(count, len(self._experiment_queue))
            for _ in range(actual_count):
                if self._experiment_queue:
                    _, experiment = heapq.heappop(self._experiment_queue)
                    selected.append(self._experiment_spec_to_dict(experiment))
                    
        except Exception as e:
            self.logger.error(f"Experiment selection failed: {e}")
            # Rex: Return partial results instead of failing completely
        
        self.logger.info(f"Selected {len(selected)} experiments for execution")
        return selected
    
    def generate_variations(self, base_experiment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate variations using Rex-designed robust approach
        
        Key safeguards:
        1. Time-bounded generation to prevent hanging
        2. Multiple generation strategies with fallbacks
        3. Comprehensive validation of all variations
        4. Conservative parameter bounds
        5. Graceful error handling
        """
        start_time = time.perf_counter()
        variations = []
        
        try:
            # Convert input to ExperimentSpec
            base_spec = self._dict_to_experiment_spec(base_experiment)
            
            # Determine generation strategy
            strategy = self._select_generation_strategy(base_spec)
            
            # Generate variations with time limit
            if strategy in self.generators:
                generator = self.generators[strategy]
                raw_variations = generator.generate_variations(
                    base_spec, 
                    self.config.max_variations_per_batch, 
                    self.config
                )
                
                # Validate all variations
                for variation in raw_variations:
                    if time.perf_counter() - start_time > self.config.max_generation_time_ms / 1000:
                        self.logger.warning("Variation generation timeout reached")
                        break
                        
                    validation = self.validator.validate_variation(variation)
                    if validation.is_valid:
                        variations.append(self._experiment_spec_to_dict(variation))
                    else:
                        self.logger.debug(f"Invalid variation rejected: {validation.violations}")
            
        except Exception as e:
            self.logger.error(f"Variation generation failed: {e}")
            # Rex: Try fallback strategy
            variations = self._generate_fallback_variations(base_experiment)
        
        # Update statistics
        generation_time = time.perf_counter() - start_time
        self._generation_stats['total_calls'] += 1
        self._generation_stats['total_variations'] += len(variations)
        self._generation_stats['total_time_ms'] += generation_time * 1000
        
        self.logger.info(f"Generated {len(variations)} variations in {generation_time*1000:.2f}ms")
        return variations
    
    def calculate_priority(self, experiment: Dict[str, Any]) -> float:
        """Calculate experiment priority using conservative heuristics"""
        try:
            base_priority = 1.0
            
            # Rex: Simple, safe priority calculation
            if 'estimated_duration' in experiment:
                duration = experiment['estimated_duration']
                # Prefer shorter experiments (lower risk)
                base_priority *= (1.0 / max(1.0, duration / 60.0))  # Normalize by minutes
            
            if 'strategy_type' in experiment:
                # Rex: Prefer proven strategies
                safe_strategies = {'navigation', 'basic_movement'}
                if experiment['strategy_type'] in safe_strategies:
                    base_priority *= 1.5
            
            return min(10.0, max(0.1, base_priority))  # Rex: Bounded priorities
            
        except Exception as e:
            self.logger.error(f"Priority calculation failed: {e}")
            return 1.0  # Rex: Safe default
    
    def mark_completed(self, experiment_id: str, result: Dict[str, Any]) -> None:
        """Mark experiment as completed and store result"""
        try:
            completion_record = {
                'experiment_id': experiment_id,
                'result': result,
                'completed_at': time.time()
            }
            self._completed_experiments.append(completion_record)
            self.logger.debug(f"Marked experiment {experiment_id} as completed")
            
        except Exception as e:
            self.logger.error(f"Failed to mark experiment completed: {e}")
    
    def get_experiment_history(self) -> List[Dict[str, Any]]:
        """Get history of completed experiments"""
        return list(self._completed_experiments)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get variation generation performance statistics"""
        stats = dict(self._generation_stats)
        total_calls = stats.get('total_calls', 0)
        if total_calls > 0:
            stats['average_variations_per_call'] = stats.get('total_variations', 0) / total_calls
            stats['average_time_per_call_ms'] = stats.get('total_time_ms', 0) / total_calls
        return stats
    
    # Private helper methods
    
    def _dict_to_experiment_spec(self, experiment_dict: Dict[str, Any]) -> ExperimentSpec:
        """Convert dictionary to ExperimentSpec with safe defaults"""
        return ExperimentSpec(
            experiment_id=experiment_dict.get('experiment_id', f"exp_{int(time.time())}"),
            parameters=experiment_dict.get('parameters', {}),
            strategy_type=experiment_dict.get('strategy_type', 'unknown'),
            dsl_script=experiment_dict.get('dsl_script', '# Empty script'),
            estimated_duration=experiment_dict.get('estimated_duration', 60.0)
        )
    
    def _experiment_spec_to_dict(self, experiment: ExperimentSpec) -> Dict[str, Any]:
        """Convert ExperimentSpec to dictionary"""
        return {
            'experiment_id': experiment.experiment_id,
            'parameters': experiment.parameters,
            'strategy_type': experiment.strategy_type,
            'dsl_script': experiment.dsl_script,
            'estimated_duration': experiment.estimated_duration,
            'created_timestamp': experiment.created_timestamp,
            'source_experiment_id': experiment.source_experiment_id
        }
    
    def _select_generation_strategy(self, experiment: ExperimentSpec) -> VariationStrategy:
        """Select appropriate generation strategy based on experiment characteristics"""
        # Rex: Start conservative, expand later based on success
        return VariationStrategy.CONSERVATIVE
    
    def _generate_fallback_variations(self, base_experiment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate minimal variations as fallback when main generation fails"""
        try:
            fallback_variation = base_experiment.copy()
            fallback_variation['experiment_id'] = f"{base_experiment.get('experiment_id', 'unknown')}_fallback"
            fallback_variation['estimated_duration'] = base_experiment.get('estimated_duration', 60) * 0.9
            return [fallback_variation]
        except Exception:
            self.logger.error("Even fallback variation generation failed")
            return []


# Factory function for easy instantiation
def create_experiment_selector(config: Optional[VariationConfig] = None) -> ExperimentSelector:
    """Factory function to create ExperimentSelector with default configuration"""
    return ExperimentSelector(config or VariationConfig())


if __name__ == "__main__":
    # Rex: Simple verification that the module loads correctly
    logging.basicConfig(level=logging.INFO)
    selector = create_experiment_selector()
    print(f"ExperimentSelector created successfully with {len(selector.generators)} generators")