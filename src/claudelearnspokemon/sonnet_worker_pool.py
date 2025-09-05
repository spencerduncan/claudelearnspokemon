"""
SonnetWorkerPool - Worker Pool Initialization and Management

This module provides a high-level abstraction layer over ClaudeCodeManager's
tactical processes for managing Sonnet workers in parallel script development.
It implements worker pool initialization, task assignment, and lifecycle management
following Clean Architecture principles.

Key Responsibilities:
- Initialize configurable number of Sonnet workers
- Assign unique worker IDs and track worker status
- Verify worker health and connectivity during initialization
- Support worker pool scaling and dynamic reconfiguration
- Provide load balancing for task assignment

Performance Targets:
- Worker pool initialization: <500ms
- Worker health checks: <10ms per worker
- Task assignment: <50ms
- Worker status queries: <10ms
"""

import logging
import queue
import random
import threading
import time
import uuid
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Forward references to avoid circular imports
    pass

from .claude_code_manager import ClaudeCodeManager
from .mcp_data_patterns import PokemonStrategy, QueryBuilder
from .script_compiler import ScriptCompiler
from .script_quality_assessor import ScriptQualityAssessor, PatternRefiner
from .config import CONFIG

logger = logging.getLogger(__name__)


class SemanticPatternEngine:
    """Semantic pattern understanding layer with context-aware matching."""
    
    def __init__(self):
        self.pattern_embeddings = {}  # pattern_id -> embedding vector
        self.context_embeddings = {}  # context hash -> embedding vector
        self.pattern_success_contexts = {}  # pattern_id -> list of successful contexts
        self.similarity_cache = {}  # (pattern_id, context_hash) -> similarity score
        
    def create_pattern_embedding(self, pattern: Dict[str, Any]) -> List[float]:
        """Create a semantic embedding for a pattern based on its characteristics."""
        # Simple feature-based embedding approach
        embedding = []
        
        # Pattern complexity features
        pattern_sequence = pattern.get('pattern_sequence', [])
        embedding.append(len(pattern_sequence) / 20.0)  # Normalized length
        
        # Command type diversity
        movement_commands = sum(1 for cmd in pattern_sequence if cmd in ['UP', 'DOWN', 'LEFT', 'RIGHT'])
        action_commands = sum(1 for cmd in pattern_sequence if cmd in ['A', 'B', 'START', 'SELECT'])
        embedding.append(movement_commands / max(len(pattern_sequence), 1))
        embedding.append(action_commands / max(len(pattern_sequence), 1))
        
        # Success rate features
        success_rate = pattern.get('success_rate', 0.0)
        embedding.append(success_rate)
        
        # Context-based features
        context = pattern.get('context', {})
        if 'location' in context:
            # Simple location encoding (could be enhanced with learned embeddings)
            location_hash = hash(str(context['location'])) % 1000 / 1000.0
            embedding.append(location_hash)
        else:
            embedding.append(0.0)
            
        if 'objective' in context:
            objective_hash = hash(str(context['objective'])) % 1000 / 1000.0
            embedding.append(objective_hash)
        else:
            embedding.append(0.0)
            
        # Resource requirement features
        resources = pattern.get('resource_requirements', {})
        embedding.append(resources.get('time_limit', 0.0) / 3600.0)  # Normalized hours
        embedding.append(resources.get('difficulty', 0.0) / 10.0)  # Normalized difficulty
        
        return embedding
    
    def create_context_embedding(self, context: Dict[str, Any]) -> List[float]:
        """Create an embedding for a task context."""
        embedding = []
        
        # Location context
        if 'location' in context:
            location_hash = hash(str(context['location'])) % 1000 / 1000.0
            embedding.append(location_hash)
        else:
            embedding.append(0.0)
            
        # Objective context
        if 'objective' in context:
            objective_hash = hash(str(context['objective'])) % 1000 / 1000.0
            embedding.append(objective_hash)
        else:
            embedding.append(0.0)
            
        # Task complexity indicators
        task_text = str(context.get('objective', '')) + str(context.get('description', ''))
        embedding.append(len(task_text.split()) / 50.0)  # Normalized word count
        
        # Time constraints
        embedding.append(context.get('time_limit', 0.0) / 3600.0)
        embedding.append(context.get('difficulty', 5.0) / 10.0)  # Default medium difficulty
        
        # Pad to match pattern embedding size
        while len(embedding) < 8:
            embedding.append(0.0)
            
        return embedding[:8]  # Ensure consistent size
    
    def compute_similarity(self, pattern_embedding: List[float], context_embedding: List[float]) -> float:
        """Compute cosine similarity between pattern and context embeddings."""
        if len(pattern_embedding) != len(context_embedding):
            # Pad shorter vector with zeros
            max_len = max(len(pattern_embedding), len(context_embedding))
            pattern_embedding = pattern_embedding + [0.0] * (max_len - len(pattern_embedding))
            context_embedding = context_embedding + [0.0] * (max_len - len(context_embedding))
        
        # Compute cosine similarity
        dot_product = sum(p * c for p, c in zip(pattern_embedding, context_embedding))
        pattern_norm = sum(p * p for p in pattern_embedding) ** 0.5
        context_norm = sum(c * c for c in context_embedding) ** 0.5
        
        if pattern_norm == 0.0 or context_norm == 0.0:
            return 0.0
            
        return dot_product / (pattern_norm * context_norm)
    
    def get_contextual_pattern_recommendations(self, patterns: List[Dict[str, Any]], 
                                             context: Dict[str, Any], 
                                             top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top-k patterns most relevant to the given context."""
        context_embedding = self.create_context_embedding(context)
        context_hash = hash(str(sorted(context.items())))
        
        pattern_scores = []
        
        for pattern in patterns:
            pattern_id = pattern.get('pattern_id', pattern.get('strategy_id', 'unknown'))
            
            # Check cache first
            cache_key = (pattern_id, context_hash)
            if cache_key in self.similarity_cache:
                similarity = self.similarity_cache[cache_key]
            else:
                # Compute similarity
                if pattern_id not in self.pattern_embeddings:
                    self.pattern_embeddings[pattern_id] = self.create_pattern_embedding(pattern)
                
                pattern_embedding = self.pattern_embeddings[pattern_id]
                similarity = self.compute_similarity(pattern_embedding, context_embedding)
                
                # Cache result
                self.similarity_cache[cache_key] = similarity
            
            # Boost score based on historical success in similar contexts
            boost = self._get_contextual_success_boost(pattern_id, context)
            final_score = similarity * (1.0 + boost)
            
            pattern_scores.append((pattern, final_score))
        
        # Sort by score and return top-k
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        return [pattern for pattern, score in pattern_scores[:top_k]]
    
    def update_pattern_success(self, pattern_id: str, context: Dict[str, Any], success: bool) -> None:
        """Update pattern success history for better contextual recommendations."""
        if pattern_id not in self.pattern_success_contexts:
            self.pattern_success_contexts[pattern_id] = []
            
        context_record = {
            'context': context,
            'success': success,
            'timestamp': time.time()
        }
        
        self.pattern_success_contexts[pattern_id].append(context_record)
        
        # Keep only recent history (last 50 records)
        if len(self.pattern_success_contexts[pattern_id]) > 50:
            self.pattern_success_contexts[pattern_id] = self.pattern_success_contexts[pattern_id][-25:]
    
    def _get_contextual_success_boost(self, pattern_id: str, context: Dict[str, Any]) -> float:
        """Get boost factor based on historical success in similar contexts."""
        if pattern_id not in self.pattern_success_contexts:
            return 0.0
            
        history = self.pattern_success_contexts[pattern_id]
        if not history:
            return 0.0
            
        context_embedding = self.create_context_embedding(context)
        
        # Calculate weighted success rate based on context similarity
        weighted_success = 0.0
        total_weight = 0.0
        
        for record in history:
            historical_context_embedding = self.create_context_embedding(record['context'])
            similarity = self.compute_similarity(context_embedding, historical_context_embedding)
            
            if similarity > 0.5:  # Only consider reasonably similar contexts
                weight = similarity ** 2  # Square to emphasize high similarity
                weighted_success += weight * (1.0 if record['success'] else 0.0)
                total_weight += weight
        
        if total_weight == 0.0:
            return 0.0
            
        success_rate = weighted_success / total_weight
        return success_rate * 0.3  # Max 30% boost


class ExperimentSelector:
    """ML-guided strategy selection for script development experiments.
    
    Coordinates between ReinforcementLearningEngine, MultiObjectiveOptimizer, and
    SemanticPatternEngine to intelligently select optimal development strategies
    based on task context and historical performance.
    
    Performance Target: <50ms strategy selection
    """
    
    def __init__(self):
        self.strategy_performance_history: Dict[str, List[float]] = {}
        self.context_strategy_mapping: Dict[str, List[Tuple[str, float]]] = {}
        self.selection_cache: Dict[str, Tuple[str, float]] = {}  # context_hash -> (strategy, confidence)
        
        # Available development strategies
        self.development_strategies = {
            'genetic_evolution': {
                'description': 'Genetic algorithm with adaptive operators',
                'best_contexts': ['optimization', 'exploration', 'long_sequences'],
                'performance_target': 'balanced',
                'computational_cost': 0.7
            },
            'reinforcement_guided': {
                'description': 'RL-guided script development',
                'best_contexts': ['learning', 'adaptation', 'complex_patterns'],
                'performance_target': 'innovation', 
                'computational_cost': 0.8
            },
            'pattern_synthesis': {
                'description': 'Cross-worker pattern combination',
                'best_contexts': ['synthesis', 'collaboration', 'known_patterns'],
                'performance_target': 'reliability',
                'computational_cost': 0.5
            },
            'semantic_search': {
                'description': 'Context-aware pattern matching',
                'best_contexts': ['similar_contexts', 'quick_solutions', 'proven_patterns'],
                'performance_target': 'speed',
                'computational_cost': 0.3
            },
            'multi_objective': {
                'description': 'Balanced multi-objective optimization',
                'best_contexts': ['trade_offs', 'multiple_goals', 'performance_critical'],
                'performance_target': 'balanced',
                'computational_cost': 0.6
            }
        }
        
        # Performance tracking
        self.selection_times: List[float] = []
        self.strategy_success_rates: Dict[str, List[bool]] = {
            strategy: [] for strategy in self.development_strategies.keys()
        }
        
    def select_optimal_strategy(self, task_context: Dict[str, Any], 
                              rl_engine: 'ReinforcementLearningEngine',
                              semantic_engine: 'SemanticPatternEngine',
                              performance_constraints: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Select optimal development strategy based on ML guidance and context analysis.
        
        Args:
            task_context: Task context including location, objectives, constraints
            rl_engine: Reinforcement learning engine for strategy learning
            semantic_engine: Semantic pattern engine for context matching
            performance_constraints: Optional performance requirements
            
        Returns:
            Dictionary containing selected strategy, confidence, and metadata
        """
        selection_start = time.time()
        
        try:
            # Create context hash for caching
            context_hash = self._hash_context(task_context)
            
            # Check cache first for performance
            if context_hash in self.selection_cache:
                cached_result = self.selection_cache[context_hash]
                self._record_selection_time(time.time() - selection_start)
                return {
                    'strategy': cached_result[0],
                    'confidence': cached_result[1],
                    'source': 'cache',
                    'selection_time_ms': (time.time() - selection_start) * 1000
                }
            
            # Get ML-guided recommendations
            strategy_scores = self._compute_ml_strategy_scores(
                task_context, rl_engine, semantic_engine
            )
            
            # Apply performance constraints if specified
            if performance_constraints:
                strategy_scores = self._apply_performance_constraints(
                    strategy_scores, performance_constraints
                )
            
            # Select best strategy with confidence scoring
            selected_strategy, confidence = self._select_best_strategy(
                strategy_scores, task_context
            )
            
            # Cache result for future use
            self.selection_cache[context_hash] = (selected_strategy, confidence)
            
            selection_time = time.time() - selection_start
            self._record_selection_time(selection_time)
            
            return {
                'strategy': selected_strategy,
                'confidence': confidence,
                'strategy_scores': strategy_scores,
                'source': 'ml_guided',
                'selection_time_ms': selection_time * 1000,
                'context_hash': context_hash
            }
            
        except Exception as e:
            logger.warning(f"Strategy selection failed: {e}, falling back to default")
            return self._get_fallback_strategy(task_context)
    
    def update_strategy_performance(self, strategy: str, task_context: Dict[str, Any], 
                                  success: bool, performance_metrics: Dict[str, float]) -> None:
        """Update strategy performance history for learning."""
        # Track success rate
        if strategy in self.strategy_success_rates:
            self.strategy_success_rates[strategy].append(success)
            # Keep only recent history
            if len(self.strategy_success_rates[strategy]) > 100:
                self.strategy_success_rates[strategy] = self.strategy_success_rates[strategy][-50:]
        
        # Track detailed performance
        context_hash = self._hash_context(task_context)
        if context_hash not in self.context_strategy_mapping:
            self.context_strategy_mapping[context_hash] = []
        
        performance_score = self._compute_performance_score(performance_metrics, success)
        self.context_strategy_mapping[context_hash].append((strategy, performance_score))
        
        # Invalidate cache for this context
        if context_hash in self.selection_cache:
            del self.selection_cache[context_hash]
            
        logger.debug(f"Updated {strategy} performance: success={success}, score={performance_score:.3f}")
    
    def get_strategy_recommendations(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get top-k strategy recommendations with performance data."""
        strategy_rankings = []
        
        for strategy, info in self.development_strategies.items():
            success_history = self.strategy_success_rates.get(strategy, [])
            success_rate = sum(success_history) / max(len(success_history), 1)
            
            strategy_rankings.append({
                'strategy': strategy,
                'description': info['description'],
                'success_rate': success_rate,
                'usage_count': len(success_history),
                'computational_cost': info['computational_cost'],
                'performance_target': info['performance_target']
            })
        
        # Sort by success rate and usage
        strategy_rankings.sort(
            key=lambda x: (x['success_rate'], x['usage_count']), 
            reverse=True
        )
        
        return strategy_rankings[:top_k]
    
    def _compute_ml_strategy_scores(self, task_context: Dict[str, Any],
                                  rl_engine: 'ReinforcementLearningEngine',
                                  semantic_engine: 'SemanticPatternEngine') -> Dict[str, float]:
        """Compute ML-guided scores for each strategy."""
        scores = {}
        
        # Get RL recommendations
        try:
            state = self._encode_context_for_rl(task_context)
            rl_action = rl_engine.select_action(state)
            rl_strategy_preference = self._decode_rl_action(rl_action)
        except Exception as e:
            logger.debug(f"RL strategy selection failed: {e}")
            rl_strategy_preference = {}
        
        # Compute context-based scores
        for strategy, info in self.development_strategies.items():
            # Base score from strategy characteristics
            base_score = 0.5
            
            # RL preference boost
            if strategy in rl_strategy_preference:
                base_score += rl_strategy_preference[strategy] * 0.3
            
            # Context matching score
            context_match_score = self._compute_context_match(task_context, info['best_contexts'])
            base_score += context_match_score * 0.2
            
            # Historical performance in similar contexts
            historical_score = self._get_historical_performance(strategy, task_context)
            base_score += historical_score * 0.3
            
            scores[strategy] = min(1.0, max(0.0, base_score))
        
        return scores
    
    def _apply_performance_constraints(self, scores: Dict[str, float], 
                                     constraints: Dict[str, float]) -> Dict[str, float]:
        """Apply performance constraints to strategy scores."""
        constrained_scores = scores.copy()
        
        for strategy, score in scores.items():
            strategy_info = self.development_strategies[strategy]
            
            # Check computational cost constraint
            if 'max_computational_cost' in constraints:
                max_cost = constraints['max_computational_cost']
                if strategy_info['computational_cost'] > max_cost:
                    penalty = (strategy_info['computational_cost'] - max_cost) * 0.5
                    constrained_scores[strategy] = max(0.0, score - penalty)
            
            # Check time constraint
            if 'max_time_ms' in constraints:
                max_time = constraints['max_time_ms']
                if strategy in ['reinforcement_guided', 'genetic_evolution'] and max_time < 100:
                    constrained_scores[strategy] *= 0.5  # Penalize slow strategies for fast requirements
        
        return constrained_scores
    
    def _select_best_strategy(self, scores: Dict[str, float], 
                            task_context: Dict[str, Any]) -> Tuple[str, float]:
        """Select best strategy with confidence scoring."""
        if not scores:
            return 'semantic_search', 0.5  # Safe fallback
        
        # Find best strategy
        best_strategy = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_strategy]
        
        # Compute confidence based on score distribution
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            confidence = (sorted_scores[0] - sorted_scores[1]) + 0.5  # Margin + base confidence
        else:
            confidence = best_score
        
        confidence = min(1.0, max(0.0, confidence))
        
        return best_strategy, confidence
    
    def _get_fallback_strategy(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get safe fallback strategy when ML guidance fails."""
        # Simple heuristic fallback
        if 'speed' in str(task_context).lower():
            fallback = 'semantic_search'
        elif 'complex' in str(task_context).lower():
            fallback = 'genetic_evolution'
        else:
            fallback = 'multi_objective'
        
        return {
            'strategy': fallback,
            'confidence': 0.4,
            'source': 'fallback',
            'selection_time_ms': 1.0
        }
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create hash for context caching."""
        # Simple context hashing for caching
        context_str = str(sorted(context.items()))
        return str(hash(context_str))
    
    def _encode_context_for_rl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Encode context for RL engine consumption."""
        return {
            'location_hash': hash(str(context.get('location', 'unknown'))) % 1000,
            'objective_hash': hash(str(context.get('objective', 'unknown'))) % 1000,
            'complexity': len(str(context)) / 100.0,  # Normalized complexity
        }
    
    def _decode_rl_action(self, action: Dict[str, Any]) -> Dict[str, float]:
        """Decode RL action into strategy preferences."""
        # Simple mapping from RL action to strategy preferences
        preferences = {}
        
        if 'exploration_weight' in action:
            weight = action['exploration_weight']
            preferences['genetic_evolution'] = weight
            preferences['reinforcement_guided'] = weight * 0.8
        
        if 'exploitation_weight' in action:
            weight = action['exploitation_weight']
            preferences['semantic_search'] = weight
            preferences['pattern_synthesis'] = weight * 0.9
        
        return preferences
    
    def _compute_context_match(self, context: Dict[str, Any], best_contexts: List[str]) -> float:
        """Compute how well context matches strategy's best contexts."""
        context_str = str(context).lower()
        matches = sum(1 for bc in best_contexts if bc in context_str)
        return matches / max(len(best_contexts), 1)
    
    def _get_historical_performance(self, strategy: str, context: Dict[str, Any]) -> float:
        """Get historical performance for strategy in similar contexts."""
        context_hash = self._hash_context(context)
        
        if context_hash in self.context_strategy_mapping:
            history = self.context_strategy_mapping[context_hash]
            strategy_history = [score for strat, score in history if strat == strategy]
            if strategy_history:
                return sum(strategy_history) / len(strategy_history)
        
        # Fallback to global strategy performance
        if strategy in self.strategy_success_rates:
            success_history = self.strategy_success_rates[strategy]
            if success_history:
                return sum(success_history) / len(success_history)
        
        return 0.5  # Neutral score
    
    def _compute_performance_score(self, metrics: Dict[str, float], success: bool) -> float:
        """Compute overall performance score from metrics."""
        base_score = 1.0 if success else 0.0
        
        # Adjust based on performance metrics
        if 'execution_time_ms' in metrics:
            time_score = max(0.0, 1.0 - metrics['execution_time_ms'] / 1000.0)  # Normalize to 1s
            base_score = base_score * 0.7 + time_score * 0.3
        
        if 'quality_score' in metrics:
            quality_score = min(1.0, metrics['quality_score'])
            base_score = base_score * 0.8 + quality_score * 0.2
        
        return min(1.0, max(0.0, base_score))
    
    def _record_selection_time(self, selection_time: float) -> None:
        """Record selection time for performance monitoring."""
        self.selection_times.append(selection_time)
        
        # Keep only recent times
        if len(self.selection_times) > 100:
            self.selection_times = self.selection_times[-50:]
        
        # Log warning if selection is slow
        if selection_time > 0.05:  # 50ms target
            logger.warning(f"Strategy selection took {selection_time*1000:.1f}ms, exceeds 50ms target")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        if not self.selection_times:
            return {'status': 'no_data'}
        
        avg_time_ms = (sum(self.selection_times) / len(self.selection_times)) * 1000
        max_time_ms = max(self.selection_times) * 1000
        
        return {
            'average_selection_time_ms': avg_time_ms,
            'max_selection_time_ms': max_time_ms,
            'total_selections': len(self.selection_times),
            'performance_target_met': avg_time_ms < 50.0,
            'cache_size': len(self.selection_cache),
            'strategy_usage': {
                strategy: len(history) 
                for strategy, history in self.strategy_success_rates.items()
            }
        }


class MultiObjectiveOptimizer:
    """Multi-objective optimization for balancing speed, reliability, and innovation."""
    
    def __init__(self, speed_weight: float = 0.4, reliability_weight: float = 0.4, innovation_weight: float = 0.2):
        self.speed_weight = speed_weight
        self.reliability_weight = reliability_weight
        self.innovation_weight = innovation_weight
        self.pareto_front = []  # Store Pareto-optimal solutions
        
    def compute_multi_objective_fitness(self, variant: 'ScriptVariant', 
                                       base_quality_score: float,
                                       execution_time_estimate: float = 0.0,
                                       pattern_novelty_score: float = 0.0) -> Dict[str, float]:
        """Compute multi-objective fitness scores."""
        # Objective 1: Speed (minimize execution time, maximize efficiency)
        script_length = len(variant.script_text.split('\n'))
        speed_score = max(0.0, 1.0 - (script_length / 100.0))  # Shorter scripts generally faster
        
        # Add execution time estimate if available
        if execution_time_estimate > 0:
            time_penalty = min(1.0, execution_time_estimate / 300.0)  # 300 frame penalty threshold
            speed_score = speed_score * (1.0 - time_penalty)
            
        # Objective 2: Reliability (based on quality score and validation)
        reliability_score = base_quality_score
        
        # Boost reliability for patterns with successful history
        if hasattr(variant, 'mutation_history'):
            successful_mutations = sum(1 for m in variant.mutation_history if 'successful' in m)
            total_mutations = len(variant.mutation_history)
            if total_mutations > 0:
                mutation_success_rate = successful_mutations / total_mutations
                reliability_score = reliability_score * (0.8 + 0.2 * mutation_success_rate)
        
        # Objective 3: Innovation (pattern diversity and novelty)
        innovation_score = self._compute_innovation_score(variant, pattern_novelty_score)
        
        # Combine objectives using weighted sum
        combined_fitness = (
            self.speed_weight * speed_score +
            self.reliability_weight * reliability_score +
            self.innovation_weight * innovation_score
        )
        
        return {
            'combined_fitness': combined_fitness,
            'speed_score': speed_score,
            'reliability_score': reliability_score,
            'innovation_score': innovation_score
        }
    
    def _compute_innovation_score(self, variant: 'ScriptVariant', pattern_novelty_score: float) -> float:
        """Compute innovation score based on pattern diversity and novelty."""
        script_lines = [line.strip() for line in variant.script_text.split('\n') if line.strip()]
        
        # Command diversity score
        unique_commands = set()
        for line in script_lines:
            if line.upper() in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT', 'REPEAT', 'END', 'OBSERVE']:
                unique_commands.add(line.upper())
        
        diversity_score = len(unique_commands) / 11.0  # 11 possible unique commands
        
        # Pattern novelty (if provided)
        novelty_score = pattern_novelty_score
        
        # Sequence innovation (reward unusual command combinations)
        innovation_bonus = 0.0
        for i in range(len(script_lines) - 1):
            current = script_lines[i].upper()
            next_cmd = script_lines[i + 1].upper()
            
            # Reward creative sequences (these are heuristics, could be learned)
            if current in ['A', 'B'] and next_cmd in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                innovation_bonus += 0.1  # Action followed by movement
            elif current == 'START' and next_cmd in ['UP', 'DOWN']:
                innovation_bonus += 0.05  # Menu navigation patterns
        
        innovation_bonus = min(innovation_bonus, 0.3)  # Cap bonus at 30%
        
        return (diversity_score * 0.5 + novelty_score * 0.3 + innovation_bonus * 0.2)
    
    def update_pareto_front(self, variant: 'ScriptVariant', fitness_scores: Dict[str, float]) -> bool:
        """Update Pareto front with new variant if it's non-dominated."""
        objectives = [fitness_scores['speed_score'], fitness_scores['reliability_score'], fitness_scores['innovation_score']]
        
        # Check if this variant dominates any existing solutions
        dominated_indices = []
        is_dominated = False
        
        for i, (existing_variant, existing_objectives) in enumerate(self.pareto_front):
            dominance = self._check_dominance(objectives, existing_objectives)
            
            if dominance == 1:  # New variant dominates existing
                dominated_indices.append(i)
            elif dominance == -1:  # New variant is dominated
                is_dominated = True
                break
        
        if not is_dominated:
            # Remove dominated solutions
            for i in sorted(dominated_indices, reverse=True):
                del self.pareto_front[i]
            
            # Add new solution
            self.pareto_front.append((variant, objectives))
            return True
            
        return False
    
    def _check_dominance(self, obj1: List[float], obj2: List[float]) -> int:
        """Check dominance relationship between two objective vectors.
        
        Returns:
            1 if obj1 dominates obj2
            -1 if obj2 dominates obj1  
            0 if neither dominates
        """
        better_count = 0
        worse_count = 0
        
        for v1, v2 in zip(obj1, obj2):
            if v1 > v2:
                better_count += 1
            elif v1 < v2:
                worse_count += 1
        
        if better_count > 0 and worse_count == 0:
            return 1  # obj1 dominates obj2
        elif worse_count > 0 and better_count == 0:
            return -1  # obj2 dominates obj1
        else:
            return 0  # Neither dominates
    
    def get_pareto_optimal_solutions(self) -> List[Tuple['ScriptVariant', List[float]]]:
        """Get current Pareto-optimal solutions."""
        return self.pareto_front.copy()
    
    def select_solution_from_pareto_front(self, preference_weights: Optional[List[float]] = None) -> Optional['ScriptVariant']:
        """Select best solution from Pareto front based on preference weights."""
        if not self.pareto_front:
            return None
            
        if preference_weights is None:
            preference_weights = [self.speed_weight, self.reliability_weight, self.innovation_weight]
            
        best_variant = None
        best_score = -1.0
        
        for variant, objectives in self.pareto_front:
            weighted_score = sum(w * obj for w, obj in zip(preference_weights, objectives))
            if weighted_score > best_score:
                best_score = weighted_score
                best_variant = variant
                
        return best_variant


class ScriptVariant:
    """Represents a single script variant in the genetic algorithm population."""
    
    def __init__(self, script_text: str, fitness: float = 0.0, generation: int = 0):
        self.script_text = script_text
        self.fitness = fitness
        self.generation = generation
        self.parent_ids: List[str] = []
        self.variant_id = f"variant_{uuid.uuid4().hex[:8]}"
        self.mutation_history: List[str] = []
        
    def copy(self) -> 'ScriptVariant':
        """Create a deep copy of this variant."""
        new_variant = ScriptVariant(self.script_text, self.fitness, self.generation + 1)
        new_variant.parent_ids = [self.variant_id]
        new_variant.mutation_history = self.mutation_history.copy()
        return new_variant


class AdvancedGeneticOperators:
    """Enhanced DSL-aware genetic operators with evolutionary strategies."""
    
    # Enhanced command categorization
    MOVEMENT_COMMANDS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    ACTION_COMMANDS = ['A', 'B', 'START', 'SELECT']  
    CONTROL_COMMANDS = ['REPEAT', 'END', 'OBSERVE']
    ALL_COMMANDS = MOVEMENT_COMMANDS + ACTION_COMMANDS + CONTROL_COMMANDS
    
    # Command interaction patterns for intelligent mutations
    COMMAND_SYNERGIES = {
        'A': ['B', 'START'],  # Actions that work well with A
        'B': ['A', 'SELECT'],
        'UP': ['A', 'B'],  # Movement + action combinations
        'DOWN': ['A', 'B'],
        'LEFT': ['A', 'B'],
        'RIGHT': ['A', 'B'],
        'START': ['UP', 'DOWN', 'SELECT'],  # Menu navigation
        'SELECT': ['START', 'A', 'B']
    }
    
    # Timing optimization patterns
    TIMING_PATTERNS = {
        'fast_execution': [1, 2, 3],
        'standard_execution': [3, 4, 5, 6],
        'cautious_execution': [8, 10, 12, 15],
        'synchronization': [30, 60]  # Frame-perfect timing
    }
    
    def __init__(self):
        self.mutation_success_history: Dict[str, List[float]] = {}
        self.crossover_success_history: Dict[str, List[float]] = {}
        self.adaptive_mutation_rates: Dict[str, float] = {
            'insert_command': 0.3,
            'delete_command': 0.2,
            'modify_timing': 0.25,
            'swap_sequence': 0.15,
            'context_aware_mutation': 0.1
        }
    
    def context_aware_insert_mutation(self, script: str, context: Dict[str, Any] = None) -> str:
        """Insert commands based on context and command synergies."""
        lines = [line.strip() for line in script.split('\n') if line.strip()]
        if not lines:
            return random.choice(self.ALL_COMMANDS)
        
        # Find strategic insertion point based on context
        insert_pos = self._find_strategic_insertion_point(lines, context)
        
        # Choose command based on context and surrounding commands
        if context and 'objective' in context and 'menu' in context['objective'].lower():
            command = random.choice(self.ACTION_COMMANDS + ['START', 'SELECT'])
        elif context and 'movement' in str(context).lower():
            command = random.choice(self.MOVEMENT_COMMANDS)
        else:
            # Use synergy-based selection
            command = self._select_synergistic_command(lines, insert_pos)
        
        # Add context-appropriate timing
        timing = self._select_contextual_timing(command, context)
        if timing:
            lines.insert(insert_pos, command)
            lines.insert(insert_pos + 1, str(timing))
        else:
            lines.insert(insert_pos, command)
            
        return '\n'.join(lines)
    
    def adaptive_delete_mutation(self, script: str, quality_feedback: float = 0.5) -> str:
        """Remove commands based on quality feedback and redundancy analysis."""
        lines = [line.strip() for line in script.split('\n') if line.strip()]
        if len(lines) <= 2:
            return script
            
        # Analyze command patterns to find redundancies
        redundancy_scores = self._compute_redundancy_scores(lines)
        
        # Select deletion target based on redundancy and quality feedback
        if quality_feedback < 0.5:  # Poor performance, remove more aggressively
            delete_candidates = [i for i, score in enumerate(redundancy_scores) if score > 0.3]
        else:  # Good performance, be more conservative
            delete_candidates = [i for i, score in enumerate(redundancy_scores) if score > 0.7]
            
        if delete_candidates:
            delete_pos = random.choice(delete_candidates)
        else:
            delete_pos = random.randint(0, len(lines) - 1)
            
        lines.pop(delete_pos)
        return '\n'.join(lines)
    
    def intelligent_timing_mutation(self, script: str, performance_target: str = 'balanced') -> str:
        """Adjust timing based on performance targets and execution patterns."""
        lines = script.split('\n')
        modified_lines = []
        
        # Select timing pattern based on target
        if performance_target == 'speed':
            timing_pool = self.TIMING_PATTERNS['fast_execution']
        elif performance_target == 'accuracy':
            timing_pool = self.TIMING_PATTERNS['cautious_execution']
        elif performance_target == 'sync':
            timing_pool = self.TIMING_PATTERNS['synchronization']
        else:
            timing_pool = self.TIMING_PATTERNS['standard_execution']
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.isdigit():
                current_delay = int(line)
                
                # Apply intelligent timing adjustment
                if performance_target == 'speed' and current_delay > 5:
                    new_delay = random.choice(timing_pool)
                elif performance_target == 'sync' and i > 0 and not lines[i-1].strip().isdigit():
                    # Add synchronization timing after commands
                    new_delay = random.choice(timing_pool)
                else:
                    # Standard optimization
                    new_delay = random.choice(timing_pool + [current_delay])
                
                modified_lines.append(str(new_delay))
            else:
                modified_lines.append(line)
                
        return '\n'.join(modified_lines)
    
    def multi_point_crossover(self, parent1: str, parent2: str, crossover_points: int = 2) -> str:
        """Advanced multi-point crossover with adaptive point selection."""
        lines1 = [line.strip() for line in parent1.split('\n') if line.strip()]
        lines2 = [line.strip() for line in parent2.split('\n') if line.strip()]
        
        if not lines1 or not lines2:
            return parent1 if lines1 else parent2
            
        min_length = min(len(lines1), len(lines2))
        if min_length < crossover_points + 1:
            # Fallback to single-point crossover
            return self.single_point_crossover(parent1, parent2)
            
        # Select adaptive crossover points based on command boundaries
        crossover_positions = self._select_crossover_points(lines1, lines2, crossover_points)
        
        offspring = []
        current_parent = 1
        last_pos = 0
        
        for pos in sorted(crossover_positions) + [min_length]:
            if current_parent == 1:
                offspring.extend(lines1[last_pos:pos])
                current_parent = 2
            else:
                offspring.extend(lines2[last_pos:pos])
                current_parent = 1
            last_pos = pos
            
        return '\n'.join(offspring)
    
    def semantic_crossover(self, parent1: str, parent2: str, context: Dict[str, Any] = None) -> str:
        """Crossover that preserves semantic blocks and functional units."""
        # Analyze parents for semantic blocks
        blocks1 = self._extract_semantic_blocks(parent1)
        blocks2 = self._extract_semantic_blocks(parent2)
        
        if not blocks1 or not blocks2:
            return self.single_point_crossover(parent1, parent2)
        
        # Select blocks that complement each other
        offspring_blocks = []
        
        # Start with setup blocks
        setup_blocks1 = [b for b in blocks1 if b['type'] == 'setup']
        setup_blocks2 = [b for b in blocks2 if b['type'] == 'setup']
        if setup_blocks1 and setup_blocks2:
            offspring_blocks.append(random.choice(setup_blocks1 + setup_blocks2))
        
        # Add execution blocks
        exec_blocks1 = [b for b in blocks1 if b['type'] == 'execution']
        exec_blocks2 = [b for b in blocks2 if b['type'] == 'execution']
        offspring_blocks.extend(random.sample(exec_blocks1 + exec_blocks2, 
                                           min(3, len(exec_blocks1) + len(exec_blocks2))))
        
        # Add validation blocks
        val_blocks1 = [b for b in blocks1 if b['type'] == 'validation']
        val_blocks2 = [b for b in blocks2 if b['type'] == 'validation']
        if val_blocks1 or val_blocks2:
            all_val = val_blocks1 + val_blocks2
            offspring_blocks.append(random.choice(all_val))
        
        # Combine blocks into script
        offspring_lines = []
        for block in offspring_blocks:
            offspring_lines.extend(block['commands'])
            
        return '\n'.join(offspring_lines) if offspring_lines else self.single_point_crossover(parent1, parent2)
    
    def single_point_crossover(self, parent1: str, parent2: str) -> str:
        """Traditional single-point crossover."""
        lines1 = [line.strip() for line in parent1.split('\n') if line.strip()]
        lines2 = [line.strip() for line in parent2.split('\n') if line.strip()]
        
        if not lines1 or not lines2:
            return parent1 if lines1 else parent2
            
        cross_point1 = random.randint(0, len(lines1))
        cross_point2 = random.randint(0, len(lines2))
        
        offspring = lines1[:cross_point1] + lines2[cross_point2:]
        return '\n'.join(offspring)
    
    def _find_strategic_insertion_point(self, lines: List[str], context: Dict[str, Any] = None) -> int:
        """Find strategic points for command insertion based on script analysis."""
        if not lines:
            return 0
            
        # Look for natural breakpoints (after timing commands, before control commands)
        strategic_points = []
        
        for i, line in enumerate(lines):
            if line.isdigit():  # After timing commands
                strategic_points.append(i + 1)
            elif line.upper() in ['START', 'SELECT', 'OBSERVE']:  # Before control commands
                strategic_points.append(i)
                
        if strategic_points:
            return random.choice(strategic_points)
        else:
            return random.randint(0, len(lines))
    
    def _select_synergistic_command(self, lines: List[str], insert_pos: int) -> str:
        """Select command that synergizes with surrounding commands."""
        if insert_pos > 0 and insert_pos <= len(lines):
            # Look at preceding command
            prev_cmd = lines[insert_pos - 1].upper()
            if prev_cmd in self.COMMAND_SYNERGIES:
                synergy_options = self.COMMAND_SYNERGIES[prev_cmd]
                if random.random() < 0.7:  # 70% chance to use synergy
                    return random.choice(synergy_options)
        
        # Fallback to random selection
        return random.choice(self.ALL_COMMANDS)
    
    def _select_contextual_timing(self, command: str, context: Dict[str, Any] = None) -> Optional[int]:
        """Select appropriate timing based on command and context."""
        if random.random() < 0.4:  # 40% chance to add timing
            if context and 'speed' in str(context).lower():
                return random.choice(self.TIMING_PATTERNS['fast_execution'])
            elif context and 'precision' in str(context).lower():
                return random.choice(self.TIMING_PATTERNS['cautious_execution'])
            else:
                return random.choice(self.TIMING_PATTERNS['standard_execution'])
        return None
    
    def _compute_redundancy_scores(self, lines: List[str]) -> List[float]:
        """Compute redundancy scores for each line to guide deletion."""
        scores = []
        command_counts = {}
        
        # Count command frequencies
        for line in lines:
            if line.upper() in self.ALL_COMMANDS:
                command_counts[line.upper()] = command_counts.get(line.upper(), 0) + 1
        
        # Compute redundancy based on frequency and position
        for i, line in enumerate(lines):
            if line.upper() in self.ALL_COMMANDS:
                frequency = command_counts.get(line.upper(), 1)
                position_penalty = 0.1 if i < len(lines) * 0.8 else 0.0  # Penalize early commands less
                redundancy = (frequency / len(lines)) + position_penalty
            elif line.isdigit():
                # Timing commands - moderate redundancy
                redundancy = 0.3
            else:
                # Unknown commands - high redundancy (remove preferentially)
                redundancy = 0.8
                
            scores.append(min(1.0, redundancy))
            
        return scores
    
    def _select_crossover_points(self, lines1: List[str], lines2: List[str], num_points: int) -> List[int]:
        """Select intelligent crossover points based on command boundaries."""
        min_length = min(len(lines1), len(lines2))
        
        # Find natural breakpoints
        breakpoints = set()
        for i in range(min_length - 1):
            # After control commands
            if lines1[i].upper() in ['START', 'SELECT', 'OBSERVE'] or lines2[i].upper() in ['START', 'SELECT', 'OBSERVE']:
                breakpoints.add(i + 1)
            # After timing sequences
            if (i < len(lines1) - 1 and lines1[i].isdigit() and not lines1[i+1].isdigit()) or \
               (i < len(lines2) - 1 and lines2[i].isdigit() and not lines2[i+1].isdigit()):
                breakpoints.add(i + 1)
        
        # Convert to list and select random subset
        available_points = list(breakpoints) if breakpoints else list(range(1, min_length))
        
        if len(available_points) >= num_points:
            return random.sample(available_points, num_points)
        else:
            # Add additional random points if needed
            additional_points = random.sample(list(range(1, min_length)), 
                                           num_points - len(available_points))
            return list(breakpoints) + additional_points
    
    def _extract_semantic_blocks(self, script: str) -> List[Dict[str, Any]]:
        """Extract semantic blocks from script for intelligent crossover."""
        lines = [line.strip() for line in script.split('\n') if line.strip()]
        blocks = []
        
        current_block = {'type': 'execution', 'commands': [], 'start_idx': 0}
        
        for i, line in enumerate(lines):
            if line.upper() in ['START', 'SELECT']:
                # Finish current block and start setup block
                if current_block['commands']:
                    blocks.append(current_block)
                current_block = {'type': 'setup', 'commands': [line], 'start_idx': i}
            elif line.upper() == 'OBSERVE':
                # Finish current block and start validation block
                if current_block['commands']:
                    blocks.append(current_block)
                current_block = {'type': 'validation', 'commands': [line], 'start_idx': i}
            else:
                current_block['commands'].append(line)
        
        # Add final block
        if current_block['commands']:
            blocks.append(current_block)
            
        return blocks
    
    def update_mutation_success(self, mutation_type: str, success_score: float):
        """Update success history for adaptive mutation rates."""
        if mutation_type not in self.mutation_success_history:
            self.mutation_success_history[mutation_type] = []
            
        self.mutation_success_history[mutation_type].append(success_score)
        
        # Keep only recent history
        if len(self.mutation_success_history[mutation_type]) > 20:
            self.mutation_success_history[mutation_type] = self.mutation_success_history[mutation_type][-20:]
        
        # Adapt mutation rate based on success
        if len(self.mutation_success_history[mutation_type]) >= 5:
            avg_success = sum(self.mutation_success_history[mutation_type][-5:]) / 5
            if avg_success > 0.7:
                # Increase rate for successful mutations
                self.adaptive_mutation_rates[mutation_type] = min(0.5, 
                    self.adaptive_mutation_rates.get(mutation_type, 0.3) * 1.1)
            elif avg_success < 0.3:
                # Decrease rate for unsuccessful mutations
                self.adaptive_mutation_rates[mutation_type] = max(0.05,
                    self.adaptive_mutation_rates.get(mutation_type, 0.3) * 0.9)
    
    def get_adaptive_mutation_rate(self, mutation_type: str) -> float:
        """Get current adaptive mutation rate for a specific mutation type."""
        return self.adaptive_mutation_rates.get(mutation_type, 0.3)


class GeneticOperators:
    """Legacy DSL-aware genetic operators for backward compatibility."""
    
    # Basic DSL commands and their categories  
    MOVEMENT_COMMANDS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    ACTION_COMMANDS = ['A', 'B', 'START', 'SELECT']
    CONTROL_COMMANDS = ['REPEAT', 'END', 'OBSERVE']
    ALL_COMMANDS = MOVEMENT_COMMANDS + ACTION_COMMANDS + CONTROL_COMMANDS
    
    @staticmethod
    def insert_command_mutation(script: str) -> str:
        """Insert a random valid command at a strategic point."""
        lines = [line.strip() for line in script.split('\n') if line.strip()]
        if not lines:
            return random.choice(GeneticOperators.ALL_COMMANDS)
            
        insert_pos = random.randint(0, len(lines))
        command = random.choice(GeneticOperators.ALL_COMMANDS)
        
        if random.random() < 0.3:
            command += f"\n{random.randint(1, 10)}"
            
        lines.insert(insert_pos, command)
        return '\n'.join(lines)
    
    @staticmethod
    def delete_command_mutation(script: str) -> str:
        """Remove a redundant or inefficient command sequence."""
        lines = [line.strip() for line in script.split('\n') if line.strip()]
        if len(lines) <= 2:
            return script
            
        delete_pos = random.randint(0, len(lines) - 1)
        lines.pop(delete_pos)
        return '\n'.join(lines)
    
    @staticmethod
    def modify_timing_mutation(script: str) -> str:
        """Adjust frame delays for optimization."""
        lines = script.split('\n')
        modified_lines = []
        
        for line in lines:
            line = line.strip()
            if line.isdigit():
                current_delay = int(line)
                modifier = random.uniform(0.5, 1.5)
                new_delay = max(1, int(current_delay * modifier))
                modified_lines.append(str(new_delay))
            else:
                modified_lines.append(line)
                
        return '\n'.join(modified_lines)
    
    @staticmethod
    def crossover_sequences(parent1: str, parent2: str) -> str:
        """Combine successful patterns from two parent scripts."""
        lines1 = [line.strip() for line in parent1.split('\n') if line.strip()]
        lines2 = [line.strip() for line in parent2.split('\n') if line.strip()]
        
        if not lines1 or not lines2:
            return parent1 if lines1 else parent2
            
        cross_point1 = random.randint(0, len(lines1))
        cross_point2 = random.randint(0, len(lines2))
        
        offspring = lines1[:cross_point1] + lines2[cross_point2:]
        return '\n'.join(offspring)


class GeneticPopulation:
    """Manages a population of script variants using enhanced genetic algorithm principles."""
    
    def __init__(self, population_size: int = 8, elite_size: int = 2):
        self.population_size = population_size
        self.elite_size = elite_size
        self.variants: List[ScriptVariant] = []
        self.generation = 0
        self.operators = GeneticOperators()  # Legacy operators for compatibility
        self.advanced_operators = AdvancedGeneticOperators()  # Enhanced operators
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        
        # Enhanced evolution parameters
        self.selection_pressure = 1.5  # Tournament size multiplier
        self.elitism_rate = 0.25  # Percentage of population to preserve as elite
        self.diversity_maintenance_rate = 0.15  # Percentage for diversity injection
        self.adaptive_crossover_rate = 0.3  # Dynamic crossover rate
        
        # Performance tracking for operator adaptation
        self.operator_performance: Dict[str, List[float]] = {
            'context_aware_insert': [],
            'adaptive_delete': [],
            'intelligent_timing': [],
            'multi_point_crossover': [],
            'semantic_crossover': []
        }
        
    def initialize_from_patterns(self, patterns: List[Dict[str, Any]], base_script: Optional[str] = None) -> None:
        """Initialize population from successful patterns and base script."""
        self.variants = []
        
        # Add base script if provided
        if base_script:
            self.variants.append(ScriptVariant(base_script, generation=self.generation))
            
        # Generate variants from patterns
        for pattern in patterns[:self.population_size - 1]:
            if 'pattern_sequence' in pattern:
                pattern_script = '\n'.join(pattern['pattern_sequence'])
                variant = ScriptVariant(pattern_script, generation=self.generation)
                self.variants.append(variant)
                
        # Fill remaining slots with mutations of existing variants
        while len(self.variants) < self.population_size:
            if self.variants:
                base_variant = random.choice(self.variants)
                mutated_script = self._apply_random_mutation(base_variant.script_text)
                new_variant = ScriptVariant(mutated_script, generation=self.generation)
                new_variant.parent_ids = [base_variant.variant_id]
                self.variants.append(new_variant)
            else:
                # Generate random script if no patterns available
                random_script = self._generate_random_script()
                self.variants.append(ScriptVariant(random_script, generation=self.generation))
    
    def evaluate_fitness(self, quality_assessor: ScriptQualityAssessor, task: Dict[str, Any]) -> None:
        """Evaluate fitness scores for all variants in population using multi-objective optimization."""
        for variant in self.variants:
            try:
                # Get base quality assessment
                assessment = quality_assessor.assess_script_quality(variant.script_text, task)
                base_quality_score = assessment.get('quality_score', 0.0)
                
                # Estimate execution time from script analysis (simple heuristic)
                script_lines = [line.strip() for line in variant.script_text.split('\n') if line.strip() and not line.strip().isdigit()]
                estimated_execution_time = len(script_lines) * 2.0  # Rough estimate: 2 frames per command
                
                # Compute pattern novelty score (simple approach)
                pattern_novelty_score = self._compute_pattern_novelty(variant)
                
                # Use multi-objective optimizer to compute fitness scores
                fitness_scores = self.multi_objective_optimizer.compute_multi_objective_fitness(
                    variant, base_quality_score, estimated_execution_time, pattern_novelty_score
                )
                
                # Store combined fitness as main fitness
                variant.fitness = fitness_scores['combined_fitness']
                
                # Store detailed scores as attributes for analysis
                variant.speed_score = fitness_scores['speed_score']
                variant.reliability_score = fitness_scores['reliability_score']
                variant.innovation_score = fitness_scores['innovation_score']
                
                # Update Pareto front
                self.multi_objective_optimizer.update_pareto_front(variant, fitness_scores)
                
            except Exception as e:
                logger.debug(f"Fitness evaluation failed for variant {variant.variant_id}: {e}")
                variant.fitness = 0.0
                variant.speed_score = 0.0
                variant.reliability_score = 0.0
                variant.innovation_score = 0.0
                
    def tournament_selection(self, tournament_size: Optional[int] = None) -> ScriptVariant:
        """Select variant using tournament selection."""
        if tournament_size is None:
            tournament_size = getattr(CONFIG.script_development, 'TOURNAMENT_SIZE', 3)
        tournament = random.sample(self.variants, min(tournament_size, len(self.variants)))
        return max(tournament, key=lambda v: v.fitness)
    
    def get_elite(self) -> List[ScriptVariant]:
        """Get the best performing variants (elite)."""
        sorted_variants = sorted(self.variants, key=lambda v: v.fitness, reverse=True)
        return sorted_variants[:self.elite_size]
    
    def get_best_variant(self) -> Optional[ScriptVariant]:
        """Get the highest fitness variant."""
        if not self.variants:
            return None
        return max(self.variants, key=lambda v: v.fitness)
    
    def evolve_generation(self, task_context: Dict[str, Any] = None) -> None:
        """Create next generation using enhanced evolutionary strategies."""
        if not self.variants:
            return
            
        # Calculate adaptive parameters based on population performance
        self._update_evolutionary_parameters()
        
        # Preserve elite with dynamic elitism rate
        elite_count = max(1, int(self.population_size * self.elitism_rate))
        elite = self.get_elite()[:elite_count]
        new_generation = [variant.copy() for variant in elite]
        
        # Maintain diversity through diverse selection, but ensure total doesn't exceed population size
        remaining_slots = self.population_size - elite_count
        diversity_count = min(remaining_slots, int(self.population_size * self.diversity_maintenance_rate))
        if diversity_count > 0:
            # Select diverse variants excluding already selected elite variants
            elite_ids = {variant.variant_id for variant in elite}
            non_elite_variants = [v for v in self.variants if v.variant_id not in elite_ids]
            diverse_variants = self._select_diverse_variants_from_pool(non_elite_variants, diversity_count)
            new_generation.extend([variant.copy() for variant in diverse_variants])
        
        # Generate offspring using advanced operators
        while len(new_generation) < self.population_size:
            # Adaptive selection between mutation and crossover
            if random.random() < self._get_adaptive_mutation_rate():
                # Advanced mutation
                parent = self.enhanced_tournament_selection()
                offspring = parent.copy()
                mutation_type, mutated_script = self._apply_adaptive_mutation(
                    offspring.script_text, task_context, parent.fitness
                )
                offspring.script_text = mutated_script
                offspring.mutation_history.append(f'enhanced_{mutation_type}')
                
                # Update operator performance tracking
                self._track_operator_performance(mutation_type, parent.fitness)
                
            else:
                # Advanced crossover
                parent1 = self.enhanced_tournament_selection()
                parent2 = self.enhanced_tournament_selection()
                
                crossover_type, offspring_script = self._apply_adaptive_crossover(
                    parent1.script_text, parent2.script_text, task_context
                )
                offspring = ScriptVariant(offspring_script, generation=self.generation + 1)
                offspring.parent_ids = [parent1.variant_id, parent2.variant_id]
                offspring.mutation_history.append(f'enhanced_{crossover_type}')
                
                # Update operator performance tracking
                combined_parent_fitness = (parent1.fitness + parent2.fitness) / 2
                self._track_operator_performance(crossover_type, combined_parent_fitness)
                
            new_generation.append(offspring)
            
        # Ensure exact population size is maintained
        new_generation = new_generation[:self.population_size]
        self.variants = new_generation
        self.generation += 1
        
        # Adaptive parameter updates
        self._adapt_evolutionary_parameters()
    
    def enhanced_tournament_selection(self, tournament_size: Optional[int] = None) -> ScriptVariant:
        """Enhanced tournament selection with adaptive pressure."""
        if tournament_size is None:
            base_size = getattr(CONFIG.script_development, 'TOURNAMENT_SIZE', 3)
            tournament_size = max(2, int(base_size * self.selection_pressure))
            
        # Select tournament participants
        tournament = random.sample(self.variants, min(tournament_size, len(self.variants)))
        
        # Multi-objective selection considering fitness, diversity, and age
        scored_participants = []
        for variant in tournament:
            # Primary fitness score
            fitness_score = variant.fitness
            
            # Diversity bonus (reward unique patterns)
            diversity_bonus = self._compute_pattern_novelty(variant) * 0.1
            
            # Age penalty (slight preference for newer variants)
            age_penalty = (self.generation - variant.generation) * 0.02
            
            # Combined score
            total_score = fitness_score + diversity_bonus - age_penalty
            scored_participants.append((variant, total_score))
        
        # Return variant with highest combined score
        return max(scored_participants, key=lambda x: x[1])[0]
    
    def _select_diverse_variants(self, count: int) -> List[ScriptVariant]:
        """Select variants that maximize population diversity."""
        return self._select_diverse_variants_from_pool(self.variants, count)
    
    def _select_diverse_variants_from_pool(self, variant_pool: List[ScriptVariant], count: int) -> List[ScriptVariant]:
        """Select variants that maximize population diversity from a specific pool."""
        if count <= 0 or not variant_pool:
            return []
            
        # Calculate diversity metrics for all variants in pool
        diversity_scores = []
        for variant in variant_pool:
            novelty_score = self._compute_pattern_novelty(variant)
            diversity_scores.append((variant, novelty_score))
        
        # Sort by diversity and select top diverse variants
        diversity_scores.sort(key=lambda x: x[1], reverse=True)
        selected_variants = [variant for variant, _ in diversity_scores[:count]]
        
        return selected_variants
    
    def _apply_adaptive_mutation(self, script: str, context: Dict[str, Any], parent_fitness: float) -> Tuple[str, str]:
        """Apply adaptive mutation based on context and performance."""
        # Select mutation operator based on adaptive rates and context
        available_mutations = []
        
        # Context-aware insertion (good for incomplete scripts)
        if parent_fitness < 0.6:  # Poor performance, try adding commands
            available_mutations.append(('context_aware_insert', 
                                      self.advanced_operators.get_adaptive_mutation_rate('context_aware_insert')))
        
        # Adaptive deletion (good for overly complex scripts)  
        if len(script.split('\n')) > 15:  # Long script, try reducing
            available_mutations.append(('adaptive_delete',
                                      self.advanced_operators.get_adaptive_mutation_rate('adaptive_delete')))
        
        # Intelligent timing (always useful)
        available_mutations.append(('intelligent_timing',
                                  self.advanced_operators.get_adaptive_mutation_rate('intelligent_timing')))
        
        # Select mutation based on weighted probabilities
        if available_mutations:
            weights = [rate for _, rate in available_mutations]
            selected_mutation, _ = random.choices(available_mutations, weights=weights)[0]
        else:
            selected_mutation = 'context_aware_insert'
        
        # Apply selected mutation
        if selected_mutation == 'context_aware_insert':
            mutated_script = self.advanced_operators.context_aware_insert_mutation(script, context)
        elif selected_mutation == 'adaptive_delete':
            mutated_script = self.advanced_operators.adaptive_delete_mutation(script, parent_fitness)
        elif selected_mutation == 'intelligent_timing':
            # Determine performance target from context
            if context and 'speed' in str(context).lower():
                target = 'speed'
            elif context and ('precision' in str(context).lower() or 'accuracy' in str(context).lower()):
                target = 'accuracy'
            else:
                target = 'balanced'
            mutated_script = self.advanced_operators.intelligent_timing_mutation(script, target)
        else:
            # Fallback to legacy mutation
            mutated_script = self._apply_random_mutation(script)
            selected_mutation = 'legacy_mutation'
        
        return selected_mutation, mutated_script
    
    def _apply_adaptive_crossover(self, parent1_script: str, parent2_script: str, 
                                 context: Dict[str, Any]) -> Tuple[str, str]:
        """Apply adaptive crossover based on context and parent characteristics."""
        # Analyze parents to select appropriate crossover
        parent1_length = len(parent1_script.split('\n'))
        parent2_length = len(parent2_script.split('\n'))
        
        # Select crossover operator based on parent characteristics and context
        if abs(parent1_length - parent2_length) > 10:
            # Very different lengths - use semantic crossover to preserve structure
            crossover_type = 'semantic_crossover'
            offspring_script = self.advanced_operators.semantic_crossover(parent1_script, parent2_script, context)
        elif parent1_length > 20 or parent2_length > 20:
            # Long parents - use multi-point crossover for better mixing
            crossover_type = 'multi_point_crossover'
            offspring_script = self.advanced_operators.multi_point_crossover(parent1_script, parent2_script, 2)
        else:
            # Standard parents - use single-point crossover
            crossover_type = 'single_point_crossover'
            offspring_script = self.advanced_operators.single_point_crossover(parent1_script, parent2_script)
        
        return crossover_type, offspring_script
    
    def _get_adaptive_mutation_rate(self) -> float:
        """Get adaptive mutation rate based on population performance and diversity."""
        base_rate = getattr(CONFIG.script_development, 'MUTATION_RATE', 0.7)
        
        # Adjust based on population diversity
        diversity = self.get_diversity_score()
        if diversity < 0.2:  # Low diversity, increase mutation
            return min(0.9, base_rate + 0.2)
        elif diversity > 0.7:  # High diversity, reduce mutation
            return max(0.3, base_rate - 0.2)
        else:
            return base_rate
    
    def _update_evolutionary_parameters(self):
        """Update evolutionary parameters based on population performance."""
        if not self.variants or self.generation < 2:
            return
            
        # Calculate average fitness improvement
        current_avg_fitness = sum(v.fitness for v in self.variants) / len(self.variants)
        
        # Adjust selection pressure based on fitness variance
        fitness_variance = self._calculate_fitness_variance()
        if fitness_variance < 0.05:  # Low variance, increase pressure
            self.selection_pressure = min(2.0, self.selection_pressure + 0.1)
        elif fitness_variance > 0.3:  # High variance, decrease pressure
            self.selection_pressure = max(1.0, self.selection_pressure - 0.1)
    
    def _adapt_evolutionary_parameters(self):
        """Adapt evolutionary parameters based on generation performance."""
        # Track performance trends
        if len(self.variants) > 0:
            best_fitness = max(v.fitness for v in self.variants)
            
            # Adjust crossover rate based on best fitness improvement
            if best_fitness > 0.8:  # High performance, maintain current strategy
                pass
            elif best_fitness < 0.4:  # Poor performance, increase crossover for more exploration
                self.adaptive_crossover_rate = min(0.5, self.adaptive_crossover_rate + 0.05)
            else:  # Moderate performance, slightly favor mutation
                self.adaptive_crossover_rate = max(0.2, self.adaptive_crossover_rate - 0.02)
    
    def _calculate_fitness_variance(self) -> float:
        """Calculate variance in population fitness."""
        if len(self.variants) <= 1:
            return 0.0
            
        fitnesses = [v.fitness for v in self.variants]
        mean_fitness = sum(fitnesses) / len(fitnesses)
        variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)
        
        return variance
    
    def _track_operator_performance(self, operator_type: str, baseline_fitness: float):
        """Track performance of genetic operators for adaptation."""
        if operator_type in self.operator_performance:
            self.operator_performance[operator_type].append(baseline_fitness)
            
            # Keep only recent performance data
            if len(self.operator_performance[operator_type]) > 20:
                self.operator_performance[operator_type] = self.operator_performance[operator_type][-20:]
    
    def get_operator_performance_summary(self) -> Dict[str, float]:
        """Get performance summary for all operators."""
        summary = {}
        for operator, performances in self.operator_performance.items():
            if performances:
                summary[operator] = sum(performances) / len(performances)
            else:
                summary[operator] = 0.5  # Default neutral performance
                
        return summary
        
    def _apply_random_mutation(self, script: str) -> str:
        """Apply a random mutation operator to the script."""
        mutations = [
            self.operators.insert_command_mutation,
            self.operators.delete_command_mutation,
            self.operators.modify_timing_mutation
        ]
        mutation = random.choice(mutations)
        return mutation(script)
    
    def _generate_random_script(self) -> str:
        """Generate a basic random script as fallback."""
        commands = []
        length = random.randint(5, 15)
        
        for _ in range(length):
            command = random.choice(self.operators.ALL_COMMANDS)
            commands.append(command)
            
            # Add timing occasionally
            if random.random() < 0.4:
                commands.append(str(random.randint(1, 5)))
                
        return '\n'.join(commands)
    
    def get_diversity_score(self) -> float:
        """Calculate population diversity to prevent premature convergence."""
        if len(self.variants) <= 1:
            return 0.0
            
        # Simple diversity metric based on script length variance
        lengths = [len(v.script_text.split('\n')) for v in self.variants]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        return min(variance / mean_length, 1.0) if mean_length > 0 else 0.0
    
    def _compute_pattern_novelty(self, variant: ScriptVariant) -> float:
        """Compute novelty score based on how different this variant is from population."""
        if len(self.variants) <= 1:
            return 0.5  # Medium novelty for single variant
            
        # Simple novelty metric: inverse of average similarity to other variants
        similarities = []
        variant_commands = set(line.strip().upper() for line in variant.script_text.split('\\n') 
                             if line.strip().upper() in self.operators.ALL_COMMANDS)
        
        for other_variant in self.variants:
            if other_variant.variant_id == variant.variant_id:
                continue
                
            other_commands = set(line.strip().upper() for line in other_variant.script_text.split('\\n')
                               if line.strip().upper() in self.operators.ALL_COMMANDS)
            
            # Jaccard similarity
            if len(variant_commands | other_commands) == 0:
                similarity = 1.0  # Both empty
            else:
                similarity = len(variant_commands & other_commands) / len(variant_commands | other_commands)
                
            similarities.append(similarity)
        
        if not similarities:
            return 0.5
            
        # Novelty is inverse of average similarity
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity


class ReinforcementLearningEngine:
    """Advanced reinforcement learning system with Q-learning and experience replay."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 replay_buffer_size: int = 1000, exploration_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Q-table: state-action -> value mapping
        # State: (task_context_hash, script_pattern_hash, quality_level)
        # Action: (mutation_type, crossover_type, parameter_adjustment)
        self.q_table: Dict[tuple, Dict[str, float]] = {}
        
        # Experience replay buffer
        self.experience_buffer: List[Dict[str, Any]] = []
        self.replay_buffer_size = replay_buffer_size
        
        # Action space for genetic algorithm decisions
        self.mutation_actions = ['insert_command', 'delete_command', 'modify_timing', 'swap_sequence']
        self.crossover_actions = ['single_point', 'two_point', 'uniform', 'semantic_aware']
        self.parameter_actions = ['increase_population', 'decrease_population', 'adjust_mutation_rate']
        
        # Performance tracking for adaptive learning
        self.action_success_history: Dict[str, List[float]] = {}
        self.recent_experiences: List[Dict[str, Any]] = []
        
        # State encoding for pattern recognition
        self.state_encoder = self._initialize_state_encoder()
    
    def _initialize_state_encoder(self) -> Dict[str, int]:
        """Initialize state encoding dictionary for efficient state representation."""
        return {
            'context_location': {},
            'context_objective': {},
            'script_pattern': {},
            'quality_level': {
                'low': 0,
                'medium': 1, 
                'high': 2
            }
        }
    
    def encode_state(self, task_context: Dict[str, Any], script_pattern: str, 
                     current_quality: float) -> tuple:
        """Encode current state for Q-learning."""
        # Hash task context for consistent state representation
        context_items = sorted(task_context.items()) if task_context else []
        context_hash = hash(str(context_items)) % 10000
        
        # Hash script pattern
        pattern_hash = hash(script_pattern) % 10000 if script_pattern else 0
        
        # Discretize quality level
        if current_quality >= 0.8:
            quality_level = 'high'
        elif current_quality >= 0.5:
            quality_level = 'medium'
        else:
            quality_level = 'low'
            
        return (context_hash, pattern_hash, quality_level)
    
    def select_action(self, state: tuple) -> Dict[str, str]:
        """Select action using epsilon-greedy policy with Q-learning."""
        if random.random() < self.exploration_rate:
            # Exploration: random action
            return {
                'mutation': random.choice(self.mutation_actions),
                'crossover': random.choice(self.crossover_actions),
                'parameter': random.choice(self.parameter_actions)
            }
        
        # Exploitation: best known action
        if state not in self.q_table:
            self.q_table[state] = self._initialize_q_values()
            
        best_action_key = max(self.q_table[state], key=self.q_table[state].get)
        return self._decode_action(best_action_key)
    
    def _initialize_q_values(self) -> Dict[str, float]:
        """Initialize Q-values for all possible action combinations."""
        q_values = {}
        for mutation in self.mutation_actions:
            for crossover in self.crossover_actions:
                for parameter in self.parameter_actions:
                    action_key = f"{mutation}_{crossover}_{parameter}"
                    q_values[action_key] = random.uniform(0.0, 0.1)  # Small random initialization
        return q_values
    
    def _decode_action(self, action_key: str) -> Dict[str, str]:
        """Decode action key back to action dictionary."""
        parts = action_key.split('_', 2)
        mutation_part = parts[0]
        crossover_part = parts[1] 
        parameter_part = '_'.join(parts[2:]) if len(parts) > 2 else parts[2]
        
        return {
            'mutation': mutation_part,
            'crossover': crossover_part,
            'parameter': parameter_part
        }
    
    def update_q_value(self, state: tuple, action: Dict[str, str], reward: float, 
                       next_state: tuple, done: bool = False):
        """Update Q-value using Q-learning algorithm."""
        action_key = f"{action['mutation']}_{action['crossover']}_{action['parameter']}"
        
        if state not in self.q_table:
            self.q_table[state] = self._initialize_q_values()
        if next_state not in self.q_table:
            self.q_table[next_state] = self._initialize_q_values()
            
        current_q = self.q_table[state][action_key]
        
        if done:
            max_next_q = 0.0  # Terminal state
        else:
            max_next_q = max(self.q_table[next_state].values())
            
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action_key] = new_q
    
    def store_experience(self, experience: Dict[str, Any]):
        """Store experience in replay buffer for batch learning."""
        self.experience_buffer.append(experience)
        
        # Maintain buffer size limit
        if len(self.experience_buffer) > self.replay_buffer_size:
            self.experience_buffer.pop(0)
        
        # Track recent experiences for trend analysis
        self.recent_experiences.append(experience)
        if len(self.recent_experiences) > 50:  # Keep last 50 experiences
            self.recent_experiences.pop(0)
    
    def replay_learning(self, batch_size: int = 32):
        """Perform experience replay learning from stored experiences."""
        if len(self.experience_buffer) < batch_size:
            return
            
        # Sample random batch from experience buffer
        batch = random.sample(self.experience_buffer, batch_size)
        
        for experience in batch:
            self.update_q_value(
                experience['state'],
                experience['action'],
                experience['reward'],
                experience['next_state'],
                experience['done']
            )
    
    def compute_reward(self, quality_improvement: float, iteration_efficiency: float,
                       diversity_score: float, pattern_novelty: float) -> float:
        """Compute reward signal for reinforcement learning."""
        # Multi-component reward combining different objectives
        quality_reward = quality_improvement * 2.0  # Primary reward
        efficiency_reward = (1.0 - iteration_efficiency) * 0.5  # Reward faster solutions
        diversity_reward = diversity_score * 0.3  # Reward population diversity
        novelty_reward = pattern_novelty * 0.2  # Reward innovative patterns
        
        total_reward = quality_reward + efficiency_reward + diversity_reward + novelty_reward
        
        # Normalize reward to [-1, 1] range
        return max(-1.0, min(1.0, total_reward))
    
    def get_success_trends(self) -> Dict[str, float]:
        """Analyze success trends from recent experiences."""
        if not self.recent_experiences:
            return {'trend_score': 0.5, 'confidence': 0.0}
            
        recent_rewards = [exp['reward'] for exp in self.recent_experiences[-20:]]
        if len(recent_rewards) < 3:
            return {'trend_score': 0.5, 'confidence': 0.0}
            
        # Calculate trend using simple linear regression
        n = len(recent_rewards)
        x_mean = (n - 1) / 2
        y_mean = sum(recent_rewards) / n
        
        numerator = sum((i - x_mean) * (reward - y_mean) for i, reward in enumerate(recent_rewards))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            trend_slope = 0
        else:
            trend_slope = numerator / denominator
            
        # Convert slope to trend score
        trend_score = max(0.0, min(1.0, 0.5 + trend_slope * 0.5))
        confidence = min(1.0, n / 20.0)  # Higher confidence with more data
        
        return {'trend_score': trend_score, 'confidence': confidence}
    
    def adapt_exploration_rate(self, performance_trend: float):
        """Adapt exploration rate based on learning performance."""
        if performance_trend > 0.7:  # Good performance, reduce exploration
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
        elif performance_trend < 0.3:  # Poor performance, increase exploration
            self.exploration_rate = min(0.3, self.exploration_rate * 1.1)


class RealTimePerformanceMonitor:
    """Real-time performance monitoring and adaptive parameter tuning system."""
    
    def __init__(self):
        self.performance_metrics = {
            'development_time': [],
            'quality_scores': [],
            'success_rates': [],
            'iteration_counts': [],
            'pattern_usage_effectiveness': {}
        }
        
        # Adaptive thresholds and parameters
        self.adaptive_parameters = {
            'max_iterations': CONFIG.script_development['DEFAULT_MAX_ITERATIONS'],
            'population_size': CONFIG.script_development['GENETIC_POPULATION_SIZE'],
            'mutation_rate': CONFIG.script_development['mutation_rate'],
            'success_threshold': CONFIG.script_development['SUCCESS_QUALITY_THRESHOLD']
        }
        
        # Performance trend analysis
        self.trend_window = 10  # Number of recent sessions to analyze
        self.adjustment_sensitivity = 0.1  # How aggressively to adjust parameters
        
        # System resource monitoring
        self.resource_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'response_times': []
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'avg_development_time': 45000,  # 45 seconds
            'low_success_rate': 0.3,
            'high_iteration_usage': 0.8  # 80% of max iterations used
        }
        
        # Optimization strategies
        self.optimization_strategies = {
            'speed_optimization': self._apply_speed_optimization,
            'quality_optimization': self._apply_quality_optimization,
            'efficiency_optimization': self._apply_efficiency_optimization,
            'balanced_optimization': self._apply_balanced_optimization
        }
        
        # Current optimization mode
        self.current_optimization_mode = 'balanced_optimization'
        
    def record_development_session(self, session_data: Dict[str, Any]) -> None:
        """Record performance data from a development session."""
        # Core metrics
        self.performance_metrics['development_time'].append(session_data.get('development_time_ms', 0))
        self.performance_metrics['quality_scores'].append(session_data.get('quality_score', 0.0))
        self.performance_metrics['iteration_counts'].append(session_data.get('refinement_iterations', 0))
        
        # Calculate success rate
        success = 1.0 if session_data.get('quality_score', 0.0) >= self.adaptive_parameters['success_threshold'] else 0.0
        self.performance_metrics['success_rates'].append(success)
        
        # Pattern effectiveness tracking
        patterns_used = session_data.get('patterns_used', [])
        for pattern_id in patterns_used:
            if pattern_id not in self.performance_metrics['pattern_usage_effectiveness']:
                self.performance_metrics['pattern_usage_effectiveness'][pattern_id] = []
            self.performance_metrics['pattern_usage_effectiveness'][pattern_id].append(session_data.get('quality_score', 0.0))
        
        # Resource metrics (if available)
        if 'system_metrics' in session_data:
            system_metrics = session_data['system_metrics']
            self.resource_metrics['response_times'].append(system_metrics.get('response_time', 0))
        
        # Maintain sliding window
        self._maintain_metrics_window()
        
        # Trigger real-time analysis and adaptation
        self._analyze_and_adapt()
    
    def _maintain_metrics_window(self) -> None:
        """Maintain sliding window of recent metrics."""
        max_history = 50  # Keep last 50 sessions
        
        for metric_name, metric_values in self.performance_metrics.items():
            if isinstance(metric_values, list) and len(metric_values) > max_history:
                self.performance_metrics[metric_name] = metric_values[-max_history:]
        
        for resource_name, resource_values in self.resource_metrics.items():
            if len(resource_values) > max_history:
                self.resource_metrics[resource_name] = resource_values[-max_history:]
    
    def _analyze_and_adapt(self) -> None:
        """Analyze recent performance and adapt parameters accordingly."""
        if len(self.performance_metrics['development_time']) < 5:
            return  # Need more data points
        
        # Calculate recent performance trends
        recent_window = min(self.trend_window, len(self.performance_metrics['development_time']))
        
        # Development time trend
        recent_times = self.performance_metrics['development_time'][-recent_window:]
        avg_dev_time = sum(recent_times) / len(recent_times)
        
        # Quality score trend
        recent_quality = self.performance_metrics['quality_scores'][-recent_window:]
        avg_quality = sum(recent_quality) / len(recent_quality)
        
        # Success rate trend
        recent_success = self.performance_metrics['success_rates'][-recent_window:]
        success_rate = sum(recent_success) / len(recent_success)
        
        # Iteration usage trend
        recent_iterations = self.performance_metrics['iteration_counts'][-recent_window:]
        avg_iterations = sum(recent_iterations) / len(recent_iterations)
        max_iterations = self.adaptive_parameters['max_iterations']
        iteration_usage_ratio = avg_iterations / max_iterations if max_iterations > 0 else 0
        
        # Determine optimization strategy
        self._select_optimization_strategy(avg_dev_time, avg_quality, success_rate, iteration_usage_ratio)
        
        # Apply adaptations
        self._apply_adaptive_parameters(avg_dev_time, avg_quality, success_rate, iteration_usage_ratio)
        
        # Check for performance alerts
        self._check_performance_alerts(avg_dev_time, success_rate, iteration_usage_ratio)
    
    def _select_optimization_strategy(self, avg_dev_time: float, avg_quality: float, 
                                     success_rate: float, iteration_usage_ratio: float) -> None:
        """Select appropriate optimization strategy based on current performance."""
        if avg_dev_time > self.alert_thresholds['avg_development_time']:
            # Development time is too high - prioritize speed
            self.current_optimization_mode = 'speed_optimization'
        elif success_rate < self.alert_thresholds['low_success_rate']:
            # Success rate is too low - prioritize quality
            self.current_optimization_mode = 'quality_optimization'
        elif iteration_usage_ratio > self.alert_thresholds['high_iteration_usage']:
            # Using too many iterations - prioritize efficiency
            self.current_optimization_mode = 'efficiency_optimization'
        else:
            # Performance is acceptable - maintain balance
            self.current_optimization_mode = 'balanced_optimization'
    
    def _apply_adaptive_parameters(self, avg_dev_time: float, avg_quality: float, 
                                  success_rate: float, iteration_usage_ratio: float) -> None:
        """Apply adaptive parameter adjustments based on performance analysis."""
        # Apply selected optimization strategy
        if self.current_optimization_mode in self.optimization_strategies:
            self.optimization_strategies[self.current_optimization_mode](
                avg_dev_time, avg_quality, success_rate, iteration_usage_ratio
            )
        
        # Ensure parameters stay within reasonable bounds
        self._clamp_parameters()
    
    def _apply_speed_optimization(self, avg_dev_time: float, avg_quality: float, 
                                 success_rate: float, iteration_usage_ratio: float) -> None:
        """Apply optimizations focused on reducing development time."""
        # Reduce max iterations to speed up development
        if avg_dev_time > 30000:  # 30 seconds
            adjustment = -int(self.adaptive_parameters['max_iterations'] * 0.1)
            self.adaptive_parameters['max_iterations'] = max(2, self.adaptive_parameters['max_iterations'] + adjustment)
        
        # Reduce population size for faster evolution
        if avg_dev_time > 25000:  # 25 seconds
            adjustment = -max(1, int(self.adaptive_parameters['population_size'] * 0.1))
            self.adaptive_parameters['population_size'] = max(4, self.adaptive_parameters['population_size'] + adjustment)
        
        # Increase mutation rate for faster exploration
        self.adaptive_parameters['mutation_rate'] = min(0.9, self.adaptive_parameters['mutation_rate'] + 0.1)
        
        logger.info("Applied speed optimization adjustments")
    
    def _apply_quality_optimization(self, avg_dev_time: float, avg_quality: float, 
                                   success_rate: float, iteration_usage_ratio: float) -> None:
        """Apply optimizations focused on improving solution quality."""
        # Increase max iterations to allow more refinement
        if success_rate < 0.4:
            adjustment = int(self.adaptive_parameters['max_iterations'] * 0.2)
            self.adaptive_parameters['max_iterations'] = min(8, self.adaptive_parameters['max_iterations'] + adjustment)
        
        # Increase population size for better exploration
        if avg_quality < 0.5:
            adjustment = max(1, int(self.adaptive_parameters['population_size'] * 0.15))
            self.adaptive_parameters['population_size'] = min(12, self.adaptive_parameters['population_size'] + adjustment)
        
        # Reduce mutation rate for more stable evolution
        self.adaptive_parameters['mutation_rate'] = max(0.4, self.adaptive_parameters['mutation_rate'] - 0.1)
        
        # Lower success threshold temporarily to build success momentum
        if success_rate < 0.3:
            self.adaptive_parameters['success_threshold'] = max(0.5, self.adaptive_parameters['success_threshold'] - 0.05)
        
        logger.info("Applied quality optimization adjustments")
    
    def _apply_efficiency_optimization(self, avg_dev_time: float, avg_quality: float, 
                                      success_rate: float, iteration_usage_ratio: float) -> None:
        """Apply optimizations focused on improving efficiency."""
        # Optimize iteration usage
        if iteration_usage_ratio > 0.9:
            # Using almost all iterations - reduce max iterations slightly
            adjustment = -1
            self.adaptive_parameters['max_iterations'] = max(2, self.adaptive_parameters['max_iterations'] + adjustment)
        elif iteration_usage_ratio < 0.5 and avg_quality > 0.7:
            # Not using many iterations but quality is good - can reduce further
            adjustment = -1
            self.adaptive_parameters['max_iterations'] = max(2, self.adaptive_parameters['max_iterations'] + adjustment)
        
        # Balance population size based on quality vs time trade-off
        if avg_quality > 0.7 and avg_dev_time > 20000:  # Good quality but slow
            adjustment = -1
            self.adaptive_parameters['population_size'] = max(4, self.adaptive_parameters['population_size'] + adjustment)
        
        logger.info("Applied efficiency optimization adjustments")
    
    def _apply_balanced_optimization(self, avg_dev_time: float, avg_quality: float, 
                                    success_rate: float, iteration_usage_ratio: float) -> None:
        """Apply balanced optimizations to maintain overall system performance."""
        # Make small adjustments to maintain stability
        base_max_iter = CONFIG.script_development['DEFAULT_MAX_ITERATIONS']
        base_pop_size = CONFIG.script_development['GENETIC_POPULATION_SIZE']
        base_mutation_rate = CONFIG.script_development['mutation_rate']
        
        # Gradually return parameters to baseline if performance is stable
        if abs(self.adaptive_parameters['max_iterations'] - base_max_iter) > 1:
            direction = 1 if base_max_iter > self.adaptive_parameters['max_iterations'] else -1
            self.adaptive_parameters['max_iterations'] += direction * 0.5
        
        if abs(self.adaptive_parameters['population_size'] - base_pop_size) > 1:
            direction = 1 if base_pop_size > self.adaptive_parameters['population_size'] else -1
            self.adaptive_parameters['population_size'] += direction * 0.5
        
        # Smooth mutation rate adjustments
        rate_diff = abs(self.adaptive_parameters['mutation_rate'] - base_mutation_rate)
        if rate_diff > 0.05:
            direction = 1 if base_mutation_rate > self.adaptive_parameters['mutation_rate'] else -1
            self.adaptive_parameters['mutation_rate'] += direction * 0.02
    
    def _clamp_parameters(self) -> None:
        """Ensure all adaptive parameters stay within acceptable bounds."""
        # Max iterations bounds
        self.adaptive_parameters['max_iterations'] = max(2, min(10, int(self.adaptive_parameters['max_iterations'])))
        
        # Population size bounds
        self.adaptive_parameters['population_size'] = max(4, min(16, int(self.adaptive_parameters['population_size'])))
        
        # Mutation rate bounds
        self.adaptive_parameters['mutation_rate'] = max(0.2, min(0.9, self.adaptive_parameters['mutation_rate']))
        
        # Success threshold bounds
        self.adaptive_parameters['success_threshold'] = max(0.4, min(0.95, self.adaptive_parameters['success_threshold']))
    
    def _check_performance_alerts(self, avg_dev_time: float, success_rate: float, 
                                 iteration_usage_ratio: float) -> None:
        """Check for performance issues that require attention."""
        alerts = []
        
        if avg_dev_time > self.alert_thresholds['avg_development_time']:
            alerts.append(f"High average development time: {avg_dev_time/1000:.1f}s")
        
        if success_rate < self.alert_thresholds['low_success_rate']:
            alerts.append(f"Low success rate: {success_rate:.2%}")
        
        if iteration_usage_ratio > self.alert_thresholds['high_iteration_usage']:
            alerts.append(f"High iteration usage: {iteration_usage_ratio:.1%}")
        
        if alerts:
            logger.warning(f"Performance alerts: {'; '.join(alerts)}")
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current adaptive parameters for use in script development."""
        return self.adaptive_parameters.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_metrics['development_time']:
            return {'status': 'insufficient_data'}
        
        recent_window = min(10, len(self.performance_metrics['development_time']))
        
        recent_times = self.performance_metrics['development_time'][-recent_window:]
        recent_quality = self.performance_metrics['quality_scores'][-recent_window:]
        recent_success = self.performance_metrics['success_rates'][-recent_window:]
        recent_iterations = self.performance_metrics['iteration_counts'][-recent_window:]
        
        return {
            'status': 'active',
            'optimization_mode': self.current_optimization_mode,
            'sessions_analyzed': len(self.performance_metrics['development_time']),
            'recent_metrics': {
                'avg_development_time_ms': sum(recent_times) / len(recent_times),
                'avg_quality_score': sum(recent_quality) / len(recent_quality),
                'success_rate': sum(recent_success) / len(recent_success),
                'avg_iterations': sum(recent_iterations) / len(recent_iterations)
            },
            'adaptive_parameters': self.adaptive_parameters,
            'pattern_effectiveness': self._get_pattern_effectiveness_summary()
        }
    
    def _get_pattern_effectiveness_summary(self) -> Dict[str, float]:
        """Get summary of pattern effectiveness."""
        effectiveness_summary = {}
        
        for pattern_id, quality_scores in self.performance_metrics['pattern_usage_effectiveness'].items():
            if quality_scores:
                effectiveness_summary[pattern_id] = sum(quality_scores) / len(quality_scores)
        
        # Sort by effectiveness
        sorted_patterns = sorted(effectiveness_summary.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_patterns[:10])  # Top 10 most effective patterns
    
    def reset_adaptations(self) -> None:
        """Reset adaptive parameters to baseline configuration."""
        self.adaptive_parameters = {
            'max_iterations': CONFIG.script_development['DEFAULT_MAX_ITERATIONS'],
            'population_size': CONFIG.script_development['GENETIC_POPULATION_SIZE'],
            'mutation_rate': CONFIG.script_development['mutation_rate'],
            'success_threshold': CONFIG.script_development['SUCCESS_QUALITY_THRESHOLD']
        }
        self.current_optimization_mode = 'balanced_optimization'
        logger.info("Reset adaptive parameters to baseline configuration")


class CrossWorkerPatternSynthesis:
    """Advanced pattern synthesis system for combining successful patterns across workers."""
    
    def __init__(self):
        self.pattern_combination_cache: Dict[str, Dict[str, Any]] = {}
        self.synthesis_success_history: Dict[str, List[float]] = {}
        self.cross_pollination_matrix: Dict[str, Dict[str, float]] = {}
        self.pattern_affinity_scores: Dict[str, Dict[str, float]] = {}
        
        # Synthesis strategies with success tracking
        self.synthesis_strategies = {
            'sequential_fusion': self._sequential_fusion_synthesis,
            'interleaved_merge': self._interleaved_merge_synthesis,
            'hierarchical_composition': self._hierarchical_composition_synthesis,
            'semantic_interpolation': self._semantic_interpolation_synthesis,
            'evolutionary_recombination': self._evolutionary_recombination_synthesis
        }
        
        # Strategy performance tracking
        self.strategy_performance: Dict[str, List[float]] = {
            strategy: [] for strategy in self.synthesis_strategies.keys()
        }
    
    def synthesize_patterns(self, patterns: List[Dict[str, Any]], 
                          target_context: Dict[str, Any], 
                          synthesis_count: int = 3) -> List[Dict[str, Any]]:
        """Synthesize new patterns by combining successful patterns from different workers."""
        if len(patterns) < 2:
            return patterns
        
        synthesized_patterns = []
        
        # Group patterns by worker source for cross-pollination
        worker_patterns = self._group_patterns_by_worker(patterns)
        
        # Apply different synthesis strategies
        for strategy_name, strategy_func in self.synthesis_strategies.items():
            if len(synthesized_patterns) >= synthesis_count:
                break
                
            try:
                synthetic_pattern = strategy_func(patterns, target_context)
                if synthetic_pattern:
                    synthetic_pattern['synthesis_strategy'] = strategy_name
                    synthetic_pattern['synthesis_source_count'] = len(patterns)
                    synthetic_pattern['synthesis_timestamp'] = time.time()
                    synthesized_patterns.append(synthetic_pattern)
                    
            except Exception as e:
                logger.debug(f"Pattern synthesis strategy {strategy_name} failed: {e}")
        
        # Cross-validate synthesized patterns
        validated_patterns = self._validate_synthesized_patterns(synthesized_patterns, target_context)
        
        return validated_patterns
    
    def _group_patterns_by_worker(self, patterns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group patterns by their originating worker for cross-pollination analysis."""
        worker_groups = {}
        
        for pattern in patterns:
            worker_source = pattern.get('discovered_by', 'unknown')
            if worker_source not in worker_groups:
                worker_groups[worker_source] = []
            worker_groups[worker_source].append(pattern)
            
        return worker_groups
    
    def _sequential_fusion_synthesis(self, patterns: List[Dict[str, Any]], 
                                   target_context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize by sequentially chaining high-success patterns."""
        # Sort patterns by success rate
        sorted_patterns = sorted(patterns, key=lambda p: p.get('success_rate', 0.0), reverse=True)
        
        # Take top patterns and chain their sequences
        fusion_sequence = []
        combined_success = 1.0
        
        for pattern in sorted_patterns[:3]:  # Use top 3 patterns
            sequence = pattern.get('pattern_sequence', [])
            if sequence:
                fusion_sequence.extend(sequence)
                combined_success *= pattern.get('success_rate', 0.5)
        
        if not fusion_sequence:
            return None
            
        # Calculate synthesized success rate (geometric mean with diversity bonus)
        diversity_bonus = min(0.2, len(set(fusion_sequence)) / len(fusion_sequence))
        synthesized_success = (combined_success ** (1/min(3, len(sorted_patterns)))) + diversity_bonus
        
        return {
            'pattern_id': f'fusion_{uuid.uuid4().hex[:8]}',
            'name': 'Sequential Fusion Pattern',
            'pattern_sequence': fusion_sequence,
            'success_rate': min(1.0, synthesized_success),
            'context': target_context,
            'synthesis_type': 'sequential_fusion',
            'source_patterns': [p.get('pattern_id', 'unknown') for p in sorted_patterns[:3]]
        }
    
    def _interleaved_merge_synthesis(self, patterns: List[Dict[str, Any]], 
                                   target_context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize by interleaving commands from different patterns."""
        if len(patterns) < 2:
            return None
            
        # Get sequences from different patterns
        sequences = []
        for pattern in patterns[:4]:  # Use up to 4 patterns
            seq = pattern.get('pattern_sequence', [])
            if seq:
                sequences.append((seq, pattern.get('success_rate', 0.5)))
        
        if len(sequences) < 2:
            return None
            
        # Interleave sequences based on success rates (higher success = more representation)
        merged_sequence = []
        max_length = max(len(seq[0]) for seq in sequences)
        
        for i in range(max_length):
            for seq, success_rate in sequences:
                if i < len(seq) and random.random() < success_rate:
                    merged_sequence.append(seq[i])
        
        if not merged_sequence:
            return None
            
        # Calculate combined success rate
        avg_success = sum(success for _, success in sequences) / len(sequences)
        interleave_bonus = 0.1 * (len(sequences) - 1)  # Bonus for more patterns
        
        return {
            'pattern_id': f'interleaved_{uuid.uuid4().hex[:8]}',
            'name': 'Interleaved Merge Pattern',
            'pattern_sequence': merged_sequence,
            'success_rate': min(1.0, avg_success + interleave_bonus),
            'context': target_context,
            'synthesis_type': 'interleaved_merge',
            'source_patterns': [p.get('pattern_id', 'unknown') for p in patterns[:4]]
        }
    
    def _hierarchical_composition_synthesis(self, patterns: List[Dict[str, Any]], 
                                          target_context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize using hierarchical composition of successful sub-patterns."""
        # Identify common sub-patterns and their success contexts
        sub_patterns = self._extract_sub_patterns(patterns)
        if not sub_patterns:
            return None
            
        # Build hierarchical structure: setup -> execution -> validation
        setup_patterns = [sp for sp in sub_patterns if self._classify_sub_pattern_phase(sp) == 'setup']
        exec_patterns = [sp for sp in sub_patterns if self._classify_sub_pattern_phase(sp) == 'execution']
        valid_patterns = [sp for sp in sub_patterns if self._classify_sub_pattern_phase(sp) == 'validation']
        
        # Compose hierarchical sequence
        composed_sequence = []
        
        # Add setup phase
        if setup_patterns:
            best_setup = max(setup_patterns, key=lambda p: p.get('frequency', 0))
            composed_sequence.extend(best_setup['sequence'])
        
        # Add execution phase
        if exec_patterns:
            best_exec = max(exec_patterns, key=lambda p: p.get('frequency', 0))
            composed_sequence.extend(best_exec['sequence'])
        
        # Add validation phase
        if valid_patterns:
            best_valid = max(valid_patterns, key=lambda p: p.get('frequency', 0))
            composed_sequence.extend(best_valid['sequence'])
        
        if not composed_sequence:
            return None
            
        # Calculate hierarchical success rate
        phase_count = sum(1 for phases in [setup_patterns, exec_patterns, valid_patterns] if phases)
        hierarchical_bonus = 0.15 * phase_count  # Bonus for more phases
        base_success = sum(p.get('success_rate', 0.5) for p in patterns) / len(patterns)
        
        return {
            'pattern_id': f'hierarchical_{uuid.uuid4().hex[:8]}',
            'name': 'Hierarchical Composition Pattern',
            'pattern_sequence': composed_sequence,
            'success_rate': min(1.0, base_success + hierarchical_bonus),
            'context': target_context,
            'synthesis_type': 'hierarchical_composition',
            'source_patterns': [p.get('pattern_id', 'unknown') for p in patterns]
        }
    
    def _semantic_interpolation_synthesis(self, patterns: List[Dict[str, Any]], 
                                        target_context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize using semantic interpolation between similar patterns."""
        # Find patterns with similar contexts
        context_similarities = []
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                similarity = self._compute_context_similarity(
                    pattern1.get('context', {}), pattern2.get('context', {})
                )
                if similarity > 0.6:  # High similarity threshold
                    context_similarities.append((i, j, similarity, pattern1, pattern2))
        
        if not context_similarities:
            return None
            
        # Select best similar pair
        best_pair = max(context_similarities, key=lambda x: x[2])
        _, _, similarity, pattern1, pattern2 = best_pair
        
        # Interpolate between the two patterns
        seq1 = pattern1.get('pattern_sequence', [])
        seq2 = pattern2.get('pattern_sequence', [])
        
        interpolated_sequence = self._interpolate_sequences(seq1, seq2, similarity)
        
        if not interpolated_sequence:
            return None
            
        # Calculate interpolated success rate
        success1 = pattern1.get('success_rate', 0.5)
        success2 = pattern2.get('success_rate', 0.5)
        interpolated_success = (success1 + success2) / 2 + (similarity * 0.1)
        
        return {
            'pattern_id': f'interpolated_{uuid.uuid4().hex[:8]}',
            'name': 'Semantic Interpolation Pattern',
            'pattern_sequence': interpolated_sequence,
            'success_rate': min(1.0, interpolated_success),
            'context': target_context,
            'synthesis_type': 'semantic_interpolation',
            'source_patterns': [pattern1.get('pattern_id', 'unknown'), pattern2.get('pattern_id', 'unknown')]
        }
    
    def _evolutionary_recombination_synthesis(self, patterns: List[Dict[str, Any]], 
                                            target_context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize using evolutionary recombination principles."""
        if len(patterns) < 2:
            return None
            
        # Select two parent patterns based on fitness (success rate)
        weights = [p.get('success_rate', 0.5) for p in patterns]
        parent1, parent2 = random.choices(patterns, weights=weights, k=2)
        
        seq1 = parent1.get('pattern_sequence', [])
        seq2 = parent2.get('pattern_sequence', [])
        
        if not seq1 or not seq2:
            return None
            
        # Single-point crossover
        crossover_point = random.randint(1, min(len(seq1), len(seq2)) - 1)
        offspring_sequence = seq1[:crossover_point] + seq2[crossover_point:]
        
        # Apply mutation with low probability
        if random.random() < 0.3:  # 30% mutation rate
            mutation_point = random.randint(0, len(offspring_sequence) - 1)
            new_command = random.choice(['A', 'B', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'START', 'SELECT'])
            offspring_sequence[mutation_point] = new_command
        
        # Calculate evolutionary fitness
        parent_fitness = (parent1.get('success_rate', 0.5) + parent2.get('success_rate', 0.5)) / 2
        mutation_bonus = 0.05 if random.random() < 0.3 else 0.0
        evolutionary_fitness = min(1.0, parent_fitness + mutation_bonus)
        
        return {
            'pattern_id': f'evolved_{uuid.uuid4().hex[:8]}',
            'name': 'Evolutionary Recombination Pattern',
            'pattern_sequence': offspring_sequence,
            'success_rate': evolutionary_fitness,
            'context': target_context,
            'synthesis_type': 'evolutionary_recombination',
            'source_patterns': [parent1.get('pattern_id', 'unknown'), parent2.get('pattern_id', 'unknown')]
        }
    
    def _extract_sub_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract common sub-patterns from a collection of patterns."""
        all_sequences = []
        for pattern in patterns:
            seq = pattern.get('pattern_sequence', [])
            if len(seq) >= 3:  # Minimum sub-pattern length
                all_sequences.append(seq)
        
        if not all_sequences:
            return []
            
        # Find common subsequences
        sub_pattern_counts = {}
        
        for seq in all_sequences:
            for i in range(len(seq) - 2):
                for j in range(i + 3, len(seq) + 1):
                    sub_seq = tuple(seq[i:j])
                    sub_pattern_counts[sub_seq] = sub_pattern_counts.get(sub_seq, 0) + 1
        
        # Filter for frequently occurring sub-patterns
        frequent_sub_patterns = []
        for sub_seq, count in sub_pattern_counts.items():
            if count >= 2 and len(sub_seq) <= 6:  # Appears in multiple patterns
                frequent_sub_patterns.append({
                    'sequence': list(sub_seq),
                    'frequency': count,
                    'length': len(sub_seq)
                })
        
        return frequent_sub_patterns
    
    def _classify_sub_pattern_phase(self, sub_pattern: Dict[str, Any]) -> str:
        """Classify sub-pattern into execution phases."""
        sequence = sub_pattern['sequence']
        
        # Simple heuristic-based classification
        if any(cmd in sequence for cmd in ['START', 'SELECT']):
            return 'setup'
        elif any(cmd in sequence for cmd in ['OBSERVE']):
            return 'validation'
        else:
            return 'execution'
    
    def _compute_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Compute similarity between two pattern contexts."""
        if not context1 or not context2:
            return 0.0
            
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
            
        similarity_score = 0.0
        for key in common_keys:
            val1, val2 = str(context1[key]), str(context2[key])
            if val1 == val2:
                similarity_score += 1.0
            elif val1.lower() in val2.lower() or val2.lower() in val1.lower():
                similarity_score += 0.5
                
        return similarity_score / len(common_keys)
    
    def _interpolate_sequences(self, seq1: List[str], seq2: List[str], similarity: float) -> List[str]:
        """Interpolate between two command sequences based on similarity."""
        min_length = min(len(seq1), len(seq2))
        max_length = max(len(seq1), len(seq2))
        
        # Target length based on similarity
        target_length = int(min_length + (max_length - min_length) * similarity)
        
        interpolated = []
        for i in range(target_length):
            if i < len(seq1) and i < len(seq2):
                # Both sequences have commands at this position
                if seq1[i] == seq2[i]:
                    interpolated.append(seq1[i])
                else:
                    # Choose based on weighted random selection
                    if random.random() < similarity:
                        interpolated.append(random.choice([seq1[i], seq2[i]]))
                    else:
                        interpolated.append(seq1[i] if random.random() < 0.5 else seq2[i])
            elif i < len(seq1):
                interpolated.append(seq1[i])
            elif i < len(seq2):
                interpolated.append(seq2[i])
        
        return interpolated
    
    def _validate_synthesized_patterns(self, patterns: List[Dict[str, Any]], 
                                     target_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate synthesized patterns for correctness and relevance."""
        validated = []
        
        for pattern in patterns:
            sequence = pattern.get('pattern_sequence', [])
            
            # Basic validation checks
            if not sequence or len(sequence) < 2:
                continue
                
            # Check for valid commands
            valid_commands = {'A', 'B', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'START', 'SELECT', 'OBSERVE'}
            invalid_commands = [cmd for cmd in sequence if cmd not in valid_commands and not cmd.isdigit()]
            
            if invalid_commands:
                logger.debug(f"Synthesized pattern contains invalid commands: {invalid_commands}")
                continue
                
            # Check for reasonable length
            if len(sequence) > 50:  # Too long
                pattern['pattern_sequence'] = sequence[:50]  # Truncate
                
            # Add synthesis metadata
            pattern['validated'] = True
            pattern['validation_timestamp'] = time.time()
            
            validated.append(pattern)
        
        return validated
    
    def update_synthesis_performance(self, pattern_id: str, success_score: float):
        """Update performance tracking for synthesis strategies."""
        # Find pattern and update strategy performance
        for strategy in self.strategy_performance:
            # This would be called with actual performance data
            # For now, we track it generally
            pass
    
    def get_best_synthesis_strategy(self) -> str:
        """Get the currently best performing synthesis strategy."""
        best_strategy = 'sequential_fusion'  # Default
        best_score = 0.0
        
        for strategy, scores in self.strategy_performance.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_strategy = strategy
        
        return best_strategy


class PatternProcessor:
    """Advanced pattern processing and cross-worker pattern synthesis.
    
    Provides a clean API for pattern synthesis, validation, and cross-worker
    pattern sharing while leveraging the existing CrossWorkerPatternSynthesis
    infrastructure for advanced synthesis capabilities.
    """
    
    def __init__(self, cross_worker_synthesis: Optional['CrossWorkerPatternSynthesis'] = None):
        # Use provided synthesis engine or create new one
        self.synthesis_engine = cross_worker_synthesis or CrossWorkerPatternSynthesis()
        
        # Pattern processing metrics
        self.processing_times: List[float] = []
        self.synthesis_success_rate = 0.0
        self.processed_pattern_count = 0
        
        # Pattern validation and quality tracking
        self.pattern_quality_scores: Dict[str, List[float]] = {}
        self.synthesis_strategy_performance: Dict[str, List[float]] = {}
        
        # Pattern classification and tagging
        self.pattern_categories = {
            'movement': ['UP', 'DOWN', 'LEFT', 'RIGHT'],
            'action': ['A', 'B', 'START', 'SELECT'],
            'timing': ['WAIT'],
            'complex': ['OBSERVE', 'IF', 'THEN', 'ELSE', 'REPEAT']
        }
    
    def process_patterns(self, source_patterns: List[Dict[str, Any]], 
                        target_context: Dict[str, Any],
                        processing_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process and synthesize patterns for a target context.
        
        Args:
            source_patterns: List of source patterns from different workers
            target_context: Target context for pattern synthesis
            processing_options: Optional processing configuration
            
        Returns:
            Dictionary with processed patterns and synthesis metadata
        """
        processing_start = time.time()
        
        try:
            # Set default processing options
            options = processing_options or {}
            synthesis_count = options.get('synthesis_count', 3)
            quality_threshold = options.get('quality_threshold', 0.6)
            enable_validation = options.get('enable_validation', True)
            
            # Pre-process patterns for synthesis
            validated_patterns = self._validate_source_patterns(source_patterns)
            if not validated_patterns:
                return self._get_empty_processing_result("no_valid_patterns")
            
            # Classify patterns for better synthesis
            classified_patterns = self._classify_patterns(validated_patterns)
            
            # Perform pattern synthesis using the underlying engine
            synthesized_patterns = self.synthesis_engine.synthesize_patterns(
                validated_patterns, target_context, synthesis_count
            )
            
            # Post-process and validate synthesized patterns
            if enable_validation:
                synthesized_patterns = self._validate_synthesized_patterns(
                    synthesized_patterns, target_context, quality_threshold
                )
            
            # Track performance metrics
            processing_time = time.time() - processing_start
            self._record_processing_metrics(processing_time, len(synthesized_patterns) > 0)
            
            # Build comprehensive result
            result = {
                'synthesized_patterns': synthesized_patterns,
                'pattern_metadata': {
                    'source_pattern_count': len(source_patterns),
                    'validated_pattern_count': len(validated_patterns),
                    'synthesis_count': len(synthesized_patterns),
                    'pattern_classification': classified_patterns['classification_summary'],
                    'processing_time_ms': processing_time * 1000
                },
                'quality_metrics': self._compute_quality_metrics(synthesized_patterns),
                'synthesis_strategies_used': [
                    p.get('synthesis_strategy', 'unknown') for p in synthesized_patterns
                ],
                'processing_success': len(synthesized_patterns) > 0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern processing failed: {e}")
            return self._get_empty_processing_result("processing_error", str(e))
    
    def merge_worker_patterns(self, worker_patterns: Dict[str, List[Dict[str, Any]]], 
                            merge_strategy: str = 'quality_weighted') -> List[Dict[str, Any]]:
        """Merge patterns from multiple workers using specified strategy.
        
        Args:
            worker_patterns: Dict mapping worker_id to their patterns
            merge_strategy: Strategy for merging ('quality_weighted', 'diversity_focused', 'performance_based')
            
        Returns:
            List of merged patterns with worker attribution
        """
        if not worker_patterns:
            return []
        
        # Flatten all patterns with worker attribution
        all_patterns = []
        for worker_id, patterns in worker_patterns.items():
            for pattern in patterns:
                attributed_pattern = pattern.copy()
                attributed_pattern['source_worker'] = worker_id
                attributed_pattern['merge_timestamp'] = time.time()
                all_patterns.append(attributed_pattern)
        
        # Apply merge strategy
        if merge_strategy == 'quality_weighted':
            return self._merge_quality_weighted(all_patterns)
        elif merge_strategy == 'diversity_focused':
            return self._merge_diversity_focused(all_patterns)
        elif merge_strategy == 'performance_based':
            return self._merge_performance_based(all_patterns)
        else:
            # Default: simple concatenation
            return all_patterns
    
    def analyze_pattern_evolution(self, pattern_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how patterns have evolved over time across workers."""
        if not pattern_history:
            return {'status': 'no_data'}
        
        # Sort patterns by timestamp
        sorted_patterns = sorted(pattern_history, 
                               key=lambda p: p.get('timestamp', p.get('synthesis_timestamp', 0)))
        
        evolution_metrics = {
            'pattern_count_over_time': [],
            'quality_trend': [],
            'complexity_trend': [],
            'worker_contribution_trend': {},
            'synthesis_strategy_evolution': {}
        }
        
        # Track metrics over time windows
        window_size = max(1, len(sorted_patterns) // 10)  # 10 time windows
        
        for i in range(0, len(sorted_patterns), window_size):
            window_patterns = sorted_patterns[i:i+window_size]
            
            # Pattern count in window
            evolution_metrics['pattern_count_over_time'].append(len(window_patterns))
            
            # Average quality in window
            qualities = [p.get('success_rate', p.get('quality_score', 0.5)) for p in window_patterns]
            avg_quality = sum(qualities) / len(qualities) if qualities else 0.5
            evolution_metrics['quality_trend'].append(avg_quality)
            
            # Average complexity in window
            complexities = [self._compute_pattern_complexity(p) for p in window_patterns]
            avg_complexity = sum(complexities) / len(complexities) if complexities else 0.5
            evolution_metrics['complexity_trend'].append(avg_complexity)
            
            # Worker contributions in window
            for pattern in window_patterns:
                worker = pattern.get('discovered_by', pattern.get('source_worker', 'unknown'))
                if worker not in evolution_metrics['worker_contribution_trend']:
                    evolution_metrics['worker_contribution_trend'][worker] = []
                evolution_metrics['worker_contribution_trend'][worker].append(1)
            
            # Synthesis strategies used
            for pattern in window_patterns:
                strategy = pattern.get('synthesis_strategy', 'none')
                if strategy not in evolution_metrics['synthesis_strategy_evolution']:
                    evolution_metrics['synthesis_strategy_evolution'][strategy] = []
                evolution_metrics['synthesis_strategy_evolution'][strategy].append(1)
        
        return evolution_metrics
    
    def optimize_pattern_selection(self, available_patterns: List[Dict[str, Any]], 
                                 optimization_criteria: Dict[str, float]) -> List[Dict[str, Any]]:
        """Optimize pattern selection based on multiple criteria.
        
        Args:
            available_patterns: List of available patterns to choose from
            optimization_criteria: Dict with criteria weights (quality, diversity, performance, etc.)
            
        Returns:
            Optimized list of selected patterns
        """
        if not available_patterns:
            return []
        
        # Score each pattern based on optimization criteria
        pattern_scores = []
        
        for pattern in available_patterns:
            score = 0.0
            
            # Quality criterion
            quality_weight = optimization_criteria.get('quality', 0.4)
            quality_score = pattern.get('success_rate', pattern.get('quality_score', 0.5))
            score += quality_weight * quality_score
            
            # Diversity criterion
            diversity_weight = optimization_criteria.get('diversity', 0.3)
            diversity_score = self._compute_pattern_diversity_score(pattern, available_patterns)
            score += diversity_weight * diversity_score
            
            # Performance criterion
            performance_weight = optimization_criteria.get('performance', 0.2)
            performance_score = self._estimate_pattern_performance(pattern)
            score += performance_weight * performance_score
            
            # Recency criterion
            recency_weight = optimization_criteria.get('recency', 0.1)
            recency_score = self._compute_pattern_recency_score(pattern)
            score += recency_weight * recency_score
            
            pattern_scores.append((pattern, score))
        
        # Sort by score and return top patterns
        sorted_patterns = sorted(pattern_scores, key=lambda x: x[1], reverse=True)
        
        # Select top patterns (up to optimization limit)
        max_patterns = optimization_criteria.get('max_patterns', 10)
        selected_patterns = [pattern for pattern, score in sorted_patterns[:max_patterns]]
        
        return selected_patterns
    
    def _validate_source_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate source patterns for synthesis processing."""
        validated = []
        
        for pattern in patterns:
            # Check required fields
            if not pattern.get('pattern_sequence'):
                continue
                
            # Check pattern sequence validity
            sequence = pattern['pattern_sequence']
            if not isinstance(sequence, list) or len(sequence) == 0:
                continue
            
            # Check for valid commands
            valid_commands = set()
            for category, commands in self.pattern_categories.items():
                valid_commands.update(commands)
            
            invalid_commands = [cmd for cmd in sequence if cmd not in valid_commands and not isinstance(cmd, int)]
            if len(invalid_commands) > len(sequence) * 0.5:  # More than 50% invalid
                continue
            
            validated.append(pattern)
        
        return validated
    
    def _classify_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify patterns by type and complexity."""
        classification = {
            'by_category': {category: [] for category in self.pattern_categories.keys()},
            'by_complexity': {'simple': [], 'medium': [], 'complex': []},
            'by_length': {'short': [], 'medium': [], 'long': []},
            'classification_summary': {}
        }
        
        for pattern in patterns:
            sequence = pattern.get('pattern_sequence', [])
            
            # Classify by command category
            for category, commands in self.pattern_categories.items():
                if any(cmd in commands for cmd in sequence):
                    classification['by_category'][category].append(pattern)
            
            # Classify by complexity
            complexity = self._compute_pattern_complexity(pattern)
            if complexity < 0.3:
                classification['by_complexity']['simple'].append(pattern)
            elif complexity < 0.7:
                classification['by_complexity']['medium'].append(pattern)
            else:
                classification['by_complexity']['complex'].append(pattern)
            
            # Classify by length
            length = len(sequence)
            if length < 5:
                classification['by_length']['short'].append(pattern)
            elif length < 15:
                classification['by_length']['medium'].append(pattern)
            else:
                classification['by_length']['long'].append(pattern)
        
        # Create summary
        classification['classification_summary'] = {
            'total_patterns': len(patterns),
            'category_distribution': {cat: len(pats) for cat, pats in classification['by_category'].items()},
            'complexity_distribution': {comp: len(pats) for comp, pats in classification['by_complexity'].items()},
            'length_distribution': {length: len(pats) for length, pats in classification['by_length'].items()}
        }
        
        return classification
    
    def _validate_synthesized_patterns(self, patterns: List[Dict[str, Any]], 
                                     context: Dict[str, Any], 
                                     quality_threshold: float) -> List[Dict[str, Any]]:
        """Validate synthesized patterns against quality threshold."""
        validated = []
        
        for pattern in patterns:
            # Basic validation
            if not pattern.get('pattern_sequence'):
                continue
            
            # Estimate quality score if not present
            if 'estimated_quality' not in pattern:
                pattern['estimated_quality'] = self._estimate_pattern_quality(pattern, context)
            
            # Apply quality threshold
            quality_score = pattern.get('estimated_quality', pattern.get('success_rate', 0.5))
            if quality_score >= quality_threshold:
                validated.append(pattern)
        
        return validated
    
    def _compute_quality_metrics(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute quality metrics for synthesized patterns."""
        if not patterns:
            return {'status': 'no_patterns'}
        
        qualities = [p.get('estimated_quality', p.get('success_rate', 0.5)) for p in patterns]
        complexities = [self._compute_pattern_complexity(p) for p in patterns]
        
        return {
            'average_quality': sum(qualities) / len(qualities),
            'max_quality': max(qualities),
            'min_quality': min(qualities),
            'quality_variance': self._compute_variance(qualities),
            'average_complexity': sum(complexities) / len(complexities),
            'pattern_count': len(patterns),
            'synthesis_strategies': list(set(p.get('synthesis_strategy', 'unknown') for p in patterns))
        }
    
    def _compute_pattern_complexity(self, pattern: Dict[str, Any]) -> float:
        """Compute complexity score for a pattern."""
        sequence = pattern.get('pattern_sequence', [])
        if not sequence:
            return 0.0
        
        complexity_factors = 0.0
        
        # Length factor
        complexity_factors += min(1.0, len(sequence) / 20.0) * 0.3
        
        # Command diversity factor
        unique_commands = len(set(sequence))
        complexity_factors += min(1.0, unique_commands / 10.0) * 0.3
        
        # Control structure factor (IF, REPEAT, etc.)
        control_commands = sum(1 for cmd in sequence if cmd in self.pattern_categories.get('complex', []))
        complexity_factors += min(1.0, control_commands / 5.0) * 0.4
        
        return complexity_factors
    
    def _merge_quality_weighted(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge patterns using quality-weighted selection."""
        # Sort by quality and select top patterns
        quality_scored = [(p, p.get('success_rate', p.get('quality_score', 0.5))) for p in patterns]
        sorted_patterns = sorted(quality_scored, key=lambda x: x[1], reverse=True)
        
        # Take top 50% by quality
        top_count = max(1, len(patterns) // 2)
        return [pattern for pattern, quality in sorted_patterns[:top_count]]
    
    def _merge_diversity_focused(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge patterns focusing on diversity."""
        if not patterns:
            return []
        
        selected = [patterns[0]]  # Start with first pattern
        
        for pattern in patterns[1:]:
            # Check diversity against already selected patterns
            is_diverse = True
            for selected_pattern in selected:
                similarity = self._compute_pattern_similarity(pattern, selected_pattern)
                if similarity > 0.8:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(pattern)
        
        return selected
    
    def _merge_performance_based(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge patterns based on performance estimates."""
        performance_scored = [(p, self._estimate_pattern_performance(p)) for p in patterns]
        sorted_patterns = sorted(performance_scored, key=lambda x: x[1], reverse=True)
        
        # Take top 60% by performance
        top_count = max(1, int(len(patterns) * 0.6))
        return [pattern for pattern, performance in sorted_patterns[:top_count]]
    
    def _compute_pattern_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Compute similarity between two patterns."""
        seq1 = pattern1.get('pattern_sequence', [])
        seq2 = pattern2.get('pattern_sequence', [])
        
        if not seq1 or not seq2:
            return 0.0
        
        # Simple similarity based on common commands
        set1 = set(seq1)
        set2 = set(seq2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / max(len(union), 1)
    
    def _estimate_pattern_quality(self, pattern: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Estimate quality score for a pattern in context."""
        base_quality = 0.5
        
        # Factor in pattern complexity
        complexity = self._compute_pattern_complexity(pattern)
        base_quality += complexity * 0.2
        
        # Factor in context relevance (simplified)
        sequence = pattern.get('pattern_sequence', [])
        context_relevance = 0.0
        
        if 'location' in context:
            # Simple heuristic: movement commands more relevant for location-based tasks
            movement_ratio = sum(1 for cmd in sequence if cmd in self.pattern_categories.get('movement', [])) / max(len(sequence), 1)
            context_relevance += movement_ratio * 0.3
        
        return min(1.0, base_quality + context_relevance)
    
    def _estimate_pattern_performance(self, pattern: Dict[str, Any]) -> float:
        """Estimate performance score for pattern."""
        # Simple heuristic based on pattern characteristics
        sequence = pattern.get('pattern_sequence', [])
        if not sequence:
            return 0.0
        
        # Shorter patterns generally perform better
        length_score = max(0.0, 1.0 - len(sequence) / 30.0)
        
        # Patterns with fewer timing commands may be faster
        timing_commands = sum(1 for cmd in sequence if isinstance(cmd, int) or cmd in ['WAIT'])
        timing_penalty = timing_commands / max(len(sequence), 1) * 0.3
        
        return max(0.1, length_score - timing_penalty)
    
    def _compute_pattern_diversity_score(self, pattern: Dict[str, Any], all_patterns: List[Dict[str, Any]]) -> float:
        """Compute diversity score for pattern against all patterns."""
        if len(all_patterns) <= 1:
            return 1.0
        
        similarities = [self._compute_pattern_similarity(pattern, other) 
                       for other in all_patterns if other != pattern]
        
        # Diversity is inverse of average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        return 1.0 - avg_similarity
    
    def _compute_pattern_recency_score(self, pattern: Dict[str, Any]) -> float:
        """Compute recency score for pattern."""
        timestamp = pattern.get('synthesis_timestamp', pattern.get('timestamp', 0))
        if timestamp == 0:
            return 0.5  # Neutral for unknown timestamps
        
        current_time = time.time()
        age_hours = (current_time - timestamp) / 3600.0
        
        # Patterns from last 24 hours get full score, older patterns decay
        return max(0.1, 1.0 - age_hours / 24.0)
    
    def _compute_variance(self, values: List[float]) -> float:
        """Compute variance of a list of values."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]
        return sum(squared_diffs) / len(squared_diffs)
    
    def _record_processing_metrics(self, processing_time: float, success: bool) -> None:
        """Record processing metrics for monitoring."""
        self.processing_times.append(processing_time)
        self.processed_pattern_count += 1
        
        # Update success rate
        success_value = 1.0 if success else 0.0
        alpha = 0.1
        self.synthesis_success_rate = alpha * success_value + (1 - alpha) * self.synthesis_success_rate
        
        # Keep only recent processing times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-50:]
    
    def _get_empty_processing_result(self, reason: str, error_details: Optional[str] = None) -> Dict[str, Any]:
        """Get empty processing result with error information."""
        return {
            'synthesized_patterns': [],
            'pattern_metadata': {
                'source_pattern_count': 0,
                'validated_pattern_count': 0,
                'synthesis_count': 0,
                'pattern_classification': {},
                'processing_time_ms': 1.0
            },
            'quality_metrics': {'status': 'no_patterns'},
            'synthesis_strategies_used': [],
            'processing_success': False,
            'error_reason': reason,
            'error_details': error_details
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get pattern processing performance statistics."""
        if not self.processing_times:
            return {'status': 'no_data'}
        
        avg_time_ms = (sum(self.processing_times) / len(self.processing_times)) * 1000
        max_time_ms = max(self.processing_times) * 1000
        
        return {
            'average_processing_time_ms': avg_time_ms,
            'max_processing_time_ms': max_time_ms,
            'total_patterns_processed': self.processed_pattern_count,
            'synthesis_success_rate': self.synthesis_success_rate,
            'processing_success': True
        }


class ParallelExecutionCoordinator:
    """Coordinates parallel execution across workers with performance monitoring.
    
    Orchestrates concurrent script development tasks across multiple workers,
    ensuring optimal resource utilization while maintaining performance targets.
    
    Performance Target: <100ms coordination overhead
    """
    
    def __init__(self):
        self.active_executions: Dict[str, Dict[str, Any]] = {}  # execution_id -> execution info
        self.worker_load_tracker: Dict[str, float] = {}  # worker_id -> current load score
        self.coordination_times: List[float] = []
        self.execution_queue = queue.PriorityQueue()  # (priority, execution_request)
        self.coordination_lock = threading.Lock()
        
        # Performance monitoring
        self.coordination_metrics = {
            'total_coordinated_executions': 0,
            'average_coordination_time_ms': 0.0,
            'concurrent_execution_peak': 0,
            'coordination_success_rate': 0.0
        }
        
        # Execution coordination strategies
        self.coordination_strategies = {
            'load_balanced': self._coordinate_load_balanced,
            'priority_based': self._coordinate_priority_based,
            'affinity_aware': self._coordinate_affinity_aware,
            'performance_optimized': self._coordinate_performance_optimized
        }
        
        self.default_coordination_strategy = 'load_balanced'
    
    def coordinate_parallel_execution(self, execution_requests: List[Dict[str, Any]], 
                                    worker_pool: 'SonnetWorkerPool',
                                    coordination_strategy: Optional[str] = None) -> Dict[str, Any]:
        """Coordinate parallel execution of multiple requests across workers.
        
        Args:
            execution_requests: List of execution requests with task details
            worker_pool: SonnetWorkerPool instance for worker access
            coordination_strategy: Strategy for coordinating execution ('load_balanced', 'priority_based', etc.)
            
        Returns:
            Dictionary with coordination results and execution tracking info
        """
        coordination_start = time.time()
        
        try:
            with self.coordination_lock:
                # Select coordination strategy
                strategy = coordination_strategy or self.default_coordination_strategy
                strategy_func = self.coordination_strategies.get(strategy, self._coordinate_load_balanced)
                
                # Execute coordination strategy
                coordination_result = strategy_func(execution_requests, worker_pool)
                
                # Track coordination time
                coordination_time = time.time() - coordination_start
                self._record_coordination_time(coordination_time)
                
                # Update metrics
                self.coordination_metrics['total_coordinated_executions'] += len(execution_requests)
                
                # Add coordination metadata
                coordination_result.update({
                    'coordination_strategy': strategy,
                    'coordination_time_ms': coordination_time * 1000,
                    'total_requests': len(execution_requests),
                    'coordination_timestamp': time.time()
                })
                
                return coordination_result
                
        except Exception as e:
            logger.error(f"Parallel execution coordination failed: {e}")
            return self._get_fallback_coordination(execution_requests, worker_pool)
    
    def monitor_execution_progress(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Monitor progress of a coordinated execution."""
        if execution_id not in self.active_executions:
            return None
        
        execution_info = self.active_executions[execution_id]
        current_time = time.time()
        
        # Calculate progress metrics
        start_time = execution_info.get('start_time', current_time)
        elapsed_time = current_time - start_time
        estimated_duration = execution_info.get('estimated_duration_ms', 5000) / 1000.0
        progress_percentage = min(100.0, (elapsed_time / estimated_duration) * 100)
        
        return {
            'execution_id': execution_id,
            'worker_id': execution_info.get('worker_id'),
            'status': execution_info.get('status', 'unknown'),
            'progress_percentage': progress_percentage,
            'elapsed_time_ms': elapsed_time * 1000,
            'estimated_completion_ms': max(0, (estimated_duration - elapsed_time) * 1000)
        }
    
    def handle_execution_completion(self, execution_id: str, result: Dict[str, Any]) -> None:
        """Handle completion of a coordinated execution."""
        if execution_id not in self.active_executions:
            logger.warning(f"Completion reported for unknown execution {execution_id}")
            return
        
        execution_info = self.active_executions[execution_id]
        worker_id = execution_info.get('worker_id')
        
        # Update worker load tracking
        if worker_id in self.worker_load_tracker:
            self.worker_load_tracker[worker_id] = max(0.0, self.worker_load_tracker[worker_id] - 0.3)
        
        # Record completion metrics
        start_time = execution_info.get('start_time', time.time())
        completion_time = time.time() - start_time
        success = result.get('success', False)
        
        # Update success rate
        total_executions = self.coordination_metrics['total_coordinated_executions']
        if total_executions > 0:
            current_success_rate = self.coordination_metrics['coordination_success_rate']
            new_success_value = 1.0 if success else 0.0
            alpha = 0.1  # Learning rate
            self.coordination_metrics['coordination_success_rate'] = (
                alpha * new_success_value + (1 - alpha) * current_success_rate
            )
        
        # Clean up execution tracking
        del self.active_executions[execution_id]
        
        logger.debug(f"Execution {execution_id} completed in {completion_time*1000:.1f}ms, success={success}")
    
    def get_optimal_worker_assignment(self, task_requirements: Dict[str, Any], 
                                    available_workers: List[str]) -> Optional[str]:
        """Get optimal worker for task assignment based on current coordination state."""
        if not available_workers:
            return None
        
        # Calculate worker scores based on multiple factors
        worker_scores = {}
        
        for worker_id in available_workers:
            score = 1.0
            
            # Factor 1: Current load (lower is better)
            current_load = self.worker_load_tracker.get(worker_id, 0.0)
            load_score = max(0.0, 1.0 - current_load)
            score *= load_score * 0.4
            
            # Factor 2: Task affinity (based on requirements)
            affinity_score = self._compute_task_affinity(worker_id, task_requirements)
            score *= affinity_score * 0.3
            
            # Factor 3: Recent performance
            performance_score = self._get_worker_recent_performance(worker_id)
            score *= performance_score * 0.3
            
            worker_scores[worker_id] = score
        
        # Select worker with highest score
        optimal_worker = max(worker_scores.keys(), key=lambda w: worker_scores[w])
        return optimal_worker
    
    def update_worker_load(self, worker_id: str, load_delta: float) -> None:
        """Update worker load tracking for coordination decisions."""
        if worker_id not in self.worker_load_tracker:
            self.worker_load_tracker[worker_id] = 0.0
        
        # Update load with bounds checking
        new_load = self.worker_load_tracker[worker_id] + load_delta
        self.worker_load_tracker[worker_id] = max(0.0, min(1.0, new_load))
    
    def _coordinate_load_balanced(self, requests: List[Dict[str, Any]], 
                                worker_pool: 'SonnetWorkerPool') -> Dict[str, Any]:
        """Coordinate using load balancing strategy."""
        assigned_executions = []
        failed_assignments = []
        
        for request in requests:
            available_workers = [wid for wid, info in worker_pool.workers.items() 
                               if info.get('healthy', False) and info.get('status') == 'ready']
            
            if not available_workers:
                failed_assignments.append({
                    'request': request,
                    'reason': 'no_available_workers'
                })
                continue
            
            # Select worker with lowest load
            optimal_worker = min(available_workers, 
                               key=lambda w: self.worker_load_tracker.get(w, 0.0))
            
            # Create execution tracking
            execution_id = f"exec_{uuid.uuid4().hex[:8]}"
            execution_info = {
                'execution_id': execution_id,
                'worker_id': optimal_worker,
                'request': request,
                'start_time': time.time(),
                'status': 'assigned',
                'estimated_duration_ms': request.get('estimated_duration_ms', 5000)
            }
            
            self.active_executions[execution_id] = execution_info
            self.update_worker_load(optimal_worker, 0.3)  # Increase load
            
            assigned_executions.append(execution_info)
        
        return {
            'assigned_executions': assigned_executions,
            'failed_assignments': failed_assignments,
            'strategy': 'load_balanced'
        }
    
    def _coordinate_priority_based(self, requests: List[Dict[str, Any]], 
                                 worker_pool: 'SonnetWorkerPool') -> Dict[str, Any]:
        """Coordinate using priority-based strategy."""
        # Sort requests by priority
        prioritized_requests = sorted(requests, 
                                    key=lambda r: r.get('priority', 0.5), 
                                    reverse=True)
        
        return self._coordinate_load_balanced(prioritized_requests, worker_pool)
    
    def _coordinate_affinity_aware(self, requests: List[Dict[str, Any]], 
                                 worker_pool: 'SonnetWorkerPool') -> Dict[str, Any]:
        """Coordinate using task-worker affinity strategy."""
        assigned_executions = []
        failed_assignments = []
        
        for request in requests:
            available_workers = [wid for wid, info in worker_pool.workers.items() 
                               if info.get('healthy', False) and info.get('status') == 'ready']
            
            if not available_workers:
                failed_assignments.append({
                    'request': request,
                    'reason': 'no_available_workers'
                })
                continue
            
            # Select worker based on task affinity
            optimal_worker = self.get_optimal_worker_assignment(
                request.get('task_requirements', {}), 
                available_workers
            )
            
            if not optimal_worker:
                failed_assignments.append({
                    'request': request,
                    'reason': 'no_suitable_worker'
                })
                continue
            
            # Create execution tracking
            execution_id = f"exec_{uuid.uuid4().hex[:8]}"
            execution_info = {
                'execution_id': execution_id,
                'worker_id': optimal_worker,
                'request': request,
                'start_time': time.time(),
                'status': 'assigned',
                'estimated_duration_ms': request.get('estimated_duration_ms', 5000)
            }
            
            self.active_executions[execution_id] = execution_info
            self.update_worker_load(optimal_worker, 0.3)
            
            assigned_executions.append(execution_info)
        
        return {
            'assigned_executions': assigned_executions,
            'failed_assignments': failed_assignments,
            'strategy': 'affinity_aware'
        }
    
    def _coordinate_performance_optimized(self, requests: List[Dict[str, Any]], 
                                        worker_pool: 'SonnetWorkerPool') -> Dict[str, Any]:
        """Coordinate using performance optimization strategy."""
        # Group requests by complexity/requirements
        high_priority = [r for r in requests if r.get('priority', 0.5) > 0.7]
        normal_priority = [r for r in requests if 0.3 <= r.get('priority', 0.5) <= 0.7]
        low_priority = [r for r in requests if r.get('priority', 0.5) < 0.3]
        
        # Process high priority first with best workers
        assigned_executions = []
        failed_assignments = []
        
        for priority_group in [high_priority, normal_priority, low_priority]:
            result = self._coordinate_affinity_aware(priority_group, worker_pool)
            assigned_executions.extend(result['assigned_executions'])
            failed_assignments.extend(result['failed_assignments'])
        
        return {
            'assigned_executions': assigned_executions,
            'failed_assignments': failed_assignments,
            'strategy': 'performance_optimized'
        }
    
    def _get_fallback_coordination(self, requests: List[Dict[str, Any]], 
                                 worker_pool: 'SonnetWorkerPool') -> Dict[str, Any]:
        """Provide fallback coordination when main coordination fails."""
        return {
            'assigned_executions': [],
            'failed_assignments': [{'request': r, 'reason': 'coordination_failure'} for r in requests],
            'strategy': 'fallback',
            'coordination_time_ms': 1.0,
            'total_requests': len(requests)
        }
    
    def _compute_task_affinity(self, worker_id: str, task_requirements: Dict[str, Any]) -> float:
        """Compute affinity between worker and task requirements."""
        # Simple affinity computation based on task characteristics
        base_affinity = 0.5
        
        # Check for complexity match
        task_complexity = task_requirements.get('complexity', 'medium')
        if task_complexity == 'high':
            # Prefer workers that aren't overloaded for complex tasks
            current_load = self.worker_load_tracker.get(worker_id, 0.0)
            base_affinity += (1.0 - current_load) * 0.3
        
        # Check for performance requirements
        if 'performance_target' in task_requirements:
            target = task_requirements['performance_target']
            if target == 'speed':
                base_affinity += 0.2  # All workers are equally fast for now
            elif target == 'quality':
                base_affinity += 0.1  # Slight preference for less loaded workers
        
        return min(1.0, base_affinity)
    
    def _get_worker_recent_performance(self, worker_id: str) -> float:
        """Get recent performance score for worker (simplified)."""
        # For now, return based on current load (less loaded = better recent performance assumption)
        current_load = self.worker_load_tracker.get(worker_id, 0.0)
        return max(0.3, 1.0 - current_load * 0.5)
    
    def _record_coordination_time(self, coordination_time: float) -> None:
        """Record coordination time for performance monitoring."""
        self.coordination_times.append(coordination_time)
        
        # Keep only recent times
        if len(self.coordination_times) > 100:
            self.coordination_times = self.coordination_times[-50:]
        
        # Update average coordination time
        avg_time_ms = (sum(self.coordination_times) / len(self.coordination_times)) * 1000
        self.coordination_metrics['average_coordination_time_ms'] = avg_time_ms
        
        # Log warning if coordination is slow
        if coordination_time > 0.1:  # 100ms target
            logger.warning(f"Coordination took {coordination_time*1000:.1f}ms, exceeds 100ms target")
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination performance statistics."""
        if not self.coordination_times:
            return {'status': 'no_data'}
        
        avg_time_ms = (sum(self.coordination_times) / len(self.coordination_times)) * 1000
        max_time_ms = max(self.coordination_times) * 1000
        current_executions = len(self.active_executions)
        
        # Update peak concurrent executions
        if current_executions > self.coordination_metrics['concurrent_execution_peak']:
            self.coordination_metrics['concurrent_execution_peak'] = current_executions
        
        return {
            'average_coordination_time_ms': avg_time_ms,
            'max_coordination_time_ms': max_time_ms,
            'performance_target_met': avg_time_ms < 100.0,
            'active_executions': current_executions,
            'worker_load_status': dict(self.worker_load_tracker),
            'coordination_metrics': self.coordination_metrics.copy()
        }


class AdaptiveQualityThresholds:
    """Dynamic quality threshold system that adapts based on worker performance history."""
    
    def __init__(self):
        self.worker_performance_history: Dict[str, List[float]] = {}
        self.global_performance_trend = 0.5  # Start at middle
        self.threshold_adaptation_rate = 0.1
        self.min_threshold = 0.3
        self.max_threshold = 0.95
        
        # Base thresholds from config
        self.base_success_threshold = CONFIG.script_development['SUCCESS_QUALITY_THRESHOLD']
        self.base_acceptable_threshold = CONFIG.script_development['ACCEPTABLE_QUALITY_THRESHOLD']
        
        # Adaptive thresholds
        self.current_success_threshold = self.base_success_threshold
        self.current_acceptable_threshold = self.base_acceptable_threshold
    
    def update_worker_performance(self, worker_id: str, quality_score: float):
        """Update performance history for specific worker."""
        if worker_id not in self.worker_performance_history:
            self.worker_performance_history[worker_id] = []
            
        self.worker_performance_history[worker_id].append(quality_score)
        
        # Keep only recent history (last 20 attempts)
        if len(self.worker_performance_history[worker_id]) > 20:
            self.worker_performance_history[worker_id] = self.worker_performance_history[worker_id][-20:]
        
        self._update_global_trend()
        self._adapt_thresholds()
    
    def _update_global_trend(self):
        """Update global performance trend based on all workers."""
        if not self.worker_performance_history:
            return
            
        # Collect recent performance scores from all workers
        recent_scores = []
        for worker_history in self.worker_performance_history.values():
            if worker_history:
                # Take last 5 scores from each worker
                recent_scores.extend(worker_history[-5:])
        
        if len(recent_scores) < 3:
            return
            
        # Calculate moving average trend
        current_avg = sum(recent_scores[-10:]) / min(10, len(recent_scores))
        historical_avg = sum(recent_scores) / len(recent_scores)
        
        # Update trend with smoothing
        trend_indicator = current_avg / max(historical_avg, 0.1)  # Avoid division by zero
        self.global_performance_trend = (
            0.8 * self.global_performance_trend + 
            0.2 * min(1.0, max(0.0, trend_indicator))
        )
    
    def _adapt_thresholds(self):
        """Adapt quality thresholds based on performance trends."""
        if self.global_performance_trend > 0.8:
            # Performance is good, can raise thresholds
            threshold_adjustment = self.threshold_adaptation_rate * 0.1
            self.current_success_threshold = min(
                self.max_threshold,
                self.current_success_threshold + threshold_adjustment
            )
            self.current_acceptable_threshold = min(
                self.current_success_threshold - 0.1,
                self.current_acceptable_threshold + threshold_adjustment
            )
        elif self.global_performance_trend < 0.4:
            # Performance is poor, lower thresholds
            threshold_adjustment = self.threshold_adaptation_rate * 0.1
            self.current_acceptable_threshold = max(
                self.min_threshold,
                self.current_acceptable_threshold - threshold_adjustment
            )
            self.current_success_threshold = max(
                self.current_acceptable_threshold + 0.1,
                self.current_success_threshold - threshold_adjustment
            )
    
    def get_adaptive_thresholds(self, worker_id: str) -> Dict[str, float]:
        """Get adaptive thresholds for specific worker."""
        worker_performance = self.get_worker_performance_score(worker_id)
        
        # Adjust thresholds based on individual worker capability
        if worker_performance > 0.8:
            # High-performing worker, use higher thresholds
            success_threshold = min(0.95, self.current_success_threshold + 0.1)
            acceptable_threshold = min(success_threshold - 0.1, self.current_acceptable_threshold + 0.05)
        elif worker_performance < 0.4:
            # Struggling worker, use lower thresholds
            acceptable_threshold = max(0.3, self.current_acceptable_threshold - 0.1)
            success_threshold = max(acceptable_threshold + 0.1, self.current_success_threshold - 0.05)
        else:
            # Average worker, use current adaptive thresholds
            success_threshold = self.current_success_threshold
            acceptable_threshold = self.current_acceptable_threshold
        
        return {
            'success_threshold': success_threshold,
            'acceptable_threshold': acceptable_threshold,
            'worker_performance': worker_performance,
            'global_trend': self.global_performance_trend
        }
    
    def get_worker_performance_score(self, worker_id: str) -> float:
        """Get performance score for specific worker."""
        if worker_id not in self.worker_performance_history:
            return 0.5  # Default average performance
            
        history = self.worker_performance_history[worker_id]
        if not history:
            return 0.5
            
        # Calculate recent performance with more weight on recent scores
        if len(history) >= 5:
            recent_scores = history[-5:]
            weights = [0.4, 0.3, 0.2, 0.08, 0.02]  # More weight on recent
            weighted_score = sum(score * weight for score, weight in zip(recent_scores, weights))
        else:
            weighted_score = sum(history) / len(history)
            
        return weighted_score


class SonnetWorkerPool:
    """
    High-level abstraction for managing a pool of Claude Sonnet workers.

    This class provides clean task assignment and worker lifecycle management
    by wrapping ClaudeCodeManager's tactical processes with pool semantics.
    It focuses on operational concerns like load balancing, health monitoring,
    and worker status tracking.
    """

    def __init__(self, claude_manager: ClaudeCodeManager):
        """
        Initialize SonnetWorkerPool with Claude manager dependency.

        Args:
            claude_manager: ClaudeCodeManager instance providing tactical processes
        """
        self.claude_manager = claude_manager
        self.workers: dict[str, dict[str, Any]] = {}
        self._current_assignment_index = 0
        self._initialized = False

        # Task queueing system
        self.task_queue: queue.Queue = queue.Queue()
        self.worker_assignments: dict[str, dict[str, Any]] = {}  # worker_id -> task info
        self._queue_lock = threading.Lock()

        # MCP integration and script development (lazy initialization for testing)
        self._query_builder = None
        self._script_compiler = None
        self._quality_assessor = None
        self._pattern_refiner = None
        self._semantic_pattern_engine = None
        
        # Enhanced AI systems
        self._reinforcement_learning_engine = None
        self._adaptive_quality_thresholds = None
        self._cross_worker_pattern_synthesis = None
        self._performance_monitor = None
        self._multi_objective_optimizer = None
        self._genetic_population = None
        
        # Core script development engine components
        self._experiment_selector = None
        self._parallel_execution_coordinator = None
        self._pattern_processor = None
        self._worker_distributor = None
        
        # Performance tracking
        self.development_metrics = {
            "total_scripts_developed": 0,
            "average_development_time_ms": 0.0,
            "success_rate": 0.0,
            "total_refinements": 0
        }

        logger.info("SonnetWorkerPool initialized with enhanced AI systems and task queueing")
    
    @property
    def query_builder(self) -> QueryBuilder:
        """Lazy initialization of QueryBuilder for testing compatibility."""
        if self._query_builder is None:
            self._query_builder = QueryBuilder()
        return self._query_builder
    
    @property  
    def script_compiler(self) -> ScriptCompiler:
        """Lazy initialization of ScriptCompiler for testing compatibility."""
        if self._script_compiler is None:
            self._script_compiler = ScriptCompiler()
        return self._script_compiler
    
    @property
    def quality_assessor(self) -> ScriptQualityAssessor:
        """Lazy initialization of ScriptQualityAssessor for testing compatibility."""
        if self._quality_assessor is None:
            self._quality_assessor = ScriptQualityAssessor(self.script_compiler)
        return self._quality_assessor
    
    @quality_assessor.setter
    def quality_assessor(self, value: ScriptQualityAssessor):
        """Setter for testing compatibility."""
        self._quality_assessor = value
    
    @quality_assessor.deleter
    def quality_assessor(self):
        """Deleter for testing compatibility."""
        self._quality_assessor = None
    
    @property
    def pattern_refiner(self) -> PatternRefiner:
        """Lazy initialization of PatternRefiner for testing compatibility."""
        if self._pattern_refiner is None:
            self._pattern_refiner = PatternRefiner(self.quality_assessor)
        return self._pattern_refiner
    
    @property
    def semantic_pattern_engine(self) -> SemanticPatternEngine:
        """Lazy initialization of SemanticPatternEngine for testing compatibility."""
        if self._semantic_pattern_engine is None:
            self._semantic_pattern_engine = SemanticPatternEngine()
        return self._semantic_pattern_engine
    
    @property
    def reinforcement_learning_engine(self) -> ReinforcementLearningEngine:
        """Lazy initialization of ReinforcementLearningEngine for testing compatibility."""
        if self._reinforcement_learning_engine is None:
            self._reinforcement_learning_engine = ReinforcementLearningEngine()
        return self._reinforcement_learning_engine
    
    @property
    def adaptive_quality_thresholds(self) -> AdaptiveQualityThresholds:
        """Lazy initialization of AdaptiveQualityThresholds for testing compatibility."""
        if self._adaptive_quality_thresholds is None:
            self._adaptive_quality_thresholds = AdaptiveQualityThresholds()
        return self._adaptive_quality_thresholds
    
    @property
    def cross_worker_pattern_synthesis(self) -> CrossWorkerPatternSynthesis:
        """Lazy initialization of CrossWorkerPatternSynthesis for testing compatibility."""
        if self._cross_worker_pattern_synthesis is None:
            self._cross_worker_pattern_synthesis = CrossWorkerPatternSynthesis()
        return self._cross_worker_pattern_synthesis
    
    @property 
    def performance_monitor(self) -> RealTimePerformanceMonitor:
        """Lazy initialization of RealTimePerformanceMonitor for testing compatibility."""
        if self._performance_monitor is None:
            self._performance_monitor = RealTimePerformanceMonitor()
        return self._performance_monitor
        
    @property
    def multi_objective_optimizer(self) -> MultiObjectiveOptimizer:
        """Lazy initialization of MultiObjectiveOptimizer for testing compatibility."""
        if self._multi_objective_optimizer is None:
            self._multi_objective_optimizer = MultiObjectiveOptimizer()
        return self._multi_objective_optimizer
    
    @property
    def genetic_population(self) -> GeneticPopulation:
        """Lazy initialization of GeneticPopulation for testing compatibility."""
        if self._genetic_population is None:
            self._genetic_population = GeneticPopulation(
                population_size=CONFIG.script_development['GENETIC_POPULATION_SIZE'],
                elite_size=CONFIG.script_development['GENETIC_ELITE_SIZE']
            )
        return self._genetic_population
    
    @genetic_population.setter
    def genetic_population(self, value: GeneticPopulation):
        """Setter for testing compatibility."""
        self._genetic_population = value
    
    @genetic_population.deleter
    def genetic_population(self):
        """Deleter for testing compatibility."""
        self._genetic_population = None
    
    @property
    def experiment_selector(self) -> 'ExperimentSelector':
        """Lazy initialization of ExperimentSelector for testing compatibility."""
        if self._experiment_selector is None:
            self._experiment_selector = ExperimentSelector()
        return self._experiment_selector
    
    @property
    def parallel_execution_coordinator(self) -> 'ParallelExecutionCoordinator':
        """Lazy initialization of ParallelExecutionCoordinator for testing compatibility."""
        if self._parallel_execution_coordinator is None:
            self._parallel_execution_coordinator = ParallelExecutionCoordinator()
        return self._parallel_execution_coordinator
    
    @property
    def pattern_processor(self) -> 'PatternProcessor':
        """Lazy initialization of PatternProcessor for testing compatibility."""
        if self._pattern_processor is None:
            self._pattern_processor = PatternProcessor(self.cross_worker_pattern_synthesis)
        return self._pattern_processor
    
    @property
    def worker_distributor(self) -> 'WorkerDistributor':
        """Lazy initialization of WorkerDistributor for testing compatibility."""
        if self._worker_distributor is None:
            self._worker_distributor = WorkerDistributor()
        return self._worker_distributor

    def initialize(self, worker_count: int = None) -> bool:
        """
        Initialize worker pool with specified number of workers.

        This method:
        1. Ensures ClaudeCodeManager processes are started
        2. Gets tactical (Sonnet) processes from the manager
        3. Assigns unique worker IDs to healthy processes
        4. Verifies worker health and connectivity
        5. Sets up worker status tracking

        Args:
            worker_count: Number of workers to initialize (default from config)

        Returns:
            True if initialization successful, False otherwise

        Performance Requirements:
            - Total initialization time: <500ms
            - Health verification: <10ms per worker
        """
        if worker_count is None:
            worker_count = CONFIG.worker.DEFAULT_WORKER_COUNT
            
        start_time = time.time()
        logger.info(f"Initializing SonnetWorkerPool with {worker_count} workers")

        try:
            # Ensure ClaudeCodeManager processes are started
            if not self.claude_manager.start_all_processes():
                logger.error("Failed to start ClaudeCodeManager processes")
                return False

            # Get tactical processes from ClaudeCodeManager
            tactical_processes = self.claude_manager.get_tactical_processes()

            if len(tactical_processes) < worker_count:
                logger.warning(
                    f"Requested {worker_count} workers but only {len(tactical_processes)} available"
                )

            # Initialize workers from available tactical processes
            healthy_worker_count = 0
            for _i, process in enumerate(tactical_processes[:worker_count]):
                # Verify worker health during initialization
                health_check_start = time.time()
                if not process.health_check():
                    logger.warning(f"Process {process.process_id} failed health check, skipping")
                    continue

                health_check_time = time.time() - health_check_start
                if health_check_time > CONFIG.performance.WORKER_HEALTH_CHECK_TIMEOUT:
                    logger.warning(
                        f"Health check took {health_check_time:.3f}s, exceeds {CONFIG.performance.WORKER_HEALTH_CHECK_TIMEOUT*1000:.0f}ms target"
                    )

                # Assign unique worker ID
                worker_id = f"{CONFIG.worker.WORKER_ID_PREFIX}_{uuid.uuid4().hex[:CONFIG.performance.UUID_SHORT_LENGTH]}"

                # Create worker registration
                worker_info = {
                    "worker_id": worker_id,
                    "process_id": process.process_id,
                    "process": process,
                    "status": "ready",
                    "healthy": True,
                    "task_count": 0,
                    "last_health_check": time.time(),
                    "created_at": time.time(),
                }

                self.workers[worker_id] = worker_info
                healthy_worker_count += 1

                logger.debug(f"Registered worker {worker_id} with process {process.process_id}")

            # Validate initialization success
            self._initialized = healthy_worker_count > 0

            initialization_time = time.time() - start_time
            logger.info(
                f"Initialized {healthy_worker_count}/{worker_count} workers in {initialization_time:.3f}s"
            )

            # Check performance requirement
            if initialization_time > CONFIG.performance.WORKER_POOL_INIT_TIMEOUT:
                logger.warning(
                    f"Initialization took {initialization_time:.3f}s, exceeds {CONFIG.performance.WORKER_POOL_INIT_TIMEOUT*1000:.0f}ms target"
                )

            return self._initialized

        except Exception as e:
            logger.error(f"Worker pool initialization failed: {e}")
            return False

    def get_worker_status(self, worker_id: str) -> dict[str, Any] | None:
        """
        Get detailed status information for a specific worker.

        Args:
            worker_id: Unique worker identifier

        Returns:
            Dictionary with worker status or None if worker not found

        Status includes:
            - healthy: Boolean indicating worker health
            - process_id: Associated ClaudeProcess ID
            - worker_id: Unique worker identifier
            - status: Current worker status string
            - task_count: Number of tasks processed
            - last_health_check: Timestamp of last health verification
            - created_at: Worker creation timestamp
        """
        if worker_id not in self.workers:
            logger.debug(f"Worker {worker_id} not found")
            return None

        worker_info = self.workers[worker_id]
        process = worker_info["process"]

        # Refresh health status
        current_health = process.is_healthy()
        worker_info["healthy"] = current_health
        worker_info["last_health_check"] = time.time()

        # Return comprehensive status
        status = {
            "healthy": current_health,
            "process_id": worker_info["process_id"],
            "worker_id": worker_id,
            "status": worker_info["status"],
            "task_count": worker_info["task_count"],
            "last_health_check": worker_info["last_health_check"],
            "created_at": worker_info["created_at"],
        }

        return status

    def restart_worker(self, worker_id: str) -> bool:
        """
        Restart a specific worker conversation.

        This method restarts the underlying ClaudeProcess and updates
        worker tracking information accordingly.

        Args:
            worker_id: Unique worker identifier

        Returns:
            True if restart successful, False otherwise
        """
        if worker_id not in self.workers:
            logger.warning(f"Cannot restart unknown worker {worker_id}")
            return False

        worker_info = self.workers[worker_id]
        process = worker_info["process"]

        try:
            logger.info(f"Restarting worker {worker_id}")
            restart_success = process.restart()

            if restart_success:
                # Update worker tracking
                worker_info["status"] = "ready"
                worker_info["task_count"] = 0  # Reset task count after restart
                worker_info["last_health_check"] = time.time()
                worker_info["healthy"] = True

                logger.info(f"Worker {worker_id} restarted successfully")
                return True
            else:
                logger.error(f"Failed to restart worker {worker_id}")
                worker_info["healthy"] = False
                worker_info["status"] = "failed"
                return False

        except Exception as e:
            logger.error(f"Exception during worker restart: {e}")
            worker_info["healthy"] = False
            worker_info["status"] = "error"
            return False

    def assign_task(self, task: dict[str, Any]) -> str:
        """
        Assign a task to an available worker or queue it if all workers are busy.

        This method finds a healthy, available worker and assigns the task immediately,
        or queues the task if no workers are available. Returns a task ID for tracking.
        Uses simple round-robin selection for load balancing.

        Args:
            task: Task dictionary with objective and context

        Returns:
            Task ID for tracking (either immediate assignment or queued)

        Performance Requirements:
            - Task assignment: <50ms (including queueing)
        """
        if not self._initialized or not self.workers:
            raise ValueError("Worker pool not initialized, cannot assign task")

        start_time = time.time()
        task_id = f"{CONFIG.worker.TASK_ID_PREFIX}_{uuid.uuid4().hex[:CONFIG.performance.UUID_SHORT_LENGTH]}"

        # Add task metadata
        task_with_metadata = {
            "task_id": task_id,
            "task": task,
            "queued_at": time.time(),
            "attempts": 0,
        }

        with self._queue_lock:
            # Get list of available workers (healthy and not assigned)
            # Update cached health state before filtering
            for worker_info in self.workers.values():
                if worker_info["healthy"] and not worker_info["process"].is_healthy():
                    worker_info["healthy"] = False  # Update stale cached state

            available_workers = [
                (worker_id, worker_info)
                for worker_id, worker_info in self.workers.items()
                if worker_info["healthy"] and worker_id not in self.worker_assignments
            ]

            if available_workers:
                # Immediate assignment - worker available
                worker_index = self._current_assignment_index % len(available_workers)
                selected_worker_id, selected_worker_info = available_workers[worker_index]
                self._current_assignment_index += 1

                # Update worker tracking
                selected_worker_info["status"] = "assigned"
                selected_worker_info["task_count"] += 1

                # Track active assignment
                self.worker_assignments[selected_worker_id] = task_with_metadata

                assignment_time = time.time() - start_time
                if assignment_time > CONFIG.performance.TASK_ASSIGNMENT_TIMEOUT:
                    logger.warning(
                        f"Task assignment took {assignment_time:.3f}s, exceeds {CONFIG.performance.TASK_ASSIGNMENT_TIMEOUT*1000:.0f}ms target"
                    )

                logger.info(f"Assigned task {task_id} to worker {selected_worker_id}")
                return task_id
            else:
                # Queue task - no workers available
                self.task_queue.put(task_with_metadata)
                assignment_time = time.time() - start_time

                if assignment_time > CONFIG.performance.TASK_ASSIGNMENT_TIMEOUT:
                    logger.warning(
                        f"Task queueing took {assignment_time:.3f}s, exceeds {CONFIG.performance.TASK_ASSIGNMENT_TIMEOUT*1000:.0f}ms target"
                    )

                logger.info(f"Queued task {task_id} (queue size: {self.task_queue.qsize()})")
                return task_id

    def develop_script(self, worker_id: str, task: dict[str, Any]) -> dict[str, Any] | None:
        """
        Develop script using advanced AI-guided script generation with pattern reuse.

        Args:
            worker_id: Unique worker identifier
            task: Task dictionary with development requirements

        Returns:
            Script development result dictionary
        """
        if not self._validate_worker_for_development(worker_id):
            return None

        start_time = time.time()
        try:
            # Phase 1: Pattern retrieval and context analysis
            relevant_patterns = self._retrieve_relevant_patterns(task)
            
            # Phase 2: Iterative script development
            development_result = self._develop_script_iteratively(
                worker_id, task, relevant_patterns
            )
            
            # Phase 3: Compile final result
            final_result = self._compile_development_result(
                worker_id, development_result, start_time
            )
            
            # Update worker and metrics
            self._finalize_development_session(worker_id, final_result)
            
            return final_result

        except Exception as e:
            return self._handle_development_exception(worker_id, e, start_time)

    def _validate_worker_for_development(self, worker_id: str) -> bool:
        """Validate worker is available for script development."""
        if worker_id not in self.workers:
            logger.warning(f"Cannot develop script with unknown worker {worker_id}")
            return False

        worker_info = self.workers[worker_id]
        process = worker_info["process"]

        if not process.is_healthy():
            logger.warning(f"Worker {worker_id} is not healthy, cannot develop script")
            return False

        return True

    def _retrieve_relevant_patterns(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        """Retrieve and synthesize relevant patterns for script development using advanced techniques."""
        context_filter = self._extract_context_filter(task)
        all_patterns = self.get_shared_patterns(context_filter)
        
        if not all_patterns:
            logger.debug("No patterns found in memory, using empty pattern list")
            return []
        
        # Use semantic pattern engine for better contextual matching
        task_context = task.get('context', {})
        if 'objective' in task:
            task_context['objective'] = task['objective']
            
        # Get semantically relevant patterns
        semantically_relevant = self.semantic_pattern_engine.get_contextual_pattern_recommendations(
            all_patterns, task_context, top_k=CONFIG.script_development['MAX_PATTERNS_IN_PROMPT']
        )
        
        # Apply cross-worker pattern synthesis for innovation
        if len(semantically_relevant) >= 2:
            synthesized_patterns = self.cross_worker_pattern_synthesis.synthesize_patterns(
                semantically_relevant, task_context, synthesis_count=2
            )
            
            # Combine original patterns with synthesized ones
            # Give preference to synthesized patterns as they represent innovation
            combined_patterns = synthesized_patterns + semantically_relevant
            
            # Limit total patterns to avoid overwhelming the prompt
            max_patterns = CONFIG.script_development['MAX_PATTERNS_IN_PROMPT']
            final_patterns = combined_patterns[:max_patterns]
            
            logger.debug(f"Retrieved {len(semantically_relevant)} semantic patterns and {len(synthesized_patterns)} synthesized patterns")
        else:
            final_patterns = semantically_relevant
            logger.debug(f"Retrieved {len(semantically_relevant)} semantically relevant patterns (insufficient for synthesis)")
        
        return final_patterns

    def _develop_script_iteratively(
        self, worker_id: str, task: dict[str, Any], relevant_patterns: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Perform iterative script development with enhanced adaptive refinement."""
        start_time = time.time()
        
        worker_info = self.workers[worker_id]
        process = worker_info["process"]
        
        # Get adaptive parameters from performance monitor
        adaptive_params = self.performance_monitor.get_current_parameters()
        max_iterations = adaptive_params.get('max_iterations', task.get("max_iterations", CONFIG.script_development['DEFAULT_MAX_ITERATIONS']))
        
        current_script = None
        quality_score = 0.0
        validation_errors = []
        patterns_used = []
        
        for iteration in range(max_iterations):
            iteration_result = self._perform_development_iteration(
                process, worker_id, task, relevant_patterns, 
                current_script, validation_errors, iteration, max_iterations
            )
            
            if iteration_result is None:
                break
                
            current_script = iteration_result["script"]
            quality_score = iteration_result["quality_score"]
            patterns_used = iteration_result["patterns_used"]
            validation_errors = iteration_result["validation_errors"]
            
            # Check success criteria with adaptive threshold
            adaptive_success_threshold = adaptive_params.get('success_threshold', CONFIG.script_development['SUCCESS_QUALITY_THRESHOLD'])
            if quality_score >= adaptive_success_threshold or iteration == max_iterations - 1:
                break
        
        # Calculate development time
        development_time_ms = (time.time() - start_time) * 1000
        
        # Record session data for performance monitoring
        session_data = {
            'development_time_ms': development_time_ms,
            'quality_score': quality_score,
            'refinement_iterations': iteration + 1,
            'patterns_used': [p.get('pattern_id', 'unknown') for p in patterns_used if isinstance(p, dict)],
            'worker_id': worker_id,
            'max_iterations_used': max_iterations,
            'task_context': task.get('context', {}),
            'validation_errors_count': len(validation_errors),
            'success': quality_score >= adaptive_success_threshold
        }
        
        self.performance_monitor.record_development_session(session_data)
        
        return {
            "script": current_script,
            "quality_score": quality_score,
            "patterns_used": patterns_used,
            "validation_errors": validation_errors,
            "refinement_iterations": iteration + 1,
            "development_time_ms": development_time_ms,
            "adaptive_parameters_used": adaptive_params
        }

    def _perform_development_iteration(
        self, process, worker_id: str, task: dict[str, Any], 
        relevant_patterns: list[dict[str, Any]], current_script: str | None,
        validation_errors: list[str], iteration: int, max_iterations: int
    ) -> dict[str, Any] | None:
        """Perform enhanced genetic algorithm-based script development iteration with RL guidance."""
        logger.debug(f"Enhanced genetic script development iteration {iteration + 1}/{max_iterations}")
        
        # Get adaptive quality thresholds for this worker
        thresholds = self.adaptive_quality_thresholds.get_adaptive_thresholds(worker_id)
        success_threshold = thresholds['success_threshold']
        acceptable_threshold = thresholds['acceptable_threshold']
        
        # Initialize genetic population on first iteration
        if iteration == 0:
            self._genetic_population = GeneticPopulation(
                population_size=CONFIG.script_development['GENETIC_POPULATION_SIZE'],
                elite_size=CONFIG.script_development['GENETIC_ELITE_SIZE']
            )
            self._genetic_population.initialize_from_patterns(relevant_patterns, current_script)
            
            # Initialize RL state for this development session
            task_context = task.get('context', {})
            self._current_rl_state = self.reinforcement_learning_engine.encode_state(
                task_context, current_script or "", 0.0
            )
        
        # Get RL action recommendations for genetic algorithm parameters
        rl_action = self.reinforcement_learning_engine.select_action(self._current_rl_state)
        
        # Apply RL recommendations to genetic algorithm
        self._apply_rl_guidance_to_genetic_algorithm(rl_action, self._genetic_population)
        
        # Evaluate fitness for all variants
        self._genetic_population.evaluate_fitness(self.quality_assessor, task)
        
        # Get current best variant
        best_variant = self._genetic_population.get_best_variant()
        if not best_variant:
            logger.error(f"No viable variants in population on iteration {iteration + 1}")
            return None
            
        # Check for early termination with adaptive threshold
        if best_variant.fitness >= success_threshold:
            logger.info(f"Early termination: Found high-quality solution (fitness: {best_variant.fitness:.3f}, threshold: {success_threshold:.3f})")
            
            # Store successful RL experience
            self._store_rl_experience(worker_id, task, best_variant, rl_action, iteration, max_iterations, success=True)
            
            return self._create_iteration_result(best_variant, [])
        
        # Apply enhanced reinforcement learning feedback
        self._apply_enhanced_reinforcement_feedback(worker_id, best_variant, task, rl_action, iteration, max_iterations)
        
        # Evolve population for next iteration (unless it's the last iteration)
        if iteration < max_iterations - 1:
            # Check population diversity to prevent premature convergence
            diversity = self._genetic_population.get_diversity_score()
            if diversity < CONFIG.script_development['DIVERSITY_THRESHOLD']:
                logger.debug("Low diversity detected, injecting new variants with RL guidance")
                self._inject_diversity_with_rl_guidance(rl_action)
            
            self._genetic_population.evolve_generation(task.get('context', {}))
            
        # Return best result from current generation
        validation_result = self.quality_assessor.assess_script_quality(best_variant.script_text, task)
        
        # Update adaptive quality thresholds with performance data
        self.adaptive_quality_thresholds.update_worker_performance(worker_id, best_variant.fitness)
        
        return {
            "script": best_variant.script_text,
            "quality_score": best_variant.fitness,
            "patterns_used": validation_result.get("patterns_detected", []),
            "validation_errors": [] if validation_result.get("is_valid", False) else validation_result.get("errors", []),
            "genetic_info": {
                "variant_id": best_variant.variant_id,
                "generation": best_variant.generation,
                "population_diversity": self._genetic_population.get_diversity_score(),
                "mutation_history": best_variant.mutation_history
            },
            "adaptive_info": {
                "success_threshold": success_threshold,
                "acceptable_threshold": acceptable_threshold,
                "worker_performance": thresholds['worker_performance'],
                "global_trend": thresholds['global_trend']
            },
            "rl_info": {
                "action_taken": rl_action,
                "state_encoded": self._current_rl_state,
                "exploration_rate": self.reinforcement_learning_engine.exploration_rate
            }
        }
    
    def _create_iteration_result(self, variant: ScriptVariant, validation_errors: List[str]) -> Dict[str, Any]:
        """Create standardized iteration result from variant."""
        return {
            "script": variant.script_text,
            "quality_score": variant.fitness,
            "patterns_used": [],  # Will be filled by quality assessor
            "validation_errors": validation_errors,
            "genetic_info": {
                "variant_id": variant.variant_id,
                "generation": variant.generation,
                "mutation_history": variant.mutation_history
            }
        }
    
    def _apply_rl_guidance_to_genetic_algorithm(self, rl_action: Dict[str, str], population: GeneticPopulation) -> None:
        """Apply reinforcement learning action recommendations to genetic algorithm parameters."""
        # Adjust population parameters based on RL recommendations
        if rl_action['parameter'] == 'increase_population':
            # Temporarily expand population by adding more variants
            if len(population.variants) < CONFIG.script_development['GENETIC_POPULATION_SIZE'] * 1.5:
                for _ in range(2):  # Add 2 new variants
                    if population.variants:
                        base_variant = random.choice(population.variants)
                        new_script = population._apply_random_mutation(base_variant.script_text)
                        new_variant = ScriptVariant(new_script, generation=population.generation)
                        new_variant.mutation_history = ['rl_guided_expansion']
                        population.variants.append(new_variant)
        
        elif rl_action['parameter'] == 'decrease_population':
            # Remove weakest variants to focus evolution
            if len(population.variants) > CONFIG.script_development['GENETIC_POPULATION_SIZE'] // 2:
                population.variants.sort(key=lambda v: v.fitness, reverse=True)
                population.variants = population.variants[:max(4, len(population.variants) - 2)]
        
        elif rl_action['parameter'] == 'adjust_mutation_rate':
            # Modify the genetic operators based on RL recommendation
            # This affects the next generation's mutation probability
            trends = self.reinforcement_learning_engine.get_success_trends()
            if trends['trend_score'] < 0.4:  # Poor performance trend
                # Increase exploration through higher mutation rate
                self._temporary_mutation_rate_boost = 0.2
            else:
                self._temporary_mutation_rate_boost = 0.0
    
    def _store_rl_experience(self, worker_id: str, task: Dict[str, Any], variant: ScriptVariant, 
                           action: Dict[str, str], iteration: int, max_iterations: int, success: bool) -> None:
        """Store experience for reinforcement learning."""
        task_context = task.get('context', {})
        current_state = self.reinforcement_learning_engine.encode_state(
            task_context, variant.script_text, variant.fitness
        )
        
        # Compute multi-component reward
        quality_improvement = variant.fitness - (variant.fitness * 0.5)  # Assume improvement from baseline
        iteration_efficiency = 1.0 - (iteration / max_iterations)
        diversity_score = self._genetic_population.get_diversity_score()
        pattern_novelty = self._genetic_population._compute_pattern_novelty(variant)
        
        reward = self.reinforcement_learning_engine.compute_reward(
            quality_improvement, iteration_efficiency, diversity_score, pattern_novelty
        )
        
        # Store experience
        experience = {
            'state': self._current_rl_state,
            'action': action,
            'reward': reward,
            'next_state': current_state,
            'done': success or iteration >= max_iterations - 1,
            'worker_id': worker_id,
            'quality_score': variant.fitness,
            'iteration': iteration
        }
        
        self.reinforcement_learning_engine.store_experience(experience)
        
        # Perform experience replay learning periodically
        if len(self.reinforcement_learning_engine.experience_buffer) >= 32:
            self.reinforcement_learning_engine.replay_learning(batch_size=16)
        
        # Update current RL state for next iteration
        self._current_rl_state = current_state
    
    def _apply_enhanced_reinforcement_feedback(self, worker_id: str, best_variant: ScriptVariant, 
                                             task: Dict[str, Any], rl_action: Dict[str, str], 
                                             iteration: int, max_iterations: int) -> None:
        """Apply enhanced reinforcement learning feedback with Q-learning updates."""
        # Store experience for this iteration
        success = best_variant.fitness >= self.adaptive_quality_thresholds.get_adaptive_thresholds(worker_id)['acceptable_threshold']
        self._store_rl_experience(worker_id, task, best_variant, rl_action, iteration, max_iterations, success)
        
        # Adapt exploration rate based on performance trends
        trends = self.reinforcement_learning_engine.get_success_trends()
        self.reinforcement_learning_engine.adapt_exploration_rate(trends['trend_score'])
        
        # Update semantic pattern engine with success/failure information
        if hasattr(self, '_genetic_population') and self._genetic_population.variants:
            for pattern in task.get('patterns_used', []):
                if 'pattern_id' in pattern:
                    self.semantic_pattern_engine.update_pattern_success(
                        pattern['pattern_id'], task.get('context', {}), success
                    )
    
    def _inject_diversity_with_rl_guidance(self, rl_action: Dict[str, str]) -> None:
        """Inject population diversity with RL-guided strategies."""
        if not hasattr(self, '_genetic_population'):
            return
            
        # Get diversity injection strategies based on RL action
        injection_strategies = []
        
        if rl_action['mutation'] == 'insert_command':
            injection_strategies.append('command_insertion_variants')
        elif rl_action['mutation'] == 'delete_command':
            injection_strategies.append('minimalist_variants')
        elif rl_action['mutation'] == 'modify_timing':
            injection_strategies.append('timing_optimized_variants')
        elif rl_action['mutation'] == 'swap_sequence':
            injection_strategies.append('sequence_permutation_variants')
        
        # Replace worst performing variants with strategically generated ones
        sorted_variants = sorted(self._genetic_population.variants, key=lambda v: v.fitness)
        replace_count = max(1, len(sorted_variants) // 4)
        
        for i in range(replace_count):
            strategy = random.choice(injection_strategies) if injection_strategies else 'random_variant'
            
            if strategy == 'command_insertion_variants':
                new_script = self._generate_command_rich_variant()
            elif strategy == 'minimalist_variants':
                new_script = self._generate_minimal_variant()
            elif strategy == 'timing_optimized_variants':
                new_script = self._generate_timing_focused_variant()
            elif strategy == 'sequence_permutation_variants':
                new_script = self._generate_permutation_variant()
            else:
                new_script = self._genetic_population._generate_random_script()
            
            new_variant = ScriptVariant(new_script, generation=self._genetic_population.generation)
            new_variant.mutation_history = [f'rl_guided_{strategy}']
            sorted_variants[i] = new_variant
            
        self._genetic_population.variants = sorted_variants
    
    def _generate_command_rich_variant(self) -> str:
        """Generate a variant with diverse command usage."""
        commands = []
        command_types = ['A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'OBSERVE']
        
        for _ in range(random.randint(8, 15)):
            commands.append(random.choice(command_types))
            if random.random() < 0.3:  # Add timing
                commands.append(str(random.randint(1, 8)))
        
        return '\n'.join(commands)
    
    def _generate_minimal_variant(self) -> str:
        """Generate a minimal, efficient variant."""
        essential_commands = ['A', 'B', 'START']
        commands = []
        
        for _ in range(random.randint(3, 6)):
            commands.append(random.choice(essential_commands))
            if random.random() < 0.2:  # Less timing
                commands.append(str(random.randint(1, 3)))
        
        return '\n'.join(commands)
    
    def _generate_timing_focused_variant(self) -> str:
        """Generate a variant optimized for timing."""
        commands = []
        basic_commands = ['A', 'B', 'UP', 'DOWN']
        
        for _ in range(random.randint(5, 10)):
            commands.append(random.choice(basic_commands))
            # Higher probability of timing commands
            if random.random() < 0.6:
                commands.append(str(random.randint(1, 12)))
        
        return '\n'.join(commands)
    
    def _generate_permutation_variant(self) -> str:
        """Generate a variant by permuting existing successful patterns."""
        if hasattr(self, '_genetic_population') and self._genetic_population.variants:
            # Get best variant and permute its sequence
            best_variant = max(self._genetic_population.variants, key=lambda v: v.fitness)
            script_lines = [line.strip() for line in best_variant.script_text.split('\n') if line.strip()]
            
            # Apply random permutations
            if len(script_lines) > 3:
                # Swap adjacent pairs
                for _ in range(random.randint(1, 3)):
                    i = random.randint(0, len(script_lines) - 2)
                    script_lines[i], script_lines[i + 1] = script_lines[i + 1], script_lines[i]
            
            return '\n'.join(script_lines)
        else:
            return self._generate_command_rich_variant()
    
    def _inject_diversity(self) -> None:
        """Inject new variants to maintain population diversity."""
        if not hasattr(self, '_genetic_population'):
            return
            
        # Replace worst 25% with new random variants
        sorted_variants = sorted(self._genetic_population.variants, key=lambda v: v.fitness)
        replace_count = max(1, len(sorted_variants) // 4)
        
        for i in range(replace_count):
            new_script = self._genetic_population._generate_random_script()
            new_variant = ScriptVariant(new_script, generation=self._genetic_population.generation)
            new_variant.mutation_history = ['diversity_injection']
            sorted_variants[i] = new_variant
            
        self._genetic_population.variants = sorted_variants

    def _compile_development_result(
        self, worker_id: str, development_result: dict[str, Any], start_time: float
    ) -> dict[str, Any]:
        """Compile final development result."""
        development_time_ms = (time.time() - start_time) * 1000
        script = development_result["script"]
        validation_errors = development_result["validation_errors"]
        
        # Determine status
        if script and not validation_errors:
            status = "completed"
        elif script:
            status = "validation_error"
        else:
            status = "failed"
        
        # Get compiled script if valid
        compiled_script = None
        if script and not validation_errors:
            try:
                compiled_script = self.script_compiler.compile(script)
            except Exception:
                pass
        
        return {
            "worker_id": worker_id,
            "script": script or "",
            "compiled_script": compiled_script,
            "quality_score": development_result["quality_score"],
            "patterns_used": development_result["patterns_used"],
            "refinement_iterations": development_result["refinement_iterations"],
            "development_time_ms": development_time_ms,
            "status": status,
            "validation_errors": validation_errors,
            "timestamp": time.time(),
        }

    def _finalize_development_session(self, worker_id: str, result: dict[str, Any]) -> None:
        """Finalize development session and update metrics."""
        # Update worker status
        worker_info = self.workers[worker_id]
        worker_info["status"] = "ready"
        
        # Update performance metrics
        success = result["quality_score"] >= CONFIG.script_development['ACCEPTABLE_QUALITY_THRESHOLD']
        self._update_development_metrics(result["development_time_ms"], success)
        
        logger.info(f"Worker {worker_id} completed script development in {result['development_time_ms']:.1f}ms with quality score {result['quality_score']:.2f}")

    def _handle_development_exception(self, worker_id: str, exception: Exception, start_time: float) -> dict[str, Any]:
        """Handle development exceptions and return error result."""
        development_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Script development failed for worker {worker_id}: {exception}")
        
        worker_info = self.workers[worker_id]
        worker_info["status"] = "error"
        
        return {
            "worker_id": worker_id,
            "script": "",
            "compiled_script": None,
            "quality_score": 0.0,
            "patterns_used": [],
            "refinement_iterations": 0,
            "development_time_ms": development_time_ms,
            "status": "failed",
            "validation_errors": [str(exception)],
            "timestamp": time.time(),
        }

    def complete_task(self, worker_id: str) -> bool:
        """
        Mark a task as completed for the specified worker and process queued tasks.

        This method should be called after a worker finishes processing a task
        to make the worker available for new assignments and process any queued tasks.

        Args:
            worker_id: Unique worker identifier

        Returns:
            True if task completion processed successfully, False otherwise
        """
        if worker_id not in self.workers:
            logger.warning(f"Cannot complete task for unknown worker {worker_id}")
            return False

        with self._queue_lock:
            # Remove worker assignment
            if worker_id in self.worker_assignments:
                completed_task = self.worker_assignments.pop(worker_id)
                logger.info(f"Completed task {completed_task['task_id']} for worker {worker_id}")

            # Update worker status
            worker_info = self.workers[worker_id]
            worker_info["status"] = "ready"

            # Process next task from queue if available
            self._process_queue()

        return True

    def _process_queue(self):
        """
        Process queued tasks by assigning them to available workers.

        This internal method is called when workers become available to
        handle any pending tasks in the queue. Uses thread-safe queue operations.

        Note: Should be called with _queue_lock held.
        """
        if self.task_queue.empty():
            return

        # Find available workers (healthy and not assigned)
        available_workers = [
            (worker_id, worker_info)
            for worker_id, worker_info in self.workers.items()
            if worker_info["healthy"] and worker_id not in self.worker_assignments
        ]

        # Assign queued tasks to available workers
        tasks_assigned = 0
        while available_workers and not self.task_queue.empty():
            try:
                # Get next task from queue
                task_with_metadata = self.task_queue.get_nowait()

                # Select next available worker (round-robin)
                worker_index = self._current_assignment_index % len(available_workers)
                selected_worker_id, selected_worker_info = available_workers[worker_index]
                self._current_assignment_index += 1

                # Update worker tracking
                selected_worker_info["status"] = "assigned"
                selected_worker_info["task_count"] += 1

                # Track active assignment
                self.worker_assignments[selected_worker_id] = task_with_metadata

                # Remove assigned worker from available list
                available_workers.pop(worker_index)

                tasks_assigned += 1
                logger.info(
                    f"Assigned queued task {task_with_metadata['task_id']} to worker {selected_worker_id}"
                )

            except queue.Empty:
                break

        if tasks_assigned > 0:
            logger.info(f"Processed {tasks_assigned} queued tasks")

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """
        Get status information for a specific task.

        Args:
            task_id: Task identifier returned by assign_task()

        Returns:
            Dictionary with task status or None if task not found
        """
        # Check active assignments
        for worker_id, task_info in self.worker_assignments.items():
            if task_info["task_id"] == task_id:
                return {
                    "task_id": task_id,
                    "status": "assigned",
                    "worker_id": worker_id,
                    "queued_at": task_info["queued_at"],
                    "assigned_at": time.time(),
                }

        # Check if task is queued (expensive operation)
        with self._queue_lock:
            temp_tasks: list[dict[str, Any]] = []
            task_found = None

            # Drain queue to search for task
            while not self.task_queue.empty():
                try:
                    task_info = self.task_queue.get_nowait()
                    if task_info["task_id"] == task_id:
                        task_found = {
                            "task_id": task_id,
                            "status": "queued",
                            "worker_id": None,
                            "queued_at": task_info["queued_at"],
                            "queue_position": len(temp_tasks),
                        }
                    temp_tasks.append(task_info)
                except queue.Empty:
                    break

            # Restore queue
            for task_info in temp_tasks:
                self.task_queue.put(task_info)

            return task_found

    def get_queue_size(self) -> int:
        """Get the current number of queued tasks."""
        return self.task_queue.qsize()

    def share_pattern(self, pattern_data: dict[str, Any], discovered_by: str | None = None) -> bool:
        """
        Share a discovered pattern across all workers in the pool.

        This method stores a successful pattern or strategy using the MCP memory system
        and distributes it to all active workers for improved consistency and learning.

        Args:
            pattern_data: Dictionary containing pattern information (strategy, success_rate, context)
            discovered_by: Optional worker ID that discovered this pattern

        Returns:
            True if pattern was successfully shared, False otherwise
        """
        try:
            # Create a PokemonStrategy object from pattern data
            strategy_id = pattern_data.get("strategy_id", f"strategy_{uuid.uuid4().hex[:8]}")
            strategy = PokemonStrategy(
                id=strategy_id,
                name=pattern_data.get("name", "Discovered Strategy"),
                pattern_sequence=pattern_data.get("pattern_sequence", ["DISCOVERED_PATTERN"]),
                success_rate=pattern_data.get("success_rate", 0.0),
                estimated_time=pattern_data.get("estimated_time"),
                resource_requirements=pattern_data.get("resource_requirements", {}),
                risk_assessment=pattern_data.get("risk_assessment", {}),
                alternatives=pattern_data.get("alternatives", []),
                optimization_history=pattern_data.get("optimization_history", []),
            )

<<<<<<< HEAD
            # Store pattern using MCP memory integration via QueryBuilder
            result = self.query_builder.store_pattern(strategy)
=======
            # Store pattern using MCP memory integration
            query_builder = QueryBuilder()
            result = query_builder.store_pattern(strategy)
>>>>>>> origin/main

            if result.get("success", False):
                pattern_id = result.get("memory_id")
                logger.info(f"Stored pattern {strategy.id} with MCP ID {pattern_id}")

                # Distribute pattern to all active workers
                if self._distribute_pattern_to_workers(strategy, discovered_by):
                    logger.info(f"Successfully shared pattern {strategy.id} to all workers")
                    return True
                else:
                    logger.warning("Pattern stored but distribution to workers failed")
                    return False
            else:
                logger.error(f"Failed to store pattern in MCP system: {result}")
                return False

        except Exception as e:
            logger.error(f"Failed to share pattern: {e}")
            return False

    def get_shared_patterns(
        self, context_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve shared patterns from the MCP memory system.

        Args:
            context_filter: Optional filter criteria for pattern retrieval

        Returns:
            List of pattern dictionaries matching the filter criteria
        """
        try:
            # Build query based on filter
<<<<<<< HEAD
=======
            query_builder = QueryBuilder()
>>>>>>> origin/main
            if context_filter:
                # Use context information to build targeted query
                search_terms = []
                if "location" in context_filter:
                    search_terms.append(f"location:{context_filter['location']}")
                if "objective" in context_filter:
                    search_terms.append(f"objective:{context_filter['objective']}")
                query = " ".join(search_terms) if search_terms else "PokemonStrategy"
            else:
                query = "PokemonStrategy"

<<<<<<< HEAD
            # Use QueryBuilder for pattern retrieval
            result = self.query_builder.search_patterns(query)
=======
            result = query_builder.search_patterns(query)
>>>>>>> origin/main

            if result.get("success", False):
                patterns = []
                results = result.get("results", [])
                if isinstance(results, list):
                    for pattern_data in results:
                        # Convert MCP result to pattern dictionary
                        pattern_dict = {
                            "pattern_id": pattern_data.get("id"),
                            "strategy_id": pattern_data.get("strategy_id"),
                            "name": pattern_data.get("name"),
                            "description": pattern_data.get("description"),
                            "success_rate": pattern_data.get("success_rate", 0.0),
                            "usage_count": pattern_data.get("usage_count", 0),
                            "context": pattern_data.get("context", {}),
                            "discovered_at": pattern_data.get("discovered_at"),
                        }
                        patterns.append(pattern_dict)

                logger.info(f"Retrieved {len(patterns)} shared patterns")
                return patterns
            else:
                logger.warning(f"Failed to retrieve patterns: {result}")
                return []

        except Exception as e:
            logger.error(f"Failed to retrieve shared patterns: {e}")
            return []

    def _distribute_pattern_to_workers(
        self, strategy: PokemonStrategy, discovered_by: str | None = None
    ) -> bool:
        """
        Internal method to distribute a pattern to all active workers.

        Args:
            strategy: PokemonStrategy object to distribute
            discovered_by: Worker ID that discovered the pattern (will be excluded from distribution)

        Returns:
            True if pattern distributed successfully to at least one worker
        """
        distribution_count = 0

        for worker_id, worker_info in self.workers.items():
            # Skip the worker that discovered this pattern (they already know it)
            if worker_id == discovered_by:
                continue

            # Only send to healthy workers
            if not worker_info["healthy"]:
                continue

            try:
                # Format pattern for worker consumption
                pattern_message = self._format_pattern_message(strategy)

                # Send pattern to worker process
                process = worker_info["process"]
                response = process.send_message(pattern_message, timeout=CONFIG.performance.PATTERN_DISTRIBUTION_TIMEOUT)

                if response:
                    logger.debug(f"Distributed pattern {strategy.id} to worker {worker_id}")
                    distribution_count += 1
                else:
                    logger.warning(f"No response from worker {worker_id} when distributing pattern")

            except Exception as e:
                logger.error(f"Failed to distribute pattern to worker {worker_id}: {e}")

        return distribution_count > 0

    def _format_pattern_message(self, strategy: PokemonStrategy) -> str:
        """
        Format a pattern/strategy for distribution to Sonnet workers.

        Args:
            strategy: PokemonStrategy object to format

        Returns:
            Formatted message string for worker consumption
        """
        message = f"""SHARED PATTERN UPDATE

Pattern: {strategy.name}
Strategy ID: {strategy.id}
Success Rate: {strategy.success_rate:.2%}
Pattern Sequence: {', '.join(strategy.pattern_sequence)}
Estimated Time: {strategy.estimated_time}s

Resource Requirements: {strategy.resource_requirements}
Risk Assessment: {strategy.risk_assessment}

Please incorporate this successful pattern into your script development approach.
Focus on the techniques and strategies that contributed to its success."""

        return message

    def analyze_result(self, worker_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
        """
        Get refinement suggestions from worker based on execution result.

        Args:
            worker_id: Unique worker identifier
            result: Execution result to analyze

        Returns:
            Analysis with refinement suggestions or None if failed
        """
        if worker_id not in self.workers:
            logger.warning(f"Cannot analyze result with unknown worker {worker_id}")
            return None

        worker_info = self.workers[worker_id]
        process = worker_info["process"]

        if not process.is_healthy():
            logger.warning(f"Worker {worker_id} is not healthy, cannot analyze result")
            return None

        try:
            # Format result analysis request
            analysis_message = self._format_analysis_message(result)

            # Send analysis request to worker
            response = process.send_message(analysis_message, timeout=CONFIG.performance.RESULT_ANALYSIS_TIMEOUT)

            if response is None:
                logger.error(f"No analysis response from worker {worker_id}")
                return None

            # Format analysis result
            analysis = {"worker_id": worker_id, "suggestions": response, "timestamp": time.time()}

            logger.debug(f"Worker {worker_id} completed result analysis")
            return analysis

        except Exception as e:
            logger.error(f"Result analysis failed for worker {worker_id}: {e}")
            return None

    def _format_task_message(self, task: dict[str, Any]) -> str:
        """Format task dictionary into message for Sonnet tactical worker."""
        objective = task.get("objective", "")
        context = task.get("context", "")

        message = f"""Task: {objective}

Context: {context}

Please develop a Pokemon Red speedrun script to accomplish this objective.
Focus on frame-perfect execution and optimal path planning."""

        return message

    def _extract_context_filter(self, task: dict[str, Any]) -> dict[str, Any]:
        """Extract context information for pattern filtering from task."""
        context_filter = {}
        
        task_context = task.get("context", {})
        
        # Extract location information
        if "location" in task_context:
            context_filter["location"] = task_context["location"]
        elif "current_location" in task_context:
            context_filter["location"] = task_context["current_location"]
        
        # Extract objective information  
        if "objective" in task:
            context_filter["objective"] = task["objective"]
        elif "goal" in task:
            context_filter["objective"] = task["goal"]
            
        return context_filter
    
    def _build_enhanced_prompt(
        self, 
        task: dict[str, Any], 
        relevant_patterns: list[dict[str, Any]], 
        current_script: str | None,
        validation_errors: list[str],
        iteration: int
    ) -> str:
        """Build an enhanced prompt for script development with pattern context."""
        objective = task.get("objective", "")
        context = task.get("context", "")
        
        if iteration == 0:
            base_prompt = self._build_initial_prompt(objective, context, relevant_patterns)
        else:
            base_prompt = self._build_refinement_prompt(
                objective, context, current_script, validation_errors, iteration
            )
        
        # Enhance with pattern insights
        if relevant_patterns:
            pattern_insights = self.pattern_refiner.get_pattern_insights(relevant_patterns)
            base_prompt += f"\n\nPattern Insights: {pattern_insights}"
        
        return base_prompt

    def _build_initial_prompt(
        self, objective: str, context: str, relevant_patterns: list[dict[str, Any]]
    ) -> str:
        """Build initial development prompt."""
        prompt_parts = [
            "POKEMON RED SPEEDRUN SCRIPT DEVELOPMENT",
            f"Objective: {objective}",
            f"Context: {context}",
            "",
            "Available Patterns:",
        ]
        
        if relevant_patterns:
            for pattern in relevant_patterns[:CONFIG.script_development['MAX_PATTERNS_IN_PROMPT']]:
                success_rate = pattern.get("success_rate", 0.0) * 100
                prompt_parts.append(f"- {pattern.get('name', 'Unknown')}: {pattern.get('description', 'No description')} (Success: {success_rate:.1f}%)")
        else:
            prompt_parts.append("- No specific patterns available, create from scratch")
        
        prompt_parts.extend([
            "",
            "Generate a Pokemon Red speedrun script using our DSL syntax:",
            "- Use basic inputs: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT",
            "- Add delays with numbers (e.g., 5 for 5-frame delay)",
            "- Use REPEAT constructs for loops: repeat 3 times ... end",
            "- Include OBSERVE statements for checkpoints",
            "- Focus on frame-perfect execution and optimal paths",
            "",
            "Return ONLY the DSL script, no explanations."
        ])
        
        return "\n".join(prompt_parts)

    def _build_refinement_prompt(
        self, objective: str, context: str, current_script: str | None, 
        validation_errors: list[str], iteration: int
    ) -> str:
        """Build refinement prompt with error feedback."""
        prompt_parts = [
            "POKEMON RED SPEEDRUN SCRIPT REFINEMENT",
            f"Iteration {iteration + 1} - Previous script had issues",
            "",
            "Previous Script:",
            current_script or "(No previous script)",
            "",
            "Issues Found:",
        ]
        
        if validation_errors:
            for error in validation_errors:
                prompt_parts.append(f"- {error}")
        else:
            prompt_parts.append("- Script quality below threshold, needs optimization")
        
        prompt_parts.extend([
            "",
            f"Objective: {objective}",
            f"Context: {context}",
            "",
            "Please refine the script to fix the issues above.",
            "Focus on:",
            "- Correct DSL syntax",
            "- Logical sequence of actions",
            "- Optimal timing and efficiency",
            "- Error handling and robustness",
            "",
            "Return ONLY the improved DSL script, no explanations."
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_script_from_response(self, response: str) -> str | None:
        """Extract DSL script content from Sonnet worker response."""
        if not response or not isinstance(response, str):
            return None
        
        # Clean up the response
        lines = []
        for line in response.split('\n'):
            stripped = line.strip()
            
            # Skip empty lines and obvious non-script content
            if not stripped:
                continue
            if stripped.startswith('```'):
                continue
            if stripped.startswith('#'):
                lines.append(stripped)  # Keep comments
                continue
            if any(word in stripped.lower() for word in ['explanation', 'note:', 'this script', 'the above']):
                continue
                
            # Keep DSL commands
            if any(word in stripped.upper() for word in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT', 'REPEAT', 'END', 'OBSERVE']) or stripped.isdigit():
                lines.append(stripped.upper())
        
        if not lines:
            return None
            
        return '\n'.join(lines)
    
    
    
    def _update_development_metrics(self, development_time_ms: float, success: bool):
        """Update performance tracking metrics."""
        metrics = self.development_metrics
        
        # Update totals
        metrics["total_scripts_developed"] += 1
        
        # Update average development time (exponential moving average)
        if metrics["total_scripts_developed"] == 1:
            metrics["average_development_time_ms"] = development_time_ms
        else:
            alpha = CONFIG.script_development['METRICS_LEARNING_RATE']
            metrics["average_development_time_ms"] = (
                alpha * development_time_ms + 
                (1 - alpha) * metrics["average_development_time_ms"]
            )
        
        # Update success rate (exponential moving average)
        success_value = CONFIG.script_development['SUCCESS_RATE_INIT'] if success else CONFIG.script_development['FAILED_RATE_INIT']
        if metrics["total_scripts_developed"] == 1:
            metrics["success_rate"] = success_value
        else:
            alpha = CONFIG.script_development['METRICS_LEARNING_RATE']
            metrics["success_rate"] = (
                alpha * success_value + 
                (1 - alpha) * metrics["success_rate"]
            )

    def _format_analysis_message(self, result: dict[str, Any]) -> str:
        """Format execution result into analysis request message."""
        message = f"""Analyze this execution result and provide refinement suggestions:

Result: {result}

Please provide specific suggestions for improving the script based on this result."""

        return message

    def get_worker_count(self) -> int:
        """Get the current number of registered workers."""
        return len(self.workers)

    def get_healthy_worker_count(self) -> int:
        """Get the number of currently healthy workers."""
        return sum(1 for worker_info in self.workers.values() if worker_info["healthy"])

    def is_initialized(self) -> bool:
        """Check if the worker pool has been successfully initialized."""
        return self._initialized

    def shutdown(self):
        """Clean shutdown of worker pool."""
        logger.info("Shutting down SonnetWorkerPool")

        # Clear task queue
        with self._queue_lock:
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break
            self.worker_assignments.clear()

        self.workers.clear()
        self._initialized = False


class WorkerDistributor:
    """Intelligent load balancing and worker distribution system.
    
    Provides advanced worker selection and load balancing capabilities that
    go beyond simple round-robin assignment, using performance metrics,
    task affinity, and predictive modeling for optimal task distribution.
    """
    
    def __init__(self):
        # Worker performance tracking
        self.worker_performance_history: Dict[str, List[float]] = {}
        self.worker_task_completion_times: Dict[str, List[float]] = {}
        self.worker_success_rates: Dict[str, float] = {}
        self.worker_specializations: Dict[str, Dict[str, float]] = {}
        
        # Load tracking and prediction
        self.current_worker_loads: Dict[str, float] = {}
        self.predicted_completion_times: Dict[str, float] = {}
        self.task_complexity_cache: Dict[str, float] = {}
        
        # Distribution strategies
        self.distribution_strategies = {
            'round_robin': self._distribute_round_robin,
            'least_loaded': self._distribute_least_loaded,
            'performance_weighted': self._distribute_performance_weighted,
            'affinity_based': self._distribute_affinity_based,
            'predictive': self._distribute_predictive,
            'hybrid': self._distribute_hybrid
        }
        
        self.default_strategy = 'hybrid'
        self.distribution_metrics = {
            'total_distributions': 0,
            'distribution_times': [],
            'strategy_success_rates': {}
        }
        
        # Worker affinity learning
        self.task_type_performance: Dict[str, Dict[str, List[float]]] = {}
        self.context_worker_mapping: Dict[str, List[Tuple[str, float]]] = {}
        
    def distribute_task(self, task: Dict[str, Any], 
                       available_workers: List[str],
                       distribution_strategy: Optional[str] = None,
                       constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Distribute task to optimal worker using intelligent selection.
        
        Args:
            task: Task to be distributed with context and requirements
            available_workers: List of available worker IDs
            distribution_strategy: Strategy to use for distribution
            constraints: Optional constraints for distribution
            
        Returns:
            Distribution result with selected worker and metadata
        """
        distribution_start = time.time()
        
        try:
            if not available_workers:
                return self._get_empty_distribution_result("no_available_workers")
            
            # Select distribution strategy
            strategy = distribution_strategy or self.default_strategy
            strategy_func = self.distribution_strategies.get(strategy, self._distribute_hybrid)
            
            # Apply constraints if specified
            filtered_workers = self._apply_distribution_constraints(available_workers, constraints)
            if not filtered_workers:
                return self._get_empty_distribution_result("constraints_too_restrictive")
            
            # Perform distribution
            distribution_result = strategy_func(task, filtered_workers)
            
            # Track distribution metrics
            distribution_time = time.time() - distribution_start
            self._record_distribution_metrics(strategy, distribution_time)
            
            # Add metadata to result
            distribution_result.update({
                'distribution_strategy': strategy,
                'distribution_time_ms': distribution_time * 1000,
                'available_worker_count': len(available_workers),
                'filtered_worker_count': len(filtered_workers),
                'distribution_timestamp': time.time()
            })
            
            # Update worker load if distribution succeeded
            if distribution_result.get('selected_worker'):
                self._update_worker_load(distribution_result['selected_worker'], task)
            
            return distribution_result
            
        except Exception as e:
            logger.error(f"Task distribution failed: {e}")
            return self._get_empty_distribution_result("distribution_error", str(e))
    
    def update_worker_performance(self, worker_id: str, task: Dict[str, Any], 
                                 result: Dict[str, Any]) -> None:
        """Update worker performance history for better distribution decisions."""
        success = result.get('success', False)
        completion_time = result.get('execution_time_ms', 0) / 1000.0
        quality_score = result.get('quality_score', 0.5)
        
        # Update performance history
        if worker_id not in self.worker_performance_history:
            self.worker_performance_history[worker_id] = []
        
        performance_score = 0.6 * (1.0 if success else 0.0) + 0.4 * quality_score
        self.worker_performance_history[worker_id].append(performance_score)
        
        # Keep only recent history
        if len(self.worker_performance_history[worker_id]) > 50:
            self.worker_performance_history[worker_id] = self.worker_performance_history[worker_id][-25:]
        
        # Update completion times
        if worker_id not in self.worker_task_completion_times:
            self.worker_task_completion_times[worker_id] = []
        
        if completion_time > 0:
            self.worker_task_completion_times[worker_id].append(completion_time)
            if len(self.worker_task_completion_times[worker_id]) > 20:
                self.worker_task_completion_times[worker_id] = self.worker_task_completion_times[worker_id][-10:]
        
        # Update success rate
        current_success_rate = self.worker_success_rates.get(worker_id, 0.5)
        alpha = 0.1
        self.worker_success_rates[worker_id] = (
            alpha * (1.0 if success else 0.0) + (1 - alpha) * current_success_rate
        )
        
        # Update task type performance
        task_type = self._classify_task_type(task)
        if task_type not in self.task_type_performance:
            self.task_type_performance[task_type] = {}
        if worker_id not in self.task_type_performance[task_type]:
            self.task_type_performance[task_type][worker_id] = []
        
        self.task_type_performance[task_type][worker_id].append(performance_score)
        if len(self.task_type_performance[task_type][worker_id]) > 10:
            self.task_type_performance[task_type][worker_id] = (
                self.task_type_performance[task_type][worker_id][-5:]
            )
        
        # Update worker load (decrease after task completion)
        if worker_id in self.current_worker_loads:
            task_complexity = self._estimate_task_complexity(task)
            load_decrease = min(task_complexity * 0.3, self.current_worker_loads[worker_id])
            self.current_worker_loads[worker_id] = max(0.0, 
                self.current_worker_loads[worker_id] - load_decrease
            )
        
        logger.debug(f"Updated performance for {worker_id}: success={success}, "
                    f"quality={quality_score:.3f}, time={completion_time:.1f}s")
    
    def predict_worker_performance(self, worker_id: str, task: Dict[str, Any]) -> Dict[str, float]:
        """Predict worker performance for a specific task."""
        predictions = {
            'success_probability': 0.5,
            'expected_completion_time': 5.0,
            'expected_quality_score': 0.5,
            'confidence': 0.5
        }
        
        # Get historical performance
        history = self.worker_performance_history.get(worker_id, [])
        if not history:
            return predictions
        
        # Base success probability on recent performance
        recent_performance = history[-5:] if len(history) >= 5 else history
        predictions['success_probability'] = sum(recent_performance) / len(recent_performance)
        
        # Predict completion time from historical data
        completion_times = self.worker_task_completion_times.get(worker_id, [])
        if completion_times:
            task_complexity = self._estimate_task_complexity(task)
            base_time = sum(completion_times) / len(completion_times)
            predictions['expected_completion_time'] = base_time * (0.5 + task_complexity)
        
        # Predict quality based on task type affinity
        task_type = self._classify_task_type(task)
        if task_type in self.task_type_performance and worker_id in self.task_type_performance[task_type]:
            type_performance = self.task_type_performance[task_type][worker_id]
            predictions['expected_quality_score'] = sum(type_performance) / len(type_performance)
        
        # Confidence based on data availability
        data_points = len(history) + len(completion_times)
        predictions['confidence'] = min(1.0, data_points / 20.0)
        
        return predictions
    
    def get_optimal_worker_distribution(self, tasks: List[Dict[str, Any]], 
                                      available_workers: List[str]) -> Dict[str, Any]:
        """Get optimal distribution of multiple tasks across workers."""
        if not tasks or not available_workers:
            return {'distributions': [], 'optimization_score': 0.0}
        
        distributions = []
        worker_loads = {worker: 0.0 for worker in available_workers}
        
        # Sort tasks by priority and complexity
        sorted_tasks = sorted(tasks, 
                            key=lambda t: (t.get('priority', 0.5), 
                                         self._estimate_task_complexity(t)),
                            reverse=True)
        
        for task in sorted_tasks:
            # Find best worker considering current load distribution
            best_worker = self._find_optimal_worker_for_batch(
                task, available_workers, worker_loads
            )
            
            if best_worker:
                distributions.append({
                    'task': task,
                    'worker': best_worker,
                    'estimated_load': worker_loads[best_worker]
                })
                
                # Update load for next iteration
                task_complexity = self._estimate_task_complexity(task)
                worker_loads[best_worker] += task_complexity * 0.3
            else:
                distributions.append({
                    'task': task,
                    'worker': None,
                    'reason': 'no_suitable_worker'
                })
        
        # Calculate optimization score
        successful_distributions = [d for d in distributions if d.get('worker')]
        if successful_distributions:
            load_variance = self._calculate_load_variance(worker_loads)
            optimization_score = 1.0 - load_variance  # Lower variance = better distribution
        else:
            optimization_score = 0.0
        
        return {
            'distributions': distributions,
            'optimization_score': optimization_score,
            'worker_load_balance': worker_loads
        }
    
    def _distribute_round_robin(self, task: Dict[str, Any], workers: List[str]) -> Dict[str, Any]:
        """Simple round-robin distribution."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected_worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        
        return {
            'selected_worker': selected_worker,
            'confidence': 0.5,
            'selection_reason': 'round_robin'
        }
    
    def _distribute_least_loaded(self, task: Dict[str, Any], workers: List[str]) -> Dict[str, Any]:
        """Distribute to least loaded worker."""
        worker_loads = [(w, self.current_worker_loads.get(w, 0.0)) for w in workers]
        selected_worker = min(worker_loads, key=lambda x: x[1])[0]
        
        return {
            'selected_worker': selected_worker,
            'confidence': 0.7,
            'selection_reason': 'least_loaded',
            'worker_load': self.current_worker_loads.get(selected_worker, 0.0)
        }
    
    def _distribute_performance_weighted(self, task: Dict[str, Any], workers: List[str]) -> Dict[str, Any]:
        """Distribute based on historical performance."""
        worker_scores = []
        
        for worker in workers:
            performance_history = self.worker_performance_history.get(worker, [0.5])
            avg_performance = sum(performance_history) / len(performance_history)
            
            # Penalize for high current load
            current_load = self.current_worker_loads.get(worker, 0.0)
            load_penalty = current_load * 0.3
            
            score = avg_performance - load_penalty
            worker_scores.append((worker, score))
        
        selected_worker = max(worker_scores, key=lambda x: x[1])[0]
        confidence = min(1.0, max(worker_scores, key=lambda x: x[1])[1])
        
        return {
            'selected_worker': selected_worker,
            'confidence': confidence,
            'selection_reason': 'performance_weighted'
        }
    
    def _distribute_affinity_based(self, task: Dict[str, Any], workers: List[str]) -> Dict[str, Any]:
        """Distribute based on task-worker affinity."""
        task_type = self._classify_task_type(task)
        worker_scores = []
        
        for worker in workers:
            # Get task type performance
            if task_type in self.task_type_performance and worker in self.task_type_performance[task_type]:
                type_performance = self.task_type_performance[task_type][worker]
                affinity_score = sum(type_performance) / len(type_performance)
            else:
                affinity_score = 0.5  # Neutral for unknown combinations
            
            # Factor in current load
            current_load = self.current_worker_loads.get(worker, 0.0)
            adjusted_score = affinity_score * (1.0 - current_load * 0.4)
            
            worker_scores.append((worker, adjusted_score))
        
        selected_worker = max(worker_scores, key=lambda x: x[1])[0]
        confidence = max(worker_scores, key=lambda x: x[1])[1]
        
        return {
            'selected_worker': selected_worker,
            'confidence': confidence,
            'selection_reason': 'affinity_based',
            'task_type': task_type
        }
    
    def _distribute_predictive(self, task: Dict[str, Any], workers: List[str]) -> Dict[str, Any]:
        """Distribute using predictive performance modeling."""
        predictions = {}
        best_worker = None
        best_score = -1
        
        for worker in workers:
            prediction = self.predict_worker_performance(worker, task)
            
            # Composite score considering multiple factors
            score = (
                0.4 * prediction['success_probability'] +
                0.3 * (1.0 - min(1.0, prediction['expected_completion_time'] / 10.0)) +
                0.2 * prediction['expected_quality_score'] +
                0.1 * prediction['confidence']
            )
            
            predictions[worker] = prediction
            if score > best_score:
                best_score = score
                best_worker = worker
        
        return {
            'selected_worker': best_worker,
            'confidence': predictions[best_worker]['confidence'] if best_worker else 0.0,
            'selection_reason': 'predictive',
            'prediction': predictions.get(best_worker, {})
        }
    
    def _distribute_hybrid(self, task: Dict[str, Any], workers: List[str]) -> Dict[str, Any]:
        """Hybrid distribution combining multiple strategies."""
        # Get scores from different strategies
        performance_result = self._distribute_performance_weighted(task, workers)
        affinity_result = self._distribute_affinity_based(task, workers)
        predictive_result = self._distribute_predictive(task, workers)
        
        # Combine scores with weights
        worker_combined_scores = {}
        
        for worker in workers:
            score = 0.0
            
            # Performance weight (30%)
            if performance_result['selected_worker'] == worker:
                score += 0.3 * performance_result['confidence']
            
            # Affinity weight (30%)
            if affinity_result['selected_worker'] == worker:
                score += 0.3 * affinity_result['confidence']
            
            # Predictive weight (40%)
            if predictive_result['selected_worker'] == worker:
                score += 0.4 * predictive_result['confidence']
            
            worker_combined_scores[worker] = score
        
        # Select best worker
        selected_worker = max(worker_combined_scores.keys(), 
                            key=lambda w: worker_combined_scores[w])
        
        return {
            'selected_worker': selected_worker,
            'confidence': worker_combined_scores[selected_worker],
            'selection_reason': 'hybrid',
            'strategy_contributions': {
                'performance': performance_result['selected_worker'] == selected_worker,
                'affinity': affinity_result['selected_worker'] == selected_worker,
                'predictive': predictive_result['selected_worker'] == selected_worker
            }
        }
    
    def _apply_distribution_constraints(self, workers: List[str], 
                                      constraints: Optional[Dict[str, Any]]) -> List[str]:
        """Apply constraints to filter available workers."""
        if not constraints:
            return workers
        
        filtered_workers = workers[:]
        
        # Max load constraint
        if 'max_load' in constraints:
            max_load = constraints['max_load']
            filtered_workers = [w for w in filtered_workers 
                              if self.current_worker_loads.get(w, 0.0) <= max_load]
        
        # Min performance constraint
        if 'min_performance' in constraints:
            min_performance = constraints['min_performance']
            filtered_workers = [w for w in filtered_workers 
                              if self.worker_success_rates.get(w, 0.5) >= min_performance]
        
        # Excluded workers
        if 'excluded_workers' in constraints:
            excluded = set(constraints['excluded_workers'])
            filtered_workers = [w for w in filtered_workers if w not in excluded]
        
        # Required workers (if specified, only these are considered)
        if 'required_workers' in constraints:
            required = set(constraints['required_workers'])
            filtered_workers = [w for w in filtered_workers if w in required]
        
        return filtered_workers
    
    def _classify_task_type(self, task: Dict[str, Any]) -> str:
        """Classify task type for affinity-based distribution."""
        context = task.get('context', {})
        
        # Simple classification based on context
        if 'optimization' in str(context).lower():
            return 'optimization'
        elif 'exploration' in str(context).lower():
            return 'exploration'
        elif 'battle' in str(context).lower():
            return 'battle'
        elif 'movement' in str(context).lower():
            return 'movement'
        else:
            return 'general'
    
    def _estimate_task_complexity(self, task: Dict[str, Any]) -> float:
        """Estimate task complexity for load balancing."""
        # Create task signature for caching
        task_signature = str(hash(str(sorted(task.items()))))
        
        if task_signature in self.task_complexity_cache:
            return self.task_complexity_cache[task_signature]
        
        complexity = 0.5  # Base complexity
        
        # Factor in context complexity
        context = task.get('context', {})
        context_str = str(context)
        
        # More complex contexts indicate higher complexity
        if len(context_str) > 100:
            complexity += 0.2
        
        if 'complex' in context_str.lower() or 'difficult' in context_str.lower():
            complexity += 0.3
        
        # Factor in estimated duration
        if 'estimated_duration_ms' in task:
            duration_factor = min(0.3, task['estimated_duration_ms'] / 10000.0)  # Max 0.3 for 10s+
            complexity += duration_factor
        
        complexity = min(1.0, max(0.1, complexity))
        
        # Cache result
        self.task_complexity_cache[task_signature] = complexity
        
        return complexity
    
    def _update_worker_load(self, worker_id: str, task: Dict[str, Any]) -> None:
        """Update worker load when task is assigned."""
        if worker_id not in self.current_worker_loads:
            self.current_worker_loads[worker_id] = 0.0
        
        task_complexity = self._estimate_task_complexity(task)
        load_increase = task_complexity * 0.3
        
        self.current_worker_loads[worker_id] = min(1.0, 
            self.current_worker_loads[worker_id] + load_increase
        )
    
    def _find_optimal_worker_for_batch(self, task: Dict[str, Any], 
                                     workers: List[str], 
                                     current_loads: Dict[str, float]) -> Optional[str]:
        """Find optimal worker considering current batch load distribution."""
        best_worker = None
        best_score = -1
        
        for worker in workers:
            # Consider multiple factors
            base_performance = self.worker_success_rates.get(worker, 0.5)
            current_load = current_loads.get(worker, 0.0)
            load_penalty = current_load * 0.4
            
            # Task affinity
            task_type = self._classify_task_type(task)
            if (task_type in self.task_type_performance and 
                worker in self.task_type_performance[task_type]):
                type_performance = self.task_type_performance[task_type][worker]
                affinity_bonus = (sum(type_performance) / len(type_performance) - 0.5) * 0.2
            else:
                affinity_bonus = 0.0
            
            score = base_performance - load_penalty + affinity_bonus
            
            if score > best_score:
                best_score = score
                best_worker = worker
        
        return best_worker
    
    def _calculate_load_variance(self, worker_loads: Dict[str, float]) -> float:
        """Calculate variance in worker loads."""
        if len(worker_loads) <= 1:
            return 0.0
        
        loads = list(worker_loads.values())
        mean_load = sum(loads) / len(loads)
        variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
        
        return variance
    
    def _record_distribution_metrics(self, strategy: str, distribution_time: float) -> None:
        """Record distribution metrics for performance monitoring."""
        self.distribution_metrics['total_distributions'] += 1
        self.distribution_metrics['distribution_times'].append(distribution_time)
        
        # Keep only recent times
        if len(self.distribution_metrics['distribution_times']) > 100:
            self.distribution_metrics['distribution_times'] = (
                self.distribution_metrics['distribution_times'][-50:]
            )
        
        # Track strategy usage
        if strategy not in self.distribution_metrics['strategy_success_rates']:
            self.distribution_metrics['strategy_success_rates'][strategy] = []
    
    def _get_empty_distribution_result(self, reason: str, 
                                     error_details: Optional[str] = None) -> Dict[str, Any]:
        """Get empty distribution result with error information."""
        return {
            'selected_worker': None,
            'confidence': 0.0,
            'selection_reason': 'error',
            'error_reason': reason,
            'error_details': error_details,
            'distribution_success': False
        }
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get distribution performance statistics."""
        distribution_times = self.distribution_metrics['distribution_times']
        
        if not distribution_times:
            return {'status': 'no_data'}
        
        avg_time_ms = (sum(distribution_times) / len(distribution_times)) * 1000
        max_time_ms = max(distribution_times) * 1000
        
        # Worker load statistics
        load_stats = {}
        if self.current_worker_loads:
            loads = list(self.current_worker_loads.values())
            load_stats = {
                'average_load': sum(loads) / len(loads),
                'max_load': max(loads),
                'min_load': min(loads),
                'load_variance': self._calculate_load_variance(self.current_worker_loads)
            }
        
        return {
            'total_distributions': self.distribution_metrics['total_distributions'],
            'average_distribution_time_ms': avg_time_ms,
            'max_distribution_time_ms': max_time_ms,
            'load_statistics': load_stats,
            'strategy_usage': self.distribution_metrics['strategy_success_rates'],
            'worker_performance_summary': {
                worker: {
                    'success_rate': self.worker_success_rates.get(worker, 0.5),
                    'avg_completion_time': (
                        sum(times) / len(times) if times else 0.0
                        for times in [self.worker_task_completion_times.get(worker, [])]
                    )
                }
                for worker in self.current_worker_loads.keys()
            }
        }
