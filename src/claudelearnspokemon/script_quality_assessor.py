"""
Script Quality Assessment and Pattern Refinement for Pokémon Learning System.

This module provides quality assessment capabilities for generated scripts
and pattern refinement based on performance feedback.

Design follows Clean Code principles:
- Single Responsibility: Quality assessment and pattern refinement
- Open/Closed: Extensible for new assessment criteria
- Interface Segregation: Clear separation of concerns
"""

import random
import re
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ScriptQualityAssessor:
    """
    Assesses the quality of generated scripts using multiple evaluation criteria.
    
    Evaluates scripts based on:
    - Syntax correctness
    - Command diversity
    - Pattern complexity
    - Execution safety
    """
    
    def __init__(self, script_compiler=None):
        """
        Initialize script quality assessor.
        
        Args:
            script_compiler: Optional script compiler for syntax validation
        """
        self.script_compiler = script_compiler
        
        # Quality assessment weights
        self.weights = {
            'syntax_correctness': 0.3,
            'command_diversity': 0.25,
            'pattern_complexity': 0.25,
            'execution_safety': 0.2
        }
    
    def assess_script_quality(self, script_text: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality of a generated script.
        
        Args:
            script_text: The script content to assess
            task: Task context for assessment
            
        Returns:
            Assessment results including quality_score and detailed metrics
        """
        if not script_text or not script_text.strip():
            return {
                'quality_score': 0.0,
                'syntax_score': 0.0,
                'diversity_score': 0.0,
                'complexity_score': 0.0,
                'safety_score': 0.0,
                'issues': ['Empty script content']
            }
        
        # Perform individual assessments
        syntax_score = self._assess_syntax_correctness(script_text)
        diversity_score = self._assess_command_diversity(script_text)
        complexity_score = self._assess_pattern_complexity(script_text, task)
        safety_score = self._assess_execution_safety(script_text)
        
        # Calculate weighted quality score
        quality_score = (
            self.weights['syntax_correctness'] * syntax_score +
            self.weights['command_diversity'] * diversity_score +
            self.weights['pattern_complexity'] * complexity_score +
            self.weights['execution_safety'] * safety_score
        )
        
        return {
            'quality_score': quality_score,
            'syntax_score': syntax_score,
            'diversity_score': diversity_score,
            'complexity_score': complexity_score,
            'safety_score': safety_score,
            'issues': self._identify_issues(script_text)
        }
    
    def _assess_syntax_correctness(self, script_text: str) -> float:
        """Assess script syntax correctness."""
        try:
            # Basic Python syntax validation
            compile(script_text, '<script>', 'exec')
            return 1.0
        except SyntaxError:
            # Check for common command patterns instead
            lines = [line.strip() for line in script_text.split('\n') if line.strip()]
            valid_lines = 0
            
            for line in lines:
                if self._is_valid_command_pattern(line):
                    valid_lines += 1
            
            return valid_lines / len(lines) if lines else 0.0
    
    def _assess_command_diversity(self, script_text: str) -> float:
        """Assess diversity of commands used in script."""
        lines = [line.strip() for line in script_text.split('\n') if line.strip()]
        if not lines:
            return 0.0
        
        # Extract command types (first word of each line)
        commands = []
        for line in lines:
            words = line.split()
            if words:
                commands.append(words[0].lower())
        
        unique_commands = len(set(commands))
        total_commands = len(commands)
        
        # Higher diversity score for more unique command types
        return min(unique_commands / max(total_commands * 0.7, 1), 1.0)
    
    def _assess_pattern_complexity(self, script_text: str, task: Dict[str, Any]) -> float:
        """Assess complexity and sophistication of patterns in script."""
        lines = [line.strip() for line in script_text.split('\n') if line.strip()]
        
        complexity_indicators = {
            'conditional_logic': len(re.findall(r'\bif\b|\belse\b|\belif\b', script_text.lower())),
            'loops': len(re.findall(r'\bfor\b|\bwhile\b', script_text.lower())),
            'functions': len(re.findall(r'\bdef\b|\blambda\b', script_text.lower())),
            'data_structures': len(re.findall(r'\[|\{|\(', script_text)),
            'length': len(lines)
        }
        
        # Normalize complexity score
        complexity_score = min(sum(complexity_indicators.values()) / 20.0, 1.0)
        return complexity_score
    
    def _assess_execution_safety(self, script_text: str) -> float:
        """Assess execution safety of script."""
        dangerous_patterns = [
            r'\brm\s+-rf\b', r'\bformat\s+c:\b', r'\bdel\s+/\b',
            r'\bsudo\s+rm\b', r'\bshutil\.rmtree\b', r'\bos\.system\b'
        ]
        
        safety_score = 1.0
        for pattern in dangerous_patterns:
            if re.search(pattern, script_text, re.IGNORECASE):
                safety_score -= 0.3
        
        return max(safety_score, 0.0)
    
    def _is_valid_command_pattern(self, line: str) -> bool:
        """Check if line matches valid command patterns."""
        common_patterns = [
            r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=',  # Variable assignment
            r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\(',  # Function call
            r'^(if|else|elif|for|while|def|class|import|from)\b',  # Keywords
            r'^#',  # Comments
            r'^print\s*\(',  # Print statements
            r'^return\b',  # Return statements
        ]
        
        return any(re.match(pattern, line) for pattern in common_patterns)
    
    def _identify_issues(self, script_text: str) -> List[str]:
        """Identify potential issues in the script."""
        issues = []
        
        if len(script_text.strip()) < 10:
            issues.append("Script is too short")
        
        if not any(line.strip() for line in script_text.split('\n')):
            issues.append("Script contains only empty lines")
        
        return issues


class PatternRefiner:
    """
    Refines patterns based on quality assessment feedback.
    
    Implements pattern learning and adaptation based on script performance
    and quality metrics from the quality assessor.
    """
    
    def __init__(self, quality_assessor: ScriptQualityAssessor):
        """
        Initialize pattern refiner with quality assessor dependency.
        
        Args:
            quality_assessor: ScriptQualityAssessor instance for evaluation
        """
        self.quality_assessor = quality_assessor
        self.pattern_cache = {}
        self.refinement_history = []
    
    def refine_patterns(self, patterns: List[Dict[str, Any]], feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Refine patterns based on performance feedback.
        
        Args:
            patterns: List of patterns to refine
            feedback: Performance feedback data
            
        Returns:
            Refined patterns with improved quality metrics
        """
        refined_patterns = []
        
        for pattern in patterns:
            refined_pattern = self._refine_single_pattern(pattern, feedback)
            refined_patterns.append(refined_pattern)
        
        # Record refinement for learning
        self.refinement_history.append({
            'input_patterns': len(patterns),
            'output_patterns': len(refined_patterns),
            'feedback_quality': feedback.get('quality_score', 0.0)
        })
        
        return refined_patterns
    
    def _refine_single_pattern(self, pattern: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Refine a single pattern based on feedback."""
        refined_pattern = pattern.copy()
        
        # Apply refinement based on feedback quality
        quality_score = feedback.get('quality_score', 0.0)
        
        if quality_score < 0.5:
            # Low quality - add diversity
            refined_pattern['diversity_boost'] = True
            refined_pattern['complexity_target'] = 'increased'
        elif quality_score > 0.8:
            # High quality - maintain stability
            refined_pattern['stability_priority'] = True
        else:
            # Medium quality - balanced refinement
            refined_pattern['balanced_approach'] = True
        
        return refined_pattern
    
    def get_refinement_statistics(self) -> Dict[str, Any]:
        """Get statistics about pattern refinement performance."""
        if not self.refinement_history:
            return {'total_refinements': 0}
        
        total_refinements = len(self.refinement_history)
        avg_input_patterns = sum(r['input_patterns'] for r in self.refinement_history) / total_refinements
        avg_quality_improvement = sum(r['feedback_quality'] for r in self.refinement_history) / total_refinements
        
        return {
            'total_refinements': total_refinements,
            'average_input_patterns': avg_input_patterns,
            'average_quality_score': avg_quality_improvement,
            'refinement_trend': 'improving' if avg_quality_improvement > 0.6 else 'stable'
        }


__all__ = ["ScriptQualityAssessor", "PatternRefiner"]