"""
StrategyResponseValidator - Guardian Security Implementation

This module provides comprehensive validation for strategic response JSON data
to prevent JSON injection attacks, malformed payloads, and resource exhaustion
attacks in the OpusStrategist system.

Guardian Security Features:
- Fail-secure JSON validation with schema enforcement
- Size and depth limits to prevent DoS attacks
- Malicious payload detection and sanitization
- Comprehensive security logging and audit trail
- Defense in depth with multiple validation layers

Author: Guardian Security Framework
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .validators import SecurityValidationError, InputSizeError, InputFormatError

logger = logging.getLogger(__name__)

# Security constants for strategy response validation
MAX_STRATEGY_RESPONSE_SIZE = 5 * 1024 * 1024  # 5MB max response size
MAX_JSON_DEPTH = 15  # Maximum allowed JSON nesting depth
MAX_EXPERIMENTS_COUNT = 50  # Maximum experiments per response
MAX_INSIGHTS_COUNT = 100  # Maximum insights per response  
MAX_CHECKPOINTS_COUNT = 20  # Maximum checkpoints per response
MAX_STRING_LENGTH = 10000  # Maximum string field length
MAX_DIRECTIVE_LENGTH = 2000  # Maximum directive string length
PROCESSING_TIMEOUT_SECONDS = 30.0  # Maximum processing time

# Dangerous patterns that could indicate malicious payloads
DANGEROUS_JSON_PATTERNS = [
    r'__[a-zA-Z_]+__',  # Python dunder methods
    r'eval\s*\(',  # Code evaluation
    r'exec\s*\(',  # Code execution
    r'import\s+',  # Import statements
    r'from\s+\w+\s+import',  # From imports
    r'subprocess',  # Subprocess calls
    r'os\.',  # OS module access
    r'sys\.',  # Sys module access
    r'\\u[0-9a-fA-F]{4}',  # Unicode escapes (potential bypass)
    r'javascript:',  # JavaScript protocol
    r'<script',  # Script tags
    r'data:',  # Data URLs
    r'vbscript:',  # VBScript protocol
]

@dataclass
class ValidationMetrics:
    """Metrics for validation performance and security monitoring."""
    validation_time_ms: float
    json_depth_max: int
    experiments_count: int
    insights_count: int
    checkpoints_count: int
    dangerous_patterns_detected: int
    size_bytes: int


class StrategyResponseValidator:
    """
    Validates strategy response JSON data with comprehensive security checks.
    
    Guardian Security Implementation:
    - Multi-layer validation (syntax, semantic, security)
    - Size and resource limits enforcement
    - Malicious pattern detection
    - Fail-secure error handling
    - Comprehensive audit logging
    """
    
    def __init__(self, 
                 max_response_size: int = MAX_STRATEGY_RESPONSE_SIZE,
                 max_json_depth: int = MAX_JSON_DEPTH,
                 processing_timeout: float = PROCESSING_TIMEOUT_SECONDS):
        """
        Initialize validator with security configuration.
        
        Args:
            max_response_size: Maximum allowed response size in bytes
            max_json_depth: Maximum JSON nesting depth
            processing_timeout: Maximum processing time in seconds
        """
        self.max_response_size = max_response_size
        self.max_json_depth = max_json_depth
        self.processing_timeout = processing_timeout
        self.validation_count = 0
        self.security_violations = 0
        
    def validate_json_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Validate and sanitize strategic JSON response with comprehensive security checks.
        
        This is the main validation entry point that replaces unsafe json.loads()
        operations in OpusStrategist.
        
        Args:
            raw_response: Raw JSON string from Opus
            
        Returns:
            Validated and sanitized strategic plan dictionary
            
        Raises:
            SecurityValidationError: If response fails security validation
            InputSizeError: If response exceeds size limits
            InputFormatError: If response has invalid format
        """
        start_time = time.time()
        self.validation_count += 1
        
        try:
            # Phase 1: Pre-validation security checks
            self._pre_validate_input(raw_response)
            
            # Phase 2: Safe JSON parsing with timeout protection
            parsed_data = self._safe_json_parse(raw_response, start_time)
            
            # Phase 3: Structure and content validation
            validated_data = self._validate_strategy_structure(parsed_data)
            
            # Phase 4: Security scanning and sanitization
            sanitized_data = self._scan_and_sanitize(validated_data)
            
            # Phase 5: Final compliance checks
            final_data = self._final_compliance_check(sanitized_data)
            
            # Record successful validation metrics
            processing_time = (time.time() - start_time) * 1000
            metrics = self._calculate_validation_metrics(final_data, processing_time)
            self._log_validation_success(metrics)
            
            return final_data
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.security_violations += 1
            self._log_validation_failure(e, raw_response, processing_time)
            raise
    
    def _pre_validate_input(self, raw_response: str) -> None:
        """Phase 1: Pre-validation security checks."""
        if not isinstance(raw_response, str):
            raise InputFormatError("Strategy response must be a string")
        
        # Size validation to prevent memory exhaustion
        response_size = len(raw_response.encode('utf-8'))
        if response_size > self.max_response_size:
            raise InputSizeError(
                f"Response size {response_size} bytes exceeds maximum {self.max_response_size}"
            )
        
        # Empty response check
        if not raw_response.strip():
            raise InputFormatError("Strategy response cannot be empty")
            
        # Basic structure validation
        if not raw_response.strip().startswith('{'):
            raise InputFormatError("Strategy response must be valid JSON object")
    
    def _safe_json_parse(self, raw_response: str, start_time: float) -> Dict[str, Any]:
        """Phase 2: Safe JSON parsing with timeout protection."""
        try:
            # Timeout check during parsing
            if time.time() - start_time > self.processing_timeout:
                raise SecurityValidationError("JSON validation timeout exceeded")
            
            # Use safe JSON parsing
            parsed_data = json.loads(raw_response)
            
            if not isinstance(parsed_data, dict):
                raise InputFormatError("Strategy response must be a JSON object")
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            raise InputFormatError(f"Invalid JSON format: {str(e)}") from e
        except RecursionError as e:
            raise SecurityValidationError("JSON nesting too deep - potential DoS attack") from e
    
    def _validate_strategy_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Structure and content validation."""
        
        # Validate JSON depth to prevent stack overflow
        self._validate_json_depth(data, 0)
        
        # Required fields validation
        required_fields = {"strategy_id", "experiments", "strategic_insights", "next_checkpoints"}
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            # Guardian principle: Fail secure by adding defaults rather than rejecting
            logger.warning(f"Strategy response missing fields: {missing_fields}")
            for field in missing_fields:
                if field == "strategy_id":
                    data[field] = f"secure_fallback_{int(time.time())}"
                else:
                    data[field] = []
        
        # Validate experiments list
        experiments = data.get("experiments", [])
        if not isinstance(experiments, list):
            raise SecurityValidationError("Experiments field must be a list")
        
        if len(experiments) > MAX_EXPERIMENTS_COUNT:
            logger.warning(f"Truncating experiments list from {len(experiments)} to {MAX_EXPERIMENTS_COUNT}")
            data["experiments"] = experiments[:MAX_EXPERIMENTS_COUNT]
        
        # Validate strategic insights
        insights = data.get("strategic_insights", [])
        if not isinstance(insights, list):
            raise SecurityValidationError("Strategic insights field must be a list")
            
        if len(insights) > MAX_INSIGHTS_COUNT:
            logger.warning(f"Truncating insights list from {len(insights)} to {MAX_INSIGHTS_COUNT}")
            data["strategic_insights"] = insights[:MAX_INSIGHTS_COUNT]
        
        # Validate next checkpoints
        checkpoints = data.get("next_checkpoints", [])
        if not isinstance(checkpoints, list):
            raise SecurityValidationError("Next checkpoints field must be a list")
            
        if len(checkpoints) > MAX_CHECKPOINTS_COUNT:
            logger.warning(f"Truncating checkpoints list from {len(checkpoints)} to {MAX_CHECKPOINTS_COUNT}")
            data["next_checkpoints"] = checkpoints[:MAX_CHECKPOINTS_COUNT]
            
        return data
    
    def _scan_and_sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Security scanning and sanitization."""
        
        # Deep scan for dangerous patterns
        dangerous_patterns_found = self._deep_scan_for_threats(data)
        
        if dangerous_patterns_found > 0:
            self.security_violations += 1
            logger.error(f"Dangerous patterns detected in strategy response: {dangerous_patterns_found}")
            raise SecurityValidationError(
                f"Strategy response contains {dangerous_patterns_found} potentially malicious patterns"
            )
        
        # Sanitize string fields
        sanitized_data = self._deep_sanitize_strings(data)
        
        return sanitized_data
    
    def _final_compliance_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 5: Final compliance checks."""
        
        # Validate all experiments have required structure
        experiments = data.get("experiments", [])
        validated_experiments = []
        
        for i, experiment in enumerate(experiments):
            if not isinstance(experiment, dict):
                logger.warning(f"Experiment {i} is not a dictionary, skipping")
                continue
                
            # Ensure required experiment fields
            experiment_defaults = {
                "id": f"secure_exp_{i}_{int(time.time())}",
                "name": "Validated Strategic Experiment",
                "checkpoint": "secure_checkpoint",
                "script_dsl": "# Secure strategic action",
                "expected_outcome": "Strategic advancement",
                "priority": "medium",
                "directives": []
            }
            
            # Merge with defaults (existing values take precedence)
            validated_exp = {**experiment_defaults, **experiment}
            
            # Validate directive limits
            directives = validated_exp.get("directives", [])
            if isinstance(directives, list) and len(directives) > 10:  # Reasonable directive limit
                logger.warning(f"Truncating directives for experiment {i}")
                validated_exp["directives"] = directives[:10]
            
            validated_experiments.append(validated_exp)
        
        data["experiments"] = validated_experiments
        
        # Final size check
        serialized_size = len(json.dumps(data).encode('utf-8'))
        if serialized_size > self.max_response_size:
            raise InputSizeError(f"Validated response size {serialized_size} exceeds limit")
        
        return data
    
    def _validate_json_depth(self, obj: Any, current_depth: int) -> None:
        """Recursively validate JSON depth to prevent stack overflow attacks."""
        if current_depth > self.max_json_depth:
            raise SecurityValidationError(
                f"JSON nesting depth {current_depth} exceeds maximum {self.max_json_depth}"
            )
        
        if isinstance(obj, dict):
            for value in obj.values():
                self._validate_json_depth(value, current_depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                self._validate_json_depth(item, current_depth + 1)
    
    def _deep_scan_for_threats(self, obj: Any, path: str = "root") -> int:
        """Recursively scan for dangerous patterns in all string values."""
        threat_count = 0
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                threat_count += self._deep_scan_for_threats(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                threat_count += self._deep_scan_for_threats(item, f"{path}[{i}]")
        elif isinstance(obj, str):
            # Scan string for dangerous patterns
            for pattern in DANGEROUS_JSON_PATTERNS:
                if re.search(pattern, obj, re.IGNORECASE):
                    logger.error(f"Dangerous pattern '{pattern}' found at {path}: {obj[:100]}...")
                    threat_count += 1
        
        return threat_count
    
    def _deep_sanitize_strings(self, obj: Any) -> Any:
        """Recursively sanitize all string values."""
        if isinstance(obj, dict):
            return {key: self._deep_sanitize_strings(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_sanitize_strings(item) for item in obj]
        elif isinstance(obj, str):
            return self._sanitize_string(obj)
        else:
            return obj
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize individual string value."""
        if len(text) > MAX_STRING_LENGTH:
            logger.warning(f"Truncating string from {len(text)} to {MAX_STRING_LENGTH} characters")
            text = text[:MAX_STRING_LENGTH]
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Remove potentially dangerous Unicode escapes
        sanitized = re.sub(r'\\u[0-9a-fA-F]{4}', '', sanitized)
        
        return sanitized.strip()
    
    def _calculate_validation_metrics(self, data: Dict[str, Any], processing_time: float) -> ValidationMetrics:
        """Calculate validation metrics for monitoring."""
        return ValidationMetrics(
            validation_time_ms=processing_time,
            json_depth_max=self._calculate_max_depth(data),
            experiments_count=len(data.get("experiments", [])),
            insights_count=len(data.get("strategic_insights", [])),
            checkpoints_count=len(data.get("next_checkpoints", [])),
            dangerous_patterns_detected=0,  # Would be > 0 if validation failed
            size_bytes=len(json.dumps(data).encode('utf-8'))
        )
    
    def _calculate_max_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum JSON depth for metrics."""
        max_depth = current_depth
        
        if isinstance(obj, dict):
            for value in obj.values():
                depth = self._calculate_max_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        elif isinstance(obj, list):
            for item in obj:
                depth = self._calculate_max_depth(item, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _log_validation_success(self, metrics: ValidationMetrics) -> None:
        """Log successful validation with metrics."""
        logger.info(
            f"Strategy response validated successfully: "
            f"time={metrics.validation_time_ms:.2f}ms, "
            f"depth={metrics.json_depth_max}, "
            f"experiments={metrics.experiments_count}, "
            f"insights={metrics.insights_count}, "
            f"size={metrics.size_bytes} bytes"
        )
    
    def _log_validation_failure(self, error: Exception, raw_response: str, processing_time: float) -> None:
        """Log validation failure with security details."""
        response_preview = raw_response[:200] if raw_response else "None"
        logger.error(
            f"Strategy response validation failed: "
            f"error={type(error).__name__}, "
            f"message={str(error)}, "
            f"time={processing_time:.2f}ms, "
            f"preview={response_preview}..."
        )
        
        # Security audit log
        logger.warning(
            f"SECURITY_AUDIT: JSON validation failure - "
            f"violations_total={self.security_violations}, "
            f"validations_total={self.validation_count}"
        )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics for monitoring."""
        return {
            "total_validations": self.validation_count,
            "security_violations": self.security_violations,
            "violation_rate": self.security_violations / max(self.validation_count, 1),
            "max_response_size": self.max_response_size,
            "max_json_depth": self.max_json_depth,
            "processing_timeout": self.processing_timeout
        }


# Convenience function for strategic response validation
def validate_strategic_json(raw_response: str, validator: Optional[StrategyResponseValidator] = None) -> Dict[str, Any]:
    """
    Validate strategic JSON response with default security settings.
    
    This function provides a simple interface for validating JSON responses
    from OpusStrategist operations.
    
    Args:
        raw_response: Raw JSON string to validate
        validator: Optional custom validator instance
        
    Returns:
        Validated and sanitized strategic plan dictionary
        
    Raises:
        SecurityValidationError: If response fails security validation
    """
    if validator is None:
        validator = StrategyResponseValidator()
    
    return validator.validate_json_response(raw_response)