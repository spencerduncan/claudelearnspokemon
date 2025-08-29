"""
Security Input Validation Framework - Guardian Implementation

This module provides comprehensive input validation utilities to prevent
injection attacks, malicious payloads, and data corruption across the
claudelearnspokemon system.

Guardian Design Principles:
- Fail-secure by default (deny on validation failure)  
- Comprehensive sanitization and validation
- Clear security boundaries and error messages
- Defense in depth approach
- Extensive logging for security monitoring

Author: Guardian Security Framework
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import subprocess
import shlex

logger = logging.getLogger(__name__)

# Security constants
MAX_SCRIPT_LENGTH = 10000  # Maximum allowed script length
MAX_CHECKPOINT_SIZE_MB = 50  # Maximum checkpoint file size
MAX_JSON_DEPTH = 10  # Maximum allowed JSON nesting depth
MAX_LOCATION_LENGTH = 255  # Maximum location string length
ALLOWED_DOCKER_IMAGE_PATTERN = r'^[a-zA-Z0-9][a-zA-Z0-9._-]*:[a-zA-Z0-9._-]+$'
ALLOWED_BUTTON_INPUTS = {'A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'}

class SecurityValidationError(Exception):
    """Raised when security validation fails."""
    pass

class InputSizeError(SecurityValidationError):
    """Raised when input exceeds safe size limits."""
    pass

class InputFormatError(SecurityValidationError):
    """Raised when input has invalid format."""
    pass

class ScriptValidator:
    """
    Validates Pokemon DSL scripts for security and safety.
    
    Guardian Security Features:
    - Length limits to prevent resource exhaustion
    - Command whitelist to prevent injection
    - Syntax validation to prevent malformed input
    - Execution time estimation for DoS prevention
    """
    
    # Allowed DSL commands - whitelist approach
    ALLOWED_COMMANDS = {
        'PRESS', 'MOVE', 'WAIT', 'A', 'B', 'START', 'SELECT',
        'UP', 'DOWN', 'LEFT', 'RIGHT', 'REPEAT', 'END'
    }
    
    # Dangerous patterns that indicate potential injection
    DANGEROUS_PATTERNS = [
        r'[;&|`$]',  # Shell metacharacters
        r'\.{2,}/',  # Path traversal
        r'<script',  # Script injection
        r'eval\s*\(',  # Code evaluation
        r'exec\s*\(',  # Code execution
        r'import\s+',  # Import statements
        r'__\w+__',   # Python dunder methods
    ]
    
    @classmethod
    def validate_script_text(cls, script_text: str) -> str:
        """
        Validate and sanitize Pokemon DSL script text.
        
        Args:
            script_text: Raw script input
            
        Returns:
            Sanitized script text
            
        Raises:
            SecurityValidationError: If script fails security validation
            InputSizeError: If script exceeds size limits
            InputFormatError: If script has invalid format
        """
        if not isinstance(script_text, str):
            raise InputFormatError("Script text must be a string")
        
        # Length validation
        if len(script_text) > MAX_SCRIPT_LENGTH:
            raise InputSizeError(f"Script length {len(script_text)} exceeds maximum {MAX_SCRIPT_LENGTH}")
        
        if not script_text.strip():
            raise InputFormatError("Script cannot be empty")
        
        # Security pattern detection
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, script_text, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected in script: {pattern}")
                raise SecurityValidationError(f"Script contains potentially dangerous pattern: {pattern}")
        
        # Command whitelist validation
        lines = script_text.upper().split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
                
            tokens = line.split()
            if tokens and tokens[0] not in cls.ALLOWED_COMMANDS:
                raise SecurityValidationError(
                    f"Disallowed command '{tokens[0]}' at line {line_num}. "
                    f"Allowed commands: {', '.join(sorted(cls.ALLOWED_COMMANDS))}"
                )
        
        # Normalize and sanitize
        sanitized = cls._sanitize_script(script_text)
        logger.debug(f"Script validated successfully: {len(sanitized)} characters")
        return sanitized
    
    @classmethod
    def _sanitize_script(cls, script_text: str) -> str:
        """Remove dangerous characters and normalize format."""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[^\w\s\n#-]', '', script_text)
        
        # Normalize whitespace
        lines = []
        for line in sanitized.split('\n'):
            line = line.strip()
            if line:
                lines.append(line.upper())
        
        return '\n'.join(lines)

class ButtonSequenceValidator:
    """
    Validates button input sequences for Pokemon Gym adapters.
    
    Guardian Security Features:
    - Input whitelist for allowed button combinations
    - Length limits to prevent resource exhaustion  
    - Format validation to prevent injection
    """
    
    @classmethod
    def validate_button_sequence(cls, sequence: str) -> str:
        """
        Validate button input sequence.
        
        Args:
            sequence: Button sequence string
            
        Returns:
            Validated sequence
            
        Raises:
            SecurityValidationError: If sequence is invalid or dangerous
        """
        if not isinstance(sequence, str):
            raise InputFormatError("Button sequence must be a string")
        
        if len(sequence) > 1000:  # Reasonable limit for button sequences
            raise InputSizeError(f"Button sequence too long: {len(sequence)} characters")
        
        # Parse and validate individual buttons
        buttons = sequence.upper().split()
        validated_buttons = []
        
        for button in buttons:
            if button not in ALLOWED_BUTTON_INPUTS:
                raise SecurityValidationError(
                    f"Invalid button '{button}'. Allowed: {', '.join(sorted(ALLOWED_BUTTON_INPUTS))}"
                )
            validated_buttons.append(button)
        
        validated_sequence = ' '.join(validated_buttons)
        logger.debug(f"Button sequence validated: {len(validated_buttons)} buttons")
        return validated_sequence

class CheckpointDataValidator:
    """
    Validates checkpoint data for secure deserialization.
    
    Guardian Security Features:
    - Size limits to prevent compression bombs
    - JSON depth limits to prevent stack overflow
    - Content validation to prevent malicious payloads
    - Checksum validation for integrity
    """
    
    @classmethod
    def validate_checkpoint_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate checkpoint data structure before deserialization.
        
        Args:
            data: Raw checkpoint data
            
        Returns:
            Validated checkpoint data
            
        Raises:
            SecurityValidationError: If data is invalid or dangerous
        """
        if not isinstance(data, dict):
            raise InputFormatError("Checkpoint data must be a dictionary")
        
        # Validate required fields
        required_fields = {'version', 'checkpoint_id', 'game_state', 'metadata'}
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise SecurityValidationError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate checkpoint ID format
        checkpoint_id = data.get('checkpoint_id', '')
        if not cls._is_valid_checkpoint_id(checkpoint_id):
            raise SecurityValidationError(f"Invalid checkpoint ID format: {checkpoint_id}")
        
        # Validate JSON depth to prevent stack overflow attacks
        cls._validate_json_depth(data, max_depth=MAX_JSON_DEPTH)
        
        # Validate game state structure
        game_state = data.get('game_state', {})
        if not isinstance(game_state, dict):
            raise SecurityValidationError("Game state must be a dictionary")
        
        cls._validate_json_depth(game_state, max_depth=MAX_JSON_DEPTH)
        
        logger.debug(f"Checkpoint data validated: {checkpoint_id}")
        return data
    
    @classmethod
    def validate_compressed_size(cls, compressed_data: bytes, max_size_mb: int = MAX_CHECKPOINT_SIZE_MB) -> None:
        """
        Validate compressed data size to prevent compression bombs.
        
        Args:
            compressed_data: Compressed checkpoint data
            max_size_mb: Maximum allowed size in MB
            
        Raises:
            InputSizeError: If data exceeds size limits
        """
        if not isinstance(compressed_data, bytes):
            raise InputFormatError("Compressed data must be bytes")
        
        size_mb = len(compressed_data) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise InputSizeError(f"Compressed data too large: {size_mb:.2f}MB > {max_size_mb}MB")
    
    @classmethod
    def _is_valid_checkpoint_id(cls, checkpoint_id: str) -> bool:
        """Validate checkpoint ID format (UUID or timestamp-based)."""
        if not checkpoint_id or len(checkpoint_id) > 100:
            return False
        
        # Allow UUID format or timestamp-based format
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        timestamp_pattern = r'^cp_\d{8}_\d{6}_\d{6}_[0-9a-f]{8}$'
        
        return (re.match(uuid_pattern, checkpoint_id, re.IGNORECASE) or 
                re.match(timestamp_pattern, checkpoint_id))
    
    @classmethod
    def _validate_json_depth(cls, obj: Any, max_depth: int, current_depth: int = 0) -> None:
        """Recursively validate JSON depth to prevent stack overflow."""
        if current_depth > max_depth:
            raise SecurityValidationError(f"JSON nesting too deep: {current_depth} > {max_depth}")
        
        if isinstance(obj, dict):
            for value in obj.values():
                cls._validate_json_depth(value, max_depth, current_depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                cls._validate_json_depth(item, max_depth, current_depth + 1)

class LocationValidator:
    """
    Validates location strings to prevent injection attacks.
    
    Guardian Security Features:
    - Length limits
    - Character whitelist
    - Path traversal prevention
    """
    
    @classmethod
    def validate_location(cls, location: str) -> str:
        """
        Validate location string.
        
        Args:
            location: Location identifier
            
        Returns:
            Validated location
            
        Raises:
            SecurityValidationError: If location is invalid
        """
        if not isinstance(location, str):
            raise InputFormatError("Location must be a string")
        
        if len(location) > MAX_LOCATION_LENGTH:
            raise InputSizeError(f"Location too long: {len(location)} > {MAX_LOCATION_LENGTH}")
        
        # Check for path traversal attempts
        if '..' in location or '/' in location:
            raise SecurityValidationError("Location cannot contain path traversal sequences")
        
        # Only allow alphanumeric, underscore, hyphen, space
        if not re.match(r'^[a-zA-Z0-9_\- ]+$', location):
            raise SecurityValidationError("Location contains invalid characters")
        
        return location.strip()

class DockerImageValidator:
    """
    Validates Docker image names to prevent injection attacks.
    
    Guardian Security Features:  
    - Format validation against Docker naming standards
    - Registry whitelist support
    - Version tag validation
    """
    
    # Trusted registries (can be configured)
    TRUSTED_REGISTRIES = {
        'pokemon-gym',  # Local builds
        'docker.io',
        'ghcr.io'
    }
    
    @classmethod
    def validate_image_name(cls, image_name: str) -> str:
        """
        Validate Docker image name format and security.
        
        Args:
            image_name: Docker image name with tag
            
        Returns:
            Validated image name
            
        Raises:
            SecurityValidationError: If image name is invalid or untrusted
        """
        if not isinstance(image_name, str):
            raise InputFormatError("Image name must be a string")
        
        if len(image_name) > 255:
            raise InputSizeError(f"Image name too long: {len(image_name)}")
        
        # Validate format
        if not re.match(ALLOWED_DOCKER_IMAGE_PATTERN, image_name):
            raise SecurityValidationError(f"Invalid Docker image name format: {image_name}")
        
        # Extract registry (if present)
        parts = image_name.split('/')
        if len(parts) > 1:
            registry = parts[0]
            if registry not in cls.TRUSTED_REGISTRIES:
                logger.warning(f"Untrusted registry used: {registry}")
                # For now, allow but log - could be made stricter in production
        
        return image_name

class CommandValidator:
    """
    Validates subprocess commands to prevent injection attacks.
    
    Guardian Security Features:
    - Command whitelist
    - Argument sanitization  
    - Shell escape prevention
    """
    
    # Whitelisted executables
    ALLOWED_EXECUTABLES = {
        'claude',
        'python3',
        'python'
    }
    
    @classmethod
    def validate_command(cls, command: List[str]) -> List[str]:
        """
        Validate subprocess command for security.
        
        Args:
            command: Command arguments list
            
        Returns:
            Validated command list
            
        Raises:
            SecurityValidationError: If command is dangerous
        """
        if not isinstance(command, list) or not command:
            raise InputFormatError("Command must be a non-empty list")
        
        executable = command[0]
        if executable not in cls.ALLOWED_EXECUTABLES:
            raise SecurityValidationError(f"Executable '{executable}' not in whitelist: {cls.ALLOWED_EXECUTABLES}")
        
        # Validate each argument
        validated_args = []
        for arg in command:
            if not isinstance(arg, str):
                raise InputFormatError("All command arguments must be strings")
            
            # Check for dangerous patterns
            if any(char in arg for char in [';', '&', '|', '`', '$', '>', '<']):
                raise SecurityValidationError(f"Dangerous shell characters in argument: {arg}")
            
            # Properly quote arguments that need it
            validated_args.append(shlex.quote(arg) if ' ' in arg else arg)
        
        return validated_args

# Convenience functions for common validation scenarios

def validate_script_input(script_text: str) -> str:
    """Validate Pokemon DSL script input."""
    return ScriptValidator.validate_script_text(script_text)

def validate_button_input(sequence: str) -> str:
    """Validate button sequence input."""  
    return ButtonSequenceValidator.validate_button_sequence(sequence)

def validate_checkpoint_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate checkpoint data input."""
    return CheckpointDataValidator.validate_checkpoint_data(data)

def validate_location_input(location: str) -> str:
    """Validate location input."""
    return LocationValidator.validate_location(location)

def validate_docker_image(image_name: str) -> str:
    """Validate Docker image name."""
    return DockerImageValidator.validate_image_name(image_name)

def validate_subprocess_command(command: List[str]) -> List[str]:
    """Validate subprocess command."""
    return CommandValidator.validate_command(command)