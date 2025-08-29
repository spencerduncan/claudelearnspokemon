"""
Factory for creating and configuring Claude CLI processes.

This module handles process creation, command building, and configuration,
following the Single Responsibility Principle and Factory pattern to
separate creation concerns from process lifecycle management.
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from .prompts import ProcessType, PromptRepository
from .validators import validate_subprocess_command, SecurityValidationError

logger = logging.getLogger(__name__)


@dataclass
class ProcessConfig:
    """Configuration for Claude process initialization."""

    process_type: ProcessType
    model_name: str
    system_prompt: str
    max_retries: int = 3
    startup_timeout: float = 30.0
    health_check_interval: float = 5.0
    memory_limit_mb: int = 100

    # Performance optimization settings
    stdout_buffer_size: int = 8192  # 8KB
    stderr_buffer_size: int = 4096  # 4KB
    use_process_group: bool = True


class ProcessCommandBuilder:
    """
    Builds command-line arguments for Claude CLI processes.

    This class encapsulates the logic for constructing the correct
    command-line arguments based on process type and configuration.
    """

    # Model mappings for different process types
    MODEL_MAPPING = {
        ProcessType.OPUS_STRATEGIC: "claude-3-opus-20240229",
        ProcessType.SONNET_TACTICAL: "claude-3-5-sonnet-20241022",
    }

    @classmethod
    def build_command(cls, config: ProcessConfig) -> list[str]:
        """
        Build Claude CLI command based on process configuration.

        Args:
            config: Process configuration containing type and settings

        Returns:
            List of command arguments ready for subprocess execution

        Raises:
            ValueError: If process type is not supported
        """
        base_cmd = ["claude", "chat"]

        # Add model specification based on process type
        if config.process_type not in cls.MODEL_MAPPING:
            raise ValueError(f"Unsupported process type: {config.process_type}")

        model_name = cls.MODEL_MAPPING[config.process_type]
        base_cmd.extend(["--model", model_name])

        # Add performance and output flags
        base_cmd.extend(
            [
                "--no-stream",  # Disable streaming for batch processing
                "--format",
                "json",  # Structured output for parsing
            ]
        )

        logger.debug(f"Built command for {config.process_type.value}: {' '.join(base_cmd)}")
        return base_cmd

    @classmethod
    def get_supported_models(cls) -> dict[ProcessType, str]:
        """Get mapping of process types to their model names."""
        return cls.MODEL_MAPPING.copy()

    @classmethod
    def validate_process_type(cls, process_type: ProcessType) -> bool:
        """
        Validate that a process type is supported for command building.

        Args:
            process_type: The process type to validate

        Returns:
            True if supported, False otherwise
        """
        return process_type in cls.MODEL_MAPPING


class ProcessEnvironmentBuilder:
    """
    Builds environment variables for Claude CLI processes.

    This class manages environment setup for optimal subprocess performance
    and proper Claude CLI operation.
    """

    @staticmethod
    def build_environment(config: ProcessConfig) -> dict[str, str]:
        """
        Build environment variables for process execution.

        Args:
            config: Process configuration

        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()

        # Performance optimizations
        env["PYTHONUNBUFFERED"] = "1"  # Disable Python output buffering

        # Process-specific settings
        if config.memory_limit_mb:
            env["CLAUDE_MEMORY_LIMIT"] = str(config.memory_limit_mb)

        # Timeout settings
        env["CLAUDE_STARTUP_TIMEOUT"] = str(config.startup_timeout)

        logger.debug(f"Built environment for {config.process_type.value} with {len(env)} variables")
        return env


class ClaudeProcessFactory:
    """
    Factory for creating properly configured Claude CLI subprocess instances.

    This class is responsible for:
    - Creating subprocess instances with correct configuration
    - Setting up environment variables and process groups
    - Validating configuration before process creation
    - Providing different creation strategies for different process types
    """

    def __init__(self):
        """Initialize the process factory."""
        self.command_builder = ProcessCommandBuilder()
        self.env_builder = ProcessEnvironmentBuilder()

    def create_subprocess(self, config: ProcessConfig) -> subprocess.Popen:
        """
        Create a Claude CLI subprocess with hardened security configuration.
        
        Guardian Security Features:
        - Command validation to prevent injection attacks
        - Environment variable sanitization
        - Secure subprocess creation with restricted privileges
        - Process group isolation

        Args:
            config: Process configuration specifying type, model, and settings

        Returns:
            Configured subprocess.Popen instance

        Raises:
            ValueError: If configuration is invalid
            SecurityValidationError: If command validation fails  
            OSError: If subprocess creation fails
        """
        # Validate configuration
        self._validate_config(config)

        # Build command and environment
        command = self.command_builder.build_command(config)
        environment = self.env_builder.build_environment(config)

        try:
            # SECURITY: Validate command for injection attacks
            validated_command = validate_subprocess_command(command)
            logger.debug(f"Command validation passed: {' '.join(validated_command)}")
            
            # SECURITY: Sanitize environment variables
            safe_environment = self._sanitize_environment(environment)
            
            # Create subprocess with hardened security settings
            process = subprocess.Popen(
                validated_command,
                env=safe_environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered for real-time communication
                preexec_fn=os.setsid if config.use_process_group else None,
                # SECURITY: Additional hardening
                shell=False,  # Never use shell=True to prevent injection
                close_fds=True,  # Close file descriptors for security
                start_new_session=True,  # Start new session for isolation
            )

            logger.info(f"Created secure {config.process_type.value} subprocess (PID: {process.pid})")

            return process

        except SecurityValidationError as e:
            logger.error(f"Command validation failed for {config.process_type.value}: {e}")
            raise SecurityValidationError(f"Subprocess creation blocked - security validation failed: {e}") from e
        except Exception as e:
            logger.error(f"Failed to create subprocess for {config.process_type.value}: {e}")
            raise OSError(f"Subprocess creation failed: {e}") from e

    def create_strategic_config(
        self, startup_timeout: float = 10.0, memory_limit_mb: int = 100
    ) -> ProcessConfig:
        """
        Create configuration for an Opus strategic planning process.

        Args:
            startup_timeout: Maximum time to wait for process startup
            memory_limit_mb: Memory limit in megabytes

        Returns:
            ProcessConfig for strategic process
        """
        return ProcessConfig(
            process_type=ProcessType.OPUS_STRATEGIC,
            model_name=self.command_builder.MODEL_MAPPING[ProcessType.OPUS_STRATEGIC],
            system_prompt=PromptRepository.get_prompt(ProcessType.OPUS_STRATEGIC),
            startup_timeout=startup_timeout,
            memory_limit_mb=memory_limit_mb,
        )

    def create_tactical_config(
        self, startup_timeout: float = 5.0, memory_limit_mb: int = 75
    ) -> ProcessConfig:
        """
        Create configuration for a Sonnet tactical execution process.

        Args:
            startup_timeout: Maximum time to wait for process startup
            memory_limit_mb: Memory limit in megabytes

        Returns:
            ProcessConfig for tactical process
        """
        return ProcessConfig(
            process_type=ProcessType.SONNET_TACTICAL,
            model_name=self.command_builder.MODEL_MAPPING[ProcessType.SONNET_TACTICAL],
            system_prompt=PromptRepository.get_prompt(ProcessType.SONNET_TACTICAL),
            startup_timeout=startup_timeout,
            memory_limit_mb=memory_limit_mb,
        )

    def create_standard_process_set(self) -> list[ProcessConfig]:
        """
        Create the standard set of process configurations for the system.

        This creates 1 Opus strategic process and 4 Sonnet tactical processes
        with optimized settings for the Pokemon speedrun learning agent.

        Returns:
            List of ProcessConfig objects for all required processes
        """
        configs = []

        # 1 Opus Strategic process
        configs.append(
            self.create_strategic_config(
                startup_timeout=10.0, memory_limit_mb=100  # Opus gets more time
            )
        )

        # 4 Sonnet Tactical processes
        for _ in range(4):
            configs.append(
                self.create_tactical_config(
                    startup_timeout=5.0, memory_limit_mb=75  # Sonnet should start faster
                )
            )

        logger.info("Created standard process configuration set: 1 strategic + 4 tactical")
        return configs

    def _validate_config(self, config: ProcessConfig):
        """
        Validate process configuration before subprocess creation.

        Args:
            config: Process configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate process type
        if not self.command_builder.validate_process_type(config.process_type):
            raise ValueError(f"Unsupported process type: {config.process_type}")

        # Validate prompt availability
        if not PromptRepository.validate_prompt(config.process_type):
            raise ValueError(f"No valid prompt available for type: {config.process_type}")

        # Validate timeout values
        if config.startup_timeout <= 0:
            raise ValueError("Startup timeout must be positive")

        if config.health_check_interval <= 0:
            raise ValueError("Health check interval must be positive")

        # Validate memory limits
        if config.memory_limit_mb <= 0:
            raise ValueError("Memory limit must be positive")

        # Validate buffer sizes
        if config.stdout_buffer_size <= 0 or config.stderr_buffer_size <= 0:
            raise ValueError("Buffer sizes must be positive")

    def _sanitize_environment(self, env: dict[str, str]) -> dict[str, str]:
        """
        Sanitize environment variables for security.
        
        Guardian Security Features:
        - Remove dangerous environment variables
        - Validate environment variable values
        - Prevent environment variable injection
        
        Args:
            env: Original environment dictionary
            
        Returns:
            Sanitized environment dictionary
        """
        # Dangerous environment variables to remove
        DANGEROUS_ENV_VARS = {
            'LD_PRELOAD',  # Library preloading attack vector
            'LD_LIBRARY_PATH',  # Library path manipulation
            'DYLD_INSERT_LIBRARIES',  # macOS library injection
            'DYLD_LIBRARY_PATH',  # macOS library path
            'IFS',  # Input Field Separator manipulation
            'PATH',  # We'll set our own safe PATH
        }
        
        # Start with base environment (filtered)
        safe_env = {}
        
        for key, value in env.items():
            # Skip dangerous variables
            if key in DANGEROUS_ENV_VARS:
                logger.debug(f"Removing dangerous environment variable: {key}")
                continue
                
            # Validate variable name (alphanumeric + underscore only)
            if not key.replace('_', '').isalnum():
                logger.warning(f"Skipping environment variable with invalid name: {key}")
                continue
                
            # Validate variable value (no shell metacharacters)
            if any(char in str(value) for char in [';', '&', '|', '`', '$', '>', '<', '\n', '\r']):
                logger.warning(f"Skipping environment variable with dangerous value: {key}")
                continue
                
            # Limit value length to prevent resource exhaustion
            if len(str(value)) > 1000:
                logger.warning(f"Truncating long environment variable: {key}")
                value = str(value)[:1000]
                
            safe_env[key] = str(value)
        
        # Set secure PATH  
        safe_env['PATH'] = '/usr/local/bin:/usr/bin:/bin'
        
        # Ensure critical security variables
        safe_env['PYTHONUNBUFFERED'] = '1'
        safe_env['PYTHONDONTWRITEBYTECODE'] = '1'  # Don't write .pyc files
        
        logger.debug(f"Environment sanitized: {len(env)} -> {len(safe_env)} variables")
        return safe_env

    def get_factory_stats(self) -> dict[str, Any]:
        """
        Get factory statistics and supported configurations.

        Returns:
            Dictionary with factory information
        """
        return {
            "supported_process_types": [ptype.value for ptype in ProcessType],
            "supported_models": {
                ptype.value: model
                for ptype, model in self.command_builder.get_supported_models().items()
            },
            "available_prompts": [ptype.value for ptype in PromptRepository.get_available_types()],
        }
