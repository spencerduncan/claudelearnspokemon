"""Test environment configuration loader with Docker fallback support.

This module provides unified configuration for tests that can run in either:
1. Docker mode (using TestContainers with real Pokemon-gym service)
2. HTTP mock mode (using responses library for mocking)

The configuration automatically detects Docker availability and gracefully
falls back to HTTP mocking when Docker is not available.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional

from .docker_environment_detector import get_docker_detector


logger = logging.getLogger(__name__)


class TestMode(Enum):
    """Test execution modes."""
    DOCKER = "docker"
    HTTP_MOCK = "http_mock"
    AUTO = "auto"  # Automatic detection based on Docker availability


@dataclass
class TestEnvironmentConfig:
    """Configuration for test environment setup."""
    
    # Test execution mode
    mode: TestMode
    
    # Docker-specific configuration
    docker_compose_file: str = "docker-compose.test.yml"
    docker_service_name: str = "pokemon-gym"
    docker_service_port: int = 8080
    docker_health_check_timeout: int = 30
    docker_health_check_interval: float = 0.5
    
    # HTTP mock configuration  
    mock_base_url: str = "http://localhost:8080"
    mock_response_delay: float = 0.01  # Simulate realistic response times
    
    # Performance thresholds
    max_response_time_ms: float = 100.0
    max_setup_time_ms: float = 5000.0
    
    # Logging configuration
    log_level: str = "INFO"
    log_docker_output: bool = False
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0


class TestEnvironmentConfigLoader:
    """Loads test environment configuration with Docker fallback logic."""
    
    def __init__(self):
        self._config_cache: Optional[TestEnvironmentConfig] = None
        self._docker_detector = get_docker_detector()
    
    def load_config(
        self, 
        preferred_mode: TestMode = TestMode.AUTO,
        force_refresh: bool = False
    ) -> TestEnvironmentConfig:
        """Load test environment configuration.
        
        Args:
            preferred_mode: Preferred test mode (AUTO will detect Docker availability)
            force_refresh: Force refresh of Docker detection cache
            
        Returns:
            TestEnvironmentConfig with appropriate mode and settings
        """
        # Check cache first
        if not force_refresh and self._config_cache is not None:
            logger.debug("Using cached test environment configuration")
            return self._config_cache
        
        # Determine the actual test mode
        actual_mode = self._determine_test_mode(preferred_mode, force_refresh)
        
        # Create configuration
        config = self._create_config(actual_mode)
        
        # Cache the configuration
        self._config_cache = config
        
        # Log the selected mode
        self._log_configuration(config)
        
        return config
    
    def _determine_test_mode(self, preferred_mode: TestMode, force_refresh: bool) -> TestMode:
        """Determine the actual test mode based on preferences and Docker availability."""
        
        if preferred_mode == TestMode.DOCKER:
            # Docker explicitly requested - check if available
            if self._docker_detector.is_docker_available(force_refresh):
                return TestMode.DOCKER
            else:
                logger.warning(
                    "Docker mode requested but Docker not available. "
                    "Falling back to HTTP mock mode."
                )
                return TestMode.HTTP_MOCK
        
        elif preferred_mode == TestMode.HTTP_MOCK:
            # HTTP mock explicitly requested
            return TestMode.HTTP_MOCK
        
        elif preferred_mode == TestMode.AUTO:
            # Auto-detection based on Docker availability
            if self._docker_detector.is_docker_available(force_refresh):
                logger.info("Docker detected - using Docker mode for integration tests")
                return TestMode.DOCKER
            else:
                logger.info("Docker not available - using HTTP mock mode for tests")
                return TestMode.HTTP_MOCK
        
        else:
            raise ValueError(f"Unknown test mode: {preferred_mode}")
    
    def _create_config(self, mode: TestMode) -> TestEnvironmentConfig:
        """Create configuration object with environment variable overrides."""
        
        # Start with base configuration
        config = TestEnvironmentConfig(mode=mode)
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        # Apply mode-specific adjustments
        if mode == TestMode.DOCKER:
            config = self._apply_docker_config(config)
        elif mode == TestMode.HTTP_MOCK:
            config = self._apply_http_mock_config(config)
        
        return config
    
    def _apply_env_overrides(self, config: TestEnvironmentConfig) -> TestEnvironmentConfig:
        """Apply environment variable overrides to configuration."""
        
        # Docker configuration
        if os.environ.get("DOCKER_COMPOSE_FILE"):
            config.docker_compose_file = os.environ["DOCKER_COMPOSE_FILE"]
        
        if os.environ.get("DOCKER_SERVICE_NAME"):
            config.docker_service_name = os.environ["DOCKER_SERVICE_NAME"]
        
        if os.environ.get("DOCKER_SERVICE_PORT"):
            config.docker_service_port = int(os.environ["DOCKER_SERVICE_PORT"])
        
        # HTTP mock configuration
        if os.environ.get("MOCK_BASE_URL"):
            config.mock_base_url = os.environ["MOCK_BASE_URL"]
        
        # Performance thresholds
        if os.environ.get("MAX_RESPONSE_TIME_MS"):
            config.max_response_time_ms = float(os.environ["MAX_RESPONSE_TIME_MS"])
        
        # Logging configuration
        if os.environ.get("TEST_LOG_LEVEL"):
            config.log_level = os.environ["TEST_LOG_LEVEL"]
        
        if os.environ.get("LOG_DOCKER_OUTPUT"):
            config.log_docker_output = os.environ["LOG_DOCKER_OUTPUT"].lower() == "true"
        
        return config
    
    def _apply_docker_config(self, config: TestEnvironmentConfig) -> TestEnvironmentConfig:
        """Apply Docker-specific configuration adjustments."""
        
        # Docker mode typically has higher setup overhead
        config.max_setup_time_ms = 10000.0  # 10 seconds for container startup
        
        # Slightly more generous response times for real service
        config.max_response_time_ms = max(config.max_response_time_ms, 150.0)
        
        return config
    
    def _apply_http_mock_config(self, config: TestEnvironmentConfig) -> TestEnvironmentConfig:
        """Apply HTTP mock-specific configuration adjustments."""
        
        # HTTP mocks should be very fast
        config.max_setup_time_ms = 1000.0  # 1 second setup
        config.max_response_time_ms = min(config.max_response_time_ms, 50.0)  # 50ms max
        
        return config
    
    def _log_configuration(self, config: TestEnvironmentConfig) -> None:
        """Log the selected configuration."""
        
        mode_name = config.mode.value.upper()
        logger.info(f"Test environment configured: {mode_name}")
        
        if config.mode == TestMode.DOCKER:
            logger.info(f"Docker compose file: {config.docker_compose_file}")
            logger.info(f"Docker service: {config.docker_service_name}:{config.docker_service_port}")
            logger.info(f"Max setup time: {config.max_setup_time_ms}ms")
        elif config.mode == TestMode.HTTP_MOCK:
            logger.info(f"Mock base URL: {config.mock_base_url}")
            logger.info(f"Mock response delay: {config.mock_response_delay}ms")
        
        logger.info(f"Max response time: {config.max_response_time_ms}ms")
        logger.debug(f"Full configuration: {config}")
    
    def clear_cache(self) -> None:
        """Clear configuration cache to force reload on next call."""
        self._config_cache = None
        logger.debug("Test environment configuration cache cleared")


# Global configuration loader instance
_default_loader = None
_loader_lock = None


def get_config_loader() -> TestEnvironmentConfigLoader:
    """Get the global test environment configuration loader.
    
    Returns:
        Shared TestEnvironmentConfigLoader instance
    """
    global _default_loader, _loader_lock
    
    if _loader_lock is None:
        import threading
        _loader_lock = threading.Lock()
    
    with _loader_lock:
        if _default_loader is None:
            _default_loader = TestEnvironmentConfigLoader()
        return _default_loader


def load_test_config(
    preferred_mode: TestMode = TestMode.AUTO,
    force_refresh: bool = False
) -> TestEnvironmentConfig:
    """Convenience function to load test environment configuration.
    
    Args:
        preferred_mode: Preferred test mode
        force_refresh: Force refresh of caches
        
    Returns:
        TestEnvironmentConfig with appropriate settings
    """
    return get_config_loader().load_config(preferred_mode, force_refresh)


def is_docker_mode_available() -> bool:
    """Check if Docker mode is available for testing.
    
    Returns:
        True if Docker mode can be used, False otherwise
    """
    return get_docker_detector().is_docker_available()