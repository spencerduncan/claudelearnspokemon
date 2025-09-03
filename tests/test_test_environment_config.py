"""Unit tests for test environment configuration functionality."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from claudelearnspokemon.test_environment_config import (
    TestMode,
    TestEnvironmentConfig,
    TestEnvironmentConfigLoader,
    get_config_loader,
    load_test_config,
    is_docker_mode_available
)


class TestTestEnvironmentConfig:
    """Test cases for TestEnvironmentConfig dataclass."""
    
    def test_default_config_creation(self):
        """Test creation with default values."""
        config = TestEnvironmentConfig(mode=TestMode.DOCKER)
        
        assert config.mode == TestMode.DOCKER
        assert config.docker_compose_file == "docker-compose.test.yml"
        assert config.docker_service_name == "pokemon-gym"
        assert config.docker_service_port == 8080
        assert config.mock_base_url == "http://localhost:8080"
        assert config.max_response_time_ms == 100.0
        assert config.max_setup_time_ms == 5000.0
    
    def test_custom_config_creation(self):
        """Test creation with custom values."""
        config = TestEnvironmentConfig(
            mode=TestMode.HTTP_MOCK,
            docker_compose_file="custom-compose.yml",
            max_response_time_ms=50.0
        )
        
        assert config.mode == TestMode.HTTP_MOCK
        assert config.docker_compose_file == "custom-compose.yml"
        assert config.max_response_time_ms == 50.0


class TestTestEnvironmentConfigLoader:
    """Test cases for TestEnvironmentConfigLoader."""
    
    def setup_method(self):
        """Set up test instance for each test."""
        self.loader = TestEnvironmentConfigLoader()
    
    @pytest.mark.unit
    def test_auto_mode_docker_available(self):
        """Test AUTO mode when Docker is available."""
        with patch.object(self.loader, '_docker_detector') as mock_detector:
            mock_detector.is_docker_available.return_value = True
            
            config = self.loader.load_config(TestMode.AUTO)
            
            assert config.mode == TestMode.DOCKER
            mock_detector.is_docker_available.assert_called_once()
    
    @pytest.mark.unit
    def test_auto_mode_docker_unavailable(self):
        """Test AUTO mode when Docker is not available."""
        with patch.object(self.loader, '_docker_detector') as mock_detector:
            mock_detector.is_docker_available.return_value = False
            
            config = self.loader.load_config(TestMode.AUTO)
            
            assert config.mode == TestMode.HTTP_MOCK
            mock_detector.is_docker_available.assert_called_once()
    
    @pytest.mark.unit
    def test_explicit_docker_mode_available(self):
        """Test explicit Docker mode when available."""
        with patch.object(self.loader, '_docker_detector') as mock_detector:
            mock_detector.is_docker_available.return_value = True
            
            config = self.loader.load_config(TestMode.DOCKER)
            
            assert config.mode == TestMode.DOCKER
    
    @pytest.mark.unit
    def test_explicit_docker_mode_unavailable(self):
        """Test explicit Docker mode when not available (should fallback)."""
        with patch.object(self.loader, '_docker_detector') as mock_detector:
            mock_detector.is_docker_available.return_value = False
            
            config = self.loader.load_config(TestMode.DOCKER)
            
            assert config.mode == TestMode.HTTP_MOCK  # Should fallback
    
    @pytest.mark.unit
    def test_explicit_http_mock_mode(self):
        """Test explicit HTTP mock mode."""
        # No need to mock Docker detector for explicit HTTP mock mode
        config = self.loader.load_config(TestMode.HTTP_MOCK)
        
        assert config.mode == TestMode.HTTP_MOCK
    
    @pytest.mark.unit
    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown test mode"):
            self.loader._determine_test_mode("INVALID_MODE", False)
    
    @pytest.mark.unit
    def test_config_caching(self):
        """Test that configuration is properly cached."""
        with patch.object(self.loader, '_determine_test_mode') as mock_determine:
            mock_determine.return_value = TestMode.HTTP_MOCK
            
            # First call should trigger determination
            config1 = self.loader.load_config(TestMode.AUTO)
            assert mock_determine.call_count == 1
            
            # Second call should use cache
            config2 = self.loader.load_config(TestMode.AUTO)
            assert mock_determine.call_count == 1  # Should not increase
            assert config2 is config1
    
    @pytest.mark.unit
    def test_force_refresh_bypasses_cache(self):
        """Test that force_refresh bypasses cache."""
        with patch.object(self.loader, '_determine_test_mode') as mock_determine:
            mock_determine.return_value = TestMode.HTTP_MOCK
            
            # First call
            self.loader.load_config(TestMode.AUTO)
            assert mock_determine.call_count == 1
            
            # Force refresh should trigger new determination
            self.loader.load_config(TestMode.AUTO, force_refresh=True)
            assert mock_determine.call_count == 2
    
    @pytest.mark.unit
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        with patch.object(self.loader, '_determine_test_mode') as mock_determine:
            mock_determine.return_value = TestMode.HTTP_MOCK
            
            # First call to populate cache
            self.loader.load_config(TestMode.AUTO)
            assert mock_determine.call_count == 1
            
            # Clear cache
            self.loader.clear_cache()
            
            # Next call should trigger new determination
            self.loader.load_config(TestMode.AUTO)
            assert mock_determine.call_count == 2


class TestEnvironmentVariableOverrides:
    """Test cases for environment variable configuration overrides."""
    
    def setup_method(self):
        """Set up test instance."""
        self.loader = TestEnvironmentConfigLoader()
        # Save original environment
        self.original_env = dict(os.environ)
    
    def teardown_method(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    @pytest.mark.unit
    def test_docker_compose_file_override(self):
        """Test DOCKER_COMPOSE_FILE environment variable override."""
        os.environ["DOCKER_COMPOSE_FILE"] = "custom-docker-compose.yml"
        
        with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.DOCKER):
            config = self.loader.load_config()
            assert config.docker_compose_file == "custom-docker-compose.yml"
    
    @pytest.mark.unit
    def test_docker_service_name_override(self):
        """Test DOCKER_SERVICE_NAME environment variable override."""
        os.environ["DOCKER_SERVICE_NAME"] = "custom-service"
        
        with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.DOCKER):
            config = self.loader.load_config()
            assert config.docker_service_name == "custom-service"
    
    @pytest.mark.unit
    def test_docker_service_port_override(self):
        """Test DOCKER_SERVICE_PORT environment variable override."""
        os.environ["DOCKER_SERVICE_PORT"] = "9090"
        
        with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.DOCKER):
            config = self.loader.load_config()
            assert config.docker_service_port == 9090
    
    @pytest.mark.unit
    def test_mock_base_url_override(self):
        """Test MOCK_BASE_URL environment variable override."""
        os.environ["MOCK_BASE_URL"] = "http://custom-mock:8080"
        
        with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.HTTP_MOCK):
            config = self.loader.load_config()
            assert config.mock_base_url == "http://custom-mock:8080"
    
    @pytest.mark.unit
    def test_max_response_time_override(self):
        """Test MAX_RESPONSE_TIME_MS environment variable override."""
        os.environ["MAX_RESPONSE_TIME_MS"] = "200.5"
        
        with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.DOCKER):
            config = self.loader.load_config()
            assert config.max_response_time_ms == 200.5
    
    @pytest.mark.unit
    def test_log_level_override(self):
        """Test TEST_LOG_LEVEL environment variable override."""
        os.environ["TEST_LOG_LEVEL"] = "DEBUG"
        
        with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.HTTP_MOCK):
            config = self.loader.load_config()
            assert config.log_level == "DEBUG"
    
    @pytest.mark.unit
    def test_log_docker_output_override(self):
        """Test LOG_DOCKER_OUTPUT environment variable override."""
        os.environ["LOG_DOCKER_OUTPUT"] = "true"
        
        with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.DOCKER):
            config = self.loader.load_config()
            assert config.log_docker_output is True
        
        os.environ["LOG_DOCKER_OUTPUT"] = "false"
        self.loader.clear_cache()  # Clear cache to force reload
        
        with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.DOCKER):
            config = self.loader.load_config()
            assert config.log_docker_output is False


class TestModeSpecificConfigurations:
    """Test cases for mode-specific configuration adjustments."""
    
    def setup_method(self):
        """Set up test instance."""
        self.loader = TestEnvironmentConfigLoader()
    
    @pytest.mark.unit
    def test_docker_mode_adjustments(self):
        """Test Docker mode-specific configuration adjustments."""
        with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.DOCKER):
            config = self.loader.load_config()
            
            # Docker mode should have longer setup time
            assert config.max_setup_time_ms == 10000.0  # 10 seconds
            
            # Response time should be at least 150ms for real service
            assert config.max_response_time_ms >= 150.0
    
    @pytest.mark.unit
    def test_http_mock_mode_adjustments(self):
        """Test HTTP mock mode-specific configuration adjustments."""
        with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.HTTP_MOCK):
            config = self.loader.load_config()
            
            # HTTP mock should have shorter setup time
            assert config.max_setup_time_ms == 1000.0  # 1 second
            
            # Response time should be capped at 50ms for mocks
            assert config.max_response_time_ms <= 50.0
    
    @pytest.mark.unit
    def test_http_mock_preserves_lower_response_time(self):
        """Test that HTTP mock mode preserves lower response time settings."""
        # Set a very low response time via environment
        with patch.dict(os.environ, {"MAX_RESPONSE_TIME_MS": "25.0"}):
            with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.HTTP_MOCK):
                config = self.loader.load_config()
                
                # Should preserve the lower setting
                assert config.max_response_time_ms == 25.0


class TestGlobalFunctions:
    """Test cases for global convenience functions."""
    
    @pytest.mark.unit
    def test_get_config_loader_singleton(self):
        """Test that get_config_loader returns same instance."""
        loader1 = get_config_loader()
        loader2 = get_config_loader()
        assert loader1 is loader2
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.test_environment_config.get_config_loader')
    def test_load_test_config_global(self, mock_get_loader):
        """Test global load_test_config function."""
        mock_loader = Mock()
        expected_config = TestEnvironmentConfig(mode=TestMode.HTTP_MOCK)
        mock_loader.load_config.return_value = expected_config
        mock_get_loader.return_value = mock_loader
        
        result = load_test_config(TestMode.AUTO, force_refresh=True)
        
        assert result is expected_config
        mock_loader.load_config.assert_called_once_with(TestMode.AUTO, True)
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.test_environment_config.get_docker_detector')
    def test_is_docker_mode_available_global(self, mock_get_detector):
        """Test global is_docker_mode_available function."""
        mock_detector = Mock()
        mock_detector.is_docker_available.return_value = True
        mock_get_detector.return_value = mock_detector
        
        result = is_docker_mode_available()
        
        assert result is True
        mock_detector.is_docker_available.assert_called_once()


class TestLoggingBehavior:
    """Test cases for logging behavior in configuration loading."""
    
    def setup_method(self):
        """Set up test instance."""
        self.loader = TestEnvironmentConfigLoader()
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.test_environment_config.logger')
    def test_docker_mode_logging(self, mock_logger):
        """Test logging for Docker mode configuration."""
        with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.DOCKER):
            self.loader.load_config()
            
            # Should log Docker mode selection and configuration
            mock_logger.info.assert_any_call("Test environment configured: DOCKER")
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.test_environment_config.logger')
    def test_http_mock_mode_logging(self, mock_logger):
        """Test logging for HTTP mock mode configuration."""
        with patch.object(self.loader, '_determine_test_mode', return_value=TestMode.HTTP_MOCK):
            self.loader.load_config()
            
            # Should log HTTP mock mode selection
            mock_logger.info.assert_any_call("Test environment configured: HTTP_MOCK")
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.test_environment_config.logger')
    @patch('claudelearnspokemon.test_environment_config.get_docker_detector')
    def test_fallback_warning_logging(self, mock_get_detector, mock_logger):
        """Test logging for Docker fallback scenario."""
        mock_detector = Mock()
        mock_detector.is_docker_available.return_value = False
        mock_get_detector.return_value = mock_detector
        
        self.loader.load_config(TestMode.DOCKER)
        
        # Should log warning about fallback
        mock_logger.warning.assert_any_call(
            "Docker mode requested but Docker not available. "
            "Falling back to HTTP mock mode."
        )