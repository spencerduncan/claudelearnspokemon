"""Unit tests for Docker environment detection functionality."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from docker.errors import DockerException, APIError

from claudelearnspokemon.docker_environment_detector import (
    DockerEnvironmentDetector,
    DockerDetectionResult,
    get_docker_detector,
    is_docker_available,
    get_docker_info,
    DOCKER_AVAILABLE
)


class TestDockerDetectionResult:
    """Test cases for DockerDetectionResult."""
    
    def test_available_result_creation(self):
        """Test creation of successful detection result."""
        version_info = {"Server": {"Version": "20.10.0"}}
        result = DockerDetectionResult(
            is_available=True,
            version_info=version_info,
            detection_time_ms=15.5
        )
        
        assert result.is_available is True
        assert result.version_info == version_info
        assert result.error_message is None
        assert result.detection_time_ms == 15.5
        assert result.timestamp > 0
    
    def test_unavailable_result_creation(self):
        """Test creation of failed detection result."""
        error_msg = "Docker daemon not running"
        result = DockerDetectionResult(
            is_available=False,
            error_message=error_msg,
            detection_time_ms=5.0
        )
        
        assert result.is_available is False
        assert result.version_info == {}
        assert result.error_message == error_msg
        assert result.detection_time_ms == 5.0
    
    def test_result_repr(self):
        """Test string representation of results."""
        available_result = DockerDetectionResult(True, detection_time_ms=10.0)
        unavailable_result = DockerDetectionResult(False, detection_time_ms=5.0)
        
        assert "available" in str(available_result)
        assert "10.0ms" in str(available_result)
        assert "unavailable" in str(unavailable_result)
        assert "5.0ms" in str(unavailable_result)


class TestDockerEnvironmentDetector:
    """Test cases for DockerEnvironmentDetector."""
    
    def setup_method(self):
        """Set up test instance for each test."""
        self.detector = DockerEnvironmentDetector(cache_timeout_seconds=1)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = DockerEnvironmentDetector(cache_timeout_seconds=30)
        assert detector._cache_timeout == 30
        assert detector._cache is None
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.docker_environment_detector.DOCKER_AVAILABLE', False)
    def test_docker_library_not_available(self):
        """Test detection when docker library is not installed."""
        result = self.detector.detect_docker_environment()
        
        assert result.is_available is False
        assert "Docker Python library not available" in result.error_message
        assert "pip install docker" in result.error_message
        assert result.detection_time_ms > 0
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.docker_environment_detector.DOCKER_AVAILABLE', True)
    @patch('claudelearnspokemon.docker_environment_detector.docker.from_env')
    def test_docker_daemon_available(self, mock_from_env):
        """Test successful Docker daemon detection."""
        # Mock successful Docker client
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.version.return_value = {
            "Server": {"Version": "20.10.0"},
            "Client": {"Version": "20.10.0"}
        }
        mock_from_env.return_value = mock_client
        
        result = self.detector.detect_docker_environment()
        
        assert result.is_available is True
        assert result.version_info["Server"]["Version"] == "20.10.0"
        assert result.error_message is None
        assert result.detection_time_ms > 0
        
        mock_from_env.assert_called_once_with(timeout=5)
        mock_client.ping.assert_called_once()
        mock_client.version.assert_called_once()
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.docker_environment_detector.DOCKER_AVAILABLE', True)
    @patch('claudelearnspokemon.docker_environment_detector.docker.from_env')
    def test_docker_daemon_connection_refused(self, mock_from_env):
        """Test Docker daemon connection refused."""
        mock_from_env.side_effect = APIError("Connection refused")
        
        result = self.detector.detect_docker_environment()
        
        assert result.is_available is False
        assert "Docker daemon not running" in result.error_message
        assert "docker --version" in result.error_message
        assert result.detection_time_ms > 0
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.docker_environment_detector.DOCKER_AVAILABLE', True)
    @patch('claudelearnspokemon.docker_environment_detector.docker.from_env')
    def test_docker_permission_denied(self, mock_from_env):
        """Test Docker daemon permission denied."""
        mock_from_env.side_effect = DockerException("Permission denied")
        
        result = self.detector.detect_docker_environment()
        
        assert result.is_available is False
        assert "Permission denied" in result.error_message
        assert "sudo" in result.error_message or "docker group" in result.error_message
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.docker_environment_detector.DOCKER_AVAILABLE', True)
    @patch('claudelearnspokemon.docker_environment_detector.docker.from_env')
    def test_docker_timeout(self, mock_from_env):
        """Test Docker daemon timeout."""
        mock_from_env.side_effect = DockerException("timeout")
        
        result = self.detector.detect_docker_environment()
        
        assert result.is_available is False
        assert "Timeout connecting" in result.error_message
        assert "slow to respond" in result.error_message
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.docker_environment_detector.DOCKER_AVAILABLE', True)
    @patch('claudelearnspokemon.docker_environment_detector.docker.from_env')
    def test_unexpected_error(self, mock_from_env):
        """Test unexpected error during detection."""
        mock_from_env.side_effect = RuntimeError("Unexpected error")
        
        result = self.detector.detect_docker_environment()
        
        assert result.is_available is False
        assert "Unexpected error during Docker detection" in result.error_message
        assert "Unexpected error" in result.error_message
    
    @pytest.mark.unit
    def test_caching_behavior(self):
        """Test that detection results are properly cached."""
        with patch.object(self.detector, '_detect_docker_daemon') as mock_detect:
            mock_result = DockerDetectionResult(True, detection_time_ms=10.0)
            mock_detect.return_value = mock_result
            
            # First call should trigger detection
            result1 = self.detector.detect_docker_environment()
            assert mock_detect.call_count == 1
            
            # Second call should use cache
            result2 = self.detector.detect_docker_environment()
            assert mock_detect.call_count == 1  # Should not increase
            assert result2 is mock_result
    
    @pytest.mark.unit
    def test_force_refresh_bypasses_cache(self):
        """Test that force_refresh bypasses cache."""
        with patch.object(self.detector, '_detect_docker_daemon') as mock_detect:
            mock_result1 = DockerDetectionResult(True, detection_time_ms=10.0)
            mock_result2 = DockerDetectionResult(False, detection_time_ms=5.0)
            mock_detect.side_effect = [mock_result1, mock_result2]
            
            # First call
            result1 = self.detector.detect_docker_environment()
            assert mock_detect.call_count == 1
            
            # Force refresh should trigger new detection
            result2 = self.detector.detect_docker_environment(force_refresh=True)
            assert mock_detect.call_count == 2
            assert result2 is mock_result2
    
    @pytest.mark.unit
    def test_cache_expiration(self):
        """Test that cache expires after timeout."""
        # Use very short timeout for testing
        detector = DockerEnvironmentDetector(cache_timeout_seconds=0.1)
        
        with patch.object(detector, '_detect_docker_daemon') as mock_detect:
            mock_result = DockerDetectionResult(True, detection_time_ms=10.0)
            mock_detect.return_value = mock_result
            
            # First call
            detector.detect_docker_environment()
            assert mock_detect.call_count == 1
            
            # Wait for cache to expire
            time.sleep(0.2)
            
            # Should trigger new detection
            detector.detect_docker_environment()
            assert mock_detect.call_count == 2
    
    @pytest.mark.unit
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        with patch.object(self.detector, '_detect_docker_daemon') as mock_detect:
            mock_result = DockerDetectionResult(True, detection_time_ms=10.0)
            mock_detect.return_value = mock_result
            
            # First call to populate cache
            self.detector.detect_docker_environment()
            assert mock_detect.call_count == 1
            
            # Clear cache
            self.detector.clear_cache()
            
            # Next call should trigger new detection
            self.detector.detect_docker_environment()
            assert mock_detect.call_count == 2
    
    @pytest.mark.unit
    def test_is_docker_available_convenience_method(self):
        """Test is_docker_available convenience method."""
        with patch.object(self.detector, 'detect_docker_environment') as mock_detect:
            mock_detect.return_value = DockerDetectionResult(True)
            assert self.detector.is_docker_available() is True
            
            mock_detect.return_value = DockerDetectionResult(False)
            assert self.detector.is_docker_available() is False
    
    @pytest.mark.unit
    def test_get_docker_info_convenience_method(self):
        """Test get_docker_info convenience method."""
        version_info = {"Server": {"Version": "20.10.0"}}
        
        with patch.object(self.detector, 'detect_docker_environment') as mock_detect:
            # Available case
            mock_detect.return_value = DockerDetectionResult(True, version_info=version_info)
            assert self.detector.get_docker_info() == version_info
            
            # Unavailable case
            mock_detect.return_value = DockerDetectionResult(False)
            assert self.detector.get_docker_info() is None
    
    @pytest.mark.unit
    def test_thread_safety(self):
        """Test that detector is thread-safe."""
        import threading
        import concurrent.futures
        
        results = []
        call_count = 0
        
        def mock_detect():
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate detection time
            return DockerDetectionResult(True, detection_time_ms=10.0)
        
        with patch.object(self.detector, '_detect_docker_daemon', side_effect=mock_detect):
            # Run multiple detections concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(self.detector.detect_docker_environment) 
                    for _ in range(10)
                ]
                results = [f.result() for f in futures]
        
        # All should succeed
        assert len(results) == 10
        assert all(r.is_available for r in results)
        
        # Only one detection should have occurred (others should use cache)
        assert call_count == 1


class TestGlobalFunctions:
    """Test cases for global convenience functions."""
    
    @pytest.mark.unit
    def test_get_docker_detector_singleton(self):
        """Test that get_docker_detector returns same instance."""
        detector1 = get_docker_detector()
        detector2 = get_docker_detector()
        assert detector1 is detector2
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.docker_environment_detector.get_docker_detector')
    def test_is_docker_available_global(self, mock_get_detector):
        """Test global is_docker_available function."""
        mock_detector = Mock()
        mock_detector.is_docker_available.return_value = True
        mock_get_detector.return_value = mock_detector
        
        result = is_docker_available(force_refresh=True)
        
        assert result is True
        mock_detector.is_docker_available.assert_called_once_with(True)
    
    @pytest.mark.unit
    @patch('claudelearnspokemon.docker_environment_detector.get_docker_detector')
    def test_get_docker_info_global(self, mock_get_detector):
        """Test global get_docker_info function."""
        version_info = {"Server": {"Version": "20.10.0"}}
        mock_detector = Mock()
        mock_detector.get_docker_info.return_value = version_info
        mock_get_detector.return_value = mock_detector
        
        result = get_docker_info()
        
        assert result == version_info
        mock_detector.get_docker_info.assert_called_once()


class TestErrorMessageFormatting:
    """Test cases for Docker error message formatting."""
    
    def setup_method(self):
        """Set up test instance."""
        self.detector = DockerEnvironmentDetector()
    
    def test_connection_refused_formatting(self):
        """Test connection refused error formatting."""
        error = DockerException("Connection refused")
        formatted = self.detector._format_docker_error(error)
        
        assert "Docker daemon not running" in formatted
        assert "docker --version" in formatted
    
    def test_permission_denied_formatting(self):
        """Test permission denied error formatting."""
        error = DockerException("permission denied")
        formatted = self.detector._format_docker_error(error)
        
        assert "Permission denied" in formatted
        assert "sudo" in formatted or "docker group" in formatted
    
    def test_timeout_formatting(self):
        """Test timeout error formatting."""
        error = DockerException("timeout")
        formatted = self.detector._format_docker_error(error)
        
        assert "Timeout connecting" in formatted
        assert "slow to respond" in formatted
    
    def test_not_found_formatting(self):
        """Test not found error formatting."""
        error = DockerException("not found")
        formatted = self.detector._format_docker_error(error)
        
        assert "Docker daemon not found" in formatted
        assert "properly installed" in formatted
    
    def test_generic_error_formatting(self):
        """Test generic error formatting."""
        error = DockerException("Some other error")
        formatted = self.detector._format_docker_error(error)
        
        assert "Docker daemon error" in formatted
        assert "Some other error" in formatted


# Performance and integration tests
@pytest.mark.slow
@pytest.mark.integration
def test_real_docker_detection_performance():
    """Test real Docker detection performance (only if Docker available)."""
    detector = DockerEnvironmentDetector()
    
    # Test detection speed
    start_time = time.perf_counter()
    result = detector.detect_docker_environment()
    detection_time = (time.perf_counter() - start_time) * 1000
    
    # Should be reasonably fast (under 5 seconds even for slow systems)
    assert detection_time < 5000
    
    # If Docker is available, version info should be present
    if result.is_available:
        assert result.version_info is not None
        assert "Server" in result.version_info or "Client" in result.version_info
    
    # Test cached performance
    start_time = time.perf_counter()
    cached_result = detector.detect_docker_environment()
    cached_time = (time.perf_counter() - start_time) * 1000
    
    # Cached calls should be very fast (under 10ms)
    assert cached_time < 10
    assert cached_result.is_available == result.is_available