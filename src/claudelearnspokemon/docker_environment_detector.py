"""Docker Environment Detector for runtime Docker daemon detection.

This utility provides robust Docker daemon detection with caching, graceful fallback,
and clear error messaging for hybrid test infrastructure.

Key Features:
- Runtime detection (not import-time) for better error handling
- Performance-optimized with caching to avoid repeated Docker API calls
- Support for various Docker configurations (Docker Desktop, Docker Engine, etc.)
- Clear logging and error messages for troubleshooting
- Graceful degradation when Docker is unavailable
"""

import logging
import time
from typing import Dict, Optional, Tuple
import threading

try:
    import docker
    from docker.errors import DockerException
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


logger = logging.getLogger(__name__)


class DockerDetectionResult:
    """Result object for Docker environment detection."""
    
    def __init__(
        self,
        is_available: bool,
        version_info: Optional[Dict] = None,
        error_message: Optional[str] = None,
        detection_time_ms: float = 0.0
    ):
        self.is_available = is_available
        self.version_info = version_info or {}
        self.error_message = error_message
        self.detection_time_ms = detection_time_ms
        self.timestamp = time.time()
    
    def __repr__(self) -> str:
        status = "available" if self.is_available else "unavailable"
        return f"DockerDetectionResult(status={status}, time={self.detection_time_ms:.1f}ms)"


class DockerEnvironmentDetector:
    """Thread-safe Docker environment detector with caching and graceful fallback."""
    
    def __init__(self, cache_timeout_seconds: int = 60):
        """Initialize the detector.
        
        Args:
            cache_timeout_seconds: How long to cache detection results (default: 60 seconds)
        """
        self._cache_timeout = cache_timeout_seconds
        self._cache: Optional[DockerDetectionResult] = None
        self._lock = threading.Lock()
    
    def detect_docker_environment(self, force_refresh: bool = False) -> DockerDetectionResult:
        """Detect Docker daemon availability with caching.
        
        Args:
            force_refresh: If True, bypass cache and perform fresh detection
            
        Returns:
            DockerDetectionResult with availability status and metadata
        """
        with self._lock:
            # Check cache first (unless forced refresh)
            if not force_refresh and self._is_cache_valid():
                logger.debug("Using cached Docker detection result")
                return self._cache
            
            # Perform fresh detection
            logger.info("Performing Docker environment detection...")
            result = self._detect_docker_daemon()
            
            # Cache the result
            self._cache = result
            
            # Log the result
            if result.is_available:
                version = result.version_info.get('Server', {}).get('Version', 'unknown')
                logger.info(f"Docker daemon detected (version: {version}, "
                           f"detection time: {result.detection_time_ms:.1f}ms)")
            else:
                logger.warning(f"Docker daemon not available: {result.error_message}")
            
            return result
    
    def is_docker_available(self, force_refresh: bool = False) -> bool:
        """Simple boolean check for Docker availability.
        
        Args:
            force_refresh: If True, bypass cache and perform fresh detection
            
        Returns:
            True if Docker daemon is available, False otherwise
        """
        return self.detect_docker_environment(force_refresh).is_available
    
    def get_docker_info(self) -> Optional[Dict]:
        """Get Docker daemon information if available.
        
        Returns:
            Docker daemon version info if available, None otherwise
        """
        result = self.detect_docker_environment()
        return result.version_info if result.is_available else None
    
    def clear_cache(self) -> None:
        """Clear the detection cache to force fresh detection on next call."""
        with self._lock:
            self._cache = None
            logger.debug("Docker detection cache cleared")
    
    def _is_cache_valid(self) -> bool:
        """Check if cached result is still valid."""
        if self._cache is None:
            return False
        
        age_seconds = time.time() - self._cache.timestamp
        return age_seconds < self._cache_timeout
    
    def _detect_docker_daemon(self) -> DockerDetectionResult:
        """Perform actual Docker daemon detection."""
        start_time = time.perf_counter()
        
        # Check if docker library is available
        if not DOCKER_AVAILABLE:
            detection_time = (time.perf_counter() - start_time) * 1000
            return DockerDetectionResult(
                is_available=False,
                error_message="Docker Python library not available. Install with: pip install docker",
                detection_time_ms=detection_time
            )
        
        try:
            # Try to create Docker client and ping daemon
            client = docker.from_env(timeout=5)  # 5 second timeout for responsiveness
            
            # Attempt to ping the Docker daemon
            client.ping()
            
            # If ping succeeds, get version information
            version_info = client.version()
            
            detection_time = (time.perf_counter() - start_time) * 1000
            
            return DockerDetectionResult(
                is_available=True,
                version_info=version_info,
                detection_time_ms=detection_time
            )
            
        except DockerException as e:
            # Docker API errors (daemon not running, permission issues, etc.)
            detection_time = (time.perf_counter() - start_time) * 1000
            error_msg = self._format_docker_error(e)
            
            return DockerDetectionResult(
                is_available=False,
                error_message=error_msg,
                detection_time_ms=detection_time
            )
            
        except Exception as e:
            # Other unexpected errors
            detection_time = (time.perf_counter() - start_time) * 1000
            error_msg = f"Unexpected error during Docker detection: {str(e)}"
            
            return DockerDetectionResult(
                is_available=False,
                error_message=error_msg,
                detection_time_ms=detection_time
            )
    
    def _format_docker_error(self, error: DockerException) -> str:
        """Format Docker errors into user-friendly messages."""
        error_str = str(error).lower()
        
        if "connection refused" in error_str or "cannot connect" in error_str:
            return ("Docker daemon not running or not accessible. "
                   "Try: 'docker --version' and 'docker info' to verify Docker installation.")
        
        elif "permission denied" in error_str:
            return ("Permission denied accessing Docker daemon. "
                   "Try running with sudo or add your user to the docker group.")
        
        elif "timeout" in error_str:
            return ("Timeout connecting to Docker daemon. "
                   "The daemon may be slow to respond or under heavy load.")
        
        elif "not found" in error_str:
            return ("Docker daemon not found. "
                   "Ensure Docker is properly installed and running.")
        
        else:
            return f"Docker daemon error: {str(error)}"


# Global detector instance for convenience
_default_detector = None
_detector_lock = threading.Lock()


def get_docker_detector(cache_timeout_seconds: int = 60) -> DockerEnvironmentDetector:
    """Get the global Docker environment detector instance.
    
    Args:
        cache_timeout_seconds: Cache timeout for the detector
        
    Returns:
        Shared DockerEnvironmentDetector instance
    """
    global _default_detector
    
    with _detector_lock:
        if _default_detector is None:
            _default_detector = DockerEnvironmentDetector(cache_timeout_seconds)
        return _default_detector


def is_docker_available(force_refresh: bool = False) -> bool:
    """Convenience function to check Docker availability.
    
    Args:
        force_refresh: If True, bypass cache and perform fresh detection
        
    Returns:
        True if Docker daemon is available, False otherwise
    """
    return get_docker_detector().is_docker_available(force_refresh)


def get_docker_info() -> Optional[Dict]:
    """Convenience function to get Docker daemon information.
    
    Returns:
        Docker daemon version info if available, None otherwise
    """
    return get_docker_detector().get_docker_info()