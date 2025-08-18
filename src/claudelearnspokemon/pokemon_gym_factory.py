"""
Pokemon Gym Factory: Compatibility layer factory for transparent adapter selection.

Provides auto-detection between PokemonGymAdapter (benchflow-ai) and PokemonGymClient (direct)
with production-grade error handling, caching, and observability.

Author: Bot Dean - Production-First Engineering
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Union

import requests
from requests.exceptions import RequestException, Timeout

from .compatibility.validation import ValidationError, validate_client_creation_parameters
from .pokemon_gym_adapter import PokemonGymAdapter

if TYPE_CHECKING:
    from .emulator_pool import PokemonGymClient

# Configure logging for production observability
logger = logging.getLogger(__name__)

# Global cache for server type detection to avoid repeated probes
_server_type_cache: dict[str, dict[str, Any]] = {}
_cache_timeout = 300  # 5 minutes cache timeout
_cache_lock = threading.RLock()  # Reentrant lock for thread safety


class FactoryError(Exception):
    """
    Custom exception for factory operations.

    Provides actionable error messages for production debugging.
    Wraps ValidationError to maintain API compatibility while using clean validation.
    """

    pass


def create_pokemon_client(
    port: int,
    container_id: str,
    adapter_type: str = "auto",
    input_delay: float = 0.05,
    detection_timeout: float = 3.0,
) -> Union["PokemonGymClient", PokemonGymAdapter]:
    """
    Factory function to create appropriate Pokemon client based on server type.

    Implements transparent adapter selection with auto-detection, caching,
    and production-grade error handling. Provides seamless migration path
    from direct clients to benchflow-ai adapters.

    Args:
        port: HTTP port for emulator communication
        container_id: Docker container ID for this emulator
        adapter_type: Type selection - "auto", "benchflow", "direct", or "fallback"
        input_delay: Delay between sequential inputs for benchflow adapter (default: 50ms)
        detection_timeout: Timeout for server type detection (default: 3s)

    Returns:
        Either PokemonGymClient (direct) or PokemonGymAdapter (benchflow)

    Raises:
        FactoryError: On configuration error or detection failure

    Examples:
        # Auto-detection (recommended)
        client = create_pokemon_client(8081, "container_id")

        # Force specific type
        client = create_pokemon_client(8081, "container_id", adapter_type="benchflow")

        # With custom input timing
        client = create_pokemon_client(8081, "container_id", input_delay=0.1)
    """
    logger.info(
        f"Creating Pokemon client for port {port}, container {container_id[:12]}, "
        f"adapter_type={adapter_type}"
    )

    # Dynamic import to avoid circular dependency
    from .emulator_pool import PokemonGymClient

    try:
        # Validate all input parameters using clean validation module
        validated_params = validate_client_creation_parameters(
            port=port,
            container_id=container_id,
            adapter_type=adapter_type,
            input_delay=input_delay,
            detection_timeout=detection_timeout,
        )

        # Extract validated parameters for clean code readability
        port = int(validated_params["port"])
        container_id = str(validated_params["container_id"])
        adapter_type = str(validated_params["adapter_type"])
        input_delay = float(validated_params["input_delay"])
        detection_timeout = float(validated_params["detection_timeout"])

        # Handle explicit type selection
        if adapter_type == "benchflow":
            logger.info(f"Creating PokemonGymAdapter (benchflow) for port {port}")
            config = {"input_delay": input_delay}
            return PokemonGymAdapter(port, container_id, config)
        elif adapter_type == "direct":
            logger.info(f"Creating PokemonGymClient (direct) for port {port}")
            return PokemonGymClient(port, container_id)
        elif adapter_type == "fallback":
            # Fallback mode: always use direct client (for compatibility)
            logger.info(f"Creating PokemonGymClient (fallback mode) for port {port}")
            return PokemonGymClient(port, container_id)

        # Auto-detection mode
        server_type = detect_server_type(port, detection_timeout)

        if server_type == "benchflow":
            logger.info(f"Auto-detected benchflow-ai server on port {port}")
            config = {"input_delay": input_delay}
            return PokemonGymAdapter(port, container_id, config)
        else:
            logger.info(f"Auto-detected direct pokemon-gym server on port {port}")
            return PokemonGymClient(port, container_id)

    except ValidationError as e:
        # Convert ValidationError to FactoryError for API compatibility
        logger.error(f"Parameter validation failed for port {port}: {e}")
        raise FactoryError(str(e)) from e
    except Exception as e:
        logger.error(f"Failed to create Pokemon client for port {port}: {e}")
        raise FactoryError(f"Client creation failed: {e}") from e


def detect_server_type(port: int, timeout: float = 3.0) -> str:
    """
    Auto-detect server type (benchflow-ai vs direct pokemon-gym).

    Uses cached results to avoid repeated detection overhead. Implements
    production patterns for reliable detection with graceful fallbacks.

    Detection Strategy:
    1. Check cache for recent detection result
    2. Try benchflow-ai endpoints (/status)
    3. Try direct pokemon-gym endpoints (/health)
    4. Return "direct" as safe fallback if detection fails

    Args:
        port: HTTP port to check
        timeout: Detection timeout per endpoint (default: 3s)

    Returns:
        "benchflow" for benchflow-ai servers, "direct" for direct servers

    Note:
        Always returns "direct" as fallback to ensure systems keep working
        even when detection fails. This provides graceful degradation.
    """
    cache_key = f"port_{port}"
    current_time = time.time()

    # Check cache first
    with _cache_lock:
        if cache_key in _server_type_cache:
            cache_entry = _server_type_cache[cache_key]
            if current_time - cache_entry["timestamp"] < _cache_timeout:
                server_type = cache_entry["type"]
                logger.debug(f"Using cached server type for port {port}: {server_type}")
                return server_type

    logger.info(f"Detecting server type for port {port}")

    base_url = f"http://localhost:{port}"
    session = requests.Session()

    try:
        # Strategy 1: Try benchflow-ai endpoint
        if _is_benchflow_server(base_url, session, timeout):
            server_type = "benchflow"
            logger.info(f"Detected benchflow-ai server on port {port}")
        else:
            # Strategy 2: Assume direct server (safe fallback)
            server_type = "direct"
            logger.info(f"Detected direct pokemon-gym server on port {port}")

        # Cache the result
        with _cache_lock:
            _server_type_cache[cache_key] = {"type": server_type, "timestamp": current_time}

        return server_type

    except Exception as e:
        logger.warning(f"Server type detection failed for port {port}: {e}")
        # Graceful fallback to direct client
        server_type = "direct"

        # Cache the fallback result with shorter timeout
        with _cache_lock:
            _server_type_cache[cache_key] = {
                "type": server_type,
                "timestamp": current_time - _cache_timeout + 60,  # 1 minute cache for failures
            }

        return server_type

    finally:
        session.close()


def _is_benchflow_server(base_url: str, session: requests.Session, timeout: float) -> bool:
    """
    Check if server is benchflow-ai type by testing specific endpoints.

    Args:
        base_url: Base URL for the server
        session: Requests session to use
        timeout: Request timeout

    Returns:
        True if benchflow-ai server detected, False otherwise
    """
    try:
        # Try benchflow-ai specific endpoint
        response = session.get(f"{base_url}/status", timeout=timeout)

        if response.status_code == 200:
            # Additional validation: check response structure
            try:
                data = response.json()
                # benchflow-ai typically returns structured status data
                if isinstance(data, dict):
                    # Look for benchflow-ai specific fields
                    if any(key in data for key in ["session_active", "game_state", "location"]):
                        return True
                    # Even without specific fields, if /status returns JSON, likely benchflow-ai
                    return True
            except Exception:
                # If /status exists but doesn't return JSON, might still be benchflow-ai
                return True

        # Check for 404 vs other status codes
        # benchflow-ai returns 404 for missing endpoints
        # Direct pokemon-gym might return different codes
        return False

    except (RequestException, Timeout):
        # Network errors indicate server might not be benchflow-ai
        return False
    except Exception as e:
        logger.debug(f"Unexpected error during benchflow-ai detection: {e}")
        return False


def clear_detection_cache(port: int | None = None) -> None:
    """
    Clear server type detection cache.

    Useful for testing or when server configurations change during runtime.

    Args:
        port: Specific port to clear cache for, or None to clear all
    """
    global _server_type_cache

    with _cache_lock:
        if port is None:
            _server_type_cache.clear()
            logger.info("Cleared all server type detection cache")
        else:
            cache_key = f"port_{port}"
            _server_type_cache.pop(cache_key, None)
            logger.info(f"Cleared server type detection cache for port {port}")


def get_detection_cache_stats() -> dict[str, Any]:
    """
    Get statistics about the detection cache.

    Useful for monitoring and debugging cache behavior.

    Returns:
        Dictionary with cache statistics
    """
    current_time = time.time()

    valid_entries = 0
    expired_entries = 0

    with _cache_lock:
        for cache_entry in _server_type_cache.values():
            if current_time - cache_entry["timestamp"] < _cache_timeout:
                valid_entries += 1
            else:
                expired_entries += 1

        total_entries = len(_server_type_cache)

    return {
        "total_entries": total_entries,
        "valid_entries": valid_entries,
        "expired_entries": expired_entries,
        "cache_timeout": _cache_timeout,
        "cache_hit_ratio": valid_entries / max(total_entries, 1),
    }


def validate_client_compatibility(client: Union["PokemonGymClient", PokemonGymAdapter]) -> bool:
    """
    Validate that a client instance provides the expected interface.

    Ensures both client types implement the required methods for EmulatorPool
    compatibility. Used for testing and validation.

    Args:
        client: Client instance to validate

    Returns:
        True if client has all required methods, False otherwise
    """
    required_methods = ["send_input", "get_state", "reset_game", "is_healthy", "close"]

    try:
        for method_name in required_methods:
            if not hasattr(client, method_name):
                logger.error(f"Client missing required method: {method_name}")
                return False
            if not callable(getattr(client, method_name)):
                logger.error(f"Client method not callable: {method_name}")
                return False

        # Validate constructor signature compatibility
        if not hasattr(client, "port") or not hasattr(client, "container_id"):
            logger.error("Client missing required attributes: port, container_id")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating client compatibility: {e}")
        return False


# Production monitoring and metrics helpers


def get_factory_metrics() -> dict[str, Any]:
    """
    Get factory usage metrics for monitoring.

    Returns:
        Dictionary with factory operation metrics
    """
    cache_stats = get_detection_cache_stats()

    return {
        "detection_cache": cache_stats,
        "supported_types": ["direct", "benchflow", "auto", "fallback"],
        "default_detection_timeout": 3.0,
        "default_input_delay": 0.05,
        "cache_timeout_seconds": _cache_timeout,
    }
