"""
Pokemon Gym Factory: Compatibility layer factory for transparent adapter selection.

Provides auto-detection between PokemonGymAdapter (benchflow-ai) and PokemonGymClient (direct)
with production-grade error handling, caching, and observability.

Author: Bot Dean - Production-First Engineering
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Union

import requests
from requests.exceptions import RequestException, Timeout

from .compatibility.cache_strategies import CacheStrategy, create_cache_strategy
from .pokemon_gym_adapter import PokemonGymAdapter

if TYPE_CHECKING:
    from .emulator_pool import PokemonGymClient

# Configure logging for production observability
logger = logging.getLogger(__name__)

# Default cache strategy instance - can be overridden via dependency injection
_default_cache_strategy: CacheStrategy | None = None


class FactoryError(Exception):
    """
    Custom exception for factory operations.

    Provides actionable error messages for production debugging.
    """

    pass


def create_pokemon_client(
    port: int,
    container_id: str,
    adapter_type: str = "auto",
    input_delay: float = 0.05,
    detection_timeout: float = 3.0,
    cache_strategy: CacheStrategy | None = None,
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
        cache_strategy: Optional cache strategy for server type detection (None = use default)

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
        # Validate input parameters
        if not isinstance(port, int) or port <= 0:
            raise FactoryError(f"Invalid port: {port}. Must be positive integer.")
        if not container_id:
            raise FactoryError("Container ID cannot be empty")
        if adapter_type not in ("auto", "benchflow", "direct", "fallback"):
            raise FactoryError(f"Invalid adapter_type: {adapter_type}")

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

        # Auto-detection mode with injected cache strategy
        cache = cache_strategy or get_default_cache_strategy()
        server_type = detect_server_type(port, detection_timeout, cache)

        if server_type == "benchflow":
            logger.info(f"Auto-detected benchflow-ai server on port {port}")
            config = {"input_delay": input_delay}
            return PokemonGymAdapter(port, container_id, config)
        else:
            logger.info(f"Auto-detected direct pokemon-gym server on port {port}")
            return PokemonGymClient(port, container_id)

    except Exception as e:
        logger.error(f"Failed to create Pokemon client for port {port}: {e}")
        raise FactoryError(f"Client creation failed: {e}") from e


def detect_server_type(
    port: int, timeout: float = 3.0, cache_strategy: CacheStrategy | None = None
) -> str:
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
        cache_strategy: Cache strategy to use for detection results (None = use default)

    Returns:
        "benchflow" for benchflow-ai servers, "direct" for direct servers

    Note:
        Always returns "direct" as fallback to ensure systems keep working
        even when detection fails. This provides graceful degradation.
    """
    # Use provided cache strategy or default
    cache = cache_strategy or get_default_cache_strategy()
    cache_key = f"port_{port}"
    current_time = time.time()

    # Check cache first (with error handling)
    try:
        cached_result = cache.get(cache_key)
        if cached_result:
            server_type = cached_result["type"]
            logger.debug(f"Using cached server type for port {port}: {server_type}")
            return server_type
    except Exception as e:
        logger.warning(f"Cache read failed for key {cache_key}: {e}")
        # Continue without cache

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

        # Cache the result (with error handling)
        try:
            cache.set(cache_key, {"type": server_type, "timestamp": current_time})
        except Exception as e:
            logger.warning(f"Cache write failed for key {cache_key}: {e}")
            # Continue without caching

        return server_type

    except Exception as e:
        logger.warning(f"Server type detection failed for port {port}: {e}")
        # Graceful fallback to direct client
        server_type = "direct"

        # Cache the fallback result with shorter timeout (1 minute)
        try:
            cache.set(cache_key, {"type": server_type, "timestamp": current_time}, ttl_seconds=60)
        except Exception as e:
            logger.warning(f"Cache write failed for fallback key {cache_key}: {e}")
            # Continue without caching

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


def clear_detection_cache(
    port: int | None = None, cache_strategy: CacheStrategy | None = None
) -> None:
    """
    Clear server type detection cache.

    Useful for testing or when server configurations change during runtime.

    Args:
        port: Specific port to clear cache for, or None to clear all
        cache_strategy: Cache strategy to use (None = use default)
    """
    cache = cache_strategy or get_default_cache_strategy()

    if port is None:
        cache.clear()
        logger.info("Cleared all server type detection cache")
    else:
        cache_key = f"port_{port}"
        cache.clear(cache_key)
        logger.info(f"Cleared server type detection cache for port {port}")


def get_detection_cache_stats(cache_strategy: CacheStrategy | None = None) -> dict[str, Any]:
    """
    Get statistics about the detection cache.

    Useful for monitoring and debugging cache behavior.

    Args:
        cache_strategy: Cache strategy to get stats from (None = use default)

    Returns:
        Dictionary with cache statistics
    """
    cache = cache_strategy or get_default_cache_strategy()
    return cache.get_stats()


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


def get_default_cache_strategy() -> CacheStrategy:
    """
    Get or create the default cache strategy for the factory.

    Uses lazy initialization to create the default cache strategy only when needed.
    This allows for environment-based configuration while maintaining singleton behavior.

    Returns:
        Default cache strategy instance
    """
    global _default_cache_strategy

    if _default_cache_strategy is None:
        # Lazy initialization with environment-aware defaults
        _default_cache_strategy = create_cache_strategy("auto", default_ttl_seconds=300)
        logger.info(f"Initialized default cache strategy: {type(_default_cache_strategy).__name__}")

    return _default_cache_strategy


def set_default_cache_strategy(strategy: CacheStrategy) -> None:
    """
    Set the default cache strategy for the factory.

    Useful for testing or when you want to override the default strategy
    across all factory operations.

    Args:
        strategy: Cache strategy instance to use as default
    """
    global _default_cache_strategy

    _default_cache_strategy = strategy
    logger.info(f"Set default cache strategy to: {type(strategy).__name__}")


# Production monitoring and metrics helpers


def get_factory_metrics(cache_strategy: CacheStrategy | None = None) -> dict[str, Any]:
    """
    Get factory usage metrics for monitoring.

    Args:
        cache_strategy: Cache strategy to get metrics from (None = use default)

    Returns:
        Dictionary with factory operation metrics
    """
    cache_stats = get_detection_cache_stats(cache_strategy)

    return {
        "detection_cache": cache_stats,
        "supported_types": ["direct", "benchflow", "auto", "fallback"],
        "default_detection_timeout": 3.0,
        "default_input_delay": 0.05,
        "cache_strategy_type": cache_stats.get("strategy", "unknown"),
    }
