"""
PokemonGymAdapter: Production-ready HTTP client for benchflow-ai endpoints.
# mypy: disable-error-code=no-any-return

This adapter extends PokemonGymClient to support the benchflow-ai API specification
while maintaining full backward compatibility. Built with Google-scale engineering
principles for <100ms response times and production reliability.

Author: Bot Dean - Production Systems Engineering
"""

import logging
from typing import Any

try:
    import httpx
except ImportError:
    # Graceful fallback if httpx not available
    class httpx:  # type: ignore[misc]
        class Client:
            def __init__(self, **kwargs):
                pass

            def post(self, *args, **kwargs):
                pass

            def get(self, *args, **kwargs):
                pass

            def close(self):
                pass

            @property
            def is_closed(self):
                return True

        class Limits:
            def __init__(self, **kwargs):
                pass

        class TimeoutException(Exception):
            pass

        class HTTPStatusError(Exception):
            def __init__(self, message, request=None, response=None):
                super().__init__(message)
                self.response = response


from .emulator_pool import EmulatorPoolError, PokemonGymClient

# Configure production-grade logging
logger = logging.getLogger(__name__)


class PokemonGymAdapterError(EmulatorPoolError):
    """
    Custom exception for PokemonGymAdapter operations.

    Extends EmulatorPoolError to maintain error hierarchy while providing
    adapter-specific error context for production debugging.
    """

    pass


class PokemonGymAdapter(PokemonGymClient):
    """
    Production-ready adapter for benchflow-ai Pokemon gym environments.

    Extends PokemonGymClient with benchflow-ai specific endpoints while maintaining
    full backward compatibility. Uses httpx with connection pooling for optimal
    performance (<100ms target processing time).

    Supports both pokemon-gym and benchflow-ai API endpoints:
    - Legacy: /input, /state, /reset, /health
    - New: /initialize, /action, /status, /stop
    """

    def __init__(
        self,
        port: int,
        container_id: str,
        server_url: str | None = None,
        timeout_config: dict[str, float] | None = None,
        connection_limits: dict[str, int | float] | None = None,
    ):
        """
        Initialize adapter with production configuration.

        Args:
            port: HTTP port for emulator communication
            container_id: Docker container ID for this emulator
            server_url: Override base server URL (default: localhost)
            timeout_config: Custom timeout settings for different operations
            connection_limits: HTTP connection pool configuration
        """
        # Initialize parent class with existing functionality
        super().__init__(port, container_id)

        # Production-grade timeout configuration (<100ms target)
        self.timeout_config = timeout_config or {
            "initialize": 5.0,  # Environment setup can take longer
            "action": 0.1,  # Critical path - must be fast
            "status": 0.05,  # Health checks should be instant
            "stop": 2.0,  # Cleanup operations
            "default": 0.1,  # Default for all operations
        }

        # HTTP connection pool configuration for production scale
        connection_limits = connection_limits or {
            "max_keepalive_connections": 10,
            "max_connections": 20,
            "keepalive_expiry": 30.0,
        }

        # Override base URL if custom server specified
        if server_url:
            self.base_url = server_url.rstrip("/")

        # Initialize high-performance HTTP client with connection pooling
        limits = httpx.Limits(
            max_keepalive_connections=int(connection_limits.get("max_keepalive_connections", 10)),
            max_connections=int(connection_limits.get("max_connections", 20)),
            keepalive_expiry=float(connection_limits.get("keepalive_expiry", 30.0)),
        )

        self.http_client = httpx.Client(
            base_url=self.base_url,
            limits=limits,
            timeout=None,  # We set timeouts per-request for fine control
            headers={
                "User-Agent": f"PokemonGymAdapter/1.0 (container-{container_id[:12]})",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

        # Track adapter state for production observability
        self._session_initialized = False
        self._last_action_time: float | None = None

        logger.info(
            f"PokemonGymAdapter initialized: port={port}, "
            f"container={container_id[:12]}, base_url={self.base_url}"
        )

    def initialize_session(self, config: dict[str, Any] | None = None) -> dict[str, Any]:  # type: ignore[misc]
        """
        Initialize a new benchflow-ai session.

        This is the entry point for benchflow-ai environments, equivalent to
        starting a new game or resetting to a specific state.

        Args:
            config: Session configuration (game state, difficulty, etc.)

        Returns:
            Session initialization response with initial state

        Raises:
            PokemonGymAdapterError: On initialization failure
        """
        try:
            start_time = self._get_current_time()

            payload = config or {}

            response = self.http_client.post(
                "/initialize", json=payload, timeout=self.timeout_config["initialize"]
            )
            response.raise_for_status()

            result: dict[str, Any] = response.json()
            self._session_initialized = True

            execution_time = self._get_current_time() - start_time
            logger.info(
                f"Session initialized on {self}: config={len(payload)} keys, "
                f"time={execution_time:.3f}s"
            )

            return result  # type: ignore[no-any-return]

        except httpx.TimeoutException as e:
            raise PokemonGymAdapterError(
                f"Session initialization timeout on {self}: {e}. "
                f"Consider increasing timeout or checking server performance."
            ) from e
        except httpx.HTTPStatusError as e:
            raise PokemonGymAdapterError(
                f"Session initialization failed on {self}: {e.response.status_code} "
                f"{e.response.text}. Check server status and configuration."
            ) from e
        except Exception as e:
            raise PokemonGymAdapterError(
                f"Unexpected error during session initialization on {self}: {e}"
            ) from e

    def execute_action(self, action: str | dict[str, Any]) -> dict[str, Any]:  # type: ignore[misc]
        """
        Execute action on benchflow-ai environment.

        This is the critical path method for game interaction. Optimized for
        <100ms response time with comprehensive error handling.

        Args:
            action: Action to execute (input sequence or structured action)

        Returns:
            Action execution response with new state and rewards

        Raises:
            PokemonGymAdapterError: On execution failure or timeout
        """
        import time

        start_time = time.perf_counter()

        try:
            # Ensure session is initialized before actions
            if not self._session_initialized:
                logger.warning(f"Action attempted on uninitialized session {self}")

            # Normalize action format for API
            if isinstance(action, str):
                payload = {"action": action}
            else:
                payload = action

            response = self.http_client.post(
                "/action", json=payload, timeout=self.timeout_config["action"]
            )
            response.raise_for_status()

            result: dict[str, Any] = response.json()

            execution_time = time.perf_counter() - start_time
            self._last_action_time = execution_time

            # Log performance warning if approaching timeout
            if execution_time > 0.08:  # 80% of 100ms target
                logger.warning(
                    f"Action execution approaching timeout on {self}: "
                    f"{execution_time:.3f}s (target: <0.1s)"
                )
            else:
                logger.debug(f"Action executed on {self}: time={execution_time:.3f}s")

            return result  # type: ignore[no-any-return]

        except httpx.TimeoutException as e:
            execution_time = time.perf_counter() - start_time
            raise PokemonGymAdapterError(
                f"Action execution timeout on {self} after {execution_time:.3f}s: {e}. "
                f"This violates <100ms performance requirement."
            ) from e
        except httpx.HTTPStatusError as e:
            raise PokemonGymAdapterError(
                f"Action execution failed on {self}: {e.response.status_code} "
                f"{e.response.text}. Action: {action}"
            ) from e
        except Exception as e:
            raise PokemonGymAdapterError(
                f"Unexpected error during action execution on {self}: {e}"
            ) from e

    def get_session_status(self) -> dict[str, Any]:  # type: ignore[misc]
        """
        Get current session status from benchflow-ai environment.

        Provides detailed state information for monitoring and debugging.
        Optimized for minimal latency as this is called frequently.

        Returns:
            Current session status with state, metrics, and diagnostics

        Raises:
            PokemonGymAdapterError: On status retrieval failure
        """
        import time

        start_time = time.perf_counter()

        try:
            response = self.http_client.get("/status", timeout=self.timeout_config["status"])
            response.raise_for_status()

            result: dict[str, Any] = response.json()

            execution_time = time.perf_counter() - start_time
            logger.debug(f"Status retrieved from {self}: time={execution_time:.3f}s")

            return result  # type: ignore[no-any-return]

        except httpx.TimeoutException as e:
            raise PokemonGymAdapterError(
                f"Status retrieval timeout on {self}: {e}. " f"Expected <50ms response time."
            ) from e
        except httpx.HTTPStatusError as e:
            raise PokemonGymAdapterError(
                f"Status retrieval failed on {self}: {e.response.status_code} " f"{e.response.text}"
            ) from e
        except Exception as e:
            raise PokemonGymAdapterError(
                f"Unexpected error during status retrieval on {self}: {e}"
            ) from e

    def stop_session(self, save_state: bool = False) -> dict[str, Any]:  # type: ignore[misc]
        """
        Stop current benchflow-ai session.

        Gracefully terminates the session with optional state saving.
        Handles cleanup and resource deallocation.

        Args:
            save_state: Whether to save current state for future restoration

        Returns:
            Session termination response with final metrics

        Raises:
            PokemonGymAdapterError: On session termination failure
        """
        try:
            payload = {"save_state": save_state}

            response = self.http_client.post(
                "/stop", json=payload, timeout=self.timeout_config["stop"]
            )
            response.raise_for_status()

            result: dict[str, Any] = response.json()
            self._session_initialized = False

            logger.info(f"Session stopped on {self}: save_state={save_state}")

            return result  # type: ignore[no-any-return]

        except httpx.TimeoutException as e:
            raise PokemonGymAdapterError(
                f"Session stop timeout on {self}: {e}. " f"Server may be unresponsive."
            ) from e
        except httpx.HTTPStatusError as e:
            raise PokemonGymAdapterError(
                f"Session stop failed on {self}: {e.response.status_code} " f"{e.response.text}"
            ) from e
        except Exception as e:
            raise PokemonGymAdapterError(
                f"Unexpected error during session stop on {self}: {e}"
            ) from e

    def is_session_active(self) -> bool:
        """
        Check if benchflow-ai session is currently active.

        Fast health check that combines local state with server verification.

        Returns:
            True if session is active and responsive, False otherwise
        """
        try:
            # Quick local check first
            if not self._session_initialized:
                return False

            # Verify with server (fast status check)
            status = self.get_session_status()
            return bool(status.get("active", False))

        except Exception as e:
            logger.debug(f"Session activity check failed on {self}: {e}")
            return False

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get adapter performance metrics for monitoring.

        Provides production telemetry for performance analysis and optimization.

        Returns:
            Performance metrics including timing and resource usage
        """
        return {
            "session_initialized": self._session_initialized,
            "last_action_time": self._last_action_time,
            "timeout_config": self.timeout_config,
            "connection_pool_info": {"is_closed": self.http_client.is_closed},
        }

    def close(self) -> None:
        """
        Close adapter and release resources.

        Extends parent close() method to handle httpx client cleanup.
        Production pattern: always clean up resources explicitly.
        """
        try:
            # Close httpx client first
            if hasattr(self, "http_client") and not self.http_client.is_closed:
                self.http_client.close()
                logger.debug(f"HTTP client closed for {self}")
        except Exception as e:
            logger.error(f"Error closing HTTP client for {self}: {e}")
        finally:
            # Always call parent cleanup
            super().close()

    @staticmethod
    def create_adapter(
        port: int, container_id: str, adapter_type: str = "benchflow", **kwargs
    ) -> "PokemonGymAdapter":
        """
        Factory method for creating adapters with different configurations.

        Provides a clean API for adapter creation with preset configurations
        for different use cases and performance requirements.

        Args:
            port: HTTP port for emulator communication
            container_id: Docker container ID
            adapter_type: Preset configuration type
            **kwargs: Override any configuration parameters

        Returns:
            Configured PokemonGymAdapter instance

        Raises:
            ValueError: For unknown adapter types
        """
        presets = {
            "benchflow": {
                "timeout_config": {
                    "initialize": 5.0,
                    "action": 0.1,
                    "status": 0.05,
                    "stop": 2.0,
                    "default": 0.1,
                },
                "connection_limits": {
                    "max_keepalive_connections": 10,
                    "max_connections": 20,
                    "keepalive_expiry": 30.0,
                },
            },
            "high_performance": {
                "timeout_config": {
                    "initialize": 3.0,
                    "action": 0.05,
                    "status": 0.025,
                    "stop": 1.0,
                    "default": 0.05,
                },
                "connection_limits": {
                    "max_keepalive_connections": 20,
                    "max_connections": 50,
                    "keepalive_expiry": 60.0,
                },
            },
            "development": {
                "timeout_config": {
                    "initialize": 10.0,
                    "action": 1.0,
                    "status": 0.5,
                    "stop": 5.0,
                    "default": 1.0,
                },
                "connection_limits": {
                    "max_keepalive_connections": 5,
                    "max_connections": 10,
                    "keepalive_expiry": 10.0,
                },
            },
        }

        if adapter_type not in presets:
            raise ValueError(
                f"Unknown adapter type: {adapter_type}. " f"Available: {list(presets.keys())}"
            )

        # Merge preset with overrides
        config = presets[adapter_type].copy()
        config.update(kwargs)

        logger.info(
            f"Creating {adapter_type} adapter: port={port}, " f"container={container_id[:12]}"
        )

        # Extract properly typed parameters with explicit type casting
        server_url: str | None = None
        timeout_config: dict[str, float] | None = None
        connection_limits: dict[str, int | float] | None = None

        if "server_url" in config:
            server_url = config["server_url"]
        if "timeout_config" in config:
            timeout_config = config["timeout_config"]
        if "connection_limits" in config:
            connection_limits = config["connection_limits"]

        return PokemonGymAdapter(
            port,
            container_id,
            server_url=server_url,
            timeout_config=timeout_config,
            connection_limits=connection_limits,
        )

    def _get_current_time(self) -> float:
        """Get high-precision current time for performance measurement."""
        import time

        return time.perf_counter()

    def __str__(self) -> str:
        """String representation for logging and debugging."""
        session_status = "active" if self._session_initialized else "inactive"
        return f"PokemonGymAdapter(port={self.port}, container={self.container_id[:12]}, session={session_status})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"PokemonGymAdapter(port={self.port}, container_id='{self.container_id}', "
            f"base_url='{self.base_url}', session_initialized={self._session_initialized})"
        )
