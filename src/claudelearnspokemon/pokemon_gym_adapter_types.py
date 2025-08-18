"""
Type definitions for PokemonGymAdapter interface specification.

This module defines all TypedDict classes for request/response formats,
configuration options, and error handling contracts used by the PokemonGymAdapter.

The adapter bridges between our EmulatorPool API expectations and the
benchflow-ai/pokemon-gym server API through well-defined type contracts.

Performance Requirements:
- All operations must complete within specified timeout limits
- Batch input processing < 100ms per operation
- State retrieval < 50ms
- Session management overhead < 20ms
- Network timeouts configurable per operation type

Author: John Botmack - Interface Specification and Performance Engineering
"""

from enum import Enum
from typing import Any, Literal, TypedDict

# Use typing_extensions for NotRequired to ensure Python 3.10 compatibility
from typing_extensions import NotRequired

# =============================================================================
# PokemonGymClient Interface Types (Our Expected API)
# =============================================================================


class InputRequest(TypedDict):
    """Request format for sending input sequences to emulator."""

    inputs: str  # Space-separated button sequence: "A B START UP DOWN"


class StateResponse(TypedDict):
    """Response format for game state queries."""

    # Core game state fields
    player_position: NotRequired[dict[str, int | float]]  # {x, y, map_id}
    tile_grid: NotRequired[list[list[int]]]  # 20x18 tile matrix
    inventory: NotRequired[dict[str, Any]]  # Player inventory state
    party: NotRequired[list[dict[str, Any]]]  # Pokemon party data
    flags: NotRequired[dict[str, bool]]  # Story progress flags

    # Technical fields
    timestamp: NotRequired[float]  # Unix timestamp
    frame_count: NotRequired[int]  # Current frame number
    checksum: NotRequired[str]  # State integrity checksum


class ResetResponse(TypedDict):
    """Response format for game reset operations."""

    success: bool
    message: str
    initial_state: NotRequired[StateResponse]
    timestamp: float


class HealthCheckResponse(TypedDict):
    """Response format for health check operations."""

    healthy: bool
    response_time_ms: float
    session_active: NotRequired[bool]
    last_activity: NotRequired[float]
    error: NotRequired[str]


# =============================================================================
# benchflow-ai Pokemon-gym API Types (Target Server API)
# =============================================================================


class GameConfig(TypedDict):
    """Configuration for initializing a Pokemon-gym session."""

    rom_path: NotRequired[str]  # Path to Pokemon ROM file
    save_state: NotRequired[str]  # Initial save state file
    speed_multiplier: NotRequired[float]  # Emulation speed (1.0 = normal)
    audio_enabled: NotRequired[bool]  # Enable audio output
    video_enabled: NotRequired[bool]  # Enable video rendering
    headless: NotRequired[bool]  # Run without GUI
    debug_mode: NotRequired[bool]  # Enable debug logging


class InitializeRequest(TypedDict):
    """Request format for initializing Pokemon-gym session."""

    config: GameConfig


class InitializeResponse(TypedDict):
    """Response format from Pokemon-gym session initialization."""

    session_id: str  # Unique session identifier
    status: Literal["initialized", "error"]
    message: str
    config: GameConfig  # Applied configuration
    timestamp: float


class ActionType(str, Enum):
    """Valid action types for Pokemon-gym API."""

    PRESS_KEY = "press_key"
    HOLD_KEY = "hold_key"
    RELEASE_KEY = "release_key"
    WAIT = "wait"
    RESET = "reset"
    SAVE_STATE = "save_state"
    LOAD_STATE = "load_state"


class ActionRequest(TypedDict):
    """Request format for Pokemon-gym action execution."""

    action_type: ActionType
    keys: NotRequired[list[str]]  # Button names: ["A", "B", "START"]
    duration_ms: NotRequired[int]  # For hold/wait actions
    frames: NotRequired[int]  # Alternative duration in frames
    state_data: NotRequired[str]  # For save/load state actions


class ActionResponse(TypedDict):
    """Response format from Pokemon-gym action execution."""

    success: bool
    action_type: ActionType
    keys_processed: NotRequired[list[str]]
    frames_elapsed: NotRequired[int]
    timestamp: float
    error: NotRequired[str]


class GameStatus(TypedDict):
    """Game status information from Pokemon-gym."""

    session_id: str
    state: Literal["running", "paused", "stopped", "error"]
    frame_count: int
    last_input: NotRequired[float]  # Timestamp of last input
    performance: NotRequired[dict[str, float]]  # FPS, frame time, etc.


class StatusResponse(TypedDict):
    """Response format from Pokemon-gym status queries."""

    session: GameStatus
    game_state: NotRequired[StateResponse]  # Optional detailed game state
    timestamp: float


class StopRequest(TypedDict, total=False):
    """Request format for stopping Pokemon-gym session."""

    save_state: bool  # Whether to save state before stopping
    cleanup: bool  # Whether to cleanup session resources


class StopResponse(TypedDict):
    """Response format from Pokemon-gym session stop."""

    success: bool
    session_id: str
    final_state: NotRequired[str]  # Saved state data if requested
    message: str
    timestamp: float


# =============================================================================
# Error Handling Types
# =============================================================================


class ErrorResponse(TypedDict):
    """Standard error response format."""

    error: bool
    error_type: str  # Error classification
    message: str  # Human-readable error message
    details: NotRequired[dict[str, Any]]  # Additional error context
    timestamp: float
    retry_after: NotRequired[float]  # Suggested retry delay in seconds


class ValidationError(TypedDict):
    """Validation error details."""

    field: str  # Field that failed validation
    value: Any  # Invalid value provided
    expected: str  # Expected value format/type
    constraint: str  # Validation rule that failed


class AdapterError(TypedDict):
    """Adapter-specific error information."""

    adapter_operation: str  # Which adapter method failed
    underlying_error: str  # Original error from benchflow-ai API
    translation_context: dict[str, Any]  # Request translation details
    session_state: NotRequired[str]  # Session state at time of error


# =============================================================================
# Configuration Types
# =============================================================================


class NetworkConfig(TypedDict):
    """Network configuration for adapter operations."""

    connect_timeout: float  # Connection timeout in seconds (default: 5.0)
    read_timeout: float  # Read timeout in seconds (default: 30.0)
    retry_attempts: int  # Number of retry attempts (default: 3)
    retry_backoff: float  # Exponential backoff multiplier (default: 2.0)
    retry_jitter: bool  # Add jitter to retry delays (default: True)
    max_retry_delay: float  # Maximum retry delay in seconds (default: 60.0)


class SessionConfig(TypedDict):
    """Session management configuration."""

    auto_initialize: bool  # Auto-initialize on first use (default: True)
    session_timeout: float  # Session timeout in seconds (default: 300.0)
    heartbeat_interval: float  # Heartbeat ping interval (default: 30.0)
    cleanup_on_exit: bool  # Cleanup session on adapter close (default: True)
    persist_session: bool  # Try to reuse existing sessions (default: False)


class PerformanceConfig(TypedDict):
    """Performance tuning configuration."""

    input_batch_size: int  # Max inputs to batch together (default: 10)
    input_delay_ms: float  # Delay between individual inputs (default: 50.0)
    state_cache_ttl: float  # State cache TTL in seconds (default: 1.0)
    parallel_requests: bool  # Allow parallel API requests (default: False)
    connection_pool_size: int  # HTTP connection pool size (default: 5)


class LoggingConfig(TypedDict):
    """Logging configuration for adapter operations."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_requests: bool  # Log all API requests (default: True)
    log_responses: bool  # Log all API responses (default: False)
    log_performance: bool  # Log performance metrics (default: True)
    log_session_events: bool  # Log session lifecycle events (default: True)


class AdapterConfig(TypedDict):
    """Complete adapter configuration."""

    base_url: str  # Pokemon-gym server URL (default: "http://localhost:8080")
    port: int  # Server port number
    container_id: str  # Docker container identifier

    # Sub-configurations
    network: NotRequired[NetworkConfig]
    session: NotRequired[SessionConfig]
    performance: NotRequired[PerformanceConfig]
    logging: NotRequired[LoggingConfig]
    game: NotRequired[GameConfig]


# =============================================================================
# Performance Monitoring Types
# =============================================================================


class PerformanceMetrics(TypedDict):
    """Performance metrics for adapter operations."""

    operation: str  # Operation name (send_input, get_state, etc.)
    start_time: float  # Operation start timestamp
    end_time: float  # Operation end timestamp
    duration_ms: float  # Total duration in milliseconds
    network_time_ms: float  # Network request time
    translation_time_ms: float  # Request/response translation time
    success: bool  # Whether operation succeeded
    retry_count: int  # Number of retries performed
    bytes_sent: NotRequired[int]  # Request payload size
    bytes_received: NotRequired[int]  # Response payload size


class SessionMetrics(TypedDict):
    """Session-level performance metrics."""

    session_id: str
    created_at: float  # Session creation timestamp
    last_activity: float  # Last successful operation
    total_operations: int  # Total operations performed
    successful_operations: int  # Successful operations count
    failed_operations: int  # Failed operations count
    average_response_time_ms: float  # Average response time
    total_bytes_transferred: int  # Total network traffic


# =============================================================================
# Contract Test Types
# =============================================================================


class TestScenario(TypedDict):
    """Test scenario definition for contract validation."""

    name: str  # Test scenario name
    description: str  # Test description
    setup: dict[str, Any]  # Test setup parameters
    input: dict[str, Any]  # Test input data
    expected_output: dict[str, Any]  # Expected response
    performance_requirements: NotRequired[dict[str, float]]  # Performance SLAs
    error_conditions: NotRequired[list[str]]  # Expected error scenarios


class ContractValidation(TypedDict):
    """Contract validation result."""

    test_name: str
    passed: bool
    actual_output: dict[str, Any]
    expected_output: dict[str, Any]
    performance_metrics: PerformanceMetrics
    validation_errors: list[ValidationError]
    timestamp: float


# =============================================================================
# Utility Types
# =============================================================================

# Union types for common scenarios
AnyRequest = InputRequest | InitializeRequest | ActionRequest | StopRequest
AnyResponse = (
    StateResponse
    | ResetResponse
    | InitializeResponse
    | ActionResponse
    | StatusResponse
    | StopResponse
    | ErrorResponse
)

# Type aliases for better readability
ButtonSequence = str  # Space-separated button sequence
SessionId = str  # Unique session identifier
Timestamp = float  # Unix timestamp
ContainerId = str  # Docker container ID


# =============================================================================
# Constants
# =============================================================================

# Default timeout values (in seconds)
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 30.0
DEFAULT_SESSION_TIMEOUT = 300.0
DEFAULT_INPUT_DELAY_MS = 50.0

# Performance thresholds (in milliseconds)
MAX_INPUT_PROCESSING_TIME = 100.0
MAX_STATE_RETRIEVAL_TIME = 50.0
MAX_SESSION_OVERHEAD = 20.0

# Retry policy defaults
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_MAX_RETRY_DELAY = 60.0

# Valid button names for Pokemon-gym
VALID_BUTTONS = [
    "A",
    "B",
    "START",
    "SELECT",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "L",
    "R",  # For games that support shoulder buttons
]

# Valid action types
VALID_ACTION_TYPES = [item.value for item in ActionType]
