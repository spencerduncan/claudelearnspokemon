"""Claude Learns Pokemon - Speedrun learning agent system."""

__version__ = "0.1.0"

from .circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitConfig, CircuitState
from .emulator_pool import EmulatorPool, EmulatorPoolError, ExecutionResult, PokemonGymClient
from .enhanced_client import EnhancedPokemonGymClient

# Error recovery and retry logic components
from .retry_manager import FailureType, RetryConfig, RetryManager, RetryOutcome
from .session_recovery import SessionConfig, SessionRecovery, SessionRecoveryError, SessionState

# ML-guided experiment selection components
from .config import CONFIG, ConfigManager
from .experiment_models import (
    ExperimentCandidate, ExperimentResult, ExperimentMetrics,
    ExperimentStatus, SelectionStrategy
)

# Import compatibility layer components
try:
    from .pokemon_gym_adapter import PokemonGymAdapter  # noqa: F401
    from .pokemon_gym_factory import create_pokemon_client  # noqa: F401

    COMPATIBILITY_LAYER_AVAILABLE = True
except ImportError:
    COMPATIBILITY_LAYER_AVAILABLE = False

__all__ = [
    # Core components
    "EmulatorPool",
    "EmulatorPoolError",
    "ExecutionResult",
    "PokemonGymClient",
    # Error recovery components
    "RetryManager",
    "RetryConfig",
    "RetryOutcome",
    "FailureType",
    "CircuitBreaker",
    "CircuitConfig",
    "CircuitState",
    "CircuitBreakerError",
    "SessionRecovery",
    "SessionConfig",
    "SessionState",
    "SessionRecoveryError",
    "EnhancedPokemonGymClient",
    # ML-guided experiment selection components
    "CONFIG",
    "ConfigManager",
    "ExperimentCandidate",
    "ExperimentResult", 
    "ExperimentMetrics",
    "ExperimentStatus",
    "SelectionStrategy",
]

# Add compatibility layer to exports if available
if COMPATIBILITY_LAYER_AVAILABLE:
    __all__.extend(["PokemonGymAdapter", "create_pokemon_client"])
