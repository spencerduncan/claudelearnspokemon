"""Claude Learns Pokemon - Speedrun learning agent system."""

__version__ = "0.1.0"

from .circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitConfig, CircuitState
from .emulator_pool import EmulatorPool, EmulatorPoolError, ExecutionResult, PokemonGymClient
from .enhanced_client import EnhancedPokemonGymClient
from .pokemon_gym_adapter import PokemonGymAdapter, PokemonGymAdapterError

# Error recovery and retry logic components
from .retry_manager import FailureType, RetryConfig, RetryManager, RetryOutcome
from .session_recovery import SessionConfig, SessionRecovery, SessionRecoveryError, SessionState

__all__ = [
    # Core components
    "EmulatorPool",
    "EmulatorPoolError",
    "ExecutionResult",
    "PokemonGymClient",
    "PokemonGymAdapter",
    "PokemonGymAdapterError",
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
]
