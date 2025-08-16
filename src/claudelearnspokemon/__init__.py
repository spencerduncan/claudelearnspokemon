"""Claude Learns Pokemon - Speedrun learning agent system."""

__version__ = "0.1.0"

from .emulator_pool import EmulatorPool, EmulatorPoolError, ExecutionResult, PokemonGymClient
from .pokemon_gym_adapter import PokemonGymAdapter, PokemonGymAdapterError

__all__ = [
    "EmulatorPool",
    "EmulatorPoolError",
    "ExecutionResult",
    "PokemonGymClient",
    "PokemonGymAdapter",
    "PokemonGymAdapterError",
]
