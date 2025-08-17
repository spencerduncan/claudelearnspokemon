"""Claude Learns Pokemon - Speedrun learning agent system."""

__version__ = "0.1.0"

from .emulator_pool import EmulatorPool, EmulatorPoolError, ExecutionResult, PokemonGymClient

# Import compatibility layer components
try:
    from .pokemon_gym_adapter import PokemonGymAdapter
    from .pokemon_gym_factory import create_pokemon_client

    COMPATIBILITY_LAYER_AVAILABLE = True

    __all__ = [
        "EmulatorPool",
        "EmulatorPoolError",
        "ExecutionResult",
        "PokemonGymClient",
        "PokemonGymAdapter",
        "create_pokemon_client",
    ]
except ImportError:
    COMPATIBILITY_LAYER_AVAILABLE = False
    __all__ = ["EmulatorPool", "EmulatorPoolError", "ExecutionResult", "PokemonGymClient"]
