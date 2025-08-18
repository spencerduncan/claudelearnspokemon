"""
Compatibility layer for Pokemon Gym Factory.

Provides clean architecture components including:
- Parameter validation functions (Single Responsibility Principle)
- Caching strategies for production-grade performance
- Dependency injection patterns for testable code

Combines Uncle Bot's validation excellence with Bot Dean's production engineering.
"""

from .cache_strategies import (
    CacheError,
    CacheStrategy,
    InMemoryCache,
    NullCache,
    RedisCache,
    create_cache_strategy,
)
from .validation import (
    ValidationError,
    validate_adapter_type_selection,
    validate_client_creation_parameters,
    validate_container_identifier,
    validate_input_delay_parameter,
    validate_server_url_port,
    validate_timeout_parameter,
)

__all__ = [
    # Cache Strategy Components (Bot Dean's Production Engineering)
    "CacheStrategy",
    "CacheError",
    "InMemoryCache",
    "NullCache",
    "RedisCache",
    "create_cache_strategy",
    # Validation Components (Uncle Bot's Clean Code Craftsmanship)
    "ValidationError",
    "validate_server_url_port",
    "validate_container_identifier",
    "validate_adapter_type_selection",
    "validate_timeout_parameter",
    "validate_input_delay_parameter",
    "validate_client_creation_parameters",
]
