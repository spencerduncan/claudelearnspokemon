"""
Compatibility layer for Pokemon Gym Factory.

Provides clean architecture components for dependency injection,
caching strategies, and production-ready patterns.

Author: Bot Dean - Production-First Engineering
"""

from .cache_strategies import (
    CacheError,
    CacheStrategy,
    InMemoryCache,
    NullCache,
    RedisCache,
    create_cache_strategy,
)

__all__ = [
    "CacheStrategy",
    "CacheError",
    "InMemoryCache",
    "NullCache",
    "RedisCache",
    "create_cache_strategy",
]
