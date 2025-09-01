"""
Abstract logging manager interface for Pokemon game logging operations.

Provides a standardized interface for logging Pokemon-specific actions, game states,
performance metrics, and debug information with proper context injection.

This follows the Abstract Interface First pattern from Issue #241 to ensure
consistent logging behavior across all components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration for consistent level management."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class PokemonContext:
    """Standard context structure for Pokemon-specific logging."""
    
    def __init__(
        self,
        game_state: Optional[str] = None,
        location: Optional[str] = None,
        action: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        player_position: Optional[Dict[str, int]] = None,
        pokemon_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        """
        Initialize Pokemon-specific logging context.
        
        Args:
            game_state: Current state of the Pokemon game
            location: Current location in the game world
            action: The action being performed
            timestamp: When the action occurred
            player_position: Player coordinates {x, y}
            pokemon_data: Data about encountered/caught Pokemon
            session_id: Unique session identifier
            request_id: Request tracking identifier
        """
        self.game_state = game_state
        self.location = location
        self.action = action
        self.timestamp = timestamp or datetime.now()
        self.player_position = player_position
        self.pokemon_data = pokemon_data
        self.session_id = session_id
        self.request_id = request_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for structured logging."""
        return {
            key: value for key, value in {
                "game_state": self.game_state,
                "location": self.location, 
                "action": self.action,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                "player_position": self.player_position,
                "pokemon_data": self.pokemon_data,
                "session_id": self.session_id,
                "request_id": self.request_id,
            }.items() if value is not None
        }


class LoggingManager(ABC):
    """
    Abstract base class for Pokemon game logging operations.
    
    Defines the standard interface for all logging implementations,
    ensuring consistent behavior across production and test environments.
    
    This interface supports:
    - Pokemon-specific action logging
    - Game state transitions
    - Performance metrics collection
    - Debug context injection
    - Async logging for high-throughput scenarios
    """
    
    @abstractmethod
    def log_pokemon_action(
        self,
        level: LogLevel,
        message: str,
        context: PokemonContext,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a Pokemon-specific game action.
        
        Args:
            level: Log level for the message
            message: Descriptive message about the action
            context: Pokemon-specific context information
            extra_data: Additional structured data to include
        """
        pass
    
    @abstractmethod
    def log_game_state(
        self,
        level: LogLevel,
        message: str,
        state_data: Dict[str, Any],
        context: Optional[PokemonContext] = None,
    ) -> None:
        """
        Log game state transitions and updates.
        
        Args:
            level: Log level for the message
            message: Description of the state change
            state_data: Structured game state information
            context: Optional Pokemon context
        """
        pass
    
    @abstractmethod
    def log_performance_metric(
        self,
        metric_name: str,
        metric_value: Union[int, float],
        unit: str,
        context: Optional[PokemonContext] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Log performance metrics for monitoring.
        
        Args:
            metric_name: Name of the performance metric
            metric_value: Measured value
            unit: Unit of measurement (ms, seconds, count, etc.)
            context: Optional Pokemon context
            tags: Optional tags for metric categorization
        """
        pass
    
    @abstractmethod
    def log_debug_context(
        self,
        message: str,
        debug_data: Dict[str, Any],
        context: Optional[PokemonContext] = None,
    ) -> None:
        """
        Log detailed debug information.
        
        Args:
            message: Debug message
            debug_data: Structured debug information
            context: Optional Pokemon context
        """
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """
        Force flush any buffered log messages.
        
        This is particularly important for async logging implementations
        to ensure all messages are written before shutdown.
        """
        pass
    
    @abstractmethod
    def set_log_level(self, level: LogLevel) -> None:
        """
        Set the minimum log level for this manager.
        
        Args:
            level: Minimum level to log
        """
        pass
    
    @abstractmethod
    def is_enabled_for(self, level: LogLevel) -> bool:
        """
        Check if logging is enabled for the specified level.
        
        Args:
            level: Log level to check
            
        Returns:
            True if logging is enabled for this level
        """
        pass