"""
Conversation lifecycle management for Claude CLI processes.

This module handles conversation turn tracking, persistence, and lifecycle management,
following the Single Responsibility Principle by separating conversation concerns 
from process lifecycle and communication.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from .process_metrics_collector import ProcessMetrics
from .prompts import ProcessType

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurnMetrics(ProcessMetrics):
    """
    Extended metrics for conversation turn tracking.
    
    Extends ProcessMetrics with conversation-specific metrics
    following the established metrics pattern.
    """
    
    # Conversation turn tracking
    total_turns: int = 0
    opus_turns: int = 0
    sonnet_turns: int = 0
    
    # Turn limits and alerts
    opus_turn_limit: int = 100
    sonnet_turn_limit: int = 20
    alert_threshold_percent: float = 80.0
    
    # Turn timing metrics
    average_turn_processing_time: float = 0.0
    last_turn_timestamp: float = 0.0
    turns_per_minute: float = 0.0
    
    def get_opus_turns_remaining(self) -> int:
        """Get remaining Opus turns before limit."""
        return max(0, self.opus_turn_limit - self.opus_turns)
    
    def get_sonnet_turns_remaining(self) -> int:
        """Get remaining Sonnet turns before limit."""
        return max(0, self.sonnet_turn_limit - self.sonnet_turns)
    
    def get_opus_alert_threshold(self) -> int:
        """Get Opus turn count that triggers alert."""
        return int(self.opus_turn_limit * (self.alert_threshold_percent / 100))
    
    def get_sonnet_alert_threshold(self) -> int:
        """Get Sonnet turn count that triggers alert."""
        return int(self.sonnet_turn_limit * (self.alert_threshold_percent / 100))
    
    def should_alert_opus(self) -> bool:
        """Check if Opus turns have exceeded alert threshold."""
        return self.opus_turns >= self.get_opus_alert_threshold()
    
    def should_alert_sonnet(self) -> bool:
        """Check if Sonnet turns have exceeded alert threshold."""
        return self.sonnet_turns >= self.get_sonnet_alert_threshold()


@dataclass
class TurnLimitConfiguration:
    """
    Configuration for conversation turn limits.
    
    Follows the ProcessConfig pattern for consistent configuration management.
    """
    
    # Turn limits per process type
    opus_turn_limit: int = 100
    sonnet_turn_limit: int = 20
    
    # Alert configuration
    alert_threshold_percent: float = 80.0  # Trigger alert at 80% of limit
    alert_cooldown_seconds: float = 300.0  # 5 minute cooldown between alerts
    
    # Persistence configuration
    persistence_file: str = "conversation_turns.json"
    persistence_backup_count: int = 3
    auto_save_interval_seconds: float = 30.0
    
    # Performance tuning
    max_conversation_history: int = 1000  # Limit stored conversation IDs
    cleanup_interval_hours: float = 24.0  # Cleanup old conversations daily


class ConversationLifecycleManager:
    """
    Manages conversation turn tracking, limits, and lifecycle.
    
    This class is responsible for:
    - Thread-safe tracking of conversation turns by process type
    - Persistent storage of conversation state across restarts
    - Turn limit validation and alerting
    - Integration with existing health monitoring systems
    """
    
    def __init__(self, config: TurnLimitConfiguration = None, data_dir: str = "."):
        """
        Initialize conversation lifecycle manager.
        
        Args:
            config: Turn limit configuration (uses defaults if None)
            data_dir: Directory for persistent data storage
        """
        self.config = config or TurnLimitConfiguration()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Conversation turn tracking
        self.conversation_turns: Dict[str, ConversationTurnMetrics] = {}
        
        # Global metrics aggregation
        self.global_metrics = ConversationTurnMetrics(
            process_id=-1,  # Use -1 for global metrics
            opus_turn_limit=self.config.opus_turn_limit,
            sonnet_turn_limit=self.config.sonnet_turn_limit,
            alert_threshold_percent=self.config.alert_threshold_percent
        )
        
        # Alert tracking
        self._last_opus_alert = 0.0
        self._last_sonnet_alert = 0.0
        
        # Persistence setup
        self.persistence_file = self.data_dir / self.config.persistence_file
        self._load_persisted_data()
        
        # Auto-save scheduling
        self._auto_save_thread = None
        self._running = False
        self._start_auto_save()
        
        logger.info(f"ConversationLifecycleManager initialized with limits: Opus={self.config.opus_turn_limit}, Sonnet={self.config.sonnet_turn_limit}")
    
    def increment_turn_count(self, conversation_id: str, process_type: ProcessType) -> ConversationTurnMetrics:
        """
        Increment turn count for a conversation and process type.
        
        Args:
            conversation_id: Unique conversation identifier
            process_type: Type of Claude process (Opus or Sonnet)
            
        Returns:
            Updated ConversationTurnMetrics for the conversation
        """
        start_time = time.time()
        
        with self._lock:
            # Get or create conversation metrics
            if conversation_id not in self.conversation_turns:
                self.conversation_turns[conversation_id] = ConversationTurnMetrics(
                    process_id=hash(conversation_id) % 10000,  # Pseudo process ID
                    opus_turn_limit=self.config.opus_turn_limit,
                    sonnet_turn_limit=self.config.sonnet_turn_limit,
                    alert_threshold_percent=self.config.alert_threshold_percent
                )
            
            metrics = self.conversation_turns[conversation_id]
            
            # Increment appropriate counter
            if process_type == ProcessType.OPUS_STRATEGIC:
                metrics.opus_turns += 1
                self.global_metrics.opus_turns += 1
            elif process_type == ProcessType.SONNET_TACTICAL:
                metrics.sonnet_turns += 1
                self.global_metrics.sonnet_turns += 1
            
            # Update global counters
            metrics.total_turns += 1
            self.global_metrics.total_turns += 1
            
            # Update timing metrics
            current_time = time.time()
            processing_time = current_time - start_time
            
            # Update average processing time (simple moving average)
            if metrics.average_turn_processing_time == 0:
                metrics.average_turn_processing_time = processing_time
            else:
                metrics.average_turn_processing_time = (
                    metrics.average_turn_processing_time * 0.9 + processing_time * 0.1
                )
            
            metrics.last_turn_timestamp = current_time
            
            # Calculate turns per minute
            if metrics.total_turns > 1 and metrics.get_uptime() > 0:
                metrics.turns_per_minute = (metrics.total_turns / metrics.get_uptime()) * 60
            
            logger.debug(
                f"Turn incremented for conversation {conversation_id}: "
                f"{process_type.value} (total: {metrics.total_turns}, "
                f"opus: {metrics.opus_turns}, sonnet: {metrics.sonnet_turns})"
            )
            
            return metrics
    
    def check_turn_limits(self, conversation_id: str = None) -> Dict[str, Any]:
        """
        Check turn limits and generate alerts if necessary.
        
        Args:
            conversation_id: Specific conversation to check (checks global if None)
            
        Returns:
            Dictionary with limit check results and alert information
        """
        with self._lock:
            current_time = time.time()
            
            if conversation_id:
                # Check specific conversation
                if conversation_id not in self.conversation_turns:
                    return {"conversation_id": conversation_id, "status": "not_found"}
                
                metrics = self.conversation_turns[conversation_id]
            else:
                # Check global metrics
                metrics = self.global_metrics
            
            # Check Opus limits
            opus_alert_needed = (
                metrics.should_alert_opus() and 
                (current_time - self._last_opus_alert) > self.config.alert_cooldown_seconds
            )
            
            # Check Sonnet limits
            sonnet_alert_needed = (
                metrics.should_alert_sonnet() and
                (current_time - self._last_sonnet_alert) > self.config.alert_cooldown_seconds
            )
            
            # Update alert timestamps
            if opus_alert_needed:
                self._last_opus_alert = current_time
            if sonnet_alert_needed:
                self._last_sonnet_alert = current_time
            
            result = {
                "conversation_id": conversation_id or "global",
                "opus_turns": metrics.opus_turns,
                "sonnet_turns": metrics.sonnet_turns,
                "opus_limit": metrics.opus_turn_limit,
                "sonnet_limit": metrics.sonnet_turn_limit,
                "opus_remaining": metrics.get_opus_turns_remaining(),
                "sonnet_remaining": metrics.get_sonnet_turns_remaining(),
                "opus_alert_needed": opus_alert_needed,
                "sonnet_alert_needed": sonnet_alert_needed,
                "opus_at_limit": metrics.opus_turns >= metrics.opus_turn_limit,
                "sonnet_at_limit": metrics.sonnet_turns >= metrics.sonnet_turn_limit,
                "timestamp": current_time
            }
            
            if opus_alert_needed or sonnet_alert_needed:
                logger.warning(f"Turn limit alert for {conversation_id or 'global'}: {result}")
            
            return result
    
    def get_conversation_metrics(self, conversation_id: str = None) -> Dict[str, Any]:
        """
        Get conversation metrics for reporting.
        
        Args:
            conversation_id: Specific conversation (returns global if None)
            
        Returns:
            Dictionary with conversation metrics
        """
        with self._lock:
            if conversation_id:
                if conversation_id not in self.conversation_turns:
                    return {}
                metrics = self.conversation_turns[conversation_id]
            else:
                metrics = self.global_metrics
            
            return {
                "conversation_id": conversation_id or "global",
                "total_turns": metrics.total_turns,
                "opus_turns": metrics.opus_turns,
                "sonnet_turns": metrics.sonnet_turns,
                "opus_remaining": metrics.get_opus_turns_remaining(),
                "sonnet_remaining": metrics.get_sonnet_turns_remaining(),
                "average_processing_time_ms": round(metrics.average_turn_processing_time * 1000, 2),
                "turns_per_minute": round(metrics.turns_per_minute, 1),
                "last_turn_timestamp": metrics.last_turn_timestamp,
                "uptime_seconds": round(metrics.get_uptime(), 1)
            }
    
    def get_all_conversation_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all tracked conversations."""
        with self._lock:
            result = {"global": self.get_conversation_metrics()}
            
            for conv_id in self.conversation_turns.keys():
                result[conv_id] = self.get_conversation_metrics(conv_id)
            
            return result
    
    def cleanup_old_conversations(self, max_age_hours: float = 24.0) -> int:
        """
        Remove old conversation data to prevent memory leaks.
        
        Args:
            max_age_hours: Maximum age of conversations to keep
            
        Returns:
            Number of conversations cleaned up
        """
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - (max_age_hours * 3600)
            
            old_conversations = [
                conv_id for conv_id, metrics in self.conversation_turns.items()
                if metrics.last_turn_timestamp < cutoff_time
            ]
            
            for conv_id in old_conversations:
                del self.conversation_turns[conv_id]
            
            if old_conversations:
                logger.info(f"Cleaned up {len(old_conversations)} old conversations")
            
            return len(old_conversations)
    
    def _load_persisted_data(self):
        """Load conversation data from persistent storage."""
        try:
            if self.persistence_file.exists():
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
                
                # Restore global metrics
                if 'global_metrics' in data:
                    global_data = data['global_metrics']
                    self.global_metrics.total_turns = global_data.get('total_turns', 0)
                    self.global_metrics.opus_turns = global_data.get('opus_turns', 0)
                    self.global_metrics.sonnet_turns = global_data.get('sonnet_turns', 0)
                
                # Restore conversation data (simplified for performance)
                if 'conversations' in data and len(data['conversations']) <= self.config.max_conversation_history:
                    for conv_id, conv_data in data['conversations'].items():
                        metrics = ConversationTurnMetrics(
                            process_id=conv_data.get('process_id', hash(conv_id) % 10000),
                            opus_turn_limit=self.config.opus_turn_limit,
                            sonnet_turn_limit=self.config.sonnet_turn_limit,
                            alert_threshold_percent=self.config.alert_threshold_percent
                        )
                        metrics.total_turns = conv_data.get('total_turns', 0)
                        metrics.opus_turns = conv_data.get('opus_turns', 0)
                        metrics.sonnet_turns = conv_data.get('sonnet_turns', 0)
                        metrics.last_turn_timestamp = conv_data.get('last_turn_timestamp', time.time())
                        
                        self.conversation_turns[conv_id] = metrics
                
                logger.info(f"Loaded persisted conversation data: {self.global_metrics.total_turns} total turns")
        
        except Exception as e:
            logger.error(f"Failed to load persisted conversation data: {e}")
    
    def _save_persisted_data(self):
        """Save conversation data to persistent storage."""
        try:
            # Prepare data for serialization
            data = {
                'global_metrics': {
                    'total_turns': self.global_metrics.total_turns,
                    'opus_turns': self.global_metrics.opus_turns,
                    'sonnet_turns': self.global_metrics.sonnet_turns,
                    'timestamp': time.time()
                },
                'conversations': {}
            }
            
            # Save recent conversations only (performance optimization)
            recent_conversations = sorted(
                self.conversation_turns.items(),
                key=lambda x: x[1].last_turn_timestamp,
                reverse=True
            )[:self.config.max_conversation_history]
            
            for conv_id, metrics in recent_conversations:
                data['conversations'][conv_id] = {
                    'process_id': metrics.process_id,
                    'total_turns': metrics.total_turns,
                    'opus_turns': metrics.opus_turns,
                    'sonnet_turns': metrics.sonnet_turns,
                    'last_turn_timestamp': metrics.last_turn_timestamp
                }
            
            # Atomic write with backup
            temp_file = self.persistence_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Create backup if original exists
            if self.persistence_file.exists():
                backup_file = self.persistence_file.with_suffix('.bak')
                backup_file.write_bytes(self.persistence_file.read_bytes())
            
            # Atomic replace
            temp_file.replace(self.persistence_file)
            
            logger.debug("Conversation data persisted successfully")
        
        except Exception as e:
            logger.error(f"Failed to persist conversation data: {e}")
    
    def _start_auto_save(self):
        """Start background auto-save thread."""
        if self._auto_save_thread and self._auto_save_thread.is_alive():
            return
        
        self._running = True
        self._auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self._auto_save_thread.start()
        
        logger.debug("Auto-save thread started")
    
    def _auto_save_loop(self):
        """Background loop for periodic data persistence."""
        while self._running:
            try:
                time.sleep(self.config.auto_save_interval_seconds)
                if self._running:  # Check again after sleep
                    self._save_persisted_data()
            except Exception as e:
                logger.error(f"Error in auto-save loop: {e}")
                time.sleep(5.0)  # Brief pause before retrying
    
    def shutdown(self):
        """Gracefully shutdown the conversation lifecycle manager."""
        logger.info("Shutting down ConversationLifecycleManager...")
        
        self._running = False
        
        # Wait for auto-save thread to finish
        if self._auto_save_thread and self._auto_save_thread.is_alive():
            self._auto_save_thread.join(timeout=2.0)
        
        # Final save
        self._save_persisted_data()
        
        logger.info("ConversationLifecycleManager shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()