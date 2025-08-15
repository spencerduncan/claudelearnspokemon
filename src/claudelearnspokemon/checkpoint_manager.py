"""
CheckpointManager - Core save/load operations for Pokemon speedrun checkpoints.

Manages saved game states for deterministic replay and experimentation with:
- LZ4 compression for efficiency 
- UUID-based checkpoint identifiers
- Atomic write operations
- Performance target: <500ms for save/load operations
"""

import json
import lz4.frame
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import time
import structlog

logger = structlog.get_logger(__name__)


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""
    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint cannot be found."""
    pass


class CheckpointCorruptionError(CheckpointError):
    """Raised when a checkpoint is corrupted or invalid."""
    pass


class CheckpointManager:
    """
    Manages Pokemon game state checkpoints with LZ4 compression.
    
    Provides atomic save/load operations with UUID-based identifiers.
    Designed for high-frequency use in parallel execution environments.
    """
    
    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint storage. 
                          Defaults to ~/.claudelearnspokemon/checkpoints
        """
        if checkpoint_dir is None:
            self.checkpoint_dir = Path.home() / ".claudelearnspokemon" / "checkpoints"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self._save_times = []
        self._load_times = []
        
        logger.info("CheckpointManager initialized", checkpoint_dir=str(self.checkpoint_dir))

    def save_checkpoint(self, game_state: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """
        Save game state with metadata to compressed checkpoint file.
        
        Args:
            game_state: Complete game state dictionary
            metadata: Checkpoint metadata (location, progress, etc.)
            
        Returns:
            Checkpoint identifier (UUID string)
            
        Raises:
            CheckpointError: If save operation fails
        """
        start_time = time.monotonic()
        
        # Generate unique checkpoint ID
        checkpoint_id = str(uuid.uuid4())
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.lz4"
        
        try:
            # Prepare checkpoint data structure
            checkpoint_data = {
                "version": "1.0",
                "checkpoint_id": checkpoint_id,
                "timestamp": time.time(),
                "game_state": game_state,
                "metadata": metadata
            }
            
            # Serialize to JSON
            json_data = json.dumps(checkpoint_data, separators=(',', ':'))
            json_bytes = json_data.encode('utf-8')
            
            # Compress with LZ4 for speed
            compressed_data = lz4.frame.compress(json_bytes)
            
            # Atomic write: write to temp file, then rename
            temp_file = checkpoint_file.with_suffix('.tmp')
            try:
                with temp_file.open('wb') as f:
                    f.write(compressed_data)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure data reaches disk
                
                # Atomic rename
                temp_file.rename(checkpoint_file)
                
            except Exception as e:
                # Cleanup temp file on failure
                if temp_file.exists():
                    temp_file.unlink()
                raise
            
            # Track performance
            duration = time.monotonic() - start_time
            self._save_times.append(duration)
            
            logger.info(
                "Checkpoint saved",
                checkpoint_id=checkpoint_id,
                duration_ms=int(duration * 1000),
                compressed_size=len(compressed_data),
                original_size=len(json_bytes)
            )
            
            return checkpoint_id
            
        except Exception as e:
            logger.error(
                "Failed to save checkpoint", 
                error=str(e), 
                checkpoint_id=checkpoint_id,
                duration_ms=int((time.monotonic() - start_time) * 1000)
            )
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load game state from compressed checkpoint file.
        
        Args:
            checkpoint_id: Checkpoint identifier (UUID string)
            
        Returns:
            Game state dictionary
            
        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
            CheckpointCorruptionError: If checkpoint is corrupted
            CheckpointError: If load operation fails
        """
        start_time = time.monotonic()
        
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.lz4"
        
        if not checkpoint_file.exists():
            raise CheckpointNotFoundError(f"Checkpoint {checkpoint_id} not found")
        
        try:
            # Read compressed data
            with checkpoint_file.open('rb') as f:
                compressed_data = f.read()
            
            if not compressed_data:
                raise CheckpointCorruptionError(f"Checkpoint {checkpoint_id} is empty")
            
            # Decompress
            try:
                json_bytes = lz4.frame.decompress(compressed_data)
            except (RuntimeError, Exception) as e:
                # LZ4 raises RuntimeError for decompression failures
                if "LZ4F_" in str(e) or "decompress" in str(e).lower():
                    raise CheckpointCorruptionError(f"Failed to decompress checkpoint {checkpoint_id}") from e
                else:
                    raise
            
            # Parse JSON
            try:
                json_data = json_bytes.decode('utf-8')
                checkpoint_data = json.loads(json_data)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise CheckpointCorruptionError(f"Failed to parse checkpoint {checkpoint_id}") from e
            
            # Validate checkpoint structure
            required_fields = ["version", "checkpoint_id", "game_state", "metadata"]
            for field in required_fields:
                if field not in checkpoint_data:
                    raise CheckpointCorruptionError(f"Checkpoint {checkpoint_id} missing field: {field}")
            
            # Validate checkpoint ID matches
            if checkpoint_data["checkpoint_id"] != checkpoint_id:
                raise CheckpointCorruptionError(
                    f"Checkpoint ID mismatch: file {checkpoint_id}, content {checkpoint_data['checkpoint_id']}"
                )
            
            # Extract game state
            game_state = checkpoint_data["game_state"]
            
            # Track performance
            duration = time.monotonic() - start_time
            self._load_times.append(duration)
            
            logger.info(
                "Checkpoint loaded",
                checkpoint_id=checkpoint_id,
                duration_ms=int(duration * 1000),
                compressed_size=len(compressed_data),
                decompressed_size=len(json_bytes)
            )
            
            return game_state
            
        except (CheckpointNotFoundError, CheckpointCorruptionError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            logger.error(
                "Failed to load checkpoint",
                error=str(e),
                checkpoint_id=checkpoint_id,
                duration_ms=int((time.monotonic() - start_time) * 1000)
            )
            raise CheckpointError(f"Failed to load checkpoint {checkpoint_id}: {e}") from e

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for checkpoint operations.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            "save_operations": len(self._save_times),
            "load_operations": len(self._load_times),
        }
        
        if self._save_times:
            stats.update({
                "avg_save_time_ms": int(sum(self._save_times) * 1000 / len(self._save_times)),
                "max_save_time_ms": int(max(self._save_times) * 1000),
                "min_save_time_ms": int(min(self._save_times) * 1000)
            })
        
        if self._load_times:
            stats.update({
                "avg_load_time_ms": int(sum(self._load_times) * 1000 / len(self._load_times)),
                "max_load_time_ms": int(max(self._load_times) * 1000),
                "min_load_time_ms": int(min(self._load_times) * 1000)
            })
        
        return stats

    def checkpoint_exists(self, checkpoint_id: str) -> bool:
        """
        Check if a checkpoint exists.
        
        Args:
            checkpoint_id: Checkpoint identifier
            
        Returns:
            True if checkpoint exists
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.lz4"
        return checkpoint_file.exists()

    def get_checkpoint_size(self, checkpoint_id: str) -> int:
        """
        Get compressed size of checkpoint file.
        
        Args:
            checkpoint_id: Checkpoint identifier
            
        Returns:
            Size in bytes
            
        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.lz4"
        
        if not checkpoint_file.exists():
            raise CheckpointNotFoundError(f"Checkpoint {checkpoint_id} not found")
        
        return checkpoint_file.stat().st_size