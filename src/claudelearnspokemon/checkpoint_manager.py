"""
CheckpointManager - Manages saved game states with metadata management.

Provides checkpoint save/load functionality with comprehensive metadata storage,
validation, and fast querying capabilities for deterministic replay and experimentation.
Uses LZ4 compression for optimal performance.
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lz4.frame

logger = logging.getLogger(__name__)


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""

    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint cannot be found."""

    pass


class CheckpointCorruptionError(CheckpointError):
    """Raised when a checkpoint is corrupted or invalid."""

    pass


@dataclass
class CheckpointMetadata:
    """Structured metadata for checkpoints."""

    checkpoint_id: str
    created_at: str  # ISO format timestamp
    game_location: str
    progress_markers: list[str]
    performance_metrics: dict[str, float]
    tags: list[str]
    custom_fields: dict[str, Any]
    file_size: int = 0
    checksum: str = ""
    schema_version: int = 1


class CheckpointManager:
    """
    Manages saved game states with metadata management and LZ4 compression.

    Provides compression, metadata management, indexing, and fast querying
    for efficient checkpoint operations across parallel executions.
    """

    SCHEMA_VERSION = 1
    METADATA_CACHE_SIZE = 1000

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize CheckpointManager with directory and metadata database.
        Args:
            checkpoint_dir: Directory to store checkpoint files and database
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Database for metadata indexing
        self.db_path = self.checkpoint_dir / "metadata.db"
        self._metadata_cache: dict[str, dict[str, Any]] = {}
        self._cache_access_times: dict[str, float] = {}

        # Performance tracking
        self._save_times: list[float] = []
        self._load_times: list[float] = []

        self._init_database()

        # Verify database was created successfully
        if not self.db_path.exists():
            raise RuntimeError(f"Failed to create database file at {self.db_path}")

        logger.info(f"CheckpointManager initialized with directory: {self.checkpoint_dir}")

    def _init_database(self) -> None:
        """Initialize SQLite database for metadata indexing."""
        try:
            logger.info(f"Creating database at: {self.db_path}")
            with sqlite3.connect(self.db_path) as conn:
                logger.info("Database connection established")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoint_metadata (
                        checkpoint_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        game_location TEXT,
                        progress_markers TEXT,
                        performance_metrics TEXT,
                        tags TEXT,
                        custom_fields TEXT,
                        file_size INTEGER,
                        checksum TEXT,
                        schema_version INTEGER DEFAULT 1
                    )
                    """
                )
                logger.info("Main table created")

                # Create indexes for fast querying
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_created_at ON checkpoint_metadata(created_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_game_location ON checkpoint_metadata(game_location)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_checksum ON checkpoint_metadata(checksum)"
                )
                logger.info("Indexes created")

                # Explicit commit to ensure database file is created
                conn.commit()
                logger.info("Database committed")

            logger.info(f"Database file exists after init: {self.db_path.exists()}")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def save_checkpoint(self, game_state: dict[str, Any], metadata: dict[str, Any]) -> str:
        """
        Save game state with metadata to LZ4 compressed checkpoint file.

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
                "metadata": metadata,
            }

            # Serialize to JSON
            json_data = json.dumps(checkpoint_data, separators=(",", ":"))
            json_bytes = json_data.encode("utf-8")

            # Compress with LZ4 for speed
            compressed_data = lz4.frame.compress(json_bytes)

            # Atomic write: write to temp file, then rename
            temp_file = checkpoint_file.with_suffix(".tmp")
            try:
                with temp_file.open("wb") as f:
                    f.write(compressed_data)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure data reaches disk

                # Atomic rename
                temp_file.rename(checkpoint_file)

            except Exception:
                # Cleanup temp file on failure
                if temp_file.exists():
                    temp_file.unlink()
                raise

            # Create structured metadata
            structured_metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                game_location=metadata.get("game_location", ""),
                progress_markers=metadata.get("progress_markers", []),
                performance_metrics=metadata.get("performance_metrics", {}),
                tags=metadata.get("tags", []),
                custom_fields=metadata.get("custom_fields", {}),
                file_size=len(compressed_data),
                checksum=self._calculate_checksum_from_data(compressed_data),
                schema_version=self.SCHEMA_VERSION,
            )

            # Store metadata
            self._store_metadata(structured_metadata)

            # Track performance
            duration = time.monotonic() - start_time
            self._save_times.append(duration)

            logger.info(
                f"Checkpoint saved: {checkpoint_id} at {structured_metadata.game_location}, "
                f"duration: {int(duration * 1000)}ms, compressed: {len(compressed_data)} bytes, "
                f"original: {len(json_bytes)} bytes"
            )

            return checkpoint_id

        except Exception as e:
            logger.error(
                f"Failed to save checkpoint {checkpoint_id}: {e}, "
                f"duration: {int((time.monotonic() - start_time) * 1000)}ms"
            )
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    def load_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """
        Load game state from LZ4 compressed checkpoint file.

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
            with checkpoint_file.open("rb") as f:
                compressed_data = f.read()

            if not compressed_data:
                raise CheckpointCorruptionError(f"Checkpoint {checkpoint_id} is empty")

            # Decompress
            try:
                json_bytes = lz4.frame.decompress(compressed_data)
            except (RuntimeError, Exception) as e:
                # LZ4 raises RuntimeError for decompression failures
                if "LZ4F_" in str(e) or "decompress" in str(e).lower():
                    raise CheckpointCorruptionError(
                        f"Failed to decompress checkpoint {checkpoint_id}"
                    ) from e
                else:
                    raise

            # Parse JSON
            try:
                json_data = json_bytes.decode("utf-8")
                checkpoint_data = json.loads(json_data)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise CheckpointCorruptionError(
                    f"Failed to parse checkpoint {checkpoint_id}"
                ) from e

            # Validate checkpoint structure
            required_fields = ["version", "checkpoint_id", "game_state", "metadata"]
            for field in required_fields:
                if field not in checkpoint_data:
                    raise CheckpointCorruptionError(
                        f"Checkpoint {checkpoint_id} missing field: {field}"
                    )

            # Validate checkpoint ID matches
            if checkpoint_data["checkpoint_id"] != checkpoint_id:
                raise CheckpointCorruptionError(
                    f"Checkpoint ID mismatch: file {checkpoint_id}, content {checkpoint_data['checkpoint_id']}"
                )

            # Extract game state
            game_state: dict[str, Any] = checkpoint_data["game_state"]

            # Track performance
            duration = time.monotonic() - start_time
            self._load_times.append(duration)

            logger.info(
                f"Checkpoint loaded: {checkpoint_id}, duration: {int(duration * 1000)}ms, "
                f"compressed: {len(compressed_data)} bytes, decompressed: {len(json_bytes)} bytes"
            )

            return game_state

        except (CheckpointNotFoundError, CheckpointCorruptionError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            logger.error(
                f"Failed to load checkpoint {checkpoint_id}: {e}, "
                f"duration: {int((time.monotonic() - start_time) * 1000)}ms"
            )
            raise CheckpointError(f"Failed to load checkpoint {checkpoint_id}: {e}") from e

    def get_checkpoint_metadata(self, checkpoint_id: str) -> dict[str, Any] | None:
        """
        Get metadata for a checkpoint with caching support.

        Args:
            checkpoint_id: Unique identifier of checkpoint

        Returns:
            Metadata dictionary or None if checkpoint not found
        """
        # Check cache first
        current_time = time.time()
        if checkpoint_id in self._metadata_cache:
            self._cache_access_times[checkpoint_id] = current_time
            return self._metadata_cache[checkpoint_id].copy()

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM checkpoint_metadata WHERE checkpoint_id = ?", (checkpoint_id,)
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                # Convert to dict and parse JSON fields
                metadata = dict(row)
                metadata["progress_markers"] = json.loads(metadata["progress_markers"] or "[]")
                metadata["performance_metrics"] = json.loads(
                    metadata["performance_metrics"] or "{}"
                )
                metadata["tags"] = json.loads(metadata["tags"] or "[]")
                metadata["custom_fields"] = json.loads(metadata["custom_fields"] or "{}")

                # Update cache
                self._update_cache(checkpoint_id, metadata)

                return metadata

        except Exception as e:
            logger.error(f"Failed to get metadata for {checkpoint_id}: {e}")
            return None

    def update_checkpoint_metadata(self, checkpoint_id: str, updates: dict[str, Any]) -> bool:
        """
        Update metadata for an existing checkpoint.

        Args:
            checkpoint_id: Unique identifier of checkpoint
            updates: Dictionary of fields to update

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Get current metadata first
            current_metadata = self.get_checkpoint_metadata(checkpoint_id)
            if current_metadata is None:
                logger.error(f"Cannot update metadata: checkpoint {checkpoint_id} not found")
                return False

            # Apply updates
            current_metadata.update(updates)

            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE checkpoint_metadata
                    SET game_location = ?, progress_markers = ?, performance_metrics = ?,
                        tags = ?, custom_fields = ?
                    WHERE checkpoint_id = ?
                    """,
                    (
                        current_metadata.get("game_location"),
                        json.dumps(current_metadata.get("progress_markers", [])),
                        json.dumps(current_metadata.get("performance_metrics", {})),
                        json.dumps(current_metadata.get("tags", [])),
                        json.dumps(current_metadata.get("custom_fields", {})),
                        checkpoint_id,
                    ),
                )
                conn.commit()

            # Update cache
            if checkpoint_id in self._metadata_cache:
                self._metadata_cache[checkpoint_id] = current_metadata

            logger.info(f"Updated metadata for checkpoint {checkpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update metadata for {checkpoint_id}: {e}")
            return False

    def search_checkpoints(self, criteria: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Search checkpoints based on metadata criteria.

        Args:
            criteria: Search criteria dictionary

        Returns:
            List of matching checkpoint metadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                query = "SELECT * FROM checkpoint_metadata WHERE 1=1"
                params = []

                # Build dynamic query based on criteria
                if "game_location" in criteria:
                    query += " AND game_location LIKE ?"
                    params.append(f"%{criteria['game_location']}%")

                if "tags" in criteria and criteria["tags"]:
                    for tag in criteria["tags"]:
                        query += " AND tags LIKE ?"
                        params.append(f'%"{tag}"%')

                if "created_after" in criteria:
                    query += " AND created_at > ?"
                    params.append(criteria["created_after"])

                if "created_before" in criteria:
                    query += " AND created_at < ?"
                    params.append(criteria["created_before"])

                query += " ORDER BY created_at DESC"

                # Apply limit if specified
                if "limit" in criteria:
                    query += " LIMIT ?"
                    params.append(criteria["limit"])

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    metadata = dict(row)
                    # Parse JSON fields
                    metadata["progress_markers"] = json.loads(metadata["progress_markers"] or "[]")
                    metadata["performance_metrics"] = json.loads(
                        metadata["performance_metrics"] or "{}"
                    )
                    metadata["tags"] = json.loads(metadata["tags"] or "[]")
                    metadata["custom_fields"] = json.loads(metadata["custom_fields"] or "{}")
                    results.append(metadata)

                return results

        except Exception as e:
            logger.error(f"Failed to search checkpoints: {e}")
            return []

    def validate_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Validate checkpoint integrity using checksum.

        Args:
            checkpoint_id: Unique identifier of checkpoint

        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.lz4"
            if not checkpoint_file.exists():
                return False

            # Get stored checksum
            metadata = self.get_checkpoint_metadata(checkpoint_id)
            if not metadata or "checksum" not in metadata:
                return False

            stored_checksum = metadata["checksum"]

            # Calculate current checksum
            current_checksum = self._calculate_checksum(checkpoint_file)

            return stored_checksum == current_checksum

        except Exception as e:
            logger.error(f"Failed to validate checkpoint {checkpoint_id}: {e}")
            return False

    def list_checkpoints(self, criteria: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        List all checkpoints with optional filtering.

        Args:
            criteria: Optional filtering criteria

        Returns:
            List of checkpoint metadata
        """
        if criteria is None:
            criteria = {}

        return self.search_checkpoints(criteria)

    def find_nearest_checkpoint(self, location: str) -> str | None:
        """
        Find the most recent checkpoint near a given location.

        Args:
            location: Game location to search near

        Returns:
            Checkpoint ID of nearest checkpoint, or None if none found
        """
        results = self.search_checkpoints({"game_location": location, "limit": 1})

        return results[0]["checkpoint_id"] if results else None

    def get_performance_stats(self) -> dict[str, Any]:
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
            stats.update(
                {
                    "avg_save_time_ms": int(sum(self._save_times) * 1000 / len(self._save_times)),
                    "max_save_time_ms": int(max(self._save_times) * 1000),
                    "min_save_time_ms": int(min(self._save_times) * 1000),
                }
            )

        if self._load_times:
            stats.update(
                {
                    "avg_load_time_ms": int(sum(self._load_times) * 1000 / len(self._load_times)),
                    "max_load_time_ms": int(max(self._load_times) * 1000),
                    "min_load_time_ms": int(min(self._load_times) * 1000),
                }
            )

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

    def _store_metadata(self, metadata: CheckpointMetadata) -> None:
        """Store structured metadata in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO checkpoint_metadata
                    (checkpoint_id, created_at, game_location, progress_markers,
                     performance_metrics, tags, custom_fields, file_size, checksum, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metadata.checkpoint_id,
                        metadata.created_at,
                        metadata.game_location,
                        json.dumps(metadata.progress_markers),
                        json.dumps(metadata.performance_metrics),
                        json.dumps(metadata.tags),
                        json.dumps(metadata.custom_fields),
                        metadata.file_size,
                        metadata.checksum,
                        metadata.schema_version,
                    ),
                )
                conn.commit()

            # Update cache
            metadata_dict = asdict(metadata)
            self._update_cache(metadata.checkpoint_id, metadata_dict)

        except Exception as e:
            logger.error(f"Failed to store metadata for {metadata.checkpoint_id}: {e}")
            raise

    def _update_cache(self, checkpoint_id: str, metadata: dict[str, Any]) -> None:
        """Update metadata cache with LRU eviction."""
        current_time = time.time()

        # Add to cache
        self._metadata_cache[checkpoint_id] = metadata
        self._cache_access_times[checkpoint_id] = current_time

        # Evict if cache too large
        if len(self._metadata_cache) > self.METADATA_CACHE_SIZE:
            # Find least recently used item
            oldest_id = min(
                self._cache_access_times.keys(), key=lambda k: self._cache_access_times[k]
            )
            del self._metadata_cache[oldest_id]
            del self._cache_access_times[oldest_id]

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _calculate_checksum_from_data(self, data: bytes) -> str:
        """Calculate SHA-256 checksum from data."""
        hasher = hashlib.sha256()
        hasher.update(data)
        return hasher.hexdigest()
