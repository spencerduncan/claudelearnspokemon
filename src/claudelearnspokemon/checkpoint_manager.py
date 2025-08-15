"""
CheckpointManager - Manages saved game states with metadata management.

Provides checkpoint save/load functionality with comprehensive metadata storage,
validation, and fast querying capabilities for deterministic replay and experimentation.
"""

import gzip
import hashlib
import json
import logging
import pickle
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    Manages saved game states for deterministic replay and experimentation.

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

        self._init_database()
        logger.info(f"CheckpointManager initialized with directory: {checkpoint_dir}")

    def _init_database(self) -> None:
        """Initialize SQLite database for metadata indexing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_metadata (
                    checkpoint_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    game_location TEXT,
                    progress_markers TEXT,  -- JSON array
                    performance_metrics TEXT,  -- JSON object
                    tags TEXT,  -- JSON array
                    custom_fields TEXT,  -- JSON object
                    file_size INTEGER,
                    checksum TEXT,
                    schema_version INTEGER DEFAULT 1
                )
            """
            )

            # Create indexes for fast queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON checkpoint_metadata(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_game_location ON checkpoint_metadata(game_location)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_checksum ON checkpoint_metadata(checksum)")

            conn.commit()

    def save_checkpoint(self, game_state: dict[str, Any], metadata: dict[str, Any]) -> str:
        """
        Save game state with metadata to compressed file.

        Args:
            game_state: Game state dictionary to save
            metadata: Metadata dictionary with checkpoint information

        Returns:
            checkpoint_id: Unique identifier for the saved checkpoint

        Raises:
            ValueError: If metadata is invalid or required fields are missing
        """
        checkpoint_id = str(uuid.uuid4())

        # Validate and structure metadata
        structured_metadata = self._validate_and_structure_metadata(metadata, checkpoint_id)

        # Save compressed checkpoint data
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl.gz"
        with gzip.open(checkpoint_file, "wb") as f:
            pickle.dump(game_state, f)

        # Calculate file size and checksum
        structured_metadata.file_size = checkpoint_file.stat().st_size
        structured_metadata.checksum = self._calculate_checksum(checkpoint_file)

        # Store metadata
        self._store_metadata(structured_metadata)

        logger.info(f"Checkpoint saved: {checkpoint_id} at {structured_metadata.game_location}")
        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """
        Load game state from compressed checkpoint file.

        Args:
            checkpoint_id: Unique identifier of checkpoint to load

        Returns:
            game_state: Loaded game state dictionary

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint is corrupted or invalid
        """
        if not self._checkpoint_exists(checkpoint_id):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl.gz"

        # Verify integrity if checksum exists
        metadata = self.get_checkpoint_metadata(checkpoint_id)
        if metadata and metadata.get("checksum"):
            current_checksum = self._calculate_checksum(checkpoint_file)
            if current_checksum != metadata["checksum"]:
                raise ValueError(f"Checkpoint corrupted: checksum mismatch for {checkpoint_id}")

        try:
            with gzip.open(checkpoint_file, "rb") as f:
                game_state = pickle.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return game_state
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint {checkpoint_id}: {str(e)}") from e

    def get_checkpoint_metadata(self, checkpoint_id: str) -> dict[str, Any] | None:
        """
        Get metadata for a specific checkpoint.

        Args:
            checkpoint_id: Unique identifier of checkpoint

        Returns:
            metadata: Checkpoint metadata dictionary or None if not found
        """
        # Check cache first
        if checkpoint_id in self._metadata_cache:
            self._cache_access_times[checkpoint_id] = time.time()
            return self._metadata_cache[checkpoint_id]

        # Query database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM checkpoint_metadata WHERE checkpoint_id = ?", (checkpoint_id,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        metadata = self._row_to_metadata_dict(row)

        # Update cache
        self._update_metadata_cache(checkpoint_id, metadata)

        return metadata

    def update_checkpoint_metadata(
        self, checkpoint_id: str, metadata_updates: dict[str, Any]
    ) -> bool:
        """
        Update metadata for an existing checkpoint.

        Args:
            checkpoint_id: Unique identifier of checkpoint
            metadata_updates: Dictionary of metadata fields to update

        Returns:
            success: True if update was successful, False otherwise
        """
        if not self._checkpoint_exists(checkpoint_id):
            logger.warning(f"Cannot update metadata: checkpoint not found: {checkpoint_id}")
            return False

        current_metadata = self.get_checkpoint_metadata(checkpoint_id)
        if not current_metadata:
            logger.warning(f"Cannot update metadata: no existing metadata for: {checkpoint_id}")
            return False

        # Merge updates with current metadata
        updated_metadata = current_metadata.copy()

        # Handle special fields that need JSON serialization
        json_fields = ["progress_markers", "performance_metrics", "tags", "custom_fields"]

        for field, value in metadata_updates.items():
            if field == "checkpoint_id":
                continue  # Don't allow changing ID

            updated_metadata[field] = value

        # Validate updated metadata
        try:
            self._validate_metadata_dict(updated_metadata)
        except ValueError as e:
            logger.error(f"Invalid metadata update for {checkpoint_id}: {e}")
            return False

        # Update database
        with sqlite3.connect(self.db_path) as conn:
            # Prepare update statement
            set_clause = []
            values = []

            for field in ["created_at", "game_location", "file_size", "checksum", "schema_version"]:
                if field in metadata_updates:
                    set_clause.append(f"{field} = ?")
                    values.append(updated_metadata[field])

            for field in json_fields:
                if field in metadata_updates:
                    set_clause.append(f"{field} = ?")
                    values.append(json.dumps(updated_metadata[field]))

            if set_clause:
                values.append(checkpoint_id)
                query = f"UPDATE checkpoint_metadata SET {', '.join(set_clause)} WHERE checkpoint_id = ?"
                conn.execute(query, values)
                conn.commit()

        # Update cache
        if checkpoint_id in self._metadata_cache:
            self._metadata_cache[checkpoint_id] = updated_metadata

        logger.info(f"Metadata updated for checkpoint: {checkpoint_id}")
        return True

    def search_checkpoints(self, criteria: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Search checkpoints by metadata criteria.

        Args:
            criteria: Search criteria dictionary with field filters
                     Supports: game_location, tags, created_after, created_before,
                     performance_min, performance_max, custom fields

        Returns:
            matching_checkpoints: List of matching checkpoint metadata
        """
        query_parts = []
        values = []

        # Build WHERE clause from criteria
        if "game_location" in criteria:
            query_parts.append("game_location LIKE ?")
            values.append(f"%{criteria['game_location']}%")

        if "created_after" in criteria:
            query_parts.append("created_at >= ?")
            values.append(criteria["created_after"])

        if "created_before" in criteria:
            query_parts.append("created_at <= ?")
            values.append(criteria["created_before"])

        if "tags" in criteria:
            # Tags are stored as JSON, need to check if any tag matches
            for tag in criteria["tags"]:
                query_parts.append("tags LIKE ?")
                values.append(f'%"{tag}"%')

        # Build full query
        base_query = "SELECT * FROM checkpoint_metadata"
        if query_parts:
            where_clause = " AND ".join(query_parts)
            query = f"{base_query} WHERE {where_clause}"
        else:
            query = base_query

        query += " ORDER BY created_at DESC"

        # Execute query
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, values)
            rows = cursor.fetchall()

        results = [self._row_to_metadata_dict(row) for row in rows]

        # Apply additional filtering for complex criteria
        if "performance_min" in criteria or "performance_max" in criteria:
            results = self._filter_by_performance(results, criteria)

        if "custom_fields" in criteria:
            results = self._filter_by_custom_fields(results, criteria["custom_fields"])

        logger.info(f"Found {len(results)} checkpoints matching search criteria")
        return results

    def list_checkpoints(self, criteria: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        List checkpoints matching optional criteria.

        Args:
            criteria: Optional search criteria (same as search_checkpoints)

        Returns:
            checkpoints: List of checkpoint metadata dictionaries
        """
        if criteria:
            return self.search_checkpoints(criteria)
        else:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM checkpoint_metadata ORDER BY created_at DESC")
                rows = cursor.fetchall()

            return [self._row_to_metadata_dict(row) for row in rows]

    def validate_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Validate checkpoint integrity and metadata consistency.

        Args:
            checkpoint_id: Unique identifier of checkpoint to validate

        Returns:
            is_valid: True if checkpoint is valid, False otherwise
        """
        try:
            # Check if files exist
            if not self._checkpoint_exists(checkpoint_id):
                logger.warning(f"Checkpoint file missing: {checkpoint_id}")
                return False

            # Check metadata exists
            metadata = self.get_checkpoint_metadata(checkpoint_id)
            if not metadata:
                logger.warning(f"Checkpoint metadata missing: {checkpoint_id}")
                return False

            # Validate file integrity
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl.gz"
            if metadata.get("checksum"):
                current_checksum = self._calculate_checksum(checkpoint_file)
                if current_checksum != metadata["checksum"]:
                    logger.warning(f"Checkpoint checksum mismatch: {checkpoint_id}")
                    return False

            # Try loading the checkpoint
            try:
                self.load_checkpoint(checkpoint_id)
            except Exception as e:
                logger.warning(f"Checkpoint load failed: {checkpoint_id}: {e}")
                return False

            return True

        except Exception as e:
            logger.error(f"Checkpoint validation error for {checkpoint_id}: {e}")
            return False

    def prune_checkpoints(self, max_count: int) -> int:
        """
        Remove low-value checkpoints to manage storage.

        Args:
            max_count: Maximum number of checkpoints to keep

        Returns:
            pruned_count: Number of checkpoints removed
        """
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= max_count:
            return 0

        # Sort by value score (simple heuristic: newer + better performance)
        def value_score(checkpoint):
            created_at = datetime.fromisoformat(checkpoint["created_at"])
            age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600

            performance = checkpoint.get("performance_metrics", {})
            success_rate = performance.get("success_rate", 0.5)
            execution_time = performance.get("execution_time", float("inf"))

            # Higher score = higher value (newer, faster, more successful)
            score = success_rate * 100 - age_hours * 0.1
            if execution_time < float("inf"):
                score += max(0, 100 - execution_time)

            return score

        checkpoints.sort(key=value_score, reverse=True)

        # Remove lowest value checkpoints
        to_remove = checkpoints[max_count:]
        pruned_count = 0

        for checkpoint in to_remove:
            checkpoint_id = checkpoint["checkpoint_id"]
            if self._remove_checkpoint(checkpoint_id):
                pruned_count += 1

        logger.info(f"Pruned {pruned_count} checkpoints, keeping {max_count}")
        return pruned_count

    def find_nearest_checkpoint(self, location: str) -> str | None:
        """
        Find checkpoint closest to specified location.

        Args:
            location: Game location to find nearest checkpoint for

        Returns:
            checkpoint_id: ID of nearest checkpoint or None if none found
        """
        checkpoints = self.search_checkpoints({"game_location": location})

        if not checkpoints:
            # Try broader search
            checkpoints = self.list_checkpoints()

        if not checkpoints:
            return None

        # Simple distance heuristic - exact match first, then lexicographic similarity
        def location_distance(checkpoint):
            checkpoint_location = checkpoint.get("game_location", "")
            if checkpoint_location == location:
                return 0

            # Simple string similarity
            return len(set(location) ^ set(checkpoint_location))

        nearest = min(checkpoints, key=location_distance)
        return nearest["checkpoint_id"]

    # Private helper methods

    def _validate_and_structure_metadata(
        self, metadata: dict[str, Any], checkpoint_id: str
    ) -> CheckpointMetadata:
        """Validate metadata and convert to structured format."""
        # Set defaults
        now = datetime.now(timezone.utc).isoformat()

        structured = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            created_at=metadata.get("created_at", now),
            game_location=metadata.get("game_location", ""),
            progress_markers=metadata.get("progress_markers", []),
            performance_metrics=metadata.get("performance_metrics", {}),
            tags=metadata.get("tags", []),
            custom_fields=metadata.get("custom_fields", {}),
            schema_version=self.SCHEMA_VERSION,
        )

        # Validate
        self._validate_metadata_structure(structured)

        return structured

    def _validate_metadata_structure(self, metadata: CheckpointMetadata) -> None:
        """Validate structured metadata."""
        if not metadata.checkpoint_id:
            raise ValueError("checkpoint_id is required")

        if not isinstance(metadata.progress_markers, list):
            raise ValueError("progress_markers must be a list")

        if not isinstance(metadata.performance_metrics, dict):
            raise ValueError("performance_metrics must be a dict")

        if not isinstance(metadata.tags, list):
            raise ValueError("tags must be a list")

        if not isinstance(metadata.custom_fields, dict):
            raise ValueError("custom_fields must be a dict")

        # Validate datetime format
        try:
            datetime.fromisoformat(metadata.created_at)
        except ValueError as e:
            raise ValueError("created_at must be valid ISO format timestamp") from e

    def _validate_metadata_dict(self, metadata: dict[str, Any]) -> None:
        """Validate metadata dictionary format."""
        required_fields = ["checkpoint_id", "created_at"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Required field missing: {field}")

        list_fields = ["progress_markers", "tags"]
        for field in list_fields:
            if field in metadata and not isinstance(metadata[field], list):
                raise ValueError(f"{field} must be a list")

        dict_fields = ["performance_metrics", "custom_fields"]
        for field in dict_fields:
            if field in metadata and not isinstance(metadata[field], dict):
                raise ValueError(f"{field} must be a dict")

    def _store_metadata(self, metadata: CheckpointMetadata) -> None:
        """Store metadata in database."""
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
        self._update_metadata_cache(metadata.checkpoint_id, asdict(metadata))

    def _checkpoint_exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint file exists."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl.gz"
        return checkpoint_file.exists()

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _row_to_metadata_dict(self, row) -> dict[str, Any]:
        """Convert database row to metadata dictionary."""
        columns = [
            "checkpoint_id",
            "created_at",
            "game_location",
            "progress_markers",
            "performance_metrics",
            "tags",
            "custom_fields",
            "file_size",
            "checksum",
            "schema_version",
        ]

        metadata = dict(zip(columns, row, strict=False))

        # Parse JSON fields
        json_fields = ["progress_markers", "performance_metrics", "tags", "custom_fields"]
        for field in json_fields:
            if metadata[field]:
                try:
                    metadata[field] = json.loads(metadata[field])
                except json.JSONDecodeError:
                    metadata[field] = [] if field in ["progress_markers", "tags"] else {}

        return metadata

    def _update_metadata_cache(self, checkpoint_id: str, metadata: dict[str, Any]) -> None:
        """Update metadata cache with LRU eviction."""
        # Add to cache
        self._metadata_cache[checkpoint_id] = metadata
        self._cache_access_times[checkpoint_id] = time.time()

        # Evict if cache is full
        if len(self._metadata_cache) > self.METADATA_CACHE_SIZE:
            # Remove oldest accessed item
            oldest_id = min(
                self._cache_access_times.keys(), key=lambda x: self._cache_access_times[x]
            )
            del self._metadata_cache[oldest_id]
            del self._cache_access_times[oldest_id]

    def _filter_by_performance(
        self, checkpoints: list[dict[str, Any]], criteria: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Filter checkpoints by performance criteria."""
        filtered = []

        for checkpoint in checkpoints:
            performance = checkpoint.get("performance_metrics", {})

            if "performance_min" in criteria:
                success_rate = performance.get("success_rate", 0)
                if success_rate < criteria["performance_min"]:
                    continue

            if "performance_max" in criteria:
                execution_time = performance.get("execution_time", float("inf"))
                if execution_time > criteria["performance_max"]:
                    continue

            filtered.append(checkpoint)

        return filtered

    def _filter_by_custom_fields(
        self, checkpoints: list[dict[str, Any]], custom_criteria: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Filter checkpoints by custom field criteria."""
        filtered = []

        for checkpoint in checkpoints:
            custom_fields = checkpoint.get("custom_fields", {})

            matches = True
            for key, expected_value in custom_criteria.items():
                if key not in custom_fields or custom_fields[key] != expected_value:
                    matches = False
                    break

            if matches:
                filtered.append(checkpoint)

        return filtered

    def _remove_checkpoint(self, checkpoint_id: str) -> bool:
        """Remove checkpoint file and metadata."""
        try:
            # Remove file
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl.gz"
            if checkpoint_file.exists():
                checkpoint_file.unlink()

            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM checkpoint_metadata WHERE checkpoint_id = ?", (checkpoint_id,)
                )
                conn.commit()

            # Remove from cache
            if checkpoint_id in self._metadata_cache:
                del self._metadata_cache[checkpoint_id]
                del self._cache_access_times[checkpoint_id]

            return True

        except Exception as e:
            logger.error(f"Failed to remove checkpoint {checkpoint_id}: {e}")
            return False
