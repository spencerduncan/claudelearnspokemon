"""
MemgraphCheckpointDiscovery - Ultra-fast checkpoint discovery using memgraph.

Leveraging lessons from memoryalpha's 393x performance improvement:
- Pre-calculated scores on nodes (like TF-IDF scores)
- Weighted relationships for instant ranking
- Graph-native operations for <5ms queries

Performance targets:
- Discovery queries: <5ms (inspired by memoryalpha's 4.3ms)
- Checkpoint save with scoring: <50ms
- Fuzzy matching: <2ms

Author: Ron Swanson - Get It Done Right
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import Levenshtein
import mgclient
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LocationScore:
    """Pre-calculated location scoring data."""

    location_name: str
    distance: float
    relevance_score: float
    checkpoint_count: int


@dataclass
class CheckpointDiscoveryResult:
    """Result from checkpoint discovery with performance metrics."""

    checkpoint_id: str
    location_name: str
    confidence_score: float
    distance: float
    query_time_ms: float


@dataclass
class CheckpointSuggestion:
    """Single checkpoint suggestion with ranking and distance information."""

    checkpoint_id: str
    location_name: str
    confidence_score: float
    relevance_score: float
    distance_score: float
    final_score: float
    fuzzy_match_distance: int = 0  # Levenshtein distance for fuzzy matches


@dataclass
class CheckpointSuggestions:
    """Multiple checkpoint suggestions with metadata."""

    suggestions: list[CheckpointSuggestion]
    query_location: str
    query_time_ms: float
    total_matches_found: int
    fuzzy_matches_used: bool = False


class MemgraphConnectionError(Exception):
    """Raised when memgraph connection fails."""

    pass


class MemgraphCheckpointDiscovery:
    """
    Ultra-fast checkpoint discovery using pre-calculated graph scores.

    Based on memoryalpha's success with pre-calculated TF-IDF weights,
    this implementation stores all scores in the graph for instant lookup.
    """

    # Performance constants (based on memoryalpha results)
    TARGET_DISCOVERY_TIME_MS = 5.0
    TARGET_SAVE_TIME_MS = 50.0
    TARGET_FUZZY_MATCH_TIME_MS = 2.0

    # Graph schema constants
    LOCATION_LABEL = "Location"
    CHECKPOINT_LABEL = "Checkpoint"
    SAVED_AT_REL = "SAVED_AT"
    CONNECTED_TO_REL = "CONNECTED_TO"
    SIMILAR_TO_REL = "SIMILAR_TO"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7688,  # Separate instance from coding knowledge (7687)
        username: str = "memgraph",
        password: str = "memgraph",
        database: str = "checkpoint_discovery",  # Dedicated database for checkpoint discovery
        enable_metrics: bool = True,
    ):
        """
        Initialize memgraph connection and prepare for pre-calculated discovery.

        Uses a dedicated memgraph instance (port 7688) separate from coding knowledge system.

        Args:
            host: Memgraph server host
            port: Memgraph server port (default 7688, separate from coding knowledge on 7687)
            username: Database username
            password: Database password
            database: Database name for checkpoint discovery (isolated namespace)
            enable_metrics: Enable performance metrics collection
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.enable_metrics = enable_metrics

        # Connection management
        self._connection: mgclient.Connection | None = None
        self._connection_lock = threading.Lock()

        # Performance metrics
        self._metrics: dict[str, Any] = {
            "discovery_queries": 0,
            "discovery_total_time_ms": 0.0,
            "save_operations": 0,
            "save_total_time_ms": 0.0,
            "fuzzy_matches": 0,
            "fuzzy_match_total_time_ms": 0.0,
        }

        # Initialize connection and schema
        self._connect()
        self._initialize_schema()

        logger.info(
            "MemgraphCheckpointDiscovery initialized",
            host=host,
            port=port,
            database=database,
            enable_metrics=enable_metrics,
            note="Using separate memgraph instance from coding knowledge system",
        )

    def _connect(self) -> None:
        """Establish connection to memgraph."""
        try:
            self._connection = mgclient.connect(
                host=self.host, port=self.port, username=self.username, password=self.password
            )
            logger.info("Connected to memgraph")
        except Exception as e:
            logger.error("Failed to connect to memgraph", error=str(e))
            raise MemgraphConnectionError(f"Connection failed: {e}") from e

    def _initialize_schema(self) -> None:
        """
        Initialize memgraph schema with indexes for performance.
        Sets up the foundation for pre-calculated scoring.
        """
        try:
            with self._connection_lock:
                if self._connection is None:
                    raise MemgraphConnectionError("Connection not established")
                cursor = self._connection.cursor()

                # Create indexes for fast lookups (like memoryalpha's Term indexes)
                cursor.execute("CREATE INDEX ON :Location(name)")
                cursor.execute("CREATE INDEX ON :Checkpoint(id)")
                cursor.execute("CREATE INDEX ON :Checkpoint(composite_score)")

                # Create constraint for unique checkpoint IDs
                cursor.execute("CREATE CONSTRAINT ON (c:Checkpoint) ASSERT c.id IS UNIQUE")

                logger.info("Memgraph schema initialized with indexes")

        except Exception as e:
            logger.warning("Schema initialization failed (may already exist)", error=str(e))

    def save_checkpoint_with_scoring(
        self, checkpoint_id: str, location: str, metadata: dict[str, Any]
    ) -> None:
        """
        Save checkpoint with pre-calculated composite scoring.

        Like memoryalpha's TF-IDF pre-calculation, we compute all scores
        at save time for instant discovery later.

        Args:
            checkpoint_id: Unique checkpoint identifier
            location: Game location name
            metadata: Checkpoint metadata including success_rate, strategic_value, etc.
        """
        start_time = time.perf_counter()

        try:
            with self._connection_lock:
                if self._connection is None:
                    raise MemgraphConnectionError("Connection not established")
                cursor = self._connection.cursor()

                # Extract scoring components
                success_rate = metadata.get("success_rate", 0.0)
                strategic_value = metadata.get("strategic_value", 0.0)
                access_count = metadata.get("access_count", 0)
                created_at = metadata.get("created_at", datetime.now().isoformat())

                # Pre-calculate composite score (the key insight from memoryalpha!)
                composite_score = self._calculate_composite_score(
                    success_rate, strategic_value, access_count, created_at
                )

                # Create or update checkpoint node with pre-calculated score
                query = """
                MERGE (c:Checkpoint {id: $checkpoint_id})
                SET c.success_rate = $success_rate,
                    c.strategic_value = $strategic_value,
                    c.access_count = $access_count,
                    c.created_at = $created_at,
                    c.composite_score = $composite_score,
                    c.file_path = $file_path
                """

                cursor.execute(
                    query,
                    {
                        "checkpoint_id": checkpoint_id,
                        "success_rate": success_rate,
                        "strategic_value": strategic_value,
                        "access_count": access_count,
                        "created_at": created_at,
                        "composite_score": composite_score,
                        "file_path": metadata.get("file_path", ""),
                    },
                )

                # Create location node and relationship with pre-calculated relevance
                self._create_location_relationship(cursor, checkpoint_id, location)

                if self.enable_metrics:
                    self._metrics["save_operations"] += 1
                    self._metrics["save_total_time_ms"] += (time.perf_counter() - start_time) * 1000

                logger.debug(
                    "Checkpoint saved with pre-calculated scoring",
                    checkpoint_id=checkpoint_id,
                    location=location,
                    composite_score=composite_score,
                )

        except Exception as e:
            logger.error(
                "Failed to save checkpoint with scoring", error=str(e), checkpoint_id=checkpoint_id
            )
            raise

    def find_nearest_checkpoint(self, location: str) -> str:
        """
        Ultra-fast checkpoint discovery using pre-calculated scores.

        Target: <5ms (like memoryalpha's optimized queries)

        Args:
            location: Target location name (supports fuzzy matching)

        Returns:
            str: Checkpoint ID of best match, empty string if none found
        """
        start_time = time.perf_counter()

        try:
            with self._connection_lock:
                if self._connection is None:
                    raise MemgraphConnectionError("Connection not established")
                cursor = self._connection.cursor()

                # Use pre-calculated similarities for fuzzy matching
                matched_locations = self._fuzzy_match_location(cursor, location)

                if not matched_locations:
                    return ""

                # Query uses pre-calculated scores - no computation needed!
                query = """
                MATCH (c:Checkpoint)-[r:SAVED_AT]->(l:Location)
                WHERE l.name IN $locations
                RETURN c.id, r.relevance_score * c.composite_score as final_score
                ORDER BY final_score DESC
                LIMIT 1
                """

                cursor.execute(query, {"locations": matched_locations})
                result = cursor.fetchone()

                query_time_ms = (time.perf_counter() - start_time) * 1000

                if self.enable_metrics:
                    self._metrics["discovery_queries"] += 1
                    self._metrics["discovery_total_time_ms"] += query_time_ms

                if result:
                    checkpoint_id = result[0]
                    logger.debug(
                        "Checkpoint discovered",
                        checkpoint_id=checkpoint_id,
                        query_time_ms=query_time_ms,
                        final_score=result[1],
                    )
                    return checkpoint_id

                return ""

        except Exception as e:
            logger.error("Checkpoint discovery failed", error=str(e), location=location)
            return ""

    def find_nearest_checkpoints(
        self, location: str, max_suggestions: int = 5, include_distance: bool = True
    ) -> CheckpointSuggestions:
        """
        Find multiple checkpoint suggestions with ranking and distance information.

        Enhanced discovery method supporting the full requirement from Issue #82.
        Provides multiple suggestions with comprehensive scoring and distance metrics.

        Target: <10ms for multiple results (2x single result allowance)

        Args:
            location: Target location name (supports fuzzy matching)
            max_suggestions: Maximum number of suggestions to return (default: 5)
            include_distance: Include distance/score information (default: True)

        Returns:
            CheckpointSuggestions: Complete suggestions with rankings and metadata
        """
        start_time = time.perf_counter()

        try:
            with self._connection_lock:
                if self._connection is None:
                    raise MemgraphConnectionError("Connection not established")
                cursor = self._connection.cursor()

                # Use pre-calculated similarities for fuzzy matching
                matched_locations = self._fuzzy_match_location(cursor, location)

                if not matched_locations:
                    query_time_ms = (time.perf_counter() - start_time) * 1000
                    return CheckpointSuggestions(
                        suggestions=[],
                        query_location=location,
                        query_time_ms=query_time_ms,
                        total_matches_found=0,
                        fuzzy_matches_used=False,
                    )

                # Enhanced query for multiple results with detailed scoring
                query = """
                MATCH (c:Checkpoint)-[r:SAVED_AT]->(l:Location)
                WHERE l.name IN $locations
                RETURN c.id, l.name, r.relevance_score, c.composite_score,
                       r.relevance_score * c.composite_score as final_score
                ORDER BY final_score DESC
                LIMIT $max_suggestions
                """

                cursor.execute(
                    query, {"locations": matched_locations, "max_suggestions": max_suggestions}
                )
                results = cursor.fetchall()

                query_time_ms = (time.perf_counter() - start_time) * 1000

                # Build suggestions list with comprehensive information
                suggestions = []
                for result in results:
                    (
                        checkpoint_id,
                        matched_location,
                        relevance_score,
                        composite_score,
                        final_score,
                    ) = result

                    # Calculate fuzzy match distance for transparency
                    fuzzy_distance = self._calculate_fuzzy_distance(location, matched_location)

                    suggestion = CheckpointSuggestion(
                        checkpoint_id=checkpoint_id,
                        location_name=matched_location,
                        confidence_score=composite_score,
                        relevance_score=relevance_score,
                        distance_score=1.0 - (fuzzy_distance / 10.0),  # Normalize distance to score
                        final_score=final_score,
                        fuzzy_match_distance=fuzzy_distance,
                    )
                    suggestions.append(suggestion)

                # Determine if fuzzy matching was used
                fuzzy_used = any(loc.lower() != location.lower() for loc in matched_locations)

                # Update metrics
                if self.enable_metrics:
                    self._metrics["discovery_queries"] += 1
                    self._metrics["discovery_total_time_ms"] += query_time_ms

                checkpoint_suggestions = CheckpointSuggestions(
                    suggestions=suggestions,
                    query_location=location,
                    query_time_ms=query_time_ms,
                    total_matches_found=len(results),
                    fuzzy_matches_used=fuzzy_used,
                )

                logger.debug(
                    "Multiple checkpoint suggestions generated",
                    location=location,
                    suggestions_count=len(suggestions),
                    query_time_ms=query_time_ms,
                    fuzzy_matches_used=fuzzy_used,
                )

                return checkpoint_suggestions

        except Exception as e:
            logger.error("Multiple checkpoint discovery failed", error=str(e), location=location)
            query_time_ms = (time.perf_counter() - start_time) * 1000
            return CheckpointSuggestions(
                suggestions=[],
                query_location=location,
                query_time_ms=query_time_ms,
                total_matches_found=0,
                fuzzy_matches_used=False,
            )

    def _calculate_composite_score(
        self, success_rate: float, strategic_value: float, access_count: int, created_at: str
    ) -> float:
        """
        Pre-calculate composite checkpoint score.

        Like TF-IDF calculation, this runs once at save time
        rather than on every query.
        """
        try:
            # Parse timestamp and calculate age factor
            created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            age_days = (datetime.now(created_time.tzinfo) - created_time).days
            recency_factor = 1.0 / (age_days + 1)  # Newer checkpoints score higher

            # Access frequency factor (logarithmic scaling)
            import math

            frequency_factor = math.log(access_count + 1) / math.log(10)  # Log base 10

            # Weighted combination (tunable weights)
            composite_score = (
                success_rate * 0.3
                + strategic_value * 0.3
                + recency_factor * 0.2
                + frequency_factor * 0.2
            )

            return max(0.0, min(1.0, composite_score))  # Clamp to [0,1]

        except Exception as e:
            logger.warning("Score calculation failed, using default", error=str(e))
            return 0.5  # Default score

    def _create_location_relationship(self, cursor, checkpoint_id: str, location: str) -> None:
        """Create location node and relationship with pre-calculated relevance."""
        try:
            # Create location node (merge to avoid duplicates)
            location_query = """
            MERGE (l:Location {name: $location})
            """
            cursor.execute(location_query, {"location": location})

            # Create relationship with perfect relevance for exact match
            relationship_query = """
            MATCH (c:Checkpoint {id: $checkpoint_id})
            MATCH (l:Location {name: $location})
            MERGE (c)-[r:SAVED_AT]->(l)
            SET r.relevance_score = 1.0,
                r.created_at = datetime()
            """
            cursor.execute(
                relationship_query, {"checkpoint_id": checkpoint_id, "location": location}
            )

        except Exception as e:
            logger.error("Failed to create location relationship", error=str(e))
            raise

    def _fuzzy_match_location(self, cursor, location: str) -> list[str]:
        """
        Fast fuzzy matching using pre-calculated similarities.

        Target: <2ms for fuzzy matching
        """
        start_time = time.perf_counter()

        try:
            # First try exact match
            exact_query = """
            MATCH (l:Location {name: $location})
            RETURN l.name
            """
            cursor.execute(exact_query, {"location": location})
            exact_result = cursor.fetchone()

            if exact_result:
                return [exact_result[0]]

            # Fuzzy matching with Levenshtein distance
            # Get all locations for comparison
            all_locations_query = "MATCH (l:Location) RETURN l.name"
            cursor.execute(all_locations_query)
            all_locations = [row[0] for row in cursor.fetchall()]

            # Find best matches using Levenshtein distance
            matches = []
            for loc_name in all_locations:
                distance = Levenshtein.distance(location.lower(), loc_name.lower())
                # Accept matches with edit distance <= 2
                if distance <= 2:
                    matches.append((loc_name, distance))

            # Sort by distance (best matches first)
            matches.sort(key=lambda x: x[1])

            # Return top 3 matches
            matched_locations = [match[0] for match in matches[:3]]

            if self.enable_metrics:
                query_time_ms = (time.perf_counter() - start_time) * 1000
                self._metrics["fuzzy_matches"] += 1
                self._metrics["fuzzy_match_total_time_ms"] += query_time_ms

            return matched_locations

        except Exception as e:
            logger.error("Fuzzy matching failed", error=str(e))
            return []

    def _calculate_fuzzy_distance(self, query_location: str, matched_location: str) -> int:
        """
        Calculate fuzzy match distance for transparency in results.

        Args:
            query_location: Original query location
            matched_location: Matched location from database

        Returns:
            int: Levenshtein distance (0 = exact match)
        """
        if query_location.lower() == matched_location.lower():
            return 0
        return Levenshtein.distance(query_location.lower(), matched_location.lower())

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for monitoring."""
        if not self.enable_metrics:
            return {"metrics_disabled": True}

        metrics = self._metrics.copy()

        # Calculate averages
        if metrics["discovery_queries"] > 0:
            metrics["avg_discovery_time_ms"] = (
                metrics["discovery_total_time_ms"] / metrics["discovery_queries"]
            )
        else:
            metrics["avg_discovery_time_ms"] = 0.0

        if metrics["save_operations"] > 0:
            metrics["avg_save_time_ms"] = metrics["save_total_time_ms"] / metrics["save_operations"]
        else:
            metrics["avg_save_time_ms"] = 0.0

        if metrics["fuzzy_matches"] > 0:
            metrics["avg_fuzzy_match_time_ms"] = (
                metrics["fuzzy_match_total_time_ms"] / metrics["fuzzy_matches"]
            )
        else:
            metrics["avg_fuzzy_match_time_ms"] = 0.0

        # Performance status
        metrics["performance_status"] = {
            "discovery_target_met": metrics["avg_discovery_time_ms"]
            <= self.TARGET_DISCOVERY_TIME_MS,
            "save_target_met": metrics["avg_save_time_ms"] <= self.TARGET_SAVE_TIME_MS,
            "fuzzy_target_met": metrics["avg_fuzzy_match_time_ms"]
            <= self.TARGET_FUZZY_MATCH_TIME_MS,
        }

        return metrics

    def close(self) -> None:
        """Close memgraph connection."""
        if self._connection:
            with self._connection_lock:
                self._connection.close()
                self._connection = None
            logger.info("Memgraph connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
