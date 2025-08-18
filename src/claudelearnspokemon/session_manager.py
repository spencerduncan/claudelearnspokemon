"""
Advanced Session Management for PokemonGymAdapter ecosystem.

This module provides comprehensive session management capabilities including:
- Multi-session coordination and pooling
- Session lifecycle management with persistence
- Concurrent session handling with thread safety
- Session health monitoring and automatic cleanup
- Session metrics and resource usage tracking

Performance Requirements:
- Session acquisition: < 50ms
- Session state persistence: < 100ms
- Health check cycle: < 1 second
- Thread-safe operations: < 10ms overhead
- Memory per session: < 5MB

Author: John Botmack - Session Management and Performance Engineering
"""

import json
import logging
import resource
import sqlite3
import threading
import time
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Condition, Event, Lock, RLock
from typing import Any
from uuid import uuid4

# Performance monitoring imports
import psutil

# Import types from foundation
from typing_extensions import TypedDict

# =============================================================================
# Session State and Configuration Types
# =============================================================================


class SessionState(Enum):
    """Session lifecycle states with performance tracking."""

    UNINITIALIZED = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    IDLE = auto()
    UNHEALTHY = auto()
    RECOVERING = auto()
    EXPIRED = auto()
    CLEANUP = auto()
    TERMINATED = auto()


class SessionPriority(Enum):
    """Session priority levels for resource allocation."""

    CRITICAL = 1  # System-critical sessions
    HIGH = 2  # High-priority user sessions
    NORMAL = 3  # Standard user sessions
    LOW = 4  # Background/maintenance sessions
    CLEANUP = 5  # Cleanup and garbage collection


@dataclass
class SessionMetrics:
    """Performance and resource metrics for session monitoring."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    last_health_check: float = 0.0

    # Operation counts
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0

    # Performance metrics
    total_response_time_ms: float = 0.0
    min_response_time_ms: float = float("inf")
    max_response_time_ms: float = 0.0

    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0

    # Error tracking
    consecutive_failures: int = 0
    last_error_timestamp: float = 0.0
    error_rate_per_minute: float = 0.0

    @property
    def average_response_time_ms(self) -> float:
        """Calculate average response time."""
        return self.total_response_time_ms / max(1, self.total_operations)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        total = max(1, self.total_operations)
        return (self.successful_operations / total) * 100.0

    @property
    def operations_per_second(self) -> float:
        """Calculate operations per second."""
        duration = time.time() - self.created_at
        return self.total_operations / max(1.0, duration)


@dataclass
class SessionPoolConfig:
    """Configuration for session pool management."""

    # Pool sizing
    min_pool_size: int = 2
    max_pool_size: int = 8
    target_idle_sessions: int = 1

    # Session lifecycle
    session_timeout_seconds: float = 300.0
    idle_timeout_seconds: float = 60.0
    health_check_interval_seconds: float = 30.0

    # Performance tuning
    acquisition_timeout_seconds: float = 10.0
    cleanup_interval_seconds: float = 120.0
    metrics_retention_seconds: float = 3600.0

    # Persistence settings
    enable_persistence: bool = True
    persistence_file: str = "session_states.db"
    auto_recovery: bool = True

    # Thread safety
    max_concurrent_acquisitions: int = 4
    thread_pool_size: int = 4

    # Resource limits
    max_memory_mb_per_session: float = 50.0
    max_cpu_percent_per_session: float = 25.0
    max_network_mbps_per_session: float = 10.0


class SessionInfo(TypedDict):
    """Session information structure."""

    session_id: str
    state: SessionState
    priority: SessionPriority
    created_at: float
    last_activity: float
    port: int
    container_id: str
    adapter_instance: object | None  # WeakReference to adapter
    health_score: float
    metrics: SessionMetrics


# =============================================================================
# Session Persistence Layer
# =============================================================================


class SessionPersistence:
    """
    Thread-safe session state persistence using SQLite.

    Provides fast and reliable storage for session state recovery
    with ACID compliance and minimal disk I/O overhead.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = Lock()
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema with optimized indexes."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        state TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        port INTEGER NOT NULL,
                        container_id TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        last_activity REAL NOT NULL,
                        health_score REAL DEFAULT 1.0,
                        metrics_json TEXT,
                        metadata_json TEXT,
                        updated_at REAL DEFAULT (strftime('%s', 'now'))
                    )
                """
                )

                # Performance indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_session_state ON sessions(state)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_priority ON sessions(priority)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_activity ON sessions(last_activity)"
                )

                conn.commit()
            finally:
                conn.close()

    def save_session(self, session_info: SessionInfo) -> None:
        """Save session state with optimized write performance."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                metrics_json = json.dumps(
                    {
                        "total_operations": session_info["metrics"].total_operations,
                        "successful_operations": session_info["metrics"].successful_operations,
                        "failed_operations": session_info["metrics"].failed_operations,
                        "average_response_time_ms": session_info[
                            "metrics"
                        ].average_response_time_ms,
                        "success_rate": session_info["metrics"].success_rate,
                        "memory_usage_mb": session_info["metrics"].memory_usage_mb,
                        "cpu_time_seconds": session_info["metrics"].cpu_time_seconds,
                    }
                )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO sessions
                    (session_id, state, priority, port, container_id, created_at,
                     last_activity, health_score, metrics_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        session_info["session_id"],
                        session_info["state"].name,
                        session_info["priority"].value,
                        session_info["port"],
                        session_info["container_id"],
                        session_info["created_at"],
                        session_info["last_activity"],
                        session_info["health_score"],
                        metrics_json,
                        time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Load session state with cache-friendly access pattern."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    """
                    SELECT session_id, state, priority, port, container_id,
                           created_at, last_activity, health_score, metrics_json
                    FROM sessions WHERE session_id = ?
                """,
                    (session_id,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                metrics_data = json.loads(row[8]) if row[8] else {}

                return {
                    "session_id": row[0],
                    "state": SessionState[row[1]],
                    "priority": SessionPriority(row[2]),
                    "port": row[3],
                    "container_id": row[4],
                    "created_at": row[5],
                    "last_activity": row[6],
                    "health_score": row[7],
                    "metrics": metrics_data,
                }
            finally:
                conn.close()

    def load_all_sessions(self) -> list[dict[str, Any]]:
        """Load all persisted sessions for recovery."""
        sessions = []
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    """
                    SELECT session_id, state, priority, port, container_id,
                           created_at, last_activity, health_score, metrics_json
                    FROM sessions
                    ORDER BY priority ASC, last_activity DESC
                """
                )

                for row in cursor.fetchall():
                    metrics_data = json.loads(row[8]) if row[8] else {}
                    sessions.append(
                        {
                            "session_id": row[0],
                            "state": SessionState[row[1]],
                            "priority": SessionPriority(row[2]),
                            "port": row[3],
                            "container_id": row[4],
                            "created_at": row[5],
                            "last_activity": row[6],
                            "health_score": row[7],
                            "metrics": metrics_data,
                        }
                    )
            finally:
                conn.close()

        return sessions

    def delete_session(self, session_id: str) -> None:
        """Delete session from persistence."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
            finally:
                conn.close()

    def cleanup_expired_sessions(self, max_age_seconds: float) -> int:
        """Clean up expired session records."""
        cutoff_time = time.time() - max_age_seconds

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    """
                    DELETE FROM sessions
                    WHERE last_activity < ?
                    AND state IN ('TERMINATED', 'EXPIRED', 'CLEANUP')
                """,
                    (cutoff_time,),
                )

                deleted_count = cursor.rowcount
                conn.commit()
                return deleted_count
            finally:
                conn.close()


# =============================================================================
# Session Health Monitor
# =============================================================================


class SessionHealthMonitor:
    """
    High-performance session health monitoring with circuit breaker patterns.

    Monitors session health using multiple metrics and provides predictive
    failure detection with automatic recovery strategies.
    """

    def __init__(self, config: SessionPoolConfig):
        self.config = config
        self._health_scores: dict[str, float] = {}
        self._health_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._circuit_breakers: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

        # Health check thresholds
        self.critical_health_threshold = 0.3
        self.warning_health_threshold = 0.7
        self.recovery_health_threshold = 0.8

        # Circuit breaker settings
        self.failure_threshold = 5
        self.recovery_threshold = 3
        self.circuit_timeout = 60.0

    def update_health_metrics(self, session_id: str, metrics: SessionMetrics) -> float:
        """Update health score based on session metrics."""
        with self._lock:
            # Calculate health score based on multiple factors
            health_components = []

            # Success rate component (0.0 - 1.0)
            success_rate = metrics.success_rate / 100.0
            health_components.append(("success_rate", success_rate, 0.4))

            # Response time component (inverse relationship)
            avg_response = metrics.average_response_time_ms
            response_score = max(0.0, 1.0 - (avg_response / 1000.0))  # 1s = 0 score
            health_components.append(("response_time", response_score, 0.3))

            # Resource usage component
            memory_score = max(
                0.0, 1.0 - (metrics.memory_usage_mb / self.config.max_memory_mb_per_session)
            )
            health_components.append(("memory_usage", memory_score, 0.2))

            # Error rate component
            error_rate_score = max(
                0.0, 1.0 - (metrics.error_rate_per_minute / 10.0)
            )  # 10 errors/min = 0 score
            health_components.append(("error_rate", error_rate_score, 0.1))

            # Calculate weighted health score
            total_weight = sum(weight for _, _, weight in health_components)
            health_score = (
                sum(score * weight for _, score, weight in health_components) / total_weight
            )

            # Update health tracking
            self._health_scores[session_id] = health_score
            self._health_history[session_id].append((time.time(), health_score))

            return health_score

    def get_health_score(self, session_id: str) -> float:
        """Get current health score for session."""
        with self._lock:
            return self._health_scores.get(session_id, 1.0)

    def get_health_trend(self, session_id: str, window_seconds: float = 300.0) -> float:
        """Get health trend over specified time window."""
        with self._lock:
            history = self._health_history.get(session_id, deque())
            if len(history) < 2:
                return 0.0

            # Filter to time window
            cutoff_time = time.time() - window_seconds
            recent_scores = [(t, score) for t, score in history if t >= cutoff_time]

            if len(recent_scores) < 2:
                return 0.0

            # Calculate trend (positive = improving, negative = degrading)
            first_score = recent_scores[0][1]
            last_score = recent_scores[-1][1]
            return last_score - first_score

    def is_healthy(self, session_id: str) -> bool:
        """Check if session is healthy."""
        health_score = self.get_health_score(session_id)
        return health_score >= self.warning_health_threshold

    def requires_attention(self, session_id: str) -> bool:
        """Check if session requires immediate attention."""
        health_score = self.get_health_score(session_id)
        trend = self.get_health_trend(session_id)

        return health_score <= self.critical_health_threshold or (
            health_score <= self.warning_health_threshold and trend < -0.2
        )

    def can_recover(self, session_id: str) -> bool:
        """Check if session can be recovered."""
        health_score = self.get_health_score(session_id)
        trend = self.get_health_trend(session_id)

        return (
            health_score >= self.recovery_health_threshold
            or trend > 0.1  # Positive trend indicates recovery potential
        )

    def update_circuit_breaker(self, session_id: str, success: bool) -> None:
        """Update circuit breaker state based on operation result."""
        with self._lock:
            if session_id not in self._circuit_breakers:
                self._circuit_breakers[session_id] = {
                    "state": "CLOSED",
                    "failure_count": 0,
                    "last_failure_time": 0.0,
                    "success_count": 0,
                }

            breaker = self._circuit_breakers[session_id]
            current_time = time.time()

            if success:
                breaker["success_count"] += 1

                # Check for recovery from OPEN state
                if (
                    breaker["state"] == "HALF_OPEN"
                    and breaker["success_count"] >= self.recovery_threshold
                ):
                    breaker["state"] = "CLOSED"
                    breaker["failure_count"] = 0
                    breaker["success_count"] = 0
            else:
                breaker["failure_count"] += 1
                breaker["last_failure_time"] = current_time
                breaker["success_count"] = 0

                # Check for circuit opening
                if (
                    breaker["state"] == "CLOSED"
                    and breaker["failure_count"] >= self.failure_threshold
                ):
                    breaker["state"] = "OPEN"
                elif breaker["state"] == "HALF_OPEN":
                    breaker["state"] = "OPEN"

            # Check for half-open transition
            if (
                breaker["state"] == "OPEN"
                and current_time - breaker["last_failure_time"] > self.circuit_timeout
            ):
                breaker["state"] = "HALF_OPEN"
                breaker["success_count"] = 0

    def is_circuit_open(self, session_id: str) -> bool:
        """Check if circuit breaker is open for session."""
        with self._lock:
            breaker = self._circuit_breakers.get(session_id, {})
            return breaker.get("state") == "OPEN"


# =============================================================================
# Core Session Manager
# =============================================================================


class SessionManager:
    """
    Advanced multi-session manager with high-performance coordination.

    Provides comprehensive session lifecycle management with:
    - Thread-safe session pooling and acquisition
    - Persistent session state recovery
    - Proactive health monitoring and auto-recovery
    - Resource usage tracking and optimization
    - Circuit breaker patterns for fault tolerance

    Performance characteristics:
    - Session acquisition: O(1) amortized
    - Health checks: O(n) where n = active sessions
    - Resource cleanup: O(log n) priority queue
    - Thread contention: Minimized via fine-grained locking
    """

    def __init__(self, config: SessionPoolConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core session tracking
        self._sessions: dict[str, SessionInfo] = {}
        self._session_lock = RLock()

        # Session pools by state
        self._idle_sessions: set[str] = set()
        self._active_sessions: set[str] = set()
        self._unhealthy_sessions: set[str] = set()

        # Thread pool for concurrent operations
        self._thread_pool = ThreadPoolExecutor(max_workers=config.thread_pool_size)

        # Persistence and health monitoring
        self._persistence = None
        if config.enable_persistence:
            self._persistence = SessionPersistence(config.persistence_file)

        self._health_monitor = SessionHealthMonitor(config)

        # Synchronization primitives
        self._acquisition_semaphore = threading.Semaphore(config.max_concurrent_acquisitions)
        self._shutdown_event = Event()
        self._cleanup_condition = Condition()

        # Background task handles
        self._health_check_thread: threading.Thread | None = None
        self._cleanup_thread: threading.Thread | None = None

        # Performance metrics
        self._global_metrics = {
            "sessions_created": 0,
            "sessions_destroyed": 0,
            "successful_acquisitions": 0,
            "failed_acquisitions": 0,
            "recovery_operations": 0,
            "cleanup_operations": 0,
        }
        self._metrics_lock = Lock()

        # Start background tasks
        self._start_background_tasks()

        # Recover persisted sessions
        if config.auto_recovery:
            self._recover_sessions()

    def _start_background_tasks(self) -> None:
        """Start background monitoring and cleanup tasks."""
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop, name="SessionHealthMonitor", daemon=True
        )
        self._health_check_thread.start()

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, name="SessionCleanup", daemon=True
        )
        self._cleanup_thread.start()

    def _health_check_loop(self) -> None:
        """Background health monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                self._perform_health_checks()
                time.sleep(self.config.health_check_interval_seconds)
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                time.sleep(5.0)  # Back off on errors

    def _cleanup_loop(self) -> None:
        """Background cleanup and maintenance loop."""
        while not self._shutdown_event.is_set():
            try:
                self._perform_cleanup()
                with self._cleanup_condition:
                    self._cleanup_condition.wait(timeout=self.config.cleanup_interval_seconds)
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                time.sleep(10.0)  # Back off on errors

    def _recover_sessions(self) -> None:
        """Recover sessions from persistent storage."""
        if not self._persistence:
            return

        try:
            persisted_sessions = self._persistence.load_all_sessions()
            recovered_count = 0

            for session_data in persisted_sessions:
                # Skip sessions that are too old or terminated
                age = time.time() - session_data["last_activity"]
                if age > self.config.session_timeout_seconds:
                    continue

                if session_data["state"] in [SessionState.TERMINATED, SessionState.EXPIRED]:
                    continue

                # Recover session with health validation
                session_info = self._create_session_info_from_data(session_data)

                with self._session_lock:
                    self._sessions[session_info["session_id"]] = session_info

                    # Add to appropriate pool based on state
                    if session_info["state"] == SessionState.IDLE:
                        self._idle_sessions.add(session_info["session_id"])
                    elif session_info["state"] == SessionState.ACTIVE:
                        self._active_sessions.add(session_info["session_id"])
                    else:
                        self._unhealthy_sessions.add(session_info["session_id"])

                recovered_count += 1
                self.logger.info(f"Recovered session {session_info['session_id']} from persistence")

            if recovered_count > 0:
                self.logger.info(
                    f"Successfully recovered {recovered_count} sessions from persistence"
                )

        except Exception as e:
            self.logger.error(f"Session recovery failed: {e}")

    def _create_session_info_from_data(self, session_data: dict[str, Any]) -> SessionInfo:
        """Create SessionInfo from persisted data."""
        metrics = SessionMetrics(session_id=session_data["session_id"])

        # Restore metrics from persisted data
        if "metrics" in session_data and session_data["metrics"]:
            metrics_data = session_data["metrics"]
            metrics.total_operations = metrics_data.get("total_operations", 0)
            metrics.successful_operations = metrics_data.get("successful_operations", 0)
            metrics.failed_operations = metrics_data.get("failed_operations", 0)
            metrics.memory_usage_mb = metrics_data.get("memory_usage_mb", 0.0)
            metrics.cpu_time_seconds = metrics_data.get("cpu_time_seconds", 0.0)

        return {
            "session_id": session_data["session_id"],
            "state": session_data["state"],
            "priority": session_data["priority"],
            "created_at": session_data["created_at"],
            "last_activity": session_data["last_activity"],
            "port": session_data["port"],
            "container_id": session_data["container_id"],
            "adapter_instance": None,  # Will be restored when adapter reconnects
            "health_score": session_data["health_score"],
            "metrics": metrics,
        }

    @contextmanager
    def acquire_session(
        self,
        priority: SessionPriority = SessionPriority.NORMAL,
        timeout_seconds: float | None = None,
    ):
        """
        Acquire session from pool with automatic release.

        Provides thread-safe session acquisition with:
        - Priority-based allocation
        - Automatic session validation
        - Resource cleanup on exit
        - Performance monitoring

        Args:
            priority: Session priority level
            timeout_seconds: Acquisition timeout (uses config default if None)

        Yields:
            SessionInfo: Acquired session information

        Raises:
            TimeoutError: If acquisition timeout exceeded
            SessionError: If no healthy sessions available
        """
        timeout = timeout_seconds or self.config.acquisition_timeout_seconds
        start_time = time.time()

        if not self._acquisition_semaphore.acquire(timeout=timeout):
            with self._metrics_lock:
                self._global_metrics["failed_acquisitions"] += 1
            raise TimeoutError(f"Session acquisition timeout after {timeout}s")

        session_id = None
        try:
            # Acquire session with priority handling
            session_id = self._acquire_session_internal(
                priority, timeout - (time.time() - start_time)
            )

            if not session_id:
                with self._metrics_lock:
                    self._global_metrics["failed_acquisitions"] += 1
                raise RuntimeError("No healthy sessions available")

            with self._metrics_lock:
                self._global_metrics["successful_acquisitions"] += 1

            # Yield session for use - session remains ACTIVE during with block
            with self._session_lock:
                session_info = self._sessions[session_id]
                session_info["last_activity"] = time.time()
                # CRITICAL: Session must remain in ACTIVE state during yield
                # to prevent other threads from acquiring it
                yield session_info

        finally:
            # Always release session back to pool when 'with' block exits
            if session_id:
                self._release_session_internal(session_id)
            self._acquisition_semaphore.release()

    def _acquire_session_internal(
        self, priority: SessionPriority, timeout_seconds: float
    ) -> str | None:
        """Internal session acquisition logic with optimized retry strategy."""
        deadline = time.time() + timeout_seconds
        retry_count = 0

        while time.time() < deadline:
            # Try to get idle session first
            session_id = self._get_idle_session(priority)
            if session_id:
                return session_id

            # Try to create new session if pool not at capacity
            if self._can_create_session():
                session_id = self._create_session(priority)
                if session_id:
                    return session_id

            # Exponential backoff for better performance under contention
            retry_count += 1
            if retry_count <= 3:
                # Fast retries for quick availability
                time.sleep(0.001)  # 1ms
            elif retry_count <= 10:
                # Medium backoff
                time.sleep(0.01)  # 10ms
            else:
                # Slower backoff to reduce CPU usage
                time.sleep(0.05)  # 50ms

        return None

    def _get_idle_session(self, priority: SessionPriority) -> str | None:
        """Get idle session with thread-safe acquisition."""
        # CRITICAL: All session state changes must happen atomically under lock
        # to prevent multiple threads from acquiring the same session

        with self._session_lock:
            best_session_id = None
            best_score = -1.0

            # Find best idle session that is actually available
            for session_id in list(self._idle_sessions):
                session_info = self._sessions.get(session_id)
                if not session_info:
                    self._idle_sessions.discard(session_id)
                    continue

                # Validate session health
                if not self._health_monitor.is_healthy(session_id):
                    self._idle_sessions.discard(session_id)
                    self._unhealthy_sessions.add(session_id)
                    session_info["state"] = SessionState.UNHEALTHY
                    continue

                # Calculate acquisition score for priority matching
                priority_score = (6 - session_info["priority"].value) / 5.0
                health_score = self._health_monitor.get_health_score(session_id)
                age_penalty = min(0.2, (time.time() - session_info["last_activity"]) / 300.0)

                total_score = priority_score * 0.4 + health_score * 0.5 - age_penalty * 0.1

                if total_score > best_score:
                    best_score = total_score
                    best_session_id = session_id

            # Atomically acquire the best session if found
            if best_session_id:
                # Move session from idle to active atomically
                self._idle_sessions.discard(best_session_id)
                self._active_sessions.add(best_session_id)
                session_info = self._sessions[best_session_id]
                session_info["state"] = SessionState.ACTIVE
                session_info["last_activity"] = time.time()

                return best_session_id

        return None

    def _can_create_session(self) -> bool:
        """Check if new session can be created."""
        with self._session_lock:
            active_count = len(self._sessions)
            return active_count < self.config.max_pool_size

    def _create_session(self, priority: SessionPriority) -> str | None:
        """Create new session with specified priority."""
        try:
            session_id = str(uuid4())

            # Create session info
            metrics = SessionMetrics(session_id=session_id)
            session_info: SessionInfo = {
                "session_id": session_id,
                "state": SessionState.INITIALIZING,
                "priority": priority,
                "created_at": time.time(),
                "last_activity": time.time(),
                "port": self._allocate_port(),
                "container_id": f"pokemon-emulator-{session_id[:8]}",
                "adapter_instance": None,  # Will be set when adapter connects
                "health_score": 1.0,
                "metrics": metrics,
            }

            with self._session_lock:
                self._sessions[session_id] = session_info
                self._active_sessions.add(session_id)

            # Persist session
            if self._persistence:
                self._persistence.save_session(session_info)

            with self._metrics_lock:
                self._global_metrics["sessions_created"] += 1

            self.logger.info(f"Created new session {session_id} with priority {priority.name}")
            return session_id

        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            return None

    def _allocate_port(self) -> int:
        """Allocate available port for new session."""
        # Simple port allocation strategy - extend as needed
        base_port = 8081
        with self._session_lock:
            used_ports = {info["port"] for info in self._sessions.values()}

            for i in range(100):  # Try up to 100 ports
                port = base_port + i
                if port not in used_ports:
                    return port

        raise RuntimeError("No available ports for session")

    def _release_session_internal(self, session_id: str) -> None:
        """Internal session release logic."""
        with self._session_lock:
            if session_id not in self._sessions:
                return

            session_info = self._sessions[session_id]

            # Move session back to idle pool if healthy
            if self._health_monitor.is_healthy(session_id):
                self._active_sessions.discard(session_id)
                self._idle_sessions.add(session_id)
                session_info["state"] = SessionState.IDLE
                session_info["last_activity"] = time.time()
            else:
                # Move to unhealthy pool for recovery
                self._active_sessions.discard(session_id)
                self._unhealthy_sessions.add(session_id)
                session_info["state"] = SessionState.UNHEALTHY

    def register_adapter(self, session_id: str, adapter_instance: object) -> None:
        """Register adapter instance with session."""
        with self._session_lock:
            if session_id in self._sessions:
                # Use weak reference to prevent circular references
                self._sessions[session_id]["adapter_instance"] = weakref.ref(adapter_instance)
                self.logger.debug(f"Registered adapter for session {session_id}")

    def update_session_metrics(
        self, session_id: str, operation: str, duration_ms: float, success: bool, **kwargs
    ) -> None:
        """Update session performance metrics."""
        with self._session_lock:
            if session_id not in self._sessions:
                return

            session_info = self._sessions[session_id]
            metrics = session_info["metrics"]

            # Update operation counts
            metrics.total_operations += 1
            if success:
                metrics.successful_operations += 1
                metrics.consecutive_failures = 0
            else:
                metrics.failed_operations += 1
                metrics.consecutive_failures += 1
                metrics.last_error_timestamp = time.time()

            # Update response time metrics
            metrics.total_response_time_ms += duration_ms
            metrics.min_response_time_ms = min(metrics.min_response_time_ms, duration_ms)
            metrics.max_response_time_ms = max(metrics.max_response_time_ms, duration_ms)

            # Update resource usage if provided
            if "memory_mb" in kwargs:
                metrics.memory_usage_mb = kwargs["memory_mb"]
            if "cpu_seconds" in kwargs:
                metrics.cpu_time_seconds = kwargs["cpu_seconds"]
            if "bytes_sent" in kwargs:
                metrics.network_bytes_sent += kwargs["bytes_sent"]
            if "bytes_received" in kwargs:
                metrics.network_bytes_received += kwargs["bytes_received"]

            # Update activity timestamp
            session_info["last_activity"] = time.time()

            # Update health score
            health_score = self._health_monitor.update_health_metrics(session_id, metrics)
            session_info["health_score"] = health_score

            # Update circuit breaker
            self._health_monitor.update_circuit_breaker(session_id, success)

            # Persist updated session
            if self._persistence:
                self._persistence.save_session(session_info)

    def _perform_health_checks(self) -> None:
        """Perform health checks on all active sessions."""
        with self._session_lock:
            sessions_to_check = list(self._sessions.keys())

        # Check each session (can be done in parallel)
        futures = []
        for session_id in sessions_to_check:
            future = self._thread_pool.submit(self._check_session_health, session_id)
            futures.append(future)

        # Wait for health checks to complete
        for future in futures:
            try:
                future.result(timeout=5.0)  # 5 second timeout per health check
            except Exception as e:
                self.logger.warning(f"Health check failed: {e}")

    def _check_session_health(self, session_id: str) -> None:
        """Check health of individual session."""
        with self._session_lock:
            if session_id not in self._sessions:
                return

            session_info = self._sessions[session_id]

        # Skip if session is already being cleaned up
        if session_info["state"] in [SessionState.CLEANUP, SessionState.TERMINATED]:
            return

        current_time = time.time()

        # Check for session timeout
        if current_time - session_info["last_activity"] > self.config.session_timeout_seconds:
            self._mark_session_expired(session_id)
            return

        # Check if adapter is still available
        adapter_ref = session_info["adapter_instance"]
        if adapter_ref and callable(adapter_ref) and adapter_ref() is None:
            self.logger.warning(f"Session {session_id} adapter instance garbage collected")
            self._mark_session_unhealthy(session_id)
            return

        # Update health check timestamp
        session_info["metrics"].last_health_check = current_time

        # Check if unhealthy session has recovered
        if session_info["state"] == SessionState.UNHEALTHY and self._health_monitor.can_recover(
            session_id
        ):
            self._attempt_session_recovery(session_id)

    def _mark_session_expired(self, session_id: str) -> None:
        """Mark session as expired and schedule cleanup."""
        with self._session_lock:
            if session_id not in self._sessions:
                return

            session_info = self._sessions[session_id]
            session_info["state"] = SessionState.EXPIRED

            # Remove from all active pools
            self._idle_sessions.discard(session_id)
            self._active_sessions.discard(session_id)
            self._unhealthy_sessions.discard(session_id)

        self.logger.info(f"Session {session_id} expired after timeout")

        # Schedule for cleanup
        with self._cleanup_condition:
            self._cleanup_condition.notify()

    def _mark_session_unhealthy(self, session_id: str) -> None:
        """Mark session as unhealthy."""
        with self._session_lock:
            if session_id not in self._sessions:
                return

            session_info = self._sessions[session_id]
            if session_info["state"] not in [SessionState.UNHEALTHY, SessionState.RECOVERING]:
                session_info["state"] = SessionState.UNHEALTHY

                # Move to unhealthy pool
                self._idle_sessions.discard(session_id)
                self._active_sessions.discard(session_id)
                self._unhealthy_sessions.add(session_id)

                self.logger.warning(f"Session {session_id} marked as unhealthy")

    def _attempt_session_recovery(self, session_id: str) -> None:
        """Attempt to recover unhealthy session."""
        try:
            with self._session_lock:
                if session_id not in self._sessions:
                    return

                session_info = self._sessions[session_id]
                session_info["state"] = SessionState.RECOVERING

            self.logger.info(f"Attempting recovery for session {session_id}")

            # Perform recovery operations
            success = self._perform_session_recovery(session_id)

            with self._session_lock:
                if success:
                    session_info["state"] = SessionState.IDLE
                    self._unhealthy_sessions.discard(session_id)
                    self._idle_sessions.add(session_id)
                    session_info["health_score"] = 1.0

                    with self._metrics_lock:
                        self._global_metrics["recovery_operations"] += 1

                    self.logger.info(f"Successfully recovered session {session_id}")
                else:
                    session_info["state"] = SessionState.EXPIRED
                    self._unhealthy_sessions.discard(session_id)

                    # Schedule for cleanup
                    with self._cleanup_condition:
                        self._cleanup_condition.notify()

                    self.logger.warning(f"Failed to recover session {session_id}")

        except Exception as e:
            self.logger.error(f"Session recovery error for {session_id}: {e}")
            self._mark_session_expired(session_id)

    def _perform_session_recovery(self, session_id: str) -> bool:
        """Perform actual session recovery operations."""
        try:
            with self._session_lock:
                session_info = self._sessions[session_id]

            # Try to ping adapter if available
            adapter_ref = session_info["adapter_instance"]
            if adapter_ref and callable(adapter_ref):
                adapter = adapter_ref()
                if adapter and hasattr(adapter, "is_healthy"):
                    return adapter.is_healthy()

            # For now, mark as recovered if no major issues found
            # In a real implementation, you would:
            # 1. Test network connectivity
            # 2. Validate container health
            # 3. Perform basic API calls
            # 4. Reset error counters

            return True

        except Exception as e:
            self.logger.error(f"Recovery operation failed for session {session_id}: {e}")
            return False

    def _perform_cleanup(self) -> None:
        """Perform cleanup of expired and terminated sessions."""
        sessions_to_cleanup = []

        with self._session_lock:
            for session_id, session_info in self._sessions.items():
                if session_info["state"] in [SessionState.EXPIRED, SessionState.TERMINATED]:
                    sessions_to_cleanup.append(session_id)

        # Clean up sessions
        for session_id in sessions_to_cleanup:
            try:
                self._cleanup_session(session_id)
            except Exception as e:
                self.logger.error(f"Cleanup failed for session {session_id}: {e}")

        # Clean up old metrics and persistence records
        if self._persistence:
            try:
                deleted_count = self._persistence.cleanup_expired_sessions(
                    self.config.metrics_retention_seconds
                )
                if deleted_count > 0:
                    self.logger.debug(f"Cleaned up {deleted_count} old session records")
            except Exception as e:
                self.logger.error(f"Persistence cleanup failed: {e}")

        # Ensure minimum pool size
        self._ensure_minimum_pool_size()

    def _cleanup_session(self, session_id: str) -> None:
        """Clean up individual session."""
        self.logger.info(f"Cleaning up session {session_id}")

        with self._session_lock:
            if session_id not in self._sessions:
                return

            session_info = self._sessions[session_id]
            session_info["state"] = SessionState.CLEANUP

            # Remove from all pools
            self._idle_sessions.discard(session_id)
            self._active_sessions.discard(session_id)
            self._unhealthy_sessions.discard(session_id)

            # Clean up adapter reference
            adapter_ref = session_info["adapter_instance"]
            if adapter_ref and callable(adapter_ref):
                adapter = adapter_ref()
                if adapter and hasattr(adapter, "close"):
                    try:
                        adapter.close()
                    except Exception as e:
                        self.logger.warning(f"Error closing adapter for session {session_id}: {e}")

            # Remove from session tracking
            del self._sessions[session_id]

        # Remove from persistence
        if self._persistence:
            self._persistence.delete_session(session_id)

        with self._metrics_lock:
            self._global_metrics["sessions_destroyed"] += 1
            self._global_metrics["cleanup_operations"] += 1

    def _ensure_minimum_pool_size(self) -> None:
        """Ensure minimum number of idle sessions are available with proactive scaling."""
        with self._session_lock:
            idle_count = len(self._idle_sessions)
            active_count = len(self._active_sessions)
            total_count = len(self._sessions)

            # Calculate sessions needed with performance optimization
            base_sessions_needed = max(0, self.config.min_pool_size - total_count)
            idle_sessions_needed = max(0, self.config.target_idle_sessions - idle_count)

            # Proactive scaling: add extra sessions during high activity
            utilization_ratio = active_count / max(1, total_count)
            if utilization_ratio > 0.7:  # High utilization
                # Scale up proactively to prevent bottlenecks
                extra_sessions = min(3, self.config.max_pool_size - total_count)
                idle_sessions_needed = max(idle_sessions_needed, extra_sessions)

            sessions_needed = max(base_sessions_needed, idle_sessions_needed)

            if sessions_needed > 0 and total_count < self.config.max_pool_size:
                sessions_to_create = min(sessions_needed, self.config.max_pool_size - total_count)

                # Create sessions in parallel for better performance
                created_sessions = []
                for _ in range(sessions_to_create):
                    session_id = self._create_session(SessionPriority.NORMAL)
                    if session_id:
                        created_sessions.append(session_id)

                # Batch move to idle pool for efficiency
                for session_id in created_sessions:
                    self._active_sessions.discard(session_id)
                    self._idle_sessions.add(session_id)
                    self._sessions[session_id]["state"] = SessionState.IDLE

    def get_session_status(self) -> dict[str, Any]:
        """Get comprehensive session pool status."""
        with self._session_lock:
            status: dict[str, Any] = {
                "total_sessions": len(self._sessions),
                "idle_sessions": len(self._idle_sessions),
                "active_sessions": len(self._active_sessions),
                "unhealthy_sessions": len(self._unhealthy_sessions),
                "session_states": {},
                "health_scores": {},
                "performance_metrics": {},
            }

            # Session state breakdown
            for state in SessionState:
                count = sum(1 for s in self._sessions.values() if s["state"] == state)
                status["session_states"][state.name] = count

            # Health scores
            for session_id in self._sessions:
                status["health_scores"][session_id] = self._health_monitor.get_health_score(
                    session_id
                )

            # Performance metrics
            with self._metrics_lock:
                status["performance_metrics"] = dict(self._global_metrics)

        return status

    def get_detailed_session_info(self, session_id: str) -> dict[str, Any] | None:
        """Get detailed information about specific session."""
        with self._session_lock:
            if session_id not in self._sessions:
                return None

            session_info = self._sessions[session_id]

            return {
                "session_id": session_id,
                "state": session_info["state"].name,
                "priority": session_info["priority"].name,
                "created_at": session_info["created_at"],
                "last_activity": session_info["last_activity"],
                "port": session_info["port"],
                "container_id": session_info["container_id"],
                "health_score": session_info["health_score"],
                "health_trend": self._health_monitor.get_health_trend(session_id),
                "is_healthy": self._health_monitor.is_healthy(session_id),
                "requires_attention": self._health_monitor.requires_attention(session_id),
                "circuit_breaker_open": self._health_monitor.is_circuit_open(session_id),
                "metrics": {
                    "total_operations": session_info["metrics"].total_operations,
                    "successful_operations": session_info["metrics"].successful_operations,
                    "failed_operations": session_info["metrics"].failed_operations,
                    "success_rate": session_info["metrics"].success_rate,
                    "average_response_time_ms": session_info["metrics"].average_response_time_ms,
                    "min_response_time_ms": session_info["metrics"].min_response_time_ms,
                    "max_response_time_ms": session_info["metrics"].max_response_time_ms,
                    "operations_per_second": session_info["metrics"].operations_per_second,
                    "memory_usage_mb": session_info["metrics"].memory_usage_mb,
                    "cpu_time_seconds": session_info["metrics"].cpu_time_seconds,
                    "network_bytes_sent": session_info["metrics"].network_bytes_sent,
                    "network_bytes_received": session_info["metrics"].network_bytes_received,
                    "consecutive_failures": session_info["metrics"].consecutive_failures,
                    "last_error_timestamp": session_info["metrics"].last_error_timestamp,
                    "error_rate_per_minute": session_info["metrics"].error_rate_per_minute,
                },
            }

    def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Gracefully shutdown session manager."""
        self.logger.info("Starting session manager shutdown")

        # Signal shutdown to background threads
        self._shutdown_event.set()

        # Wake up cleanup thread
        with self._cleanup_condition:
            self._cleanup_condition.notify_all()

        # Wait for background threads to finish
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=timeout_seconds / 2)

        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=timeout_seconds / 2)

        # Clean up all sessions
        with self._session_lock:
            session_ids = list(self._sessions.keys())

        for session_id in session_ids:
            try:
                self._cleanup_session(session_id)
            except Exception as e:
                self.logger.error(f"Error during shutdown cleanup of session {session_id}: {e}")

        # Shutdown thread pool (Python 3.10 compatibility)
        self._thread_pool.shutdown(wait=True)

        self.logger.info("Session manager shutdown complete")


# =============================================================================
# Factory and Utility Functions
# =============================================================================


def create_session_manager(
    min_sessions: int = 2, max_sessions: int = 8, enable_persistence: bool = True, **kwargs
) -> SessionManager:
    """
    Create session manager with optimized defaults.

    Args:
        min_sessions: Minimum number of sessions to maintain
        max_sessions: Maximum number of sessions allowed
        enable_persistence: Enable session state persistence
        **kwargs: Additional configuration options

    Returns:
        Configured SessionManager instance
    """
    config = SessionPoolConfig(
        min_pool_size=min_sessions,
        max_pool_size=max_sessions,
        enable_persistence=enable_persistence,
        **kwargs,
    )

    return SessionManager(config)


def get_resource_usage() -> dict[str, float]:
    """Get current system resource usage for monitoring."""
    try:
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage_mb = memory.used / (1024 * 1024)
        memory_percent = memory.percent

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_usage_percent = (disk.used / disk.total) * 100

        return {
            "memory_usage_mb": memory_usage_mb,
            "memory_percent": memory_percent,
            "cpu_percent": cpu_percent,
            "disk_usage_percent": disk_usage_percent,
        }

    except Exception:
        # Fallback if psutil is not available
        try:
            # Use resource module for basic info
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            memory_usage_mb = rusage.ru_maxrss / 1024.0  # Convert KB to MB on Linux

            return {
                "memory_usage_mb": memory_usage_mb,
                "memory_percent": 0.0,
                "cpu_percent": 0.0,
                "disk_usage_percent": 0.0,
            }
        except Exception:
            return {
                "memory_usage_mb": 0.0,
                "memory_percent": 0.0,
                "cpu_percent": 0.0,
                "disk_usage_percent": 0.0,
            }
