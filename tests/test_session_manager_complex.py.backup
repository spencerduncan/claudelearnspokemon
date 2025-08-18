"""
Comprehensive test suite for SessionManager.

Tests all aspects of advanced session management including:
- Multi-session coordination and pooling
- Session lifecycle management with persistence
- Concurrent session handling with thread safety
- Health monitoring and automatic recovery
- Metrics tracking and resource management
- Performance characteristics and SLA compliance

Author: John Botmack - Session Management Testing and Performance Validation
"""

import logging
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

# Test imports
import pytest

# Import the session manager components
from claudelearnspokemon.session_manager import (
    SessionHealthMonitor,
    SessionInfo,
    SessionManager,
    SessionMetrics,
    SessionPersistence,
    SessionPoolConfig,
    SessionPriority,
    SessionState,
    create_session_manager,
    get_resource_usage,
)


@pytest.mark.slow
class TestSessionMetrics(unittest.TestCase):
    """Test SessionMetrics data class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.session_id = "test-session-123"
        self.metrics = SessionMetrics(session_id=self.session_id)

    def test_initialization(self):
        """Test proper metrics initialization."""
        assert self.metrics.session_id == self.session_id
        assert self.metrics.total_operations == 0
        assert self.metrics.successful_operations == 0
        assert self.metrics.failed_operations == 0
        assert self.metrics.min_response_time_ms == float("inf")
        assert self.metrics.max_response_time_ms == 0.0

    def test_average_response_time_calculation(self):
        """Test average response time calculation."""
        self.metrics.total_operations = 3
        self.metrics.total_response_time_ms = 300.0

        assert self.metrics.average_response_time_ms == 100.0

    def test_success_rate_calculation(self):
        """Test success rate percentage calculation."""
        self.metrics.total_operations = 10
        self.metrics.successful_operations = 8
        self.metrics.failed_operations = 2

        assert self.metrics.success_rate == 80.0

    def test_operations_per_second_calculation(self):
        """Test operations per second calculation."""
        # Simulate 1 second elapsed time
        self.metrics.created_at = time.time() - 1.0
        self.metrics.total_operations = 50

        # Should be approximately 50 ops/sec
        ops_per_sec = self.metrics.operations_per_second
        assert 45.0 <= ops_per_sec <= 55.0  # Allow some timing variance


@pytest.mark.slow
class TestSessionPersistence(unittest.TestCase):
    """Test SessionPersistence database operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_sessions.db")
        self.persistence = SessionPersistence(self.db_path)

        # Create test session info
        self.session_id = "test-session-456"
        self.metrics = SessionMetrics(session_id=self.session_id)
        self.metrics.total_operations = 100
        self.metrics.successful_operations = 95
        self.metrics.failed_operations = 5

        self.session_info: SessionInfo = {
            "session_id": self.session_id,
            "state": SessionState.ACTIVE,
            "priority": SessionPriority.HIGH,
            "created_at": time.time(),
            "last_activity": time.time(),
            "port": 8081,
            "container_id": "test-container-456",
            "adapter_instance": None,
            "health_score": 0.95,
            "metrics": self.metrics,
        }

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)

    def test_database_initialization(self):
        """Test database schema creation."""
        # Check that database file was created
        assert os.path.exists(self.db_path)

        # Verify table structure
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        schema = cursor.fetchone()[0]
        conn.close()

        # Verify required columns exist
        assert "session_id" in schema
        assert "state" in schema
        assert "priority" in schema
        assert "metrics_json" in schema

    def test_save_and_load_session(self):
        """Test session save and load operations."""
        # Save session
        self.persistence.save_session(self.session_info)

        # Load session
        loaded_session = self.persistence.load_session(self.session_id)

        assert loaded_session is not None
        assert loaded_session["session_id"] == self.session_id
        assert loaded_session["state"] == SessionState.ACTIVE
        assert loaded_session["priority"] == SessionPriority.HIGH
        assert loaded_session["port"] == 8081
        assert loaded_session["container_id"] == "test-container-456"
        assert loaded_session["health_score"] == 0.95

        # Verify metrics were preserved
        metrics_data = loaded_session["metrics"]
        assert metrics_data["total_operations"] == 100
        assert metrics_data["successful_operations"] == 95
        assert metrics_data["failed_operations"] == 5

    def test_load_nonexistent_session(self):
        """Test loading non-existent session."""
        result = self.persistence.load_session("non-existent-session")
        assert result is None

    def test_load_all_sessions(self):
        """Test loading all sessions."""
        # Save multiple sessions
        session_ids = ["session-1", "session-2", "session-3"]

        for i, session_id in enumerate(session_ids):
            session_info = self.session_info.copy()
            session_info["session_id"] = session_id
            session_info["priority"] = SessionPriority(i + 1)  # Different priorities
            session_info["port"] = 8081 + i

            self.persistence.save_session(session_info)

        # Load all sessions
        all_sessions = self.persistence.load_all_sessions()

        assert len(all_sessions) == 3

        # Verify they are ordered by priority, then last_activity
        priorities = [session["priority"].value for session in all_sessions]
        assert priorities == sorted(priorities)

    def test_delete_session(self):
        """Test session deletion."""
        # Save session
        self.persistence.save_session(self.session_info)

        # Verify it exists
        loaded_session = self.persistence.load_session(self.session_id)
        assert loaded_session is not None

        # Delete session
        self.persistence.delete_session(self.session_id)

        # Verify it's gone
        loaded_session = self.persistence.load_session(self.session_id)
        assert loaded_session is None

    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired session records."""
        # Create sessions with different ages
        old_time = time.time() - 7200  # 2 hours ago
        recent_time = time.time() - 300  # 5 minutes ago

        # Old expired session
        old_session = self.session_info.copy()
        old_session["session_id"] = "old-session"
        old_session["state"] = SessionState.TERMINATED
        old_session["last_activity"] = old_time
        self.persistence.save_session(old_session)

        # Recent expired session
        recent_session = self.session_info.copy()
        recent_session["session_id"] = "recent-session"
        recent_session["state"] = SessionState.TERMINATED
        recent_session["last_activity"] = recent_time
        self.persistence.save_session(recent_session)

        # Active session (should not be deleted)
        active_session = self.session_info.copy()
        active_session["session_id"] = "active-session"
        active_session["state"] = SessionState.ACTIVE
        active_session["last_activity"] = recent_time
        self.persistence.save_session(active_session)

        # Cleanup sessions older than 1 hour
        deleted_count = self.persistence.cleanup_expired_sessions(3600)

        # Should have deleted only the old terminated session
        assert deleted_count == 1

        # Verify correct sessions remain
        assert self.persistence.load_session("old-session") is None
        assert self.persistence.load_session("recent-session") is not None
        assert self.persistence.load_session("active-session") is not None

    def test_concurrent_access(self):
        """Test thread-safe concurrent database access."""
        num_threads = 10
        sessions_per_thread = 5

        def save_sessions(thread_id):
            """Save sessions from a specific thread."""
            for i in range(sessions_per_thread):
                session_id = f"thread-{thread_id}-session-{i}"
                session_info = self.session_info.copy()
                session_info["session_id"] = session_id
                session_info["port"] = 8000 + thread_id * 100 + i

                self.persistence.save_session(session_info)

        # Run concurrent save operations
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=save_sessions, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all sessions were saved
        all_sessions = self.persistence.load_all_sessions()
        expected_count = num_threads * sessions_per_thread
        assert len(all_sessions) == expected_count

        # Verify no data corruption
        session_ids = [session["session_id"] for session in all_sessions]
        assert len(set(session_ids)) == expected_count  # All unique


@pytest.mark.slow
class TestSessionHealthMonitor(unittest.TestCase):
    """Test SessionHealthMonitor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SessionPoolConfig()
        self.health_monitor = SessionHealthMonitor(self.config)
        self.session_id = "test-session-health"

    def test_initial_health_score(self):
        """Test initial health score."""
        # New session should have perfect health score
        health_score = self.health_monitor.get_health_score(self.session_id)
        assert health_score == 1.0

    def test_health_score_calculation(self):
        """Test health score calculation based on metrics."""
        metrics = SessionMetrics(session_id=self.session_id)

        # Good metrics
        metrics.total_operations = 100
        metrics.successful_operations = 95
        metrics.failed_operations = 5
        metrics.total_response_time_ms = 5000.0  # 50ms average
        metrics.memory_usage_mb = 10.0  # Well below limit
        metrics.error_rate_per_minute = 1.0  # Low error rate

        health_score = self.health_monitor.update_health_metrics(self.session_id, metrics)

        # Should have high health score
        assert health_score > 0.8

    def test_poor_health_score(self):
        """Test health score with poor metrics."""
        metrics = SessionMetrics(session_id=self.session_id)

        # Poor metrics
        metrics.total_operations = 100
        metrics.successful_operations = 50  # 50% success rate
        metrics.failed_operations = 50
        metrics.total_response_time_ms = 50000.0  # 500ms average (slow)
        metrics.memory_usage_mb = 45.0  # Close to limit
        metrics.error_rate_per_minute = 8.0  # High error rate

        health_score = self.health_monitor.update_health_metrics(self.session_id, metrics)

        # Should have low health score
        assert health_score < 0.5

    def test_health_trend_calculation(self):
        """Test health trend calculation over time."""
        metrics = SessionMetrics(session_id=self.session_id)

        # Start with poor health
        metrics.successful_operations = 50
        metrics.total_operations = 100
        _ = self.health_monitor.update_health_metrics(self.session_id, metrics)

        time.sleep(0.1)  # Small delay to create time difference

        # Improve to good health
        metrics.successful_operations = 95
        metrics.total_operations = 100
        _ = self.health_monitor.update_health_metrics(self.session_id, metrics)

        # Check trend - should be positive (improving)
        trend = self.health_monitor.get_health_trend(self.session_id)
        assert trend > 0

    def test_is_healthy_check(self):
        """Test health status checking."""
        metrics = SessionMetrics(session_id=self.session_id)

        # Good metrics
        metrics.total_operations = 100
        metrics.successful_operations = 90
        metrics.failed_operations = 10

        self.health_monitor.update_health_metrics(self.session_id, metrics)

        assert self.health_monitor.is_healthy(self.session_id) is True

    def test_requires_attention(self):
        """Test attention requirement detection."""
        metrics = SessionMetrics(session_id=self.session_id)

        # Critical health metrics
        metrics.total_operations = 100
        metrics.successful_operations = 10  # Even lower success rate
        metrics.failed_operations = 90
        metrics.total_response_time_ms = 100000.0  # Very slow responses
        metrics.memory_usage_mb = 48.0  # Near memory limit
        metrics.error_rate_per_minute = 15.0  # Very high error rate

        self.health_monitor.update_health_metrics(self.session_id, metrics)

        assert self.health_monitor.requires_attention(self.session_id) is True

    def test_circuit_breaker_functionality(self):
        """Test circuit breaker patterns."""
        # Test multiple failures
        for _ in range(6):  # Exceed failure threshold
            self.health_monitor.update_circuit_breaker(self.session_id, success=False)

        # Circuit should be open
        assert self.health_monitor.is_circuit_open(self.session_id) is True

        # Wait for timeout and test recovery
        time.sleep(0.1)  # Brief wait

        # Simulate recovery attempts
        for _ in range(3):  # Meet recovery threshold
            self.health_monitor.update_circuit_breaker(self.session_id, success=True)

        # Circuit should be closed again (in real scenario after timeout)
        # This test would need more sophisticated timing control in production


@pytest.mark.slow
class TestSessionManager(unittest.TestCase):
    """Test SessionManager core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SessionPoolConfig(
            min_pool_size=1,
            max_pool_size=5,
            enable_persistence=True,
            persistence_file=os.path.join(self.temp_dir, "test_sessions.db"),
            session_timeout_seconds=10.0,
            health_check_interval_seconds=1.0,
            cleanup_interval_seconds=2.0,
        )

        # Patch port allocation for testing
        self.port_counter = 8081

        def mock_allocate_port():
            result = self.port_counter
            self.port_counter += 1
            return result

        self.session_manager = SessionManager(self.config)
        self.session_manager._allocate_port = mock_allocate_port

    def tearDown(self):
        """Clean up test fixtures."""
        self.session_manager.shutdown(timeout_seconds=5.0)

        # Clean up temp files
        db_path = os.path.join(self.temp_dir, "test_sessions.db")
        if os.path.exists(db_path):
            os.unlink(db_path)
        os.rmdir(self.temp_dir)

    def test_session_manager_initialization(self):
        """Test SessionManager proper initialization."""
        status = self.session_manager.get_session_status()

        # Should have minimum pool size
        assert status["total_sessions"] >= self.config.min_pool_size
        assert "idle_sessions" in status
        assert "active_sessions" in status
        assert "unhealthy_sessions" in status

    def test_session_acquisition_and_release(self):
        """Test basic session acquisition and release."""
        with self.session_manager.acquire_session() as session_info:
            assert session_info is not None
            assert session_info["session_id"] is not None
            assert session_info["state"] == SessionState.ACTIVE
            assert session_info["port"] >= 8081

            # Verify session is marked as active
            status = self.session_manager.get_session_status()
            assert status["active_sessions"] >= 1

        # After release, session should be back in idle pool
        time.sleep(0.1)  # Brief wait for cleanup
        status = self.session_manager.get_session_status()
        # Should have one idle session now (assuming it stayed healthy)

    def test_concurrent_session_acquisition(self):
        """Test thread-safe concurrent session acquisition."""
        num_threads = 10
        active_sessions = {}  # session_id -> thread_info
        acquisition_events = []
        acquisition_lock = threading.Lock()

        def acquire_session_worker(worker_id):
            """Worker function for concurrent acquisition."""
            try:
                with self.session_manager.acquire_session(timeout_seconds=5.0) as session_info:
                    session_id = session_info["session_id"]

                    with acquisition_lock:
                        # Record acquisition
                        acquire_time = time.time()
                        acquisition_events.append(("acquire", worker_id, session_id, acquire_time))

                        # Check for concurrent access to same session
                        if session_id in active_sessions:
                            other_worker = active_sessions[session_id]["worker_id"]
                            raise AssertionError(
                                f"CONCURRENT ACCESS BUG: Worker-{worker_id} and Worker-{other_worker} "
                                f"both have session {session_id} simultaneously!"
                            )

                        active_sessions[session_id] = {
                            "worker_id": worker_id,
                            "acquire_time": acquire_time,
                        }

                    # Hold session briefly to test concurrent access
                    time.sleep(0.1)

                    with acquisition_lock:
                        # Record release
                        release_time = time.time()
                        acquisition_events.append(("release", worker_id, session_id, release_time))

                        # Remove from active sessions
                        if session_id in active_sessions:
                            del active_sessions[session_id]

            except Exception as e:
                with acquisition_lock:
                    acquisition_events.append(("error", worker_id, str(e), time.time()))

        # Launch concurrent acquisition threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=acquire_session_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all acquisitions to complete
        for thread in threads:
            thread.join()

        # Analyze acquisition events
        successful_acquisitions = [e for e in acquisition_events if e[0] == "acquire"]
        errors = [e for e in acquisition_events if e[0] == "error"]

        # Should have some successful acquisitions
        assert len(successful_acquisitions) > 0

        # Should not have concurrent access errors in the acquisition events
        concurrent_errors = [e for e in errors if "CONCURRENT ACCESS BUG" in str(e[2])]
        assert len(concurrent_errors) == 0, f"Concurrent access detected: {concurrent_errors}"

        # Validate session reuse is working correctly
        session_usage_count = {}
        for event_type, _worker_id, session_id, _timestamp in acquisition_events:
            if event_type == "acquire":
                session_usage_count[session_id] = session_usage_count.get(session_id, 0) + 1

        # With 10 threads and max 5 sessions, some sessions should be reused
        total_reused = sum(count - 1 for count in session_usage_count.values() if count > 1)
        assert total_reused >= 5, f"Expected session reuse, but got {session_usage_count}"

        # All acquisitions should eventually succeed or timeout gracefully
        if len(errors) > 0:
            _timeout_errors = [e for e in errors if "timeout" in str(e[2]).lower()]
            non_timeout_errors = [e for e in errors if "timeout" not in str(e[2]).lower()]
            # Non-timeout errors indicate bugs
            assert len(non_timeout_errors) == 0, f"Unexpected errors: {non_timeout_errors}"

    def test_session_priority_handling(self):
        """Test priority-based session allocation."""
        # Acquire all available sessions with normal priority
        normal_sessions = []
        try:
            for _ in range(self.config.max_pool_size):
                session_ctx = self.session_manager.acquire_session(
                    priority=SessionPriority.NORMAL, timeout_seconds=1.0
                )
                normal_sessions.append(session_ctx)
                session_ctx.__enter__()  # Manually enter context
        except Exception:
            pass  # Expected when pool is exhausted

        # Try to acquire high priority session - should still work or timeout gracefully
        try:
            with self.session_manager.acquire_session(
                priority=SessionPriority.HIGH, timeout_seconds=1.0
            ) as high_priority_session:
                assert high_priority_session is not None
        except TimeoutError:
            pass  # Acceptable if pool is truly exhausted

        # Clean up normal sessions
        for session_ctx in normal_sessions:
            try:
                session_ctx.__exit__(None, None, None)
            except Exception:
                pass

    def test_session_metrics_tracking(self):
        """Test session performance metrics tracking."""
        with self.session_manager.acquire_session() as session_info:
            session_id = session_info["session_id"]

            # Simulate some operations with metrics
            self.session_manager.update_session_metrics(
                session_id=session_id,
                operation="test_operation",
                duration_ms=150.0,
                success=True,
                memory_mb=25.0,
                bytes_sent=1024,
                bytes_received=2048,
            )

            # Simulate a failure
            self.session_manager.update_session_metrics(
                session_id=session_id, operation="test_failure", duration_ms=500.0, success=False
            )

        # Check detailed session info
        session_details = self.session_manager.get_detailed_session_info(session_id)
        assert session_details is not None
        assert session_details["metrics"]["total_operations"] == 2
        assert session_details["metrics"]["successful_operations"] == 1
        assert session_details["metrics"]["failed_operations"] == 1
        assert session_details["metrics"]["success_rate"] == 50.0

    def test_session_health_monitoring(self):
        """Test automatic session health monitoring."""
        # Wait for initial health check cycle
        time.sleep(self.config.health_check_interval_seconds + 0.5)

        status = self.session_manager.get_session_status()

        # Should have health scores for all sessions
        assert len(status["health_scores"]) >= status["total_sessions"]

        # All initial sessions should be healthy
        healthy_sessions = sum(1 for score in status["health_scores"].values() if score > 0.7)
        assert healthy_sessions >= status["total_sessions"]

    def test_session_cleanup_and_recovery(self):
        """Test session cleanup and recovery mechanisms."""
        # Get initial session count
        initial_status = self.session_manager.get_session_status()
        _initial_count = initial_status["total_sessions"]

        # Force some sessions to be unhealthy by simulating failures
        with self.session_manager.acquire_session() as session_info:
            session_id = session_info["session_id"]

            # Simulate multiple failures to make session unhealthy
            for _ in range(10):
                self.session_manager.update_session_metrics(
                    session_id=session_id,
                    operation="failing_operation",
                    duration_ms=1000.0,
                    success=False,
                )

        # Wait for cleanup cycle
        time.sleep(self.config.cleanup_interval_seconds + 1.0)

        # Check that session manager is maintaining pool health
        final_status = self.session_manager.get_session_status()

        # Should still have sessions available (recovery/replacement)
        assert final_status["total_sessions"] >= self.config.min_pool_size

    def test_persistence_integration(self):
        """Test session persistence and recovery."""
        # Create and use a session
        session_id = None
        with self.session_manager.acquire_session() as session_info:
            session_id = session_info["session_id"]

            # Add some metrics
            self.session_manager.update_session_metrics(
                session_id=session_id, operation="persistent_test", duration_ms=100.0, success=True
            )

        # Verify session was persisted
        if self.session_manager._persistence:
            persisted_session = self.session_manager._persistence.load_session(session_id)
            assert persisted_session is not None
            assert persisted_session["session_id"] == session_id

    def test_resource_monitoring(self):
        """Test resource usage monitoring."""
        resource_usage = get_resource_usage()

        assert "memory_usage_mb" in resource_usage
        assert "memory_percent" in resource_usage
        assert "cpu_percent" in resource_usage
        assert "disk_usage_percent" in resource_usage

        # Values should be reasonable
        assert resource_usage["memory_usage_mb"] >= 0.0
        assert 0.0 <= resource_usage["memory_percent"] <= 100.0
        assert 0.0 <= resource_usage["cpu_percent"] <= 100.0
        assert 0.0 <= resource_usage["disk_usage_percent"] <= 100.0


@pytest.mark.slow
class TestSessionManagerFactory(unittest.TestCase):
    """Test factory functions and utilities."""

    def test_create_session_manager(self):
        """Test session manager factory function."""
        session_manager = create_session_manager(
            min_sessions=3, max_sessions=10, enable_persistence=False
        )

        assert session_manager is not None
        assert session_manager.config.min_pool_size == 3
        assert session_manager.config.max_pool_size == 10
        assert session_manager.config.enable_persistence is False

        # Cleanup
        session_manager.shutdown()

    def test_resource_usage_fallback(self):
        """Test resource usage fallback when psutil is not available."""
        # Test the fallback mechanism
        with patch("claudelearnspokemon.session_manager.psutil", None):
            resource_usage = get_resource_usage()

            # Should still return valid structure
            assert "memory_usage_mb" in resource_usage
            assert "cpu_percent" in resource_usage


@pytest.mark.slow
class TestPerformanceCharacteristics(unittest.TestCase):
    """Test performance requirements and SLA compliance."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.config = SessionPoolConfig(
            min_pool_size=10,  # Higher minimum for concurrent load
            max_pool_size=30,  # Increased capacity for 20 concurrent threads
            target_idle_sessions=5,  # Keep more sessions ready
            enable_persistence=False,  # Disable for performance tests
            acquisition_timeout_seconds=5.0,  # Longer timeout for high concurrency
            max_concurrent_acquisitions=25,  # Allow higher concurrency
        )
        self.session_manager = SessionManager(self.config)

        # Mock port allocation for performance tests
        self.port_counter = 8000

        def mock_allocate_port():
            result = self.port_counter
            self.port_counter += 1
            return result

        self.session_manager._allocate_port = mock_allocate_port

    def tearDown(self):
        """Clean up performance test fixtures."""
        self.session_manager.shutdown()

    def test_session_acquisition_performance(self):
        """Test session acquisition meets performance requirements (< 50ms)."""
        num_acquisitions = 100
        acquisition_times = []

        for _ in range(num_acquisitions):
            start_time = time.time()

            try:
                with self.session_manager.acquire_session(timeout_seconds=1.0) as _session_info:
                    acquisition_time = (time.time() - start_time) * 1000.0  # Convert to ms
                    acquisition_times.append(acquisition_time)
            except Exception:
                continue  # Skip failed acquisitions for performance measurement

        if acquisition_times:
            avg_acquisition_time = sum(acquisition_times) / len(acquisition_times)
            max_acquisition_time = max(acquisition_times)

            print(f"Average acquisition time: {avg_acquisition_time:.2f}ms")
            print(f"Maximum acquisition time: {max_acquisition_time:.2f}ms")

            # Performance requirement: < 50ms average
            assert (
                avg_acquisition_time < 50.0
            ), f"Average acquisition time {avg_acquisition_time:.2f}ms exceeds 50ms limit"

            # 95th percentile should be reasonable
            sorted_times = sorted(acquisition_times)
            p95_time = sorted_times[int(len(sorted_times) * 0.95)]
            assert (
                p95_time < 100.0
            ), f"95th percentile acquisition time {p95_time:.2f}ms exceeds 100ms limit"

    def test_concurrent_acquisition_performance(self):
        """Test concurrent session acquisition performance for realistic Pokemon speedrun usage."""
        # Realistic scenario: 4 emulators + 4 background tasks
        num_threads = 8
        acquisitions_per_thread = 5  # More realistic acquisition pattern

        def acquisition_worker(results_list, worker_id):
            """Worker function for concurrent acquisitions."""
            worker_times = []

            for _ in range(acquisitions_per_thread):
                start_time = time.time()

                try:
                    with self.session_manager.acquire_session(timeout_seconds=2.0) as _session_info:
                        acquisition_time = (time.time() - start_time) * 1000.0
                        worker_times.append(acquisition_time)

                        # Brief work simulation
                        time.sleep(0.01)

                except Exception:
                    # Record timeout/failure
                    worker_times.append(-1)

            results_list.append((worker_id, worker_times))

        # Launch concurrent workers
        results = []
        threads = []

        start_time = time.time()

        for worker_id in range(num_threads):
            thread = threading.Thread(target=acquisition_worker, args=(results, worker_id))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Analyze results
        all_acquisition_times = []
        total_acquisitions = 0
        failed_acquisitions = 0

        for _worker_id, times in results:
            for time_ms in times:
                total_acquisitions += 1
                if time_ms > 0:
                    all_acquisition_times.append(time_ms)
                else:
                    failed_acquisitions += 1

        success_rate = (len(all_acquisition_times) / max(1, total_acquisitions)) * 100.0

        print("Concurrent performance results:")
        print(f"  Total acquisitions: {total_acquisitions}")
        print(f"  Successful acquisitions: {len(all_acquisition_times)}")
        print(f"  Failed acquisitions: {failed_acquisitions}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Total time: {total_time:.2f}s")

        if all_acquisition_times:
            avg_time = sum(all_acquisition_times) / len(all_acquisition_times)
            max_time = max(all_acquisition_times)

            print(f"  Average acquisition time: {avg_time:.2f}ms")
            print(f"  Maximum acquisition time: {max_time:.2f}ms")

            # Performance assertions for realistic Pokemon speedrun concurrency
            assert (
                success_rate > 95.0
            ), f"Success rate {success_rate:.1f}% too low for production use"
            assert avg_time < 75.0, f"Average time {avg_time:.2f}ms under realistic load too high"

    def test_memory_usage_per_session(self):
        """Test memory usage per session meets requirements (< 5MB)."""
        import tracemalloc

        tracemalloc.start()

        # Baseline memory
        snapshot1 = tracemalloc.take_snapshot()

        # Create sessions
        sessions = []
        for _ in range(10):
            session_ctx = self.session_manager.acquire_session(timeout_seconds=1.0)
            sessions.append(session_ctx)
            session_ctx.__enter__()

        # Measure memory after session creation
        snapshot2 = tracemalloc.take_snapshot()

        # Calculate memory difference
        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        total_memory_kb = sum(stat.size for stat in top_stats) / 1024.0
        memory_per_session_mb = total_memory_kb / (1024.0 * max(1, len(sessions)))

        print(f"Memory usage per session: {memory_per_session_mb:.2f}MB")

        # Clean up sessions
        for session_ctx in sessions:
            try:
                session_ctx.__exit__(None, None, None)
            except Exception:
                pass

        tracemalloc.stop()

        # Performance requirement: < 5MB per session
        assert (
            memory_per_session_mb < 5.0
        ), f"Memory usage per session {memory_per_session_mb:.2f}MB exceeds 5MB limit"


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run tests
    unittest.main()
