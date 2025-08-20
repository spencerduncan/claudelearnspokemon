"""
Tests for SonnetWorkerPool - Worker Pool Initialization and Management

These tests validate the SonnetWorkerPool component which provides
an abstraction layer over ClaudeCodeManager's tactical processes
for managing Sonnet workers in parallel script development.

Test Focus:
- Worker pool initialization with configurable worker count
- Worker ID assignment and tracking
- Health monitoring and status reporting
- Task assignment and load balancing
- Individual worker lifecycle management

Performance Requirements:
- Worker pool initialization: <500ms
- Worker health checks: <10ms per worker
- Task assignment: <50ms
"""

import time
from unittest.mock import Mock

import pytest

from claudelearnspokemon.claude_code_manager import ClaudeCodeManager
from claudelearnspokemon.claude_process import ClaudeProcess
from claudelearnspokemon.prompts import ProcessType
from claudelearnspokemon.sonnet_worker_pool import SonnetWorkerPool


@pytest.mark.fast
class TestSonnetWorkerPoolInitialization:
    """Test suite for SonnetWorkerPool initialization functionality."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.mock_claude_manager = Mock(spec=ClaudeCodeManager)
        self.sonnet_pool = SonnetWorkerPool(claude_manager=self.mock_claude_manager)

    def test_sonnet_pool_initializes_specified_number_of_workers(self):
        """
        Test that SonnetWorkerPool initializes the specified number of workers.

        This validates the core requirement from Issue #115:
        - Initialize configurable number of Sonnet workers (default 4)
        - Assign unique worker IDs and track worker status
        - Verify worker health and connectivity during initialization
        """
        # Arrange
        worker_count = 4
        mock_processes = []

        # Create mock tactical processes that ClaudeCodeManager would return
        for i in range(worker_count):
            mock_process = Mock(spec=ClaudeProcess)
            mock_process.process_id = i
            mock_process.config = Mock()
            mock_process.config.process_type = ProcessType.SONNET_TACTICAL
            mock_process.is_healthy.return_value = True
            mock_process.health_check.return_value = True
            mock_processes.append(mock_process)

        self.mock_claude_manager.get_tactical_processes.return_value = mock_processes
        self.mock_claude_manager.start_all_processes.return_value = True

        # Act
        start_time = time.time()
        result = self.sonnet_pool.initialize(worker_count)
        initialization_time = time.time() - start_time

        # Assert - Core functionality
        assert result is True, "Worker pool initialization should succeed"
        assert len(self.sonnet_pool.workers) == worker_count, f"Should have {worker_count} workers"

        # Assert - Worker ID assignment
        worker_ids = list(self.sonnet_pool.workers.keys())
        assert len(set(worker_ids)) == worker_count, "All worker IDs should be unique"
        assert all(
            isinstance(worker_id, str) for worker_id in worker_ids
        ), "Worker IDs should be strings"

        # Assert - Worker status tracking
        for worker_id in worker_ids:
            status = self.sonnet_pool.get_worker_status(worker_id)
            assert status is not None, f"Worker {worker_id} should have status"
            assert status["healthy"] is True, f"Worker {worker_id} should be healthy"
            assert "process_id" in status, f"Worker {worker_id} should have process_id"

        # Assert - Health verification during initialization
        assert (
            self.mock_claude_manager.get_tactical_processes.called
        ), "Should get tactical processes"
        for mock_process in mock_processes:
            mock_process.health_check.assert_called_once(), "Should verify health during initialization"

        # Assert - Performance requirement
        assert (
            initialization_time < 0.5
        ), f"Initialization took {initialization_time:.3f}s, should be <500ms"

    def test_sonnet_pool_handles_different_worker_counts(self):
        """Test that SonnetWorkerPool can handle different worker count configurations."""
        test_counts = [1, 2, 3, 4, 8]

        for count in test_counts:
            # Reset for each test
            self.setup_method()

            # Arrange
            mock_processes = []
            for i in range(count):
                mock_process = Mock(spec=ClaudeProcess)
                mock_process.process_id = i
                mock_process.is_healthy.return_value = True
                mock_process.health_check.return_value = True
                mock_processes.append(mock_process)

            self.mock_claude_manager.get_tactical_processes.return_value = mock_processes
            self.mock_claude_manager.start_all_processes.return_value = True

            # Act
            result = self.sonnet_pool.initialize(count)

            # Assert
            assert result is True, f"Should initialize with {count} workers"
            assert len(self.sonnet_pool.workers) == count, f"Should have exactly {count} workers"

    def test_sonnet_pool_default_worker_count_is_four(self):
        """Test that SonnetWorkerPool defaults to 4 workers when no count specified."""
        # Arrange - Set up 4 mock processes (default)
        mock_processes = []
        for i in range(4):
            mock_process = Mock(spec=ClaudeProcess)
            mock_process.process_id = i
            mock_process.is_healthy.return_value = True
            mock_process.health_check.return_value = True
            mock_processes.append(mock_process)

        self.mock_claude_manager.get_tactical_processes.return_value = mock_processes
        self.mock_claude_manager.start_all_processes.return_value = True

        # Act - Call initialize without specifying worker_count
        result = self.sonnet_pool.initialize()

        # Assert
        assert result is True, "Should initialize with default worker count"
        assert len(self.sonnet_pool.workers) == 4, "Default should be 4 workers"

    def test_sonnet_pool_initialization_fails_gracefully(self):
        """Test that SonnetWorkerPool handles initialization failures gracefully."""
        # Arrange - Simulate ClaudeCodeManager failure
        self.mock_claude_manager.start_all_processes.return_value = False
        self.mock_claude_manager.get_tactical_processes.return_value = []

        # Act
        result = self.sonnet_pool.initialize(4)

        # Assert
        assert result is False, "Should return False when initialization fails"
        assert len(self.sonnet_pool.workers) == 0, "Should have no workers on failure"

    def test_sonnet_pool_verifies_worker_health_during_init(self):
        """Test that SonnetWorkerPool verifies worker health during initialization."""
        # Arrange - Mix of healthy and unhealthy processes
        healthy_process = Mock(spec=ClaudeProcess)
        healthy_process.process_id = 0
        healthy_process.is_healthy.return_value = True
        healthy_process.health_check.return_value = True

        unhealthy_process = Mock(spec=ClaudeProcess)
        unhealthy_process.process_id = 1
        unhealthy_process.is_healthy.return_value = False
        unhealthy_process.health_check.return_value = False

        self.mock_claude_manager.get_tactical_processes.return_value = [
            healthy_process,
            unhealthy_process,
        ]
        self.mock_claude_manager.start_all_processes.return_value = True

        # Act
        result = self.sonnet_pool.initialize(2)

        # Assert - Should only include healthy workers
        assert result is True, "Should succeed even with some unhealthy processes"
        assert len(self.sonnet_pool.workers) == 1, "Should only register healthy workers"

        # Find the registered worker
        worker_id = list(self.sonnet_pool.workers.keys())[0]
        status = self.sonnet_pool.get_worker_status(worker_id)
        assert status["healthy"] is True, "Registered worker should be healthy"


@pytest.mark.medium
class TestSonnetWorkerPoolWorkerManagement:
    """Test suite for SonnetWorkerPool worker management functionality."""

    def setup_method(self):
        """Set up test fixtures with initialized worker pool."""
        self.mock_claude_manager = Mock(spec=ClaudeCodeManager)
        self.sonnet_pool = SonnetWorkerPool(claude_manager=self.mock_claude_manager)

        # Set up mock processes
        self.mock_processes = []
        for i in range(4):
            mock_process = Mock(spec=ClaudeProcess)
            mock_process.process_id = i
            mock_process.is_healthy.return_value = True
            mock_process.health_check.return_value = True
            mock_process.send_message.return_value = "Mock response"
            self.mock_processes.append(mock_process)

        self.mock_claude_manager.get_tactical_processes.return_value = self.mock_processes
        self.mock_claude_manager.start_all_processes.return_value = True

        # Initialize the pool
        self.sonnet_pool.initialize(4)

    def test_get_worker_status_returns_detailed_information(self):
        """Test that get_worker_status returns comprehensive worker information."""
        # Act
        worker_id = list(self.sonnet_pool.workers.keys())[0]
        status = self.sonnet_pool.get_worker_status(worker_id)

        # Assert
        assert isinstance(status, dict), "Status should be a dictionary"

        required_fields = ["healthy", "process_id", "worker_id", "status"]
        for field in required_fields:
            assert field in status, f"Status should include '{field}' field"

        assert isinstance(status["healthy"], bool), "healthy should be boolean"
        assert isinstance(status["process_id"], int), "process_id should be integer"
        assert isinstance(status["worker_id"], str), "worker_id should be string"

    def test_get_worker_status_handles_invalid_worker_id(self):
        """Test that get_worker_status handles invalid worker IDs gracefully."""
        # Act
        status = self.sonnet_pool.get_worker_status("invalid_worker_id")

        # Assert
        assert status is None, "Should return None for invalid worker ID"

    def test_restart_worker_functionality(self):
        """Test that restart_worker can restart a specific worker."""
        # Arrange
        worker_id = list(self.sonnet_pool.workers.keys())[0]
        worker_info = self.sonnet_pool.workers[worker_id]
        mock_process = worker_info["process"]
        mock_process.restart.return_value = True

        # Act
        result = self.sonnet_pool.restart_worker(worker_id)

        # Assert
        assert result is True, "Worker restart should succeed"
        mock_process.restart.assert_called_once(), "Should call restart on the process"

    def test_restart_worker_handles_invalid_worker_id(self):
        """Test that restart_worker handles invalid worker IDs gracefully."""
        # Act
        result = self.sonnet_pool.restart_worker("invalid_worker_id")

        # Assert
        assert result is False, "Should return False for invalid worker ID"

    def test_assign_task_returns_valid_worker_id(self):
        """Test that assign_task returns a valid worker ID."""
        # Arrange
        test_task = {"objective": "test objective", "context": "test context"}

        # Act
        worker_id = self.sonnet_pool.assign_task(test_task)

        # Assert
        assert worker_id is not None, "Should return a worker ID"
        assert worker_id in self.sonnet_pool.workers, "Returned worker ID should exist"

    def test_assign_task_handles_no_available_workers(self):
        """Test that assign_task handles the case when no workers are available."""
        # Arrange - Make all workers unhealthy
        for worker_info in self.sonnet_pool.workers.values():
            worker_info["process"].is_healthy.return_value = False
            worker_info["process"].health_check.return_value = False

        test_task = {"objective": "test objective", "context": "test context"}

        # Act
        worker_id = self.sonnet_pool.assign_task(test_task)

        # Assert
        assert worker_id is None, "Should return None when no workers available"
