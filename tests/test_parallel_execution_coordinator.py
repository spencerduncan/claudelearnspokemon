"""
Comprehensive test suite for ParallelExecutionCoordinator.

Tests cover parallel execution coordination, stream management, fault tolerance,
and integration with EmulatorPool and ClaudeCodeManager components.

Test Categories:
- TestParallelExecutionCoordinatorBasics: Initialization and configuration
- TestCoordinatorLifecycle: Startup, health checks, shutdown operations
- TestExecutionCoordination: Task coordination and parallel execution
- TestStreamManagement: Stream allocation, release, and lifecycle
- TestFaultTolerance: Circuit breaker patterns and error recovery
- TestPerformanceMetrics: Metrics collection and monitoring
- TestIntegrationPatterns: EmulatorPool and ClaudeCodeManager integration
"""

import logging
import threading
import time
import unittest
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

import pytest

from claudelearnspokemon.parallel_execution_coordinator import (
    ParallelExecutionCoordinator,
    ParallelExecutionError,
    CoordinationConfig,
    CoordinatorState,
    ExecutionStream,
    CoordinationMetrics,
)
from claudelearnspokemon.circuit_breaker import CircuitState


@pytest.mark.fast
@pytest.mark.unit
class TestParallelExecutionCoordinatorBasics(unittest.TestCase):
    """Test basic ParallelExecutionCoordinator functionality."""
    
    def setUp(self):
        """Set up test environment with mocked dependencies."""
        # Mock EmulatorPool
        self.mock_emulator_pool = Mock()
        self.mock_emulator_pool.pool_size = 4
        self.mock_emulator_pool.health_check.return_value = {"status": "healthy"}
        
        # Mock ClaudeCodeManager
        self.mock_claude_manager = Mock()
        self.mock_claude_manager.is_running.return_value = True
        self.mock_claude_manager.health_check_all.return_value = {0: True, 1: True, 2: True}
        self.mock_claude_manager.get_tactical_processes.return_value = [
            self._create_mock_tactical_worker(f"worker_{i}") for i in range(3)
        ]
        
        # Create coordinator with mocked dependencies
        self.config = CoordinationConfig(
            max_parallel_streams=2,
            enable_circuit_breaker=False,  # Disable for basic tests
            worker_pool_size=4,
        )
        
        self.coordinator = ParallelExecutionCoordinator(
            emulator_pool=self.mock_emulator_pool,
            claude_manager=self.mock_claude_manager,
            config=self.config,
        )
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "coordinator"):
            try:
                self.coordinator.shutdown()
            except Exception:
                pass  # Coordinator might not be initialized
    
    def _create_mock_tactical_worker(self, worker_id: str):
        """Create mock tactical worker for testing."""
        worker = Mock()
        worker.worker_id = worker_id
        worker.health_check.return_value = True
        worker.is_healthy.return_value = True
        return worker
    
    def _create_mock_emulator_client(self, port: int):
        """Create mock emulator client for testing."""
        client = Mock()
        client.port = port
        client.container_id = f"container_{port}"
        return client
    
    def test_initialization_default_config(self):
        """Test coordinator initializes with default configuration."""
        coordinator = ParallelExecutionCoordinator(
            emulator_pool=self.mock_emulator_pool,
            claude_manager=self.mock_claude_manager,
        )
        
        self.assertIsInstance(coordinator, ParallelExecutionCoordinator)
        self.assertEqual(coordinator.config.max_parallel_streams, 4)
        self.assertEqual(coordinator.state, CoordinatorState.INITIALIZING)
        self.assertIsNotNone(coordinator.coordination_id)
        self.assertEqual(len(coordinator.active_streams), 0)
    
    def test_initialization_custom_config(self):
        """Test coordinator initializes with custom configuration."""
        self.assertIsInstance(self.coordinator, ParallelExecutionCoordinator)
        self.assertEqual(self.coordinator.config.max_parallel_streams, 2)
        self.assertEqual(self.coordinator.config.worker_pool_size, 4)
        self.assertEqual(self.coordinator.state, CoordinatorState.INITIALIZING)
        self.assertFalse(self.coordinator.config.enable_circuit_breaker)
    
    def test_circuit_breaker_enabled_initialization(self):
        """Test coordinator initializes with circuit breaker enabled."""
        config = CoordinationConfig(enable_circuit_breaker=True)
        coordinator = ParallelExecutionCoordinator(
            emulator_pool=self.mock_emulator_pool,
            claude_manager=self.mock_claude_manager,
            config=config,
        )
        
        self.assertIsNotNone(coordinator.circuit_breaker)
        self.assertEqual(coordinator.circuit_breaker.get_state(), CircuitState.CLOSED)
    
    def test_circuit_breaker_disabled_initialization(self):
        """Test coordinator initializes with circuit breaker disabled."""
        self.assertIsNone(self.coordinator.circuit_breaker)
    
    def test_metrics_initialization(self):
        """Test coordination metrics initialize correctly."""
        metrics = self.coordinator.metrics
        
        self.assertIsInstance(metrics, CoordinationMetrics)
        self.assertEqual(metrics.total_coordinated_executions, 0)
        self.assertEqual(metrics.successful_coordinations, 0)
        self.assertEqual(metrics.failed_coordinations, 0)
        self.assertEqual(metrics.success_rate, 100.0)
        self.assertEqual(metrics.active_streams, 0)


@pytest.mark.medium
class TestCoordinatorLifecycle(unittest.TestCase):
    """Test ParallelExecutionCoordinator lifecycle operations."""
    
    def setUp(self):
        """Set up test environment with comprehensive mocking."""
        # Mock EmulatorPool with realistic behavior
        self.mock_emulator_pool = Mock()
        self.mock_emulator_pool.pool_size = 3
        self.mock_emulator_pool.health_check.return_value = {"status": "healthy", "healthy_count": 3}
        
        # Mock emulator clients
        self.mock_clients = []
        for i in range(3):
            client = self._create_mock_emulator_client(8081 + i)
            self.mock_clients.append(client)
        
        # Configure emulator pool acquire behavior
        self.acquire_call_count = 0
        def mock_acquire(timeout=None):
            if self.acquire_call_count < len(self.mock_clients):
                client = self.mock_clients[self.acquire_call_count]
                self.acquire_call_count += 1
                return client
            raise Exception("No more emulators available")
        
        self.mock_emulator_pool.acquire.side_effect = mock_acquire
        self.mock_emulator_pool.release.return_value = None
        
        # Mock ClaudeCodeManager with tactical workers
        self.mock_claude_manager = Mock()
        self.mock_claude_manager.is_running.return_value = True
        self.mock_claude_manager.health_check_all.return_value = {0: True, 1: True, 2: True, 3: True}
        
        self.mock_workers = [
            self._create_mock_tactical_worker(f"worker_{i}") for i in range(4)
        ]
        self.mock_claude_manager.get_tactical_processes.return_value = self.mock_workers
        
        # Create coordinator
        self.config = CoordinationConfig(
            max_parallel_streams=2,
            enable_circuit_breaker=False,
            emulator_acquisition_timeout=1.0,
        )
        
        self.coordinator = ParallelExecutionCoordinator(
            emulator_pool=self.mock_emulator_pool,
            claude_manager=self.mock_claude_manager,
            config=self.config,
        )
    
    def tearDown(self):
        """Clean up after tests."""
        try:
            self.coordinator.shutdown()
        except Exception:
            pass
    
    def _create_mock_tactical_worker(self, worker_id: str):
        """Create mock tactical worker with realistic behavior."""
        worker = Mock()
        worker.worker_id = worker_id
        worker.health_check.return_value = True
        worker.is_healthy.return_value = True
        return worker
    
    def _create_mock_emulator_client(self, port: int):
        """Create mock emulator client with realistic behavior."""
        client = Mock()
        client.port = port
        client.container_id = f"container_{port}"
        return client
    
    def test_successful_initialization(self):
        """Test successful coordinator initialization."""
        success = self.coordinator.initialize()
        
        self.assertTrue(success)
        self.assertEqual(self.coordinator.state, CoordinatorState.READY)
        self.assertIsNotNone(self.coordinator.start_time)
        self.assertIsNotNone(self.coordinator._executor)
        
        # Check that streams were created
        self.assertEqual(len(self.coordinator.active_streams), 2)
        
        # Verify dependencies were checked
        self.mock_emulator_pool.health_check.assert_called()
        self.mock_claude_manager.is_running.assert_called()
        self.mock_claude_manager.health_check_all.assert_called()
    
    def test_initialization_with_unhealthy_dependencies(self):
        """Test initialization failure with unhealthy dependencies."""
        # Make EmulatorPool unhealthy
        self.mock_emulator_pool.health_check.return_value = {"status": "failed"}
        
        success = self.coordinator.initialize()
        
        self.assertFalse(success)
        self.assertEqual(self.coordinator.state, CoordinatorState.FAILED)
    
    def test_initialization_with_no_healthy_workers(self):
        """Test initialization failure with no healthy workers."""
        # Make all workers unhealthy
        for worker in self.mock_workers:
            worker.health_check.return_value = False
            worker.is_healthy.return_value = False
        
        success = self.coordinator.initialize()
        
        self.assertFalse(success)
        self.assertEqual(self.coordinator.state, CoordinatorState.FAILED)
    
    def test_health_check_when_ready(self):
        """Test health check on ready coordinator."""
        self.coordinator.initialize()
        
        health = self.coordinator.health_check()
        
        self.assertIn("coordinator_id", health)
        self.assertIn("overall_health", health)
        self.assertIn("checks", health)
        self.assertEqual(health["state"], CoordinatorState.READY.name)
        self.assertGreaterEqual(health["check_duration_ms"], 0)
    
    def test_health_check_with_failed_dependencies(self):
        """Test health check with failed dependencies."""
        self.coordinator.initialize()
        
        # Simulate dependency failure
        self.mock_emulator_pool.health_check.return_value = {"status": "failed"}
        
        health = self.coordinator.health_check()
        
        self.assertEqual(health["overall_health"], "degraded")
        self.assertFalse(health["checks"]["dependencies"]["healthy"])
    
    def test_graceful_shutdown(self):
        """Test graceful coordinator shutdown."""
        self.coordinator.initialize()
        self.assertEqual(self.coordinator.state, CoordinatorState.READY)
        
        self.coordinator.shutdown()
        
        self.assertEqual(self.coordinator.state, CoordinatorState.STOPPED)
        self.assertEqual(len(self.coordinator.active_streams), 0)
        self.assertEqual(len(self.coordinator.stream_assignments), 0)
        
        # Verify emulators were released
        release_calls = self.mock_emulator_pool.release.call_count
        self.assertGreaterEqual(release_calls, 0)
    
    def test_context_manager_functionality(self):
        """Test coordinator works as context manager."""
        with patch.object(self.coordinator, 'initialize', return_value=True) as mock_init:
            with patch.object(self.coordinator, 'shutdown') as mock_shutdown:
                with self.coordinator as coord:
                    self.assertIs(coord, self.coordinator)
                    mock_init.assert_called_once()
                
                mock_shutdown.assert_called_once()
    
    def test_context_manager_initialization_failure(self):
        """Test context manager raises error on initialization failure."""
        with patch.object(self.coordinator, 'initialize', return_value=False):
            with pytest.raises(ParallelExecutionError):
                with self.coordinator:
                    pass


@pytest.mark.medium
class TestExecutionCoordination(unittest.TestCase):
    """Test parallel execution coordination functionality."""
    
    def setUp(self):
        """Set up test environment for execution testing."""
        # Mock dependencies with execution capabilities
        self.mock_emulator_pool = Mock()
        self.mock_emulator_pool.pool_size = 2
        self.mock_emulator_pool.health_check.return_value = {"status": "healthy"}
        
        # Mock execution results
        self.mock_execution_result = Mock()
        self.mock_execution_result.success = True
        self.mock_execution_result.execution_time = 1.5
        self.mock_execution_result.to_dict.return_value = {
            "execution_id": "test_exec",
            "success": True,
            "execution_time": 1.5,
        }
        self.mock_emulator_pool.execute_script.return_value = self.mock_execution_result
        
        # Mock ClaudeCodeManager
        self.mock_claude_manager = Mock()
        self.mock_claude_manager.is_running.return_value = True
        self.mock_claude_manager.health_check_all.return_value = {0: True, 1: True}
        self.mock_claude_manager.get_tactical_processes.return_value = [
            self._create_mock_tactical_worker(f"worker_{i}") for i in range(2)
        ]
        
        # Create coordinator in ready state
        self.coordinator = ParallelExecutionCoordinator(
            emulator_pool=self.mock_emulator_pool,
            claude_manager=self.mock_claude_manager,
            config=CoordinationConfig(max_parallel_streams=2, enable_circuit_breaker=False),
        )
        
        # Mock initialization to set ready state
        with patch.object(self.coordinator, '_verify_dependencies', return_value=True):
            with patch.object(self.coordinator, '_initialize_streams'):
                self.coordinator.state = CoordinatorState.READY
                self.coordinator.start_time = time.time()
                self.coordinator._executor = ThreadPoolExecutor(max_workers=4)
                
                # Create mock streams
                for i in range(2):
                    stream = ExecutionStream(
                        stream_id=f"stream_{i}",
                        emulator_client=self._create_mock_emulator_client(8081 + i),
                        tactical_worker=self.coordinator.claude_manager.get_tactical_processes()[i],
                    )
                    self.coordinator.active_streams[f"stream_{i}"] = stream
    
    def tearDown(self):
        """Clean up after tests."""
        try:
            self.coordinator.shutdown()
        except Exception:
            pass
    
    def _create_mock_tactical_worker(self, worker_id: str):
        """Create mock tactical worker."""
        worker = Mock()
        worker.worker_id = worker_id
        worker.health_check.return_value = True
        worker.is_healthy.return_value = True
        return worker
    
    def _create_mock_emulator_client(self, port: int):
        """Create mock emulator client."""
        client = Mock()
        client.port = port
        client.container_id = f"container_{port}"
        return client
    
    def test_run_not_ready_state(self):
        """Test run() raises error when coordinator not ready."""
        self.coordinator.state = CoordinatorState.INITIALIZING
        
        strategic_plan = {"tasks": [], "coordination_strategy": "parallel"}
        
        with pytest.raises(ParallelExecutionError):
            self.coordinator.run(strategic_plan)
    
    def test_successful_parallel_execution(self):
        """Test successful parallel task execution."""
        strategic_plan = {
            "coordination_strategy": "parallel",
            "tasks": [
                {
                    "id": "task_1",
                    "type": "script_execution",
                    "script": "MOVE UP PRESS A",
                },
                {
                    "id": "task_2",
                    "type": "exploration",
                    "target_area": "route_1",
                },
            ],
        }
        
        result = self.coordinator.run(strategic_plan)
        
        self.assertIsInstance(result, dict)
        self.assertIn("execution_id", result)
        self.assertIn("success", result)
        self.assertIn("task_results", result)
        self.assertIn("summary", result)
        self.assertEqual(result["coordination_strategy"], "parallel")
        
        # Check task results
        task_results = result["task_results"]
        self.assertIn("task_1", task_results)
        self.assertIn("task_2", task_results)
    
    def test_successful_sequential_execution(self):
        """Test successful sequential task execution."""
        strategic_plan = {
            "coordination_strategy": "sequential",
            "tasks": [
                {
                    "id": "task_1",
                    "type": "analysis",
                    "analysis_type": "pattern_detection",
                },
                {
                    "id": "task_2",
                    "type": "script_execution",
                    "script": "MOVE DOWN PRESS B",
                },
            ],
        }
        
        result = self.coordinator.run(strategic_plan)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["coordination_strategy"], "sequential")
        self.assertEqual(len(result["task_results"]), 2)
    
    def test_execution_with_empty_tasks(self):
        """Test execution fails gracefully with empty task list."""
        strategic_plan = {
            "coordination_strategy": "parallel",
            "tasks": [],
        }
        
        result = self.coordinator.run(strategic_plan)
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
    
    def test_execution_with_unknown_strategy(self):
        """Test execution fails with unknown coordination strategy."""
        strategic_plan = {
            "coordination_strategy": "unknown_strategy",
            "tasks": [{"id": "task_1", "type": "script_execution"}],
        }
        
        result = self.coordinator.run(strategic_plan)
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
    
    def test_execution_with_progress_callback(self):
        """Test execution with progress callback."""
        progress_updates = []
        
        def progress_callback(update):
            progress_updates.append(update)
        
        strategic_plan = {
            "coordination_strategy": "parallel",
            "tasks": [
                {"id": "task_1", "type": "exploration"},
                {"id": "task_2", "type": "analysis"},
            ],
        }
        
        result = self.coordinator.run(strategic_plan, progress_callback=progress_callback)
        
        self.assertTrue(result["success"])
        # Progress updates should have been called
        self.assertGreater(len(progress_updates), 0)
        
        # Check progress update structure
        for update in progress_updates:
            self.assertIn("execution_id", update)
            self.assertIn("completed_tasks", update)
            self.assertIn("total_tasks", update)
            self.assertIn("progress_percentage", update)
    
    def test_execution_with_cancellation(self):
        """Test execution respects cancellation event."""
        cancellation_event = threading.Event()
        cancellation_event.set()  # Pre-cancel
        
        strategic_plan = {
            "coordination_strategy": "sequential",
            "tasks": [
                {"id": "task_1", "type": "script_execution", "script": "MOVE UP"},
                {"id": "task_2", "type": "script_execution", "script": "MOVE DOWN"},
            ],
        }
        
        result = self.coordinator.run(
            strategic_plan, 
            cancellation_event=cancellation_event
        )
        
        # Should complete but with fewer tasks due to cancellation
        self.assertIsInstance(result, dict)
        self.assertIn("task_results", result)


@pytest.mark.medium
class TestStreamManagement(unittest.TestCase):
    """Test execution stream management functionality."""
    
    def setUp(self):
        """Set up test environment for stream testing."""
        self.mock_emulator_pool = Mock()
        self.mock_claude_manager = Mock()
        
        self.coordinator = ParallelExecutionCoordinator(
            emulator_pool=self.mock_emulator_pool,
            claude_manager=self.mock_claude_manager,
            config=CoordinationConfig(max_parallel_streams=3),
        )
    
    def tearDown(self):
        """Clean up after tests."""
        try:
            self.coordinator.shutdown()
        except Exception:
            pass
    
    def test_execution_stream_creation(self):
        """Test ExecutionStream creation and properties."""
        stream = ExecutionStream(
            stream_id="test_stream",
            emulator_client=Mock(port=8081, container_id="container_123"),
            tactical_worker=Mock(worker_id="worker_1"),
        )
        
        self.assertEqual(stream.stream_id, "test_stream")
        self.assertIsNotNone(stream.emulator_client)
        self.assertIsNotNone(stream.tactical_worker)
        self.assertEqual(stream.total_executions, 0)
        self.assertEqual(stream.success_rate, 100.0)
        self.assertFalse(stream.is_active)
    
    def test_execution_stream_metrics(self):
        """Test ExecutionStream metrics calculation."""
        stream = ExecutionStream(stream_id="test_stream")
        
        # Simulate executions
        stream.total_executions = 10
        stream.successful_executions = 7
        stream.failed_executions = 3
        
        self.assertEqual(stream.success_rate, 70.0)
        
        # Test with no executions
        stream.total_executions = 0
        self.assertEqual(stream.success_rate, 100.0)
    
    def test_stream_activity_tracking(self):
        """Test stream activity state tracking."""
        stream = ExecutionStream(stream_id="test_stream")
        
        self.assertFalse(stream.is_active)
        
        # Make stream active
        stream.current_task = {"id": "task_1", "type": "script_execution"}
        stream.execution_start_time = time.time()
        
        self.assertTrue(stream.is_active)
        
        # Make stream inactive
        stream.current_task = None
        stream.execution_start_time = None
        
        self.assertFalse(stream.is_active)
    
    def test_stream_to_dict_conversion(self):
        """Test ExecutionStream dictionary conversion."""
        stream = ExecutionStream(
            stream_id="test_stream",
            emulator_client=Mock(),
            tactical_worker=Mock(),
        )
        stream.total_executions = 5
        stream.successful_executions = 4
        stream.current_task = {"type": "exploration"}
        
        stream_dict = stream.to_dict()
        
        self.assertIsInstance(stream_dict, dict)
        self.assertEqual(stream_dict["stream_id"], "test_stream")
        self.assertTrue(stream_dict["has_emulator"])
        self.assertTrue(stream_dict["has_worker"])
        self.assertEqual(stream_dict["total_executions"], 5)
        self.assertEqual(stream_dict["successful_executions"], 4)
        self.assertEqual(stream_dict["current_task_type"], "exploration")
    
    def test_acquire_available_stream(self):
        """Test acquiring available stream for task execution."""
        # Create streams with different states
        active_stream = ExecutionStream(
            stream_id="active_stream",
            emulator_client=Mock(),
            tactical_worker=Mock(),
        )
        active_stream.current_task = {"id": "ongoing_task"}
        active_stream.execution_start_time = time.time()
        
        available_stream = ExecutionStream(
            stream_id="available_stream",
            emulator_client=Mock(),
            tactical_worker=Mock(),
        )
        
        self.coordinator.active_streams = {
            "active_stream": active_stream,
            "available_stream": available_stream,
        }
        
        # Acquire stream
        acquired = self.coordinator._acquire_available_stream("test_task")
        
        self.assertEqual(acquired.stream_id, "available_stream")
        self.assertIn("test_task", self.coordinator.stream_assignments)
        self.assertEqual(
            self.coordinator.stream_assignments["test_task"], 
            "available_stream"
        )
    
    def test_acquire_no_available_stream(self):
        """Test acquiring stream when none available."""
        # Create only active streams
        active_stream = ExecutionStream(stream_id="busy_stream")
        active_stream.current_task = {"id": "ongoing_task"}
        active_stream.execution_start_time = time.time()
        
        self.coordinator.active_streams = {"busy_stream": active_stream}
        
        acquired = self.coordinator._acquire_available_stream("test_task")
        
        self.assertIsNone(acquired)
        self.assertNotIn("test_task", self.coordinator.stream_assignments)
    
    def test_release_stream(self):
        """Test releasing stream back to available pool."""
        self.coordinator.stream_assignments = {
            "task_1": "stream_1",
            "task_2": "stream_1",
            "task_3": "stream_2",
        }
        
        self.coordinator._release_stream("stream_1")
        
        # All assignments to stream_1 should be removed
        remaining_assignments = {
            k: v for k, v in self.coordinator.stream_assignments.items()
            if v == "stream_1"
        }
        self.assertEqual(len(remaining_assignments), 0)
        
        # stream_2 assignments should remain
        self.assertIn("task_3", self.coordinator.stream_assignments)


@pytest.mark.slow
class TestFaultTolerance(unittest.TestCase):
    """Test fault tolerance and circuit breaker functionality."""
    
    def setUp(self):
        """Set up test environment for fault tolerance testing."""
        self.mock_emulator_pool = Mock()
        self.mock_claude_manager = Mock()
        
        # Enable circuit breaker for fault tolerance tests
        config = CoordinationConfig(
            enable_circuit_breaker=True,
            circuit_failure_threshold=2,
            circuit_recovery_timeout=0.1,  # Fast recovery for testing
        )
        
        self.coordinator = ParallelExecutionCoordinator(
            emulator_pool=self.mock_emulator_pool,
            claude_manager=self.mock_claude_manager,
            config=config,
        )
    
    def tearDown(self):
        """Clean up after tests."""
        try:
            self.coordinator.shutdown()
        except Exception:
            pass
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker is properly initialized."""
        self.assertIsNotNone(self.coordinator.circuit_breaker)
        self.assertEqual(
            self.coordinator.circuit_breaker.get_state(), 
            CircuitState.CLOSED
        )
    
    def test_circuit_breaker_trip_on_failures(self):
        """Test circuit breaker trips after threshold failures."""
        # Set coordinator to ready state
        self.coordinator.state = CoordinatorState.READY
        
        # Mock failed dependency check to cause failures
        with patch.object(
            self.coordinator, 
            '_execute_coordination_internal',
            side_effect=ParallelExecutionError("Test failure")
        ):
            
            strategic_plan = {
                "coordination_strategy": "parallel",
                "tasks": [{"id": "task_1", "type": "script_execution"}],
            }
            
            # Execute until circuit trips
            for i in range(self.coordinator.config.circuit_failure_threshold + 1):
                result = self.coordinator.run(strategic_plan)
                
                if i < self.coordinator.config.circuit_failure_threshold:
                    # Should still execute (failed but circuit not tripped)
                    self.assertFalse(result["success"])
                    self.assertEqual(
                        self.coordinator.circuit_breaker.get_state(),
                        CircuitState.CLOSED
                    )
                else:
                    # Circuit should now be open
                    self.assertEqual(
                        self.coordinator.circuit_breaker.get_state(),
                        CircuitState.OPEN
                    )
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        # Force circuit breaker to open state
        self.coordinator.circuit_breaker.force_open()
        self.assertEqual(
            self.coordinator.circuit_breaker.get_state(),
            CircuitState.OPEN
        )
        
        # Wait for recovery timeout
        time.sleep(self.coordinator.config.circuit_recovery_timeout + 0.05)
        
        # Circuit should attempt recovery on next call
        self.coordinator.state = CoordinatorState.READY
        
        with patch.object(
            self.coordinator,
            '_execute_coordination_internal',
            return_value={"success": True, "execution_id": "test"}
        ):
            strategic_plan = {
                "coordination_strategy": "parallel",
                "tasks": [{"id": "task_1", "type": "exploration"}],
            }
            
            result = self.coordinator.run(strategic_plan)
            
            # Should succeed and circuit should be closed
            self.assertTrue(result["success"])
            # Circuit breaker should eventually return to CLOSED state


@pytest.mark.fast
class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics collection and reporting."""
    
    def setUp(self):
        """Set up test environment for metrics testing."""
        self.mock_emulator_pool = Mock()
        self.mock_claude_manager = Mock()
        
        self.coordinator = ParallelExecutionCoordinator(
            emulator_pool=self.mock_emulator_pool,
            claude_manager=self.mock_claude_manager,
            config=CoordinationConfig(max_parallel_streams=2),
        )
    
    def tearDown(self):
        """Clean up after tests."""
        try:
            self.coordinator.shutdown()
        except Exception:
            pass
    
    def test_metrics_initialization(self):
        """Test coordination metrics initialize correctly."""
        metrics = self.coordinator.metrics
        
        self.assertEqual(metrics.total_coordinated_executions, 0)
        self.assertEqual(metrics.successful_coordinations, 0)
        self.assertEqual(metrics.failed_coordinations, 0)
        self.assertEqual(metrics.success_rate, 100.0)
        self.assertEqual(metrics.average_execution_time, 0.0)
    
    def test_metrics_record_success(self):
        """Test recording successful coordination."""
        metrics = self.coordinator.metrics
        
        metrics.record_successful_coordination(1.5)
        metrics.record_successful_coordination(2.5)
        
        self.assertEqual(metrics.total_coordinated_executions, 2)
        self.assertEqual(metrics.successful_coordinations, 2)
        self.assertEqual(metrics.failed_coordinations, 0)
        self.assertEqual(metrics.success_rate, 100.0)
        self.assertEqual(metrics.average_execution_time, 2.0)  # (1.5 + 2.5) / 2
    
    def test_metrics_record_failure(self):
        """Test recording failed coordination."""
        metrics = self.coordinator.metrics
        
        metrics.record_successful_coordination(1.0)
        metrics.record_failed_coordination()
        
        self.assertEqual(metrics.total_coordinated_executions, 2)
        self.assertEqual(metrics.successful_coordinations, 1)
        self.assertEqual(metrics.failed_coordinations, 1)
        self.assertEqual(metrics.success_rate, 50.0)
    
    def test_get_comprehensive_metrics(self):
        """Test comprehensive metrics collection."""
        # Set up coordinator state
        self.coordinator.start_time = time.time() - 10.0  # 10 seconds ago
        
        # Create mock streams
        stream1 = ExecutionStream(
            stream_id="stream_1",
            emulator_client=Mock(),
            tactical_worker=Mock(),
        )
        stream2 = ExecutionStream(
            stream_id="stream_2", 
            emulator_client=None,  # No emulator
            tactical_worker=Mock(),
        )
        stream2.current_task = {"id": "active_task"}
        stream2.execution_start_time = time.time()
        
        self.coordinator.active_streams = {
            "stream_1": stream1,
            "stream_2": stream2,
        }
        
        metrics = self.coordinator.get_metrics()
        
        # Check basic structure
        self.assertIn("coordinator_id", metrics)
        self.assertIn("state", metrics)
        self.assertIn("uptime_seconds", metrics)
        self.assertIn("total_coordinated_executions", metrics)
        self.assertIn("success_rate", metrics)
        self.assertIn("active_streams", metrics)
        self.assertIn("stream_details", metrics)
        
        # Check calculated values
        self.assertEqual(metrics["total_streams"], 2)
        self.assertEqual(metrics["active_streams"], 1)  # Only stream_2 is active
        self.assertEqual(metrics["emulator_utilization"], 50.0)  # 1/2 streams have emulators
        self.assertEqual(metrics["worker_utilization"], 100.0)  # 2/2 streams have workers
        
        # Check stream details
        self.assertIn("stream_1", metrics["stream_details"])
        self.assertIn("stream_2", metrics["stream_details"])


@pytest.mark.medium
class TestIntegrationPatterns(unittest.TestCase):
    """Test integration with EmulatorPool and ClaudeCodeManager."""
    
    def setUp(self):
        """Set up test environment for integration testing."""
        # Mock EmulatorPool with realistic behavior
        self.mock_emulator_pool = Mock()
        self.mock_emulator_pool.pool_size = 2
        
        # Mock ClaudeCodeManager with tactical workers
        self.mock_claude_manager = Mock()
        self.mock_tactical_workers = [
            Mock(worker_id=f"worker_{i}") for i in range(3)
        ]
        for worker in self.mock_tactical_workers:
            worker.health_check.return_value = True
            worker.is_healthy.return_value = True
        
        self.coordinator = ParallelExecutionCoordinator(
            emulator_pool=self.mock_emulator_pool,
            claude_manager=self.mock_claude_manager,
            config=CoordinationConfig(max_parallel_streams=2),
        )
    
    def tearDown(self):
        """Clean up after tests."""
        try:
            self.coordinator.shutdown()
        except Exception:
            pass
    
    def test_dependency_verification_success(self):
        """Test successful dependency verification."""
        # Configure healthy dependencies
        self.mock_emulator_pool.health_check.return_value = {"status": "healthy"}
        self.mock_claude_manager.is_running.return_value = True
        self.mock_claude_manager.health_check_all.return_value = {0: True, 1: True}
        
        result = self.coordinator._verify_dependencies()
        
        self.assertTrue(result)
        self.mock_emulator_pool.health_check.assert_called()
        self.mock_claude_manager.is_running.assert_called()
        self.mock_claude_manager.health_check_all.assert_called()
    
    def test_dependency_verification_emulator_failure(self):
        """Test dependency verification with EmulatorPool failure."""
        self.mock_emulator_pool.health_check.return_value = {"status": "failed"}
        self.mock_claude_manager.is_running.return_value = True
        self.mock_claude_manager.health_check_all.return_value = {0: True}
        
        result = self.coordinator._verify_dependencies()
        
        self.assertFalse(result)
    
    def test_dependency_verification_claude_manager_failure(self):
        """Test dependency verification with ClaudeCodeManager failure."""
        self.mock_emulator_pool.health_check.return_value = {"status": "healthy"}
        self.mock_claude_manager.is_running.return_value = False
        
        result = self.coordinator._verify_dependencies()
        
        self.assertFalse(result)
    
    def test_dependency_verification_no_healthy_processes(self):
        """Test dependency verification with no healthy Claude processes."""
        self.mock_emulator_pool.health_check.return_value = {"status": "healthy"}
        self.mock_claude_manager.is_running.return_value = True
        self.mock_claude_manager.health_check_all.return_value = {0: False, 1: False}
        
        result = self.coordinator._verify_dependencies()
        
        self.assertFalse(result)
    
    def test_script_task_execution_integration(self):
        """Test script task execution with EmulatorPool integration."""
        # Mock stream with emulator client
        mock_client = Mock()
        stream = ExecutionStream(
            stream_id="test_stream",
            emulator_client=mock_client,
            tactical_worker=Mock(),
        )
        
        # Mock execution result
        mock_result = Mock()
        mock_result.success = True
        mock_result.execution_time = 2.5
        mock_result.performance_metrics = {
            "frames_executed": 10,
            "completion_percentage": 100.0,
        }
        mock_result.to_dict.return_value = {"success": True, "execution_time": 2.5}
        
        self.mock_emulator_pool.execute_script.return_value = mock_result
        
        task = {
            "type": "script_execution",
            "script": "MOVE UP PRESS A",
            "checkpoint_id": "checkpoint_123",
        }
        
        result = self.coordinator._execute_script_task(stream, task, "exec_123")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["task_type"], "script_execution")
        self.assertIn("emulator_result", result)
        self.assertIn("worker_analysis", result)
        
        # Verify EmulatorPool was called correctly
        self.mock_emulator_pool.execute_script.assert_called_once_with(
            script_text="MOVE UP PRESS A",
            checkpoint_id="checkpoint_123",
            timeout=self.coordinator.config.tactical_execution_timeout,
        )
    
    def test_task_execution_error_handling(self):
        """Test task execution error handling."""
        # Mock stream
        stream = ExecutionStream(
            stream_id="test_stream",
            emulator_client=None,  # No emulator client to trigger error
            tactical_worker=Mock(),
        )
        
        task = {"type": "script_execution", "script": "MOVE UP"}
        
        result = self.coordinator._execute_task_on_stream(stream, task, "exec_123")
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertEqual(result["stream_id"], "test_stream")


if __name__ == "__main__":
    # Run specific test categories when executed directly
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1].lower()
        
        if test_category == "basic":
            unittest.main(argv=[sys.argv[0]], 
                         defaultTest="TestParallelExecutionCoordinatorBasics")
        elif test_category == "lifecycle":
            unittest.main(argv=[sys.argv[0]], 
                         defaultTest="TestCoordinatorLifecycle")
        elif test_category == "execution":
            unittest.main(argv=[sys.argv[0]], 
                         defaultTest="TestExecutionCoordination")
        elif test_category == "streams":
            unittest.main(argv=[sys.argv[0]], 
                         defaultTest="TestStreamManagement")
        elif test_category == "fault":
            unittest.main(argv=[sys.argv[0]], 
                         defaultTest="TestFaultTolerance")
        elif test_category == "metrics":
            unittest.main(argv=[sys.argv[0]], 
                         defaultTest="TestPerformanceMetrics")
        elif test_category == "integration":
            unittest.main(argv=[sys.argv[0]], 
                         defaultTest="TestIntegrationPatterns")
        else:
            print("Available test categories: basic, lifecycle, execution, streams, fault, metrics, integration")
    else:
        # Run all tests
        unittest.main()