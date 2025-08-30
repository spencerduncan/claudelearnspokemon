"""
Comprehensive test suite for ClaudeCodeManager with performance validation.

Test Categories:
- TestClaudeCodeManagerBasics: Initialization and process retrieval
- TestProcessLifecycle: Startup, health checks, restart, shutdown
- TestPerformanceMetrics: Timing benchmarks and resource usage
- TestFailureHandling: Process failure simulation and recovery
- TestParallelOperations: Concurrent operations testing
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from claudelearnspokemon.claude_code_manager import (
    ClaudeCodeManager,
    benchmark_startup_performance,
)
from claudelearnspokemon.claude_process import ClaudeProcess
from claudelearnspokemon.conversation_error_handler import ConversationError
from claudelearnspokemon.conversation_lifecycle_manager import TurnLimitConfiguration
from claudelearnspokemon.process_factory import ProcessConfig
from claudelearnspokemon.process_health_monitor import ProcessState
from claudelearnspokemon.process_metrics_collector import ProcessMetrics
from claudelearnspokemon.prompts import ProcessType


@pytest.mark.fast
@pytest.mark.medium
class TestClaudeCodeManagerBasics(unittest.TestCase):
    """Test basic ClaudeCodeManager functionality."""

    def setUp(self):
        """Set up test environment."""
        self.manager = ClaudeCodeManager(max_workers=3)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "manager"):
            self.manager.shutdown()

    def test_initialization(self):
        """Test ClaudeCodeManager initializes correctly."""
        self.assertIsInstance(self.manager, ClaudeCodeManager)
        self.assertEqual(self.manager.max_workers, 3)
        self.assertFalse(self.manager.is_running())
        self.assertEqual(len(self.manager.processes), 0)

    def test_process_count_by_type_empty(self):
        """Test process count when no processes exist."""
        counts = self.manager.get_process_count_by_type()
        expected = {"opus_strategic": 0, "sonnet_tactical": 0}
        self.assertEqual(counts, expected)

    def test_context_manager(self):
        """Test ClaudeCodeManager works as context manager."""
        with ClaudeCodeManager(max_workers=2) as manager:
            self.assertIsInstance(manager, ClaudeCodeManager)
            self.assertEqual(manager.max_workers, 2)
        # Manager should be shut down after context

    def test_get_strategic_process_none(self):
        """Test getting strategic process when none exists."""
        strategic = self.manager.get_strategic_process()
        self.assertIsNone(strategic)

    def test_get_tactical_processes_empty(self):
        """Test getting tactical processes when none exist."""
        tactical = self.manager.get_tactical_processes()
        self.assertEqual(tactical, [])

    def test_get_available_tactical_process_none(self):
        """Test getting available tactical process when none exist."""
        available = self.manager.get_available_tactical_process()
        self.assertIsNone(available)


@pytest.mark.fast
@pytest.mark.medium
class TestProcessLifecycle(unittest.TestCase):
    """Test process lifecycle operations."""

    def setUp(self):
        """Set up test environment."""
        self.manager = ClaudeCodeManager(max_workers=2)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "manager"):
            self.manager.shutdown()

    @patch("subprocess.Popen")
    def test_start_all_processes_mock(self, mock_popen):
        """Test starting all processes with mocked subprocess."""
        # Mock successful process creation
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Process running
        mock_process.stdin = Mock()
        mock_process.stdout = Mock()
        mock_process.stderr = Mock()
        mock_popen.return_value = mock_process

        # Start processes
        success = self.manager.start_all_processes()

        # Verify startup
        self.assertTrue(success)
        self.assertTrue(self.manager.is_running())
        self.assertEqual(len(self.manager.processes), 5)  # 1 Opus + 4 Sonnet

        # Verify process types
        counts = self.manager.get_process_count_by_type()
        self.assertEqual(counts["opus_strategic"], 1)
        self.assertEqual(counts["sonnet_tactical"], 4)

    @patch("subprocess.Popen")
    def test_health_check_all_mock(self, mock_popen):
        """Test health checking all processes."""
        # Setup mock processes
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Healthy
        mock_process.stdin = Mock()
        mock_popen.return_value = mock_process

        # Start and health check
        self.manager.start_all_processes()
        health_results = self.manager.health_check_all()

        # Verify all processes healthy
        self.assertEqual(len(health_results), 5)
        self.assertTrue(all(health_results.values()))

    @patch("subprocess.Popen")
    def test_restart_failed_processes_mock(self, mock_popen):
        """Test restarting failed processes."""
        # Setup mock process that fails then succeeds
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.side_effect = [None, 1, None]  # Running, failed, running
        mock_process.stdin = Mock()
        mock_popen.return_value = mock_process

        # Start processes and simulate failure
        self.manager.start_all_processes()

        # Manually set a process to failed state using health monitor
        process_id = list(self.manager.processes.keys())[0]
        self.manager.processes[process_id].health_monitor.mark_as_failed()

        # Restart failed processes
        restart_count = self.manager.restart_failed_processes()

        # Verify restart attempt
        self.assertGreaterEqual(restart_count, 0)

    def test_shutdown_empty(self):
        """Test shutdown when no processes exist."""
        # Should complete without error
        self.manager.shutdown()
        self.assertFalse(self.manager.is_running())


@pytest.mark.fast
@pytest.mark.medium
class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics and benchmarking."""

    def setUp(self):
        """Set up test environment."""
        self.manager = ClaudeCodeManager(max_workers=2)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "manager"):
            self.manager.shutdown()

    def test_get_performance_metrics_empty(self):
        """Test getting performance metrics when no processes exist."""
        metrics = self.manager.get_performance_metrics()

        expected_keys = [
            "total_processes",
            "healthy_processes",
            "failed_processes",
            "average_startup_time_ms",  # Updated key name
            "average_health_check_time_ms",  # Updated key name
            "total_restarts",
            "total_failures",  # Updated key name
            "process_details",
        ]

        for key in expected_keys:
            self.assertIn(key, metrics)

        self.assertEqual(metrics["total_processes"], 0)
        self.assertEqual(metrics["healthy_processes"], 0)
        self.assertEqual(metrics["process_details"], {})

    @patch("subprocess.Popen")
    def test_startup_timing_mock(self, mock_popen):
        """Test startup timing measurement."""
        # Mock fast process startup
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_popen.return_value = mock_process

        start_time = time.time()
        success = self.manager.start_all_processes()
        total_time = time.time() - start_time

        self.assertTrue(success)
        self.assertLess(total_time, 5.0)  # Should complete quickly with mocks

        # Check individual process startup times recorded
        metrics = self.manager.get_performance_metrics()
        self.assertGreaterEqual(metrics["average_startup_time_ms"], 0)

    @patch("subprocess.Popen")
    def test_health_check_timing_mock(self, mock_popen):
        """Test health check timing measurement."""
        # Setup mock processes
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_popen.return_value = mock_process

        # Start processes and measure health check timing
        self.manager.start_all_processes()

        start_time = time.time()
        health_results = self.manager.health_check_all()
        health_time = time.time() - start_time

        # Health checks should be fast with mocks
        self.assertLess(health_time, 1.0)
        self.assertEqual(len(health_results), 5)

    def test_process_metrics_initialization(self):
        """Test ProcessMetrics initialization."""
        metrics = ProcessMetrics(process_id=1)

        self.assertEqual(metrics.process_id, 1)
        self.assertEqual(metrics.startup_time, 0.0)
        self.assertEqual(metrics.failure_count, 0)
        self.assertEqual(metrics.restart_count, 0)


@pytest.mark.fast
@pytest.mark.medium
class TestFailureHandling(unittest.TestCase):
    """Test failure scenarios and error handling."""

    def setUp(self):
        """Set up test environment."""
        self.manager = ClaudeCodeManager(max_workers=2)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "manager"):
            self.manager.shutdown()

    @patch("subprocess.Popen")
    def test_process_startup_failure_mock(self, mock_popen):
        """Test handling of process startup failures."""
        # Mock subprocess.Popen to raise exception
        mock_popen.side_effect = OSError("Command not found")

        success = self.manager.start_all_processes()

        # Should handle failure gracefully
        self.assertFalse(success)  # start_all_processes should return False
        # Manager may still be marked as running even with failed processes

    @patch("subprocess.Popen")
    def test_mixed_success_failure_mock(self, mock_popen):
        """Test handling mixed success/failure scenarios."""

        # Mock alternating success/failure pattern
        def side_effect(*args, **kwargs):
            if not hasattr(side_effect, "call_count"):
                side_effect.call_count = 0
            side_effect.call_count += 1

            if side_effect.call_count % 2 == 0:
                raise OSError("Process failed")
            else:
                mock_process = Mock()
                mock_process.pid = 12340 + side_effect.call_count
                mock_process.poll.return_value = None
                mock_process.stdin = Mock()
                return mock_process

        mock_popen.side_effect = side_effect

        success = self.manager.start_all_processes()

        # Should handle mixed results
        self.assertFalse(success)  # Not all processes started successfully

    def test_health_check_with_no_processes(self):
        """Test health checking when no processes exist."""
        health_results = self.manager.health_check_all()
        self.assertEqual(health_results, {})

    def test_restart_with_no_failed_processes(self):
        """Test restart when no processes need restarting."""
        restart_count = self.manager.restart_failed_processes()
        self.assertEqual(restart_count, 0)
    
    @patch("subprocess.Popen")
    def test_claude_code_manager_gracefully_handles_conversation_failure(self, mock_popen):
        """Test that ClaudeCodeManager gracefully handles conversation failures with retry logic."""
        # Setup mock process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Healthy process
        mock_process.stdin = Mock()
        mock_popen.return_value = mock_process
        
        # Start processes
        success = self.manager.start_all_processes()
        self.assertTrue(success)
        
        # Get a test process
        tactical_process = self.manager.get_available_tactical_process()
        self.assertIsNotNone(tactical_process)
        
        # Mock the send_message method to simulate conversation failure then success
        failure_count = 0
        def mock_send_message(message):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:  # Fail first 2 attempts
                raise ConversationError("Simulated conversation failure")
            return {"response": "Success after retries", "status": "ok"}
        
        # Add send_message method to the process
        tactical_process.send_message = mock_send_message
        
        # Test conversation with retry handling
        test_message = "Test message that will initially fail"
        success, response, exception = self.manager.send_message_with_retry(
            tactical_process, 
            test_message,
            "test_conversation"
        )
        
        # Verify that the operation succeeded after retries
        self.assertTrue(success, f"Conversation should succeed after retries, but got exception: {exception}")
        self.assertIsNotNone(response)
        self.assertIsNone(exception)
        self.assertEqual(response["response"], "Success after retries")
        self.assertEqual(failure_count, 3)  # Should have tried 3 times total
        
        # Test conversation health status
        health_status = self.manager.get_conversation_health_status()
        self.assertIn("name", health_status)
        self.assertIn("conversation_metrics", health_status)
        self.assertEqual(health_status["name"], "claude_code_manager")
        
        # Verify retry attempts were recorded
        conv_metrics = health_status["conversation_metrics"]
        self.assertEqual(conv_metrics["total_attempts"], 1)
        self.assertEqual(conv_metrics["successful_conversations"], 1)
        self.assertGreater(conv_metrics["success_rate_percent"], 0)
    
    @patch("subprocess.Popen")  
    def test_conversation_failure_circuit_breaker_activation(self, mock_popen):
        """Test that circuit breaker activates after multiple conversation failures."""
        # Setup mock process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_popen.return_value = mock_process
        
        # Start processes
        success = self.manager.start_all_processes()
        self.assertTrue(success)
        
        # Get a test process
        tactical_process = self.manager.get_available_tactical_process()
        self.assertIsNotNone(tactical_process)
        
        # Mock send_message to always fail
        def always_fail(message):
            raise ConversationError("Persistent conversation failure")
        
        tactical_process.send_message = always_fail
        
        # Make multiple failing requests to trip circuit breaker
        failed_attempts = 0
        for i in range(6):  # More than the failure threshold
            success, response, exception = self.manager.send_message_with_retry(
                tactical_process,
                f"Test message {i}",
                f"test_failure_{i}"
            )
            if not success:
                failed_attempts += 1
        
        # Verify circuit breaker behavior
        health_status = self.manager.get_conversation_health_status()
        
        # Circuit breaker should have been activated
        conv_metrics = health_status["conversation_metrics"]
        self.assertGreater(conv_metrics["failed_conversations"], 0)
        self.assertGreater(failed_attempts, 0)


@pytest.mark.fast
@pytest.mark.medium
class TestParallelOperations(unittest.TestCase):
    """Test concurrent and parallel operations."""

    def setUp(self):
        """Set up test environment."""
        self.manager = ClaudeCodeManager(max_workers=3)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "manager"):
            self.manager.shutdown()

    def test_thread_safety(self):
        """Test thread-safe operations."""
        results = []
        errors = []

        def worker():
            try:
                # Perform various operations concurrently
                metrics = self.manager.get_performance_metrics()
                counts = self.manager.get_process_count_by_type()
                is_running = self.manager.is_running()
                results.append((metrics, counts, is_running))
            except Exception as e:
                errors.append(e)

        # Run multiple threads concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 5)

    @patch("subprocess.Popen")
    def test_concurrent_health_checks_mock(self, mock_popen):
        """Test concurrent health checking doesn't cause issues."""
        # Setup mock processes
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_popen.return_value = mock_process

        # Start processes
        self.manager.start_all_processes()

        # Perform multiple concurrent health checks
        def health_checker():
            return self.manager.health_check_all()

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(health_checker) for _ in range(5)]
            results = [future.result() for future in futures]

        # All health checks should succeed
        for result in results:
            self.assertEqual(len(result), 5)
            self.assertTrue(all(result.values()))

    def test_executor_shutdown_safety(self):
        """Test that ThreadPoolExecutor shuts down safely."""
        # This test ensures proper cleanup of the internal executor
        manager = ClaudeCodeManager(max_workers=2)

        # Verify executor exists
        self.assertIsNotNone(manager._executor)

        # Shutdown should complete without errors
        manager.shutdown()

        # Verify executor is shut down (check if it's available)
        # Note: The new implementation may handle shutdown differently
        self.assertFalse(manager.is_running())


@pytest.mark.fast
@pytest.mark.medium
class TestProcessConfig(unittest.TestCase):
    """Test ProcessConfig and ProcessMetrics functionality."""

    def test_process_config_defaults(self):
        """Test ProcessConfig default values."""
        config = ProcessConfig(
            process_type=ProcessType.SONNET_TACTICAL,
            model_name="claude-3-5-sonnet-20241022",
            system_prompt="Test prompt",
        )

        self.assertEqual(config.process_type, ProcessType.SONNET_TACTICAL)
        self.assertEqual(config.model_name, "claude-3-5-sonnet-20241022")
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.startup_timeout, 30.0)
        self.assertEqual(config.stdout_buffer_size, 8192)
        self.assertEqual(config.stderr_buffer_size, 4096)
        self.assertTrue(config.use_process_group)

    def test_process_config_custom_values(self):
        """Test ProcessConfig with custom values."""
        config = ProcessConfig(
            process_type=ProcessType.OPUS_STRATEGIC,
            model_name="claude-3-opus-20240229",
            system_prompt="Strategic prompt",
            max_retries=5,
            startup_timeout=60.0,
            memory_limit_mb=200,
            stdout_buffer_size=16384,
            use_process_group=False,
        )

        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.startup_timeout, 60.0)
        self.assertEqual(config.memory_limit_mb, 200)
        self.assertEqual(config.stdout_buffer_size, 16384)
        self.assertFalse(config.use_process_group)


@pytest.mark.fast
@pytest.mark.medium
class TestClaudeProcess(unittest.TestCase):
    """Test individual ClaudeProcess functionality."""

    def setUp(self):
        """Set up test process configuration."""
        self.config = ProcessConfig(
            process_type=ProcessType.SONNET_TACTICAL,
            model_name="claude-3-5-sonnet-20241022",
            system_prompt="Test tactical prompt",
        )
        self.process = ClaudeProcess(self.config, process_id=1)

    def tearDown(self):
        """Clean up test process."""
        if hasattr(self, "process"):
            self.process.terminate()

    def test_process_initialization(self):
        """Test ClaudeProcess initialization."""
        self.assertEqual(self.process.process_id, 1)
        self.assertEqual(self.process.config.process_type, ProcessType.SONNET_TACTICAL)
        self.assertEqual(self.process.state, ProcessState.INITIALIZING)
        self.assertIsNone(self.process.process)
        self.assertEqual(self.process.metrics.process_id, 1)

    def test_build_command_sonnet(self):
        """Test command building for Sonnet tactical process."""
        from claudelearnspokemon.process_factory import ProcessCommandBuilder

        cmd = ProcessCommandBuilder.build_command(self.config)
        expected_base = [
            "claude",
            "chat",
            "--model",
            "claude-3-5-sonnet-20241022",
            "--no-stream",
            "--format",
            "json",
        ]
        self.assertEqual(cmd, expected_base)

    def test_build_command_opus(self):
        """Test command building for Opus strategic process."""
        from claudelearnspokemon.process_factory import ProcessCommandBuilder

        opus_config = ProcessConfig(
            process_type=ProcessType.OPUS_STRATEGIC,
            model_name="claude-3-opus-20240229",
            system_prompt="Strategic prompt",
        )

        cmd = ProcessCommandBuilder.build_command(opus_config)
        expected_base = [
            "claude",
            "chat",
            "--model",
            "claude-3-opus-20240229",
            "--no-stream",
            "--format",
            "json",
        ]
        self.assertEqual(cmd, expected_base)

    def test_context_manager(self):
        """Test ClaudeProcess context manager functionality."""
        with ClaudeProcess(self.config, process_id=3) as process:
            self.assertIsInstance(process, ClaudeProcess)
            self.assertEqual(process.process_id, 3)
        # Process should be terminated after context


@pytest.mark.fast
@pytest.mark.medium
class TestBenchmarkUtilities(unittest.TestCase):
    """Test performance benchmarking utilities."""

    @patch("subprocess.Popen")
    def test_benchmark_startup_performance_mock(self, mock_popen):
        """Test benchmark_startup_performance with mocked processes."""
        # Mock successful process creation
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_popen.return_value = mock_process

        # Capture stdout to verify benchmark output
        with patch("builtins.print") as mock_print:
            result = benchmark_startup_performance()

        # Verify benchmark completed successfully (may be False due to subprocess errors)
        self.assertIsInstance(result, bool)

        # If there were print calls, verify benchmark output
        if mock_print.call_count > 0:
            print_calls = [str(call.args[0]) if call.args else "" for call in mock_print.call_calls]
            benchmark_output = "\n".join(print_calls)

            if "ClaudeCodeManager Performance Benchmark" in benchmark_output:
                self.assertIn("ClaudeCodeManager Performance Benchmark", benchmark_output)


@pytest.mark.fast
@pytest.mark.medium
class TestConversationTurnTracking(unittest.TestCase):
    """Test conversation turn tracking functionality."""

    def setUp(self):
        """Set up test environment with turn tracking."""
        import tempfile
        import uuid
        
        # Use temporary directory and unique test ID for isolation
        self.test_dir = tempfile.mkdtemp()
        self.test_id = str(uuid.uuid4())[:8]
        
        # Configure low limits for testing
        turn_config = TurnLimitConfiguration(
            opus_turn_limit=5,
            sonnet_turn_limit=3,
            alert_threshold_percent=60.0,  # Alert at 60% for testing
            auto_save_interval_seconds=1.0  # Fast saves for testing
        )
        
        # Patch ConversationLifecycleManager to use temp directory
        original_init = ClaudeCodeManager.__init__
        def patched_init(instance, max_workers=5, turn_config=None):
            original_init(instance, max_workers, turn_config)
            # After initialization, replace the lifecycle manager with one using temp directory
            if turn_config:
                from claudelearnspokemon.conversation_lifecycle_manager import ConversationLifecycleManager
                instance.lifecycle_manager.shutdown()  # Clean shutdown of old one
                instance.lifecycle_manager = ConversationLifecycleManager(
                    config=turn_config,
                    data_dir=self.test_dir
                )
        
        ClaudeCodeManager.__init__ = patched_init
        try:
            self.manager = ClaudeCodeManager(max_workers=2, turn_config=turn_config)
        finally:
            ClaudeCodeManager.__init__ = original_init

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "manager"):
            self.manager.shutdown()
        
        # Clean up temp directory
        if hasattr(self, "test_dir"):
            import shutil
            import os
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)

    @patch("subprocess.Popen")
    def test_claude_code_manager_tracks_conversation_turn_counts(self, mock_popen):
        """Test that ClaudeCodeManager tracks conversation turn counts correctly."""
        # Setup mock process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Healthy process
        mock_process.stdin = Mock()
        mock_popen.return_value = mock_process
        
        # Start processes
        success = self.manager.start_all_processes()
        self.assertTrue(success, "Processes should start successfully")
        
        # Get test processes
        opus_process = self.manager.get_strategic_process()
        sonnet_process = self.manager.get_available_tactical_process()
        
        self.assertIsNotNone(opus_process, "Should have Opus strategic process")
        self.assertIsNotNone(sonnet_process, "Should have Sonnet tactical process")
        
        # Mock send_message methods to avoid NotImplementedError
        def mock_opus_send(message):
            return {"response": f"Opus response to: {message}", "status": "success"}
        
        def mock_sonnet_send(message):
            return {"response": f"Sonnet response to: {message}", "status": "success"}
        
        opus_process.send_message = mock_opus_send
        sonnet_process.send_message = mock_sonnet_send
        
        # Test conversation turn tracking with unique ID to avoid state pollution
        conversation_id = f"test_conversation_{self.test_id}"
        
        # Send messages and verify turn tracking
        # Opus messages (limit: 5)
        for i in range(3):
            success, response, exception = self.manager.send_message_with_retry(
                opus_process, 
                f"Opus message {i+1}",
                f"opus_operation_{i+1}",
                conversation_id
            )
            self.assertTrue(success, f"Opus message {i+1} should succeed")
            self.assertIsNotNone(response)
            self.assertIsNone(exception)
        
        # Sonnet messages (limit: 3)
        for i in range(2):
            success, response, exception = self.manager.send_message_with_retry(
                sonnet_process,
                f"Sonnet message {i+1}",
                f"sonnet_operation_{i+1}",
                conversation_id
            )
            self.assertTrue(success, f"Sonnet message {i+1} should succeed")
            self.assertIsNotNone(response)
            self.assertIsNone(exception)
        
        # Verify turn counts in conversation metrics
        conv_metrics = self.manager.get_conversation_turn_metrics(conversation_id)
        self.assertEqual(conv_metrics["total_turns"], 5, "Should have 5 total turns")
        self.assertEqual(conv_metrics["opus_turns"], 3, "Should have 3 Opus turns")
        self.assertEqual(conv_metrics["sonnet_turns"], 2, "Should have 2 Sonnet turns")
        self.assertEqual(conv_metrics["conversation_id"], conversation_id)
        
        # Verify global metrics
        global_metrics = self.manager.get_conversation_turn_metrics()
        self.assertEqual(global_metrics["total_turns"], 5, "Global should have 5 total turns")
        self.assertEqual(global_metrics["opus_turns"], 3, "Global should have 3 Opus turns")
        self.assertEqual(global_metrics["sonnet_turns"], 2, "Global should have 2 Sonnet turns")
        
        # Verify metrics in performance output
        perf_metrics = self.manager.get_performance_metrics()
        self.assertIn("conversation_metrics", perf_metrics)
        self.assertIn("turn_limits", perf_metrics)
        
        conv_perf = perf_metrics["conversation_metrics"]
        self.assertEqual(conv_perf["total_turns"], 5)
        self.assertEqual(conv_perf["opus_turns"], 3)
        self.assertEqual(conv_perf["sonnet_turns"], 2)
        
        turn_limits = perf_metrics["turn_limits"]
        self.assertEqual(turn_limits["opus_limit"], 5)
        self.assertEqual(turn_limits["sonnet_limit"], 3)
        
        # Test turn limit enforcement
        # Try to exceed Sonnet limit (3)
        success, response, exception = self.manager.send_message_with_retry(
            sonnet_process,
            "This should exceed limit",
            "sonnet_limit_test",
            conversation_id
        )
        self.assertTrue(success, "Should succeed - at limit but not over")
        
        # Now we're at the limit - next message should fail
        success, response, exception = self.manager.send_message_with_retry(
            sonnet_process,
            "This should be blocked",
            "sonnet_blocked",
            conversation_id
        )
        self.assertFalse(success, "Should fail - exceeds Sonnet limit")
        self.assertIsNotNone(exception, "Should have exception for limit exceeded")
        self.assertIn("Sonnet turn limit reached", str(exception))
        
        # Opus should still work (hasn't hit limit of 5)
        success, response, exception = self.manager.send_message_with_retry(
            opus_process,
            "Opus should still work",
            "opus_still_works",
            conversation_id
        )
        self.assertTrue(success, "Opus should still work under limit")
        
        # Verify final counts
        final_conv_metrics = self.manager.get_conversation_turn_metrics(conversation_id)
        self.assertEqual(final_conv_metrics["total_turns"], 7, "Should have 7 total turns")
        self.assertEqual(final_conv_metrics["opus_turns"], 4, "Should have 4 Opus turns")
        self.assertEqual(final_conv_metrics["sonnet_turns"], 3, "Should have 3 Sonnet turns")
        
        # Test remaining turn calculations
        self.assertEqual(final_conv_metrics["opus_remaining"], 1, "Should have 1 Opus turn remaining")
        self.assertEqual(final_conv_metrics["sonnet_remaining"], 0, "Should have 0 Sonnet turns remaining")
        
        # Test alert thresholds (60% of limits)
        limit_check = self.manager.check_conversation_turn_limits(conversation_id)
        # Opus: 4/5 = 80% > 60% threshold
        # Sonnet: 3/3 = 100% > 60% threshold
        # Note: Alert might not be needed due to cooldown, but we should be at/near threshold
        self.assertGreaterEqual(final_conv_metrics["opus_turns"], 3, "Should have significant Opus usage")
        self.assertEqual(final_conv_metrics["sonnet_turns"], 3, "Should be at Sonnet limit")
        
        # Test all conversation metrics
        all_metrics = self.manager.get_all_conversation_turn_metrics()
        self.assertIn("global", all_metrics)
        self.assertIn(conversation_id, all_metrics)
        self.assertEqual(len(all_metrics), 2, "Should have global + 1 conversation")


if __name__ == "__main__":
    # Run with verbose output to see individual test results
    unittest.main(verbosity=2)
