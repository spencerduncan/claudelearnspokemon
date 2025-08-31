"""
Performance benchmark suite for conversation setup and parallel initialization.

This test suite validates that conversation setup meets John Botmack's
performance targets: <100ms tactical, <500ms strategic startup.
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

import pytest

from claudelearnspokemon.claude_code_manager import ClaudeCodeManager
from claudelearnspokemon.claude_process import ClaudeProcess
from claudelearnspokemon.conversation_state import ConversationState
from claudelearnspokemon.process_factory import ProcessConfig
from claudelearnspokemon.prompts import ProcessType


@pytest.mark.fast
@pytest.mark.medium
class TestConversationSetupPerformance(unittest.TestCase):
    """Performance benchmarks for conversation setup operations."""

    def test_conversation_state_initialization_performance(self):
        """Test ConversationState initialization meets <1ms target."""
        results = []

        # Run multiple initialization benchmarks
        for i in range(100):
            start_time = time.perf_counter()
            ConversationState(ProcessType.SONNET_TACTICAL, i)  # Performance measurement only
            init_time = (time.perf_counter() - start_time) * 1000
            results.append(init_time)

        avg_init_time = sum(results) / len(results)
        max_init_time = max(results)

        print("\n=== ConversationState Initialization Performance ===")
        print(f"Average initialization time: {avg_init_time:.3f}ms")
        print(f"Maximum initialization time: {max_init_time:.3f}ms")
        print("Target: <1ms")

        # Performance assertion
        self.assertLess(avg_init_time, 1.0, "ConversationState initialization should be <1ms")
        self.assertLess(max_init_time, 2.0, "Max ConversationState initialization should be <2ms")

    def test_turn_tracking_performance(self):
        """Test turn tracking operations meet <1ms target."""
        state = ConversationState(ProcessType.SONNET_TACTICAL, 1)
        state.initialize_conversation("Test prompt")

        turn_times = []

        # Benchmark turn recording
        for i in range(20):  # Full turn limit for tactical
            start_time = time.perf_counter()
            state.record_message_exchange(f"Message {i}", f"Response {i}", 100.0)
            turn_time = (time.perf_counter() - start_time) * 1000
            turn_times.append(turn_time)

        avg_turn_time = sum(turn_times) / len(turn_times)
        max_turn_time = max(turn_times)

        print("\n=== Turn Tracking Performance ===")
        print(f"Average turn recording time: {avg_turn_time:.3f}ms")
        print(f"Maximum turn recording time: {max_turn_time:.3f}ms")
        print("Target: <1ms")

        # Performance assertion
        self.assertLess(avg_turn_time, 1.0, "Turn tracking should be <1ms average")
        self.assertLess(max_turn_time, 2.0, "Turn tracking should be <2ms max")

    def test_status_summary_performance(self):
        """Test status summary generation performance."""
        state = ConversationState(ProcessType.OPUS_STRATEGIC, 1)
        state.initialize_conversation("Test prompt")

        # Add some history
        for i in range(10):
            state.record_message_exchange(f"Message {i}", f"Response {i}", 100.0)

        summary_times = []

        # Benchmark status summary generation
        for _i in range(100):
            start_time = time.perf_counter()
            state.get_status_summary()  # Performance measurement only
            summary_time = (time.perf_counter() - start_time) * 1000
            summary_times.append(summary_time)

        avg_summary_time = sum(summary_times) / len(summary_times)
        max_summary_time = max(summary_times)

        print("\n=== Status Summary Performance ===")
        print(f"Average summary generation time: {avg_summary_time:.3f}ms")
        print(f"Maximum summary generation time: {max_summary_time:.3f}ms")
        print("Target: <10ms")

        # Performance assertion
        self.assertLess(avg_summary_time, 10.0, "Status summary should be <10ms average")


@pytest.mark.fast
@pytest.mark.medium
class TestParallelConversationInitialization(unittest.TestCase):
    """Performance benchmarks for parallel conversation initialization."""

    @patch("subprocess.Popen")
    def test_parallel_tactical_process_initialization(self, mock_popen):
        """Test parallel tactical process startup meets <100ms target."""
        # Mock successful process creation for performance testing
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_process.stdout = Mock()
        mock_process.stderr = Mock()
        mock_popen.return_value = mock_process

        # Test parallel initialization of 4 tactical processes
        configs = []
        for _i in range(4):
            config = ProcessConfig(
                process_type=ProcessType.SONNET_TACTICAL,
                model_name="claude-3-5-sonnet-20241022",
                system_prompt="Tactical prompt",
                startup_timeout=2.0,  # Reduced for testing
            )
            configs.append(config)

        start_time = time.perf_counter()

        # Parallel process creation
        processes = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, config in enumerate(configs):
                future = executor.submit(self._create_and_initialize_process, config, i)
                futures.append(future)

            # Wait for all processes to initialize
            for future in as_completed(futures):
                process = future.result()
                processes.append(process)

        total_time = (time.perf_counter() - start_time) * 1000

        print("\n=== Parallel Tactical Process Initialization ===")
        print(f"4 tactical processes initialized in: {total_time:.1f}ms")
        print("Target: <100ms per tactical process")
        print(f"Average per process: {total_time/4:.1f}ms")

        # Performance assertion - tactical processes should start quickly
        self.assertLess(total_time / 4, 100.0, "Average tactical startup should be <100ms")
        self.assertEqual(len(processes), 4, "All processes should initialize successfully")

        # Cleanup
        for process in processes:
            if hasattr(process, "terminate"):
                process.terminate()

    @patch("subprocess.Popen")
    def test_parallel_strategic_process_initialization(self, mock_popen):
        """Test strategic process startup meets <500ms target."""
        # Mock successful process creation for performance testing
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_process.stdout = Mock()
        mock_process.stderr = Mock()
        mock_popen.return_value = mock_process

        config = ProcessConfig(
            process_type=ProcessType.OPUS_STRATEGIC,
            model_name="claude-3-opus-20240229",
            system_prompt="Strategic prompt",
            startup_timeout=5.0,  # Strategic processes get more time
        )

        start_time = time.perf_counter()
        process = self._create_and_initialize_process(config, 0)
        total_time = (time.perf_counter() - start_time) * 1000

        print("\n=== Strategic Process Initialization ===")
        print(f"Strategic process initialized in: {total_time:.1f}ms")
        print("Target: <500ms")

        # Performance assertion - strategic process startup
        self.assertLess(total_time, 500.0, "Strategic startup should be <500ms")
        self.assertIsNotNone(process, "Strategic process should initialize successfully")

        # Cleanup
        if hasattr(process, "terminate"):
            process.terminate()

    def _create_and_initialize_process(
        self, config: ProcessConfig, process_id: int
    ) -> ClaudeProcess:
        """Helper method to create and initialize a ClaudeProcess."""
        process = ClaudeProcess(config, process_id)

        # Start the process (mocked subprocess)
        success = process.start()
        if not success:
            raise RuntimeError(f"Failed to start process {process_id}")

        return process

    @patch("subprocess.Popen")
    def test_full_system_parallel_initialization(self, mock_popen):
        """Test full ClaudeCodeManager parallel initialization performance."""
        # Mock successful process creation
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_process.stdout = Mock()
        mock_process.stderr = Mock()
        mock_popen.return_value = mock_process

        start_time = time.perf_counter()

        # Test full system initialization (1 strategic + 4 tactical)
        with ClaudeCodeManager(max_workers=5) as manager:
            success = manager.start_all_processes()
            initialization_time = (time.perf_counter() - start_time) * 1000

            print("\n=== Full System Parallel Initialization ===")
            print(
                f"Full system (1 strategic + 4 tactical) initialized in: {initialization_time:.1f}ms"
            )
            print("Target: <5000ms total")
            print(f"Processes created: {len(manager.processes)}")

            # Verify all processes initialized
            self.assertTrue(success, "All processes should start successfully")
            self.assertEqual(len(manager.processes), 5, "Should have 5 processes total")

            # Performance assertion
            self.assertLess(initialization_time, 5000.0, "Full system startup should be <5s")

            # Verify conversation states are properly initialized
            conversation_states = []
            for process in manager.processes.values():
                if hasattr(process, "conversation_state"):
                    conversation_states.append(process.conversation_state)

            self.assertEqual(
                len(conversation_states), 5, "All processes should have conversation states"
            )


@pytest.mark.fast
@pytest.mark.medium
class TestConversationMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency of conversation state management."""

    def test_circular_buffer_memory_usage(self):
        """Test that conversation history uses circular buffer efficiently."""
        state = ConversationState(ProcessType.SONNET_TACTICAL, 1)
        state.initialize_conversation("Test prompt")

        # Fill beyond circular buffer size (should be limited to 50 or turn limit)
        expected_buffer_size = min(50, 20)  # 20 is turn limit for tactical

        # Add more messages than buffer size
        for i in range(expected_buffer_size + 10):
            state.record_message_exchange(f"Message {i}", f"Response {i}", 100.0)

        # Verify buffer doesn't grow beyond expected size
        actual_history_size = len(state._message_history)

        print("\n=== Circular Buffer Memory Efficiency ===")
        print(f"Messages added: {expected_buffer_size + 10}")
        print(f"Actual history size: {actual_history_size}")
        print(f"Expected buffer limit: {expected_buffer_size}")

        # Should be limited by turn limit since we hit it first
        self.assertLessEqual(
            actual_history_size,
            expected_buffer_size,
            "History buffer should not exceed expected size",
        )

    def test_multiple_conversations_memory_isolation(self):
        """Test that multiple conversation states don't interfere with memory."""
        conversations = []

        # Create multiple conversation states
        for i in range(10):
            state = ConversationState(ProcessType.SONNET_TACTICAL, i)
            state.initialize_conversation(f"Test prompt {i}")

            # Add different amounts of history
            for j in range(i + 1):
                state.record_message_exchange(f"Message {j}", f"Response {j}", 100.0)

            conversations.append(state)

        # Verify each conversation maintains independent state
        for i, conversation in enumerate(conversations):
            expected_turns = i + 1
            self.assertEqual(
                conversation.turn_count,
                expected_turns,
                f"Conversation {i} should have {expected_turns} turns",
            )

        print("\n=== Memory Isolation Test ===")
        print(f"Created {len(conversations)} independent conversations")
        print("✓ Each conversation maintains independent turn count and history")


if __name__ == "__main__":
    # Run performance tests with detailed output
    unittest.main(verbosity=2)
