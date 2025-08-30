"""
Integration tests for conversation setup with real Claude CLI integration.

This test file validates the new ConversationState functionality and
replaces mock-based tests with real subprocess validation where possible.
"""

import unittest

import pytest

from claudelearnspokemon.claude_process import ClaudeProcess
from claudelearnspokemon.conversation_state import (
    ConversationState,
    ConversationStateManager,
    ConversationStatus,
)
from claudelearnspokemon.process_communication import ProcessCommunicator
from claudelearnspokemon.process_factory import ProcessConfig
from claudelearnspokemon.prompts import ProcessType


@pytest.mark.fast
class TestConversationStateBasics(unittest.TestCase):
    """Test ConversationState class functionality."""

    def setUp(self):
        """Set up test conversation state."""
        self.tactical_state = ConversationState(ProcessType.SONNET_TACTICAL, 1)
        self.strategic_state = ConversationState(ProcessType.OPUS_STRATEGIC, 2)

    def test_conversation_state_initialization(self):
        """Test ConversationState initializes with correct turn limits."""
        # Tactical process (Sonnet) should have 20 turn limit
        self.assertEqual(self.tactical_state._max_turns, 20)
        self.assertEqual(self.tactical_state.turn_count, 0)
        self.assertEqual(self.tactical_state.status, ConversationStatus.INITIALIZING)

        # Strategic process (Opus) should have 100 turn limit
        self.assertEqual(self.strategic_state._max_turns, 100)
        self.assertEqual(self.strategic_state.turn_count, 0)
        self.assertEqual(self.strategic_state.status, ConversationStatus.INITIALIZING)

    def test_conversation_initialization(self):
        """Test conversation initialization with system prompt."""
        system_prompt = "Test system prompt for Pokemon speedrun agent"

        success = self.tactical_state.initialize_conversation(system_prompt)
        self.assertTrue(success)
        self.assertEqual(self.tactical_state.status, ConversationStatus.ACTIVE)
        self.assertTrue(self.tactical_state._system_prompt_sent)
        self.assertIsNotNone(self.tactical_state._initialization_time)

    def test_message_exchange_recording(self):
        """Test recording message exchanges with turn counting."""
        # Initialize conversation first
        self.tactical_state.initialize_conversation("Test prompt")

        # Record several message exchanges
        turn1 = self.tactical_state.record_message_exchange(
            "First message", "First response", 100.0
        )
        self.assertEqual(turn1, 1)
        self.assertEqual(self.tactical_state.turn_count, 1)

        turn2 = self.tactical_state.record_message_exchange(
            "Second message", "Second response", 150.0
        )
        self.assertEqual(turn2, 2)
        self.assertEqual(self.tactical_state.turn_count, 2)

        # Verify metrics update
        self.assertEqual(self.tactical_state.metrics.total_turns, 2)
        self.assertGreater(self.tactical_state.metrics.average_response_time_ms, 0)

    def test_turn_limit_enforcement(self):
        """Test turn limit enforcement and status transitions."""
        self.tactical_state.initialize_conversation("Test prompt")

        # Fill up to warning threshold (16 turns for tactical = 80% of 20)
        for i in range(16):
            self.tactical_state.record_message_exchange(f"Message {i}", f"Response {i}", 100.0)

        self.assertEqual(self.tactical_state.status, ConversationStatus.APPROACHING_LIMIT)
        self.assertTrue(self.tactical_state.needs_context_compression())

        # Fill to limit
        for i in range(16, 20):
            self.tactical_state.record_message_exchange(f"Message {i}", f"Response {i}", 100.0)

        self.assertEqual(self.tactical_state.status, ConversationStatus.LIMIT_REACHED)
        self.assertFalse(self.tactical_state.can_send_message())

    def test_conversation_context_generation(self):
        """Test conversation context generation for Claude CLI."""
        self.tactical_state.initialize_conversation("Test prompt")

        # Add some message exchanges
        for i in range(3):
            self.tactical_state.record_message_exchange(
                f"Message {i}", f"Response {i}", 100.0 + i * 10
            )

        context = self.tactical_state.get_conversation_context(recent_turns=2)
        self.assertIsInstance(context, str)
        # Should be valid JSON
        import json

        context_data = json.loads(context)
        self.assertIn("conversation_id", context_data)
        self.assertIn("turn_count", context_data)
        self.assertIn("recent_exchanges", context_data)

    def test_status_summary(self):
        """Test comprehensive status summary generation."""
        self.tactical_state.initialize_conversation("Test prompt")

        summary = self.tactical_state.get_status_summary()

        expected_keys = [
            "process_id",
            "process_type",
            "status",
            "turn_count",
            "max_turns",
            "turns_remaining",
            "warning_threshold",
            "approaching_limit",
            "limit_reached",
            "metrics",
            "memory_usage",
        ]

        for key in expected_keys:
            self.assertIn(key, summary)

        self.assertEqual(summary["process_id"], 1)
        self.assertEqual(summary["process_type"], "sonnet_tactical")
        self.assertEqual(summary["max_turns"], 20)

    def test_conversation_reset(self):
        """Test conversation reset functionality."""
        self.tactical_state.initialize_conversation("Test prompt")

        # Add some exchanges
        for i in range(5):
            self.tactical_state.record_message_exchange(f"Message {i}", f"Response {i}")

        original_metrics = self.tactical_state.metrics.total_turns
        self.assertGreater(original_metrics, 0)

        # Reset with metrics preservation
        self.tactical_state.reset_conversation(preserve_metrics=True)

        self.assertEqual(self.tactical_state.turn_count, 0)
        self.assertEqual(self.tactical_state.status, ConversationStatus.INITIALIZING)
        self.assertFalse(self.tactical_state._system_prompt_sent)
        # Metrics should still include context preservation event count
        self.assertEqual(self.tactical_state.metrics.context_preservation_events, 1)


@pytest.mark.fast
class TestConversationStateManager(unittest.TestCase):
    """Test ConversationStateManager functionality."""

    def setUp(self):
        """Set up test manager."""
        self.manager = ConversationStateManager()

    def test_manager_initialization(self):
        """Test ConversationStateManager initializes correctly."""
        self.assertEqual(len(self.manager.states), 0)

    def test_add_remove_conversations(self):
        """Test adding and removing conversation states."""
        state1 = ConversationState(ProcessType.SONNET_TACTICAL, 1)
        state2 = ConversationState(ProcessType.OPUS_STRATEGIC, 2)

        # Add states
        self.manager.add_conversation(state1)
        self.manager.add_conversation(state2)

        self.assertEqual(len(self.manager.states), 2)
        self.assertIs(self.manager.get_conversation(1), state1)
        self.assertIs(self.manager.get_conversation(2), state2)

        # Remove state
        self.manager.remove_conversation(1)
        self.assertEqual(len(self.manager.states), 1)
        self.assertIsNone(self.manager.get_conversation(1))

    def test_system_summary(self):
        """Test system-wide conversation summary."""
        # Empty system
        summary = self.manager.get_system_summary()
        self.assertEqual(summary["total_conversations"], 0)

        # Add active conversations
        state1 = ConversationState(ProcessType.SONNET_TACTICAL, 1)
        state1.initialize_conversation("Test prompt 1")
        state1.record_message_exchange("Message 1", "Response 1")

        state2 = ConversationState(ProcessType.OPUS_STRATEGIC, 2)
        state2.initialize_conversation("Test prompt 2")

        self.manager.add_conversation(state1)
        self.manager.add_conversation(state2)

        summary = self.manager.get_system_summary()
        self.assertEqual(summary["total_conversations"], 2)
        self.assertEqual(summary["active_conversations"], 2)
        self.assertEqual(summary["total_turns_across_all"], 1)  # Only state1 has a turn


@pytest.mark.fast
class TestClaudeProcessConversationIntegration(unittest.TestCase):
    """Test ConversationState integration with ClaudeProcess."""

    def setUp(self):
        """Set up test process configuration."""
        self.config = ProcessConfig(
            process_type=ProcessType.SONNET_TACTICAL,
            model_name="claude-3-5-sonnet-20241022",
            system_prompt="Test tactical prompt",
        )

    def test_claude_process_conversation_initialization(self):
        """Test ClaudeProcess includes conversation state."""
        process = ClaudeProcess(self.config, process_id=1)

        # Verify conversation state is created
        self.assertIsNotNone(process.conversation_state)
        self.assertEqual(process.conversation_state.process_type, ProcessType.SONNET_TACTICAL)
        self.assertEqual(process.conversation_state.process_id, 1)

        # Verify methods are available
        self.assertTrue(hasattr(process, "get_conversation_status"))
        self.assertTrue(hasattr(process, "can_send_message"))
        self.assertTrue(hasattr(process, "needs_context_compression"))

    def test_conversation_status_methods(self):
        """Test conversation status methods on ClaudeProcess."""
        process = ClaudeProcess(self.config, process_id=1)

        # Test can_send_message (should be False before initialization)
        self.assertFalse(process.can_send_message())

        # Initialize conversation
        process.conversation_state.initialize_conversation("Test prompt")
        self.assertTrue(process.can_send_message())

        # Test conversation status
        status = process.get_conversation_status()
        self.assertIsInstance(status, dict)
        self.assertIn("status", status)
        self.assertIn("turn_count", status)

        # Test needs compression
        self.assertFalse(process.needs_context_compression())


@pytest.mark.fast
class TestProcessCommunicatorConversationIntegration(unittest.TestCase):
    """Test ProcessCommunicator integration with ConversationState."""

    def setUp(self):
        """Set up test communicator."""
        self.conversation_state = ConversationState(ProcessType.SONNET_TACTICAL, 1)
        self.communicator = ProcessCommunicator(
            process_id=1, conversation_state=self.conversation_state
        )

    def test_communicator_initialization_with_conversation_state(self):
        """Test ProcessCommunicator accepts conversation state."""
        self.assertIs(self.communicator.conversation_state, self.conversation_state)
        self.assertTrue(self.communicator._json_communication)

    def test_turn_limit_enforcement_in_send_message(self):
        """Test that send_message respects turn limits."""
        import subprocess
        from unittest.mock import Mock

        # Mock process for testing
        mock_process = Mock(spec=subprocess.Popen)
        mock_process.stdin = Mock()
        mock_process.poll.return_value = None

        # Initialize conversation
        self.conversation_state.initialize_conversation("Test prompt")

        # Fill up turn limit
        for i in range(20):
            self.conversation_state.record_message_exchange(f"Message {i}", f"Response {i}")

        # Should not be able to send more messages
        self.assertFalse(self.conversation_state.can_send_message())

        # send_message should return False when limit reached
        result = self.communicator.send_message(mock_process, "Blocked message")
        self.assertFalse(result)


if __name__ == "__main__":
    # Run with verbose output to see individual test results
    unittest.main(verbosity=2)
