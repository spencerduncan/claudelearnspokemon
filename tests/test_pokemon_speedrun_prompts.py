"""
Tests for Pokemon speedrun system prompt effectiveness and Claude CLI integration.

This test suite validates that the system prompts are appropriate for
Pokemon speedrun learning agent tasks and conversation setup works correctly.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from claudelearnspokemon.claude_process import ClaudeProcess
from claudelearnspokemon.conversation_state import ConversationState, ConversationStatus
from claudelearnspokemon.process_factory import ProcessConfig
from claudelearnspokemon.prompts import ProcessType, PromptRepository


@pytest.mark.fast
class TestPokemonSpeedrunSystemPrompts(unittest.TestCase):
    """Test Pokemon speedrun system prompt design and effectiveness."""

    def test_opus_strategic_prompt_content(self):
        """Test Opus strategic prompt contains appropriate Pokemon speedrun context."""
        prompt = PromptRepository.get_prompt(ProcessType.OPUS_STRATEGIC)

        # Verify prompt contains key strategic concepts
        strategic_keywords = [
            "chess grandmaster",
            "Pokemon Red speedrun",
            "BIG PICTURE PLANNING",
            "PATTERN SYNTHESIS",
            "route patterns",
            "strategic checkpoint",
            "meta-patterns",
            "tactical experiments",
        ]

        for keyword in strategic_keywords:
            self.assertIn(
                keyword,
                prompt,
                f"Strategic prompt should contain '{keyword}' for Pokemon speedrun context",
            )

        # Verify prompt emphasizes strategic thinking
        strategic_concepts = [
            "Think 10+ moves ahead",
            "optimal route patterns",
            "Synthesize",
            "STRATEGIC INTELLIGENCE",
            "Coordination",
        ]

        for concept in strategic_concepts:
            self.assertIn(concept, prompt, f"Strategic prompt should emphasize '{concept}'")

    def test_sonnet_tactical_prompt_content(self):
        """Test Sonnet tactical prompt contains appropriate execution focus."""
        prompt = PromptRepository.get_prompt(ProcessType.SONNET_TACTICAL)

        # Verify prompt contains key tactical concepts
        tactical_keywords = [
            "frame-perfect speedrunner",
            "Pokemon Red execution scripts",
            "PRECISION EXECUTION",
            "MICRO-OPTIMIZATION",
            "tool-assisted speedrun",
            "input sequences",
            "DSL pattern",
            "tactical problem-solving",
        ]

        for keyword in tactical_keywords:
            self.assertIn(
                keyword,
                prompt,
                f"Tactical prompt should contain '{keyword}' for Pokemon speedrun context",
            )

        # Verify prompt emphasizes execution precision
        execution_concepts = [
            "frame-perfect timing",
            "exact input sequences",
            "TACTICAL PRECISION",
            "Micro-optimizations",
            "speedrunner precision",
        ]

        for concept in execution_concepts:
            self.assertIn(concept, prompt, f"Tactical prompt should emphasize '{concept}'")

    def test_prompt_length_optimization(self):
        """Test that prompts are appropriately sized for Claude CLI efficiency."""
        opus_prompt = PromptRepository.get_prompt(ProcessType.OPUS_STRATEGIC)
        sonnet_prompt = PromptRepository.get_prompt(ProcessType.SONNET_TACTICAL)

        # Prompts should be substantial but not excessive
        self.assertGreater(len(opus_prompt), 200, "Strategic prompt should be substantial")
        self.assertLess(len(opus_prompt), 2000, "Strategic prompt should not be excessive")

        self.assertGreater(len(sonnet_prompt), 200, "Tactical prompt should be substantial")
        self.assertLess(len(sonnet_prompt), 2000, "Tactical prompt should not be excessive")

        # Prompts should be focused and concise
        opus_lines = opus_prompt.count("\n")
        sonnet_lines = sonnet_prompt.count("\n")

        self.assertLess(opus_lines, 30, "Strategic prompt should be concise")
        self.assertLess(sonnet_lines, 30, "Tactical prompt should be concise")

    def test_prompt_role_differentiation(self):
        """Test that strategic and tactical prompts have clear role differentiation."""
        opus_prompt = PromptRepository.get_prompt(ProcessType.OPUS_STRATEGIC)
        sonnet_prompt = PromptRepository.get_prompt(ProcessType.SONNET_TACTICAL)

        # Strategic should focus on planning and synthesis
        strategic_unique = [
            "grandmaster",
            "BIG PICTURE",
            "Synthesize",
            "Coordination",
            "meta-patterns",
        ]
        for term in strategic_unique:
            self.assertIn(term, opus_prompt, f"Strategic prompt should emphasize '{term}'")
            self.assertNotIn(term, sonnet_prompt, f"Tactical prompt should not emphasize '{term}'")

        # Tactical should focus on execution and precision
        tactical_unique = [
            "frame-perfect",
            "PRECISION EXECUTION",
            "tool-assisted",
            "MICRO-OPTIMIZATION",
        ]
        for term in tactical_unique:
            self.assertIn(term, sonnet_prompt, f"Tactical prompt should emphasize '{term}'")
            self.assertNotIn(term, opus_prompt, f"Strategic prompt should not emphasize '{term}'")

    def test_prompt_pokemon_specificity(self):
        """Test that prompts are specifically tailored to Pokemon Red speedrun context."""
        opus_prompt = PromptRepository.get_prompt(ProcessType.OPUS_STRATEGIC)
        sonnet_prompt = PromptRepository.get_prompt(ProcessType.SONNET_TACTICAL)

        # Both should mention Pokemon Red specifically
        for prompt_type, prompt in [("Strategic", opus_prompt), ("Tactical", sonnet_prompt)]:
            self.assertIn(
                "Pokemon Red",
                prompt,
                f"{prompt_type} prompt should specifically mention 'Pokemon Red'",
            )
            self.assertIn(
                "speedrun", prompt, f"{prompt_type} prompt should specifically mention 'speedrun'"
            )


@pytest.mark.fast
class TestConversationSetupWithPrompts(unittest.TestCase):
    """Test complete conversation setup with Pokemon speedrun prompts."""

    def setUp(self):
        """Set up test processes with Pokemon speedrun configuration."""
        self.strategic_config = ProcessConfig(
            process_type=ProcessType.OPUS_STRATEGIC,
            model_name="claude-3-opus-20240229",
            system_prompt=PromptRepository.get_prompt(ProcessType.OPUS_STRATEGIC),
        )

        self.tactical_config = ProcessConfig(
            process_type=ProcessType.SONNET_TACTICAL,
            model_name="claude-3-5-sonnet-20241022",
            system_prompt=PromptRepository.get_prompt(ProcessType.SONNET_TACTICAL),
        )

    def test_conversation_state_initialization_with_prompts(self):
        """Test conversation state initializes correctly with Pokemon speedrun prompts."""
        # Test strategic conversation
        strategic_state = ConversationState(ProcessType.OPUS_STRATEGIC, 1)
        opus_prompt = PromptRepository.get_prompt(ProcessType.OPUS_STRATEGIC)

        success = strategic_state.initialize_conversation(opus_prompt)
        self.assertTrue(success, "Strategic conversation should initialize successfully")
        self.assertEqual(strategic_state.status, ConversationStatus.ACTIVE)
        self.assertEqual(strategic_state._max_turns, 100, "Strategic should have 100 turn limit")

        # Test tactical conversation
        tactical_state = ConversationState(ProcessType.SONNET_TACTICAL, 2)
        sonnet_prompt = PromptRepository.get_prompt(ProcessType.SONNET_TACTICAL)

        success = tactical_state.initialize_conversation(sonnet_prompt)
        self.assertTrue(success, "Tactical conversation should initialize successfully")
        self.assertEqual(tactical_state.status, ConversationStatus.ACTIVE)
        self.assertEqual(tactical_state._max_turns, 20, "Tactical should have 20 turn limit")

    @patch("subprocess.Popen")
    def test_claude_process_initialization_with_pokemon_prompts(self, mock_popen):
        """Test ClaudeProcess initializes with Pokemon speedrun prompts."""
        # Mock subprocess for testing
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_process.stdout = Mock()
        mock_process.stderr = Mock()
        mock_popen.return_value = mock_process

        # Test strategic process
        strategic_process = ClaudeProcess(self.strategic_config, 1)
        self.assertIsNotNone(strategic_process.conversation_state)
        self.assertEqual(
            strategic_process.conversation_state.process_type, ProcessType.OPUS_STRATEGIC
        )

        # Test process startup (mocked)
        success = strategic_process.start()
        self.assertTrue(success, "Strategic process should start successfully")

        # Verify conversation state is initialized
        self.assertTrue(
            strategic_process.conversation_state._system_prompt_sent,
            "System prompt should be marked as sent",
        )

        # Test tactical process
        tactical_process = ClaudeProcess(self.tactical_config, 2)
        self.assertIsNotNone(tactical_process.conversation_state)
        self.assertEqual(
            tactical_process.conversation_state.process_type, ProcessType.SONNET_TACTICAL
        )

        success = tactical_process.start()
        self.assertTrue(success, "Tactical process should start successfully")

        # Cleanup
        strategic_process.terminate()
        tactical_process.terminate()

    def test_conversation_context_generation_with_pokemon_prompts(self):
        """Test conversation context generation works with Pokemon-specific prompts."""
        state = ConversationState(ProcessType.SONNET_TACTICAL, 1)
        sonnet_prompt = PromptRepository.get_prompt(ProcessType.SONNET_TACTICAL)

        # Initialize with Pokemon speedrun prompt
        state.initialize_conversation(sonnet_prompt)

        # Add some Pokemon speedrun-related message exchanges
        speedrun_messages = [
            (
                "Generate input sequence for Pallet Town to Viridian City optimal route",
                "A_START A A A A RIGHT RIGHT RIGHT UP UP LEFT",
            ),
            (
                "Optimize Brock fight with minimal PP usage",
                "Use Bubble spam strategy with damage range manipulation",
            ),
            (
                "Calculate frame-perfect timing for Mt. Moon skip",
                "Input sequence requires 180-frame window with 2-frame precision",
            ),
        ]

        for message, response in speedrun_messages:
            state.record_message_exchange(message, response, 150.0)

        # Generate conversation context
        context = state.get_conversation_context(recent_turns=3)
        self.assertIsInstance(context, str, "Context should be generated as string")

        # Context should be valid JSON with Pokemon speedrun content
        import json

        context_data = json.loads(context)

        self.assertIn("conversation_id", context_data)
        self.assertIn("process_type", context_data)
        self.assertEqual(context_data["process_type"], "sonnet_tactical")
        self.assertIn("recent_exchanges", context_data)

        # Verify Pokemon-specific content in recent exchanges
        exchanges = context_data["recent_exchanges"]
        self.assertGreater(len(exchanges), 0, "Should have recent exchanges")

        # Check that exchanges contain Pokemon speedrun content
        exchange_text = str(exchanges)
        pokemon_terms = ["route", "input sequence", "frame", "optimal"]
        for term in pokemon_terms:
            self.assertIn(
                term,
                exchange_text.lower(),
                f"Conversation context should contain Pokemon speedrun term: {term}",
            )


@pytest.mark.fast
class TestSystemPromptValidation(unittest.TestCase):
    """Test system prompt validation and effectiveness."""

    def test_all_process_types_have_prompts(self):
        """Test that all defined process types have corresponding prompts."""
        for process_type in ProcessType:
            prompt = PromptRepository.get_prompt(process_type)
            self.assertIsInstance(prompt, str, f"Prompt for {process_type.value} should be string")
            self.assertGreater(
                len(prompt.strip()), 0, f"Prompt for {process_type.value} should not be empty"
            )

    def test_prompt_validation_functionality(self):
        """Test prompt validation methods work correctly."""
        # Test valid prompts
        for process_type in ProcessType:
            is_valid = PromptRepository.validate_prompt(process_type)
            self.assertTrue(is_valid, f"Prompt for {process_type.value} should be valid")

        # Test available types
        available_types = PromptRepository.get_available_types()
        self.assertEqual(
            len(available_types), len(ProcessType), "Should have prompts for all process types"
        )

        for process_type in ProcessType:
            self.assertIn(
                process_type, available_types, f"{process_type.value} should be in available types"
            )

    def test_prompt_template_metadata(self):
        """Test that prompt templates include proper metadata."""
        for process_type in ProcessType:
            template = PromptRepository.get_prompt_template(process_type)

            self.assertIsNotNone(template.content, "Template should have content")
            self.assertIsNotNone(template.description, "Template should have description")
            self.assertEqual(
                template.process_type, process_type, "Template should have correct process type"
            )

            # Description should be informative
            self.assertGreater(
                len(template.description), 20, "Template description should be informative"
            )

            # Description should mention Pokemon speedrun context
            self.assertIn(
                "Pokemon", template.description, "Template description should mention Pokemon"
            )


if __name__ == "__main__":
    # Run with verbose output to see Pokemon speedrun prompt validation
    unittest.main(verbosity=2)
