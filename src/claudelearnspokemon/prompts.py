"""
System prompts for Claude processes in the Pokemon speedrun learning agent.

This module centralizes all system prompts used by different Claude process types,
following Clean Code principles by separating prompt content from process logic.
"""

from dataclasses import dataclass
from enum import Enum


class ProcessType(Enum):
    """Claude process specialization types."""

    OPUS_STRATEGIC = "opus_strategic"
    SONNET_TACTICAL = "sonnet_tactical"


@dataclass
class PromptTemplate:
    """Container for a system prompt with metadata."""

    content: str
    description: str
    process_type: ProcessType


class PromptRepository:
    """Repository for managing system prompts with type safety and validation."""

    _OPUS_STRATEGIC_PROMPT = """You are a chess grandmaster AI analyzing Pokemon Red speedrun strategies.

Your role is BIG PICTURE PLANNING and PATTERN SYNTHESIS:
- Think 10+ moves ahead like a chess master
- Identify optimal route patterns and sequence dependencies
- Synthesize insights from multiple parallel experiments
- Plan strategic checkpoint placement and resource allocation
- Recognize meta-patterns across different speedrun approaches

Focus on STRATEGIC INTELLIGENCE:
- Route optimization and path planning
- Risk assessment and contingency strategies
- Pattern recognition across gameplay sequences
- Long-term learning and adaptation strategies
- Coordination of multiple tactical experiments

You receive aggregated results from tactical agents and provide high-level guidance."""

    _SONNET_TACTICAL_PROMPT = """You are a frame-perfect speedrunner developing Pokemon Red execution scripts.

Your role is PRECISION EXECUTION and MICRO-OPTIMIZATION:
- Generate exact input sequences like a tool-assisted speedrun
- Optimize for frame-perfect timing and minimal input count
- Develop reusable script patterns and movement sequences
- Focus on immediate tactical problem-solving
- Convert strategic plans into executable input sequences

Focus on TACTICAL PRECISION:
- Script development and DSL pattern creation
- Frame-by-frame execution optimization
- Input sequence generation and validation
- Real-time adaptation to game state changes
- Micro-optimizations for specific gameplay situations

You execute strategic plans with speedrunner precision and report results up."""

    _PROMPTS: dict[ProcessType, PromptTemplate] = {
        ProcessType.OPUS_STRATEGIC: PromptTemplate(
            content=_OPUS_STRATEGIC_PROMPT,
            description="Strategic planning and pattern synthesis for Pokemon Red speedrunning",
            process_type=ProcessType.OPUS_STRATEGIC,
        ),
        ProcessType.SONNET_TACTICAL: PromptTemplate(
            content=_SONNET_TACTICAL_PROMPT,
            description="Tactical execution and script development for Pokemon Red speedrunning",
            process_type=ProcessType.SONNET_TACTICAL,
        ),
    }

    @classmethod
    def get_prompt(cls, process_type: ProcessType) -> str:
        """
        Get system prompt for the specified process type.

        Args:
            process_type: The type of Claude process

        Returns:
            The system prompt string

        Raises:
            KeyError: If process_type is not supported
        """
        if process_type not in cls._PROMPTS:
            raise KeyError(f"No prompt available for process type: {process_type}")

        return cls._PROMPTS[process_type].content

    @classmethod
    def get_prompt_template(cls, process_type: ProcessType) -> PromptTemplate:
        """
        Get full prompt template with metadata for the specified process type.

        Args:
            process_type: The type of Claude process

        Returns:
            PromptTemplate with content, description, and type

        Raises:
            KeyError: If process_type is not supported
        """
        if process_type not in cls._PROMPTS:
            raise KeyError(f"No prompt template available for process type: {process_type}")

        return cls._PROMPTS[process_type]

    @classmethod
    def get_available_types(cls) -> list[ProcessType]:
        """Get list of all available process types with prompts."""
        return list(cls._PROMPTS.keys())

    @classmethod
    def validate_prompt(cls, process_type: ProcessType) -> bool:
        """
        Validate that a prompt exists and is non-empty for the given type.

        Args:
            process_type: The type of Claude process to validate

        Returns:
            True if prompt exists and is valid, False otherwise
        """
        try:
            prompt = cls.get_prompt(process_type)
            return bool(prompt and prompt.strip())
        except KeyError:
            return False


# Backward compatibility constants for existing code
OPUS_STRATEGIC_PROMPT = PromptRepository.get_prompt(ProcessType.OPUS_STRATEGIC)
SONNET_TACTICAL_PROMPT = PromptRepository.get_prompt(ProcessType.SONNET_TACTICAL)
