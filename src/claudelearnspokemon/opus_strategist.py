"""
OpusStrategist - Strategic Planning with Claude Opus

Production-ready strategic planning component that processes responses
from Claude Opus with comprehensive error handling, caching, and
circuit breaker patterns for high availability.
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .circuit_breaker import CircuitBreaker, CircuitConfig
from .claude_code_manager import ClaudeCodeManager
from .opus_strategist_exceptions import (
    DirectiveExtractionError,
    MalformedResponseError,
    OpusStrategistError,
    ResponseTimeoutError,
    StrategyValidationError,
)
from .strategy_response import FallbackStrategy, StrategyResponse
from .strategy_response_cache import ResponseCache
from .strategy_response_parser import StrategyResponseParser, ValidationRule

logger = logging.getLogger(__name__)


class StrategyPriority(Enum):
    """Strategic request priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class SummarizationStrategy(Enum):
    """Strategy types for conversation context summarization."""

    COMPREHENSIVE = "comprehensive"  # Maximum detail preservation
    STRATEGIC = "strategic"  # Focus on strategic insights only
    TACTICAL = "tactical"  # Focus on tactical patterns only
    MINIMAL = "minimal"  # Aggressive compression for space constraints


@dataclass
class StrategyRequest:
    """Structured request for strategic planning."""

    game_state: dict[str, Any]
    context: dict[str, Any]
    priority: StrategyPriority = StrategyPriority.NORMAL
    timeout_override: float | None = None
    cache_ttl_override: float | None = None


@dataclass(frozen=True)
class ConversationSummary:
    """Immutable conversation summary for context compression."""

    summary_id: str
    strategy: SummarizationStrategy
    total_messages: int
    preserved_insights: list[str]
    critical_discoveries: list[str]
    current_objectives: list[str]
    successful_patterns: list[str]
    compressed_content: str
    compression_ratio: float
    timestamp: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary for serialization."""
        return {
            "summary_id": self.summary_id,
            "strategy": self.strategy.value,
            "total_messages": self.total_messages,
            "preserved_insights": self.preserved_insights,
            "critical_discoveries": self.critical_discoveries,
            "current_objectives": self.current_objectives,
            "successful_patterns": self.successful_patterns,
            "compressed_content": self.compressed_content,
            "compression_ratio": self.compression_ratio,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationSummary":
        """Create summary from dictionary."""
        return cls(
            summary_id=data["summary_id"],
            strategy=SummarizationStrategy(data["strategy"]),
            total_messages=data["total_messages"],
            preserved_insights=data["preserved_insights"],
            critical_discoveries=data["critical_discoveries"],
            current_objectives=data["current_objectives"],
            successful_patterns=data["successful_patterns"],
            compressed_content=data["compressed_content"],
            compression_ratio=data["compression_ratio"],
            timestamp=data["timestamp"],
            metadata=data["metadata"],
        )


@runtime_checkable
class SummarizationProvider(Protocol):
    """Protocol for conversation summarization providers."""

    def summarize_learnings(
        self,
        conversation_history: list[dict[str, Any]],
        max_summary_length: int = 1000,
        strategy: SummarizationStrategy = SummarizationStrategy.COMPREHENSIVE,
    ) -> ConversationSummary:
        """Create comprehensive summary preserving critical strategic discoveries."""
        ...


class OpusStrategist:
    """
    Strategic planning component using Claude Opus.

    Processes game state into strategic responses with experiments,
    insights, and tactical directives. Implements production patterns:
    - Response caching with TTL
    - Circuit breaker for failure protection
    - Graceful degradation with fallback strategies
    - Comprehensive error recovery
    """

    def __init__(
        self,
        claude_manager: ClaudeCodeManager,
        parser_config: ValidationRule | None = None,
        cache_size: int = 100,
        cache_ttl: float = 300.0,
        circuit_breaker_threshold: float = 0.5,
        circuit_breaker_timeout: float = 60.0,
    ):
        """
        Initialize OpusStrategist with production configuration.

        Args:
            claude_manager: ClaudeCodeManager for Opus communication
            parser_config: Custom parser validation rules
            cache_size: Maximum cached responses
            cache_ttl: Cache TTL in seconds
            circuit_breaker_threshold: Failure rate threshold (0.0-1.0)
            circuit_breaker_timeout: Circuit breaker timeout in seconds
        """
        self.claude_manager = claude_manager

        # Initialize components
        self.parser = StrategyResponseParser(
            validation_timeout=5.0,
            max_response_size=10 * 1024 * 1024,  # 10MB
            validation_rules=parser_config,
        )

        self.response_cache = ResponseCache(
            max_size=cache_size, default_ttl=cache_ttl, cleanup_interval=60.0
        )

        self.circuit_breaker = CircuitBreaker(
            config=CircuitConfig(
                failure_threshold=int(10 * circuit_breaker_threshold),  # Convert to count
                recovery_timeout=circuit_breaker_timeout,
                expected_exception_types=(OpusStrategistError,),
            )
        )

        # Performance and reliability metrics
        self.metrics = {
            "total_requests": 0,
            "successful_responses": 0,
            "cache_hits": 0,
            "fallback_responses": 0,
            "circuit_breaker_trips": 0,
            "avg_response_time_ms": 0.0,
            "max_response_time_ms": 0.0,
        }
        self._metrics_lock = threading.Lock()

        logger.info("OpusStrategist initialized with production configuration")

    def get_strategy(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any] | None = None,
        priority: StrategyPriority = StrategyPriority.NORMAL,
        use_cache: bool = True,
    ) -> StrategyResponse:
        """
        Get strategic response for current game state.

        Args:
            game_state: Current game state information
            context: Additional context for strategic planning
            priority: Request priority level
            use_cache: Whether to use cached responses

        Returns:
            StrategyResponse with experiments and directives

        Raises:
            OpusStrategistError: On systematic failures
        """
        start_time = time.time()

        try:
            self._record_metric("total_requests", 1)

            # Generate cache key
            cache_key = self._generate_cache_key(game_state, context)

            # Try cache first (if enabled and not critical priority)
            if use_cache and priority != StrategyPriority.CRITICAL:
                cached_response = self.response_cache.get(cache_key)
                if cached_response:
                    self._record_metric("cache_hits", 1)
                    logger.info(f"Cache hit for strategy request: {cache_key[:16]}...")
                    return cached_response

            # Check circuit breaker
            if not self.circuit_breaker.is_available():
                self._record_metric("circuit_breaker_trips", 1)
                logger.warning("Circuit breaker open, using fallback strategy")
                return self._create_fallback_strategy(game_state, "circuit_breaker_open")

            # Request strategy from Opus
            try:
                raw_response = self._request_opus_strategy(game_state, context, priority)
                parsed_response = self.parser.parse_response(raw_response)

                # Cache successful response
                if use_cache:
                    cache_ttl = self._get_cache_ttl_for_priority(priority)
                    self.response_cache.put(cache_key, parsed_response, cache_ttl)

                # Record success
                self.circuit_breaker.metrics.record_success()
                self._record_metric("successful_responses", 1)

                processing_time = (time.time() - start_time) * 1000
                self._update_response_time_metrics(processing_time)

                logger.info(f"Successfully processed strategy response in {processing_time:.2f}ms")
                return parsed_response

            except (StrategyValidationError, MalformedResponseError, ResponseTimeoutError) as e:
                # Record failure for circuit breaker
                self.circuit_breaker.metrics.record_failure()

                # Use fallback strategy for parsing/validation errors
                logger.warning(f"Strategy processing failed: {str(e)}")
                fallback_reason = type(e).__name__.lower()
                return self._create_fallback_strategy(game_state, fallback_reason)

        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error in get_strategy: {str(e)}")
            return self._create_fallback_strategy(game_state, "unexpected_error")

    def extract_directives(self, strategy_response: StrategyResponse) -> list[str]:
        """
        Extract actionable strategic directives from response.

        Args:
            strategy_response: Parsed strategy response

        Returns:
            List of strategic directives

        Raises:
            DirectiveExtractionError: If directive extraction fails
        """
        try:
            directives = []

            # Extract directives from experiments
            for experiment in strategy_response.experiments:
                directives.extend(experiment.directives)

            # Extract directives from insights
            for insight in strategy_response.strategic_insights:
                if insight.upper().startswith("DIRECTIVE:"):
                    directive = insight[10:].strip()  # Remove 'DIRECTIVE:' prefix
                    directives.append(directive)
                elif "DIRECTIVE" in insight.upper():
                    # Handle embedded directives
                    extracted_directive = self._extract_embedded_directive(insight)
                    if extracted_directive:
                        directives.append(extracted_directive)

            # Remove duplicates while preserving order
            unique_directives = []
            seen = set()
            for directive in directives:
                directive_lower = directive.lower()
                if directive_lower not in seen:
                    unique_directives.append(directive)
                    seen.add(directive_lower)

            logger.debug(f"Extracted {len(unique_directives)} unique directives")
            return unique_directives

        except Exception as e:
            raise DirectiveExtractionError(
                f"Failed to extract directives: {str(e)}", strategy_response.to_dict()
            ) from e

    def get_metrics(self) -> dict[str, Any]:
        """Get strategist performance metrics."""
        with self._metrics_lock:
            base_metrics = self.metrics.copy()

        # Create comprehensive metrics dictionary with mixed types
        metrics: dict[str, Any] = dict(base_metrics)

        # Add component metrics
        metrics["parser_metrics"] = self.parser.get_metrics()
        metrics["cache_metrics"] = self.response_cache.get_stats()
        metrics["circuit_breaker_state"] = self.circuit_breaker.get_state().value

        return metrics

    def summarize_learnings(
        self,
        conversation_history: list[dict[str, Any]],
        max_summary_length: int = 1000,
        strategy: SummarizationStrategy = SummarizationStrategy.COMPREHENSIVE,
    ) -> ConversationSummary:
        """
        Create comprehensive summary preserving critical strategic discoveries.

        Analyzes conversation history to extract and preserve essential context
        for conversation continuity across restarts. Applies SOLID principles
        with strategy pattern for different compression approaches.

        Args:
            conversation_history: List of conversation messages with metadata
            max_summary_length: Maximum length of compressed content
            strategy: Summarization strategy to apply

        Returns:
            ConversationSummary with preserved context for continuity

        Raises:
            OpusStrategistError: On summarization failures
        """
        start_time = time.time()

        try:
            # Generate unique summary ID
            summary_id = self._generate_summary_id(conversation_history, strategy)

            # Extract critical information based on strategy
            extracted_info = self._extract_critical_information(conversation_history, strategy)

            # Compress content while preserving essential context
            compressed_content = self._compress_conversation_content(
                conversation_history, extracted_info, max_summary_length, strategy
            )

            # Calculate compression metrics
            original_length = self._calculate_conversation_length(conversation_history)
            compression_ratio = 1 - (len(compressed_content) / max(original_length, 1))

            # Create immutable summary
            summary = ConversationSummary(
                summary_id=summary_id,
                strategy=strategy,
                total_messages=len(conversation_history),
                preserved_insights=extracted_info["strategic_insights"],
                critical_discoveries=extracted_info["discoveries"],
                current_objectives=extracted_info["objectives"],
                successful_patterns=extracted_info["patterns"],
                compressed_content=compressed_content,
                compression_ratio=compression_ratio,
                timestamp=time.time(),
                metadata={
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "original_length": original_length,
                    "strategy_applied": strategy.value,
                    "compression_effective": compression_ratio > 0.5,
                },
            )

            # Validate summary completeness
            self._validate_summary_completeness(summary, extracted_info)

            # Record metrics
            processing_time = (time.time() - start_time) * 1000
            self._record_summarization_metrics(processing_time, compression_ratio)

            logger.info(
                f"Successfully created {strategy.value} summary "
                f"with {compression_ratio:.2%} compression in {processing_time:.2f}ms"
            )

            return summary

        except Exception as e:
            logger.error(f"Failed to summarize conversation history: {str(e)}")
            raise OpusStrategistError(f"Summarization failed: {str(e)}") from e

    def _request_opus_strategy(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any] | None,
        priority: StrategyPriority,
    ) -> str:
        """Send strategic planning request to Claude Opus."""

        # Build strategic planning prompt
        prompt = self._build_strategic_prompt(game_state, context, priority)

        # Send to Opus via ClaudeCodeManager
        try:
            strategic_process = self.claude_manager.get_strategic_process()
            if strategic_process is None:
                raise ConnectionError("No strategic process available")

            response = strategic_process.send_message(prompt)

            if not response or not response.strip():
                raise MalformedResponseError("Empty response from Opus")

            return response

        except Exception as e:
            logger.error(f"Failed to get response from Opus: {str(e)}")
            raise OpusStrategistError(f"Opus communication failed: {str(e)}") from e

    def _build_strategic_prompt(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any] | None,
        priority: StrategyPriority,
    ) -> str:
        """Build strategic planning prompt for Opus."""

        prompt_parts = [
            "STRATEGIC PLANNING REQUEST",
            "",
            "Current Game State:",
            f"- Location: {game_state.get('location', 'unknown')}",
            f"- Health: {game_state.get('health', 0)}",
            f"- Level: {game_state.get('level', 1)}",
            f"- Pokemon Count: {game_state.get('pokemon_count', 0)}",
            f"- Badges: {game_state.get('badges', 0)}",
            "",
        ]

        if context:
            prompt_parts.extend(
                [
                    "Strategic Context:",
                    *[f"- {key}: {value}" for key, value in context.items()],
                    "",
                ]
            )

        prompt_parts.extend(
            [
                f"Priority: {priority.name}",
                "",
                "Please provide strategic response in JSON format with:",
                "- strategy_id: unique identifier",
                "- experiments: array of parallel experiments with id, name, checkpoint, script_dsl, expected_outcome, priority",
                "- strategic_insights: array of strategic insights and directives",
                "- next_checkpoints: array of recommended checkpoint locations",
                "",
                "Focus on concrete, executable experiments that advance speedrun progress.",
            ]
        )

        return "\n".join(prompt_parts)

    def _generate_cache_key(
        self, game_state: dict[str, Any], context: dict[str, Any] | None
    ) -> str:
        """Generate cache key for request."""
        # Combine game state and context for cache key
        combined_state = dict(game_state)
        if context:
            combined_state.update({"context": context})

        return self.response_cache.generate_cache_key(combined_state)

    def _get_cache_ttl_for_priority(self, priority: StrategyPriority) -> float:
        """Get appropriate cache TTL based on request priority."""
        ttl_map = {
            StrategyPriority.LOW: 600.0,  # 10 minutes
            StrategyPriority.NORMAL: 300.0,  # 5 minutes
            StrategyPriority.HIGH: 120.0,  # 2 minutes
            StrategyPriority.CRITICAL: 60.0,  # 1 minute
        }
        return ttl_map.get(priority, 300.0)

    def _create_fallback_strategy(
        self, game_state: dict[str, Any], reason: str
    ) -> StrategyResponse:
        """Create fallback strategy when Opus communication fails."""
        self._record_metric("fallback_responses", 1)

        logger.info(f"Creating fallback strategy due to: {reason}")

        fallback = FallbackStrategy.create_default_fallback(game_state)

        # Add reason to metadata
        fallback_dict = fallback.to_dict()
        fallback_dict["metadata"]["fallback_reason"] = reason

        return StrategyResponse.from_dict(fallback_dict)

    def _extract_embedded_directive(self, insight: str) -> str | None:
        """Extract directive from insight text containing 'DIRECTIVE'."""
        # Simple pattern matching for embedded directives
        insight_upper = insight.upper()

        if "DIRECTIVE:" in insight_upper:
            # Find the directive part
            directive_start = insight_upper.index("DIRECTIVE:") + 10
            directive_text = insight[directive_start:].strip()

            # Find end of directive (stop at punctuation or new sentence)
            for end_char in [".", "!", "?", ";", "\n"]:
                if end_char in directive_text:
                    directive_text = directive_text[: directive_text.index(end_char)].strip()
                    break

            return directive_text if directive_text else None

        return None

    def _record_metric(self, metric_name: str, value: int) -> None:
        """Thread-safe metric recording."""
        with self._metrics_lock:
            self.metrics[metric_name] += value

    def _update_response_time_metrics(self, response_time_ms: float) -> None:
        """Update response time metrics."""
        with self._metrics_lock:
            current_avg = self.metrics["avg_response_time_ms"]
            current_count = self.metrics["successful_responses"]

            if current_count == 1:
                self.metrics["avg_response_time_ms"] = response_time_ms
            else:
                # Rolling average
                self.metrics["avg_response_time_ms"] = (
                    current_avg * (current_count - 1) + response_time_ms
                ) / current_count

            if response_time_ms > self.metrics["max_response_time_ms"]:
                self.metrics["max_response_time_ms"] = response_time_ms

    # Context Summarization Private Methods - Clean Code Architecture

    def _generate_summary_id(
        self, conversation_history: list[dict[str, Any]], strategy: SummarizationStrategy
    ) -> str:
        """Generate unique identifier for conversation summary."""
        import hashlib

        # Create deterministic hash based on content and strategy
        content_hash = hashlib.sha256()
        content_hash.update(f"{len(conversation_history)}{strategy.value}".encode())
        content_hash.update(str(time.time()).encode())

        return f"summary_{content_hash.hexdigest()[:12]}"

    def _extract_critical_information(
        self, conversation_history: list[dict[str, Any]], strategy: SummarizationStrategy
    ) -> dict[str, list[str]]:
        """Extract critical information based on summarization strategy."""
        extracted: dict[str, list[str]] = {
            "strategic_insights": [],
            "discoveries": [],
            "objectives": [],
            "patterns": [],
        }

        # Apply strategy-specific extraction patterns
        for message in conversation_history:
            content = message.get("content", "")
            role = message.get("role", "")

            if strategy == SummarizationStrategy.COMPREHENSIVE:
                extracted = self._extract_comprehensive_info(content, role, extracted)
            elif strategy == SummarizationStrategy.STRATEGIC:
                extracted = self._extract_strategic_info(content, role, extracted)
            elif strategy == SummarizationStrategy.TACTICAL:
                extracted = self._extract_tactical_info(content, role, extracted)
            elif strategy == SummarizationStrategy.MINIMAL:
                extracted = self._extract_minimal_info(content, role, extracted)

        # Remove duplicates while preserving order
        for key in extracted:
            extracted[key] = self._remove_duplicates_preserve_order(extracted[key])

        return extracted

    def _extract_comprehensive_info(
        self, content: str, role: str, extracted: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Extract comprehensive information from message content."""
        content_lower = content.lower()

        # Strategic insights patterns
        if any(
            keyword in content_lower for keyword in ["strategy", "strategic", "plan", "approach"]
        ):
            if len(content) > 50:  # Substantial strategic content
                extracted["strategic_insights"].append(
                    content[:200] + "..." if len(content) > 200 else content
                )

        # Discovery patterns
        if any(
            keyword in content_lower for keyword in ["discovered", "found", "identified", "learned"]
        ):
            extracted["discoveries"].append(
                content[:150] + "..." if len(content) > 150 else content
            )

        # Objective patterns
        if any(
            keyword in content_lower
            for keyword in ["objective", "goal", "target", "should", "need to"]
        ):
            extracted["objectives"].append(content[:150] + "..." if len(content) > 150 else content)

        # Pattern identification
        if any(
            keyword in content_lower
            for keyword in ["pattern", "sequence", "optimization", "speedrun"]
        ):
            extracted["patterns"].append(content[:150] + "..." if len(content) > 150 else content)

        return extracted

    def _extract_strategic_info(
        self, content: str, role: str, extracted: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Extract strategic-focused information from message content."""
        content_lower = content.lower()

        # Focus only on high-level strategic content
        if any(
            keyword in content_lower
            for keyword in ["strategy", "strategic", "plan", "approach", "direction"]
        ):
            extracted["strategic_insights"].append(
                content[:200] + "..." if len(content) > 200 else content
            )

        if any(keyword in content_lower for keyword in ["objective", "goal", "target"]):
            extracted["objectives"].append(content[:150] + "..." if len(content) > 150 else content)

        return extracted

    def _extract_tactical_info(
        self, content: str, role: str, extracted: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Extract tactical-focused information from message content."""
        content_lower = content.lower()

        # Focus on tactical patterns and discoveries
        if any(
            keyword in content_lower
            for keyword in ["pattern", "sequence", "optimization", "execution"]
        ):
            extracted["patterns"].append(content[:150] + "..." if len(content) > 150 else content)

        if any(
            keyword in content_lower for keyword in ["discovered", "found", "works", "effective"]
        ):
            extracted["discoveries"].append(
                content[:150] + "..." if len(content) > 150 else content
            )

        return extracted

    def _extract_minimal_info(
        self, content: str, role: str, extracted: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Extract minimal critical information for aggressive compression."""
        content_lower = content.lower()

        # Only preserve most critical discoveries and objectives
        if any(
            keyword in content_lower for keyword in ["critical", "important", "key", "essential"]
        ):
            if any(keyword in content_lower for keyword in ["discovered", "found"]):
                extracted["discoveries"].append(
                    content[:100] + "..." if len(content) > 100 else content
                )
            elif any(keyword in content_lower for keyword in ["objective", "goal"]):
                extracted["objectives"].append(
                    content[:100] + "..." if len(content) > 100 else content
                )

        return extracted

    def _compress_conversation_content(
        self,
        conversation_history: list[dict[str, Any]],
        extracted_info: dict[str, list[str]],
        max_length: int,
        strategy: SummarizationStrategy,
    ) -> str:
        """Compress conversation content while preserving essential information."""

        # Build structured summary
        summary_parts = [
            "CONVERSATION SUMMARY",
            f"Strategy: {strategy.value.upper()}",
            f"Total Messages: {len(conversation_history)}",
            "",
        ]

        # Add extracted information sections
        if extracted_info["strategic_insights"]:
            summary_parts.extend(
                [
                    "STRATEGIC INSIGHTS:",
                    *[f"- {insight}" for insight in extracted_info["strategic_insights"][:5]],
                    "",
                ]
            )

        if extracted_info["discoveries"]:
            summary_parts.extend(
                [
                    "CRITICAL DISCOVERIES:",
                    *[f"- {discovery}" for discovery in extracted_info["discoveries"][:5]],
                    "",
                ]
            )

        if extracted_info["objectives"]:
            summary_parts.extend(
                [
                    "CURRENT OBJECTIVES:",
                    *[f"- {objective}" for objective in extracted_info["objectives"][:3]],
                    "",
                ]
            )

        if extracted_info["patterns"]:
            summary_parts.extend(
                [
                    "SUCCESSFUL PATTERNS:",
                    *[f"- {pattern}" for pattern in extracted_info["patterns"][:3]],
                    "",
                ]
            )

        # Join and truncate if needed
        full_summary = "\n".join(summary_parts)

        if len(full_summary) <= max_length:
            return full_summary

        # Truncate while preserving structure
        return full_summary[: max_length - 3] + "..."

    def _calculate_conversation_length(self, conversation_history: list[dict[str, Any]]) -> int:
        """Calculate total character length of conversation history."""
        total_length = 0
        for message in conversation_history:
            content = message.get("content", "")
            total_length += len(content)
        return total_length

    def _validate_summary_completeness(
        self, summary: ConversationSummary, extracted_info: dict[str, list[str]]
    ) -> None:
        """Validate that summary preserves essential information."""
        # Ensure critical information is preserved
        if not summary.compressed_content:
            raise OpusStrategistError("Summary compression produced empty content")

        # Validate compression is effective but not destructive
        if summary.compression_ratio < 0.1:
            logger.warning(
                f"Low compression ratio ({summary.compression_ratio:.2%}) may indicate ineffective summarization"
            )

        # Ensure strategy-specific requirements are met (except for empty conversations)
        if summary.total_messages > 0:
            if summary.strategy == SummarizationStrategy.COMPREHENSIVE:
                if not (summary.preserved_insights or summary.critical_discoveries):
                    raise OpusStrategistError(
                        "Comprehensive strategy must preserve insights or discoveries"
                    )

            elif summary.strategy == SummarizationStrategy.STRATEGIC:
                if not summary.preserved_insights:
                    raise OpusStrategistError("Strategic strategy must preserve strategic insights")

            elif summary.strategy == SummarizationStrategy.TACTICAL:
                if not summary.successful_patterns:
                    raise OpusStrategistError("Tactical strategy must preserve successful patterns")

    def _record_summarization_metrics(
        self, processing_time_ms: float, compression_ratio: float
    ) -> None:
        """Record summarization performance metrics."""
        with self._metrics_lock:
            # Initialize summarization metrics if not present
            if "summarization_requests" not in self.metrics:
                self.metrics.update(
                    {
                        "summarization_requests": 0,
                        "avg_summarization_time_ms": 0.0,
                        "avg_compression_ratio": 0.0,
                        "max_summarization_time_ms": 0.0,
                    }
                )

            # Update counts
            self.metrics["summarization_requests"] += 1
            count = self.metrics["summarization_requests"]

            # Update rolling averages
            current_avg_time = self.metrics["avg_summarization_time_ms"]
            self.metrics["avg_summarization_time_ms"] = (
                current_avg_time * (count - 1) + processing_time_ms
            ) / count

            current_avg_compression = self.metrics["avg_compression_ratio"]
            self.metrics["avg_compression_ratio"] = (
                current_avg_compression * (count - 1) + compression_ratio
            ) / count

            # Update max time
            if processing_time_ms > self.metrics["max_summarization_time_ms"]:
                self.metrics["max_summarization_time_ms"] = processing_time_ms

    def _remove_duplicates_preserve_order(self, items: list[str]) -> list[str]:
        """Remove duplicates from list while preserving order."""
        seen = set()
        result = []
        for item in items:
            item_lower = item.lower()
            if item_lower not in seen:
                result.append(item)
                seen.add(item_lower)
        return result


# Re-export classes for easier imports
__all__ = [
    "OpusStrategist",
    "StrategyPriority",
    "StrategyRequest",
    "StrategyResponse",
    "StrategyResponseParser",
    "ResponseCache",
    "FallbackStrategy",
    "SummarizationStrategy",
    "ConversationSummary",
    "SummarizationProvider",
]
