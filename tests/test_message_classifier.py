"""
Unit Tests for MessageClassifier - Strategic vs Tactical Classification

This module provides comprehensive unit tests for the MessageClassifier system,
validating pattern matching accuracy, performance requirements, circuit breaker
behavior, and production reliability patterns.

Test Coverage:
- Pattern-based classification accuracy
- Performance requirements (<5ms per classification)
- Circuit breaker functionality
- Cache behavior and memory management
- Error handling and graceful degradation
- Metrics collection and reporting
"""

import threading
import time
from unittest.mock import patch

import pytest

from claudelearnspokemon.message_classifier import (
    ClassificationPattern,
    ClassificationResult,
    MessageClassifier,
    MessageType,
    PatternBasedClassifier,
)


@pytest.mark.fast
class TestClassificationPattern:
    """Test ClassificationPattern dataclass functionality."""

    def test_pattern_creation(self):
        """Test basic pattern creation."""
        pattern = ClassificationPattern(
            pattern=r"\bstrategy\b",
            message_type=MessageType.STRATEGIC,
            priority=10,
            description="Strategy pattern",
        )

        assert pattern.pattern == r"\bstrategy\b"
        assert pattern.message_type == MessageType.STRATEGIC
        assert pattern.priority == 10
        assert pattern.description == "Strategy pattern"

    def test_pattern_immutability(self):
        """Test that patterns are immutable (frozen dataclass)."""
        pattern = ClassificationPattern(pattern=r"\btest\b", message_type=MessageType.TACTICAL)

        with pytest.raises(AttributeError):
            pattern.priority = 20  # Should fail due to frozen=True


@pytest.mark.fast
class TestClassificationResult:
    """Test ClassificationResult dataclass functionality."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = ClassificationResult(
            message_type=MessageType.STRATEGIC,
            confidence=0.85,
            matched_patterns=["strategic:planning"],
            processing_time_ms=3.2,
        )

        assert result.message_type == MessageType.STRATEGIC
        assert result.confidence == 0.85
        assert result.matched_patterns == ["strategic:planning"]
        assert result.processing_time_ms == 3.2
        assert not result.fallback_used
        assert result.error_message is None


@pytest.mark.fast
class TestPatternBasedClassifier:
    """Test PatternBasedClassifier functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = PatternBasedClassifier(enable_caching=True)

    def test_strategic_classification(self):
        """Test classification of strategic messages."""
        strategic_messages = [
            "Develop a comprehensive strategy for Pokemon Red speedrun",
            "Analyze the overall route optimization approach",
            "Plan the experiment methodology for route discovery",
            "Design the architecture for the learning system",
        ]

        for message in strategic_messages:
            result = self.classifier.classify(message)
            assert result.message_type == MessageType.STRATEGIC
            assert result.confidence > 0.5
            assert len(result.matched_patterns) > 0
            assert "strategic:" in result.matched_patterns[0]

    def test_tactical_classification(self):
        """Test classification of tactical messages."""
        tactical_messages = [
            "Implement the script compiler module",
            "Debug the Pokemon gym connection issue",
            "Write unit tests for the checkpoint manager",
            "Execute the tile observer function",
            "Fix the bug in the emulator pool",
        ]

        for message in tactical_messages:
            result = self.classifier.classify(message)
            assert result.message_type == MessageType.TACTICAL
            assert result.confidence > 0.5
            assert len(result.matched_patterns) > 0
            assert "tactical:" in result.matched_patterns[0]

    def test_pokemon_domain_patterns(self):
        """Test Pokemon-specific domain patterns."""
        pokemon_strategic = "Optimize the Pokemon speedrun route planning"
        pokemon_tactical = "Execute the pokemon gym tile grid sequence"

        strategic_result = self.classifier.classify(pokemon_strategic)
        tactical_result = self.classifier.classify(pokemon_tactical)

        assert strategic_result.message_type == MessageType.STRATEGIC
        assert tactical_result.message_type == MessageType.TACTICAL

        # Pokemon-specific patterns should have high priority
        assert strategic_result.confidence > 0.8
        assert tactical_result.confidence > 0.8

    def test_performance_requirements(self):
        """Test that classification meets performance requirements (<5ms)."""
        message = "Implement a strategic approach to Pokemon Red optimization"

        # Test multiple classifications to get reliable timing
        times = []
        for _ in range(100):
            start_time = time.time()
            result = self.classifier.classify(message)
            end_time = time.time()

            classification_time = (end_time - start_time) * 1000
            times.append(classification_time)

            # Each classification should be under 5ms
            assert classification_time < 5.0
            assert result.processing_time_ms < 5.0

        # Average should be well under the limit
        avg_time = sum(times) / len(times)
        assert avg_time < 2.0  # Target 2ms average

    def test_caching_behavior(self):
        """Test classification caching for performance."""
        message = "Strategic planning for Pokemon speedrun"

        # First classification (cache miss)
        start_time = time.time()
        result1 = self.classifier.classify(message)
        first_time = (time.time() - start_time) * 1000

        # Second classification (cache hit)
        start_time = time.time()
        result2 = self.classifier.classify(message)
        second_time = (time.time() - start_time) * 1000

        # Results should be identical
        assert result1.message_type == result2.message_type
        assert result1.confidence == result2.confidence

        # Second call should be faster (cache hit)
        assert second_time < first_time
        assert second_time < 1.0  # Cache hits should be very fast

    def test_cache_size_limits(self):
        """Test that cache doesn't grow unbounded."""
        classifier = PatternBasedClassifier(enable_caching=True)

        # Add more than 1000 unique messages (cache limit)
        for i in range(1200):
            unique_message = f"Strategic message number {i} for testing"
            classifier.classify(unique_message)

        # Cache should be limited to prevent memory bloat
        assert len(classifier._classification_cache) <= 1000

    def test_context_hints(self):
        """Test classification with context hints."""
        ambiguous_message = "Optimize the system performance"

        strategic_context = {"requires_planning": True}
        tactical_context = {"requires_implementation": True}

        strategic_result = self.classifier.classify(ambiguous_message, strategic_context)
        tactical_result = self.classifier.classify(ambiguous_message, tactical_context)

        # Context should influence classification
        assert strategic_result.message_type == MessageType.STRATEGIC
        assert tactical_result.message_type == MessageType.TACTICAL

    def test_fallback_behavior(self):
        """Test fallback behavior for unmatched patterns."""
        unclear_message = "xyz abc def ghi"  # No meaningful patterns

        result = self.classifier.classify(unclear_message)

        # Should default to tactical with low confidence
        assert result.message_type == MessageType.TACTICAL
        assert result.confidence < 0.5
        assert result.fallback_used
        assert "fallback:" in result.matched_patterns[0]

    def test_custom_pattern_addition(self):
        """Test adding custom classification patterns."""
        custom_pattern = ClassificationPattern(
            pattern=r"\bcustom.*pattern\b",
            message_type=MessageType.STRATEGIC,
            priority=15,
            description="Custom test pattern",
        )

        success = self.classifier.add_pattern(custom_pattern)
        assert success

        # Test that custom pattern works
        result = self.classifier.classify("Use custom pattern for testing")
        assert result.message_type == MessageType.STRATEGIC
        assert "Custom test pattern" in str(result.matched_patterns)

    def test_invalid_pattern_handling(self):
        """Test handling of invalid regex patterns."""
        invalid_pattern = ClassificationPattern(
            pattern=r"[invalid regex pattern",  # Missing closing bracket
            message_type=MessageType.STRATEGIC,
            priority=10,
            description="Invalid pattern",
        )

        success = self.classifier.add_pattern(invalid_pattern)
        assert not success  # Should fail gracefully

    def test_metrics_collection(self):
        """Test metrics collection functionality."""
        messages = [
            "Strategic planning approach",
            "Implement the tactical solution",
            "Debug the code issue",
            "Analyze the strategic options",
        ]

        initial_metrics = self.classifier.get_metrics()
        initial_total = initial_metrics["total_classifications"]

        for message in messages:
            self.classifier.classify(message)

        final_metrics = self.classifier.get_metrics()

        # Metrics should be updated
        assert final_metrics["total_classifications"] == initial_total + len(messages)
        assert final_metrics["strategic_classifications"] >= 2  # At least 2 strategic
        assert final_metrics["tactical_classifications"] >= 2  # At least 2 tactical
        assert final_metrics["avg_classification_time_ms"] > 0

    def test_thread_safety(self):
        """Test thread safety of classifier operations."""
        message = "Strategic Pokemon speedrun planning"
        results = []

        def classify_worker():
            for _ in range(50):
                result = self.classifier.classify(message)
                results.append(result)

        # Run multiple threads concurrently
        threads = []
        for _ in range(4):
            thread = threading.Thread(target=classify_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All results should be consistent
        assert len(results) == 200  # 4 threads * 50 classifications
        first_result = results[0]
        for result in results:
            assert result.message_type == first_result.message_type
            assert result.confidence == first_result.confidence


@pytest.mark.fast
class TestMessageClassifier:
    """Test main MessageClassifier with circuit breaker functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = MessageClassifier()

    def test_basic_classification(self):
        """Test basic classification functionality."""
        strategic_message = "Plan the strategic approach"
        tactical_message = "Implement the function"

        strategic_result = self.classifier.classify_message(strategic_message)
        tactical_result = self.classifier.classify_message(tactical_message)

        assert strategic_result.message_type == MessageType.STRATEGIC
        assert tactical_result.message_type == MessageType.TACTICAL
        assert not strategic_result.fallback_used
        assert not tactical_result.fallback_used

    def test_circuit_breaker_functionality(self):
        """Test circuit breaker behavior on failures."""
        # Mock the primary strategy to fail
        with patch.object(self.classifier.primary_strategy, "classify") as mock_classify:
            mock_classify.side_effect = Exception("Classification failed")

            # Trigger circuit breaker by exceeding failure threshold
            for _ in range(6):  # Threshold is 5
                result = self.classifier.classify_message("test message")
                assert result.fallback_used
                assert result.message_type == MessageType.TACTICAL  # Safe default

            # Circuit breaker should now be open
            health = self.classifier.get_health_status()
            assert health["circuit_breaker_open"]
            assert health["failure_count"] >= 5

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        # Set a very short timeout for testing
        self.classifier._circuit_breaker_timeout = 0.1

        # Mock the primary strategy to fail initially
        with patch.object(self.classifier.primary_strategy, "classify") as mock_classify:
            mock_classify.side_effect = Exception("Classification failed")

            # Trigger circuit breaker
            for _ in range(6):
                self.classifier.classify_message("test message")

            assert self.classifier._circuit_open

            # Wait for timeout
            time.sleep(0.15)

            # Mock strategy to succeed now
            mock_classify.side_effect = None
            mock_classify.return_value = ClassificationResult(
                message_type=MessageType.STRATEGIC, confidence=0.8
            )

            # Circuit breaker should reset and allow normal operation
            result = self.classifier.classify_message("test message")
            assert not result.fallback_used
            assert not self.classifier._circuit_open

    def test_health_status_reporting(self):
        """Test health status reporting."""
        health = self.classifier.get_health_status()

        required_fields = [
            "circuit_breaker_open",
            "failure_count",
            "metrics",
            "primary_strategy_metrics",
        ]

        for field in required_fields:
            assert field in health

        # Metrics should include basic counters
        metrics = health["metrics"]
        assert "total_requests" in metrics
        assert "successful_classifications" in metrics
        assert "failed_classifications" in metrics

    def test_manual_circuit_breaker_reset(self):
        """Test manual reset of circuit breaker."""
        # Force circuit breaker open
        self.classifier._circuit_open = True
        self.classifier._failure_count = 10

        # Reset manually
        self.classifier.reset_circuit_breaker()

        assert not self.classifier._circuit_open
        assert self.classifier._failure_count == 0

    def test_error_handling_with_context(self):
        """Test error handling with various context scenarios."""
        # Test with invalid context
        result = self.classifier.classify_message(
            "test message", context={"invalid": object()}  # Non-serializable context
        )

        # Should still work despite invalid context
        assert result.message_type in [
            MessageType.STRATEGIC,
            MessageType.TACTICAL,
            MessageType.UNKNOWN,
        ]

    @pytest.mark.parametrize(
        "message,expected_type",
        [
            ("Strategic planning for optimization", MessageType.STRATEGIC),
            ("Implement the debugging function", MessageType.TACTICAL),
            ("Pokemon speedrun route analysis", MessageType.STRATEGIC),
            ("Execute pokemon gym sequence", MessageType.TACTICAL),
            ("", MessageType.TACTICAL),  # Empty message defaults to tactical
        ],
    )
    def test_classification_accuracy(self, message, expected_type):
        """Test classification accuracy with various message types."""
        result = self.classifier.classify_message(message)
        assert result.message_type == expected_type


@pytest.mark.fast
class TestPerformanceRequirements:
    """Test performance requirements and SLA compliance."""

    def setup_method(self):
        """Set up performance test fixtures."""
        self.classifier = MessageClassifier()

    def test_classification_sla_compliance(self):
        """Test that classification meets SLA requirements."""
        messages = [
            "Strategic Pokemon Red speedrun planning approach",
            "Implement tactical script development function",
            "Debug the emulator connection issue immediately",
            "Analyze route optimization patterns comprehensively",
            "Execute the checkpoint save sequence efficiently",
        ]

        # Test performance under concurrent load
        results = []
        start_time = time.time()

        for message in messages * 20:  # 100 total classifications
            result = self.classifier.classify_message(message)
            results.append(result)

        total_time = (time.time() - start_time) * 1000
        avg_time_per_classification = total_time / len(results)

        # Performance requirements
        assert avg_time_per_classification < 5.0  # <5ms per classification
        assert all(r.processing_time_ms < 5.0 for r in results if r.processing_time_ms)
        assert (
            len([r for r in results if r.error_message is None]) >= len(results) * 0.95
        )  # 95% success rate

    def test_memory_usage_limits(self):
        """Test memory usage stays within limits."""
        import sys

        # Get initial memory usage
        initial_size = sys.getsizeof(self.classifier)

        # Classify many unique messages to test memory growth
        for i in range(2000):
            message = f"Strategic message {i} for memory testing purposes"
            self.classifier.classify_message(message)

        # Memory growth should be controlled
        final_size = sys.getsizeof(self.classifier)
        memory_growth = final_size - initial_size

        # Should not exceed 10MB additional memory as specified
        assert memory_growth < 10 * 1024 * 1024  # 10MB limit

    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        import concurrent.futures

        def classification_worker(message_prefix):
            results = []
            for i in range(25):
                message = f"{message_prefix} message {i}"
                result = self.classifier.classify_message(message)
                results.append(result)
            return results

        # Run concurrent classifications
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(classification_worker, f"Strategic worker {i}") for i in range(4)
            ]

            all_results = []
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())

        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(all_results)

        # Concurrent performance should still meet SLA
        assert avg_time < 10.0  # Allow slightly higher latency under concurrent load
        assert len(all_results) == 100  # 4 workers * 25 messages
        assert all(
            r.message_type in [MessageType.STRATEGIC, MessageType.TACTICAL] for r in all_results
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
