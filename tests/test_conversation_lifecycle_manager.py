"""
Test suite for ConversationLifecycleManager.

Comprehensive tests for strategic conversation lifecycle management including
turn tracking, context compression, conversation restarts, and performance
validation following Clean Code testing principles.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from claudelearnspokemon.conversation_lifecycle_manager import (
    ConversationLifecycleManager,
    ConversationType,
    ConversationConfig,
    ConversationState,
    CompressionMetrics
)


class TestConversationLifecycleManagerBasics:
    """Test basic lifecycle manager functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.manager = ConversationLifecycleManager()
    
    def test_initialization(self):
        """Test manager initializes correctly."""
        assert self.manager is not None
        assert isinstance(self.manager._conversations, dict)
        assert isinstance(self.manager._configs, dict)
        assert len(self.manager._conversations) == 0
        assert len(self.manager._configs) == 0
    
    def test_register_opus_conversation(self):
        """Test registering Opus strategic conversation."""
        result = self.manager.register_conversation(
            "opus-001", 
            ConversationType.OPUS_STRATEGIC
        )
        
        assert result is True
        assert "opus-001" in self.manager._conversations
        assert "opus-001" in self.manager._configs
        
        config = self.manager._configs["opus-001"]
        assert config.conversation_id == "opus-001"
        assert config.conversation_type == ConversationType.OPUS_STRATEGIC
        assert config.max_turns == 100
        assert config.compression_threshold == 90
    
    def test_register_sonnet_conversation(self):
        """Test registering Sonnet tactical conversation."""
        result = self.manager.register_conversation(
            "sonnet-001", 
            ConversationType.SONNET_TACTICAL
        )
        
        assert result is True
        
        config = self.manager._configs["sonnet-001"]
        assert config.conversation_type == ConversationType.SONNET_TACTICAL
        assert config.max_turns == 20
        assert config.compression_threshold == 18
    
    def test_register_duplicate_conversation_fails(self):
        """Test that registering duplicate conversation ID fails."""
        # Register first conversation
        result1 = self.manager.register_conversation(
            "test-001", 
            ConversationType.OPUS_STRATEGIC
        )
        assert result1 is True
        
        # Attempt to register duplicate
        result2 = self.manager.register_conversation(
            "test-001", 
            ConversationType.SONNET_TACTICAL
        )
        assert result2 is False
        
        # Verify original configuration unchanged
        config = self.manager._configs["test-001"]
        assert config.conversation_type == ConversationType.OPUS_STRATEGIC
    
    def test_unregister_conversation(self):
        """Test unregistering a conversation."""
        self.manager.register_conversation("test-001", ConversationType.OPUS_STRATEGIC)
        
        result = self.manager.unregister_conversation("test-001")
        assert result is True
        assert "test-001" not in self.manager._conversations
        assert "test-001" not in self.manager._configs
        
        # Test unregistering non-existent conversation
        result2 = self.manager.unregister_conversation("non-existent")
        assert result2 is False


class TestTurnTracking:
    """Test turn counting and tracking functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.manager = ConversationLifecycleManager()
        self.manager.register_conversation("test-conv", ConversationType.OPUS_STRATEGIC)
    
    def test_initial_turn_count_is_zero(self):
        """Test new conversation starts with zero turn count."""
        count = self.manager.get_turn_count("test-conv")
        assert count == 0
    
    def test_increment_turn_count_basic(self):
        """Test basic turn count increment."""
        new_count = self.manager.increment_turn_count("test-conv")
        assert new_count == 1
        
        # Verify get_turn_count returns same value
        count = self.manager.get_turn_count("test-conv")
        assert count == 1
    
    def test_increment_turn_count_multiple(self):
        """Test multiple turn count increments."""
        for expected_count in range(1, 6):
            new_count = self.manager.increment_turn_count("test-conv")
            assert new_count == expected_count
    
    def test_increment_turn_count_performance(self):
        """Test turn increment meets <1ms performance target."""
        start_time = time.perf_counter()
        
        # Perform 100 increments to test performance
        for _ in range(100):
            self.manager.increment_turn_count("test-conv")
        
        total_time = time.perf_counter() - start_time
        avg_time_ms = (total_time / 100) * 1000
        
        # Each increment should be <1ms
        assert avg_time_ms < 1.0, f"Average increment time {avg_time_ms:.3f}ms exceeds 1ms target"
    
    def test_get_turn_count_nonexistent_conversation(self):
        """Test getting turn count for non-existent conversation returns -1."""
        count = self.manager.get_turn_count("non-existent")
        assert count == -1
    
    def test_increment_turn_count_nonexistent_conversation(self):
        """Test incrementing turn count for non-existent conversation returns -1."""
        result = self.manager.increment_turn_count("non-existent")
        assert result == -1
    
    def test_turn_count_thread_safety(self):
        """Test turn counting is thread-safe under concurrent access."""
        num_threads = 10
        increments_per_thread = 100
        threads = []
        
        def increment_worker():
            for _ in range(increments_per_thread):
                self.manager.increment_turn_count("test-conv")
        
        # Start all threads
        for _ in range(num_threads):
            thread = threading.Thread(target=increment_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify final count is correct
        final_count = self.manager.get_turn_count("test-conv")
        expected_count = num_threads * increments_per_thread
        assert final_count == expected_count


class TestCompressionThresholds:
    """Test context compression threshold management."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.manager = ConversationLifecycleManager()
        self.manager.register_conversation("opus-conv", ConversationType.OPUS_STRATEGIC)
        self.manager.register_conversation("sonnet-conv", ConversationType.SONNET_TACTICAL)
    
    def test_should_compress_context_below_threshold(self):
        """Test compression not needed below threshold."""
        # Opus conversation below threshold (90 turns)
        for _ in range(89):
            self.manager.increment_turn_count("opus-conv")
        
        assert not self.manager.should_compress_context("opus-conv")
        
        # Sonnet conversation below threshold (18 turns)
        for _ in range(17):
            self.manager.increment_turn_count("sonnet-conv")
        
        assert not self.manager.should_compress_context("sonnet-conv")
    
    def test_should_compress_context_at_threshold(self):
        """Test compression needed at threshold."""
        # Opus conversation at threshold (90 turns)
        for _ in range(90):
            self.manager.increment_turn_count("opus-conv")
        
        assert self.manager.should_compress_context("opus-conv")
        
        # Sonnet conversation at threshold (18 turns)
        for _ in range(18):
            self.manager.increment_turn_count("sonnet-conv")
        
        assert self.manager.should_compress_context("sonnet-conv")
    
    def test_should_compress_context_above_threshold(self):
        """Test compression needed above threshold."""
        # Opus conversation above threshold
        for _ in range(95):
            self.manager.increment_turn_count("opus-conv")
        
        assert self.manager.should_compress_context("opus-conv")
    
    def test_should_compress_context_nonexistent_conversation(self):
        """Test compression check for non-existent conversation returns False."""
        result = self.manager.should_compress_context("non-existent")
        assert result is False


class TestContextCompression:
    """Test context compression and conversation restart functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.manager = ConversationLifecycleManager()
        self.manager.register_conversation("test-conv", ConversationType.OPUS_STRATEGIC)
        
        # Set up conversation at compression threshold
        for _ in range(90):
            self.manager.increment_turn_count("test-conv")
    
    def test_compress_and_restart_basic(self):
        """Test basic context compression and restart."""
        context = "This is a long conversation context " * 100
        critical_elements = ["important strategy", "key decision"]
        
        result = self.manager.compress_and_restart(
            "test-conv",
            context,
            critical_elements
        )
        
        assert result["success"] is True
        assert "compressed_content" in result
        assert result["compression_ratio"] > 0
        assert result["new_turn_count"] == 0
        assert result["compression_count"] == 1
        
        # Verify turn count was reset
        turn_count = self.manager.get_turn_count("test-conv")
        assert turn_count == 0
    
    def test_compress_and_restart_performance(self):
        """Test compression meets <500ms performance target."""
        context = "Large conversation context " * 1000
        critical_elements = ["strategy", "decision", "outcome"]
        
        start_time = time.perf_counter()
        result = self.manager.compress_and_restart(
            "test-conv",
            context,
            critical_elements
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        assert result["success"] is True
        assert duration_ms < 500.0, f"Compression took {duration_ms:.2f}ms, exceeds 500ms target"
    
    def test_compress_and_restart_compression_ratio(self):
        """Test compression achieves >70% reduction target."""
        # Create substantial context for meaningful compression
        context = "\n".join([f"Turn {i}: Some conversation content here" for i in range(100)])
        critical_elements = ["critical strategy", "important decision"]
        
        result = self.manager.compress_and_restart(
            "test-conv",
            context,
            critical_elements
        )
        
        assert result["success"] is True
        assert result["compression_ratio"] >= 0.70, f"Compression ratio {result['compression_ratio']:.2%} below 70% target"
    
    def test_compress_and_restart_preserves_critical_elements(self):
        """Test compression preserves critical strategic elements."""
        context = "\n".join([
            "Regular conversation turn 1",
            "CRITICAL: Key strategic decision made here",
            "Regular conversation turn 2", 
            "IMPORTANT: Major breakthrough discovered",
            "Regular conversation turn 3"
        ])
        critical_elements = ["CRITICAL", "IMPORTANT"]
        
        result = self.manager.compress_and_restart(
            "test-conv",
            context,
            critical_elements
        )
        
        compressed_content = result["compressed_content"]
        
        # Verify critical elements are preserved
        for element in critical_elements:
            assert element in compressed_content
    
    def test_compress_and_restart_nonexistent_conversation(self):
        """Test compression fails gracefully for non-existent conversation."""
        result = self.manager.compress_and_restart(
            "non-existent",
            "some context",
            ["critical"]
        )
        
        assert result["success"] is False
        assert "error" in result
    
    def test_multiple_compressions_track_count(self):
        """Test multiple compressions properly track compression count."""
        context = "Context to compress"
        critical_elements = ["important"]
        
        # First compression
        result1 = self.manager.compress_and_restart("test-conv", context, critical_elements)
        assert result1["compression_count"] == 1
        
        # Set up for second compression
        for _ in range(90):
            self.manager.increment_turn_count("test-conv")
        
        # Second compression
        result2 = self.manager.compress_and_restart("test-conv", context, critical_elements)
        assert result2["compression_count"] == 2


class TestMetricsAndMonitoring:
    """Test conversation metrics and monitoring functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.manager = ConversationLifecycleManager()
        self.manager.register_conversation("opus-conv", ConversationType.OPUS_STRATEGIC)
        self.manager.register_conversation("sonnet-conv", ConversationType.SONNET_TACTICAL)
    
    def test_get_conversation_metrics_basic(self):
        """Test getting basic conversation metrics."""
        # Add some turns
        for _ in range(5):
            self.manager.increment_turn_count("opus-conv")
        
        metrics = self.manager.get_conversation_metrics("opus-conv")
        
        assert metrics["conversation_id"] == "opus-conv"
        assert metrics["conversation_type"] == "opus_strategic"
        assert metrics["current_turn_count"] == 5
        assert metrics["total_lifetime_turns"] == 5
        assert metrics["max_turns"] == 100
        assert metrics["compression_threshold"] == 90
        assert metrics["compression_count"] == 0
        assert metrics["turns_until_compression"] == 85
        assert metrics["is_active"] is True
    
    def test_get_conversation_metrics_after_compression(self):
        """Test conversation metrics after compression."""
        # Set up for compression
        for _ in range(90):
            self.manager.increment_turn_count("opus-conv")
        
        # Perform compression
        self.manager.compress_and_restart(
            "opus-conv", 
            "context",
            ["critical"]
        )
        
        # Add more turns after restart
        for _ in range(3):
            self.manager.increment_turn_count("opus-conv")
        
        metrics = self.manager.get_conversation_metrics("opus-conv")
        
        assert metrics["current_turn_count"] == 3
        assert metrics["total_lifetime_turns"] == 93  # 90 + 3
        assert metrics["compression_count"] == 1
        assert metrics["last_compression"] is not None
        assert metrics["turns_until_compression"] == 87  # 90 - 3
    
    def test_get_conversation_metrics_nonexistent(self):
        """Test getting metrics for non-existent conversation."""
        metrics = self.manager.get_conversation_metrics("non-existent")
        assert "error" in metrics
    
    def test_get_system_metrics(self):
        """Test getting system-wide metrics."""
        # Add some activity to conversations
        for _ in range(10):
            self.manager.increment_turn_count("opus-conv")
        
        for _ in range(5):
            self.manager.increment_turn_count("sonnet-conv")
        
        system_metrics = self.manager.get_system_metrics()
        
        assert system_metrics["total_conversations"] == 2
        assert system_metrics["active_conversations"] == 2
        assert system_metrics["total_lifetime_turns"] == 15
        assert system_metrics["total_compressions"] == 0
        assert system_metrics["compression_operations"] == 0
    
    def test_get_system_metrics_with_compressions(self):
        """Test system metrics with compression operations."""
        # Set up and perform compression
        for _ in range(90):
            self.manager.increment_turn_count("opus-conv")
        
        # Use larger context for meaningful compression
        large_context = "This is a large conversation context with many turns and strategic discussions. " * 20
        self.manager.compress_and_restart("opus-conv", large_context, ["critical"])
        
        system_metrics = self.manager.get_system_metrics()
        
        assert system_metrics["total_compressions"] == 1
        assert system_metrics["compression_operations"] == 1
        assert system_metrics["average_compression_ratio"] > 0


class TestImmutableDataStructures:
    """Test that data structures maintain immutability for thread safety."""
    
    def test_conversation_config_immutable(self):
        """Test ConversationConfig is properly immutable."""
        config = ConversationConfig(
            conversation_id="test",
            conversation_type=ConversationType.OPUS_STRATEGIC
        )
        
        # Attempt to modify should raise AttributeError
        with pytest.raises(AttributeError):
            config.conversation_id = "modified"
        
        with pytest.raises(AttributeError):
            config.max_turns = 200
    
    def test_conversation_state_immutable(self):
        """Test ConversationState is properly immutable."""
        state = ConversationState(
            conversation_id="test",
            current_turn_count=5,
            total_lifetime_turns=10
        )
        
        # Attempt to modify should raise AttributeError
        with pytest.raises(AttributeError):
            state.current_turn_count = 10
        
        with pytest.raises(AttributeError):
            state.total_lifetime_turns = 20
    
    def test_compression_metrics_immutable(self):
        """Test CompressionMetrics is properly immutable."""
        metrics = CompressionMetrics(
            compression_timestamp=time.time(),
            original_context_size=1000,
            compressed_context_size=300,
            compression_ratio=0.7,
            compression_duration_ms=50.0,
            critical_elements_preserved=5
        )
        
        # Attempt to modify should raise AttributeError
        with pytest.raises(AttributeError):
            metrics.compression_ratio = 0.8
        
        with pytest.raises(AttributeError):
            metrics.compression_duration_ms = 100.0


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.manager = ConversationLifecycleManager()
    
    def test_operations_on_empty_manager(self):
        """Test operations when no conversations are registered."""
        assert self.manager.get_turn_count("any-id") == -1
        assert self.manager.increment_turn_count("any-id") == -1
        assert not self.manager.should_compress_context("any-id")
        
        result = self.manager.compress_and_restart("any-id", "context", [])
        assert result["success"] is False
    
    def test_concurrent_registration_and_operations(self):
        """Test thread safety during concurrent registration and operations."""
        results = []
        thread_counter = 0
        counter_lock = threading.Lock()
        
        def register_and_increment():
            nonlocal thread_counter
            with counter_lock:
                thread_counter += 1
                conv_id = f"conv-{thread_counter}"
            
            success = self.manager.register_conversation(
                conv_id, 
                ConversationType.SONNET_TACTICAL
            )
            results.append(("register", success))
            
            if success:
                for _ in range(10):
                    count = self.manager.increment_turn_count(conv_id)
                    results.append(("increment", count))
        
        # Start multiple concurrent threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=register_and_increment)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all registrations succeeded and increments worked
        register_results = [r[1] for r in results if r[0] == "register"]
        assert all(register_results), "Some registrations failed"
        
        increment_results = [r[1] for r in results if r[0] == "increment"]
        assert all(count > 0 for count in increment_results), "Some increments failed"


# Integration test to verify expected interface for benchmark compatibility
class TestBenchmarkCompatibility:
    """Test compatibility with existing benchmark expectations."""
    
    def test_provides_expected_interface_for_benchmarks(self):
        """Test that ConversationLifecycleManager provides expected interface."""
        manager = ConversationLifecycleManager()
        manager.register_conversation("test-worker", ConversationType.SONNET_TACTICAL)
        
        # Increment some turns
        manager.increment_turn_count("test-worker")
        manager.increment_turn_count("test-worker")
        manager.increment_turn_count("test-worker")
        manager.increment_turn_count("test-worker")
        manager.increment_turn_count("test-worker")
        
        # This should match the expected interface from benchmark_health_check_performance.py
        # where mock_process.conversation_manager.get_turn_count.return_value = 5
        turn_count = manager.get_turn_count("test-worker")
        assert turn_count == 5
        
        # Test method exists and is callable
        assert hasattr(manager, 'get_turn_count')
        assert callable(getattr(manager, 'get_turn_count'))