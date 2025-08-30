"""
Comprehensive test suite for ConversationLifecycleManager.

Test Categories:
- TestConversationLifecycleManagerBasics: Basic functionality and initialization
- TestThreadSafety: Concurrent access and thread safety validation
- TestPersistence: Data persistence and recovery across restarts
- TestPerformance: Performance validation and resource usage
- TestLimitEnforcement: Turn limit validation and alert triggering
- TestCleanupAndMaintenance: Memory management and data cleanup
"""

import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import pytest

from claudelearnspokemon.conversation_lifecycle_manager import (
    ConversationLifecycleManager,
    ConversationTurnMetrics,
    TurnLimitConfiguration,
)
from claudelearnspokemon.prompts import ProcessType


@pytest.mark.fast
@pytest.mark.medium
class TestConversationLifecycleManagerBasics(unittest.TestCase):
    """Test basic ConversationLifecycleManager functionality."""

    def setUp(self):
        """Set up test environment with temporary directory."""
        self.test_dir = tempfile.mkdtemp()
        self.config = TurnLimitConfiguration(
            opus_turn_limit=10,
            sonnet_turn_limit=5,
            alert_threshold_percent=80.0,
            auto_save_interval_seconds=0.5  # Fast saves for testing
        )
        self.manager = ConversationLifecycleManager(
            config=self.config,
            data_dir=self.test_dir
        )

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "manager"):
            self.manager.shutdown()
        
        # Clean up temp directory
        import shutil
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test ConversationLifecycleManager initializes correctly."""
        self.assertEqual(self.manager.config.opus_turn_limit, 10)
        self.assertEqual(self.manager.config.sonnet_turn_limit, 5)
        self.assertEqual(self.manager.config.alert_threshold_percent, 80.0)
        self.assertEqual(len(self.manager.conversation_turns), 0)
        self.assertEqual(self.manager.global_metrics.total_turns, 0)

    def test_turn_metrics_dataclass(self):
        """Test ConversationTurnMetrics dataclass functionality."""
        metrics = ConversationTurnMetrics(
            process_id=1,
            opus_turn_limit=10,
            sonnet_turn_limit=5,
            alert_threshold_percent=80.0
        )
        
        # Test basic properties
        self.assertEqual(metrics.opus_turn_limit, 10)
        self.assertEqual(metrics.sonnet_turn_limit, 5)
        self.assertEqual(metrics.total_turns, 0)
        self.assertEqual(metrics.opus_turns, 0)
        self.assertEqual(metrics.sonnet_turns, 0)
        
        # Test remaining turns calculation
        self.assertEqual(metrics.get_opus_turns_remaining(), 10)
        self.assertEqual(metrics.get_sonnet_turns_remaining(), 5)
        
        # Test alert thresholds
        self.assertEqual(metrics.get_opus_alert_threshold(), 8)  # 80% of 10
        self.assertEqual(metrics.get_sonnet_alert_threshold(), 4)  # 80% of 5
        
        # Test alert conditions
        self.assertFalse(metrics.should_alert_opus())
        self.assertFalse(metrics.should_alert_sonnet())
        
        # Simulate turns and test alerts
        metrics.opus_turns = 8
        metrics.sonnet_turns = 4
        self.assertTrue(metrics.should_alert_opus())
        self.assertTrue(metrics.should_alert_sonnet())

    def test_basic_turn_increment(self):
        """Test basic turn increment functionality."""
        conversation_id = "test_conv_1"
        
        # Increment Opus turns
        metrics = self.manager.increment_turn_count(conversation_id, ProcessType.OPUS_STRATEGIC)
        self.assertEqual(metrics.total_turns, 1)
        self.assertEqual(metrics.opus_turns, 1)
        self.assertEqual(metrics.sonnet_turns, 0)
        
        # Increment Sonnet turns
        metrics = self.manager.increment_turn_count(conversation_id, ProcessType.SONNET_TACTICAL)
        self.assertEqual(metrics.total_turns, 2)
        self.assertEqual(metrics.opus_turns, 1)
        self.assertEqual(metrics.sonnet_turns, 1)
        
        # Check global metrics
        self.assertEqual(self.manager.global_metrics.total_turns, 2)
        self.assertEqual(self.manager.global_metrics.opus_turns, 1)
        self.assertEqual(self.manager.global_metrics.sonnet_turns, 1)

    def test_conversation_metrics_retrieval(self):
        """Test conversation metrics retrieval."""
        conversation_id = "test_conv_metrics"
        
        # Add some turns
        for _ in range(3):
            self.manager.increment_turn_count(conversation_id, ProcessType.OPUS_STRATEGIC)
        for _ in range(2):
            self.manager.increment_turn_count(conversation_id, ProcessType.SONNET_TACTICAL)
        
        # Test specific conversation metrics
        metrics = self.manager.get_conversation_metrics(conversation_id)
        self.assertEqual(metrics["total_turns"], 5)
        self.assertEqual(metrics["opus_turns"], 3)
        self.assertEqual(metrics["sonnet_turns"], 2)
        self.assertEqual(metrics["opus_remaining"], 7)  # 10 - 3
        self.assertEqual(metrics["sonnet_remaining"], 3)  # 5 - 2
        self.assertEqual(metrics["conversation_id"], conversation_id)
        
        # Test global metrics
        global_metrics = self.manager.get_conversation_metrics()
        self.assertEqual(global_metrics["total_turns"], 5)
        self.assertEqual(global_metrics["conversation_id"], "global")
        
        # Test all conversations
        all_metrics = self.manager.get_all_conversation_metrics()
        self.assertIn("global", all_metrics)
        self.assertIn(conversation_id, all_metrics)

    def test_turn_limit_checking(self):
        """Test turn limit checking and alert generation."""
        conversation_id = "test_limits"
        
        # Add turns up to alert threshold
        for _ in range(8):  # 80% of 10 Opus limit
            self.manager.increment_turn_count(conversation_id, ProcessType.OPUS_STRATEGIC)
        
        # Check limits - should need alert
        limit_check = self.manager.check_turn_limits(conversation_id)
        self.assertEqual(limit_check["opus_turns"], 8)
        self.assertTrue(limit_check["opus_alert_needed"])
        self.assertFalse(limit_check["opus_at_limit"])
        
        # Add turns to hard limit
        for _ in range(2):  # Reach limit of 10
            self.manager.increment_turn_count(conversation_id, ProcessType.OPUS_STRATEGIC)
        
        limit_check = self.manager.check_turn_limits(conversation_id)
        self.assertEqual(limit_check["opus_turns"], 10)
        self.assertTrue(limit_check["opus_at_limit"])
        self.assertEqual(limit_check["opus_remaining"], 0)


@pytest.mark.fast
@pytest.mark.medium
class TestThreadSafety(unittest.TestCase):
    """Test thread safety of ConversationLifecycleManager."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = TurnLimitConfiguration(
            opus_turn_limit=1000,  # High limits for concurrency testing
            sonnet_turn_limit=1000,
            auto_save_interval_seconds=0.1
        )
        self.manager = ConversationLifecycleManager(
            config=self.config,
            data_dir=self.test_dir
        )

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "manager"):
            self.manager.shutdown()
        
        import shutil
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_concurrent_turn_increment(self):
        """Test concurrent turn increments are thread-safe."""
        conversation_id = "concurrent_test"
        num_threads = 10
        turns_per_thread = 20
        expected_total = num_threads * turns_per_thread
        
        def worker(thread_id):
            """Worker function for concurrent turn increments."""
            for i in range(turns_per_thread):
                # Alternate between Opus and Sonnet
                if i % 2 == 0:
                    self.manager.increment_turn_count(conversation_id, ProcessType.OPUS_STRATEGIC)
                else:
                    self.manager.increment_turn_count(conversation_id, ProcessType.SONNET_TACTICAL)
        
        # Run concurrent workers
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify final counts
        metrics = self.manager.get_conversation_metrics(conversation_id)
        self.assertEqual(metrics["total_turns"], expected_total)
        
        # Should have roughly equal Opus and Sonnet turns
        self.assertAlmostEqual(metrics["opus_turns"], expected_total // 2, delta=num_threads)
        self.assertAlmostEqual(metrics["sonnet_turns"], expected_total // 2, delta=num_threads)
        
        # Global metrics should match
        global_metrics = self.manager.get_conversation_metrics()
        self.assertEqual(global_metrics["total_turns"], expected_total)

    def test_concurrent_metrics_access(self):
        """Test concurrent access to metrics doesn't cause issues."""
        conversation_id = "metrics_access_test"
        
        # Add some initial data
        for _ in range(50):
            self.manager.increment_turn_count(conversation_id, ProcessType.OPUS_STRATEGIC)
        
        results = []
        errors = []
        
        def reader_worker():
            """Worker that reads metrics concurrently."""
            try:
                for _ in range(100):
                    metrics = self.manager.get_conversation_metrics(conversation_id)
                    all_metrics = self.manager.get_all_conversation_metrics()
                    limit_check = self.manager.check_turn_limits(conversation_id)
                    
                    # Verify consistency
                    self.assertGreaterEqual(metrics["total_turns"], 50)
                    self.assertIn(conversation_id, all_metrics)
                    self.assertIsInstance(limit_check["opus_turns"], int)
                    
                    results.append((metrics, all_metrics, limit_check))
            except Exception as e:
                errors.append(e)
        
        def writer_worker():
            """Worker that writes turns concurrently."""
            try:
                for _ in range(50):
                    self.manager.increment_turn_count(conversation_id, ProcessType.SONNET_TACTICAL)
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)
        
        # Run concurrent readers and writers
        threads = []
        
        # Start readers
        for _ in range(3):
            thread = threading.Thread(target=reader_worker)
            threads.append(thread)
            thread.start()
        
        # Start writers
        for _ in range(2):
            thread = threading.Thread(target=writer_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Errors occurred during concurrent access: {errors}")
        self.assertGreater(len(results), 0)
        
        # Final verification
        final_metrics = self.manager.get_conversation_metrics(conversation_id)
        self.assertGreaterEqual(final_metrics["total_turns"], 100)  # 50 initial + 50*2 writers


@pytest.mark.fast
@pytest.mark.medium  
class TestPersistence(unittest.TestCase):
    """Test data persistence functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = TurnLimitConfiguration(
            opus_turn_limit=20,
            sonnet_turn_limit=10,
            auto_save_interval_seconds=0.1  # Fast saves for testing
        )

    def tearDown(self):
        """Clean up after tests."""
        import shutil
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_data_persistence_across_restarts(self):
        """Test that data persists across manager restarts."""
        conversation_id = "persistence_test"
        
        # Create first manager and add data
        manager1 = ConversationLifecycleManager(
            config=self.config,
            data_dir=self.test_dir
        )
        
        # Add turns
        for _ in range(5):
            manager1.increment_turn_count(conversation_id, ProcessType.OPUS_STRATEGIC)
        for _ in range(3):
            manager1.increment_turn_count(conversation_id, ProcessType.SONNET_TACTICAL)
        
        # Verify data before shutdown
        metrics1 = manager1.get_conversation_metrics(conversation_id)
        global1 = manager1.get_conversation_metrics()
        
        self.assertEqual(metrics1["total_turns"], 8)
        self.assertEqual(metrics1["opus_turns"], 5)
        self.assertEqual(metrics1["sonnet_turns"], 3)
        self.assertEqual(global1["total_turns"], 8)
        
        # Shutdown and create new manager
        manager1.shutdown()
        time.sleep(0.2)  # Allow persistence to complete
        
        manager2 = ConversationLifecycleManager(
            config=self.config,
            data_dir=self.test_dir
        )
        
        # Verify data was restored
        metrics2 = manager2.get_conversation_metrics(conversation_id)
        global2 = manager2.get_conversation_metrics()
        
        # Note: conversation-specific data might not persist for performance reasons
        # but global data should persist
        self.assertEqual(global2["total_turns"], 8)
        self.assertEqual(global2["opus_turns"], 5)
        self.assertEqual(global2["sonnet_turns"], 3)
        
        # Add more data to verify continued functionality
        manager2.increment_turn_count(conversation_id, ProcessType.OPUS_STRATEGIC)
        
        final_global = manager2.get_conversation_metrics()
        self.assertEqual(final_global["total_turns"], 9)
        
        manager2.shutdown()

    def test_persistence_file_format(self):
        """Test that persistence file has correct format."""
        conversation_id = "format_test"
        
        manager = ConversationLifecycleManager(
            config=self.config,
            data_dir=self.test_dir
        )
        
        # Add some data
        for _ in range(3):
            manager.increment_turn_count(conversation_id, ProcessType.OPUS_STRATEGIC)
        
        # Force save
        manager._save_persisted_data()
        
        # Verify file exists and has correct format
        persistence_file = Path(self.test_dir) / self.config.persistence_file
        self.assertTrue(persistence_file.exists())
        
        with open(persistence_file, 'r') as f:
            data = json.load(f)
        
        # Verify structure
        self.assertIn("global_metrics", data)
        self.assertIn("conversations", data)
        
        global_data = data["global_metrics"]
        self.assertEqual(global_data["total_turns"], 3)
        self.assertEqual(global_data["opus_turns"], 3)
        self.assertEqual(global_data["sonnet_turns"], 0)
        self.assertIn("timestamp", global_data)
        
        manager.shutdown()


@pytest.mark.fast
@pytest.mark.medium
class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = TurnLimitConfiguration(
            opus_turn_limit=10000,  # High limits for performance testing
            sonnet_turn_limit=10000,
            auto_save_interval_seconds=10.0  # Less frequent saves for performance
        )
        self.manager = ConversationLifecycleManager(
            config=self.config,
            data_dir=self.test_dir
        )

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "manager"):
            self.manager.shutdown()
        
        import shutil
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_turn_increment_performance(self):
        """Test turn increment performance meets requirements (<1ms)."""
        conversation_id = "performance_test"
        num_iterations = 1000
        
        # Warm up
        for _ in range(10):
            self.manager.increment_turn_count(conversation_id, ProcessType.OPUS_STRATEGIC)
        
        # Measure performance
        start_time = time.time()
        
        for i in range(num_iterations):
            process_type = ProcessType.OPUS_STRATEGIC if i % 2 == 0 else ProcessType.SONNET_TACTICAL
            self.manager.increment_turn_count(conversation_id, process_type)
        
        total_time = time.time() - start_time
        avg_time_per_increment = total_time / num_iterations
        
        # Verify performance target (<1ms per increment)
        self.assertLess(avg_time_per_increment, 0.001, 
                       f"Average turn increment time {avg_time_per_increment*1000:.2f}ms exceeds 1ms target")
        
        print(f"\nTurn increment performance: {avg_time_per_increment*1000:.3f}ms per operation")

    def test_metrics_retrieval_performance(self):
        """Test metrics retrieval performance."""
        # Add data across multiple conversations
        for conv_id in range(50):
            conversation_id = f"perf_conv_{conv_id}"
            for _ in range(20):
                self.manager.increment_turn_count(conversation_id, ProcessType.OPUS_STRATEGIC)
        
        # Measure metrics retrieval
        start_time = time.time()
        
        for _ in range(100):
            all_metrics = self.manager.get_all_conversation_metrics()
            global_metrics = self.manager.get_conversation_metrics()
            limit_checks = self.manager.check_turn_limits()
        
        total_time = time.time() - start_time
        avg_time_per_retrieval = total_time / 100
        
        # Should be fast even with many conversations
        self.assertLess(avg_time_per_retrieval, 0.01, 
                       f"Average metrics retrieval time {avg_time_per_retrieval*1000:.2f}ms is too slow")
        
        print(f"Metrics retrieval performance: {avg_time_per_retrieval*1000:.3f}ms per operation")

    def test_memory_usage_bounds(self):
        """Test memory usage stays within reasonable bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add significant data
        for conv_id in range(100):
            conversation_id = f"memory_test_{conv_id}"
            for _ in range(100):
                process_type = ProcessType.OPUS_STRATEGIC if conv_id % 2 == 0 else ProcessType.SONNET_TACTICAL
                self.manager.increment_turn_count(conversation_id, process_type)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (<5MB for test data)
        self.assertLess(memory_increase, 5.0,
                       f"Memory increase {memory_increase:.1f}MB exceeds 5MB target")
        
        print(f"Memory usage increase: {memory_increase:.1f}MB for 10,000 turn records")


@pytest.mark.fast
@pytest.mark.medium
class TestCleanupAndMaintenance(unittest.TestCase):
    """Test cleanup and maintenance functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = TurnLimitConfiguration(
            opus_turn_limit=20,
            sonnet_turn_limit=10,
            max_conversation_history=50,  # Low limit for testing cleanup
            auto_save_interval_seconds=0.1
        )
        self.manager = ConversationLifecycleManager(
            config=self.config,
            data_dir=self.test_dir
        )

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "manager"):
            self.manager.shutdown()
        
        import shutil
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_old_conversation_cleanup(self):
        """Test cleanup of old conversations."""
        # Create conversations with different ages
        current_time = time.time()
        old_conversations = []
        recent_conversations = []
        
        # Add old conversations (simulate old timestamps)
        for i in range(10):
            conv_id = f"old_conv_{i}"
            old_conversations.append(conv_id)
            self.manager.increment_turn_count(conv_id, ProcessType.OPUS_STRATEGIC)
            # Manually set old timestamp
            self.manager.conversation_turns[conv_id].last_turn_timestamp = current_time - (48 * 3600)  # 48 hours ago
        
        # Add recent conversations
        for i in range(5):
            conv_id = f"recent_conv_{i}"
            recent_conversations.append(conv_id)
            self.manager.increment_turn_count(conv_id, ProcessType.OPUS_STRATEGIC)
            # These will have current timestamps
        
        # Verify initial state
        self.assertEqual(len(self.manager.conversation_turns), 15)
        
        # Cleanup conversations older than 24 hours
        cleaned_count = self.manager.cleanup_old_conversations(max_age_hours=24.0)
        
        # Verify cleanup
        self.assertEqual(cleaned_count, 10, "Should have cleaned up 10 old conversations")
        self.assertEqual(len(self.manager.conversation_turns), 5, "Should have 5 conversations remaining")
        
        # Verify correct conversations remain
        for conv_id in recent_conversations:
            self.assertIn(conv_id, self.manager.conversation_turns, f"Recent conversation {conv_id} should remain")
        
        for conv_id in old_conversations:
            self.assertNotIn(conv_id, self.manager.conversation_turns, f"Old conversation {conv_id} should be cleaned")

    def test_context_manager_cleanup(self):
        """Test that context manager properly cleans up resources."""
        conversation_id = "context_test"
        
        # Use context manager
        with ConversationLifecycleManager(config=self.config, data_dir=self.test_dir) as manager:
            # Add some data
            for _ in range(5):
                manager.increment_turn_count(conversation_id, ProcessType.OPUS_STRATEGIC)
            
            # Verify data exists
            metrics = manager.get_conversation_metrics(conversation_id)
            self.assertEqual(metrics["total_turns"], 5)
        
        # After context exit, manager should be shut down
        # Data should still be persisted
        new_manager = ConversationLifecycleManager(config=self.config, data_dir=self.test_dir)
        global_metrics = new_manager.get_conversation_metrics()
        self.assertEqual(global_metrics["total_turns"], 5)
        new_manager.shutdown()


if __name__ == "__main__":
    # Run with verbose output to see individual test results
    unittest.main(verbosity=2)