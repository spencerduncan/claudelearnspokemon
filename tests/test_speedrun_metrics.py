"""
Tests for Pokemon speedrun application-specific metrics.

Validates experiment tracking, pattern discovery metrics,
SLA compliance monitoring, and performance measurement.
"""

import time
import threading
import unittest
from unittest.mock import MagicMock

import pytest

from claudelearnspokemon.speedrun_metrics import (
    ExperimentStatus,
    PatternType,
    ExperimentResult,
    PatternDiscovery,
    SpeedrunMetricsSnapshot,
    SpeedrunMetricsCollector
)


class TestExperimentResult(unittest.TestCase):
    """Test cases for ExperimentResult data structure."""
    
    def test_initialization(self):
        """Test ExperimentResult initialization."""
        result = ExperimentResult(
            experiment_id="exp_001",
            status=ExperimentStatus.SUCCESS,
            duration_seconds=125.5,
            script_length=450,
            script_compilation_time_ms=85.2,
            checkpoint_loading_time_ms=320.1,
            ai_strategy="genetic_algorithm",
            pattern_discoveries=3,
            performance_score=0.85
        )
        
        self.assertEqual(result.experiment_id, "exp_001")
        self.assertEqual(result.status, ExperimentStatus.SUCCESS)
        self.assertEqual(result.duration_seconds, 125.5)
        self.assertEqual(result.script_length, 450)
        self.assertEqual(result.script_compilation_time_ms, 85.2)
        self.assertEqual(result.checkpoint_loading_time_ms, 320.1)
        self.assertEqual(result.ai_strategy, "genetic_algorithm")
        self.assertEqual(result.pattern_discoveries, 3)
        self.assertEqual(result.performance_score, 0.85)
        self.assertIsNone(result.error_message)
        self.assertIsInstance(result.timestamp, float)


class TestPatternDiscovery(unittest.TestCase):
    """Test cases for PatternDiscovery data structure."""
    
    def test_initialization(self):
        """Test PatternDiscovery initialization."""
        pattern = PatternDiscovery(
            pattern_id="pattern_001",
            pattern_type=PatternType.MOVEMENT,
            quality_score=0.92,
            discovery_time_seconds=45.3,
            experiment_id="exp_001",
            ai_worker="worker_3"
        )
        
        self.assertEqual(pattern.pattern_id, "pattern_001")
        self.assertEqual(pattern.pattern_type, PatternType.MOVEMENT)
        self.assertEqual(pattern.quality_score, 0.92)
        self.assertEqual(pattern.discovery_time_seconds, 45.3)
        self.assertEqual(pattern.experiment_id, "exp_001")
        self.assertEqual(pattern.ai_worker, "worker_3")
        self.assertEqual(pattern.reuse_count, 0)
        self.assertIsInstance(pattern.timestamp, float)


class TestSpeedrunMetricsCollector(unittest.TestCase):
    """Test cases for SpeedrunMetricsCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = SpeedrunMetricsCollector(
            max_stored_experiments=50,
            max_stored_patterns=25
        )
    
    def test_initialization(self):
        """Test proper initialization of collector."""
        self.assertEqual(self.collector.max_stored_experiments, 50)
        self.assertEqual(self.collector.max_stored_patterns, 25)
        self.assertEqual(len(self.collector._experiments), 0)
        self.assertEqual(len(self.collector._pattern_discoveries), 0)
    
    def test_record_experiment_performance(self):
        """Test experiment recording meets performance requirements (<2ms)."""
        result = ExperimentResult(
            experiment_id="perf_test",
            status=ExperimentStatus.SUCCESS,
            duration_seconds=120.0,
            script_compilation_time_ms=75.5,
            checkpoint_loading_time_ms=425.2
        )
        
        start_time = time.time()
        self.collector.record_experiment(result)
        record_time = time.time() - start_time
        
        self.assertLess(record_time, 0.002, "Experiment recording should be <2ms")
    
    def test_record_pattern_discovery_performance(self):
        """Test pattern discovery recording meets performance requirements (<2ms)."""
        pattern = PatternDiscovery(
            pattern_id="perf_pattern",
            pattern_type=PatternType.BATTLE,
            quality_score=0.88,
            discovery_time_seconds=30.2,
            experiment_id="exp_001"
        )
        
        start_time = time.time()
        self.collector.record_pattern_discovery(pattern)
        record_time = time.time() - start_time
        
        self.assertLess(record_time, 0.002, "Pattern recording should be <2ms")
    
    def test_record_multiple_experiments(self):
        """Test recording multiple experiments with various outcomes."""
        experiments = [
            ExperimentResult("exp_001", ExperimentStatus.SUCCESS, 100.0, 
                           script_compilation_time_ms=95.0, checkpoint_loading_time_ms=450.0),
            ExperimentResult("exp_002", ExperimentStatus.SUCCESS, 110.0,
                           script_compilation_time_ms=88.5, checkpoint_loading_time_ms=380.2),
            ExperimentResult("exp_003", ExperimentStatus.FAILURE, 65.0,
                           script_compilation_time_ms=120.0, checkpoint_loading_time_ms=600.0),
            ExperimentResult("exp_004", ExperimentStatus.SUCCESS, 95.0,
                           script_compilation_time_ms=75.2, checkpoint_loading_time_ms=320.1),
        ]
        
        for exp in experiments:
            self.collector.record_experiment(exp)
        
        # Verify storage
        self.assertEqual(len(self.collector._experiments), 4)
        
        # Test metrics calculation
        snapshot = self.collector.get_metrics_snapshot()
        self.assertEqual(snapshot.total_experiments, 4)
        self.assertEqual(snapshot.successful_experiments, 3)
        self.assertEqual(snapshot.failed_experiments, 1)
        self.assertEqual(snapshot.experiment_success_rate, 75.0)
        
        # Test performance metrics
        expected_avg_compilation = (95.0 + 88.5 + 120.0 + 75.2) / 4
        expected_avg_checkpoint = (450.0 + 380.2 + 600.0 + 320.1) / 4
        
        self.assertAlmostEqual(snapshot.average_script_compilation_ms, expected_avg_compilation, places=1)
        self.assertAlmostEqual(snapshot.average_checkpoint_loading_ms, expected_avg_checkpoint, places=1)
    
    def test_target_compliance_calculation(self):
        """Test performance target compliance calculation."""
        # Create experiments with known performance characteristics
        experiments = [
            # Under target performance (good)
            ExperimentResult("exp_001", ExperimentStatus.SUCCESS, 100.0, 
                           script_compilation_time_ms=85.0, checkpoint_loading_time_ms=450.0),
            ExperimentResult("exp_002", ExperimentStatus.SUCCESS, 100.0,
                           script_compilation_time_ms=95.0, checkpoint_loading_time_ms=480.0),
            # Over target performance (bad)  
            ExperimentResult("exp_003", ExperimentStatus.SUCCESS, 100.0,
                           script_compilation_time_ms=120.0, checkpoint_loading_time_ms=600.0),
            ExperimentResult("exp_004", ExperimentStatus.SUCCESS, 100.0,
                           script_compilation_time_ms=150.0, checkpoint_loading_time_ms=750.0),
        ]
        
        for exp in experiments:
            self.collector.record_experiment(exp)
        
        snapshot = self.collector.get_metrics_snapshot()
        
        # 2 out of 4 experiments under 100ms compilation target = 50% compliance
        self.assertEqual(snapshot.compilation_target_compliance, 50.0)
        
        # 2 out of 4 experiments under 500ms checkpoint target = 50% compliance  
        self.assertEqual(snapshot.checkpoint_target_compliance, 50.0)
    
    def test_pattern_discovery_metrics(self):
        """Test pattern discovery metrics calculation."""
        base_time = time.time() - 3600  # 1 hour ago
        patterns = [
            PatternDiscovery("p1", PatternType.MOVEMENT, 0.85, 30.0, "exp_001"),
            PatternDiscovery("p2", PatternType.BATTLE, 0.92, 45.0, "exp_002"), 
            PatternDiscovery("p3", PatternType.MENU_NAVIGATION, 0.78, 20.0, "exp_003"),
            PatternDiscovery("p4", PatternType.OPTIMIZATION, 0.95, 60.0, "exp_004"),
        ]
        
        # Set timestamps to create realistic time progression
        for i, pattern in enumerate(patterns):
            pattern.timestamp = base_time + (i * 900)  # 15-minute intervals
            pattern.reuse_count = i  # Varying reuse counts
            self.collector.record_pattern_discovery(pattern)
        
        snapshot = self.collector.get_metrics_snapshot()
        
        self.assertEqual(snapshot.total_patterns_discovered, 4)
        self.assertGreater(snapshot.pattern_discovery_rate_per_hour, 0)
        
        # Average quality: (0.85 + 0.92 + 0.78 + 0.95) / 4 = 0.875
        self.assertAlmostEqual(snapshot.average_pattern_quality, 0.875, places=3)
        
        # Pattern reuse rate: (0 + 1 + 2 + 3) / 4 = 1.5
        self.assertEqual(snapshot.pattern_reuse_rate, 1.5)
    
    def test_record_pattern_reuse(self):
        """Test recording pattern reuse events."""
        pattern = PatternDiscovery("reuse_pattern", PatternType.MOVEMENT, 0.80, 25.0, "exp_001")
        self.collector.record_pattern_discovery(pattern)
        
        # Initial reuse count should be 0
        self.assertEqual(self.collector._pattern_discoveries[0].reuse_count, 0)
        
        # Record reuse
        self.collector.record_pattern_reuse("reuse_pattern")
        
        # Reuse count should be incremented
        self.assertEqual(self.collector._pattern_discoveries[0].reuse_count, 1)
        
        # Record multiple reuses
        self.collector.record_pattern_reuse("reuse_pattern")
        self.collector.record_pattern_reuse("reuse_pattern")
        
        self.assertEqual(self.collector._pattern_discoveries[0].reuse_count, 3)
    
    def test_ai_strategy_effectiveness(self):
        """Test AI strategy effectiveness calculation."""
        experiments = [
            ExperimentResult("exp_001", ExperimentStatus.SUCCESS, 100.0, ai_strategy="genetic_algorithm"),
            ExperimentResult("exp_002", ExperimentStatus.SUCCESS, 100.0, ai_strategy="genetic_algorithm"),
            ExperimentResult("exp_003", ExperimentStatus.FAILURE, 100.0, ai_strategy="genetic_algorithm"),
            ExperimentResult("exp_004", ExperimentStatus.SUCCESS, 100.0, ai_strategy="reinforcement_learning"),
            ExperimentResult("exp_005", ExperimentStatus.FAILURE, 100.0, ai_strategy="reinforcement_learning"),
            ExperimentResult("exp_006", ExperimentStatus.FAILURE, 100.0, ai_strategy="reinforcement_learning"),
        ]
        
        for exp in experiments:
            self.collector.record_experiment(exp)
        
        snapshot = self.collector.get_metrics_snapshot()
        
        # Genetic algorithm: 2 success, 1 failure = 66.67% effectiveness
        # Reinforcement learning: 1 success, 2 failures = 33.33% effectiveness
        
        effectiveness = snapshot.strategy_effectiveness
        self.assertAlmostEqual(effectiveness["genetic_algorithm"], 66.67, places=1)
        self.assertAlmostEqual(effectiveness["reinforcement_learning"], 33.33, places=1)
        self.assertEqual(snapshot.most_effective_strategy, "genetic_algorithm")
    
    def test_get_sla_compliance(self):
        """Test SLA compliance calculation."""
        # Create experiments that meet/don't meet SLA requirements
        experiments = [
            # Good performance, successful
            ExperimentResult("exp_001", ExperimentStatus.SUCCESS, 100.0,
                           script_compilation_time_ms=80.0, checkpoint_loading_time_ms=400.0),
            ExperimentResult("exp_002", ExperimentStatus.SUCCESS, 100.0,
                           script_compilation_time_ms=90.0, checkpoint_loading_time_ms=450.0),
        ]
        
        patterns = [
            PatternDiscovery("p1", PatternType.MOVEMENT, 0.85, 30.0, "exp_001"),
            PatternDiscovery("p2", PatternType.BATTLE, 0.90, 45.0, "exp_002"),
        ]
        
        for exp in experiments:
            self.collector.record_experiment(exp)
        for pattern in patterns:
            self.collector.record_pattern_discovery(pattern)
        
        sla_compliance = self.collector.get_sla_compliance()
        
        self.assertIn("overall_sla_compliant", sla_compliance)
        self.assertIn("individual_compliance", sla_compliance)
        
        # With 100% success rate, good performance, and good pattern quality,
        # should meet SLA requirements
        individual = sla_compliance["individual_compliance"]
        self.assertTrue(individual["experiment_success_sla"])  # 100% >= 95%
        self.assertTrue(individual["compilation_performance_sla"])  # 100% >= 95%
        self.assertTrue(individual["checkpoint_performance_sla"])  # 100% >= 95%
        self.assertTrue(individual["pattern_quality_sla"])  # 0.875 >= 0.7
    
    def test_storage_limits(self):
        """Test that storage respects configured limits."""
        # Generate more experiments than the limit
        for i in range(75):  # More than max_stored_experiments=50
            exp = ExperimentResult(f"exp_{i}", ExperimentStatus.SUCCESS, 100.0)
            self.collector.record_experiment(exp)
        
        # Should not exceed the limit
        self.assertEqual(len(self.collector._experiments), 50)
        
        # Should contain the most recent experiments
        last_exp = self.collector._experiments[-1]
        self.assertEqual(last_exp.experiment_id, "exp_74")
        
        # Test patterns limit
        for i in range(35):  # More than max_stored_patterns=25
            pattern = PatternDiscovery(f"pattern_{i}", PatternType.MOVEMENT, 0.8, 30.0, "exp_001")
            self.collector.record_pattern_discovery(pattern)
        
        # Should not exceed the limit
        self.assertEqual(len(self.collector._pattern_discoveries), 25)
        
        # Should contain the most recent patterns
        last_pattern = self.collector._pattern_discoveries[-1]
        self.assertEqual(last_pattern.pattern_id, "pattern_34")
    
    def test_metrics_caching(self):
        """Test metrics snapshot caching for performance."""
        # Record some data
        exp = ExperimentResult("cache_test", ExperimentStatus.SUCCESS, 100.0)
        self.collector.record_experiment(exp)
        
        # First call should generate snapshot
        start_time = time.time()
        snapshot1 = self.collector.get_metrics_snapshot()
        first_time = time.time() - start_time
        
        # Second call should use cache
        start_time = time.time() 
        snapshot2 = self.collector.get_metrics_snapshot()
        cached_time = time.time() - start_time
        
        # Cached call should be much faster
        self.assertLess(cached_time, first_time / 2, "Cached call should be significantly faster")
        self.assertEqual(snapshot1.timestamp, snapshot2.timestamp, "Should return cached data")
    
    def test_get_performance_metrics(self):
        """Test getting collector performance metrics."""
        # Record some data to generate timing measurements
        for i in range(5):
            exp = ExperimentResult(f"perf_{i}", ExperimentStatus.SUCCESS, 100.0)
            self.collector.record_experiment(exp)
        
        perf_metrics = self.collector.get_performance_metrics()
        
        self.assertIn("measurement_performance", perf_metrics)
        self.assertIn("storage_efficiency", perf_metrics)
        self.assertIn("cache_performance", perf_metrics)
        
        measurement = perf_metrics["measurement_performance"]
        self.assertIn("average_ms", measurement)
        self.assertIn("target_ms", measurement)
        self.assertEqual(measurement["target_ms"], 2.0)
        
        storage = perf_metrics["storage_efficiency"]
        self.assertEqual(storage["experiments_stored"], 5)
        self.assertEqual(storage["max_experiments"], 50)
    
    def test_reset_metrics(self):
        """Test resetting all collected metrics."""
        # Generate some data
        exp = ExperimentResult("reset_test", ExperimentStatus.SUCCESS, 100.0)
        pattern = PatternDiscovery("reset_pattern", PatternType.MOVEMENT, 0.8, 30.0, "exp_001")
        
        self.collector.record_experiment(exp)
        self.collector.record_pattern_discovery(pattern)
        
        # Verify data exists
        self.assertEqual(len(self.collector._experiments), 1)
        self.assertEqual(len(self.collector._pattern_discoveries), 1)
        
        # Reset
        self.collector.reset_metrics()
        
        # Verify data is cleared
        self.assertEqual(len(self.collector._experiments), 0)
        self.assertEqual(len(self.collector._pattern_discoveries), 0)
        
        snapshot = self.collector.get_metrics_snapshot()
        self.assertEqual(snapshot.total_experiments, 0)
        self.assertEqual(snapshot.total_patterns_discovered, 0)
    
    def test_thread_safety(self):
        """Test thread-safe operations during concurrent recording."""
        def record_experiments():
            for i in range(10):
                exp = ExperimentResult(f"thread_exp_{i}", ExperimentStatus.SUCCESS, 100.0)
                self.collector.record_experiment(exp)
                time.sleep(0.001)
        
        def record_patterns():
            for i in range(10):
                pattern = PatternDiscovery(f"thread_pattern_{i}", PatternType.MOVEMENT, 0.8, 30.0, "exp")
                self.collector.record_pattern_discovery(pattern)
                time.sleep(0.001)
        
        # Run concurrent recordings
        threads = [
            threading.Thread(target=record_experiments),
            threading.Thread(target=record_patterns),
            threading.Thread(target=record_experiments),
        ]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should have recorded all data correctly
        self.assertEqual(len(self.collector._experiments), 20)  # 2 threads * 10 each
        self.assertEqual(len(self.collector._pattern_discoveries), 10)


@pytest.mark.integration
class TestSpeedrunMetricsIntegration:
    """Integration tests for speedrun metrics system."""
    
    def test_realistic_experiment_workflow(self):
        """Test realistic experiment tracking workflow."""
        collector = SpeedrunMetricsCollector()
        
        # Simulate a series of experiments with realistic data
        experiment_scenarios = [
            ("genetic_algo_run_1", ExperimentStatus.SUCCESS, 142.5, 87.2, 445.8, 0.87),
            ("genetic_algo_run_2", ExperimentStatus.SUCCESS, 138.9, 92.1, 412.3, 0.82),
            ("genetic_algo_run_3", ExperimentStatus.FAILURE, 95.2, 125.4, 580.1, 0.45),
            ("reinforcement_run_1", ExperimentStatus.SUCCESS, 156.3, 78.9, 389.4, 0.91),
            ("reinforcement_run_2", ExperimentStatus.SUCCESS, 161.8, 83.2, 425.7, 0.88),
        ]
        
        for exp_id, status, duration, compile_ms, checkpoint_ms, score in experiment_scenarios:
            result = ExperimentResult(
                experiment_id=exp_id,
                status=status,
                duration_seconds=duration,
                script_compilation_time_ms=compile_ms,
                checkpoint_loading_time_ms=checkpoint_ms,
                ai_strategy=exp_id.split('_')[0] + "_algorithm",
                performance_score=score
            )
            collector.record_experiment(result)
        
        # Add some pattern discoveries
        patterns = [
            PatternDiscovery("movement_opt_1", PatternType.MOVEMENT, 0.89, 35.2, "genetic_algo_run_1"),
            PatternDiscovery("battle_strat_1", PatternType.BATTLE, 0.94, 48.7, "reinforcement_run_1"),
            PatternDiscovery("menu_nav_1", PatternType.MENU_NAVIGATION, 0.76, 22.1, "genetic_algo_run_2"),
        ]
        
        for pattern in patterns:
            collector.record_pattern_discovery(pattern)
        
        # Get comprehensive snapshot
        snapshot = collector.get_metrics_snapshot()
        sla_compliance = collector.get_sla_compliance()
        
        # Verify realistic metrics
        assert snapshot.total_experiments == 5
        assert snapshot.successful_experiments == 4
        assert snapshot.experiment_success_rate == 80.0
        
        # Performance metrics should be reasonable
        assert 70 < snapshot.average_script_compilation_ms < 110
        assert 400 < snapshot.average_checkpoint_loading_ms < 500
        
        # Pattern metrics
        assert snapshot.total_patterns_discovered == 3
        assert 0.75 < snapshot.average_pattern_quality < 0.95
        
        # SLA compliance should be calculated correctly
        assert "overall_sla_compliant" in sla_compliance
        assert "individual_compliance" in sla_compliance


if __name__ == "__main__":
    unittest.main()