"""
Test suite for predictive planning functionality.

Performance-focused tests following John Botmack principles:
- <100ms performance target validation for think_ahead method
- <10ms cache retrieval target validation
- Mathematical precision verification for Bayesian algorithms
- Edge case handling and graceful degradation testing
- Real-world scenario simulation for Pokemon speedrun contexts

Test Categories:
- TestPredictivePlanningDataStructures: Core data structure validation
- TestExecutionPatternAnalyzer: Pattern analysis algorithm testing
- TestBayesianPredictor: Mathematical precision and accuracy testing
- TestPredictionCache: Cache performance and reliability testing
- TestContingencyGenerator: Strategy generation and scenario coverage
- TestOpusStrategistPredictivePlanning: Integration and performance testing
"""

import time
import unittest
from unittest.mock import Mock

import pytest

from claudelearnspokemon.predictive_planning import (
    BayesianPredictor,
    ContingencyGenerator,
    ContingencyStrategy,
    ExecutionPattern,
    ExecutionPatternAnalyzer,
    OutcomePrediction,
    PredictionCache,
    PredictionConfidence,
    PredictivePlanningResult,
    ScenarioTrigger,
)
from claudelearnspokemon.strategy_response import ExperimentSpec, StrategyResponse


@pytest.mark.fast
@pytest.mark.medium
class TestPredictivePlanningDataStructures(unittest.TestCase):
    """Test core data structures for predictive planning."""

    def test_outcome_prediction_validation(self):
        """Test OutcomePrediction validates parameters correctly."""
        # Valid prediction
        prediction = OutcomePrediction(
            experiment_id="test_exp",
            success_probability=0.75,
            estimated_execution_time_ms=2500.0,
            expected_performance_score=0.85,
            confidence=PredictionConfidence.HIGH,
            contributing_factors=["high_priority", "optimal_complexity"],
            risk_factors=["time_constraint"],
            historical_accuracy=0.82,
        )

        self.assertEqual(prediction.experiment_id, "test_exp")
        self.assertEqual(prediction.success_probability, 0.75)
        self.assertEqual(prediction.confidence, PredictionConfidence.HIGH)

        # Test risk score calculation
        risk_score = prediction.get_risk_score()
        self.assertGreater(risk_score, 0.0)
        self.assertLess(risk_score, 1.0)

        # Test serialization
        prediction_dict = prediction.to_dict()
        self.assertIn("experiment_id", prediction_dict)
        self.assertIn("risk_score", prediction_dict)

        # Test invalid success probability
        with self.assertRaises(ValueError):
            OutcomePrediction(
                experiment_id="test",
                success_probability=1.5,  # Invalid: > 1.0
                estimated_execution_time_ms=1000.0,
                expected_performance_score=0.5,
                confidence=PredictionConfidence.MEDIUM,
            )

    def test_contingency_strategy_validation(self):
        """Test ContingencyStrategy validates parameters and trigger conditions."""
        # Create mock strategy
        mock_strategy = Mock(spec=StrategyResponse)
        mock_strategy.to_dict.return_value = {"strategy_id": "test_strategy"}

        contingency = ContingencyStrategy(
            scenario_id="test_scenario",
            trigger_type=ScenarioTrigger.EXECUTION_FAILURE,
            trigger_conditions=["failure_rate > 0.3", "success_probability < 0.5"],
            strategy=mock_strategy,
            activation_probability=0.7,
            confidence=PredictionConfidence.HIGH,
            priority=3,
        )

        self.assertEqual(contingency.scenario_id, "test_scenario")
        self.assertEqual(contingency.trigger_type, ScenarioTrigger.EXECUTION_FAILURE)
        self.assertEqual(contingency.priority, 3)

        # Test activation logic
        test_conditions = {
            "failure_rate": 0.4,
            "success_probability": 0.3,
            "performance_score": 0.6,  # Add this for condition matching
        }

        should_activate = contingency.should_activate(test_conditions)
        self.assertTrue(should_activate)

        # Test serialization
        contingency_dict = contingency.to_dict()
        self.assertIn("scenario_id", contingency_dict)
        self.assertIn("strategy", contingency_dict)

    def test_predictive_planning_result_creation(self):
        """Test PredictivePlanningResult creation and validation."""
        # Create mock components
        mock_strategy = Mock(spec=StrategyResponse)
        mock_strategy.to_dict.return_value = {"strategy_id": "primary"}

        mock_contingency = Mock(spec=ContingencyStrategy)
        mock_contingency.to_dict.return_value = {"scenario_id": "contingency1"}

        mock_prediction = Mock(spec=OutcomePrediction)
        mock_prediction.to_dict.return_value = {"experiment_id": "exp1"}
        mock_prediction.success_probability = 0.8

        result = PredictivePlanningResult(
            planning_id="test_planning",
            primary_strategy=mock_strategy,
            contingencies=[mock_contingency],
            outcome_predictions={"exp1": mock_prediction},
            confidence_scores={"overall": 0.8, "primary_strategy": 0.75},
            execution_time_ms=85.5,
            cache_metadata={"cache_key": "test_key", "hit": False},
        )

        self.assertEqual(result.planning_id, "test_planning")
        self.assertEqual(result.execution_time_ms, 85.5)
        self.assertIsNotNone(result.content_hash)

        # Test overall confidence calculation
        overall_confidence = result.get_overall_confidence()
        self.assertGreater(overall_confidence, 0.0)
        self.assertLessEqual(overall_confidence, 1.0)

        # Test serialization
        result_dict = result.to_dict()
        self.assertIn("planning_id", result_dict)
        self.assertIn("content_hash", result_dict)
        self.assertIn("overall_confidence", result_dict)


@pytest.mark.fast
@pytest.mark.medium
class TestExecutionPatternAnalyzer(unittest.TestCase):
    """Test execution pattern analysis algorithms."""

    def setUp(self):
        """Set up pattern analyzer for tests."""
        self.analyzer = ExecutionPatternAnalyzer(
            max_patterns=50,
            similarity_threshold=0.7,
            min_frequency_threshold=2,
        )

    def test_pattern_feature_extraction(self):
        """Test pattern feature extraction with O(n) complexity."""
        test_experiments = [
            {
                "id": "exp1",
                "priority": 3,
                "script_dsl": "MOVE_TO_LOCATION; BATTLE_TRAINER; COLLECT_ITEM",
                "checkpoint": "gym_entrance",
            },
            {
                "id": "exp2",
                "priority": 2,
                "script_dsl": "EXPLORE_AREA; SAVE_PROGRESS",
                "checkpoint": "route_12",
            },
        ]

        start_time = time.perf_counter()
        features = self.analyzer._extract_pattern_features(test_experiments)
        extraction_time = (time.perf_counter() - start_time) * 1000

        # Performance validation: <50ms for feature extraction
        self.assertLess(
            extraction_time,
            50.0,
            f"Feature extraction took {extraction_time:.2f}ms, exceeding 50ms target",
        )

        # Feature validation
        self.assertIn("experiment_count", features)
        self.assertIn("avg_priority", features)
        self.assertIn("checkpoint_diversity", features)
        self.assertIn("cmd_move_freq", features)
        self.assertIn("cmd_battle_freq", features)

        # Feature normalization validation (all values 0-1)
        for feature, value in features.items():
            self.assertGreaterEqual(value, 0.0, f"Feature {feature} has negative value: {value}")
            self.assertLessEqual(value, 1.0, f"Feature {feature} exceeds 1.0: {value}")

    def test_pattern_similarity_computation(self):
        """Test cosine similarity computation for execution patterns."""
        pattern1 = ExecutionPattern(
            pattern_id="pattern1",
            features={"feature_a": 0.8, "feature_b": 0.6, "feature_c": 0.3},
            success_rate=0.75,
            avg_execution_time=2000.0,
        )

        pattern2 = ExecutionPattern(
            pattern_id="pattern2",
            features={"feature_a": 0.7, "feature_b": 0.5, "feature_c": 0.4},
            success_rate=0.65,
            avg_execution_time=2500.0,
        )

        similarity = pattern1.compute_similarity(pattern2)

        # Similarity should be between 0 and 1
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

        # Test identical patterns
        identical_similarity = pattern1.compute_similarity(pattern1)
        self.assertAlmostEqual(identical_similarity, 1.0, places=5)

    def test_pattern_analysis_performance(self):
        """Test pattern analysis meets <50ms performance target."""
        # Create test experiments
        test_experiments = [
            {
                "id": f"exp_{i}",
                "priority": (i % 5) + 1,
                "script_dsl": f"ACTION_{i}; SAVE_PROGRESS",
                "checkpoint": f"checkpoint_{i % 3}",
            }
            for i in range(10)
        ]

        start_time = time.perf_counter()
        analysis_result = self.analyzer.analyze_execution_patterns(test_experiments)
        analysis_time = (time.perf_counter() - start_time) * 1000

        # Performance validation: <50ms target
        self.assertLess(
            analysis_time,
            50.0,
            f"Pattern analysis took {analysis_time:.2f}ms, exceeding 50ms target",
        )

        # Result validation
        self.assertIn("current_pattern", analysis_result)
        self.assertIn("similar_patterns", analysis_result)
        self.assertIn("confidence_metrics", analysis_result)
        self.assertIn("analysis_time_ms", analysis_result)

        # Performance metrics validation
        performance_metrics = self.analyzer.get_performance_metrics()
        self.assertIn("avg_analysis_time_ms", performance_metrics)

    def test_trend_analysis_accuracy(self):
        """Test trend analysis with statistical validation."""
        # Create patterns with known trend
        increasing_patterns = [
            ExecutionPattern(
                pattern_id=f"trend_{i}",
                features={"base_feature": 0.5},
                success_rate=0.5 + (i * 0.1),  # Increasing trend
                avg_execution_time=1000.0 + (i * 100),  # Increasing time
                frequency=5,
            )
            for i in range(5)
        ]

        trend_analysis = self.analyzer._compute_trend_analysis(increasing_patterns)

        # Validate trend detection
        self.assertIn("success_rate_trend", trend_analysis)
        self.assertIn("execution_time_trend", trend_analysis)
        self.assertIn("confidence", trend_analysis)

        # Success rate should show positive trend
        self.assertGreater(trend_analysis["success_rate_trend"], 0.0)

        # Execution time should show positive trend (increasing)
        self.assertGreater(trend_analysis["execution_time_trend"], 0.0)


@pytest.mark.fast
@pytest.mark.medium
class TestBayesianPredictor(unittest.TestCase):
    """Test Bayesian prediction algorithms with mathematical precision."""

    def setUp(self):
        """Set up Bayesian predictor for tests."""
        self.predictor = BayesianPredictor(
            alpha_prior=1.0,
            beta_prior=1.0,
            forgetting_factor=0.95,
            min_samples=3,
        )

    def test_bayesian_inference_accuracy(self):
        """Test Bayesian inference mathematical accuracy."""
        # Test with known parameters
        experiment_id = "test_exp"
        experiment_features = {
            "priority": 3,
            "script_dsl": "TEST_SCRIPT",
            "complexity": 0.5,
        }

        # Mock similar patterns with known success rates
        from datetime import datetime, timedelta

        now = datetime.utcnow()
        mock_patterns = [
            Mock(
                success_rate=0.8,
                frequency=10,
                last_seen=now,
                avg_execution_time=2000.0,
                features={"similarity": 0.9},
            ),
            Mock(
                success_rate=0.7,
                frequency=5,
                last_seen=now - timedelta(days=1),  # 1 day ago
                avg_execution_time=2500.0,
                features={"similarity": 0.8},
            ),
        ]

        prediction = self.predictor.predict_outcome(
            experiment_id=experiment_id,
            experiment_features=experiment_features,
            similar_patterns=mock_patterns,
        )

        # Validate prediction structure
        self.assertIsInstance(prediction, OutcomePrediction)
        self.assertEqual(prediction.experiment_id, experiment_id)

        # Validate mathematical bounds
        self.assertGreaterEqual(prediction.success_probability, 0.0)
        self.assertLessEqual(prediction.success_probability, 1.0)

        # Validate confidence assessment
        self.assertIsInstance(prediction.confidence, PredictionConfidence)

        # Test prediction updates
        self.predictor.update_with_result(
            experiment_id=experiment_id,
            actual_success=True,
            actual_execution_time=2200.0,
            actual_performance_score=0.85,
        )

        # Verify parameters were updated
        self.assertIn(experiment_id, self.predictor.experiment_priors)
        updated_params = self.predictor.experiment_priors[experiment_id]
        self.assertGreater(updated_params["alpha"], 1.0)  # Should increase with success

    def test_evidence_parameter_extraction(self):
        """Test evidence parameter extraction from similar patterns."""
        from datetime import datetime, timedelta

        now = datetime.utcnow()
        patterns = [
            Mock(success_rate=0.8, frequency=5, last_seen=now),
            Mock(success_rate=0.6, frequency=3, last_seen=now - timedelta(hours=1)),
            Mock(success_rate=0.9, frequency=8, last_seen=now - timedelta(hours=2)),
        ]

        alpha_evidence, beta_evidence = self.predictor._extract_evidence_parameters(patterns)

        # Evidence parameters should be non-negative
        self.assertGreaterEqual(alpha_evidence, 0.0)
        self.assertGreaterEqual(beta_evidence, 0.0)

        # Total evidence should reflect input patterns
        total_evidence = alpha_evidence + beta_evidence
        self.assertGreater(total_evidence, 0.0)

    def test_prediction_confidence_levels(self):
        """Test prediction confidence level mapping."""
        # Test with varying amounts of evidence
        confidence_high = self.predictor._compute_prediction_confidence(20.0, 5.0, 10)
        confidence_low = self.predictor._compute_prediction_confidence(2.0, 2.0, 1)

        # High evidence should give higher confidence
        self.assertGreaterEqual(confidence_high.value, confidence_low.value)

    def test_performance_metrics_tracking(self):
        """Test prediction performance metrics collection."""
        # Make several predictions to generate metrics
        for i in range(5):
            self.predictor.predict_outcome(
                experiment_id=f"test_{i}",
                experiment_features={"test": True},
                similar_patterns=[],
            )

        metrics = self.predictor.get_performance_metrics()

        # Validate metrics structure
        self.assertIn("avg_prediction_time_ms", metrics)
        self.assertIn("total_experiments", metrics)
        self.assertIn("total_predictions", metrics)


@pytest.mark.fast
@pytest.mark.medium
class TestPredictionCache(unittest.TestCase):
    """Test prediction cache performance and reliability."""

    def setUp(self):
        """Set up prediction cache for tests."""
        self.cache = PredictionCache(
            max_entries=10,
            default_ttl=300.0,
            cleanup_interval=5,
        )

    def test_cache_retrieval_performance(self):
        """Test cache retrieval meets <10ms target."""
        # Create test result
        mock_result = self._create_mock_planning_result()

        # Store in cache
        cache_key = "test_key"
        self.cache.put(cache_key, mock_result)

        # Test retrieval performance
        retrieval_times = []
        for _ in range(100):  # Multiple retrievals for average
            start_time = time.perf_counter()
            cached_result = self.cache.get(cache_key)
            retrieval_time = (time.perf_counter() - start_time) * 1000
            retrieval_times.append(retrieval_time)
            self.assertIsNotNone(cached_result)

        avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
        max_retrieval_time = max(retrieval_times)

        # Performance validation: <10ms average, <20ms max
        self.assertLess(
            avg_retrieval_time,
            10.0,
            f"Average retrieval time {avg_retrieval_time:.2f}ms exceeds 10ms target",
        )
        self.assertLess(
            max_retrieval_time,
            20.0,
            f"Max retrieval time {max_retrieval_time:.2f}ms exceeds 20ms acceptable limit",
        )

    def test_lru_eviction_policy(self):
        """Test LRU eviction maintains cache bounds."""
        # Fill cache to capacity
        for i in range(15):  # More than max_entries (10)
            mock_result = self._create_mock_planning_result(planning_id=f"result_{i}")
            self.cache.put(f"key_{i}", mock_result)

        # Verify cache size is bounded
        stats = self.cache.get_statistics()
        self.assertLessEqual(stats["current_entries"], self.cache.max_entries)

        # Verify LRU eviction (early entries should be evicted)
        self.assertIsNone(self.cache.get("key_0"))  # Should be evicted
        self.assertIsNone(self.cache.get("key_1"))  # Should be evicted
        self.assertIsNotNone(self.cache.get("key_14"))  # Should be present

    def test_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        mock_result = self._create_mock_planning_result()

        # Store with short TTL
        self.cache.put("test_key", mock_result, ttl=0.1)  # 100ms TTL

        # Should be available immediately
        cached_result = self.cache.get("test_key")
        self.assertIsNotNone(cached_result)

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired
        expired_result = self.cache.get("test_key")
        self.assertIsNone(expired_result)

        # Verify expiration metrics
        stats = self.cache.get_statistics()
        self.assertGreater(stats["expired_count"], 0)

    def test_cache_key_generation(self):
        """Test deterministic cache key generation."""
        components1 = ({"a": 1, "b": 2}, [3, 4, 5], "test")
        components2 = ({"b": 2, "a": 1}, [5, 4, 3], "test")  # Same data, different order

        key1 = self.cache.generate_cache_key(*components1)
        key2 = self.cache.generate_cache_key(*components2)

        # Keys should be identical for equivalent data
        self.assertEqual(key1, key2)

        # Keys should be deterministic
        key3 = self.cache.generate_cache_key(*components1)
        self.assertEqual(key1, key3)

    def _create_mock_planning_result(self, planning_id="test_planning"):
        """Create mock PredictivePlanningResult for testing."""
        mock_strategy = Mock(spec=StrategyResponse)
        mock_strategy.to_dict.return_value = {"strategy_id": "test"}

        return PredictivePlanningResult(
            planning_id=planning_id,
            primary_strategy=mock_strategy,
            contingencies=[],
            outcome_predictions={},
            confidence_scores={"overall": 0.8},
            execution_time_ms=50.0,
            cache_metadata={"test": True},
        )


@pytest.mark.fast
@pytest.mark.medium
class TestContingencyGenerator(unittest.TestCase):
    """Test contingency strategy generation algorithms."""

    def setUp(self):
        """Set up contingency generator for tests."""
        self.generator = ContingencyGenerator(
            fallback_strategy_pool_size=10,
            scenario_coverage_threshold=0.8,
            strategy_template_cache_size=20,
        )

    def test_contingency_generation_performance(self):
        """Test contingency generation meets <50ms target."""
        # Create test data
        primary_strategy = self._create_test_strategy()
        execution_patterns = {
            "similar_patterns": [],
            "confidence_metrics": {"overall_confidence": 0.6},
            "pattern_statistics": {"cache_hit_rate": 0.5},
        }
        outcome_predictions = {
            "exp1": Mock(
                success_probability=0.3,
                get_risk_score=Mock(return_value=0.7),
                estimated_execution_time_ms=8000.0,
            ),
            "exp2": Mock(
                success_probability=0.8,
                get_risk_score=Mock(return_value=0.2),
                estimated_execution_time_ms=2000.0,
            ),
        }

        start_time = time.perf_counter()
        contingencies = self.generator.generate_contingencies(
            primary_strategy=primary_strategy,
            execution_patterns=execution_patterns,
            outcome_predictions=outcome_predictions,
            horizon=3,
        )
        generation_time = (time.perf_counter() - start_time) * 1000

        # Performance validation: <50ms target
        self.assertLess(
            generation_time,
            50.0,
            f"Contingency generation took {generation_time:.2f}ms, exceeding 50ms target",
        )

        # Result validation
        self.assertIsInstance(contingencies, list)
        self.assertGreater(len(contingencies), 0)

        # Validate contingency structure
        for contingency in contingencies:
            self.assertIsInstance(contingency, ContingencyStrategy)
            self.assertIsNotNone(contingency.strategy)
            self.assertGreater(len(contingency.trigger_conditions), 0)

    def test_scenario_identification(self):
        """Test failure scenario identification logic."""
        primary_strategy = self._create_test_strategy()
        execution_patterns = {
            "pattern_statistics": {"cache_hit_rate": 0.4},  # Below 0.6 threshold
            "similar_patterns": [Mock()],  # Only 1 pattern (below 2 threshold)
            "trend_analysis": {"confidence": 0.3},  # Below 0.4 threshold
        }
        outcome_predictions = {
            "high_risk_exp": Mock(
                success_probability=0.3,  # Below 0.5 threshold
                get_risk_score=Mock(return_value=0.7),  # Above 0.6 threshold
                estimated_execution_time_ms=12000.0,  # Above 10s threshold
            )
        }

        scenarios = self.generator._identify_failure_scenarios(
            primary_strategy, execution_patterns, outcome_predictions
        )

        # Should identify multiple scenarios
        self.assertGreater(len(scenarios), 0)

        # Verify scenario types
        scenario_types = {s["type"] for s in scenarios}

        # Should identify multiple scenario types
        self.assertGreaterEqual(len(scenario_types), 2)

    def test_template_based_strategy_generation(self):
        """Test template-based contingency strategy generation."""
        scenario = {
            "type": ScenarioTrigger.EXECUTION_FAILURE,
            "probability": 0.8,
            "trigger_conditions": ["failure_rate > 0.3"],
            "priority": 4,
        }

        primary_strategy = self._create_test_strategy()
        execution_patterns = {"current_pattern": {"test": 0.5}}

        contingency = self.generator._generate_scenario_contingency(
            scenario, primary_strategy, execution_patterns, horizon=3
        )

        # Validate contingency creation
        self.assertIsInstance(contingency, ContingencyStrategy)
        self.assertEqual(contingency.trigger_type, ScenarioTrigger.EXECUTION_FAILURE)
        self.assertIsNotNone(contingency.strategy)

        # Validate strategy experiments
        strategy_experiments = contingency.strategy.experiments
        self.assertGreater(len(strategy_experiments), 0)

    def test_graceful_degradation(self):
        """Test graceful degradation with minimal contingencies."""
        primary_strategy = self._create_test_strategy()

        # Test minimal contingency creation
        minimal_contingencies = self.generator._create_minimal_contingencies(primary_strategy)

        self.assertGreater(len(minimal_contingencies), 0)

        for contingency in minimal_contingencies:
            self.assertIsInstance(contingency, ContingencyStrategy)
            self.assertIsNotNone(contingency.strategy)

    def _create_test_strategy(self):
        """Create test strategy for contingency generation."""
        mock_experiment = ExperimentSpec(
            id="test_exp",
            name="Test Experiment",
            checkpoint="test_checkpoint",
            script_dsl="TEST_ACTION; SAVE",
            expected_outcome="test_outcome",
            priority=2,
        )

        return StrategyResponse(
            strategy_id="test_strategy",
            experiments=[mock_experiment],
            strategic_insights=["Test insight"],
            next_checkpoints=["test_checkpoint"],
        )


@pytest.mark.integration
@pytest.mark.medium
class TestOpusStrategistPredictivePlanning(unittest.TestCase):
    """Integration tests for OpusStrategist predictive planning."""

    def setUp(self):
        """Set up OpusStrategist with predictive planning enabled."""
        self.mock_claude_manager = Mock()

        # Import here to avoid circular imports during module load
        from claudelearnspokemon.opus_strategist import OpusStrategist

        self.strategist = OpusStrategist(
            claude_manager=self.mock_claude_manager,
            enable_predictive_planning=True,
            prediction_cache_size=50,
            pattern_analyzer_max_patterns=100,
        )

    def test_think_ahead_performance_target(self):
        """Test think_ahead method meets <100ms performance target."""
        current_experiments = [
            {
                "id": "exp1",
                "name": "Test Experiment 1",
                "priority": 3,
                "script_dsl": "MOVE_TO_LOCATION; BATTLE_TRAINER; SAVE_PROGRESS",
                "checkpoint": "gym_entrance",
                "expected_outcome": "gym_progress",
            },
            {
                "id": "exp2",
                "name": "Test Experiment 2",
                "priority": 2,
                "script_dsl": "EXPLORE_AREA; COLLECT_ITEMS",
                "checkpoint": "route_12",
                "expected_outcome": "item_collection",
            },
        ]

        execution_patterns = {
            "current_pattern": {
                "experiment_count": 0.2,
                "avg_priority": 0.5,
                "checkpoint_diversity": 1.0,
            },
            "similar_patterns": [],
            "confidence_metrics": {"overall_confidence": 0.6},
            "pattern_statistics": {"cache_hit_rate": 0.8},
            "historical_results": [],
        }

        # Performance test
        start_time = time.perf_counter()
        result = self.strategist.think_ahead(
            current_experiments=current_experiments,
            execution_patterns=execution_patterns,
            horizon=3,
        )
        execution_time = (time.perf_counter() - start_time) * 1000

        # Performance validation: <100ms target
        self.assertLess(
            execution_time,
            100.0,
            f"think_ahead took {execution_time:.2f}ms, exceeding 100ms target",
        )

        # Result validation
        self.assertIsInstance(result, PredictivePlanningResult)
        self.assertIsNotNone(result.primary_strategy)
        self.assertIsInstance(result.contingencies, list)
        self.assertIsInstance(result.outcome_predictions, dict)
        self.assertIsInstance(result.confidence_scores, dict)

        # Performance metadata validation
        self.assertGreater(result.execution_time_ms, 0)
        self.assertIn("cache_key", result.cache_metadata)

    def test_prediction_caching(self):
        """Test prediction result caching functionality."""
        test_experiments = [{"id": "cache_test", "priority": 2, "script_dsl": "TEST"}]
        test_patterns = {"current_pattern": {}, "similar_patterns": []}

        # First call - cache miss
        result1 = self.strategist.think_ahead(test_experiments, test_patterns)

        # Second call - should be cache hit
        start_time = time.perf_counter()
        result2 = self.strategist.think_ahead(test_experiments, test_patterns)
        cached_retrieval_time = (time.perf_counter() - start_time) * 1000

        # Cache hit should be much faster
        self.assertLess(
            cached_retrieval_time,
            10.0,
            f"Cached retrieval took {cached_retrieval_time:.2f}ms, exceeding 10ms target",
        )

        # Results should be equivalent (same content hash)
        self.assertEqual(result1.content_hash, result2.content_hash)

        # Verify cache metrics
        metrics = self.strategist.get_metrics()
        self.assertGreater(metrics["predictive_planning_cache_hits"], 0)

    def test_prediction_result_updates(self):
        """Test prediction result updates for learning."""
        # Make a prediction first
        test_experiments = [{"id": "learning_test", "priority": 3}]
        test_patterns = {"current_pattern": {}, "similar_patterns": []}

        self.strategist.think_ahead(test_experiments, test_patterns)

        # Update with actual results
        self.strategist.update_prediction_results(
            experiment_id="learning_test",
            actual_success=True,
            actual_execution_time=2500.0,
            actual_performance_score=0.85,
        )

        # Verify no exceptions and metrics updated
        predictor_metrics = self.strategist.bayesian_predictor.get_performance_metrics()
        self.assertGreater(predictor_metrics.get("total_experiments", 0), 0)

    def test_comprehensive_metrics_collection(self):
        """Test comprehensive metrics collection from all components."""
        # Generate some activity
        test_experiments = [{"id": "metrics_test", "priority": 2}]
        test_patterns = {"current_pattern": {}, "similar_patterns": []}

        self.strategist.think_ahead(test_experiments, test_patterns)

        # Get comprehensive metrics
        metrics = self.strategist.get_metrics()

        # Validate predictive planning metrics
        self.assertTrue(metrics["predictive_planning_enabled"])
        self.assertIn("pattern_analyzer_metrics", metrics)
        self.assertIn("bayesian_predictor_metrics", metrics)
        self.assertIn("contingency_generator_metrics", metrics)
        self.assertIn("prediction_cache_metrics", metrics)

        # Validate performance metrics
        self.assertGreater(metrics["predictive_planning_requests"], 0)

    def test_disabled_predictive_planning(self):
        """Test behavior when predictive planning is disabled."""
        disabled_strategist = Mock()
        disabled_strategist.enable_predictive_planning = False

        # Import and patch
        from claudelearnspokemon.opus_strategist import OpusStrategist
        from claudelearnspokemon.opus_strategist_exceptions import OpusStrategistError

        # Create disabled strategist
        disabled_strategist = OpusStrategist(
            claude_manager=self.mock_claude_manager,
            enable_predictive_planning=False,
        )

        # Should raise error when think_ahead is called
        with self.assertRaises(OpusStrategistError):
            disabled_strategist.think_ahead([], {})

        # Metrics should reflect disabled state
        metrics = disabled_strategist.get_metrics()
        self.assertFalse(metrics["predictive_planning_enabled"])


if __name__ == "__main__":
    # Run performance-critical tests first
    unittest.main(verbosity=2, failfast=True)
