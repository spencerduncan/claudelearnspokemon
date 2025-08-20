"""Unit tests for OpusStrategist experiment generation functionality."""

import json
import time
from unittest.mock import Mock

import pytest

from src.claudelearnspokemon.opus_strategist import (
    ExperimentSpec,
    OpusStrategist,
)


@pytest.mark.fast
class TestOpusStrategistExperimentGeneration:
    """Test experiment generation functionality in OpusStrategist."""

    @pytest.fixture
    def mock_claude_manager(self):
        """Mock ClaudeCodeManager for testing."""
        mock_manager = Mock()
        return mock_manager

    @pytest.fixture
    def strategist(self, mock_claude_manager):
        """Create OpusStrategist instance for testing."""
        return OpusStrategist(mock_claude_manager)

    @pytest.fixture
    def sample_strategic_response(self):
        """Sample strategic response for testing."""
        return {
            "analysis": "Current strategy shows potential for optimization",
            "experiments": [
                {
                    "name": "Speed Route Optimization",
                    "description": "Test optimized routing through Viridian Forest",
                    "strategy": "aggressive",
                    "parameters": {"speed": 1.5, "risk_factor": 0.8},
                    "priority": 0.9,
                    "estimated_duration": 120.0,
                    "dependencies": [],
                },
                {
                    "name": "Conservative Approach",
                    "description": "Safe but slower route",
                    "strategy": "conservative",
                    "parameters": {"speed": 1.0, "risk_factor": 0.2},
                    "priority": 0.6,
                    "estimated_duration": 180.0,
                    "dependencies": ["Speed Route Optimization"],
                },
            ],
            "recommendations": ["Focus on early game optimization", "Test battle avoidance"],
        }

    @pytest.fixture
    def sample_experiment(self):
        """Sample experiment specification for testing."""
        return ExperimentSpec(
            id="exp_0001",
            name="Test Experiment",
            description="A test experiment for validation",
            strategy="exploratory",
            parameters={"param1": 1.0, "param2": "value"},
            priority=0.8,
            estimated_duration=60.0,
            dependencies=[],
            metadata={},
        )

    def test_opus_strategist_initialization(self, mock_claude_manager):
        """Test OpusStrategist initializes correctly."""
        strategist = OpusStrategist(mock_claude_manager)

        assert strategist.claude_manager == mock_claude_manager
        assert strategist._experiment_counter == 0
        assert len(strategist._experiment_history) == 0
        assert len(strategist._validation_rules) > 0

    def test_request_strategy_formats_prompt_correctly(self, strategist, mock_claude_manager):
        """Test strategy request formats prompt correctly."""
        game_state = {"level": 5, "location": "Viridian Forest"}
        recent_results = [{"score": 85, "success": True}]

        mock_response = json.dumps(
            {"analysis": "Test analysis", "experiments": [], "recommendations": []}
        )

        # Mock strategic process and send_message
        mock_strategic_process = Mock()
        mock_strategic_process.send_message.return_value = mock_response
        mock_claude_manager.get_strategic_process.return_value = mock_strategic_process

        result = strategist.request_strategy(game_state, recent_results)

        # Verify methods were called correctly
        mock_claude_manager.get_strategic_process.assert_called_once()
        mock_strategic_process.send_message.assert_called_once()
        call_args = mock_strategic_process.send_message.call_args[0][0]

        assert "Strategic Planning Request" in call_args
        assert str(game_state["level"]) in call_args  # Check level is in prompt
        assert game_state["location"] in call_args  # Check location is in prompt
        assert "experiments" in call_args.lower()

        assert isinstance(result, dict)
        assert "experiments" in result

    def test_extract_experiments_from_response(self, strategist, sample_strategic_response):
        """Test extraction of experiments from strategic response."""
        # Time the extraction to ensure <100ms performance requirement
        start_time = time.time()
        experiments = strategist.extract_experiments_from_response(sample_strategic_response)
        extraction_time = (time.time() - start_time) * 1000

        # Verify performance requirement
        assert (
            extraction_time < 100.0
        ), f"Extraction took {extraction_time:.1f}ms (should be <100ms)"

        # Verify extraction results
        assert len(experiments) == 2

        exp1 = experiments[0]
        assert exp1.name == "Speed Route Optimization"
        assert exp1.strategy == "aggressive"
        assert exp1.priority == 0.9
        assert exp1.parameters["speed"] == 1.5

        exp2 = experiments[1]
        assert exp2.name == "Conservative Approach"
        assert exp2.strategy == "conservative"
        assert exp2.dependencies == ["Speed Route Optimization"]

    def test_generate_experiment_variations_performance(self, strategist, sample_experiment):
        """Test experiment variation generation meets performance requirements."""
        # Time the variation generation to ensure <50ms per experiment
        start_time = time.time()
        variations = strategist.generate_experiment_variations(sample_experiment, count=3)
        generation_time = (time.time() - start_time) * 1000

        # Verify performance requirement (<50ms per experiment)
        assert (
            generation_time < 150.0
        ), f"Generation took {generation_time:.1f}ms for 3 experiments (should be <150ms total)"

        # Verify variation results
        assert len(variations) == 3

        for variation in variations:
            assert variation.id != sample_experiment.id
            assert (
                sample_experiment.name in variation.name
                or sample_experiment.strategy != variation.strategy
            )
            assert (
                variation.priority <= sample_experiment.priority
            )  # Variations have lower priority

    def test_generate_parameter_variations(self, strategist, sample_experiment):
        """Test parameter-based experiment variations."""
        variations = strategist._generate_parameter_variations(sample_experiment, count=2)

        assert len(variations) == 2

        for variation in variations:
            assert "var_" in variation.id
            assert variation.strategy == sample_experiment.strategy  # Strategy unchanged
            assert variation.metadata["variation_type"] == "parameter"
            assert variation.metadata["base_experiment"] == sample_experiment.id

            # Parameters should be modified
            if "param1" in variation.parameters:
                assert variation.parameters["param1"] != sample_experiment.parameters["param1"]

    def test_generate_strategy_variations(self, strategist, sample_experiment):
        """Test strategy-based experiment variations."""
        variations = strategist._generate_strategy_variations(sample_experiment, count=2)

        assert len(variations) <= 2  # May be fewer if strategies match

        for variation in variations:
            assert "strat_" in variation.id
            assert variation.strategy != sample_experiment.strategy  # Strategy changed
            assert variation.metadata["variation_type"] == "strategy"
            assert variation.metadata["base_experiment"] == sample_experiment.id

            # Parameters should be unchanged
            assert variation.parameters == sample_experiment.parameters

    def test_validate_experiment_executability_performance(self, strategist, sample_experiment):
        """Test experiment validation meets performance requirements."""
        # Time the validation to ensure <10ms per specification
        start_time = time.time()
        is_valid, issues = strategist.validate_experiment_executability(sample_experiment)
        validation_time = (time.time() - start_time) * 1000

        # Verify performance requirement
        assert validation_time < 10.0, f"Validation took {validation_time:.1f}ms (should be <10ms)"

        # Verify validation results
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

        if not is_valid:
            assert len(issues) > 0

    def test_validate_experiment_with_valid_spec(self, strategist, sample_experiment):
        """Test validation of valid experiment specification."""
        is_valid, issues = strategist.validate_experiment_executability(sample_experiment)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_experiment_with_invalid_spec(self, strategist):
        """Test validation of invalid experiment specification."""
        invalid_experiment = ExperimentSpec(
            id="",  # Invalid: empty ID
            name="",  # Invalid: empty name
            description="",  # Invalid: empty description
            strategy="invalid",
            parameters="not_a_dict",  # Invalid: not a dict
            priority=1.5,  # Invalid: priority > 1.0
            estimated_duration=-10.0,  # Invalid: negative duration
            dependencies="not_a_list",  # Invalid: not a list
            metadata={},
        )

        is_valid, issues = strategist.validate_experiment_executability(invalid_experiment)

        assert is_valid is False
        assert len(issues) > 0

        # Check specific validation failures
        issue_text = " ".join(issues)
        assert "has_name" in issue_text or "name" in issue_text.lower()
        assert "valid_priority" in issue_text or "priority" in issue_text.lower()
        assert "positive_duration" in issue_text or "duration" in issue_text.lower()

    def test_create_experiment_metadata(self, strategist, sample_experiment):
        """Test experiment metadata creation."""
        additional_data = {"custom_field": "custom_value"}
        metadata = strategist.create_experiment_metadata(sample_experiment, additional_data)

        # Verify required fields
        assert metadata["id"] == sample_experiment.id
        assert metadata["name"] == sample_experiment.name
        assert metadata["strategy"] == sample_experiment.strategy
        assert metadata["priority"] == sample_experiment.priority
        assert "created_at" in metadata
        assert "tags" in metadata
        assert "complexity_score" in metadata
        assert "diversity_score" in metadata

        # Verify additional data included
        assert metadata["custom_field"] == "custom_value"

        # Verify metadata stored in history
        assert len(strategist._experiment_history) == 1
        assert strategist._experiment_history[0]["id"] == sample_experiment.id

    def test_experiment_tags_generation(self, strategist):
        """Test experiment tag generation."""
        high_priority_exp = ExperimentSpec(
            id="test",
            name="Test",
            description="Test",
            strategy="aggressive",
            parameters={},
            priority=0.9,
            estimated_duration=60.0,
            dependencies=[],
            metadata={},
        )

        tags = strategist._generate_experiment_tags(high_priority_exp)

        assert "aggressive" in tags
        assert "high_priority" in tags
        assert "quick" in tags  # Duration < 300s

        low_priority_long_exp = ExperimentSpec(
            id="test2",
            name="Test2",
            description="Test2",
            strategy="conservative",
            parameters={},
            priority=0.2,
            estimated_duration=600.0,
            dependencies=["dep1"],
            metadata={},
        )

        tags2 = strategist._generate_experiment_tags(low_priority_long_exp)

        assert "conservative" in tags2
        assert "low_priority" in tags2
        assert "long_running" in tags2  # Duration > 300s
        assert "has_dependencies" in tags2

    def test_complexity_score_calculation(self, strategist):
        """Test experiment complexity score calculation."""
        simple_exp = ExperimentSpec(
            id="simple",
            name="Simple",
            description="Simple",
            strategy="basic",
            parameters={},
            priority=0.5,
            estimated_duration=60.0,
            dependencies=[],
            metadata={},
        )

        complex_exp = ExperimentSpec(
            id="complex",
            name="Complex",
            description="Complex",
            strategy="advanced",
            parameters={"p1": 1, "p2": 2, "p3": 3, "p4": 4, "p5": 5},  # 5 parameters
            priority=0.8,
            estimated_duration=600.0,
            dependencies=["d1", "d2"],
            metadata={},
        )

        simple_score = strategist._calculate_complexity_score(simple_exp)
        complex_score = strategist._calculate_complexity_score(complex_exp)

        assert 0.0 <= simple_score <= 1.0
        assert 0.0 <= complex_score <= 1.0
        assert complex_score > simple_score

    def test_diversity_score_calculation(self, strategist):
        """Test experiment diversity score calculation."""
        # First experiment should have maximum diversity
        exp1 = ExperimentSpec(
            id="exp1",
            name="Exp1",
            description="Exp1",
            strategy="strategy1",
            parameters={"p1": 1},
            priority=0.5,
            estimated_duration=60.0,
            dependencies=[],
            metadata={},
        )

        score1 = strategist._calculate_diversity_score(exp1)
        assert score1 == 1.0  # First experiment is maximally diverse

        # Add experiment to history
        strategist._experiment_history.append({"strategy": "strategy1"})

        # Same strategy experiment should have lower diversity
        exp2 = ExperimentSpec(
            id="exp2",
            name="Exp2",
            description="Exp2",
            strategy="strategy1",
            parameters={"p1": 1},
            priority=0.5,
            estimated_duration=60.0,
            dependencies=[],
            metadata={},
        )

        score2 = strategist._calculate_diversity_score(exp2)
        assert score2 < 1.0

        # Different strategy experiment should have higher diversity
        exp3 = ExperimentSpec(
            id="exp3",
            name="Exp3",
            description="Exp3",
            strategy="strategy2",
            parameters={"p1": 1, "p2": 2},
            priority=0.5,
            estimated_duration=60.0,
            dependencies=[],
            metadata={},
        )

        score3 = strategist._calculate_diversity_score(exp3)
        assert score3 > score2


@pytest.mark.fast
class TestOpusStrategistParallelResultsAnalysis:
    """Test parallel results analysis functionality (Issue #100)."""

    @pytest.fixture
    def strategist(self):
        """Create OpusStrategist instance for testing."""
        mock_manager = Mock()
        return OpusStrategist(mock_manager)

    @pytest.fixture
    def sample_results(self):
        """Sample execution results for testing."""
        return [
            {
                "success": True,
                "score": 85,
                "strategy": "aggressive",
                "performance": {"speed": 1.2, "accuracy": 0.85, "efficiency": 0.78},
            },
            {
                "success": False,
                "score": 45,
                "strategy": "conservative",
                "performance": {"speed": 0.8, "accuracy": 0.95, "efficiency": 0.65},
            },
            {
                "success": True,
                "score": 92,
                "strategy": "aggressive",
                "performance": {"speed": 1.5, "accuracy": 0.82, "efficiency": 0.88},
            },
        ]

    def test_analyze_parallel_results_performance(self, strategist, sample_results):
        """Test parallel results analysis meets performance requirements."""
        # Time the analysis to ensure <500ms strategic response requirement
        start_time = time.time()
        analysis = strategist.analyze_parallel_results(sample_results)
        analysis_time = (time.time() - start_time) * 1000

        # Verify performance requirement (should be much faster than 500ms)
        assert analysis_time < 500.0, f"Analysis took {analysis_time:.1f}ms (should be <500ms)"

        # Verify analysis results
        assert isinstance(analysis, list)
        assert len(analysis) > 0

    def test_extract_metrics_from_results(self, strategist, sample_results):
        """Test metric extraction from execution results."""
        metrics = strategist._extract_metrics(sample_results)

        assert "speed" in metrics
        assert "accuracy" in metrics
        assert "efficiency" in metrics

        assert len(metrics["speed"]) == 3
        assert len(metrics["accuracy"]) == 3
        assert len(metrics["efficiency"]) == 3

        assert metrics["speed"] == [1.2, 0.8, 1.5]
        assert metrics["accuracy"] == [0.85, 0.95, 0.82]
        assert metrics["efficiency"] == [0.78, 0.65, 0.88]

    def test_calculate_correlations(self, strategist, sample_results):
        """Test correlation calculation between metrics."""
        metrics = strategist._extract_metrics(sample_results)
        correlations = strategist._calculate_correlations(metrics)

        assert isinstance(correlations, dict)

        # Check that correlations are calculated for metric pairs
        expected_pairs = ["speed_vs_accuracy", "speed_vs_efficiency", "accuracy_vs_efficiency"]

        for pair in expected_pairs:
            if pair in correlations:
                correlation = correlations[pair]
                assert -1.0 <= correlation <= 1.0

    def test_identify_patterns(self, strategist, sample_results):
        """Test pattern identification across results."""
        patterns = strategist._identify_patterns(sample_results)

        # Verify pattern structure
        assert "success_rate" in patterns
        assert "average_score" in patterns
        assert "common_strategies" in patterns
        assert "failure_modes" in patterns

        # Verify calculated values
        assert patterns["success_rate"] == 2 / 3  # 2 successes out of 3
        assert patterns["average_score"] == (85 + 45 + 92) / 3  # Average score

        # Verify strategy analysis
        strategies = patterns["common_strategies"]
        assert len(strategies) > 0
        assert strategies[0][0] == "aggressive"  # Most common strategy
        assert strategies[0][1] == 2  # Appears 2 times

    def test_compress_analysis(self, strategist):
        """Test analysis compression for efficiency."""
        # Test with significant correlations
        correlations = {
            "speed_vs_accuracy": -0.5,  # Significant (>0.3 magnitude)
            "speed_vs_efficiency": 0.2,  # Not significant
            "accuracy_vs_efficiency": 0.8,  # Significant
        }

        patterns = {
            "success_rate": 0.75,
            "average_score": 85.5,
            "common_strategies": [("aggressive", 2), ("conservative", 1)],
        }

        compressed = strategist._compress_analysis(correlations, patterns)

        # Verify compression structure
        assert isinstance(compressed, list)
        assert len(compressed) == 2  # correlations + patterns

        # Check correlations compression
        corr_section = next(item for item in compressed if item["type"] == "correlations")
        assert len(corr_section["data"]) == 2  # Only significant correlations
        assert "speed_vs_accuracy" in corr_section["data"]
        assert "accuracy_vs_efficiency" in corr_section["data"]
        assert "speed_vs_efficiency" not in corr_section["data"]  # Filtered out

        # Check patterns compression
        pattern_section = next(item for item in compressed if item["type"] == "patterns")
        assert pattern_section["data"]["success_rate"] == 0.75
        assert pattern_section["data"]["average_score"] == 85.5
        assert pattern_section["data"]["top_strategy"] == "aggressive"

    def test_analyze_parallel_results_empty_input(self, strategist):
        """Test analysis with empty results."""
        analysis = strategist.analyze_parallel_results([])

        assert analysis == []

    def test_analyze_parallel_results_no_performance_data(self, strategist):
        """Test analysis with results missing performance data."""
        results_no_perf = [{"success": True, "score": 85}, {"success": False, "score": 45}]

        analysis = strategist.analyze_parallel_results(results_no_perf)

        # Should still return analysis even without performance metrics
        assert isinstance(analysis, list)
        assert len(analysis) > 0

        # Should contain patterns even without correlations
        pattern_section = next(item for item in analysis if item["type"] == "patterns")
        assert pattern_section["data"]["success_rate"] == 0.5
        assert pattern_section["data"]["average_score"] == 65.0


@pytest.mark.integration
@pytest.mark.fast
class TestOpusStrategistIntegration:
    """Integration tests for OpusStrategist functionality."""

    def test_full_experiment_generation_workflow(self):
        """Test complete experiment generation workflow."""
        mock_manager = Mock()
        strategist = OpusStrategist(mock_manager)

        # Mock strategic response
        strategic_response = {
            "analysis": "Test analysis",
            "experiments": [
                {
                    "name": "Test Experiment",
                    "description": "Full workflow test",
                    "strategy": "test_strategy",
                    "parameters": {"param1": 1.0},
                    "priority": 0.8,
                    "estimated_duration": 120.0,
                    "dependencies": [],
                }
            ],
            "recommendations": ["Test recommendation"],
        }

        # Extract experiments
        experiments = strategist.extract_experiments_from_response(strategic_response)
        assert len(experiments) == 1

        base_experiment = experiments[0]

        # Generate variations
        variations = strategist.generate_experiment_variations(base_experiment, count=3)
        assert len(variations) == 3

        # Validate all experiments
        all_experiments = [base_experiment] + variations
        for experiment in all_experiments:
            is_valid, issues = strategist.validate_experiment_executability(experiment)
            assert is_valid, f"Experiment {experiment.id} validation failed: {issues}"

        # Create metadata for all experiments
        metadata_list = []
        for experiment in all_experiments:
            metadata = strategist.create_experiment_metadata(experiment)
            metadata_list.append(metadata)

        assert len(metadata_list) == 4  # Base + 3 variations
        assert len(strategist._experiment_history) == 4

    @pytest.mark.performance
    def test_performance_requirements_end_to_end(self):
        """Test all performance requirements in end-to-end scenario."""
        mock_manager = Mock()
        strategist = OpusStrategist(mock_manager)

        # Prepare test data
        strategic_response = {
            "experiments": [
                {
                    "name": f"Experiment {i}",
                    "description": f"Performance test experiment {i}",
                    "strategy": "performance_test",
                    "parameters": {"param1": i * 1.0},
                    "priority": 0.7,
                    "estimated_duration": 100.0,
                    "dependencies": [],
                }
                for i in range(5)  # 5 experiments
            ]
        }

        # Test extraction performance
        start_time = time.time()
        experiments = strategist.extract_experiments_from_response(strategic_response)
        extraction_time = (time.time() - start_time) * 1000
        assert extraction_time < 100.0, f"Extraction took {extraction_time:.1f}ms"

        # Test variation generation performance
        for experiment in experiments:
            start_time = time.time()
            strategist.generate_experiment_variations(experiment, count=3)
            generation_time = (time.time() - start_time) * 1000
            assert (
                generation_time < 150.0
            ), f"Generation took {generation_time:.1f}ms for 3 variations"

        # Test validation performance
        all_experiments = experiments + [
            v for exp in experiments for v in strategist.generate_experiment_variations(exp, 1)
        ]
        for experiment in all_experiments:
            start_time = time.time()
            is_valid, issues = strategist.validate_experiment_executability(experiment)
            validation_time = (time.time() - start_time) * 1000
            assert validation_time < 10.0, f"Validation took {validation_time:.1f}ms"
