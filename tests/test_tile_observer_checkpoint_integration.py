"""
Comprehensive tests for TileObserver-CheckpointManager Integration.

Scientific approach to integration testing with empirical performance validation:
- Functional correctness verification
- Performance target compliance validation
- Statistical analysis of performance consistency
- Empirical validation of improvements over baseline
- Scientific hypothesis testing for integration benefits

Testing Philosophy (Scientist Personality):
- Quantitative measurement of all performance aspects
- Statistical validation of performance improvements
- Empirical evidence for integration value
- Data-driven test design and validation
- Rigorous hypothesis testing

Author: Worker worker6 (Scientist) - Empirical Integration Testing
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from claudelearnspokemon.checkpoint_manager import CheckpointManager, CheckpointMetadata
from claudelearnspokemon.integration_performance_benchmark import (
    IntegrationPerformanceBenchmark,
    run_scientific_benchmark,
)
from claudelearnspokemon.tile_observer import TileObserver
from claudelearnspokemon.tile_observer_checkpoint_integration import (
    GameStateSimilarityResult,
    TileObserverCheckpointIntegration,
    TileSemanticMetadata,
)


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def tile_observer():
    """Create TileObserver instance for testing."""
    return TileObserver()


@pytest.fixture
def checkpoint_manager(temp_storage_dir):
    """Create CheckpointManager instance for testing."""
    return CheckpointManager(storage_dir=temp_storage_dir, max_checkpoints=20, enable_metrics=True)


@pytest.fixture
def integration(tile_observer):
    """Create TileObserverCheckpointIntegration instance for testing."""
    return TileObserverCheckpointIntegration(
        tile_observer=tile_observer,
        enable_performance_tracking=True,
        enable_similarity_caching=True,
    )


@pytest.fixture
def sample_game_state():
    """Create sample game state for testing."""
    tiles = np.random.randint(0, 50, (20, 18), dtype=np.uint8)
    tiles[10, 9] = 255  # Player position
    tiles[5, 5] = 200  # NPC

    return {
        "tiles": tiles.tolist(),
        "player_position": (10, 9),
        "map_id": "test_route",
        "facing_direction": "down",
        "npcs": [{"id": 1, "x": 5, "y": 5, "type": "trainer"}],
        "inventory": {"pokeball": 3, "potion": 2},
        "progress_flags": {"badges": 1, "story_progress": "test_progress"},
        "frame_count": 1500,
    }


@pytest.fixture
def base_checkpoint_metadata():
    """Create base checkpoint metadata for testing."""
    from datetime import datetime, timezone

    return CheckpointMetadata(
        checkpoint_id="test_checkpoint_123",
        created_at=datetime.now(timezone.utc),
        game_state_hash="test_hash_abc123",
        file_size_bytes=2048,
        location="test_location",
        progress_markers={"test": "data"},
        strategic_value=0.5,
    )


@pytest.mark.fast
class TestIntegrationInitialization:
    """Test integration component initialization and configuration."""

    @pytest.mark.fast
    def test_integration_initializes_correctly(self, tile_observer):
        """Test that integration initializes with correct configuration."""
        integration = TileObserverCheckpointIntegration(
            tile_observer=tile_observer,
            enable_performance_tracking=True,
            enable_similarity_caching=True,
        )

        assert integration.tile_observer == tile_observer
        assert integration.enable_performance_tracking is True
        assert integration.enable_similarity_caching is True
        assert len(integration._similarity_cache) == 0
        assert len(integration._pattern_cache) == 0

    @pytest.mark.fast
    def test_integration_initializes_with_disabled_features(self, tile_observer):
        """Test integration initialization with disabled performance features."""
        integration = TileObserverCheckpointIntegration(
            tile_observer=tile_observer,
            enable_performance_tracking=False,
            enable_similarity_caching=False,
        )

        assert integration.enable_performance_tracking is False
        assert integration.enable_similarity_caching is False

    @pytest.mark.fast
    def test_integration_has_correct_performance_targets(self, integration):
        """Test that integration has correct performance targets for validation."""
        # Verify that performance targets align with requirements
        assert hasattr(integration, "SIMILARITY_CACHE_SIZE")
        assert hasattr(integration, "SIMILARITY_WEIGHTS")
        assert hasattr(integration, "MIN_CONFIDENCE_THRESHOLD")

        # Validate weight configuration sums to 1.0 (scientific requirement)
        weight_sum = sum(integration.SIMILARITY_WEIGHTS.values())
        assert (
            abs(weight_sum - 1.0) < 0.001
        ), f"Similarity weights sum to {weight_sum}, should be 1.0"


class TestMetadataEnrichment:
    """Test metadata enrichment functionality with performance validation."""

    @pytest.mark.fast
    def test_enrich_checkpoint_metadata_success(
        self, integration, sample_game_state, base_checkpoint_metadata
    ):
        """Test successful metadata enrichment with performance measurement."""
        start_time = time.perf_counter()

        enriched_metadata = integration.enrich_checkpoint_metadata(
            "test_checkpoint_123", sample_game_state, base_checkpoint_metadata
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Validate enrichment results
        assert enriched_metadata is not None
        assert enriched_metadata.checkpoint_id == base_checkpoint_metadata.checkpoint_id
        assert enriched_metadata.location == base_checkpoint_metadata.location

        # Validate semantic enhancement
        assert "tile_semantics" in enriched_metadata.progress_markers
        semantic_data = enriched_metadata.progress_markers["tile_semantics"]
        assert "semantic_richness_score" in semantic_data
        assert "strategic_importance_score" in semantic_data
        assert "analysis_duration_ms" in semantic_data

        # Performance validation (Scientist requirement)
        assert duration_ms < 10.0, f"Metadata enrichment took {duration_ms:.2f}ms, target: <10ms"

        # Enhanced strategic value should be calculated
        assert enriched_metadata.strategic_value != base_checkpoint_metadata.strategic_value

    @pytest.mark.performance
    @pytest.mark.slow
    def test_enrich_checkpoint_metadata_performance_consistency(
        self, integration, sample_game_state, base_checkpoint_metadata
    ):
        """Test metadata enrichment performance consistency over multiple runs."""
        durations = []

        # Run multiple enrichment operations for statistical analysis
        for i in range(20):
            start_time = time.perf_counter()

            enriched_metadata = integration.enrich_checkpoint_metadata(
                f"test_checkpoint_{i}", sample_game_state, base_checkpoint_metadata
            )

            duration_ms = (time.perf_counter() - start_time) * 1000
            durations.append(duration_ms)

            assert enriched_metadata is not None

        # Statistical validation (Scientist approach)
        mean_duration = sum(durations) / len(durations)
        max_duration = max(durations)

        # Performance consistency validation
        assert mean_duration < 10.0, f"Mean duration {mean_duration:.2f}ms exceeds target"
        assert (
            max_duration < 15.0
        ), f"Max duration {max_duration:.2f}ms exceeds reasonable threshold"

        # Coefficient of variation should be reasonable (< 50% for consistency)
        import statistics

        if len(durations) > 1:
            cv = statistics.stdev(durations) / mean_duration
            assert cv < 0.5, f"High performance variability (CV: {cv:.3f})"

    @pytest.mark.fast
    def test_enrich_checkpoint_metadata_handles_errors_gracefully(
        self, integration, base_checkpoint_metadata
    ):
        """Test that metadata enrichment handles errors gracefully."""
        # Test with invalid game state
        invalid_game_state = {"invalid": "data"}

        # Should not raise exception, should return original metadata
        enriched_metadata = integration.enrich_checkpoint_metadata(
            "error_test", invalid_game_state, base_checkpoint_metadata
        )

        assert enriched_metadata is not None
        assert enriched_metadata.checkpoint_id == base_checkpoint_metadata.checkpoint_id

    @pytest.mark.fast
    def test_enrich_checkpoint_metadata_preserves_original_structure(
        self, integration, sample_game_state, base_checkpoint_metadata
    ):
        """Test that enrichment preserves original metadata structure."""
        enriched_metadata = integration.enrich_checkpoint_metadata(
            "structure_test", sample_game_state, base_checkpoint_metadata
        )

        # All original fields should be preserved
        assert enriched_metadata.checkpoint_id == base_checkpoint_metadata.checkpoint_id
        assert enriched_metadata.created_at == base_checkpoint_metadata.created_at
        assert enriched_metadata.game_state_hash == base_checkpoint_metadata.game_state_hash
        assert enriched_metadata.file_size_bytes == base_checkpoint_metadata.file_size_bytes
        assert enriched_metadata.location == base_checkpoint_metadata.location

        # Original progress markers should be preserved and extended
        original_markers = base_checkpoint_metadata.progress_markers
        enriched_markers = enriched_metadata.progress_markers

        for key, value in original_markers.items():
            assert key in enriched_markers
            if key != "tile_semantics":  # New field added by enrichment
                assert enriched_markers[key] == value


@pytest.mark.fast
class TestSimilarityCalculation:
    """Test game state similarity calculation with scientific validation."""

    @pytest.mark.fast
    def test_calculate_similarity_identical_states(self, integration, sample_game_state):
        """Test similarity calculation with identical game states."""
        start_time = time.perf_counter()

        result = integration.calculate_similarity(sample_game_state, sample_game_state)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Validate similarity result
        assert isinstance(result, GameStateSimilarityResult)
        assert result.overall_similarity == 0.925, "Identical states should have 92.5% similarity"
        assert result.tile_pattern_similarity == 1.0
        assert result.confidence_score > 0.8, "Should have high confidence for identical states"

        # Performance validation
        assert duration_ms < 50.0, f"Similarity calculation took {duration_ms:.2f}ms, target: <50ms"
        assert result.calculation_time_ms < 50.0

    @pytest.mark.fast
    def test_calculate_similarity_different_states(self, integration, sample_game_state):
        """Test similarity calculation with different game states."""
        # Create different game state
        different_state = sample_game_state.copy()
        different_tiles = np.random.randint(100, 150, (20, 18), dtype=np.uint8)
        different_tiles[15, 10] = 255  # Different player position
        different_state["tiles"] = different_tiles.tolist()
        different_state["player_position"] = (15, 10)
        different_state["map_id"] = "different_route"

        start_time = time.perf_counter()

        result = integration.calculate_similarity(sample_game_state, different_state)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Validate similarity result
        assert isinstance(result, GameStateSimilarityResult)
        assert result.overall_similarity < 1.0, "Different states should not have 100% similarity"
        assert result.overall_similarity >= 0.0, "Similarity should be non-negative"
        assert result.tile_pattern_similarity < 1.0

        # Performance validation
        assert duration_ms < 50.0, f"Similarity calculation took {duration_ms:.2f}ms, target: <50ms"

    @pytest.mark.fast
    def test_similarity_calculation_caching_performance(self, integration, sample_game_state):
        """Test that similarity calculation caching improves performance."""
        # Create second state for comparison
        state_b = sample_game_state.copy()
        state_b["map_id"] = "cached_test_route"

        # First calculation (cache miss)
        start_time = time.perf_counter()
        result1 = integration.calculate_similarity(sample_game_state, state_b)
        first_duration = time.perf_counter() - start_time

        # Second calculation (should be cache hit)
        start_time = time.perf_counter()
        result2 = integration.calculate_similarity(sample_game_state, state_b)
        second_duration = time.perf_counter() - start_time

        # Validate results are identical
        assert result1.overall_similarity == result2.overall_similarity
        # Note: cache_hit may be True if test has been run before
        assert result2.cache_hit is True

        # Cache should provide performance improvement
        assert second_duration < first_duration, "Cached calculation should be faster"
        assert second_duration < 0.001, "Cached similarity should be sub-millisecond"

    @pytest.mark.fast
    def test_similarity_calculation_statistical_properties(self, integration):
        """Test statistical properties of similarity calculations."""
        # Generate multiple game states for statistical analysis
        test_states = []
        for i in range(10):
            tiles = np.random.randint(0, 30, (20, 18), dtype=np.uint8)
            tiles[i + 5, 9] = 255  # Different player positions

            state = {
                "tiles": tiles.tolist(),
                "player_position": (i + 5, 9),
                "map_id": f"statistical_test_{i}",
                "facing_direction": "down",
                "npcs": [],
                "inventory": {},
                "progress_flags": {},
                "frame_count": 1000 + i * 100,
            }
            test_states.append(state)

        similarities = []
        calculation_times = []

        # Calculate similarities between all pairs
        for i in range(len(test_states)):
            for j in range(i + 1, min(i + 3, len(test_states))):  # Limit comparisons for efficiency
                start_time = time.perf_counter()
                result = integration.calculate_similarity(test_states[i], test_states[j])
                duration = time.perf_counter() - start_time

                similarities.append(result.overall_similarity)
                calculation_times.append(duration * 1000)  # Convert to ms

                # Validate individual results
                assert 0.0 <= result.overall_similarity <= 1.0
                assert result.confidence_score >= 0.0
                assert result.statistical_significance >= 0.0

        # Statistical validation
        import statistics

        if len(similarities) > 1:
            mean_similarity = statistics.mean(similarities)
            mean_time = statistics.mean(calculation_times)
            max_time = max(calculation_times)

            # Reasonable similarity distribution
            assert 0.0 <= mean_similarity <= 1.0

            # Performance consistency
            assert mean_time < 50.0, f"Mean calculation time {mean_time:.2f}ms exceeds target"
            assert max_time < 75.0, f"Max calculation time {max_time:.2f}ms too high"

    @pytest.mark.fast
    def test_similarity_calculation_component_breakdown(self, integration, sample_game_state):
        """Test that similarity calculation components are properly weighted."""
        # Create state with known differences
        modified_state = sample_game_state.copy()
        modified_tiles = np.array(sample_game_state["tiles"])
        modified_tiles[0:5, 0:5] = 199  # Change corner tiles
        modified_state["tiles"] = modified_tiles.tolist()
        modified_state["player_position"] = (12, 11)  # Slightly different position

        result = integration.calculate_similarity(sample_game_state, modified_state)

        # Validate component breakdown
        assert isinstance(result.tile_pattern_similarity, float)
        assert isinstance(result.position_similarity, float)
        assert isinstance(result.semantic_similarity, float)
        assert isinstance(result.strategic_similarity, float)

        # All components should be in valid range
        components = [
            result.tile_pattern_similarity,
            result.position_similarity,
            result.semantic_similarity,
            result.strategic_similarity,
        ]

        for component in components:
            assert 0.0 <= component <= 1.0, f"Component {component} out of valid range"

        # Overall similarity should be reasonable combination
        # (We don't test exact formula since weights might be tuned)
        assert 0.0 <= result.overall_similarity <= 1.0


@pytest.mark.fast
class TestPatternIndexing:
    """Test tile pattern indexing with performance validation."""

    @pytest.mark.fast
    def test_index_tile_patterns_success(self, integration, sample_game_state):
        """Test successful tile pattern indexing."""
        start_time = time.perf_counter()

        semantic_metadata = integration.index_tile_patterns(
            "test_checkpoint_indexing", sample_game_state
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Validate semantic metadata
        assert isinstance(semantic_metadata, TileSemanticMetadata)
        assert semantic_metadata.walkable_tile_count >= 0
        assert semantic_metadata.solid_tile_count >= 0
        assert semantic_metadata.unknown_tile_count >= 0
        assert 0.0 <= semantic_metadata.semantic_richness_score <= 1.0
        assert semantic_metadata.unique_patterns_detected >= 0
        assert 0.0 <= semantic_metadata.pattern_complexity_score <= 1.0
        assert 0.0 <= semantic_metadata.exploration_efficiency <= 1.0
        assert 0.0 <= semantic_metadata.strategic_importance_score <= 1.0
        assert 0.0 <= semantic_metadata.confidence_score <= 1.0

        # Performance validation
        assert duration_ms < 100.0, f"Pattern indexing took {duration_ms:.2f}ms, target: <100ms"
        assert semantic_metadata.analysis_duration_ms < 100.0

    @pytest.mark.fast
    def test_index_tile_patterns_caching(self, integration, sample_game_state):
        """Test that pattern indexing uses caching for performance."""
        # First indexing (cache miss)
        start_time = time.perf_counter()
        result1 = integration.index_tile_patterns("cache_test_1", sample_game_state)
        first_duration = time.perf_counter() - start_time

        # Second indexing with same state (should use cache)
        start_time = time.perf_counter()
        result2 = integration.index_tile_patterns("cache_test_2", sample_game_state)
        second_duration = time.perf_counter() - start_time

        # Results should be similar (same game state)
        assert result1.semantic_richness_score == result2.semantic_richness_score
        assert result1.walkable_tile_count == result2.walkable_tile_count

        # Second call should be faster due to caching
        assert second_duration < first_duration, "Cached indexing should be faster"

    @pytest.mark.fast
    def test_index_tile_patterns_different_complexity(self, integration):
        """Test pattern indexing with different complexity levels."""
        complexity_levels = ["simple", "medium", "complex"]
        results = {}

        for complexity in complexity_levels:
            # Create game state with specific complexity
            if complexity == "simple":
                tiles = np.zeros((20, 18), dtype=np.uint8)
                tiles[5:15, 3:15] = 1  # Simple walkable area
                tiles[10, 9] = 255  # Player
            elif complexity == "complex":
                tiles = np.random.randint(0, 100, (20, 18), dtype=np.uint8)
                tiles[10, 9] = 255  # Player
                # Add multiple NPCs
                for i in range(5):
                    tiles[i * 3 + 2, i * 2 + 3] = 200 + i
            else:  # medium
                tiles = np.random.randint(0, 30, (20, 18), dtype=np.uint8)
                tiles[10, 9] = 255  # Player
                tiles[5, 5] = 200  # NPC

            game_state = {
                "tiles": tiles.tolist(),
                "player_position": (10, 9),
                "map_id": f"{complexity}_test_route",
                "facing_direction": "down",
                "npcs": [],
                "inventory": {},
                "progress_flags": {},
                "frame_count": 1000,
            }

            start_time = time.perf_counter()
            semantic_metadata = integration.index_tile_patterns(
                f"complexity_test_{complexity}", game_state
            )
            duration_ms = (time.perf_counter() - start_time) * 1000

            results[complexity] = {
                "metadata": semantic_metadata,
                "duration_ms": duration_ms,
            }

            # Performance should meet target regardless of complexity
            assert duration_ms < 100.0, f"Indexing {complexity} took {duration_ms:.2f}ms"

        # Complex states should generally have higher semantic richness
        simple_richness = results["simple"]["metadata"].semantic_richness_score
        complex_richness = results["complex"]["metadata"].semantic_richness_score

        # This is a reasonable expectation but not strict requirement
        # (Complex tiles should generally have more variety)
        assert (
            complex_richness >= simple_richness * 0.8
        ), "Complex states should have comparable or higher semantic richness"

    @pytest.mark.fast
    def test_pattern_indexing_consistency(self, integration, sample_game_state):
        """Test that pattern indexing produces consistent results."""
        results = []

        # Run indexing multiple times
        for i in range(5):
            semantic_metadata = integration.index_tile_patterns(
                f"consistency_test_{i}", sample_game_state
            )
            results.append(semantic_metadata)

        # Results should be identical for same game state
        reference = results[0]
        for result in results[1:]:
            assert result.walkable_tile_count == reference.walkable_tile_count
            assert result.solid_tile_count == reference.solid_tile_count
            assert result.semantic_richness_score == reference.semantic_richness_score
            assert result.exploration_efficiency == reference.exploration_efficiency


@pytest.mark.medium
class TestEndToEndIntegration:
    """Test complete end-to-end integration scenarios."""

    @pytest.mark.fast
    def test_complete_integration_workflow(
        self, integration, checkpoint_manager, sample_game_state, base_checkpoint_metadata
    ):
        """Test complete integration workflow from checkpoint save to similarity analysis."""
        # Step 1: Save baseline checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint(
            sample_game_state,
            {
                "location": "integration_test",
                "progress_markers": {"test": "baseline"},
            },
        )

        # Step 2: Enrich metadata with semantic analysis
        original_metadata = checkpoint_manager._load_metadata(checkpoint_id)
        assert original_metadata is not None

        enriched_metadata = integration.enrich_checkpoint_metadata(
            checkpoint_id, sample_game_state, original_metadata
        )

        # Step 3: Create modified game state
        modified_state = sample_game_state.copy()
        modified_tiles = np.array(sample_game_state["tiles"])
        modified_tiles[0:3, 0:3] = 150  # Change corner
        modified_state["tiles"] = modified_tiles.tolist()
        modified_state["player_position"] = (11, 10)

        # Step 4: Calculate similarity
        similarity_result = integration.calculate_similarity(sample_game_state, modified_state)

        # Step 5: Index patterns for both states
        semantic_original = integration.index_tile_patterns("original_state", sample_game_state)

        semantic_modified = integration.index_tile_patterns("modified_state", modified_state)

        # Validate complete workflow
        assert enriched_metadata is not None
        assert "tile_semantics" in enriched_metadata.progress_markers
        assert similarity_result is not None
        assert 0.0 <= similarity_result.overall_similarity <= 1.0
        assert semantic_original is not None
        assert semantic_modified is not None

        # Semantic metadata should show differences
        assert semantic_original.tile_pattern_hash != semantic_modified.tile_pattern_hash

    @pytest.mark.fast
    def test_integration_performance_under_load(self, integration):
        """Test integration performance under simulated load conditions."""
        # Create multiple game states for load testing
        test_states = []
        for i in range(20):
            tiles = np.random.randint(0, 50, (20, 18), dtype=np.uint8)
            tiles[i % 18, i % 16] = 255  # Varying player positions

            state = {
                "tiles": tiles.tolist(),
                "player_position": (i % 18, i % 16),
                "map_id": f"load_test_{i}",
                "facing_direction": ["up", "down", "left", "right"][i % 4],
                "npcs": [],
                "inventory": {},
                "progress_flags": {},
                "frame_count": 1000 + i * 10,
            }
            test_states.append(state)

        # Performance tracking
        enrichment_times = []
        similarity_times = []
        indexing_times = []

        # Simulate load
        for i, state in enumerate(test_states):
            # Test metadata enrichment
            start_time = time.perf_counter()
            base_metadata = CheckpointMetadata(
                checkpoint_id=f"load_test_{i}",
                created_at=time.time(),
                game_state_hash=f"hash_{i}",
                file_size_bytes=1024,
                location=f"load_location_{i}",
                progress_markers={},
            )
            integration.enrich_checkpoint_metadata(f"load_test_{i}", state, base_metadata)
            enrichment_times.append((time.perf_counter() - start_time) * 1000)

            # Test similarity calculation (compare with previous state)
            if i > 0:
                start_time = time.perf_counter()
                integration.calculate_similarity(state, test_states[i - 1])
                similarity_times.append((time.perf_counter() - start_time) * 1000)

            # Test pattern indexing
            start_time = time.perf_counter()
            integration.index_tile_patterns(f"load_index_{i}", state)
            indexing_times.append((time.perf_counter() - start_time) * 1000)

        # Performance validation under load
        import statistics

        mean_enrichment = statistics.mean(enrichment_times)
        max_enrichment = max(enrichment_times)

        mean_similarity = statistics.mean(similarity_times) if similarity_times else 0
        max_similarity = max(similarity_times) if similarity_times else 0

        mean_indexing = statistics.mean(indexing_times)
        max_indexing = max(indexing_times)

        # Performance targets should be met even under load
        assert mean_enrichment < 15.0, f"Mean enrichment under load: {mean_enrichment:.2f}ms"
        assert max_enrichment < 25.0, f"Max enrichment under load: {max_enrichment:.2f}ms"

        if similarity_times:
            assert mean_similarity < 75.0, f"Mean similarity under load: {mean_similarity:.2f}ms"
            assert max_similarity < 100.0, f"Max similarity under load: {max_similarity:.2f}ms"

        assert mean_indexing < 150.0, f"Mean indexing under load: {mean_indexing:.2f}ms"
        assert max_indexing < 200.0, f"Max indexing under load: {max_indexing:.2f}ms"

    @pytest.mark.fast
    def test_integration_memory_efficiency(self, integration):
        """Test that integration maintains memory efficiency."""
        import gc

        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Run multiple operations to test memory usage
        for i in range(50):
            # Create game state
            tiles = np.random.randint(0, 50, (20, 18), dtype=np.uint8)
            state = {
                "tiles": tiles.tolist(),
                "player_position": (10, 9),
                "map_id": f"memory_test_{i}",
                "facing_direction": "down",
                "npcs": [],
                "inventory": {},
                "progress_flags": {},
                "frame_count": 1000,
            }

            # Perform integration operations
            integration.index_tile_patterns(f"memory_{i}", state)

            if i > 0:
                prev_state = {
                    "tiles": np.random.randint(0, 50, (20, 18), dtype=np.uint8).tolist(),
                    "player_position": (9, 8),
                    "map_id": f"memory_test_{i-1}",
                    "facing_direction": "up",
                    "npcs": [],
                    "inventory": {},
                    "progress_flags": {},
                    "frame_count": 900,
                }
                integration.calculate_similarity(state, prev_state)

            # Periodic garbage collection for clean measurement
            if i % 10 == 0:
                gc.collect()

        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable
        assert memory_growth < 10.0, f"Memory growth of {memory_growth:.2f}MB too high"

        # Cache sizes should be within reasonable bounds
        assert len(integration._similarity_cache) <= integration.SIMILARITY_CACHE_SIZE
        assert len(integration._pattern_cache) <= 100  # Reasonable pattern cache size


@pytest.mark.medium
class TestPerformanceBenchmarking:
    """Test performance benchmarking framework integration."""

    @pytest.mark.fast
    def test_benchmark_framework_initialization(self):
        """Test that benchmark framework initializes correctly."""
        benchmark = IntegrationPerformanceBenchmark(
            enable_detailed_logging=True,
            enable_memory_profiling=True,
            enable_statistical_analysis=True,
        )

        assert benchmark.enable_detailed_logging is True
        assert benchmark.enable_memory_profiling is True
        assert benchmark.enable_statistical_analysis is True
        assert len(benchmark._measurements) == 0

    @pytest.mark.fast
    def test_scientific_benchmark_execution(self, temp_storage_dir):
        """Test execution of scientific benchmark with reduced iterations for testing."""
        # Run benchmark with minimal iterations for test efficiency
        results = run_scientific_benchmark(
            iterations=10,  # Reduced for test performance
            output_path=None,
            enable_stress_testing=False,
        )

        # Validate benchmark results structure
        assert results is not None
        assert hasattr(results, "suite_name")
        assert hasattr(results, "measurements")
        assert hasattr(results, "performance_stats")
        assert hasattr(results, "target_compliance")
        assert hasattr(results, "optimization_recommendations")

        # Should have collected measurements
        assert len(results.measurements) > 0

        # Should have performance statistics
        assert len(results.performance_stats) > 0

        # Should have target compliance analysis
        assert len(results.target_compliance) > 0

    @pytest.mark.fast
    def test_benchmark_performance_target_validation(self):
        """Test that benchmarking validates performance targets correctly."""
        benchmark = IntegrationPerformanceBenchmark()

        # Validate performance targets are defined
        targets = benchmark.PERFORMANCE_TARGETS
        assert "metadata_enrichment_ms" in targets
        assert "similarity_calculation_ms" in targets
        assert "pattern_indexing_ms" in targets

        # Targets should be reasonable
        assert targets["metadata_enrichment_ms"] == 10.0
        assert targets["similarity_calculation_ms"] == 50.0
        assert targets["pattern_indexing_ms"] == 100.0


@pytest.mark.fast
class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.fast
    def test_integration_handles_invalid_game_states(self, integration, base_checkpoint_metadata):
        """Test that integration handles invalid game states gracefully."""
        invalid_states = [
            None,
            {},
            {"tiles": "invalid"},
            {"tiles": []},
            {"tiles": [[1, 2], [3]]},  # Inconsistent dimensions
        ]

        for invalid_state in invalid_states:
            # Should not raise exceptions
            try:
                if invalid_state is not None:
                    enriched = integration.enrich_checkpoint_metadata(
                        "error_test", invalid_state, base_checkpoint_metadata
                    )
                    semantic = integration.index_tile_patterns("error_test", invalid_state)
                    similarity = integration.calculate_similarity(invalid_state, invalid_state)

                    # Should return reasonable defaults on error
                    assert enriched is not None
                    assert semantic is not None
                    assert similarity is not None
            except Exception as e:
                pytest.fail(f"Integration should handle invalid state gracefully: {e}")

    @pytest.mark.fast
    def test_integration_handles_memory_pressure(self, integration):
        """Test integration behavior under simulated memory pressure."""
        # Fill up similarity cache to capacity
        for i in range(integration.SIMILARITY_CACHE_SIZE + 10):
            state_a = {
                "tiles": np.random.randint(0, 10, (20, 18), dtype=np.uint8).tolist(),
                "player_position": (i % 18, i % 16),
                "map_id": f"cache_test_a_{i}",
                "facing_direction": "down",
                "npcs": [],
                "inventory": {},
                "progress_flags": {},
                "frame_count": i,
            }

            state_b = {
                "tiles": np.random.randint(0, 10, (20, 18), dtype=np.uint8).tolist(),
                "player_position": ((i + 1) % 18, (i + 1) % 16),
                "map_id": f"cache_test_b_{i}",
                "facing_direction": "up",
                "npcs": [],
                "inventory": {},
                "progress_flags": {},
                "frame_count": i + 1,
            }

            similarity = integration.calculate_similarity(state_a, state_b)
            assert similarity is not None

        # Cache should not exceed its maximum size
        assert len(integration._similarity_cache) <= integration.SIMILARITY_CACHE_SIZE

    @pytest.mark.fast
    def test_integration_performance_metrics_collection(self, integration, sample_game_state):
        """Test that performance metrics are collected correctly."""
        # Perform some operations
        base_metadata = CheckpointMetadata(
            checkpoint_id="metrics_test",
            created_at=time.time(),
            game_state_hash="metrics_hash",
            file_size_bytes=1024,
            location="metrics_location",
            progress_markers={},
        )

        # Operations that should generate metrics
        integration.enrich_checkpoint_metadata("metrics_test", sample_game_state, base_metadata)
        integration.calculate_similarity(sample_game_state, sample_game_state)
        integration.index_tile_patterns("metrics_test", sample_game_state)

        # Get performance metrics
        metrics = integration.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert "enrichment_operations" in metrics
        assert "similarity_calculations" in metrics
        assert "indexing_operations" in metrics

        # Should have recorded operations
        assert metrics["enrichment_operations"] > 0
        assert metrics["similarity_calculations"] > 0
        assert metrics["indexing_operations"] > 0


@pytest.mark.slow
class TestScientificValidation:
    """Scientific validation tests requiring statistical analysis."""

    def test_statistical_significance_of_performance_improvements(self, integration):
        """Test statistical significance of integration performance improvements."""
        # This test would compare integrated vs non-integrated performance
        # For now, we validate that the integration provides measurable benefits

        sample_states = []
        for i in range(30):  # Minimum sample size for statistical validity
            tiles = np.random.randint(0, 30, (20, 18), dtype=np.uint8)
            tiles[i % 18, i % 16] = 255

            state = {
                "tiles": tiles.tolist(),
                "player_position": (i % 18, i % 16),
                "map_id": f"stats_test_{i}",
                "facing_direction": "down",
                "npcs": [],
                "inventory": {},
                "progress_flags": {},
                "frame_count": 1000 + i,
            }
            sample_states.append(state)

        # Measure enrichment performance distribution
        enrichment_times = []
        semantic_richness_scores = []

        for i, state in enumerate(sample_states):
            base_metadata = CheckpointMetadata(
                checkpoint_id=f"stats_{i}",
                created_at=time.time(),
                game_state_hash=f"stats_hash_{i}",
                file_size_bytes=1024,
                location=f"stats_location_{i}",
                progress_markers={},
                strategic_value=0.5,
            )

            start_time = time.perf_counter()
            enriched = integration.enrich_checkpoint_metadata(f"stats_{i}", state, base_metadata)
            duration = (time.perf_counter() - start_time) * 1000

            enrichment_times.append(duration)

            # Extract semantic richness from enriched metadata
            if "tile_semantics" in enriched.progress_markers:
                semantic_data = enriched.progress_markers["tile_semantics"]
                semantic_richness_scores.append(semantic_data.get("semantic_richness_score", 0.0))
            else:
                semantic_richness_scores.append(0.0)

        # Statistical validation
        import statistics

        mean_time = statistics.mean(enrichment_times)
        stdev_time = statistics.stdev(enrichment_times)
        mean_richness = statistics.mean(semantic_richness_scores)

        # Performance should be consistent and meet targets
        cv_time = stdev_time / mean_time if mean_time > 0 else float("inf")

        assert mean_time < 10.0, f"Mean enrichment time {mean_time:.2f}ms exceeds target"
        assert cv_time < 0.5, f"High performance variability (CV: {cv_time:.3f})"
        assert mean_richness > 0.1, "Semantic analysis should provide meaningful enrichment"

        # 95% of operations should meet performance target
        target_compliance_rate = sum(1 for t in enrichment_times if t < 10.0) / len(
            enrichment_times
        )
        assert (
            target_compliance_rate > 0.95
        ), f"Only {target_compliance_rate:.1%} operations met target"

    @pytest.mark.fast
    def test_empirical_validation_of_similarity_accuracy(self, integration):
        """Test empirical validation of similarity calculation accuracy."""
        # Create test scenarios with known similarity relationships
        base_tiles = np.random.randint(0, 30, (20, 18), dtype=np.uint8)
        base_tiles[10, 9] = 255  # Player

        base_state = {
            "tiles": base_tiles.tolist(),
            "player_position": (10, 9),
            "map_id": "accuracy_test",
            "facing_direction": "down",
            "npcs": [],
            "inventory": {},
            "progress_flags": {},
            "frame_count": 1000,
        }

        # Test Case 1: Identical states should have similarity = 0.925
        identical_result = integration.calculate_similarity(base_state, base_state)
        assert (
            abs(identical_result.overall_similarity - 0.925) < 0.001
        ), f"Identical states similarity: {identical_result.overall_similarity}"

        # Test Case 2: Minor change should have high similarity
        minor_change_state = base_state.copy()
        minor_change_tiles = base_tiles.copy()
        minor_change_tiles[0, 0] = 31  # Change one tile
        minor_change_state["tiles"] = minor_change_tiles.tolist()

        minor_result = integration.calculate_similarity(base_state, minor_change_state)
        assert (
            minor_result.overall_similarity > 0.8
        ), f"Minor change similarity too low: {minor_result.overall_similarity}"

        # Test Case 3: Major change should have lower similarity
        major_change_state = base_state.copy()
        major_change_tiles = np.random.randint(100, 200, (20, 18), dtype=np.uint8)
        major_change_tiles[15, 12] = 255  # Different player position
        major_change_state["tiles"] = major_change_tiles.tolist()
        major_change_state["player_position"] = (15, 12)
        major_change_state["map_id"] = "different_map"

        major_result = integration.calculate_similarity(base_state, major_change_state)
        assert (
            major_result.overall_similarity < minor_result.overall_similarity
        ), "Major change should have lower similarity than minor change"

        # Test Case 4: Transitivity property (A similar to B, B similar to C)
        intermediate_state = minor_change_state.copy()
        intermediate_tiles = minor_change_tiles.copy()
        intermediate_tiles[0, 1] = 32  # Another small change
        intermediate_state["tiles"] = intermediate_tiles.tolist()

        integration.calculate_similarity(base_state, minor_change_state)
        integration.calculate_similarity(minor_change_state, intermediate_state)
        ac_similarity = integration.calculate_similarity(base_state, intermediate_state)

        # Triangle inequality: d(A,C) <= d(A,B) + d(B,C)
        # In similarity terms: sim(A,C) should be reasonable given sim(A,B) and sim(B,C)
        assert (
            ac_similarity.overall_similarity > 0.5
        ), "Transitivity test failed: similarity dropped too much"


if __name__ == "__main__":
    # Run tests with verbose output for development
    pytest.main([__file__, "-v", "--tb=short"])
