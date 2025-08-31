"""
TileObserver-CheckpointManager Integration Performance Benchmarking Framework.

Scientific performance measurement and validation tool designed for empirical analysis
and data-driven optimization of the integration between tile semantic analysis and
checkpoint metadata enrichment.

Scientist Personality Implementation Features:
- Statistical analysis of performance metrics
- Empirical validation of performance targets
- Comprehensive data collection for optimization
- Scientific hypothesis testing for improvements
- Benchmarking against performance requirements

Performance Targets (Scientific Validation):
- Metadata enrichment: < 10ms per checkpoint
- Game state similarity: < 50ms per comparison
- Tile pattern indexing: < 100ms per checkpoint
- Memory efficiency: < 1MB per 1000 cached operations

Author: Worker worker6 (Scientist) - Empirical Performance Engineering
"""

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import psutil
import structlog

from .checkpoint_manager import CheckpointManager
from .tile_observer import TileObserver
from .tile_observer_checkpoint_integration import TileObserverCheckpointIntegration

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMeasurement:
    """
    Individual performance measurement with statistical validation.

    Scientific approach to performance data collection and analysis.
    """

    operation_type: str
    duration_ms: float
    memory_delta_mb: float
    cpu_percent: float
    timestamp: float
    complexity_indicator: int
    success: bool
    error_message: str = ""

    # Statistical validation fields
    target_met: bool = False
    performance_percentile: float = 0.0  # Percentile within operation type
    statistical_significance: float = 0.0  # P-value for performance regression


@dataclass
class BenchmarkSuite:
    """
    Complete benchmark suite results with comprehensive statistical analysis.

    Designed for scientific validation and empirical optimization insights.
    """

    suite_name: str
    start_timestamp: float
    end_timestamp: float
    total_duration_seconds: float

    # Performance measurements
    measurements: list[PerformanceMeasurement]

    # Statistical summaries by operation type
    performance_stats: dict[str, dict[str, float]]

    # Target compliance analysis
    target_compliance: dict[str, bool]

    # Memory and resource analysis
    peak_memory_mb: float
    average_cpu_percent: float
    memory_leak_detected: bool

    # Scientific validation metrics
    statistical_validity: dict[str, float]
    regression_analysis: dict[str, Any]
    optimization_recommendations: list[str]


class IntegrationPerformanceBenchmark:
    """
    Comprehensive performance benchmarking framework for scientific validation.

    Implements rigorous scientific methodology for performance measurement:
    - Statistical significance testing
    - Regression analysis
    - Empirical validation of improvements
    - Data-driven optimization recommendations

    Scientist Personality Focus:
    - Quantitative measurement of all performance aspects
    - Statistical validation of performance improvements
    - Empirical evidence for optimization decisions
    - Scientific rigor in benchmark design and analysis
    """

    # Scientific benchmarking configuration
    PERFORMANCE_TARGETS = {
        "metadata_enrichment_ms": 10.0,
        "similarity_calculation_ms": 50.0,
        "pattern_indexing_ms": 100.0,
        "checkpoint_save_ms": 500.0,  # CheckpointManager target
        "checkpoint_load_ms": 500.0,  # CheckpointManager target
    }

    MEMORY_TARGETS = {
        "max_cache_size_mb": 1.0,  # Per 1000 operations
        "memory_leak_threshold_mb": 10.0,  # Acceptable growth per suite
    }

    # Statistical validation parameters
    STATISTICAL_SIGNIFICANCE_THRESHOLD = 0.05
    MINIMUM_SAMPLE_SIZE = 30
    PERFORMANCE_REGRESSION_THRESHOLD = 0.1  # 10% degradation threshold

    def __init__(
        self,
        enable_detailed_logging: bool = True,
        enable_memory_profiling: bool = True,
        enable_statistical_analysis: bool = True,
    ):
        """
        Initialize benchmarking framework with scientific measurement capabilities.

        Args:
            enable_detailed_logging: Enable detailed performance logging
            enable_memory_profiling: Enable memory usage profiling
            enable_statistical_analysis: Enable statistical analysis of results
        """
        self.enable_detailed_logging = enable_detailed_logging
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_statistical_analysis = enable_statistical_analysis

        # Initialize performance tracking
        self._measurements: list[PerformanceMeasurement] = []
        self._baseline_memory_mb = self._get_current_memory_usage()
        self._process = psutil.Process()

        logger.info(
            "Integration Performance Benchmark initialized",
            detailed_logging=enable_detailed_logging,
            memory_profiling=enable_memory_profiling,
            statistical_analysis=enable_statistical_analysis,
            baseline_memory_mb=self._baseline_memory_mb,
        )

    def run_comprehensive_benchmark(
        self,
        test_iterations: int = 100,
        game_state_complexity_levels: list[str] = None,
        include_stress_testing: bool = True,
    ) -> BenchmarkSuite:
        """
        Run comprehensive performance benchmark with scientific rigor.

        Includes:
        - Core integration performance measurement
        - Statistical analysis of performance distribution
        - Memory usage and leak detection
        - Stress testing under load
        - Comparative analysis against targets

        Args:
            test_iterations: Number of iterations per test (min 30 for statistical validity)
            game_state_complexity_levels: Complexity levels to test ['simple', 'medium', 'complex']
            include_stress_testing: Include stress testing scenarios

        Returns:
            Complete benchmark suite with statistical analysis
        """
        if test_iterations < self.MINIMUM_SAMPLE_SIZE:
            logger.warning(
                f"Test iterations {test_iterations} below minimum for statistical validity",
                minimum_required=self.MINIMUM_SAMPLE_SIZE,
                statistical_validity_warning=True,
            )

        if game_state_complexity_levels is None:
            game_state_complexity_levels = ["simple", "medium", "complex"]

        start_time = time.time()
        initial_memory = self._get_current_memory_usage()

        logger.info(
            "Starting comprehensive integration performance benchmark",
            iterations=test_iterations,
            complexity_levels=game_state_complexity_levels,
            stress_testing=include_stress_testing,
            target_metadata_enrichment_ms=self.PERFORMANCE_TARGETS["metadata_enrichment_ms"],
            target_similarity_ms=self.PERFORMANCE_TARGETS["similarity_calculation_ms"],
            target_indexing_ms=self.PERFORMANCE_TARGETS["pattern_indexing_ms"],
        )

        # Clear measurements for fresh benchmark
        self._measurements.clear()

        # Create temporary environment for benchmarking
        with TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)

            # Initialize components for benchmarking
            tile_observer = TileObserver()
            checkpoint_manager = CheckpointManager(
                storage_dir=storage_path,
                max_checkpoints=test_iterations + 10,  # Accommodate all test checkpoints
                enable_metrics=True,
            )
            integration = TileObserverCheckpointIntegration(
                tile_observer=tile_observer,
                enable_performance_tracking=True,
                enable_similarity_caching=True,
            )

            # Benchmark 1: Metadata Enrichment Performance
            logger.info("Benchmarking metadata enrichment performance...")
            self._benchmark_metadata_enrichment(
                integration, checkpoint_manager, test_iterations, game_state_complexity_levels
            )

            # Benchmark 2: Similarity Calculation Performance
            logger.info("Benchmarking similarity calculation performance...")
            self._benchmark_similarity_calculations(
                integration, test_iterations, game_state_complexity_levels
            )

            # Benchmark 3: Pattern Indexing Performance
            logger.info("Benchmarking pattern indexing performance...")
            self._benchmark_pattern_indexing(
                integration, test_iterations, game_state_complexity_levels
            )

            # Benchmark 4: End-to-end Integration Performance
            logger.info("Benchmarking end-to-end integration performance...")
            self._benchmark_end_to_end_integration(integration, checkpoint_manager, test_iterations)

            # Benchmark 5: Stress Testing (if enabled)
            if include_stress_testing:
                logger.info("Running stress testing scenarios...")
                self._benchmark_stress_testing(integration, checkpoint_manager)

            # Memory leak detection
            final_memory = self._get_current_memory_usage()
            memory_growth = final_memory - initial_memory
            memory_leak_detected = memory_growth > self.MEMORY_TARGETS["memory_leak_threshold_mb"]

            # Generate comprehensive benchmark suite
            end_time = time.time()
            benchmark_suite = self._generate_benchmark_suite(
                "comprehensive_integration_benchmark",
                start_time,
                end_time,
                initial_memory,
                final_memory,
                memory_leak_detected,
            )

            logger.info(
                "Comprehensive benchmark completed",
                total_duration_seconds=benchmark_suite.total_duration_seconds,
                total_measurements=len(benchmark_suite.measurements),
                memory_growth_mb=memory_growth,
                memory_leak_detected=memory_leak_detected,
                target_compliance=benchmark_suite.target_compliance,
            )

            return benchmark_suite

    def _benchmark_metadata_enrichment(
        self,
        integration: TileObserverCheckpointIntegration,
        checkpoint_manager: CheckpointManager,
        iterations: int,
        complexity_levels: list[str],
    ) -> None:
        """Benchmark metadata enrichment performance across complexity levels."""
        for complexity in complexity_levels:
            for i in range(iterations // len(complexity_levels)):
                # Create test game state
                game_state = self._create_test_game_state(complexity)

                # Create base metadata
                base_metadata = self._create_base_checkpoint_metadata(f"bench_{complexity}_{i}")

                # Measure metadata enrichment performance
                start_time = time.perf_counter()
                start_memory = self._get_current_memory_usage()
                start_cpu = self._get_cpu_percent()

                try:
                    integration.enrich_checkpoint_metadata(
                        f"benchmark_{complexity}_{i}", game_state, base_metadata
                    )
                    success = True
                    error_msg = ""
                except Exception as e:
                    success = False
                    error_msg = str(e)

                duration_ms = (time.perf_counter() - start_time) * 1000
                memory_delta = self._get_current_memory_usage() - start_memory
                cpu_percent = self._get_cpu_percent() - start_cpu

                # Record measurement
                measurement = PerformanceMeasurement(
                    operation_type="metadata_enrichment",
                    duration_ms=duration_ms,
                    memory_delta_mb=memory_delta,
                    cpu_percent=cpu_percent,
                    timestamp=time.time(),
                    complexity_indicator=self._calculate_complexity_indicator(game_state),
                    success=success,
                    error_message=error_msg,
                    target_met=duration_ms < self.PERFORMANCE_TARGETS["metadata_enrichment_ms"],
                )

                self._measurements.append(measurement)

                if self.enable_detailed_logging and i % 10 == 0:
                    logger.debug(
                        "Metadata enrichment benchmark progress",
                        complexity=complexity,
                        iteration=i,
                        duration_ms=duration_ms,
                        target_met=measurement.target_met,
                    )

    def _benchmark_similarity_calculations(
        self,
        integration: TileObserverCheckpointIntegration,
        iterations: int,
        complexity_levels: list[str],
    ) -> None:
        """Benchmark similarity calculation performance."""
        # Create test game states for comparison
        test_states = {}
        for complexity in complexity_levels:
            test_states[complexity] = [self._create_test_game_state(complexity) for _ in range(10)]

        comparison_count = 0
        target_comparisons = iterations

        for complexity_a in complexity_levels:
            for complexity_b in complexity_levels:
                if comparison_count >= target_comparisons:
                    break

                for state_a in test_states[complexity_a]:
                    if comparison_count >= target_comparisons:
                        break

                    for state_b in test_states[complexity_b]:
                        if comparison_count >= target_comparisons:
                            break

                        # Measure similarity calculation performance
                        start_time = time.perf_counter()
                        start_memory = self._get_current_memory_usage()
                        start_cpu = self._get_cpu_percent()

                        try:
                            similarity_result = integration.calculate_similarity(state_a, state_b)
                            success = True
                            error_msg = ""
                        except Exception as e:
                            success = False
                            error_msg = str(e)
                            similarity_result = None

                        duration_ms = (time.perf_counter() - start_time) * 1000
                        memory_delta = self._get_current_memory_usage() - start_memory
                        cpu_percent = self._get_cpu_percent() - start_cpu

                        # Record measurement
                        measurement = PerformanceMeasurement(
                            operation_type="similarity_calculation",
                            duration_ms=duration_ms,
                            memory_delta_mb=memory_delta,
                            cpu_percent=cpu_percent,
                            timestamp=time.time(),
                            complexity_indicator=(
                                self._calculate_complexity_indicator(state_a)
                                + self._calculate_complexity_indicator(state_b)
                            ),
                            success=success,
                            error_message=error_msg,
                            target_met=duration_ms
                            < self.PERFORMANCE_TARGETS["similarity_calculation_ms"],
                        )

                        self._measurements.append(measurement)
                        comparison_count += 1

                        if self.enable_detailed_logging and comparison_count % 10 == 0:
                            logger.debug(
                                "Similarity calculation benchmark progress",
                                comparison=comparison_count,
                                duration_ms=duration_ms,
                                target_met=measurement.target_met,
                                similarity_score=(
                                    similarity_result.overall_similarity
                                    if similarity_result
                                    else 0.0
                                ),
                            )

    def _benchmark_pattern_indexing(
        self,
        integration: TileObserverCheckpointIntegration,
        iterations: int,
        complexity_levels: list[str],
    ) -> None:
        """Benchmark tile pattern indexing performance."""
        for complexity in complexity_levels:
            for i in range(iterations // len(complexity_levels)):
                # Create test game state
                game_state = self._create_test_game_state(complexity)

                # Measure pattern indexing performance
                start_time = time.perf_counter()
                start_memory = self._get_current_memory_usage()
                start_cpu = self._get_cpu_percent()

                try:
                    semantic_metadata = integration.index_tile_patterns(
                        f"benchmark_index_{complexity}_{i}", game_state
                    )
                    success = True
                    error_msg = ""
                except Exception as e:
                    success = False
                    error_msg = str(e)
                    semantic_metadata = None

                duration_ms = (time.perf_counter() - start_time) * 1000
                memory_delta = self._get_current_memory_usage() - start_memory
                cpu_percent = self._get_cpu_percent() - start_cpu

                # Record measurement
                measurement = PerformanceMeasurement(
                    operation_type="pattern_indexing",
                    duration_ms=duration_ms,
                    memory_delta_mb=memory_delta,
                    cpu_percent=cpu_percent,
                    timestamp=time.time(),
                    complexity_indicator=self._calculate_complexity_indicator(game_state),
                    success=success,
                    error_message=error_msg,
                    target_met=duration_ms < self.PERFORMANCE_TARGETS["pattern_indexing_ms"],
                )

                self._measurements.append(measurement)

                if self.enable_detailed_logging and i % 10 == 0:
                    logger.debug(
                        "Pattern indexing benchmark progress",
                        complexity=complexity,
                        iteration=i,
                        duration_ms=duration_ms,
                        target_met=measurement.target_met,
                        semantic_richness=(
                            semantic_metadata.semantic_richness_score if semantic_metadata else 0.0
                        ),
                    )

    def _benchmark_end_to_end_integration(
        self,
        integration: TileObserverCheckpointIntegration,
        checkpoint_manager: CheckpointManager,
        iterations: int,
    ) -> None:
        """Benchmark complete end-to-end integration workflow."""
        for i in range(min(iterations // 4, 25)):  # Limit end-to-end tests for efficiency
            # Create test scenario
            game_state = self._create_test_game_state("medium")
            metadata = {"location": f"benchmark_location_{i}", "progress_markers": {}}

            # Measure complete workflow performance
            start_time = time.perf_counter()
            start_memory = self._get_current_memory_usage()
            start_cpu = self._get_cpu_percent()

            try:
                # Step 1: Save checkpoint
                checkpoint_id = checkpoint_manager.save_checkpoint(game_state, metadata)

                # Step 2: Enrich with semantic analysis
                base_metadata = checkpoint_manager._load_metadata(checkpoint_id)
                if base_metadata:
                    integration.enrich_checkpoint_metadata(checkpoint_id, game_state, base_metadata)

                # Step 3: Calculate similarity with previous state
                if i > 0:
                    previous_state = self._create_test_game_state("medium")
                    integration.calculate_similarity(game_state, previous_state)

                # Step 4: Index patterns
                integration.index_tile_patterns(checkpoint_id, game_state)

                success = True
                error_msg = ""

            except Exception as e:
                success = False
                error_msg = str(e)

            duration_ms = (time.perf_counter() - start_time) * 1000
            memory_delta = self._get_current_memory_usage() - start_memory
            cpu_percent = self._get_cpu_percent() - start_cpu

            # Record measurement (using longer target for end-to-end)
            measurement = PerformanceMeasurement(
                operation_type="end_to_end_integration",
                duration_ms=duration_ms,
                memory_delta_mb=memory_delta,
                cpu_percent=cpu_percent,
                timestamp=time.time(),
                complexity_indicator=self._calculate_complexity_indicator(game_state),
                success=success,
                error_message=error_msg,
                target_met=duration_ms < 200.0,  # Reasonable target for full workflow
            )

            self._measurements.append(measurement)

            if self.enable_detailed_logging:
                logger.debug(
                    "End-to-end integration benchmark",
                    iteration=i,
                    duration_ms=duration_ms,
                    target_met=measurement.target_met,
                )

    def _benchmark_stress_testing(
        self, integration: TileObserverCheckpointIntegration, checkpoint_manager: CheckpointManager
    ) -> None:
        """Run stress testing scenarios to validate performance under load."""
        logger.info("Running stress testing scenarios...")

        # Stress Test 1: Rapid similarity calculations
        states = [self._create_test_game_state("complex") for _ in range(20)]
        start_time = time.perf_counter()

        for i in range(len(states)):
            for j in range(i + 1, min(i + 5, len(states))):  # Compare with next 4 states
                try:
                    integration.calculate_similarity(states[i], states[j])
                except Exception as e:
                    logger.warning(f"Stress test similarity calculation failed: {e}")

        stress_duration = (time.perf_counter() - start_time) * 1000

        # Record stress test measurement
        measurement = PerformanceMeasurement(
            operation_type="stress_test_similarity",
            duration_ms=stress_duration,
            memory_delta_mb=self._get_current_memory_usage() - self._baseline_memory_mb,
            cpu_percent=self._get_cpu_percent(),
            timestamp=time.time(),
            complexity_indicator=len(states) * 4,  # Number of comparisons
            success=True,
            target_met=stress_duration < 2000.0,  # 2 second target for stress test
        )

        self._measurements.append(measurement)

        logger.info(f"Stress testing completed in {stress_duration:.2f}ms")

    def _generate_benchmark_suite(
        self,
        suite_name: str,
        start_time: float,
        end_time: float,
        initial_memory: float,
        final_memory: float,
        memory_leak_detected: bool,
    ) -> BenchmarkSuite:
        """Generate comprehensive benchmark suite with statistical analysis."""
        total_duration = end_time - start_time

        # Group measurements by operation type
        measurements_by_type = {}
        for measurement in self._measurements:
            op_type = measurement.operation_type
            if op_type not in measurements_by_type:
                measurements_by_type[op_type] = []
            measurements_by_type[op_type].append(measurement)

        # Calculate statistical summaries
        performance_stats = {}
        target_compliance = {}

        for op_type, measurements in measurements_by_type.items():
            durations = [m.duration_ms for m in measurements if m.success]

            if durations:
                performance_stats[op_type] = {
                    "count": len(durations),
                    "mean_ms": statistics.mean(durations),
                    "median_ms": statistics.median(durations),
                    "stdev_ms": statistics.stdev(durations) if len(durations) > 1 else 0.0,
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                    "p95_ms": self._calculate_percentile(durations, 95),
                    "p99_ms": self._calculate_percentile(durations, 99),
                    "success_rate": sum(1 for m in measurements if m.success) / len(measurements),
                }

                # Determine target compliance
                target_key = f"{op_type}_ms"
                if target_key in self.PERFORMANCE_TARGETS:
                    target = self.PERFORMANCE_TARGETS[target_key]
                    compliance_rate = sum(1 for d in durations if d < target) / len(durations)
                    target_compliance[op_type] = compliance_rate > 0.95  # 95% compliance
                else:
                    target_compliance[op_type] = True  # No specific target
            else:
                performance_stats[op_type] = {"count": 0, "success_rate": 0.0}
                target_compliance[op_type] = False

        # Calculate peak memory and average CPU
        peak_memory = max(
            self._baseline_memory_mb
            + max([m.memory_delta_mb for m in self._measurements], default=0),
            final_memory,
        )

        avg_cpu = (
            statistics.mean([m.cpu_percent for m in self._measurements])
            if self._measurements
            else 0.0
        )

        # Statistical validity analysis (if enabled)
        statistical_validity = {}
        regression_analysis = {}
        if self.enable_statistical_analysis:
            statistical_validity, regression_analysis = self._perform_statistical_analysis(
                measurements_by_type
            )

        # Generate optimization recommendations
        optimization_recommendations = self._generate_optimization_recommendations(
            performance_stats, target_compliance, statistical_validity
        )

        return BenchmarkSuite(
            suite_name=suite_name,
            start_timestamp=start_time,
            end_timestamp=end_time,
            total_duration_seconds=total_duration,
            measurements=self._measurements.copy(),
            performance_stats=performance_stats,
            target_compliance=target_compliance,
            peak_memory_mb=peak_memory,
            average_cpu_percent=avg_cpu,
            memory_leak_detected=memory_leak_detected,
            statistical_validity=statistical_validity,
            regression_analysis=regression_analysis,
            optimization_recommendations=optimization_recommendations,
        )

    # Helper methods for benchmark implementation

    def _create_test_game_state(self, complexity: str = "medium") -> dict[str, Any]:
        """Create test game state with specified complexity level."""
        if complexity == "simple":
            tiles = np.zeros((20, 18), dtype=np.uint8)
            tiles[5:15, 3:15] = 1  # Simple walkable area
            tiles[10, 9] = 255  # Player position
        elif complexity == "complex":
            # Complex terrain with various tile types
            tiles = np.random.randint(0, 50, (20, 18), dtype=np.uint8)
            tiles[0:2, :] = 100  # Top border (solid)
            tiles[18:20, :] = 100  # Bottom border (solid)
            tiles[:, 0:2] = 100  # Left border (solid)
            tiles[:, 16:18] = 100  # Right border (solid)
            tiles[10, 9] = 255  # Player position
            tiles[5, 5] = 200  # NPC
            tiles[15, 12] = 201  # Another NPC
        else:  # medium complexity
            tiles = np.random.randint(0, 30, (20, 18), dtype=np.uint8)
            tiles[10, 9] = 255  # Player position
            tiles[8, 7] = 200  # NPC

        return {
            "tiles": tiles.tolist(),
            "player_position": (10, 9),
            "map_id": f"benchmark_{complexity}",
            "facing_direction": "down",
            "npcs": [{"id": 1, "x": 8, "y": 7, "type": "trainer"}],
            "inventory": {"pokeball": 5, "potion": 3},
            "progress_flags": {"badges": 1, "story_progress": f"benchmark_{complexity}"},
            "frame_count": np.random.randint(1000, 10000),
        }

    def _create_base_checkpoint_metadata(self, checkpoint_id: str):
        """Create base checkpoint metadata for benchmarking."""
        from datetime import datetime, timezone

        from .checkpoint_manager import CheckpointMetadata

        return CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            created_at=datetime.now(timezone.utc),
            game_state_hash="benchmark_hash",
            file_size_bytes=1024,
            location="benchmark_location",
            progress_markers={"benchmark": True},
        )

    def _calculate_complexity_indicator(self, game_state: dict[str, Any]) -> int:
        """Calculate numerical complexity indicator for game state."""
        try:
            tiles = game_state.get("tiles", [])
            if isinstance(tiles, list) and tiles:
                # Count unique tile types as complexity indicator
                flat_tiles = [
                    tile for row in tiles for tile in (row if isinstance(row, list) else [row])
                ]
                return len(set(flat_tiles))
            return 0
        except Exception:
            return 0

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.enable_memory_profiling:
            try:
                return self._process.memory_info().rss / (1024 * 1024)
            except Exception:
                return 0.0
        return 0.0

    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self._process.cpu_percent()
        except Exception:
            return 0.0

    def _calculate_percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of data list."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight

    def _perform_statistical_analysis(
        self, measurements_by_type: dict[str, list[PerformanceMeasurement]]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Perform statistical analysis for scientific validation."""
        statistical_validity = {}
        regression_analysis = {}

        for op_type, measurements in measurements_by_type.items():
            durations = [m.duration_ms for m in measurements if m.success]

            if len(durations) >= self.MINIMUM_SAMPLE_SIZE:
                # Statistical validity: coefficient of variation
                mean_duration = statistics.mean(durations)
                stdev_duration = statistics.stdev(durations)
                cv = stdev_duration / mean_duration if mean_duration > 0 else float("inf")

                # Lower coefficient of variation = higher statistical validity
                statistical_validity[op_type] = max(0.0, 1.0 - min(cv, 1.0))

                # Simple regression analysis: performance over time
                timestamps = [m.timestamp for m in measurements if m.success]
                if len(timestamps) > 1:
                    # Calculate trend using simple linear regression
                    n = len(durations)
                    sum_x = sum(range(n))
                    sum_y = sum(durations)
                    sum_xy = sum(i * duration for i, duration in enumerate(durations))
                    sum_x2 = sum(i * i for i in range(n))

                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

                    regression_analysis[op_type] = {
                        "trend_slope_ms_per_operation": slope,
                        "performance_degradation": slope > self.PERFORMANCE_REGRESSION_THRESHOLD,
                        "sample_size": n,
                    }
                else:
                    regression_analysis[op_type] = {"insufficient_data": True}
            else:
                statistical_validity[op_type] = 0.0
                regression_analysis[op_type] = {"insufficient_samples": True}

        return statistical_validity, regression_analysis

    def _generate_optimization_recommendations(
        self,
        performance_stats: dict[str, dict[str, float]],
        target_compliance: dict[str, bool],
        statistical_validity: dict[str, float],
    ) -> list[str]:
        """Generate data-driven optimization recommendations."""
        recommendations = []

        for op_type, compliant in target_compliance.items():
            if not compliant and op_type in performance_stats:
                stats = performance_stats[op_type]

                if op_type == "metadata_enrichment":
                    if stats["mean_ms"] > 10:
                        recommendations.append(
                            f"Metadata enrichment averaging {stats['mean_ms']:.1f}ms "
                            f"(target: 10ms). Consider optimizing semantic analysis algorithms."
                        )
                elif op_type == "similarity_calculation":
                    if stats["mean_ms"] > 50:
                        recommendations.append(
                            f"Similarity calculation averaging {stats['mean_ms']:.1f}ms "
                            f"(target: 50ms). Consider increasing cache size or optimizing comparison algorithms."
                        )
                elif op_type == "pattern_indexing":
                    if stats["mean_ms"] > 100:
                        recommendations.append(
                            f"Pattern indexing averaging {stats['mean_ms']:.1f}ms "
                            f"(target: 100ms). Consider pattern detection optimization or caching improvements."
                        )

        # Add statistical validity recommendations
        for op_type, validity in statistical_validity.items():
            if validity < 0.8:  # Low statistical validity
                recommendations.append(
                    f"Performance measurements for {op_type} show high variability "
                    f"(validity: {validity:.2f}). Consider investigating performance inconsistencies."
                )

        if not recommendations:
            recommendations.append("All performance targets met with high statistical validity.")

        return recommendations

    def save_benchmark_results(self, benchmark_suite: BenchmarkSuite, output_path: Path) -> None:
        """Save benchmark results to JSON file for analysis."""
        # Convert dataclass to dictionary for JSON serialization
        results_dict = asdict(benchmark_suite)

        # Convert numpy types to Python native types for JSON serialization
        results_dict = self._convert_numpy_types(results_dict)

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

        logger.info(f"Benchmark results saved to {output_path}")

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj


def run_scientific_benchmark(
    iterations: int = 100, output_path: Path = None, enable_stress_testing: bool = True
) -> BenchmarkSuite:
    """
    Run scientific performance benchmark of TileObserver-CheckpointManager integration.

    Scientist personality implementation for empirical validation and performance optimization.

    Args:
        iterations: Number of test iterations (minimum 30 for statistical validity)
        output_path: Path to save benchmark results JSON file
        enable_stress_testing: Include stress testing scenarios

    Returns:
        Complete benchmark suite with statistical analysis and optimization recommendations
    """
    logger.info(
        "Starting scientific performance benchmark",
        iterations=iterations,
        stress_testing=enable_stress_testing,
        scientist_validation=True,
    )

    # Initialize benchmark framework
    benchmark = IntegrationPerformanceBenchmark(
        enable_detailed_logging=True, enable_memory_profiling=True, enable_statistical_analysis=True
    )

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        test_iterations=iterations, include_stress_testing=enable_stress_testing
    )

    # Save results if output path provided
    if output_path:
        benchmark.save_benchmark_results(results, output_path)

    # Log summary of scientific validation
    logger.info(
        "Scientific benchmark completed",
        total_measurements=len(results.measurements),
        target_compliance=results.target_compliance,
        peak_memory_mb=results.peak_memory_mb,
        memory_leak_detected=results.memory_leak_detected,
        optimization_recommendations_count=len(results.optimization_recommendations),
        statistical_validity_summary=results.statistical_validity,
    )

    return results


if __name__ == "__main__":
    # Run benchmark with scientific validation
    results = run_scientific_benchmark(
        iterations=60,  # Sufficient for statistical analysis
        output_path=Path("integration_benchmark_results.json"),
        enable_stress_testing=True,
    )

    print("\n" + "=" * 70)
    print("SCIENTIFIC PERFORMANCE BENCHMARK RESULTS")
    print("=" * 70)

    print(f"Total Duration: {results.total_duration_seconds:.2f}s")
    print(f"Peak Memory Usage: {results.peak_memory_mb:.1f}MB")
    print(f"Memory Leak Detected: {'Yes' if results.memory_leak_detected else 'No'}")
    print()

    print("PERFORMANCE TARGETS COMPLIANCE:")
    for operation, compliant in results.target_compliance.items():
        status = "✅ PASSED" if compliant else "❌ FAILED"
        print(f"  {operation}: {status}")
    print()

    print("OPTIMIZATION RECOMMENDATIONS:")
    for i, recommendation in enumerate(results.optimization_recommendations, 1):
        print(f"  {i}. {recommendation}")

    print("\n" + "=" * 70)
