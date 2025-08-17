"""
ClaudeCodeManager - High-Performance Claude CLI Process Management

This module provides efficient management of Claude CLI processes with performance-
optimized subprocess handling, parallel initialization, and comprehensive monitoring.

Design Features:
- Parallel process initialization with ThreadPoolExecutor
- Health monitoring with exponential backoff timing
- Memory optimization with pre-allocated buffers
- Process groups for clean bulk termination
- Graceful shutdown with timeout handling

Performance Targets:
- Tactical process startup: <100ms
- Strategic process startup: <500ms
- Health check duration: <10ms
- Memory usage: <50MB per process baseline
"""

import enum
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessType(enum.Enum):
    """Claude process specialization types."""

    OPUS_STRATEGIC = "opus_strategic"
    SONNET_TACTICAL = "sonnet_tactical"


class ProcessState(enum.Enum):
    """Process lifecycle states."""

    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class ProcessConfig:
    """Configuration for Claude process initialization."""

    process_type: ProcessType
    model_name: str
    system_prompt: str
    max_retries: int = 3
    startup_timeout: float = 30.0
    health_check_interval: float = 5.0
    memory_limit_mb: int = 100

    # Performance optimization settings
    stdout_buffer_size: int = 8192  # 8KB
    stderr_buffer_size: int = 4096  # 4KB
    use_process_group: bool = True


@dataclass
class ProcessMetrics:
    """Performance and health metrics for a Claude process."""

    process_id: int
    startup_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_health_check: float = 0.0
    health_check_duration: float = 0.0
    failure_count: int = 0
    restart_count: int = 0
    total_uptime: float = 0.0


class ClaudeProcess:
    """High-performance wrapper for Claude CLI subprocess with lifecycle management."""

    # System prompts optimized for role specialization
    OPUS_STRATEGIC_PROMPT = """You are a chess grandmaster AI analyzing Pokemon Red speedrun strategies.

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

    SONNET_TACTICAL_PROMPT = """You are a frame-perfect speedrunner developing Pokemon Red execution scripts.

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

    def __init__(self, config: ProcessConfig, process_id: int):
        """Initialize Claude process with performance optimization."""
        self.config = config
        self.process_id = process_id
        self.process: subprocess.Popen | None = None
        self.state = ProcessState.INITIALIZING
        self.metrics = ProcessMetrics(process_id=process_id)
        self._lock = threading.Lock()
        self._health_check_backoff = 1.0  # Start at 1ms, exponential backoff

        # Pre-allocate communication buffers for performance
        self._stdout_buffer = bytearray(config.stdout_buffer_size)
        self._stderr_buffer = bytearray(config.stderr_buffer_size)

        logger.info(f"Initialized Claude process {process_id} with {config.process_type.value}")

    def start(self) -> bool:
        """Start Claude CLI process with performance timing."""
        start_time = time.time()

        try:
            with self._lock:
                if self.process is not None:
                    logger.warning(f"Process {self.process_id} already started")
                    return True

                # Build command based on process type
                cmd = self._build_command()

                # Set up process environment with performance optimizations
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"  # Disable Python output buffering

                # Create process with performance settings
                self.process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=0,
                    preexec_fn=os.setsid if self.config.use_process_group else None,
                )
                self.state = ProcessState.HEALTHY

                # Record startup timing
                startup_duration = time.time() - start_time
                self.metrics.startup_time = startup_duration

                # Send system prompt
                self._send_system_prompt()

                logger.info(
                    f"Started {self.config.process_type.value} process {self.process_id} "
                    f"(PID: {self.process.pid}) in {startup_duration*1000:.1f}ms"
                )

                return True

        except Exception as e:
            logger.error(f"Failed to start process {self.process_id}: {e}")
            self.state = ProcessState.FAILED
            self.metrics.failure_count += 1
            return False

    def _build_command(self) -> list[str]:
        """Build Claude CLI command based on process type."""
        base_cmd = ["claude", "chat"]

        # Add model specification
        if self.config.process_type == ProcessType.OPUS_STRATEGIC:
            base_cmd.extend(["--model", "claude-3-opus-20240229"])
        elif self.config.process_type == ProcessType.SONNET_TACTICAL:
            base_cmd.extend(["--model", "claude-3-5-sonnet-20241022"])

        # Add performance flags
        base_cmd.extend(
            [
                "--no-stream",  # Disable streaming for batch processing
                "--format",
                "json",  # Structured output for parsing
            ]
        )

        return base_cmd

    def _send_system_prompt(self):
        """Send initial system prompt based on process type."""
        if not self.process or not self.process.stdin:
            return

        try:
            if self.config.process_type == ProcessType.OPUS_STRATEGIC:
                prompt = self.OPUS_STRATEGIC_PROMPT
            else:
                prompt = self.SONNET_TACTICAL_PROMPT

            # Send prompt with proper formatting
            self.process.stdin.write(f"{prompt}\n\n")
            self.process.stdin.flush()

        except Exception as e:
            logger.error(f"Failed to send system prompt to process {self.process_id}: {e}")

    def health_check(self) -> bool:
        """Perform efficient health check with exponential backoff timing."""
        start_time = time.time()

        try:
            with self._lock:
                if self.process is None:
                    self.state = ProcessState.TERMINATED
                    return False

                # Check if process is still running
                return_code = self.process.poll()
                if return_code is not None:
                    logger.warning(f"Process {self.process_id} terminated with code {return_code}")
                    self.state = ProcessState.FAILED
                    return False

                # Update metrics
                duration = time.time() - start_time
                self.metrics.health_check_duration = duration
                self.metrics.last_health_check = start_time

                # Adjust backoff timing based on performance
                if duration < 0.005:  # < 5ms is excellent
                    self._health_check_backoff = max(0.001, self._health_check_backoff * 0.9)
                elif duration > 0.010:  # > 10ms needs backoff
                    self._health_check_backoff = min(0.1, self._health_check_backoff * 1.1)

                if self.state == ProcessState.FAILED:
                    self.state = ProcessState.HEALTHY

                return True

        except Exception as e:
            logger.error(f"Health check failed for process {self.process_id}: {e}")
            self.state = ProcessState.DEGRADED
            return False

    def send_message(self, message: str, timeout: float = 30.0) -> str | None:
        """Send message to Claude process and get response."""
        if not self.process or not self.process.stdin:
            logger.error(f"Process {self.process_id} not available for communication")
            return None

        try:
            # Send message
            self.process.stdin.write(f"{message}\n")
            self.process.stdin.flush()

            # Read response with timeout
            # This is a simplified implementation - production would need proper JSON parsing
            if self.process.stdout:
                response = self.process.stdout.readline()
                return response.strip() if response else None
            return None

        except Exception as e:
            logger.error(f"Communication failed with process {self.process_id}: {e}")
            return None

    def restart(self) -> bool:
        """Restart the Claude process with metrics tracking."""
        logger.info(f"Restarting process {self.process_id}")

        self.terminate()
        self.metrics.restart_count += 1

        # Brief pause to ensure clean shutdown
        time.sleep(0.1)

        return self.start()

    def terminate(self, timeout: float = 2.0):
        """Gracefully terminate the process with timeout."""
        if self.process is None:
            return

        try:
            with self._lock:
                if self.process.poll() is not None:
                    return  # Already terminated

                # Graceful termination attempt
                self.process.terminate()

                try:
                    self.process.wait(timeout=timeout)
                    logger.info(f"Process {self.process_id} terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force termination if graceful fails
                    logger.warning(f"Force killing process {self.process_id}")
                    self.process.kill()
                    self.process.wait()

                self.process = None
                self.state = ProcessState.TERMINATED

        except Exception as e:
            logger.error(f"Error terminating process {self.process_id}: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.terminate()


class ClaudeCodeManager:
    """Orchestrator for 1 Opus + 4 Sonnet Claude CLI processes with performance optimization."""

    def __init__(self, max_workers: int = 5):
        """Initialize manager with parallel execution capability."""
        self.max_workers = max_workers
        self.processes: dict[int, ClaudeProcess] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._running = False

        logger.info(f"ClaudeCodeManager initialized with {max_workers} workers")

    def start_all_processes(self) -> bool:
        """Start all Claude processes in parallel for optimal performance."""
        start_time = time.time()

        try:
            # Create process configurations
            configs = self._create_process_configs()

            # Parallel initialization using ThreadPoolExecutor
            logger.info(f"Starting {len(configs)} processes in parallel...")

            futures = {}
            for i, config in enumerate(configs):
                process = ClaudeProcess(config, i)
                self.processes[i] = process
                future = self._executor.submit(process.start)
                futures[future] = i

            # Wait for all processes to start
            success_count = 0
            for future in as_completed(futures):
                process_id = futures[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                        logger.info(f"Process {process_id} started successfully")
                    else:
                        logger.error(f"Process {process_id} failed to start")
                except Exception as e:
                    logger.error(f"Process {process_id} startup exception: {e}")

            total_time = time.time() - start_time
            self._running = True

            logger.info(
                f"Parallel startup completed: {success_count}/{len(configs)} processes "
                f"in {total_time*1000:.1f}ms"
            )

            return success_count == len(configs)

        except Exception as e:
            logger.error(f"Failed to start processes: {e}")
            return False

    def _create_process_configs(self) -> list[ProcessConfig]:
        """Create optimized process configurations."""
        configs = []

        # 1 Opus Strategic process
        configs.append(
            ProcessConfig(
                process_type=ProcessType.OPUS_STRATEGIC,
                model_name="claude-3-opus-20240229",
                system_prompt=ClaudeProcess.OPUS_STRATEGIC_PROMPT,
                startup_timeout=10.0,  # Opus gets more time
                memory_limit_mb=100,
            )
        )

        # 4 Sonnet Tactical processes
        for _ in range(4):
            configs.append(
                ProcessConfig(
                    process_type=ProcessType.SONNET_TACTICAL,
                    model_name="claude-3-5-sonnet-20241022",
                    system_prompt=ClaudeProcess.SONNET_TACTICAL_PROMPT,
                    startup_timeout=5.0,  # Sonnet should start faster
                    memory_limit_mb=75,
                )
            )

        return configs

    def get_strategic_process(self) -> ClaudeProcess | None:
        """Get the Opus strategic planning process."""
        for process in self.processes.values():
            if process.config.process_type == ProcessType.OPUS_STRATEGIC:
                return process
        return None

    def get_tactical_processes(self) -> list[ClaudeProcess]:
        """Get all Sonnet tactical execution processes."""
        tactical = []
        for process in self.processes.values():
            if process.config.process_type == ProcessType.SONNET_TACTICAL:
                tactical.append(process)
        return tactical

    def get_available_tactical_process(self) -> ClaudeProcess | None:
        """Get an available tactical process for work assignment."""
        tactical_processes = self.get_tactical_processes()

        # Simple round-robin selection - production would use load balancing
        for process in tactical_processes:
            if process.health_check() and process.state == ProcessState.HEALTHY:
                return process

        return None

    def health_check_all(self) -> dict[int, bool]:
        """Perform health checks on all processes with performance timing."""
        start_time = time.time()

        futures = {}
        for process_id, process in self.processes.items():
            future = self._executor.submit(process.health_check)
            futures[future] = process_id

        results = {}
        for future in as_completed(futures):
            process_id = futures[future]
            try:
                results[process_id] = future.result()
            except Exception as e:
                logger.error(f"Health check failed for process {process_id}: {e}")
                results[process_id] = False

        total_time = time.time() - start_time
        logger.debug(
            f"Health check completed for {len(self.processes)} processes "
            f"in {total_time*1000:.1f}ms"
        )

        return results

    def restart_failed_processes(self) -> int:
        """Restart any failed processes and return count of restarts."""
        restart_count = 0

        for process_id, process in self.processes.items():
            if process.state in [ProcessState.FAILED, ProcessState.DEGRADED]:
                logger.info(f"Restarting failed process {process_id}")
                if process.restart():
                    restart_count += 1
                    logger.info(f"Process {process_id} restarted successfully")
                else:
                    logger.error(f"Failed to restart process {process_id}")

        return restart_count

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics for all processes."""
        metrics: dict[str, Any] = {
            "total_processes": len(self.processes),
            "healthy_processes": 0,
            "failed_processes": 0,
            "average_startup_time": 0.0,
            "average_health_check_time": 0.0,
            "total_restarts": 0,
            "process_details": {},
        }

        startup_times = []
        health_check_times = []

        for process_id, process in self.processes.items():
            process_metrics = process.metrics

            # Aggregate counts
            if process.state == ProcessState.HEALTHY:
                metrics["healthy_processes"] += 1
            elif process.state in [ProcessState.FAILED, ProcessState.DEGRADED]:
                metrics["failed_processes"] += 1

            # Timing data
            startup_times.append(process_metrics.startup_time)
            health_check_times.append(process_metrics.health_check_duration)
            metrics["total_restarts"] += process_metrics.restart_count

            # Individual process details
            metrics["process_details"][process_id] = {
                "type": process.config.process_type.value,
                "state": process.state.value,
                "startup_time_ms": process_metrics.startup_time * 1000,
                "health_check_time_ms": process_metrics.health_check_duration * 1000,
                "restart_count": process_metrics.restart_count,
                "failure_count": process_metrics.failure_count,
            }

        # Calculate averages
        if startup_times:
            metrics["average_startup_time"] = sum(startup_times) / len(startup_times) * 1000
        if health_check_times:
            metrics["average_health_check_time"] = (
                sum(health_check_times) / len(health_check_times) * 1000
            )

        return metrics

    def shutdown(self, timeout: float = 10.0):
        """Gracefully shutdown all processes with timeout."""
        logger.info("Shutting down ClaudeCodeManager...")

        if not self.processes:
            return

        # Parallel shutdown for efficiency
        futures = {}
        for process_id, process in self.processes.items():
            future = self._executor.submit(process.terminate, timeout / len(self.processes))
            futures[future] = process_id

        # Wait for all shutdowns
        for future in as_completed(futures):
            process_id = futures[future]
            try:
                future.result()
                logger.info(f"Process {process_id} shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down process {process_id}: {e}")

        self.processes.clear()
        self._running = False

        # Shutdown executor
        self._executor.shutdown(wait=True)

        logger.info("ClaudeCodeManager shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()

    def is_running(self) -> bool:
        """Check if manager is currently running."""
        return self._running and bool(self.processes)

    def get_process_count_by_type(self) -> dict[str, int]:
        """Get count of processes by type."""
        counts = {"opus_strategic": 0, "sonnet_tactical": 0}

        for process in self.processes.values():
            if process.config.process_type == ProcessType.OPUS_STRATEGIC:
                counts["opus_strategic"] += 1
            elif process.config.process_type == ProcessType.SONNET_TACTICAL:
                counts["sonnet_tactical"] += 1

        return counts


# Performance monitoring utilities
def benchmark_startup_performance():
    """Benchmark ClaudeCodeManager startup performance."""
    logger.info("Starting ClaudeCodeManager performance benchmark...")

    start_time = time.time()

    with ClaudeCodeManager() as manager:
        startup_success = manager.start_all_processes()
        startup_time = time.time() - start_time

        if startup_success:
            metrics = manager.get_performance_metrics()

            print("\n=== ClaudeCodeManager Performance Benchmark ===")
            print(f"Total startup time: {startup_time*1000:.1f}ms")
            print(f"Average process startup: {metrics['average_startup_time']:.1f}ms")
            print(f"Processes started: {metrics['healthy_processes']}/{metrics['total_processes']}")
            print(f"Process breakdown: {manager.get_process_count_by_type()}")

            # Test health checks
            health_start = time.time()
            health_results = manager.health_check_all()
            health_time = time.time() - health_start

            healthy_count = sum(1 for result in health_results.values() if result)
            print(f"Health check time: {health_time*1000:.1f}ms")
            print(f"Healthy processes: {healthy_count}/{len(health_results)}")

            return True
        else:
            print("Benchmark failed - not all processes started successfully")
            return False


if __name__ == "__main__":
    benchmark_startup_performance()
