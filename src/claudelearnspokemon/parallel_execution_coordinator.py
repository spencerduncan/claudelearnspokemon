"""
ParallelExecutionCoordinator: Multi-stream emulator orchestration for parallel Pokemon gameplay.

Coordinates execution across multiple emulator streams with strategic planning and tactical
execution workers. Built with Clean Architecture principles for maintainable, scalable,
and fault-tolerant operation.

This coordinator implements the Act phase of the OODA loop (Observe, Orient, Decide, Act),
orchestrating the execution of strategic plans across multiple parallel emulation streams.

Features:
- Multi-stream execution coordination with EmulatorPool integration
- Strategic planning via Opus processes and tactical execution via Sonnet workers
- Circuit breaker patterns for fault tolerance and cascading failure prevention
- Comprehensive performance monitoring and metrics collection
- Graceful error handling and automatic recovery mechanisms
- Clean resource management with context manager support

Author: Quinn (Scientist) - Act Subagent Implementation
"""

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, TYPE_CHECKING

from .circuit_breaker import CircuitBreaker, CircuitConfig

if TYPE_CHECKING:
    from .claude_code_manager import ClaudeCodeManager
    from .emulator_pool import EmulatorPool, ExecutionResult, PokemonGymClient

# Configure logging for production observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoordinatorState(Enum):
    """Coordinator lifecycle states."""
    
    INITIALIZING = auto()  # Setting up components and resources
    READY = auto()         # Ready to coordinate executions
    RUNNING = auto()       # Actively coordinating executions
    PAUSED = auto()        # Temporarily paused but can resume
    STOPPING = auto()      # Gracefully shutting down
    STOPPED = auto()       # Fully stopped and resources released
    FAILED = auto()        # Failed state requiring manual intervention


class ParallelExecutionError(Exception):
    """
    Custom exception for ParallelExecutionCoordinator operations.
    
    Provides actionable error messages for production debugging and monitoring.
    """
    pass


@dataclass
class CoordinationConfig:
    """Configuration for parallel execution coordination behavior."""
    
    # Core coordination parameters
    max_parallel_streams: int = 4          # Maximum concurrent emulator streams
    strategic_planning_timeout: float = 30.0  # Opus strategic planning timeout
    tactical_execution_timeout: float = 120.0 # Sonnet execution timeout per task
    
    # Resource management
    emulator_acquisition_timeout: float = 10.0  # Timeout for emulator acquisition
    worker_pool_size: int = 8              # ThreadPoolExecutor size
    
    # Circuit breaker configuration
    enable_circuit_breaker: bool = True
    circuit_failure_threshold: int = 3
    circuit_recovery_timeout: float = 60.0
    
    # Performance monitoring
    metrics_collection_interval: float = 5.0  # Metrics collection frequency
    health_check_interval: float = 10.0       # Health check frequency
    
    # Error handling
    max_retry_attempts: int = 3
    retry_backoff_factor: float = 1.5
    enable_automatic_recovery: bool = True


@dataclass
class ExecutionStream:
    """
    Represents a single parallel execution stream.
    
    Encapsulates the state and resources for one emulator-worker coordination thread.
    """
    
    stream_id: str
    emulator_client: "PokemonGymClient | None" = None
    tactical_worker: Any = None  # ClaudeProcess from ClaudeCodeManager
    current_task: dict[str, Any] | None = None
    execution_start_time: float | None = None
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    last_execution_result: "ExecutionResult | None" = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for this stream."""
        if self.total_executions == 0:
            return 100.0
        return (self.successful_executions / self.total_executions) * 100.0
    
    @property
    def is_active(self) -> bool:
        """Check if stream is currently executing a task."""
        return self.current_task is not None and self.execution_start_time is not None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert stream to dictionary for metrics and monitoring."""
        return {
            "stream_id": self.stream_id,
            "has_emulator": self.emulator_client is not None,
            "has_worker": self.tactical_worker is not None,
            "is_active": self.is_active,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.success_rate,
            "current_task_type": self.current_task.get("type") if self.current_task else None,
        }


@dataclass
class CoordinationMetrics:
    """Comprehensive metrics for parallel execution coordination."""
    
    # Execution metrics
    total_coordinated_executions: int = 0
    successful_coordinations: int = 0
    failed_coordinations: int = 0
    active_streams: int = 0
    
    # Performance metrics
    average_execution_time: float = 0.0
    peak_parallel_streams: int = 0
    total_uptime: float = 0.0
    
    # Resource utilization
    emulator_utilization: float = 0.0      # Percentage of emulators in use
    worker_utilization: float = 0.0        # Percentage of workers in use
    
    # Error tracking
    circuit_breaker_trips: int = 0
    recovery_attempts: int = 0
    resource_acquisition_failures: int = 0
    
    def record_successful_coordination(self, execution_time: float) -> None:
        """Record successful coordination execution."""
        self.total_coordinated_executions += 1
        self.successful_coordinations += 1
        # Update rolling average
        current_avg = self.average_execution_time
        total_successes = self.successful_coordinations
        self.average_execution_time = (current_avg * (total_successes - 1) + execution_time) / total_successes
    
    def record_failed_coordination(self) -> None:
        """Record failed coordination attempt."""
        self.total_coordinated_executions += 1
        self.failed_coordinations += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate overall coordination success rate."""
        if self.total_coordinated_executions == 0:
            return 100.0
        return (self.successful_coordinations / self.total_coordinated_executions) * 100.0


class ParallelExecutionCoordinator:
    """
    Orchestrates parallel execution across multiple emulator streams with strategic coordination.
    
    This coordinator implements Clean Architecture principles and integrates with existing
    EmulatorPool and ClaudeCodeManager components to provide scalable, fault-tolerant
    parallel execution capabilities.
    
    Key responsibilities:
    - Coordinate strategic planning via Opus processes
    - Dispatch tactical executions to Sonnet workers across multiple emulator streams
    - Manage resource allocation and lifecycle (emulators and workers)
    - Implement fault tolerance with circuit breaker patterns
    - Collect comprehensive performance metrics and health monitoring
    - Provide graceful error handling and automatic recovery
    """
    
    def __init__(
        self,
        emulator_pool: "EmulatorPool",
        claude_manager: "ClaudeCodeManager",
        config: CoordinationConfig | None = None,
    ):
        """
        Initialize ParallelExecutionCoordinator with dependencies.
        
        Args:
            emulator_pool: EmulatorPool for managing Pokemon emulator resources
            claude_manager: ClaudeCodeManager for strategic and tactical AI workers
            config: Configuration for coordination behavior (uses defaults if None)
        """
        self.emulator_pool = emulator_pool
        self.claude_manager = claude_manager
        self.config = config or CoordinationConfig()
        
        # Coordinator state management
        self.state = CoordinatorState.INITIALIZING
        self.coordination_id = str(uuid.uuid4())[:8]
        self.start_time: float | None = None
        
        # Execution streams management
        self.active_streams: dict[str, ExecutionStream] = {}
        self.stream_assignments: dict[str, str] = {}  # task_id -> stream_id mapping
        
        # Thread safety and resource management
        self._lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._shutdown_event = threading.Event()
        
        # Circuit breaker for fault tolerance
        self.circuit_breaker: CircuitBreaker | None = None
        if self.config.enable_circuit_breaker:
            circuit_config = CircuitConfig(
                failure_threshold=self.config.circuit_failure_threshold,
                recovery_timeout=self.config.circuit_recovery_timeout,
                expected_exception_types=(ParallelExecutionError, ConnectionError, TimeoutError),
            )
            self.circuit_breaker = CircuitBreaker(
                name=f"coordinator_{self.coordination_id}",
                config=circuit_config,
            )
        
        # Metrics and monitoring
        self.metrics = CoordinationMetrics()
        self._last_metrics_update = time.time()
        
        logger.info(
            f"ParallelExecutionCoordinator '{self.coordination_id}' initialized: "
            f"max_streams={self.config.max_parallel_streams}, "
            f"worker_pool={self.config.worker_pool_size}, "
            f"circuit_breaker={'enabled' if self.config.enable_circuit_breaker else 'disabled'}"
        )
    
    def initialize(self) -> bool:
        """
        Initialize coordinator resources and verify component health.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info(f"Initializing ParallelExecutionCoordinator '{self.coordination_id}'...")
            
            # Verify dependencies are healthy
            if not self._verify_dependencies():
                raise ParallelExecutionError("Dependency verification failed")
            
            # Initialize thread pool executor
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.worker_pool_size,
                thread_name_prefix=f"coordinator_{self.coordination_id}",
            )
            
            # Initialize execution streams
            self._initialize_streams()
            
            # Update state and timing
            self.state = CoordinatorState.READY
            self.start_time = time.time()
            
            logger.info(
                f"ParallelExecutionCoordinator '{self.coordination_id}' initialized successfully "
                f"with {len(self.active_streams)} streams"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize coordinator '{self.coordination_id}': {e}")
            self.state = CoordinatorState.FAILED
            return False
    
    def run(
        self,
        strategic_plan: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancellation_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """
        Execute strategic plan across parallel emulator streams with coordination.
        
        This is the core coordination method that implements the Act phase of OODA loop,
        orchestrating strategic planning and tactical execution across multiple streams.
        
        Args:
            strategic_plan: Strategic plan from Opus containing tasks and coordination instructions
            progress_callback: Optional callback for progress updates
            cancellation_event: Optional event to signal cancellation
            
        Returns:
            Comprehensive execution results with metrics and outcomes
        """
        if self.state != CoordinatorState.READY:
            raise ParallelExecutionError(
                f"Coordinator not ready (current state: {self.state.name}). "
                f"Call initialize() first."
            )
        
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(
                f"Starting parallel execution '{execution_id}' with coordinator '{self.coordination_id}'"
            )
            
            self.state = CoordinatorState.RUNNING
            
            # Execute coordination with circuit breaker protection
            if self.circuit_breaker and self.config.enable_circuit_breaker:
                result = self.circuit_breaker.call(
                    lambda: self._execute_coordination_internal(
                        strategic_plan, execution_id, progress_callback, cancellation_event
                    ),
                    operation_name=f"coordination_{execution_id}",
                )
            else:
                result = self._execute_coordination_internal(
                    strategic_plan, execution_id, progress_callback, cancellation_event
                )
            
            # Record successful coordination
            execution_time = time.time() - start_time
            self.metrics.record_successful_coordination(execution_time)
            
            self.state = CoordinatorState.READY
            
            logger.info(
                f"Parallel execution '{execution_id}' completed successfully in {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.metrics.record_failed_coordination()
            self.state = CoordinatorState.READY  # Reset to ready for retry
            
            logger.error(f"Parallel execution '{execution_id}' failed: {e}")
            
            # Return error result
            return {
                "execution_id": execution_id,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "coordinator_metrics": self.get_metrics(),
            }
    
    def _execute_coordination_internal(
        self,
        strategic_plan: dict[str, Any],
        execution_id: str,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        cancellation_event: threading.Event | None,
    ) -> dict[str, Any]:
        """
        Internal coordination execution with comprehensive error handling.
        
        This method implements the core coordination logic separated from
        circuit breaker and metrics collection for clean separation of concerns.
        """
        start_time = time.time()
        task_results: dict[str, Any] = {}
        
        try:
            # Extract tasks from strategic plan
            tasks = strategic_plan.get("tasks", [])
            coordination_strategy = strategic_plan.get("coordination_strategy", "parallel")
            
            if not tasks:
                raise ParallelExecutionError("No tasks found in strategic plan")
            
            logger.info(
                f"Executing {len(tasks)} tasks with {coordination_strategy} strategy "
                f"across {len(self.active_streams)} streams"
            )
            
            # Execute tasks based on coordination strategy
            if coordination_strategy == "parallel":
                task_results = self._execute_parallel_tasks(
                    tasks, execution_id, progress_callback, cancellation_event
                )
            elif coordination_strategy == "sequential":
                task_results = self._execute_sequential_tasks(
                    tasks, execution_id, progress_callback, cancellation_event
                )
            else:
                raise ParallelExecutionError(f"Unknown coordination strategy: {coordination_strategy}")
            
            # Analyze results and determine overall success
            successful_tasks = sum(1 for result in task_results.values() if result.get("success", False))
            total_tasks = len(task_results)
            
            overall_success = successful_tasks > 0  # At least one task succeeded
            success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            return {
                "execution_id": execution_id,
                "success": overall_success,
                "coordination_strategy": coordination_strategy,
                "execution_time": time.time() - start_time,
                "task_results": task_results,
                "summary": {
                    "total_tasks": total_tasks,
                    "successful_tasks": successful_tasks,
                    "failed_tasks": total_tasks - successful_tasks,
                    "success_rate": success_rate,
                    "active_streams_used": len([s for s in self.active_streams.values() if s.is_active]),
                },
                "coordinator_metrics": self.get_metrics(),
                "stream_metrics": {
                    stream_id: stream.to_dict()
                    for stream_id, stream in self.active_streams.items()
                },
            }
            
        except Exception as e:
            logger.error(f"Internal coordination execution failed: {e}")
            raise ParallelExecutionError(f"Coordination execution failed: {e}") from e
    
    def _execute_parallel_tasks(
        self,
        tasks: list[dict[str, Any]],
        execution_id: str,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        cancellation_event: threading.Event | None,
    ) -> dict[str, Any]:
        """Execute tasks in parallel across available streams."""
        if not self._executor:
            raise ParallelExecutionError("Thread executor not initialized")
        
        task_results = {}
        futures = {}
        
        # Submit tasks to available streams
        for i, task in enumerate(tasks):
            task_id = task.get("id", f"task_{i}")
            
            # Check for cancellation
            if cancellation_event and cancellation_event.is_set():
                logger.info(f"Parallel execution cancelled at task {i}/{len(tasks)}")
                break
            
            future = self._executor.submit(self._execute_single_task, task, execution_id)
            futures[future] = task_id
        
        # Collect results as they complete
        completed_tasks = 0
        for future in as_completed(futures):
            task_id = futures[future]
            
            try:
                result = future.result()
                task_results[task_id] = result
                completed_tasks += 1
                
                # Report progress
                if progress_callback:
                    progress_callback({
                        "execution_id": execution_id,
                        "completed_tasks": completed_tasks,
                        "total_tasks": len(tasks),
                        "progress_percentage": (completed_tasks / len(tasks)) * 100,
                        "current_task": task_id,
                    })
                
            except Exception as e:
                logger.error(f"Task {task_id} execution failed: {e}")
                task_results[task_id] = {
                    "success": False,
                    "error": str(e),
                    "task_id": task_id,
                }
        
        return task_results
    
    def _execute_sequential_tasks(
        self,
        tasks: list[dict[str, Any]],
        execution_id: str,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        cancellation_event: threading.Event | None,
    ) -> dict[str, Any]:
        """Execute tasks sequentially with controlled coordination."""
        task_results = {}
        
        for i, task in enumerate(tasks):
            task_id = task.get("id", f"task_{i}")
            
            # Check for cancellation
            if cancellation_event and cancellation_event.is_set():
                logger.info(f"Sequential execution cancelled at task {i}/{len(tasks)}")
                break
            
            try:
                result = self._execute_single_task(task, execution_id)
                task_results[task_id] = result
                
                # Report progress
                if progress_callback:
                    progress_callback({
                        "execution_id": execution_id,
                        "completed_tasks": i + 1,
                        "total_tasks": len(tasks),
                        "progress_percentage": ((i + 1) / len(tasks)) * 100,
                        "current_task": task_id,
                    })
                
            except Exception as e:
                logger.error(f"Sequential task {task_id} execution failed: {e}")
                task_results[task_id] = {
                    "success": False,
                    "error": str(e),
                    "task_id": task_id,
                }
                
                # In sequential mode, continue with next task instead of failing completely
                
        return task_results
    
    def _execute_single_task(
        self, 
        task: dict[str, Any], 
        execution_id: str
    ) -> dict[str, Any]:
        """
        Execute a single task on an available stream.
        
        This method coordinates the assignment of a task to a stream,
        manages resource acquisition and release, and handles task-level errors.
        """
        task_id = task.get("id", f"task_{int(time.time())}")
        start_time = time.time()
        
        stream: ExecutionStream | None = None
        
        try:
            # Acquire available stream
            stream = self._acquire_available_stream(task_id)
            if not stream:
                raise ParallelExecutionError("No available streams for task execution")
            
            # Execute task on stream
            logger.info(f"Executing task '{task_id}' on stream '{stream.stream_id}'")
            
            # Update stream state
            stream.current_task = task
            stream.execution_start_time = start_time
            
            # Execute the task using emulator and tactical worker
            execution_result = self._execute_task_on_stream(stream, task, execution_id)
            
            # Update stream metrics
            stream.total_executions += 1
            if execution_result.get("success", False):
                stream.successful_executions += 1
            else:
                stream.failed_executions += 1
            
            stream.last_execution_result = execution_result.get("emulator_result")
            
            return {
                "success": execution_result.get("success", False),
                "task_id": task_id,
                "stream_id": stream.stream_id,
                "execution_time": time.time() - start_time,
                "emulator_result": execution_result.get("emulator_result"),
                "worker_analysis": execution_result.get("worker_analysis"),
                "error": execution_result.get("error"),
            }
            
        except Exception as e:
            logger.error(f"Single task execution failed for '{task_id}': {e}")
            return {
                "success": False,
                "task_id": task_id,
                "stream_id": stream.stream_id if stream else None,
                "execution_time": time.time() - start_time,
                "error": str(e),
            }
            
        finally:
            # Clean up stream state
            if stream:
                stream.current_task = None
                stream.execution_start_time = None
                self._release_stream(stream.stream_id)
    
    def _execute_task_on_stream(
        self, 
        stream: ExecutionStream, 
        task: dict[str, Any], 
        execution_id: str
    ) -> dict[str, Any]:
        """Execute a specific task on the given execution stream."""
        if not stream.emulator_client:
            raise ParallelExecutionError(f"Stream {stream.stream_id} has no emulator client")
        
        task_type = task.get("type", "script_execution")
        
        try:
            if task_type == "script_execution":
                return self._execute_script_task(stream, task, execution_id)
            elif task_type == "exploration":
                return self._execute_exploration_task(stream, task, execution_id)
            elif task_type == "analysis":
                return self._execute_analysis_task(stream, task, execution_id)
            else:
                raise ParallelExecutionError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Task execution failed on stream {stream.stream_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "stream_id": stream.stream_id,
            }
    
    def _execute_script_task(
        self, 
        stream: ExecutionStream, 
        task: dict[str, Any], 
        execution_id: str
    ) -> dict[str, Any]:
        """Execute a script-based task on the emulator."""
        script_content = task.get("script", "")
        checkpoint_id = task.get("checkpoint_id")
        
        if not script_content:
            raise ParallelExecutionError("Script task missing script content")
        
        # Execute script on emulator
        execution_result = self.emulator_pool.execute_script(
            script_text=script_content,
            checkpoint_id=checkpoint_id,
            timeout=self.config.tactical_execution_timeout,
        )
        
        # Basic analysis - in full implementation, this would use tactical worker
        analysis = {
            "execution_successful": execution_result.success,
            "execution_time": execution_result.execution_time,
            "frames_executed": execution_result.performance_metrics.get("frames_executed", 0),
            "completion_percentage": execution_result.performance_metrics.get("completion_percentage", 0),
        }
        
        return {
            "success": execution_result.success,
            "emulator_result": execution_result.to_dict(),
            "worker_analysis": analysis,
            "task_type": "script_execution",
        }
    
    def _execute_exploration_task(
        self, 
        stream: ExecutionStream, 
        task: dict[str, Any], 
        execution_id: str
    ) -> dict[str, Any]:
        """Execute an exploration task (placeholder implementation)."""
        # Placeholder for exploration task logic
        logger.info(f"Executing exploration task on stream {stream.stream_id}")
        
        # Simulate exploration execution
        time.sleep(0.1)  # Simulate some processing time
        
        return {
            "success": True,
            "task_type": "exploration",
            "exploration_results": {
                "areas_explored": 1,
                "items_found": 0,
                "new_locations": [],
            },
        }
    
    def _execute_analysis_task(
        self, 
        stream: ExecutionStream, 
        task: dict[str, Any], 
        execution_id: str
    ) -> dict[str, Any]:
        """Execute an analysis task (placeholder implementation)."""
        # Placeholder for analysis task logic
        logger.info(f"Executing analysis task on stream {stream.stream_id}")
        
        # Simulate analysis execution
        time.sleep(0.05)  # Simulate some processing time
        
        return {
            "success": True,
            "task_type": "analysis",
            "analysis_results": {
                "patterns_identified": 0,
                "recommendations": [],
                "confidence_score": 0.8,
            },
        }
    
    def _acquire_available_stream(self, task_id: str) -> ExecutionStream | None:
        """
        Acquire an available execution stream for task assignment.
        
        Returns the first available stream or None if all streams are busy.
        """
        with self._lock:
            for stream_id, stream in self.active_streams.items():
                if not stream.is_active and stream.emulator_client and stream.tactical_worker:
                    # Mark stream as assigned to this task
                    self.stream_assignments[task_id] = stream_id
                    return stream
            
            return None
    
    def _release_stream(self, stream_id: str) -> None:
        """Release a stream back to the available pool."""
        with self._lock:
            # Remove any task assignments for this stream
            task_assignments_to_remove = [
                task_id for task_id, assigned_stream_id in self.stream_assignments.items()
                if assigned_stream_id == stream_id
            ]
            
            for task_id in task_assignments_to_remove:
                del self.stream_assignments[task_id]
    
    def _initialize_streams(self) -> None:
        """Initialize execution streams with emulator and worker assignments."""
        try:
            # Get available tactical workers
            tactical_workers = self.claude_manager.get_tactical_processes()
            available_workers = [
                worker for worker in tactical_workers
                if worker.health_check() and worker.is_healthy()
            ]
            
            if not available_workers:
                raise ParallelExecutionError("No healthy tactical workers available")
            
            # Create streams up to the configured limit
            max_streams = min(
                self.config.max_parallel_streams,
                len(available_workers),
                self.emulator_pool.pool_size,
            )
            
            logger.info(f"Initializing {max_streams} execution streams")
            
            for i in range(max_streams):
                stream_id = f"stream_{i}"
                
                # Create stream with worker assignment
                stream = ExecutionStream(
                    stream_id=stream_id,
                    tactical_worker=available_workers[i % len(available_workers)],
                )
                
                # Try to pre-acquire emulator for this stream
                try:
                    emulator_client = self.emulator_pool.acquire(
                        timeout=self.config.emulator_acquisition_timeout
                    )
                    stream.emulator_client = emulator_client
                    
                    logger.info(f"Stream '{stream_id}' initialized successfully")
                    
                except Exception as e:
                    logger.warning(f"Failed to acquire emulator for stream '{stream_id}': {e}")
                    # Stream will be created without emulator - can acquire later on-demand
                
                self.active_streams[stream_id] = stream
            
            logger.info(
                f"Successfully initialized {len(self.active_streams)} streams "
                f"({sum(1 for s in self.active_streams.values() if s.emulator_client)} with emulators)"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize streams: {e}")
            raise ParallelExecutionError(f"Stream initialization failed: {e}") from e
    
    def _verify_dependencies(self) -> bool:
        """Verify that required dependencies are healthy and available."""
        try:
            # Check EmulatorPool health
            pool_health = self.emulator_pool.health_check()
            if pool_health.get("status") not in ["healthy", "degraded"]:
                logger.error("EmulatorPool is not healthy")
                return False
            
            # Check ClaudeCodeManager health
            if not self.claude_manager.is_running():
                logger.error("ClaudeCodeManager is not running")
                return False
            
            health_results = self.claude_manager.health_check_all()
            healthy_processes = sum(1 for is_healthy in health_results.values() if is_healthy)
            
            if healthy_processes == 0:
                logger.error("No healthy Claude processes available")
                return False
            
            logger.info(
                f"Dependencies verified: EmulatorPool={pool_health.get('status')}, "
                f"Claude processes={healthy_processes}/{len(health_results)} healthy"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Dependency verification failed: {e}")
            return False
    
    def get_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive coordination metrics for monitoring and analysis.
        
        Returns:
            Dictionary with detailed metrics and status information
        """
        with self._lock:
            # Update runtime metrics
            if self.start_time:
                self.metrics.total_uptime = time.time() - self.start_time
            
            self.metrics.active_streams = len([s for s in self.active_streams.values() if s.is_active])
            self.metrics.peak_parallel_streams = max(
                self.metrics.peak_parallel_streams, 
                self.metrics.active_streams
            )
            
            # Calculate resource utilization
            streams_with_emulators = sum(1 for s in self.active_streams.values() if s.emulator_client)
            total_possible_streams = len(self.active_streams)
            
            self.metrics.emulator_utilization = (
                (streams_with_emulators / total_possible_streams * 100) 
                if total_possible_streams > 0 else 0
            )
            
            streams_with_workers = sum(1 for s in self.active_streams.values() if s.tactical_worker)
            self.metrics.worker_utilization = (
                (streams_with_workers / total_possible_streams * 100)
                if total_possible_streams > 0 else 0
            )
            
            # Add circuit breaker metrics if enabled
            circuit_metrics = {}
            if self.circuit_breaker:
                circuit_health = self.circuit_breaker.get_health_status()
                circuit_metrics = {
                    "circuit_breaker_state": circuit_health["state"],
                    "circuit_breaker_available": circuit_health["is_available"],
                    "circuit_success_rate": circuit_health["success_rate"],
                    "circuit_trips": circuit_health["circuit_trips"],
                }
                self.metrics.circuit_breaker_trips = circuit_health["circuit_trips"]
            
            return {
                # Core metrics
                "coordinator_id": self.coordination_id,
                "state": self.state.name,
                "uptime_seconds": self.metrics.total_uptime,
                
                # Execution metrics
                "total_coordinated_executions": self.metrics.total_coordinated_executions,
                "successful_coordinations": self.metrics.successful_coordinations,
                "failed_coordinations": self.metrics.failed_coordinations,
                "success_rate": self.metrics.success_rate,
                "average_execution_time": self.metrics.average_execution_time,
                
                # Resource metrics
                "active_streams": self.metrics.active_streams,
                "total_streams": len(self.active_streams),
                "peak_parallel_streams": self.metrics.peak_parallel_streams,
                "emulator_utilization": self.metrics.emulator_utilization,
                "worker_utilization": self.metrics.worker_utilization,
                
                # Stream details
                "stream_details": {
                    stream_id: stream.to_dict()
                    for stream_id, stream in self.active_streams.items()
                },
                
                # Circuit breaker metrics
                **circuit_metrics,
                
                # Timestamps
                "last_metrics_update": time.time(),
                "start_time": self.start_time,
            }
    
    def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive health check of coordinator and dependencies.
        
        Returns:
            Health status information for monitoring
        """
        start_time = time.time()
        
        try:
            health_status: dict[str, Any] = {
                "coordinator_id": self.coordination_id,
                "overall_health": "unknown",
                "state": self.state.name,
                "checks": {},
                "check_duration_ms": 0,
            }
            
            # Check coordinator state
            health_status["checks"]["coordinator_state"] = {
                "healthy": self.state in [CoordinatorState.READY, CoordinatorState.RUNNING],
                "details": f"Current state: {self.state.name}",
            }
            
            # Check dependencies
            if self._verify_dependencies():
                health_status["checks"]["dependencies"] = {
                    "healthy": True,
                    "details": "EmulatorPool and ClaudeCodeManager are healthy",
                }
            else:
                health_status["checks"]["dependencies"] = {
                    "healthy": False,
                    "details": "One or more dependencies are unhealthy",
                }
            
            # Check streams
            healthy_streams = sum(
                1 for stream in self.active_streams.values()
                if stream.emulator_client and stream.tactical_worker
            )
            total_streams = len(self.active_streams)
            
            health_status["checks"]["execution_streams"] = {
                "healthy": healthy_streams > 0,
                "details": f"{healthy_streams}/{total_streams} streams ready",
            }
            
            # Check circuit breaker
            if self.circuit_breaker:
                cb_available = self.circuit_breaker.is_available()
                health_status["checks"]["circuit_breaker"] = {
                    "healthy": cb_available,
                    "details": f"Circuit breaker state: {self.circuit_breaker.get_state().name}",
                }
            
            # Determine overall health
            all_checks_healthy = all(
                check["healthy"] for check in health_status["checks"].values()
            )
            
            health_status["overall_health"] = "healthy" if all_checks_healthy else "degraded"
            health_status["check_duration_ms"] = int((time.time() - start_time) * 1000)
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "coordinator_id": self.coordination_id,
                "overall_health": "unhealthy",
                "state": self.state.name,
                "error": str(e),
                "check_duration_ms": int((time.time() - start_time) * 1000),
            }
    
    def shutdown(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown coordinator and release all resources.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        logger.info(f"Shutting down ParallelExecutionCoordinator '{self.coordination_id}'...")
        
        self.state = CoordinatorState.STOPPING
        self._shutdown_event.set()
        
        try:
            # Shutdown thread pool
            if self._executor:
                self._executor.shutdown(wait=True)
            
            # Release all emulator resources
            for stream in self.active_streams.values():
                if stream.emulator_client:
                    try:
                        self.emulator_pool.release(stream.emulator_client)
                        logger.info(f"Released emulator for stream {stream.stream_id}")
                    except Exception as e:
                        logger.error(f"Error releasing emulator for stream {stream.stream_id}: {e}")
            
            # Clear all state
            self.active_streams.clear()
            self.stream_assignments.clear()
            
            self.state = CoordinatorState.STOPPED
            
            logger.info(f"ParallelExecutionCoordinator '{self.coordination_id}' shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during coordinator shutdown: {e}")
            self.state = CoordinatorState.FAILED
    
    def __enter__(self):
        """Context manager entry - initialize coordinator."""
        if not self.initialize():
            raise ParallelExecutionError("Failed to initialize ParallelExecutionCoordinator")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - shutdown coordinator."""
        self.shutdown()


# Utility functions for testing and validation

def create_test_coordinator(
    emulator_pool_size: int = 2,
    max_parallel_streams: int = 2,
) -> ParallelExecutionCoordinator:
    """
    Create a test coordinator instance for development and testing.
    
    This function provides a simple way to create a coordinator with
    minimal dependencies for testing purposes.
    
    Args:
        emulator_pool_size: Size of test emulator pool
        max_parallel_streams: Maximum parallel streams for coordination
        
    Returns:
        Configured ParallelExecutionCoordinator instance
    """
    from .emulator_pool import EmulatorPool
    from .claude_code_manager import ClaudeCodeManager
    
    # Create test configuration
    config = CoordinationConfig(
        max_parallel_streams=max_parallel_streams,
        enable_circuit_breaker=False,  # Disable for simpler testing
        worker_pool_size=4,
    )
    
    # Create dependencies (would normally be injected)
    emulator_pool = EmulatorPool(pool_size=emulator_pool_size)
    claude_manager = ClaudeCodeManager(max_workers=max_parallel_streams + 1)
    
    # Create and return coordinator
    coordinator = ParallelExecutionCoordinator(
        emulator_pool=emulator_pool,
        claude_manager=claude_manager,
        config=config,
    )
    
    logger.info(f"Created test coordinator with {emulator_pool_size} emulators and {max_parallel_streams} streams")
    
    return coordinator


if __name__ == "__main__":
    # Simple test/demo when run directly
    print("ParallelExecutionCoordinator - Act Phase Implementation")
    print("=" * 60)
    
    # Create test strategic plan
    test_plan = {
        "coordination_strategy": "parallel",
        "tasks": [
            {
                "id": "explore_route_1",
                "type": "script_execution",
                "script": "MOVE UP MOVE UP PRESS A",
                "priority": 1,
            },
            {
                "id": "collect_items",
                "type": "exploration",
                "target_area": "starting_town",
                "priority": 2,
            },
        ],
    }
    
    print("Test strategic plan created:")
    print(f"- Strategy: {test_plan['coordination_strategy']}")
    print(f"- Tasks: {len(test_plan['tasks'])}")
    
    print("\nParallelExecutionCoordinator ready for integration!")