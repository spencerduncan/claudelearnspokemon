"""
SonnetWorkerPool - Worker Pool Initialization and Management

This module provides a high-level abstraction layer over ClaudeCodeManager's
tactical processes for managing Sonnet workers in parallel script development.
It implements worker pool initialization, task assignment, and lifecycle management
following Clean Architecture principles.

Key Responsibilities:
- Initialize configurable number of Sonnet workers
- Assign unique worker IDs and track worker status
- Verify worker health and connectivity during initialization
- Support worker pool scaling and dynamic reconfiguration
- Provide load balancing for task assignment

Performance Targets:
- Worker pool initialization: <500ms
- Worker health checks: <10ms per worker
- Task assignment: <50ms
- Worker status queries: <10ms
"""

import logging
import queue
import threading
import time
import uuid
from typing import Any

from .claude_code_manager import ClaudeCodeManager
from .mcp_data_patterns import PokemonStrategy, QueryBuilder

logger = logging.getLogger(__name__)


class SonnetWorkerPool:
    """
    High-level abstraction for managing a pool of Claude Sonnet workers.

    This class provides clean task assignment and worker lifecycle management
    by wrapping ClaudeCodeManager's tactical processes with pool semantics.
    It focuses on operational concerns like load balancing, health monitoring,
    and worker status tracking.
    """

    def __init__(self, claude_manager: ClaudeCodeManager):
        """
        Initialize SonnetWorkerPool with Claude manager dependency.

        Args:
            claude_manager: ClaudeCodeManager instance providing tactical processes
        """
        self.claude_manager = claude_manager
        self.workers: dict[str, dict[str, Any]] = {}
        self._current_assignment_index = 0
        self._initialized = False

        # Task queueing system
        self.task_queue: queue.Queue = queue.Queue()
        self.worker_assignments: dict[str, dict[str, Any]] = {}  # worker_id -> task info
        self._queue_lock = threading.Lock()

        logger.info("SonnetWorkerPool initialized with task queueing support")

    def initialize(self, worker_count: int = 4) -> bool:
        """
        Initialize worker pool with specified number of workers.

        This method:
        1. Ensures ClaudeCodeManager processes are started
        2. Gets tactical (Sonnet) processes from the manager
        3. Assigns unique worker IDs to healthy processes
        4. Verifies worker health and connectivity
        5. Sets up worker status tracking

        Args:
            worker_count: Number of workers to initialize (default 4)

        Returns:
            True if initialization successful, False otherwise

        Performance Requirements:
            - Total initialization time: <500ms
            - Health verification: <10ms per worker
        """
        start_time = time.time()
        logger.info(f"Initializing SonnetWorkerPool with {worker_count} workers")

        try:
            # Ensure ClaudeCodeManager processes are started
            if not self.claude_manager.start_all_processes():
                logger.error("Failed to start ClaudeCodeManager processes")
                return False

            # Get tactical processes from ClaudeCodeManager
            tactical_processes = self.claude_manager.get_tactical_processes()

            if len(tactical_processes) < worker_count:
                logger.warning(
                    f"Requested {worker_count} workers but only {len(tactical_processes)} available"
                )

            # Initialize workers from available tactical processes
            healthy_worker_count = 0
            for _i, process in enumerate(tactical_processes[:worker_count]):
                # Verify worker health during initialization
                health_check_start = time.time()
                if not process.health_check():
                    logger.warning(f"Process {process.process_id} failed health check, skipping")
                    continue

                health_check_time = time.time() - health_check_start
                if health_check_time > 0.01:  # 10ms requirement
                    logger.warning(
                        f"Health check took {health_check_time:.3f}s, exceeds 10ms target"
                    )

                # Assign unique worker ID
                worker_id = f"sonnet_worker_{uuid.uuid4().hex[:8]}"

                # Create worker registration
                worker_info = {
                    "worker_id": worker_id,
                    "process_id": process.process_id,
                    "process": process,
                    "status": "ready",
                    "healthy": True,
                    "task_count": 0,
                    "last_health_check": time.time(),
                    "created_at": time.time(),
                }

                self.workers[worker_id] = worker_info
                healthy_worker_count += 1

                logger.debug(f"Registered worker {worker_id} with process {process.process_id}")

            # Validate initialization success
            self._initialized = healthy_worker_count > 0

            initialization_time = time.time() - start_time
            logger.info(
                f"Initialized {healthy_worker_count}/{worker_count} workers in {initialization_time:.3f}s"
            )

            # Check performance requirement
            if initialization_time > 0.5:
                logger.warning(
                    f"Initialization took {initialization_time:.3f}s, exceeds 500ms target"
                )

            return self._initialized

        except Exception as e:
            logger.error(f"Worker pool initialization failed: {e}")
            return False

    def get_worker_status(self, worker_id: str) -> dict[str, Any] | None:
        """
        Get detailed status information for a specific worker.

        Args:
            worker_id: Unique worker identifier

        Returns:
            Dictionary with worker status or None if worker not found

        Status includes:
            - healthy: Boolean indicating worker health
            - process_id: Associated ClaudeProcess ID
            - worker_id: Unique worker identifier
            - status: Current worker status string
            - task_count: Number of tasks processed
            - last_health_check: Timestamp of last health verification
            - created_at: Worker creation timestamp
        """
        if worker_id not in self.workers:
            logger.debug(f"Worker {worker_id} not found")
            return None

        worker_info = self.workers[worker_id]
        process = worker_info["process"]

        # Refresh health status
        current_health = process.is_healthy()
        worker_info["healthy"] = current_health
        worker_info["last_health_check"] = time.time()

        # Return comprehensive status
        status = {
            "healthy": current_health,
            "process_id": worker_info["process_id"],
            "worker_id": worker_id,
            "status": worker_info["status"],
            "task_count": worker_info["task_count"],
            "last_health_check": worker_info["last_health_check"],
            "created_at": worker_info["created_at"],
        }

        return status

    def restart_worker(self, worker_id: str) -> bool:
        """
        Restart a specific worker conversation.

        This method restarts the underlying ClaudeProcess and updates
        worker tracking information accordingly.

        Args:
            worker_id: Unique worker identifier

        Returns:
            True if restart successful, False otherwise
        """
        if worker_id not in self.workers:
            logger.warning(f"Cannot restart unknown worker {worker_id}")
            return False

        worker_info = self.workers[worker_id]
        process = worker_info["process"]

        try:
            logger.info(f"Restarting worker {worker_id}")
            restart_success = process.restart()

            if restart_success:
                # Update worker tracking
                worker_info["status"] = "ready"
                worker_info["task_count"] = 0  # Reset task count after restart
                worker_info["last_health_check"] = time.time()
                worker_info["healthy"] = True

                logger.info(f"Worker {worker_id} restarted successfully")
                return True
            else:
                logger.error(f"Failed to restart worker {worker_id}")
                worker_info["healthy"] = False
                worker_info["status"] = "failed"
                return False

        except Exception as e:
            logger.error(f"Exception during worker restart: {e}")
            worker_info["healthy"] = False
            worker_info["status"] = "error"
            return False

    def assign_task(self, task: dict[str, Any]) -> str:
        """
        Assign a task to an available worker or queue it if all workers are busy.

        This method finds a healthy, available worker and assigns the task immediately,
        or queues the task if no workers are available. Returns a task ID for tracking.
        Uses simple round-robin selection for load balancing.

        Args:
            task: Task dictionary with objective and context

        Returns:
            Task ID for tracking (either immediate assignment or queued)

        Performance Requirements:
            - Task assignment: <50ms (including queueing)
        """
        if not self._initialized or not self.workers:
            raise ValueError("Worker pool not initialized, cannot assign task")

        start_time = time.time()
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        # Add task metadata
        task_with_metadata = {
            "task_id": task_id,
            "task": task,
            "queued_at": time.time(),
            "attempts": 0,
        }

        with self._queue_lock:
            # Get list of available workers (healthy and not assigned)
            # Update cached health state before filtering
            for worker_info in self.workers.values():
                if worker_info["healthy"] and not worker_info["process"].is_healthy():
                    worker_info["healthy"] = False  # Update stale cached state

            available_workers = [
                (worker_id, worker_info)
                for worker_id, worker_info in self.workers.items()
                if worker_info["healthy"] and worker_id not in self.worker_assignments
            ]

            if available_workers:
                # Immediate assignment - worker available
                worker_index = self._current_assignment_index % len(available_workers)
                selected_worker_id, selected_worker_info = available_workers[worker_index]
                self._current_assignment_index += 1

                # Update worker tracking
                selected_worker_info["status"] = "assigned"
                selected_worker_info["task_count"] += 1

                # Track active assignment
                self.worker_assignments[selected_worker_id] = task_with_metadata

                assignment_time = time.time() - start_time
                if assignment_time > 0.05:  # 50ms requirement
                    logger.warning(
                        f"Task assignment took {assignment_time:.3f}s, exceeds 50ms target"
                    )

                logger.info(f"Assigned task {task_id} to worker {selected_worker_id}")
                return task_id
            else:
                # Queue task - no workers available
                self.task_queue.put(task_with_metadata)
                assignment_time = time.time() - start_time

                if assignment_time > 0.05:  # 50ms requirement
                    logger.warning(
                        f"Task queueing took {assignment_time:.3f}s, exceeds 50ms target"
                    )

                logger.info(f"Queued task {task_id} (queue size: {self.task_queue.qsize()})")
                return task_id

    def develop_script(self, worker_id: str, task: dict[str, Any]) -> dict[str, Any] | None:
        """
        Develop script using specified worker for given task.

        This method sends the task to the specified worker's Claude process
        and returns the script development result.

        Args:
            worker_id: Unique worker identifier
            task: Task dictionary with development requirements

        Returns:
            Script development result dictionary or None if failed
        """
        if worker_id not in self.workers:
            logger.warning(f"Cannot develop script with unknown worker {worker_id}")
            return None

        worker_info = self.workers[worker_id]
        process = worker_info["process"]

        if not process.is_healthy():
            logger.warning(f"Worker {worker_id} is not healthy, cannot develop script")
            return None

        try:
            # Format task for Sonnet tactical worker
            task_message = self._format_task_message(task)

            # Send message to worker process
            response = process.send_message(task_message, timeout=30.0)

            if response is None:
                logger.error(f"No response from worker {worker_id}")
                return None

            # Update worker status
            worker_info["status"] = "ready"  # Ready for next task

            # Format response as script development result
            result = {
                "worker_id": worker_id,
                "script": response,
                "status": "completed",
                "timestamp": time.time(),
            }

            logger.debug(f"Worker {worker_id} completed script development")
            return result

        except Exception as e:
            logger.error(f"Script development failed for worker {worker_id}: {e}")
            worker_info["status"] = "error"
            return None

    def complete_task(self, worker_id: str) -> bool:
        """
        Mark a task as completed for the specified worker and process queued tasks.

        This method should be called after a worker finishes processing a task
        to make the worker available for new assignments and process any queued tasks.

        Args:
            worker_id: Unique worker identifier

        Returns:
            True if task completion processed successfully, False otherwise
        """
        if worker_id not in self.workers:
            logger.warning(f"Cannot complete task for unknown worker {worker_id}")
            return False

        with self._queue_lock:
            # Remove worker assignment
            if worker_id in self.worker_assignments:
                completed_task = self.worker_assignments.pop(worker_id)
                logger.info(f"Completed task {completed_task['task_id']} for worker {worker_id}")

            # Update worker status
            worker_info = self.workers[worker_id]
            worker_info["status"] = "ready"

            # Process next task from queue if available
            self._process_queue()

        return True

    def _process_queue(self):
        """
        Process queued tasks by assigning them to available workers.

        This internal method is called when workers become available to
        handle any pending tasks in the queue. Uses thread-safe queue operations.

        Note: Should be called with _queue_lock held.
        """
        if self.task_queue.empty():
            return

        # Find available workers (healthy and not assigned)
        available_workers = [
            (worker_id, worker_info)
            for worker_id, worker_info in self.workers.items()
            if worker_info["healthy"] and worker_id not in self.worker_assignments
        ]

        # Assign queued tasks to available workers
        tasks_assigned = 0
        while available_workers and not self.task_queue.empty():
            try:
                # Get next task from queue
                task_with_metadata = self.task_queue.get_nowait()

                # Select next available worker (round-robin)
                worker_index = self._current_assignment_index % len(available_workers)
                selected_worker_id, selected_worker_info = available_workers[worker_index]
                self._current_assignment_index += 1

                # Update worker tracking
                selected_worker_info["status"] = "assigned"
                selected_worker_info["task_count"] += 1

                # Track active assignment
                self.worker_assignments[selected_worker_id] = task_with_metadata

                # Remove assigned worker from available list
                available_workers.pop(worker_index)

                tasks_assigned += 1
                logger.info(
                    f"Assigned queued task {task_with_metadata['task_id']} to worker {selected_worker_id}"
                )

            except queue.Empty:
                break

        if tasks_assigned > 0:
            logger.info(f"Processed {tasks_assigned} queued tasks")

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """
        Get status information for a specific task.

        Args:
            task_id: Task identifier returned by assign_task()

        Returns:
            Dictionary with task status or None if task not found
        """
        # Check active assignments
        for worker_id, task_info in self.worker_assignments.items():
            if task_info["task_id"] == task_id:
                return {
                    "task_id": task_id,
                    "status": "assigned",
                    "worker_id": worker_id,
                    "queued_at": task_info["queued_at"],
                    "assigned_at": time.time(),
                }

        # Check if task is queued (expensive operation)
        with self._queue_lock:
            temp_tasks: list[dict[str, Any]] = []
            task_found = None

            # Drain queue to search for task
            while not self.task_queue.empty():
                try:
                    task_info = self.task_queue.get_nowait()
                    if task_info["task_id"] == task_id:
                        task_found = {
                            "task_id": task_id,
                            "status": "queued",
                            "worker_id": None,
                            "queued_at": task_info["queued_at"],
                            "queue_position": len(temp_tasks),
                        }
                    temp_tasks.append(task_info)
                except queue.Empty:
                    break

            # Restore queue
            for task_info in temp_tasks:
                self.task_queue.put(task_info)

            return task_found

    def get_queue_size(self) -> int:
        """Get the current number of queued tasks."""
        return self.task_queue.qsize()

    def share_pattern(self, pattern_data: dict[str, Any], discovered_by: str | None = None) -> bool:
        """
        Share a discovered pattern across all workers in the pool.

        This method stores a successful pattern or strategy using the MCP memory system
        and distributes it to all active workers for improved consistency and learning.

        Args:
            pattern_data: Dictionary containing pattern information (strategy, success_rate, context)
            discovered_by: Optional worker ID that discovered this pattern

        Returns:
            True if pattern was successfully shared, False otherwise
        """
        try:
            # Create a PokemonStrategy object from pattern data
            strategy_id = pattern_data.get("strategy_id", f"strategy_{uuid.uuid4().hex[:8]}")
            strategy = PokemonStrategy(
                id=strategy_id,
                name=pattern_data.get("name", "Discovered Strategy"),
                pattern_sequence=pattern_data.get("pattern_sequence", ["DISCOVERED_PATTERN"]),
                success_rate=pattern_data.get("success_rate", 0.0),
                estimated_time=pattern_data.get("estimated_time"),
                resource_requirements=pattern_data.get("resource_requirements", {}),
                risk_assessment=pattern_data.get("risk_assessment", {}),
                alternatives=pattern_data.get("alternatives", []),
                optimization_history=pattern_data.get("optimization_history", []),
            )

            # Store pattern using MCP memory integration via QueryBuilder
            try:
                query_builder = QueryBuilder()
                result = query_builder.store_pattern(strategy)
                if result.get("success"):
                    logger.debug(f"Stored pattern {strategy.id} via QueryBuilder (memory_id={result.get('memory_id')})")
                else:
                    logger.warning(f"QueryBuilder storage failed: {result.get('error', 'Unknown error')}")
                    return False
            except Exception as query_error:
                logger.warning(f"QueryBuilder failed: {query_error}")
                return False

            if result.get("success", False):
                pattern_id = result.get("memory_id")
                logger.info(f"Stored pattern {strategy.id} with MCP ID {pattern_id}")

                # Distribute pattern to all active workers
                if self._distribute_pattern_to_workers(strategy, discovered_by):
                    logger.info(f"Successfully shared pattern {strategy.id} to all workers")
                    return True
                else:
                    logger.warning("Pattern stored but distribution to workers failed")
                    return False
            else:
                logger.error(f"Failed to store pattern in MCP system: {result}")
                return False

        except Exception as e:
            logger.error(f"Failed to share pattern: {e}")
            return False

    def get_shared_patterns(
        self, context_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve shared patterns from the MCP memory system.

        Args:
            context_filter: Optional filter criteria for pattern retrieval

        Returns:
            List of pattern dictionaries matching the filter criteria
        """
        try:
            # Build query based on filter (for future MCP integration)
            if context_filter:
                # Use context information to build targeted query
                search_terms = []
                if "location" in context_filter:
                    search_terms.append(f"location:{context_filter['location']}")
                if "objective" in context_filter:
                    search_terms.append(f"objective:{context_filter['objective']}")
                # query = " ".join(search_terms) if search_terms else "PokemonStrategy"
            # else:
            #     query = "PokemonStrategy"

            # Retrieve patterns using QueryBuilder
            try:
                query_builder = QueryBuilder()
                
                # Build search query from filter criteria
                if context_filter:
                    search_terms = []
                    if "location" in context_filter:
                        search_terms.append(f"location:{context_filter['location']}")
                    if "objective" in context_filter:
                        search_terms.append(f"objective:{context_filter['objective']}")
                    search_query = " ".join(search_terms) if search_terms else "PokemonStrategy"
                else:
                    search_query = "PokemonStrategy"
                
                result = query_builder.search_patterns(search_query)
                if result.get("success"):
                    logger.debug(f"Retrieved {len(result['results'])} patterns via QueryBuilder")
                else:
                    logger.warning(f"QueryBuilder search failed: {result.get('error', 'Unknown error')}")
                    result = {"success": True, "results": []}
            except Exception as query_error:
                logger.warning(f"QueryBuilder search failed: {query_error}")
                result = {"success": True, "results": []}

            if result.get("success", False):
                patterns = []
                results = result.get("results", [])
                if isinstance(results, list):
                    for pattern_data in results:
                        # Convert MCP result to pattern dictionary
                        pattern_dict = {
                            "pattern_id": pattern_data.get("id"),
                            "strategy_id": pattern_data.get("strategy_id"),
                            "name": pattern_data.get("name"),
                            "description": pattern_data.get("description"),
                            "success_rate": pattern_data.get("success_rate", 0.0),
                            "usage_count": pattern_data.get("usage_count", 0),
                            "context": pattern_data.get("context", {}),
                            "discovered_at": pattern_data.get("discovered_at"),
                        }
                        patterns.append(pattern_dict)

                logger.info(f"Retrieved {len(patterns)} shared patterns")
                return patterns
            else:
                logger.warning(f"Failed to retrieve patterns: {result}")
                return []

        except Exception as e:
            logger.error(f"Failed to retrieve shared patterns: {e}")
            return []

    def _distribute_pattern_to_workers(
        self, strategy: PokemonStrategy, discovered_by: str | None = None
    ) -> bool:
        """
        Internal method to distribute a pattern to all active workers.

        Args:
            strategy: PokemonStrategy object to distribute
            discovered_by: Worker ID that discovered the pattern (will be excluded from distribution)

        Returns:
            True if pattern distributed successfully to at least one worker
        """
        distribution_count = 0

        for worker_id, worker_info in self.workers.items():
            # Skip the worker that discovered this pattern (they already know it)
            if worker_id == discovered_by:
                continue

            # Only send to healthy workers
            if not worker_info["healthy"]:
                continue

            try:
                # Format pattern for worker consumption
                pattern_message = self._format_pattern_message(strategy)

                # Send pattern to worker process
                process = worker_info["process"]
                response = process.send_message(pattern_message, timeout=10.0)

                if response:
                    logger.debug(f"Distributed pattern {strategy.id} to worker {worker_id}")
                    distribution_count += 1
                else:
                    logger.warning(f"No response from worker {worker_id} when distributing pattern")

            except Exception as e:
                logger.error(f"Failed to distribute pattern to worker {worker_id}: {e}")

        return distribution_count > 0

    def _format_pattern_message(self, strategy: PokemonStrategy) -> str:
        """
        Format a pattern/strategy for distribution to Sonnet workers.

        Args:
            strategy: PokemonStrategy object to format

        Returns:
            Formatted message string for worker consumption
        """
        message = f"""SHARED PATTERN UPDATE

Pattern: {strategy.name}
Strategy ID: {strategy.id}
Success Rate: {strategy.success_rate:.2%}
Pattern Sequence: {', '.join(strategy.pattern_sequence)}
Estimated Time: {strategy.estimated_time}s

Resource Requirements: {strategy.resource_requirements}
Risk Assessment: {strategy.risk_assessment}

Please incorporate this successful pattern into your script development approach.
Focus on the techniques and strategies that contributed to its success."""

        return message

    def analyze_result(self, worker_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
        """
        Get refinement suggestions from worker based on execution result.

        Args:
            worker_id: Unique worker identifier
            result: Execution result to analyze

        Returns:
            Analysis with refinement suggestions or None if failed
        """
        if worker_id not in self.workers:
            logger.warning(f"Cannot analyze result with unknown worker {worker_id}")
            return None

        worker_info = self.workers[worker_id]
        process = worker_info["process"]

        if not process.is_healthy():
            logger.warning(f"Worker {worker_id} is not healthy, cannot analyze result")
            return None

        try:
            # Format result analysis request
            analysis_message = self._format_analysis_message(result)

            # Send analysis request to worker
            response = process.send_message(analysis_message, timeout=30.0)

            if response is None:
                logger.error(f"No analysis response from worker {worker_id}")
                return None

            # Format analysis result
            analysis = {"worker_id": worker_id, "suggestions": response, "timestamp": time.time()}

            logger.debug(f"Worker {worker_id} completed result analysis")
            return analysis

        except Exception as e:
            logger.error(f"Result analysis failed for worker {worker_id}: {e}")
            return None

    def _format_task_message(self, task: dict[str, Any]) -> str:
        """Format task dictionary into message for Sonnet tactical worker."""
        objective = task.get("objective", "")
        context = task.get("context", "")

        message = f"""Task: {objective}

Context: {context}

Please develop a Pokemon Red speedrun script to accomplish this objective.
Focus on frame-perfect execution and optimal path planning."""

        return message

    def _format_analysis_message(self, result: dict[str, Any]) -> str:
        """Format execution result into analysis request message."""
        message = f"""Analyze this execution result and provide refinement suggestions:

Result: {result}

Please provide specific suggestions for improving the script based on this result."""

        return message

    def get_worker_count(self) -> int:
        """Get the current number of registered workers."""
        return len(self.workers)

    def get_healthy_worker_count(self) -> int:
        """Get the number of currently healthy workers."""
        return sum(1 for worker_info in self.workers.values() if worker_info["healthy"])

    def is_initialized(self) -> bool:
        """Check if the worker pool has been successfully initialized."""
        return self._initialized

    def shutdown(self):
        """Clean shutdown of worker pool."""
        logger.info("Shutting down SonnetWorkerPool")

        # Clear task queue
        with self._queue_lock:
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break
            self.worker_assignments.clear()

        self.workers.clear()
        self._initialized = False
