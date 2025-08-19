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
import time
import uuid
from typing import Any

from .claude_code_manager import ClaudeCodeManager

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

        logger.info("SonnetWorkerPool initialized")

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

    def assign_task(self, task: dict[str, Any]) -> str | None:
        """
        Assign a task to an available worker using round-robin load balancing.

        This method finds a healthy, available worker and assigns the task,
        returning the worker ID for tracking purposes. Uses simple round-robin
        selection for load balancing.

        Args:
            task: Task dictionary with objective and context

        Returns:
            Worker ID if task assigned successfully, None if no workers available

        Performance Requirements:
            - Task assignment: <50ms
        """
        if not self._initialized or not self.workers:
            logger.warning("Worker pool not initialized, cannot assign task")
            return None

        start_time = time.time()

        # Get list of available workers (healthy and ready)
        # Update cached health state before filtering
        for worker_info in self.workers.values():
            if worker_info["healthy"] and not worker_info["process"].is_healthy():
                worker_info["healthy"] = False  # Update stale cached state

        available_workers = [
            (worker_id, worker_info)
            for worker_id, worker_info in self.workers.items()
            if worker_info["healthy"]
        ]

        if not available_workers:
            logger.warning("No healthy workers available for task assignment")
            return None

        # Simple round-robin selection for load balancing
        worker_index = self._current_assignment_index % len(available_workers)
        selected_worker_id, selected_worker_info = available_workers[worker_index]
        self._current_assignment_index += 1

        # Update worker tracking
        selected_worker_info["status"] = "assigned"
        selected_worker_info["task_count"] += 1

        assignment_time = time.time() - start_time
        if assignment_time > 0.05:  # 50ms requirement
            logger.warning(f"Task assignment took {assignment_time:.3f}s, exceeds 50ms target")

        logger.info(f"Assigned task to worker {selected_worker_id}")
        return selected_worker_id

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
        self.workers.clear()
        self._initialized = False
