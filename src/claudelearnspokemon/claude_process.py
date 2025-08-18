"""
Simplified Claude CLI process wrapper focused on lifecycle management.

This module provides a clean, focused ClaudeProcess class that orchestrates
the various specialized components (communication, health monitoring, metrics, etc.)
following Clean Code principles with single responsibility focus.
"""

import logging
import subprocess
import threading
import time

from .process_communication import ProcessCommunicator
from .process_factory import ClaudeProcessFactory, ProcessConfig
from .process_health_monitor import ProcessHealthMonitor, ProcessState
from .process_metrics_collector import ProcessMetricsCollector
from .prompts import PromptRepository

logger = logging.getLogger(__name__)


class ClaudeProcess:
    """
    Clean, focused wrapper for Claude CLI subprocess with lifecycle management.

    This class now has a single responsibility: orchestrating the lifecycle
    of a Claude CLI process by coordinating specialized components.

    Responsibilities:
    - Process startup and shutdown
    - Component coordination and lifecycle management
    - Public interface for process operations
    """

    def __init__(self, config: ProcessConfig, process_id: int):
        """
        Initialize Claude process with specialized components.

        Args:
            config: Process configuration
            process_id: Unique identifier for this process
        """
        self.config = config
        self.process_id = process_id
        self.process: subprocess.Popen | None = None
        self._lock = threading.Lock()

        # Initialize specialized components
        self.metrics_collector = ProcessMetricsCollector(process_id)
        self.health_monitor = ProcessHealthMonitor(process_id, self.metrics_collector)
        self.communicator = ProcessCommunicator(
            process_id, config.stdout_buffer_size, config.stderr_buffer_size
        )

        # Factory for process creation
        self.factory = ClaudeProcessFactory()

        logger.info(f"Initialized Claude process {process_id} with {config.process_type.value}")

    def start(self) -> bool:
        """
        Start Claude CLI process with performance timing.

        Returns:
            True if process started successfully, False otherwise
        """
        start_time = time.time()

        try:
            with self._lock:
                if self.process is not None:
                    logger.warning(f"Process {self.process_id} already started")
                    return True

                # Create subprocess using factory
                self.process = self.factory.create_subprocess(self.config)

                # Record startup timing
                startup_duration = time.time() - start_time
                self.metrics_collector.record_startup_time(startup_duration)

                # Mark as healthy and send system prompt
                self.health_monitor.mark_as_healthy()
                self._initialize_process()

                logger.info(
                    f"Started {self.config.process_type.value} process {self.process_id} "
                    f"(PID: {self.process.pid}) in {startup_duration*1000:.1f}ms"
                )

                return True

        except Exception as e:
            logger.error(f"Failed to start process {self.process_id}: {e}")
            self.health_monitor.mark_as_failed()
            return False

    def _initialize_process(self):
        """Send system prompt to initialize the process."""
        if self.process:
            prompt = PromptRepository.get_prompt(self.config.process_type)
            self.communicator.send_system_prompt(self.process, prompt)

    def health_check(self) -> bool:
        """
        Perform health check on the process.

        Returns:
            True if process is healthy, False otherwise
        """
        return self.health_monitor.check_health(self.process)

    def send_message(self, message: str, timeout: float = 30.0) -> str | None:
        """
        Send message to Claude process and get response.

        Args:
            message: Message to send
            timeout: Timeout for the operation

        Returns:
            Response string if successful, None otherwise
        """
        if not self.process:
            logger.error(f"Process {self.process_id} not available for communication")
            return None

        return self.communicator.send_and_receive(self.process, message, timeout)

    def restart(self) -> bool:
        """
        Restart the Claude process with metrics tracking.

        Returns:
            True if restart was successful, False otherwise
        """
        logger.info(f"Restarting process {self.process_id}")

        self.terminate()
        self.metrics_collector.record_restart()

        # Brief pause to ensure clean shutdown
        time.sleep(0.1)

        return self.start()

    def terminate(self, timeout: float = 2.0):
        """
        Gracefully terminate the process with timeout.

        Args:
            timeout: Maximum time to wait for graceful termination
        """
        if self.process is None:
            return

        try:
            with self._lock:
                if self.process.poll() is not None:
                    return  # Already terminated

                # Close communication channels
                self.communicator.close_communication(self.process)

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
                self.health_monitor.mark_as_terminated()

        except Exception as e:
            logger.error(f"Error terminating process {self.process_id}: {e}")

    @property
    def state(self) -> ProcessState:
        """Get current process state."""
        return self.health_monitor.get_current_state()

    @property
    def metrics(self):
        """Get process metrics snapshot."""
        return self.metrics_collector.get_metrics_snapshot()

    def is_healthy(self) -> bool:
        """Check if process is currently healthy."""
        return self.health_monitor.is_healthy()

    def is_communication_available(self) -> bool:
        """Check if communication is available with the process."""
        if self.process is None:
            return False
        return self.communicator.is_communication_available(self.process)

    def get_performance_summary(self) -> dict:
        """Get human-readable performance summary."""
        return self.metrics_collector.get_performance_summary()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.terminate()

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ClaudeProcess(id={self.process_id}, "
            f"type={self.config.process_type.value}, "
            f"state={self.state.value}, "
            f"pid={self.process.pid if self.process else 'None'})"
        )
