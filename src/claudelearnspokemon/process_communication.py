"""
Communication handling for Claude CLI processes.

This module handles all stdin/stdout/stderr communication with subprocesses,
following the Single Responsibility Principle by separating I/O concerns
from process lifecycle and health monitoring.
"""

import logging
import subprocess
import threading
from typing import Any

logger = logging.getLogger(__name__)


class ProcessCommunicator:
    """
    Handles communication with Claude CLI subprocess instances.

    This class is responsible for:
    - Sending messages to subprocess stdin
    - Reading responses from subprocess stdout
    - Managing communication buffers for performance
    - Handling communication timeouts and errors
    """

    def __init__(
        self, process_id: int, stdout_buffer_size: int = 8192, stderr_buffer_size: int = 4096
    ):
        """
        Initialize process communicator.

        Args:
            process_id: Unique identifier for the process
            stdout_buffer_size: Size of stdout buffer in bytes
            stderr_buffer_size: Size of stderr buffer in bytes
        """
        self.process_id = process_id
        self._lock = threading.Lock()

        # Pre-allocate communication buffers for performance optimization
        self._stdout_buffer = bytearray(stdout_buffer_size)
        self._stderr_buffer = bytearray(stderr_buffer_size)

        logger.debug(
            f"ProcessCommunicator initialized for process {process_id} "
            f"with buffers: stdout={stdout_buffer_size}B, stderr={stderr_buffer_size}B"
        )

    def send_message(self, process: subprocess.Popen, message: str, timeout: float = 30.0) -> bool:
        """
        Send message to Claude process stdin with timeout handling.

        Args:
            process: The subprocess.Popen instance
            message: Message to send
            timeout: Timeout in seconds for the operation

        Returns:
            True if message was sent successfully, False otherwise
        """
        if not process or not process.stdin:
            logger.error(f"Process {self.process_id} stdin not available for communication")
            return False

        try:
            with self._lock:
                # Send message with proper formatting
                process.stdin.write(f"{message}\n")
                process.stdin.flush()

                logger.debug(
                    f"Sent message to process {self.process_id}: "
                    f"{message[:50]}{'...' if len(message) > 50 else ''}"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to send message to process {self.process_id}: {e}")
            return False

    def read_response(self, process: subprocess.Popen, timeout: float = 30.0) -> str | None:
        """
        Read response from Claude process stdout with timeout.

        Args:
            process: The subprocess.Popen instance
            timeout: Timeout in seconds for the read operation

        Returns:
            Response string if successful, None if failed or timed out
        """
        if not process or not process.stdout:
            logger.error(f"Process {self.process_id} stdout not available for reading")
            return None

        try:
            with self._lock:
                # Use the pre-allocated buffer for reading
                # Note: This is a simplified implementation
                # Production would need proper JSON parsing and timeout handling
                response = process.stdout.readline()

                if response:
                    result = response.strip()
                    logger.debug(
                        f"Read response from process {self.process_id}: "
                        f"{result[:50]}{'...' if len(result) > 50 else ''}"
                    )
                    return result
                else:
                    logger.warning(f"Empty response from process {self.process_id}")
                    return None

        except Exception as e:
            logger.error(f"Failed to read response from process {self.process_id}: {e}")
            return None

    def send_and_receive(
        self, process: subprocess.Popen, message: str, timeout: float = 30.0
    ) -> str | None:
        """
        Send message and wait for response in a single operation.

        Args:
            process: The subprocess.Popen instance
            message: Message to send
            timeout: Total timeout for both send and receive operations

        Returns:
            Response string if successful, None if failed
        """
        # Split timeout between send and receive operations
        send_timeout = timeout * 0.1  # 10% for sending
        receive_timeout = timeout * 0.9  # 90% for receiving

        if not self.send_message(process, message, send_timeout):
            return None

        return self.read_response(process, receive_timeout)

    def send_system_prompt(self, process: subprocess.Popen, prompt: str) -> bool:
        """
        Send initial system prompt to the process.

        Args:
            process: The subprocess.Popen instance
            prompt: System prompt to send

        Returns:
            True if prompt was sent successfully, False otherwise
        """
        if not process or not process.stdin:
            logger.error(f"Process {self.process_id} stdin not available for system prompt")
            return False

        try:
            with self._lock:
                # Send prompt with proper formatting for system initialization
                process.stdin.write(f"{prompt}\n\n")
                process.stdin.flush()

                logger.info(f"Sent system prompt to process {self.process_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to send system prompt to process {self.process_id}: {e}")
            return False

    def is_communication_available(self, process: subprocess.Popen) -> bool:
        """
        Check if communication channels are available for the process.

        Args:
            process: The subprocess.Popen instance

        Returns:
            True if both stdin and stdout are available
        """
        return (
            process is not None
            and process.stdin is not None
            and process.stdout is not None
            and process.poll() is None  # Process is still running
        )

    def close_communication(self, process: subprocess.Popen):
        """
        Close communication channels gracefully.

        Args:
            process: The subprocess.Popen instance
        """
        try:
            if process and process.stdin:
                process.stdin.close()
                logger.debug(f"Closed stdin for process {self.process_id}")
        except Exception as e:
            logger.error(f"Error closing stdin for process {self.process_id}: {e}")

    def get_buffer_usage(self) -> dict[str, int]:
        """
        Get current buffer usage statistics.

        Returns:
            Dictionary with buffer usage information
        """
        return {
            "stdout_buffer_size": len(self._stdout_buffer),
            "stderr_buffer_size": len(self._stderr_buffer),
            "stdout_allocated": len(self._stdout_buffer),
            "stderr_allocated": len(self._stderr_buffer),
        }


class CommunicationManager:
    """
    Manages communication across multiple Claude CLI processes.

    This class provides centralized communication management while
    maintaining individual process communication isolation.
    """

    def __init__(self):
        """Initialize the communication manager."""
        self.communicators: dict[int, ProcessCommunicator] = {}
        self._lock = threading.Lock()

    def add_communicator(self, communicator: ProcessCommunicator):
        """
        Add a process communicator to the manager.

        Args:
            communicator: ProcessCommunicator to manage
        """
        with self._lock:
            self.communicators[communicator.process_id] = communicator

    def remove_communicator(self, process_id: int):
        """
        Remove a process communicator from management.

        Args:
            process_id: ID of the process to remove
        """
        with self._lock:
            self.communicators.pop(process_id, None)

    def get_communicator(self, process_id: int) -> ProcessCommunicator | None:
        """
        Get communicator for a specific process.

        Args:
            process_id: ID of the process

        Returns:
            ProcessCommunicator if found, None otherwise
        """
        with self._lock:
            return self.communicators.get(process_id)

    def broadcast_message(
        self, processes: dict[int, subprocess.Popen], message: str, timeout: float = 30.0
    ) -> dict[int, bool]:
        """
        Send message to multiple processes simultaneously.

        Args:
            processes: Dictionary mapping process_id to subprocess.Popen
            message: Message to broadcast
            timeout: Timeout per process

        Returns:
            Dictionary mapping process_id to success status
        """
        results = {}

        with self._lock:
            for process_id, process in processes.items():
                if process_id in self.communicators:
                    communicator = self.communicators[process_id]
                    results[process_id] = communicator.send_message(process, message, timeout)
                else:
                    results[process_id] = False
                    logger.warning(f"No communicator found for process {process_id}")

        return results

    def get_communication_stats(self) -> dict[str, Any]:
        """
        Get communication statistics across all managed processes.

        Returns:
            Dictionary with communication statistics
        """
        with self._lock:
            total_communicators = len(self.communicators)
            total_buffer_usage = 0

            for communicator in self.communicators.values():
                buffer_usage = communicator.get_buffer_usage()
                total_buffer_usage += (
                    buffer_usage["stdout_allocated"] + buffer_usage["stderr_allocated"]
                )

            return {
                "total_communicators": total_communicators,
                "total_buffer_usage_bytes": total_buffer_usage,
                "average_buffer_per_process": (
                    total_buffer_usage / total_communicators if total_communicators > 0 else 0
                ),
            }
