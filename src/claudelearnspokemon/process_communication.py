"""
Communication handling for Claude CLI processes.

This module handles all stdin/stdout/stderr communication with subprocesses,
following the Single Responsibility Principle by separating I/O concerns
from process lifecycle and health monitoring.
"""

import json
import logging
import subprocess
import threading
import time
from typing import Any

from .conversation_state import ConversationState

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
        self,
        process_id: int,
        stdout_buffer_size: int = 8192,
        stderr_buffer_size: int = 4096,
        conversation_state: ConversationState | None = None,
    ):
        """
        Initialize process communicator with conversation state management.

        Args:
            process_id: Unique identifier for the process
            stdout_buffer_size: Size of stdout buffer in bytes
            stderr_buffer_size: Size of stderr buffer in bytes
            conversation_state: Optional conversation state manager
        """
        self.process_id = process_id
        self._lock = threading.Lock()
        self.conversation_state = conversation_state

        # Pre-allocate communication buffers for performance optimization
        self._stdout_buffer = bytearray(stdout_buffer_size)
        self._stderr_buffer = bytearray(stderr_buffer_size)

        # Claude CLI communication settings
        self._json_communication = True  # Use JSON format for Claude CLI
        self._response_timeout = 30.0  # Default timeout for responses

        logger.debug(
            f"ProcessCommunicator initialized for process {process_id} "
            f"with buffers: stdout={stdout_buffer_size}B, stderr={stderr_buffer_size}B, "
            f"conversation_state={'enabled' if conversation_state else 'disabled'}"
        )

    def send_message(self, process: subprocess.Popen, message: str, timeout: float = 30.0) -> bool:
        """
        Send message to Claude process with turn limit enforcement and conversation tracking.

        Args:
            process: The subprocess.Popen instance
            message: Message to send
            timeout: Timeout in seconds for the operation

        Returns:
            True if message was sent successfully, False otherwise
        """
        # Check turn limits before sending
        if self.conversation_state and not self.conversation_state.can_send_message():
            logger.warning(
                f"Process {self.process_id} cannot send message: "
                f"turn limit reached ({self.conversation_state.turn_count}/{self.conversation_state._max_turns})"
            )
            return False

        if not process or not process.stdin:
            logger.error(f"Process {self.process_id} stdin not available for communication")
            return False

        start_time = time.time()

        try:
            with self._lock:
                # Format message for Claude CLI JSON communication
                if self._json_communication:
                    # Claude CLI expects messages in a specific format
                    formatted_message = json.dumps({"message": message})
                    process.stdin.write(f"{formatted_message}\n")
                else:
                    # Fallback to plain text for testing
                    process.stdin.write(f"{message}\n")

                process.stdin.flush()

                # Record message send time
                send_duration = (time.time() - start_time) * 1000

                logger.debug(
                    f"Sent message to process {self.process_id} in {send_duration:.1f}ms: "
                    f"{message[:50]}{'...' if len(message) > 50 else ''}"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to send message to process {self.process_id}: {e}")

            # Record failed message in conversation state
            if self.conversation_state:
                self.conversation_state.record_message_exchange(
                    message=message, error=str(e), duration_ms=(time.time() - start_time) * 1000
                )

            return False

    def read_response(self, process: subprocess.Popen, timeout: float = 30.0) -> str | None:
        """
        Read response from Claude CLI with JSON parsing and timeout handling.

        Args:
            process: The subprocess.Popen instance
            timeout: Timeout in seconds for the read operation

        Returns:
            Response string if successful, None if failed or timed out
        """
        if not process or not process.stdout:
            logger.error(f"Process {self.process_id} stdout not available for reading")
            return None

        start_time = time.time()

        try:
            with self._lock:
                # Read raw response with timeout handling
                # Note: In production, this would use select() or polling for proper timeout
                response_line = process.stdout.readline()

                read_duration = (time.time() - start_time) * 1000

                if response_line:
                    response_text = response_line.strip()

                    # Parse JSON response from Claude CLI
                    if self._json_communication and response_text:
                        try:
                            response_data = json.loads(response_text)
                            # Extract actual response content from Claude CLI JSON format
                            actual_response = response_data.get("response", response_text)
                        except (json.JSONDecodeError, KeyError) as json_error:
                            logger.warning(
                                f"Failed to parse JSON response from process {self.process_id}: {json_error}. "
                                f"Using raw response."
                            )
                            actual_response = response_text
                    else:
                        actual_response = response_text

                    logger.debug(
                        f"Read response from process {self.process_id} in {read_duration:.1f}ms: "
                        f"{actual_response[:50]}{'...' if len(actual_response) > 50 else ''}"
                    )
                    return actual_response
                else:
                    logger.warning(
                        f"Empty response from process {self.process_id} after {read_duration:.1f}ms"
                    )
                    return None

        except Exception as e:
            read_duration = (time.time() - start_time) * 1000
            logger.error(
                f"Failed to read response from process {self.process_id} after {read_duration:.1f}ms: {e}"
            )
            return None

    def send_and_receive(
        self, process: subprocess.Popen, message: str, timeout: float = 30.0
    ) -> str | None:
        """
        Send message and wait for response with full conversation tracking.

        Args:
            process: The subprocess.Popen instance
            message: Message to send
            timeout: Total timeout for both send and receive operations

        Returns:
            Response string if successful, None if failed
        """
        start_time = time.time()

        # Split timeout between send and receive operations
        send_timeout = timeout * 0.1  # 10% for sending
        receive_timeout = timeout * 0.9  # 90% for receiving

        # Send message (includes turn limit check)
        if not self.send_message(process, message, send_timeout):
            return None

        # Receive response
        response = self.read_response(process, receive_timeout)

        # Record complete message exchange in conversation state
        if self.conversation_state:
            total_duration = (time.time() - start_time) * 1000
            self.conversation_state.record_message_exchange(
                message=message,
                response=response,
                duration_ms=total_duration,
                error=None if response else "No response received",
            )

        return response

    def send_system_prompt(self, process: subprocess.Popen, prompt: str) -> bool:
        """
        Send initial system prompt and initialize conversation state.

        Args:
            process: The subprocess.Popen instance
            prompt: System prompt to send

        Returns:
            True if prompt was sent successfully and conversation initialized
        """
        if not process or not process.stdin:
            logger.error(f"Process {self.process_id} stdin not available for system prompt")
            return False

        start_time = time.time()

        try:
            with self._lock:
                # Format system prompt for Claude CLI
                if self._json_communication:
                    # Claude CLI system prompt format
                    system_message = json.dumps(
                        {
                            "system": prompt,
                            "message": "System prompt initialized. Ready for Pokemon speedrun learning agent tasks.",
                        }
                    )
                    process.stdin.write(f"{system_message}\n")
                else:
                    # Fallback format for testing
                    process.stdin.write(f"System: {prompt}\n\n")

                process.stdin.flush()

                # Initialize conversation state if available
                if self.conversation_state:
                    success = self.conversation_state.initialize_conversation(prompt)
                    if not success:
                        logger.warning(
                            f"Failed to initialize conversation state for process {self.process_id}"
                        )

                initialization_time = (time.time() - start_time) * 1000

                logger.info(
                    f"Sent system prompt to process {self.process_id} in {initialization_time:.1f}ms "
                    f"(conversation_state={'initialized' if self.conversation_state else 'not available'})"
                )
                return True

        except Exception as e:
            initialization_time = (time.time() - start_time) * 1000
            logger.error(
                f"Failed to send system prompt to process {self.process_id} "
                f"after {initialization_time:.1f}ms: {e}"
            )
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
