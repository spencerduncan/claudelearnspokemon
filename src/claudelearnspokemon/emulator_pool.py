"""
EmulatorPool: Docker container lifecycle management for Pokemon-gym emulators.

Production-grade container orchestration with proper failure handling,
timeouts, and resource cleanup. Built with Google-scale engineering principles.

Author: Bot Dean - Production Systems Engineering
"""

import logging
import time

import docker
from docker.errors import APIError, DockerException, ImageNotFound

# Configure logging for production observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmulatorPoolError(Exception):
    """
    Custom exception for EmulatorPool operations.

    Provides actionable error messages for production debugging.
    """

    pass


class EmulatorPool:
    """
    Manages a pool of Pokemon-gym Docker containers for parallel execution.

    Implements production patterns:
    - Graceful failure handling
    - Resource cleanup on exceptions
    - Timeout-based operations
    - Comprehensive logging for debugging
    - Idempotent operations
    """

    def __init__(
        self,
        pool_size: int = 4,
        base_port: int = 8081,
        image_name: str = "pokemon-gym:latest",
        startup_timeout: int = 30,
    ):
        """
        Initialize EmulatorPool with production configuration.

        Args:
            pool_size: Number of containers in pool (default: 4)
            base_port: Starting port for sequential allocation (default: 8081)
            image_name: Docker image name for containers (default: pokemon-gym:latest)
            startup_timeout: Max seconds to wait for container startup (default: 30)
        """
        self.pool_size = pool_size
        self.base_port = base_port
        self.image_name = image_name
        self.startup_timeout = startup_timeout

        # Container management state
        self.containers: list[docker.models.containers.Container] = []
        self.client: docker.DockerClient | None = None

        logger.info(
            f"EmulatorPool configured: size={pool_size}, base_port={base_port}, "
            f"image={image_name}, timeout={startup_timeout}s"
        )

    def initialize(self, pool_size: int | None = None) -> None:
        """
        Initialize the container pool with production-grade error handling.

        Args:
            pool_size: Override default pool size if provided

        Raises:
            EmulatorPoolError: On any initialization failure with actionable message
        """
        if pool_size is not None:
            self.pool_size = pool_size

        logger.info(f"Initializing EmulatorPool with {self.pool_size} containers")

        try:
            # Connect to Docker daemon with proper error handling
            self.client = docker.from_env()
            logger.info("Connected to Docker daemon")
        except DockerException as e:
            raise EmulatorPoolError(
                f"Docker daemon unavailable: {e}. "
                f"Ensure Docker daemon is running and accessible."
            ) from e

        # Track successfully started containers for cleanup on failure
        started_containers = []

        try:
            ports = self._get_container_ports(self.pool_size)

            for port in ports:
                logger.info(f"Starting container on port {port}")

                try:
                    container = self._start_single_container(port)
                    started_containers.append(container)
                    self.containers.append(container)

                    logger.info(f"Container {container.id} started successfully on port {port}")

                except Exception as e:
                    # Clean up successfully started containers before re-raising
                    logger.error(f"Failed to start container on port {port}: {e}")
                    self._cleanup_containers(started_containers)
                    raise EmulatorPoolError(
                        f"Failed to start container on port {port}: {e}. "
                        f"Check port availability and resource limits."
                    ) from e

        except EmulatorPoolError:
            # Already handled, just re-raise
            raise
        except Exception as e:
            # Unexpected error - clean up and provide debugging info
            logger.error(f"Unexpected error during initialization: {e}")
            self._cleanup_containers(started_containers)
            raise EmulatorPoolError(
                f"Unexpected error during pool initialization: {e}. "
                f"Check Docker daemon status and system resources."
            ) from e

        logger.info(f"EmulatorPool initialized successfully with {len(self.containers)} containers")

    def shutdown(self) -> None:
        """
        Gracefully shutdown all containers with production-grade error handling.

        Continues shutdown process even if individual containers fail to stop.
        Implements idempotent operation - safe to call multiple times.
        """
        if not self.containers:
            logger.info("EmulatorPool shutdown called - no containers to stop")
            return

        logger.info(f"Shutting down EmulatorPool with {len(self.containers)} containers")

        # Track shutdown results for logging
        shutdown_results = {"success": 0, "failed": 0}

        for container in self.containers:
            try:
                logger.info(f"Stopping container {container.id}")
                container.stop(timeout=10)  # Give containers time to gracefully stop
                shutdown_results["success"] += 1
                logger.info(f"Container {container.id} stopped successfully")

            except Exception as e:
                # Log error but continue with other containers
                logger.error(f"Failed to stop container {container.id}: {e}")
                shutdown_results["failed"] += 1

        # Clear container list - idempotent operation
        self.containers.clear()

        logger.info(
            f"EmulatorPool shutdown complete: {shutdown_results['success']} success, "
            f"{shutdown_results['failed']} failed"
        )

    def _start_single_container(self, port: int) -> docker.models.containers.Container:
        """
        Start a single container with production configuration.

        Args:
            port: Host port for container mapping

        Returns:
            Started and health-checked container

        Raises:
            EmulatorPoolError: On startup or health check failure
        """
        try:
            if not self.client:
                raise EmulatorPoolError("Docker client not initialized. Call initialize() first.")
            container = self.client.containers.run(
                image=self.image_name,
                ports={"8080/tcp": port},  # Map internal port 8080 to host port
                detach=True,
                remove=True,  # Auto-cleanup on container stop
                name=f"pokemon-emulator-{port}",
                # Production container configuration
                restart_policy={"Name": "on-failure", "MaximumRetryCount": 3},
                mem_limit="512m",  # Prevent memory exhaustion
                cpu_count=1,  # Fair CPU allocation
            )

        except ImageNotFound as e:
            raise EmulatorPoolError(
                f"Pokemon-gym image not found: {self.image_name}. "
                f"Ensure image is built and available locally."
            ) from e
        except APIError as e:
            raise EmulatorPoolError(
                f"Docker API error starting container: {e}. "
                f"Check port {port} availability and system resources."
            ) from e

        # Wait for container to become ready with timeout
        self._wait_for_container_ready(container)

        # Perform health check
        self._verify_container_health(container)

        return container

    def _wait_for_container_ready(self, container: docker.models.containers.Container) -> None:
        """
        Wait for container to reach running state with production timeout.

        Args:
            container: Container to wait for

        Raises:
            EmulatorPoolError: On timeout or container failure
        """
        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:
            container.reload()  # Refresh container status

            if container.status == "running":
                return
            elif container.status in ["exited", "dead"]:
                raise EmulatorPoolError(
                    f"Container {container.id} failed to start (status: {container.status}). "
                    f"Check container logs for details."
                )

            time.sleep(0.5)  # Poll interval

        # Timeout reached
        raise EmulatorPoolError(
            f"Container startup timeout ({self.startup_timeout}s) exceeded. "
            f"Container {container.id} status: {container.status}. "
            f"Increase timeout or check system performance."
        )

    def _verify_container_health(self, container: docker.models.containers.Container) -> None:
        """
        Verify container is healthy and responding.

        Args:
            container: Container to health check

        Raises:
            EmulatorPoolError: On health check failure
        """
        try:
            # Simple health check - verify container can execute commands
            result = container.exec_run("echo 'health_check'")

            if result.exit_code != 0:
                raise EmulatorPoolError(
                    f"Container health check failed for {container.id}. "
                    f"Exit code: {result.exit_code}, output: {result.output}"
                )

        except Exception as e:
            raise EmulatorPoolError(
                f"Container health check failed for {container.id}: {e}. "
                f"Container may not be fully initialized."
            ) from e

    def _get_container_ports(self, count: int) -> list[int]:
        """
        Calculate sequential port numbers for containers.

        Args:
            count: Number of ports needed

        Returns:
            List of sequential port numbers
        """
        return [self.base_port + i for i in range(count)]

    def _cleanup_containers(self, containers: list[docker.models.containers.Container]) -> None:
        """
        Clean up containers on failure - prevent resource leaks.

        Args:
            containers: List of containers to stop and clean up
        """
        if not containers:
            return

        logger.info(f"Cleaning up {len(containers)} containers due to initialization failure")

        for container in containers:
            try:
                container.stop(timeout=5)
                logger.info(f"Cleaned up container {container.id}")
            except Exception as e:
                # Log but don't raise - we're already in error handling
                logger.error(f"Failed to cleanup container {container.id}: {e}")

    def get_container_status(self) -> dict:
        """
        Get current status of all containers for monitoring.

        Returns:
            Dictionary with container status information
        """
        if not self.containers:
            return {"total": 0, "running": 0, "status": "not_initialized"}

        statuses = []
        running_count = 0

        for container in self.containers:
            try:
                container.reload()
                status = container.status
                if status == "running":
                    running_count += 1

                statuses.append(
                    {
                        "id": (
                            container.id[:12] if container.id else "unknown"
                        ),  # Short ID for readability
                        "status": status,
                        "name": container.name,
                    }
                )
            except Exception as e:
                statuses.append(
                    {
                        "id": (
                            container.id[:12]
                            if hasattr(container, "id") and container.id
                            else "unknown"
                        ),
                        "status": "error",
                        "error": str(e),
                    }
                )

        return {
            "total": len(self.containers),
            "running": running_count,
            "containers": statuses,
            "status": "healthy" if running_count == len(self.containers) else "degraded",
        }
