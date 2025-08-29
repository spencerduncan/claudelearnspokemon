"""
Routing Integration - Integration Layer for Message Routing Engine

This module provides the integration layer between the new message routing engine
and existing ClaudeCodeManager/SonnetWorkerPool systems. It offers clean APIs
for the Pokemon speedrun learning agent to use intelligent message routing
without requiring changes to existing code.

Performance Requirements:
- Integration overhead: <5ms additional latency
- Backward compatibility: 100% with existing interfaces
- Memory efficiency: <10MB additional overhead
- Thread safety for concurrent usage

Google SRE Patterns Applied:
- Adapter pattern for seamless integration
- Feature flags for gradual rollout
- Comprehensive monitoring of integration points
- Graceful fallback to original routing when needed
"""

import logging
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

from .claude_code_manager import ClaudeCodeManager
from .message_router import MessageRouter, RoutingRequest, RoutingResult
from .priority_queue import MessagePriority
from .routing_strategy import create_least_loaded_strategy
from .sonnet_worker_pool import SonnetWorkerPool

logger = logging.getLogger(__name__)


class IntegrationMode(Enum):
    """Integration operation modes for gradual rollout."""

    DISABLED = "disabled"  # Use original routing only
    SHADOW = "shadow"  # Run new routing alongside original (metrics only)
    PARTIAL = "partial"  # Use new routing for subset of traffic
    FULL = "full"  # Use new routing for all traffic


@dataclass
class IntegrationConfig:
    """Configuration for routing integration behavior."""

    mode: IntegrationMode = IntegrationMode.FULL
    shadow_percentage: float = 10.0  # Percentage of traffic to shadow test
    partial_percentage: float = 50.0  # Percentage of traffic for partial mode
    enable_metrics: bool = True
    enable_tracing: bool = True
    fallback_on_error: bool = True
    max_routing_time_ms: float = 50.0  # SLA for routing decisions


class RoutingAdapter:
    """
    Adapter that provides intelligent routing while maintaining API compatibility.

    This adapter sits between existing code and the new routing engine, providing
    a seamless upgrade path with feature flags and monitoring.
    """

    def __init__(
        self,
        claude_manager: ClaudeCodeManager,
        worker_pool: SonnetWorkerPool,
        config: IntegrationConfig | None = None,
    ):
        """
        Initialize routing adapter with production configuration.

        Args:
            claude_manager: Existing ClaudeCodeManager instance
            worker_pool: Existing SonnetWorkerPool instance
            config: Integration configuration (defaults to full deployment)
        """
        self.claude_manager = claude_manager
        self.worker_pool = worker_pool
        self.config = config or IntegrationConfig()

        # Initialize routing components if enabled
        self.message_router: MessageRouter | None = None
        self._original_methods: dict[str, Any] = {}
        self._integration_active = False

        if self.config.mode != IntegrationMode.DISABLED:
            self._initialize_routing_engine()

        # Integration metrics
        self._lock = threading.Lock()
        self._metrics = {
            "requests_routed_originally": 0,
            "requests_routed_intelligently": 0,
            "routing_improvements": 0,
            "routing_fallbacks": 0,
            "integration_errors": 0,
            "avg_routing_time_improvement_ms": 0.0,
        }

        logger.info(f"RoutingAdapter initialized in {self.config.mode.value} mode")

    def _initialize_routing_engine(self) -> None:
        """Initialize the new routing engine components."""
        try:
            # Create routing strategy based on system characteristics
            routing_strategy = self._create_optimal_routing_strategy()

            # Initialize message router
            self.message_router = MessageRouter(
                claude_manager=self.claude_manager,
                worker_pool=self.worker_pool,
                routing_strategy=routing_strategy,
            )

            # Start the routing engine
            if self.message_router and self.message_router.start():
                self._integration_active = True
                logger.info("Routing engine initialized successfully")
            else:
                logger.error("Failed to start routing engine")
                self._integration_active = False

        except Exception as e:
            logger.error(f"Failed to initialize routing engine: {e}")
            self._integration_active = False

    def _create_optimal_routing_strategy(self):
        """Create routing strategy optimized for Pokemon speedrun workload."""
        # For Pokemon speedrun learning, we expect:
        # - High-frequency tactical requests (script development)
        # - Lower-frequency strategic requests (planning)
        # - Variable task complexity and duration

        # Use least-loaded strategy with prediction for variable workloads
        return create_least_loaded_strategy(prediction_factor=0.7)

    def route_strategic_request(
        self,
        request_content: Any,
        context: dict[str, Any] | None = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> str | None:
        """
        Route strategic planning request with intelligent routing.

        Args:
            request_content: Request content for strategic processing
            context: Additional context for routing decisions
            priority: Request priority level

        Returns:
            Worker ID if successfully routed, None otherwise
        """
        return self._route_with_intelligence(
            request_content=request_content,
            context=context,
            priority=priority,
            force_strategic=True,
            original_method=self._get_strategic_worker_originally,
        )

    def route_tactical_request(
        self,
        request_content: Any,
        context: dict[str, Any] | None = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> str | None:
        """
        Route tactical development request with intelligent routing.

        Args:
            request_content: Request content for tactical processing
            context: Additional context for routing decisions
            priority: Request priority level

        Returns:
            Worker ID if successfully routed, None otherwise
        """
        return self._route_with_intelligence(
            request_content=request_content,
            context=context,
            priority=priority,
            force_tactical=True,
            original_method=self._get_tactical_worker_originally,
        )

    def route_auto_request(
        self,
        request_content: Any,
        context: dict[str, Any] | None = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> str | None:
        """
        Route request with automatic strategic/tactical classification.

        Args:
            request_content: Request content to classify and route
            context: Additional context for classification and routing
            priority: Request priority level

        Returns:
            Worker ID if successfully routed, None otherwise
        """
        return self._route_with_intelligence(
            request_content=request_content,
            context=context,
            priority=priority,
            force_strategic=False,
            force_tactical=False,
            original_method=self._get_any_worker_originally,
        )

    def _route_with_intelligence(
        self,
        request_content: Any,
        context: dict[str, Any] | None,
        priority: MessagePriority,
        force_strategic: bool = False,
        force_tactical: bool = False,
        original_method: Callable | None = None,
    ) -> str | None:
        """
        Core routing logic with intelligence and fallback handling.

        Handles integration modes, shadow testing, and fallback scenarios.
        """
        start_time = time.time()

        try:
            # Check if intelligent routing should be used
            should_use_intelligent = self._should_use_intelligent_routing()

            # Route request intelligently if enabled
            intelligent_result = None
            if should_use_intelligent and self._integration_active and self.message_router:
                intelligent_result = self._route_intelligently(
                    request_content, context, priority, force_strategic, force_tactical
                )

            # Route using original method for comparison/fallback
            original_result = None
            if (
                self.config.mode in [IntegrationMode.SHADOW, IntegrationMode.DISABLED]
                or intelligent_result is None
                or not intelligent_result.success
            ):

                original_result = self._route_originally(original_method)

            # Determine final result based on integration mode
            final_result = self._determine_final_result(
                intelligent_result, original_result, start_time
            )

            # Update integration metrics
            self._update_integration_metrics(intelligent_result, original_result, start_time)

            return final_result

        except Exception as e:
            logger.error(f"Integration routing failed: {e}")
            self._metrics["integration_errors"] += 1

            # Fallback to original routing on error
            if self.config.fallback_on_error and original_method:
                return self._route_originally(original_method)
            return None

    def _should_use_intelligent_routing(self) -> bool:
        """Determine if intelligent routing should be used based on config."""
        import random

        if self.config.mode == IntegrationMode.DISABLED:
            return False
        elif self.config.mode == IntegrationMode.FULL:
            return True
        elif self.config.mode == IntegrationMode.PARTIAL:
            return random.random() < (self.config.partial_percentage / 100.0)
        elif self.config.mode == IntegrationMode.SHADOW:
            return random.random() < (self.config.shadow_percentage / 100.0)

        return False

    def _route_intelligently(
        self,
        request_content: Any,
        context: dict[str, Any] | None,
        priority: MessagePriority,
        force_strategic: bool,
        force_tactical: bool,
    ) -> RoutingResult | None:
        """Route request using intelligent routing engine."""
        routing_request = RoutingRequest(
            content=request_content,
            priority=priority,
            context=context or {},
            require_strategic=force_strategic,
            require_tactical=force_tactical,
            timeout_seconds=self.config.max_routing_time_ms / 1000.0,
        )

        if not self.message_router:
            return None
        return self.message_router.route_message(routing_request)

    def _route_originally(self, original_method: Callable | None) -> str | None:
        """Route using original method for comparison/fallback."""
        if not original_method:
            return None

        try:
            return original_method()
        except Exception as e:
            logger.error(f"Original routing method failed: {e}")
            return None

    def _get_strategic_worker_originally(self) -> str | None:
        """Get strategic worker using original method."""
        process = self.claude_manager.get_strategic_process()
        if process and process.is_healthy():
            return f"strategic_{process.process_id}"
        return None

    def _get_tactical_worker_originally(self) -> str | None:
        """Get tactical worker using original method."""
        # Use SonnetWorkerPool's assign_task method
        dummy_task = {"objective": "route_request", "context": {}}
        worker_id = self.worker_pool.assign_task(dummy_task)
        return worker_id

    def _get_any_worker_originally(self) -> str | None:
        """Get any available worker using original method."""
        # Try tactical first (more workers available)
        tactical_result = self._get_tactical_worker_originally()
        if tactical_result:
            return tactical_result

        # Fall back to strategic if no tactical available
        return self._get_strategic_worker_originally()

    def _determine_final_result(
        self,
        intelligent_result: RoutingResult | None,
        original_result: str | None,
        start_time: float,
    ) -> str | None:
        """Determine final routing result based on integration mode."""
        # Check SLA compliance for intelligent routing
        intelligent_within_sla = (
            intelligent_result
            and intelligent_result.total_time_ms <= self.config.max_routing_time_ms
        )

        if self.config.mode == IntegrationMode.DISABLED:
            return original_result

        elif self.config.mode == IntegrationMode.SHADOW:
            # Always use original result in shadow mode (just collect metrics)
            return original_result

        elif self.config.mode == IntegrationMode.PARTIAL:
            # Use intelligent result if available and within SLA, otherwise original
            if intelligent_result and intelligent_result.success and intelligent_within_sla:
                return intelligent_result.worker_id
            else:
                return original_result

        elif self.config.mode == IntegrationMode.FULL:
            # Use intelligent result preferentially, fallback to original
            if intelligent_result and intelligent_result.success and intelligent_within_sla:
                return intelligent_result.worker_id
            elif original_result:
                logger.warning("Intelligent routing failed or exceeded SLA, using original")
                return original_result
            else:
                return None

        return original_result

    def _update_integration_metrics(
        self,
        intelligent_result: RoutingResult | None,
        original_result: str | None,
        start_time: float,
    ) -> None:
        """Update integration performance metrics."""
        with self._lock:
            if original_result:
                self._metrics["requests_routed_originally"] += 1

            if intelligent_result and intelligent_result.success:
                self._metrics["requests_routed_intelligently"] += 1

                # Check if intelligent routing provided improvement
                if self._is_routing_improvement(intelligent_result, original_result):
                    self._metrics["routing_improvements"] += 1

            if intelligent_result and not intelligent_result.success and original_result:
                self._metrics["routing_fallbacks"] += 1

            # Update time improvement metrics
            if intelligent_result and original_result:
                # In a full implementation, we'd track original routing time too
                # For now, assume baseline improvement based on SLA compliance
                if intelligent_result.total_time_ms <= self.config.max_routing_time_ms:
                    improvement = max(
                        0, self.config.max_routing_time_ms - intelligent_result.total_time_ms
                    )

                    # Update running average
                    current_count = self._metrics["routing_improvements"]
                    current_avg = self._metrics["avg_routing_time_improvement_ms"]

                    if current_count > 0:
                        self._metrics["avg_routing_time_improvement_ms"] = (
                            current_avg * (current_count - 1) + improvement
                        ) / current_count
                    else:
                        self._metrics["avg_routing_time_improvement_ms"] = improvement

    def _is_routing_improvement(
        self, intelligent_result: RoutingResult, original_result: str | None
    ) -> bool:
        """Determine if intelligent routing provided improvement over original."""
        if not intelligent_result.success or not original_result:
            return False

        # Consider it an improvement if:
        # 1. Routing completed within SLA
        # 2. Used appropriate worker type based on classification
        # 3. Load balancing was applied (for tactical requests)

        within_sla = intelligent_result.total_time_ms <= self.config.max_routing_time_ms
        has_classification = intelligent_result.classification_result is not None

        return within_sla and (has_classification or bool(intelligent_result.worker_type))

    @contextmanager
    def performance_monitoring(self, operation_name: str):
        """Context manager for performance monitoring of integration operations."""
        start_time = time.time()
        operation_id = str(uuid4())

        logger.debug(f"Starting {operation_name} (ID: {operation_id})")

        try:
            yield operation_id
        finally:
            duration = (time.time() - start_time) * 1000
            logger.debug(f"Completed {operation_name} (ID: {operation_id}) in {duration:.2f}ms")

            if duration > self.config.max_routing_time_ms:
                logger.warning(
                    f"{operation_name} took {duration:.2f}ms, "
                    f"exceeds SLA of {self.config.max_routing_time_ms}ms"
                )

    def get_integration_health(self) -> dict[str, Any]:
        """Get comprehensive integration health status."""
        base_health = {
            "integration_mode": self.config.mode.value,
            "routing_engine_active": self._integration_active,
            "metrics": self._metrics.copy(),
        }

        if self.message_router:
            base_health["router_health"] = self.message_router.get_health_status()

        return base_health

    def update_integration_config(self, new_config: IntegrationConfig) -> bool:
        """
        Update integration configuration dynamically.

        Args:
            new_config: New configuration to apply

        Returns:
            True if update successful, False otherwise
        """
        try:
            old_mode = self.config.mode
            self.config = new_config

            # Handle mode changes
            if old_mode != new_config.mode:
                logger.info(
                    f"Integration mode changed from {old_mode.value} to {new_config.mode.value}"
                )

                # Initialize routing engine if moving from disabled to enabled
                if (
                    old_mode == IntegrationMode.DISABLED
                    and new_config.mode != IntegrationMode.DISABLED
                ):
                    self._initialize_routing_engine()

                # Shutdown routing engine if moving to disabled
                elif new_config.mode == IntegrationMode.DISABLED and self.message_router:
                    self.message_router.shutdown()
                    self._integration_active = False

            return True

        except Exception as e:
            logger.error(f"Failed to update integration config: {e}")
            return False

    def shutdown(self) -> None:
        """Graceful shutdown of routing integration."""
        logger.info("Shutting down routing integration...")

        if self.message_router:
            self.message_router.shutdown()

        self._integration_active = False
        logger.info("Routing integration shutdown complete")


# Factory functions for common integration configurations
def create_production_adapter(
    claude_manager: ClaudeCodeManager, worker_pool: SonnetWorkerPool
) -> RoutingAdapter:
    """Create production-ready routing adapter."""
    config = IntegrationConfig(
        mode=IntegrationMode.FULL,
        enable_metrics=True,
        enable_tracing=True,
        fallback_on_error=True,
        max_routing_time_ms=50.0,
    )
    return RoutingAdapter(claude_manager, worker_pool, config)


def create_shadow_testing_adapter(
    claude_manager: ClaudeCodeManager,
    worker_pool: SonnetWorkerPool,
    shadow_percentage: float = 10.0,
) -> RoutingAdapter:
    """Create adapter for shadow testing new routing engine."""
    config = IntegrationConfig(
        mode=IntegrationMode.SHADOW,
        shadow_percentage=shadow_percentage,
        enable_metrics=True,
        enable_tracing=True,
        fallback_on_error=True,
        max_routing_time_ms=50.0,
    )
    return RoutingAdapter(claude_manager, worker_pool, config)


def create_gradual_rollout_adapter(
    claude_manager: ClaudeCodeManager,
    worker_pool: SonnetWorkerPool,
    rollout_percentage: float = 50.0,
) -> RoutingAdapter:
    """Create adapter for gradual rollout of intelligent routing."""
    config = IntegrationConfig(
        mode=IntegrationMode.PARTIAL,
        partial_percentage=rollout_percentage,
        enable_metrics=True,
        enable_tracing=True,
        fallback_on_error=True,
        max_routing_time_ms=50.0,
    )
    return RoutingAdapter(claude_manager, worker_pool, config)
