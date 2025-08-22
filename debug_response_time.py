#!/usr/bin/env python3
"""Debug script to understand response time tracking issue."""

import sys

sys.path.insert(0, "src")

from unittest.mock import Mock

from claudelearnspokemon.priority_queue import MessagePriority, QueuedMessage
from claudelearnspokemon.routing_strategy import LeastLoadedStrategy, WorkerInfo, WorkerState


def debug_response_time():
    """Debug the response time tracking."""
    print("=== Debugging Response Time Tracking ===")

    strategy = LeastLoadedStrategy(load_prediction_factor=0.7)

    # Create mock workers as in test (mimic the test setup exactly)
    workers = []
    loads = [2, 5, 1, 8]  # Different current loads

    for i, load in enumerate(loads):
        mock_process = Mock()
        worker = WorkerInfo(
            worker_id=f"worker_{i}",
            process=mock_process,
            state=WorkerState.HEALTHY,
            current_load=load,
            max_capacity=10,
            response_time_ms=100.0 + (load * 20),  # Slower with higher load
        )
        workers.append(worker)

    print(
        f"Created workers: {[(w.worker_id, w.current_load, w.response_time_ms) for w in workers]}"
    )

    worker_0 = workers[0]
    print(f"Initial worker_0: response_time={worker_0.response_time_ms}")

    # First, make the worker known to the strategy by selecting it
    message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)
    selected = strategy.select_worker(message, workers)
    print(f"Selected worker: {selected}")

    # Check strategy's internal state
    if "worker_0" in strategy._worker_states:
        internal_worker = strategy._worker_states["worker_0"]
        print(f"After select - internal worker response_time: {internal_worker.response_time_ms}")
    else:
        print("worker_0 not found in strategy._worker_states")

    # Update with faster response time
    print("Updating worker_0 with response_time=50.0")
    strategy.update_worker_state("worker_0", WorkerState.HEALTHY, 50.0)

    # Check updated state
    updated_worker = strategy._worker_states["worker_0"]
    print(f"After update - response_time: {updated_worker.response_time_ms}")
    print("Expected: between 50.0 and 140.0 (exponential moving average)")


if __name__ == "__main__":
    debug_response_time()
