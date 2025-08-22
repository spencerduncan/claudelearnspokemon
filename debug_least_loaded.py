#!/usr/bin/env python3
"""Debug script to understand least loaded strategy issue."""

import sys

sys.path.insert(0, "src")

from unittest.mock import Mock

from claudelearnspokemon.priority_queue import MessagePriority, QueuedMessage
from claudelearnspokemon.routing_strategy import LeastLoadedStrategy, WorkerInfo, WorkerState


def debug_least_loaded():
    """Debug the least loaded strategy selection."""
    print("=== Debugging LeastLoadedStrategy Health Score Consideration ===")

    strategy = LeastLoadedStrategy(load_prediction_factor=0.7)

    # Create mock workers with different loads (same as test)
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

    # Make worker_2 degraded (as in the test)
    workers[2].state = WorkerState.DEGRADED

    print("Worker states and scores:")
    for i, worker in enumerate(workers):
        print(
            f"worker_{i}: load={worker.current_load}, state={worker.state.value}, "
            f"health_score={worker.health_score:.2f}, is_available={worker.is_available}"
        )

    message = QueuedMessage(content="test", priority=MessagePriority.NORMAL)

    # Calculate scores manually to debug
    print("\nDetailed scoring:")
    with strategy._lock:
        healthy_workers = [w for w in workers if w.is_available]
        worker_scores = []
        for worker in healthy_workers:
            predicted_load = strategy._calculate_predicted_load(worker, message)
            health_score = worker.health_score

            # Apply strong penalty for degraded workers - prefer healthy workers
            if worker.state == WorkerState.DEGRADED:
                health_penalty = 3.0  # Strong penalty for degraded state
            else:
                health_penalty = 2.0 - health_score  # Normal health penalty

            combined_score = predicted_load * health_penalty
            worker_scores.append((combined_score, worker))
            print(
                f"{worker.worker_id}: predicted_load={predicted_load:.3f}, "
                f"health_score={health_score:.3f}, penalty={health_penalty:.3f}, "
                f"combined_score={combined_score:.3f}"
            )

    selected_worker = strategy.select_worker(message, workers)

    print(f"\nSelected worker: {selected_worker}")
    print("Expected: worker_0 (should prefer healthy worker_0 over degraded worker_2)")


if __name__ == "__main__":
    debug_least_loaded()
