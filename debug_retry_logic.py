#!/usr/bin/env python3
"""Debug script to understand retry logic issue."""

import sys

sys.path.insert(0, "src")

from claudelearnspokemon.priority_queue import QueuedMessage


def debug_retry_logic():
    """Debug the retry logic step by step."""
    print("=== Debugging QueuedMessage Retry Logic ===")

    # Test the retry logic step by step
    message = QueuedMessage(max_retries=3)
    print(
        f"Initial: retry_count={message.retry_count}, max_retries={message.max_retries}, can_retry={message.can_retry()}"
    )

    for i in range(4):
        message.increment_retry()
        print(
            f"After increment {i+1}: retry_count={message.retry_count}, max_retries={message.max_retries}, can_retry={message.can_retry()}"
        )

    print("\n=== Expected behavior according to test ===")
    print("After 3 increments: can_retry should be True")
    print("After 4th increment: can_retry should be False")


if __name__ == "__main__":
    debug_retry_logic()
