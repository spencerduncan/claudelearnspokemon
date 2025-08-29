#!/usr/bin/env python3
"""Debug script to understand classification issue."""

import sys

sys.path.insert(0, "src")

from claudelearnspokemon.message_classifier import PatternBasedClassifier


def debug_classification():
    """Debug the classification step by step."""
    print("=== Debugging Message Classification ===")

    classifier = PatternBasedClassifier(enable_caching=False)  # Disable cache for debugging
    message = "Optimize the system performance"

    # Test strategic context
    strategic_context = {"requires_planning": True}
    strategic_result = classifier.classify(message, strategic_context)
    print(
        f"Strategic context result: {strategic_result.message_type}, confidence: {strategic_result.confidence}"
    )
    print(f"Strategic patterns: {strategic_result.matched_patterns}")

    # Test tactical context
    tactical_context = {"requires_implementation": True}
    tactical_result = classifier.classify(message, tactical_context)
    print(
        f"Tactical context result: {tactical_result.message_type}, confidence: {tactical_result.confidence}"
    )
    print(f"Tactical patterns: {tactical_result.matched_patterns}")

    # Test no context
    no_context_result = classifier.classify(message)
    print(
        f"No context result: {no_context_result.message_type}, confidence: {no_context_result.confidence}"
    )
    print(f"No context patterns: {no_context_result.matched_patterns}")


if __name__ == "__main__":
    debug_classification()
