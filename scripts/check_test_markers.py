#!/usr/bin/env python3
"""
Script to ensure all tests have pytest speed markers (fast, medium, or slow).
Used in pre-commit hook to enforce test categorization.
"""

import argparse
import subprocess
import sys


def check_test_markers() -> bool:
    """Check that all tests have speed markers (fast, medium, or slow)."""

    # Get tests without markers
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "tests/",
            "-m",
            "not fast and not medium and not slow",
            "--collect-only",
            "-q",
        ],
        capture_output=True,
        text=True,
    )

    # Exit code 5 means no tests collected, which is good (all tests have markers)
    # Exit code 0 means some tests were collected but none selected (also good)
    if result.returncode not in [0, 5]:
        print("❌ Error running pytest to check markers:")
        if result.stderr:
            print(result.stderr)
        else:
            print(f"Pytest exited with code {result.returncode}")
        return False

    # Parse the output to get count of unmarked tests
    lines = result.stdout.strip().split("\n")
    summary_line = [
        line for line in lines if "tests collected" in line or "no tests collected" in line
    ]

    if not summary_line:
        print("❌ Could not parse pytest output")
        return False

    # Extract count from line like "93/403 tests collected (310 deselected)" or "no tests collected"
    summary = summary_line[0]
    if "no tests collected" in summary.lower():
        print("✅ All tests have proper speed markers")
        return True
    elif summary.startswith("="):
        # Parse "93/403 tests collected"
        parts = summary.split()
        count_part = parts[1]  # Should be "93/403"
        unmarked_count = int(count_part.split("/")[0])

        if unmarked_count > 0:
            print(f"❌ Found {unmarked_count} tests without speed markers (fast/medium/slow)")
            print("\nTo fix this, run:")
            print("  python scripts/profile_tests.py")
            print("  python scripts/apply_markers.py")
            print(
                "\nOr manually add @pytest.mark.fast/@pytest.mark.medium/@pytest.mark.slow to test classes"
            )
            return False
        else:
            print("✅ All tests have proper speed markers")
            return True
    else:
        print("✅ No tests found without markers")
        return True


def main():
    parser = argparse.ArgumentParser(description="Check that all tests have pytest speed markers")
    parser.parse_args()  # Parse but don't use args

    success = check_test_markers()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
