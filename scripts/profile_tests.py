#!/usr/bin/env python3
"""
Test profiling script to categorize tests by speed.

Runs tests with timeouts to categorize into fast/medium/slow buckets:
- Fast: < 1 second
- Medium: 1-5 seconds
- Slow: > 5 seconds or timeout

Usage:
    python scripts/profile_tests.py
"""

import json
import subprocess
import time
from pathlib import Path


class TestProfiler:
    """Profile test execution speeds and categorize tests."""

    def __init__(self):
        self.results = {}
        self.project_root = Path(__file__).parent.parent

    def get_test_files(self) -> list[Path]:
        """Get all test files to profile."""
        test_dir = self.project_root / "tests"
        return list(test_dir.glob("test_*.py"))

    def run_test_file_with_timeout(
        self, test_file: Path, timeout: float
    ) -> tuple[float, str, bool]:
        """
        Run a test file with timeout.

        Returns:
            (execution_time, status, timed_out)
        """
        cmd = ["python", "-m", "pytest", str(test_file), "-q", "--tb=no"]

        start_time = time.perf_counter()
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=timeout, text=True, cwd=self.project_root
            )
            execution_time = time.perf_counter() - start_time

            if result.returncode == 0:
                status = "passed"
            else:
                status = "failed"

            return execution_time, status, False

        except subprocess.TimeoutExpired:
            execution_time = timeout
            return execution_time, "timeout", True

    def profile_with_timeout(
        self, test_files: list[Path], timeout: float, category: str
    ) -> list[Path]:
        """
        Profile test files with given timeout and categorize.

        Returns list of test files that exceeded timeout.
        """
        remaining_files = []

        for test_file in test_files:
            print(f"Testing {test_file.name} with {timeout}s timeout...")

            execution_time, status, timed_out = self.run_test_file_with_timeout(test_file, timeout)

            if timed_out:
                print(f"  ⏰ {test_file.name}: TIMEOUT (>{timeout}s)")
                remaining_files.append(test_file)
            else:
                print(f"  ✓ {test_file.name}: {execution_time:.2f}s ({status})")
                self.results[str(test_file.relative_to(self.project_root))] = {
                    "duration": execution_time,
                    "status": status,
                    "category": category,
                }

        return remaining_files

    def run_profiling(self) -> dict:
        """Run complete test profiling."""
        print("🔍 Starting test profiling...")
        test_files = self.get_test_files()
        print(f"Found {len(test_files)} test files")

        # Phase 1: Identify fast tests (< 1s)
        print("\n📊 Phase 1: Profiling for FAST tests (timeout: 1s)")
        remaining_files = self.profile_with_timeout(test_files, 1.0, "fast")

        # Phase 2: Identify medium tests (< 5s)
        if remaining_files:
            print("\n📊 Phase 2: Profiling for MEDIUM tests (timeout: 5s)")
            print(f"Testing {len(remaining_files)} remaining files...")
            slow_files = self.profile_with_timeout(remaining_files, 5.0, "medium")

            # Phase 3: Mark remaining as slow
            if slow_files:
                print(f"\n📊 Phase 3: Marking {len(slow_files)} files as SLOW")
                for test_file in slow_files:
                    print(f"  🐌 {test_file.name}: SLOW (>5s)")
                    self.results[str(test_file.relative_to(self.project_root))] = {
                        "duration": ">5.0",
                        "status": "timeout",
                        "category": "slow",
                    }

        return self.results

    def save_results(self, output_file: str = "profile_results.json") -> None:
        """Save profiling results to JSON file."""
        output_path = self.project_root / output_file

        # Add summary statistics
        summary = {
            "fast": sum(1 for r in self.results.values() if r["category"] == "fast"),
            "medium": sum(1 for r in self.results.values() if r["category"] == "medium"),
            "slow": sum(1 for r in self.results.values() if r["category"] == "slow"),
            "total": len(self.results),
        }

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "results": self.results,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Results saved to {output_path}")
        return report


def main():
    """Main profiling function."""
    profiler = TestProfiler()

    try:
        profiler.run_profiling()  # Run profiling but don't store results separately
        report = profiler.save_results()

        # Display summary
        summary = report["summary"]
        print("\n🎯 PROFILING SUMMARY:")
        print(f"   Fast tests (<1s): {summary['fast']}")
        print(f"   Medium tests (1-5s): {summary['medium']}")
        print(f"   Slow tests (>5s): {summary['slow']}")
        print(f"   Total tests: {summary['total']}")

        print(f"\nFast test coverage: {summary['fast']/summary['total']*100:.1f}%")

    except KeyboardInterrupt:
        print("\n🛑 Profiling interrupted by user")
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")


if __name__ == "__main__":
    main()
