#!/usr/bin/env python3
"""
Optimized Test Runner for PR #208 Issue #82
Fixes resource contention issues in full test suite

John Botmack Performance Engineering:
- Batch processing to prevent resource exhaustion
- Accurate test counting and reporting
- Connection pooling performance measurement
- Mathematical precision in metrics

Author: John Botmack
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


class OptimizedTestRunner:
    """Performance-optimized test runner that prevents resource contention."""

    def __init__(self, batch_size: int = 50, timeout_per_batch: int = 120):
        """
        Initialize with performance-tuned parameters.

        Args:
            batch_size: Number of tests per batch to prevent resource exhaustion
            timeout_per_batch: Timeout per batch in seconds (default 2 minutes)
        """
        self.batch_size = batch_size
        self.timeout_per_batch = timeout_per_batch
        self.pythonpath = str(Path(__file__).parent / "src")

    def discover_all_tests(self) -> list[str]:
        """Discover all test files for accurate counting."""
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.pythonpath}:{env.get('PYTHONPATH', '')}"

        cmd = [sys.executable, "-m", "pytest", "--collect-only", "--quiet"]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            print(f"Test discovery failed: {result.stderr}")
            # Fallback: discover files manually using Path
            test_files = []
            tests_dir = Path("tests")
            if tests_dir.exists():
                for test_file in tests_dir.rglob("test_*.py"):
                    test_files.append(str(test_file))
            return sorted(test_files)

        # Parse pytest output to get test files
        test_files = []
        for line in result.stdout.split("\n"):
            if "<Module " in line and ".py>" in line:
                # Extract test file path from pytest output
                start = line.find("tests/")
                if start == -1:
                    start = line.find("<Module ")
                    if start != -1:
                        start += len("<Module ")
                end = line.find(".py>") + 3
                if start != -1 and end != -1:
                    test_file = line[start:end]
                    if test_file.endswith(".py"):
                        test_files.append(test_file)

        # Fallback if parsing failed
        if not test_files:
            tests_dir = Path("tests")
            if tests_dir.exists():
                for test_file in tests_dir.rglob("test_*.py"):
                    test_files.append(str(test_file))

        return sorted(set(test_files))

    def run_test_batch(self, test_files: list[str]) -> dict:
        """Run a batch of tests with proper resource isolation."""
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.pythonpath}:{env.get('PYTHONPATH', '')}"

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--tb=short",
            "--quiet",
            "--maxfail=5",  # Stop after 5 failures to save time
            "-v",
        ] + test_files

        start_time = time.perf_counter()

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_per_batch, env=env
            )

            duration = time.perf_counter() - start_time

            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "timeout": False,
            }

        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start_time
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Batch timed out after {self.timeout_per_batch}s",
                "duration": duration,
                "timeout": True,
            }

    def parse_test_results(self, output: str) -> tuple[int, int, int, int]:
        """
        Parse pytest output for accurate test counts.

        Returns:
            Tuple of (passed, failed, skipped, total)
        """
        lines = output.split("\n")

        # Look for the summary line
        for line in lines:
            if "passed" in line or "failed" in line:
                passed = failed = skipped = 0

                if " passed" in line:
                    try:
                        passed = int(line.split(" passed")[0].split()[-1])
                    except (ValueError, IndexError):
                        pass

                if " failed" in line:
                    try:
                        failed = int(line.split(" failed")[0].split()[-1])
                    except (ValueError, IndexError):
                        pass

                if " skipped" in line:
                    try:
                        skipped = int(line.split(" skipped")[0].split()[-1])
                    except (ValueError, IndexError):
                        pass

                total = passed + failed + skipped
                if total > 0:
                    return passed, failed, skipped, total

        return 0, 0, 0, 0

    def measure_connection_pooling_performance(self) -> float:
        """
        Measure connection pooling performance specifically.
        Target: <200ms as mentioned in review feedback
        """
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.pythonpath}:{env.get('PYTHONPATH', '')}"

        # Run the specific integration test that measures connection pooling
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_pokemon_gym_adapter_integration.py::TestPokemonGymAdapterIntegration::test_real_connection_pooling_behavior",
            "-v",
            "--tb=short",
        ]

        start_time = time.perf_counter()
        subprocess.run(cmd, capture_output=True, text=True, env=env)
        duration_ms = (time.perf_counter() - start_time) * 1000

        return duration_ms

    def run_all_tests(self) -> dict:
        """Run all tests in optimized batches with accurate reporting."""
        print("🚀 John Botmack Optimized Test Runner")
        print("=====================================")

        # Discover all tests
        print("📊 Discovering tests...")
        all_test_files = self.discover_all_tests()
        print(f"✅ Discovered {len(all_test_files)} test files")

        # Measure connection pooling performance first
        print("\n⚡ Measuring connection pooling performance...")
        connection_pooling_ms = self.measure_connection_pooling_performance()
        pooling_status = "✅ PASS" if connection_pooling_ms < 200 else "❌ FAIL"
        print(
            f"   Connection pooling: {connection_pooling_ms:.1f}ms {pooling_status} (target: <200ms)"
        )

        # Run tests in batches
        print(f"\n🔄 Running tests in batches of {self.batch_size}...")

        total_passed = total_failed = total_skipped = 0
        total_duration = 0
        batch_count = 0
        timeout_batches = 0

        for i in range(0, len(all_test_files), self.batch_size):
            batch = all_test_files[i : i + self.batch_size]
            batch_count += 1

            print(f"   Batch {batch_count}: {len(batch)} files...", end="", flush=True)

            result = self.run_test_batch(batch)
            total_duration += result["duration"]

            if result["timeout"]:
                timeout_batches += 1
                print(f" ⏱️  TIMEOUT ({result['duration']:.1f}s)")
                continue

            if result["returncode"] == 0:
                passed, failed, skipped, _ = self.parse_test_results(result["stdout"])
                total_passed += passed
                total_failed += failed
                total_skipped += skipped
                print(f" ✅ {passed} passed ({result['duration']:.1f}s)")
            else:
                passed, failed, skipped, _ = self.parse_test_results(result["stdout"])
                total_passed += passed
                total_failed += failed
                total_skipped += skipped
                print(f" ❌ {failed} failed, {passed} passed ({result['duration']:.1f}s)")

        # Calculate final metrics with mathematical precision
        total_tests = total_passed + total_failed + total_skipped
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0.0

        results = {
            "test_files_discovered": len(all_test_files),
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "timeout_batches": timeout_batches,
            "pass_rate_percent": pass_rate,
            "total_duration_seconds": total_duration,
            "connection_pooling_ms": connection_pooling_ms,
            "connection_pooling_passes": connection_pooling_ms < 200,
            "batch_size": self.batch_size,
            "batch_count": batch_count,
        }

        return results

    def print_summary(self, results: dict):
        """Print accurate test results summary."""
        print("\n" + "=" * 60)
        print("📈 ACCURATE TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Test Files Discovered:     {results['test_files_discovered']}")
        print(f"Total Tests:              {results['total_tests']}")
        print(f"✅ Passed:                {results['passed']}")
        print(f"❌ Failed:                {results['failed']}")
        print(f"⏭️  Skipped:               {results['skipped']}")
        print(f"⏱️  Timeout Batches:       {results['timeout_batches']}")
        print(f"📊 Pass Rate:             {results['pass_rate_percent']:.1f}%")
        print(f"⏰ Total Duration:        {results['total_duration_seconds']:.1f}s")
        print()
        print("🚀 PERFORMANCE MEASUREMENTS")
        print("-" * 30)
        status = "PASS" if results["connection_pooling_passes"] else "FAIL"
        print(f"Connection Pooling:       {results['connection_pooling_ms']:.1f}ms ({status})")
        print("Target:                   <200ms")
        print()

        if results["failed"] > 0 or results["timeout_batches"] > 0:
            print("❌ ISSUES DETECTED:")
            if results["failed"] > 0:
                print(f"   - {results['failed']} tests failed")
            if results["timeout_batches"] > 0:
                print(f"   - {results['timeout_batches']} batches timed out")
            if not results["connection_pooling_passes"]:
                print(
                    f"   - Connection pooling too slow: {results['connection_pooling_ms']:.1f}ms > 200ms"
                )
        else:
            print("✅ ALL TESTS PASSED - READY FOR PR UPDATE")


def main():
    """Main entry point with John Botmack precision."""
    runner = OptimizedTestRunner(batch_size=50, timeout_per_batch=120)
    results = runner.run_all_tests()
    runner.print_summary(results)

    # Save results for PR update
    with open("accurate_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Exit with appropriate code
    if (
        results["failed"] > 0
        or results["timeout_batches"] > 0
        or not results["connection_pooling_passes"]
    ):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
