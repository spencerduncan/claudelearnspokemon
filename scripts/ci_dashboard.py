#!/usr/bin/env python3
# mypy: ignore-errors
# ruff: noqa: UP007
"""
CI Dashboard - Terminal-based monitoring for claudelearnspokemon project
Displays git status, worktree info, test results, and code statistics
"""

import argparse
import asyncio
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Union

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class CIDashboard:
    def __init__(self, refresh_interval: int = 30):
        self.console = Console()
        self.refresh_interval = refresh_interval
        self.cache: dict[str, Any] = {}
        self.cache_times: dict[str, float] = {}
        self.project_root = Path("/home/sd/claudelearnspokemon")
        self.worktrees_root = Path("/home/sd/worktrees")

    def run_command(
        self, cmd: Union[str, list[str]], cwd: Path | None = None
    ) -> tuple[int, str, str]:
        """Run a command safely without shell injection vulnerabilities.

        Args:
            cmd: Command as string (will be parsed) or list of arguments
            cwd: Working directory for command

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        try:
            # Parse command string into safe argument list
            if isinstance(cmd, str):
                args = shlex.split(cmd)
            else:
                args = cmd

            # Validate that we have at least one argument
            if not args:
                return 1, "", "Empty command"

            result = subprocess.run(
                args,
                shell=False,  # SECURITY: Never use shell=True
                capture_output=True,
                text=True,
                cwd=cwd or self.project_root,
                timeout=10,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except (OSError, ValueError) as e:
            return 1, "", f"Command execution error: {str(e)}"
        except Exception as e:
            return 1, "", f"Unexpected error: {str(e)}"

    def get_cached_result(
        self, key: str, cache_duration: int, func: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Get cached result or compute if expired."""
        now = time.time()
        if key in self.cache_times:
            if now - self.cache_times[key] < cache_duration:
                return self.cache[key]

        result = func(*args, **kwargs)
        self.cache[key] = result
        self.cache_times[key] = now
        return result

    def _safe_git_command(self, args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
        """Execute a git command safely with argument validation.

        Args:
            args: Git command arguments (without 'git' prefix)
            cwd: Working directory

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        # Validate git arguments to prevent injection
        safe_args = ["git"] + [str(arg) for arg in args if arg is not None]
        return self.run_command(safe_args, cwd)

    def _safe_gh_command(self, args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
        """Execute a gh CLI command safely with argument validation.

        Args:
            args: GitHub CLI command arguments (without 'gh' prefix)
            cwd: Working directory

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        # Validate gh arguments to prevent injection
        safe_args = ["gh"] + [str(arg) for arg in args if arg is not None]
        return self.run_command(safe_args, cwd)

    def get_git_status(self) -> dict[str, Any]:
        """Get current git status information."""

        def _fetch_git_status() -> dict[str, Any]:
            status = {
                "branch": "unknown",
                "ahead": 0,
                "behind": 0,
                "uncommitted": 0,
                "last_commits": [],
            }

            # Get current branch
            code, stdout, _ = self._safe_git_command(["branch", "--show-current"])
            if code == 0:
                status["branch"] = stdout.strip() or "main"

            # Get ahead/behind status
            code, stdout, _ = self._safe_git_command(
                ["rev-list", "--left-right", "--count", "origin/main...HEAD"]
            )
            if code == 0 and stdout:
                parts = stdout.strip().split()
                if len(parts) == 2:
                    status["behind"] = int(parts[0])
                    status["ahead"] = int(parts[1])

            # Get uncommitted changes
            code, stdout, _ = self._safe_git_command(["status", "--porcelain"])
            if code == 0:
                status["uncommitted"] = len([line for line in stdout.splitlines() if line.strip()])

            # Get last 3 commits
            code, stdout, _ = self._safe_git_command(["log", "--oneline", "-3"])
            if code == 0:
                status["last_commits"] = stdout.strip().splitlines()[:3]

            return status

        return self.get_cached_result("git_status", 10, _fetch_git_status)

    def get_worktree_status(self) -> list[dict[str, Any]]:
        """Get status of all worktrees with PR status information."""

        def _fetch_worktree_status() -> list[dict[str, Any]]:
            worktrees: list[dict[str, Any]] = []

            if not self.worktrees_root.exists():
                return worktrees

            for worktree_dir in sorted(self.worktrees_root.iterdir()):
                if not worktree_dir.is_dir():
                    continue

                worktree = {
                    "name": worktree_dir.name,
                    "path": worktree_dir,
                    "last_modified": None,
                    "last_modified_relative": "unknown",
                    "uncommitted": 0,
                    "branch": "unknown",
                    "status": "inactive",
                    "pr_status": "no-pr",
                    "pr_state": None,
                    "pr_number": None,
                }

                # Get branch name
                code, stdout, _ = self._safe_git_command(
                    ["branch", "--show-current"], cwd=worktree_dir
                )
                if code == 0:
                    branch_name = stdout.strip()
                    worktree["branch"] = branch_name or "unknown"

                    # Check if branch is merged into main
                    if branch_name:
                        # Use git merge-base to safely check if branch is merged
                        code, stdout, _ = self._safe_git_command(
                            ["merge-base", "--is-ancestor", branch_name, "origin/main"],
                            cwd=worktree_dir,
                        )
                        if code == 0:  # Branch is ancestor of main (merged)
                            worktree["pr_status"] = "merged"
                            worktree["status"] = "merged"
                        else:
                            # Check for active PR using gh CLI (fallback gracefully if not available)
                            code, stdout, _ = self._safe_gh_command(
                                ["pr", "list", "--head", branch_name, "--json", "number,state"],
                                cwd=worktree_dir,
                            )
                            # If gh command fails, try to get empty JSON array
                            if code != 0:
                                stdout = "[]"
                            if code == 0 and stdout.strip():
                                try:
                                    import json

                                    prs = json.loads(stdout)
                                    if prs:
                                        pr = prs[0]  # Get first matching PR
                                        worktree["pr_number"] = pr["number"]
                                        worktree["pr_state"] = pr["state"]
                                        if pr["state"] == "OPEN":
                                            worktree["pr_status"] = "active-pr"
                                        elif pr["state"] == "CLOSED":
                                            worktree["pr_status"] = "closed-pr"
                                        elif pr["state"] == "MERGED":
                                            worktree["pr_status"] = "merged"
                                            worktree["status"] = "merged"
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    # Fallback to no-pr if parsing fails
                                    pass

                # Get last commit time
                code, stdout, _ = self._safe_git_command(
                    ["log", "-1", "--format=%ct"], cwd=worktree_dir
                )
                if code == 0 and stdout.strip():
                    timestamp = int(stdout.strip())
                    worktree["last_modified"] = datetime.fromtimestamp(timestamp)

                    # Calculate relative time and status (only if not already marked as merged)
                    if worktree["status"] != "merged":
                        now = datetime.now()
                        delta = now - worktree["last_modified"]

                        if delta < timedelta(hours=1):
                            worktree["last_modified_relative"] = f"{int(delta.seconds / 60)}m ago"
                            worktree["status"] = "active"
                        elif delta < timedelta(days=1):
                            worktree["last_modified_relative"] = f"{int(delta.seconds / 3600)}h ago"
                            worktree["status"] = "recent"
                        elif delta < timedelta(days=3):
                            worktree["last_modified_relative"] = f"{delta.days}d ago"
                            worktree["status"] = "stale"
                        else:
                            worktree["last_modified_relative"] = f"{delta.days}d ago"
                            worktree["status"] = "inactive"
                    else:
                        # For merged branches, still show the time
                        now = datetime.now()
                        delta = now - worktree["last_modified"]
                        if delta < timedelta(days=1):
                            worktree["last_modified_relative"] = f"{int(delta.seconds / 3600)}h ago"
                        else:
                            worktree["last_modified_relative"] = f"{delta.days}d ago"

                # Get uncommitted changes
                code, stdout, _ = self._safe_git_command(
                    ["status", "--porcelain"], cwd=worktree_dir
                )
                if code == 0:
                    worktree["uncommitted"] = len(
                        [line for line in stdout.splitlines() if line.strip()]
                    )

                worktrees.append(worktree)

            return worktrees

        return self.get_cached_result("worktree_status", 30, _fetch_worktree_status)

    def get_test_status(self) -> dict[str, Any]:
        """Get test results and coverage."""

        def _fetch_test_status() -> dict[str, Any]:
            status = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "coverage": 0,
                "last_run": "never",
            }

            # Count total tests using pytest --collect-only
            code, stdout, stderr = self.run_command(
                ["python", "-m", "pytest", "tests/", "--collect-only", "-q"]
            )
            # Check if collection succeeded (code 0) or had errors but still collected tests (code 2)
            if code == 0 or (code == 2 and "collected" in stdout):
                # Count lines that look like test functions: <Function test_*> or <Method test_*>
                import re

                test_lines = [
                    line
                    for line in stdout.splitlines()
                    if re.search(r"<(Function|Method)\s+test_", line.strip())
                ]
                status["total"] = len(test_lines)

            # Try to get real test results (quick run for small test suites)
            # Only run tests if collection was successful (no errors)
            should_run_tests = False
            if code == 0:
                should_run_tests = True
            elif code == 2 and "errors" in stdout:
                # Collection had errors - try running only tests that can be collected
                # For safety, we'll skip test execution if there are collection errors
                should_run_tests = False
            else:
                should_run_tests = False

            if should_run_tests:
                # Use timeout command safely
                code, stdout, stderr = self.run_command(
                    [
                        "timeout",
                        "30",
                        "python",
                        "-m",
                        "pytest",
                        "tests/",
                        "-v",
                        "--tb=no",
                        "--no-header",
                    ]
                )
                if code != 0:
                    stdout += "\nTIMEOUT_OR_ERROR"
            else:
                # Skip test execution due to collection errors
                stdout = "TIMEOUT_OR_ERROR - Collection errors prevent execution"
                code = 1

            if code == 0 and "TIMEOUT_OR_ERROR" not in stdout:
                # Parse pytest output for real results
                import re

                # Look for the summary line like "=== 5 passed, 1 failed, 2 skipped in 1.23s ==="
                summary_match = re.search(r"=+ (.+) in [\d\.]+s =+", stdout, re.MULTILINE)

                if summary_match:
                    summary = summary_match.group(1)

                    # Extract counts for each result type
                    passed_match = re.search(r"(\d+) passed", summary)
                    if passed_match:
                        status["passed"] = int(passed_match.group(1))

                    failed_match = re.search(r"(\d+) failed", summary)
                    if failed_match:
                        status["failed"] = int(failed_match.group(1))

                    skipped_match = re.search(r"(\d+) skipped", summary)
                    if skipped_match:
                        status["skipped"] = int(skipped_match.group(1))

                    status["last_run"] = datetime.now().strftime("%H:%M:%S")
            else:
                # Fallback: Tests haven't been run or took too long
                if "Collection errors prevent execution" in stdout:
                    status["last_run"] = "collection errors"
                else:
                    status["last_run"] = "not run"

            # Try to get real coverage using pytest-cov
            code, stdout, _ = self.run_command(
                [
                    "timeout",
                    "30",
                    "python",
                    "-m",
                    "pytest",
                    "tests/",
                    "--cov=src",
                    "--cov=claudelearnspokemon",
                    "--cov-report=term-missing",
                    "--tb=no",
                    "-q",
                ]
            )

            # Extract TOTAL line from coverage output
            if code == 0 and stdout:
                import re

                total_match = re.search(r"^TOTAL.*", stdout, re.MULTILINE)
                if total_match:
                    stdout = total_match.group(0)
                else:
                    stdout = "NO_COVERAGE"
            else:
                stdout = "NO_COVERAGE"

            if code == 0 and "NO_COVERAGE" not in stdout and "%" in stdout:
                # Parse coverage line like "TOTAL    1234    567    54%"
                import re

                coverage_match = re.search(r"TOTAL.*?(\d+)%", stdout)
                if coverage_match:
                    status["coverage"] = int(coverage_match.group(1))
            else:
                # Fallback coverage estimation
                if status["total"] > 0:
                    actual_run_total = status["passed"] + status["failed"] + status["skipped"]
                    if actual_run_total > 0:
                        # Base coverage on test success rate as rough estimate
                        status["coverage"] = int((status["passed"] / actual_run_total) * 100)
                    else:
                        status["coverage"] = 0

            return status

        return self.get_cached_result("test_status", 60, _fetch_test_status)

    def get_code_quality(self) -> dict[str, Any]:
        """Get code quality metrics from ruff, black, mypy."""

        def _fetch_code_quality() -> dict[str, Any]:
            quality = {
                "ruff_violations": 0,
                "black_status": "unknown",
                "mypy_errors": 0,
                "pre_commit": "unknown",
            }

            # Check ruff - use --statistics to get accurate count
            code, stdout, _ = self.run_command(["ruff", "check", "src/", "tests/", "--statistics"])
            if code != 0:
                # Parse statistics output for actual violation count
                # Each line looks like: "3  I001  [*] unsorted-imports"
                import re

                total_violations = 0
                for line in stdout.splitlines():
                    match = re.match(r"^\s*(\d+)\s+\w+", line.strip())
                    if match:
                        total_violations += int(match.group(1))
                quality["ruff_violations"] = total_violations
            else:
                quality["ruff_violations"] = 0
            # Check black
            code, _, _ = self.run_command(["black", "--check", "src/", "tests/"])
            quality["black_status"] = "formatted" if code == 0 else "needs formatting"

            # Check mypy
            code, stdout, _ = self.run_command(["mypy", "src/"])
            if code != 0:
                # Count actual error lines - look for lines that contain ': error:'
                import re

                error_lines = [line for line in stdout.splitlines() if re.search(r": error:", line)]
                quality["mypy_errors"] = len(error_lines)
            else:
                quality["mypy_errors"] = 0

            # Check pre-commit
            if (self.project_root / ".git" / "hooks" / "pre-commit").exists():
                quality["pre_commit"] = "installed"
            else:
                quality["pre_commit"] = "not installed"

            return quality

        return self.get_cached_result("code_quality", 60, _fetch_code_quality)

    def get_code_statistics(self) -> dict[str, Any]:
        """Get code statistics including file sizes."""

        def _fetch_code_stats() -> dict[str, Any]:
            stats = {"largest_files": [], "total_lines": 0, "file_counts": {}, "total_py_files": 0}

            # Find all Python files in project (excluding venv, .git, __pycache__ etc.)
            code, stdout, _ = self.run_command(
                [
                    "find",
                    ".",
                    "-name",
                    "*.py",
                    "-type",
                    "f",
                    "-not",
                    "-path",
                    "./venv/*",
                    "-not",
                    "-path",
                    "./.git/*",
                    "-not",
                    "-path",
                    "*/__pycache__/*",
                ]
            )

            if code == 0:
                py_files = [f.strip() for f in stdout.splitlines() if f.strip()]

                # Get line counts for each file
                file_sizes = []
                for file_path in py_files:
                    # Use wc safely with file path as separate argument
                    code, stdout, _ = self.run_command(["wc", "-l", file_path])
                    if code == 0:
                        parts = stdout.strip().split()
                        if parts:
                            lines = int(parts[0])
                            file_name = Path(file_path).name
                            file_sizes.append((file_name, lines, file_path))
                            stats["total_lines"] += lines

                # Sort by size and get top 5
                file_sizes.sort(key=lambda x: x[1], reverse=True)
                stats["largest_files"] = file_sizes[:5]
                stats["total_py_files"] = len(py_files)

            # Count files by extension (using same exclusion pattern for consistency)
            for ext in [".py", ".md", ".sh", ".toml", ".yml", ".yaml"]:
                # Build find command with safe arguments
                find_args = [
                    "find",
                    ".",
                    "-name",
                    f"*{ext}",
                    "-type",
                    "f",
                    "-not",
                    "-path",
                    "./venv/*",
                    "-not",
                    "-path",
                    "./.git/*",
                    "-not",
                    "-path",
                    "*/__pycache__/*",
                ]
                code, stdout, _ = self.run_command(find_args)
                if code == 0:
                    # Count lines in output instead of using shell pipe
                    count = len([line for line in stdout.splitlines() if line.strip()])
                    if count > 0:
                        stats["file_counts"][ext] = count

            return stats

        return self.get_cached_result("code_stats", 120, _fetch_code_stats)

    def get_pr_status(self) -> dict[str, Any]:
        """Get pull request status and information.

        Uses gh CLI if available, otherwise returns appropriate error message.
        For environments that block gh CLI, consider using GitHub MCP tools integration.
        """

        def _fetch_pr_status() -> dict[str, Any]:
            pr_data = {
                "prs": [],
                "total_open": 0,
                "error": None,
            }

            try:
                # Check if gh CLI is available
                code, _, stderr = self.run_command(["gh", "--version"])
                if code != 0:
                    pr_data["error"] = "gh CLI not available"
                    return pr_data

                # Get list of open PRs
                code, stdout, stderr = self._safe_gh_command(
                    ["pr", "list", "--state=open", "--json", "number,title,author,url"]
                )

                if code != 0:
                    pr_data["error"] = f"Failed to fetch PR list: {stderr}"
                    return pr_data

                if not stdout.strip():
                    return pr_data  # No PRs

                import json

                try:
                    prs = json.loads(stdout)
                except json.JSONDecodeError:
                    pr_data["error"] = "Failed to parse PR list JSON"
                    return pr_data

                pr_data["total_open"] = len(prs)

                # Get detailed info for each PR (limit to 10 most recent)
                for pr in prs[:10]:
                    pr_number = pr["number"]
                    pr_info = {
                        "number": pr_number,
                        "title": pr["title"][:50],  # Truncate long titles
                        "author": pr["author"]["login"],
                        "comments": 0,
                        "additions": 0,
                        "deletions": 0,
                        "commits_ahead": 0,
                        "url": pr["url"],
                        "ci_status": "unknown",
                        "review_status": "pending",
                    }

                    # Get detailed PR info including comments and diff stats
                    code, stdout, _ = self._safe_gh_command(
                        [
                            "pr",
                            "view",
                            str(pr_number),
                            "--json",
                            "comments,additions,deletions,commits",
                        ]
                    )
                    if code == 0:
                        try:
                            pr_detail = json.loads(stdout)
                            pr_info["comments"] = len(pr_detail.get("comments", []))
                            pr_info["additions"] = pr_detail.get("additions", 0)
                            pr_info["deletions"] = pr_detail.get("deletions", 0)
                            pr_info["commits_ahead"] = len(pr_detail.get("commits", []))
                        except (json.JSONDecodeError, KeyError):
                            pass  # Use defaults

                    # Get CI status
                    code, stdout, _ = self._safe_gh_command(
                        ["pr", "checks", str(pr_number), "--json", "name,state,conclusion"]
                    )
                    if code == 0 and stdout.strip():
                        try:
                            checks = json.loads(stdout)
                            if checks:
                                # Determine overall CI status
                                states = [check.get("state", "unknown") for check in checks]
                                conclusions = [
                                    check.get("conclusion", "unknown") for check in checks
                                ]

                                if any(state == "in_progress" for state in states):
                                    pr_info["ci_status"] = "pending"
                                elif any(conclusion == "failure" for conclusion in conclusions):
                                    pr_info["ci_status"] = "failing"
                                elif all(conclusion == "success" for conclusion in conclusions):
                                    pr_info["ci_status"] = "passing"
                                else:
                                    pr_info["ci_status"] = "mixed"
                            else:
                                pr_info["ci_status"] = "none"
                        except (json.JSONDecodeError, KeyError):
                            pr_info["ci_status"] = "unknown"
                    else:
                        pr_info["ci_status"] = "none"

                    # Get review status
                    code, stdout, _ = self._safe_gh_command(
                        ["pr", "view", str(pr_number), "--json", "reviews"]
                    )
                    if code == 0:
                        try:
                            pr_detail = json.loads(stdout)
                            reviews = pr_detail.get("reviews", [])
                            if reviews:
                                # Get latest review state for each reviewer
                                reviewer_states = {}
                                for review in reviews:
                                    reviewer = review.get("author", {}).get("login", "unknown")
                                    state = review.get("state", "COMMENTED")
                                    reviewer_states[reviewer] = state

                                # Determine overall review status
                                states = list(reviewer_states.values())
                                if any(state == "CHANGES_REQUESTED" for state in states):
                                    pr_info["review_status"] = "changes_requested"
                                elif any(state == "APPROVED" for state in states):
                                    pr_info["review_status"] = "approved"
                                else:
                                    pr_info["review_status"] = "commented"
                            else:
                                pr_info["review_status"] = "pending"
                        except (json.JSONDecodeError, KeyError):
                            pr_info["review_status"] = "pending"

                    pr_data["prs"].append(pr_info)

            except Exception as e:
                pr_data["error"] = f"Exception fetching PR data: {str(e)}"

            return pr_data

        return self.get_cached_result("pr_status", 60, _fetch_pr_status)

    def create_git_panel(self) -> Panel:
        """Create the git status panel."""
        git_status = self.get_git_status()

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="cyan")
        table.add_column("Value")

        # Branch info
        table.add_row("Branch:", f"[yellow]{git_status['branch']}[/yellow]")

        # Sync status
        sync_text = ""
        if git_status["ahead"] > 0:
            sync_text += f"[green]↑{git_status['ahead']}[/green] "
        if git_status["behind"] > 0:
            sync_text += f"[red]↓{git_status['behind']}[/red] "
        if not sync_text:
            sync_text = "[green]in sync[/green]"
        table.add_row("vs origin/main:", sync_text)

        # Uncommitted changes
        uncommitted_style = "green" if git_status["uncommitted"] == 0 else "yellow"
        table.add_row(
            "Uncommitted:",
            f"[{uncommitted_style}]{git_status['uncommitted']} changes[/{uncommitted_style}]",
        )

        # Add separator
        table.add_row("", "")
        table.add_row("[dim]Recent commits:[/dim]", "")

        # Last commits
        for commit in git_status["last_commits"][:3]:
            # Truncate commit message if too long
            if len(commit) > 40:
                commit = commit[:37] + "..."
            table.add_row("", f"[dim]{commit}[/dim]")

        return Panel(table, title="Git Status", border_style="blue")

    def create_worktree_panel(self) -> Panel:
        """Create the worktree status panel showing ALL worktrees with PR status."""
        worktrees = self.get_worktree_status()

        table = Table(show_header=True, box=None)
        table.add_column("Worktree", style="cyan", width=15)
        table.add_column("Last Activity", justify="right", width=10)
        table.add_column("Changes", justify="center", width=7)
        table.add_column("PR Status", justify="center", width=10)
        table.add_column("Status", justify="center", width=6)

        # Sort worktrees: active first, then recent, then stale, then merged, then inactive
        def sort_priority(w):
            status_order = {"active": 1, "recent": 2, "stale": 3, "merged": 4, "inactive": 5}
            # Primary sort by status priority, secondary by last modified time
            return (
                status_order.get(w["status"], 6),
                -(w.get("last_modified") or datetime.min).timestamp(),
            )

        sorted_worktrees = sorted(worktrees, key=sort_priority)

        for worktree in sorted_worktrees:  # Show ALL worktrees, no limit
            # Determine status style and icon
            if worktree["status"] == "active":
                status_icon = "✅"
                time_style = "green"
            elif worktree["status"] == "recent":
                status_icon = "⚠️"
                time_style = "yellow"
            elif worktree["status"] == "merged":
                status_icon = "🟢"  # Green circle for merged
                time_style = "green"
            elif worktree["status"] == "stale":
                status_icon = "🟠"  # Orange circle for stale
                time_style = "orange"
            else:  # inactive
                status_icon = "🔴"  # Red circle for inactive
                time_style = "red"

            # PR Status indicator
            pr_status = worktree["pr_status"]
            if pr_status == "active-pr":
                pr_icon = "🔄"  # Active PR
                pr_text = f"PR #{worktree['pr_number']}"
                pr_style = "blue"
            elif pr_status == "merged":
                pr_icon = "🟢"  # Merged PR
                pr_text = "merged"
                pr_style = "green"
            elif pr_status == "closed-pr":
                pr_icon = "🔴"  # Closed PR
                pr_text = "closed"
                pr_style = "red"
            else:  # no-pr
                pr_icon = "⚫"  # No PR
                pr_text = "no PR"
                pr_style = "dim"

            # Changes indicator
            changes_text = f"{worktree['uncommitted']}" if worktree["uncommitted"] > 0 else "-"
            changes_style = "yellow" if worktree["uncommitted"] > 0 else "dim"

            # Truncate worktree name if too long
            name = worktree["name"]
            if len(name) > 13:
                name = name[:10] + "..."

            table.add_row(
                name,
                f"[{time_style}]{worktree['last_modified_relative']}[/{time_style}]",
                f"[{changes_style}]{changes_text}[/{changes_style}]",
                f"[{pr_style}]{pr_icon} {pr_text}[/{pr_style}]",
                status_icon,
            )

        return Panel(table, title=f"Worktrees ({len(worktrees)} total)", border_style="green")

    def create_test_panel(self) -> Panel:
        """Create the test and quality panel."""
        test_status = self.get_test_status()
        quality = self.get_code_quality()

        # Create content
        content = Table(show_header=False, box=None, padding=(0, 1))
        content.add_column("Metric", style="cyan")
        content.add_column("Status")

        # Test results
        test_icon = "✅" if test_status["failed"] == 0 else "❌"
        test_text = f"{test_status['passed']}/{test_status['total']} passed"
        if test_status["failed"] > 0:
            test_text += f" ([red]{test_status['failed']} failed[/red])"
        content.add_row("Tests:", f"{test_icon} {test_text}")

        # Coverage bar
        coverage = test_status["coverage"]
        bar_width = 10
        filled = int(coverage / 10)
        bar = "█" * filled + "░" * (bar_width - filled)
        coverage_color = "green" if coverage >= 80 else "yellow" if coverage >= 60 else "red"
        content.add_row("Coverage:", f"[{coverage_color}]{bar} {coverage}%[/{coverage_color}]")

        # Ruff
        ruff_icon = "✅" if quality["ruff_violations"] == 0 else "⚠️"
        ruff_text = (
            f"{quality['ruff_violations']} violations"
            if quality["ruff_violations"] > 0
            else "clean"
        )
        ruff_color = "green" if quality["ruff_violations"] == 0 else "yellow"
        content.add_row("Ruff:", f"{ruff_icon} [{ruff_color}]{ruff_text}[/{ruff_color}]")

        # Black
        black_icon = "✅" if quality["black_status"] == "formatted" else "⚠️"
        black_color = "green" if quality["black_status"] == "formatted" else "yellow"
        content.add_row(
            "Black:", f"{black_icon} [{black_color}]{quality['black_status']}[/{black_color}]"
        )

        # Mypy
        mypy_icon = "✅" if quality["mypy_errors"] == 0 else "❌"
        mypy_text = (
            f"{quality['mypy_errors']} errors" if quality["mypy_errors"] > 0 else "no errors"
        )
        mypy_color = "green" if quality["mypy_errors"] == 0 else "red"
        content.add_row("Mypy:", f"{mypy_icon} [{mypy_color}]{mypy_text}[/{mypy_color}]")

        # Pre-commit
        pc_icon = "✅" if quality["pre_commit"] == "installed" else "⚠️"
        pc_color = "green" if quality["pre_commit"] == "installed" else "yellow"
        content.add_row(
            "Pre-commit:", f"{pc_icon} [{pc_color}]{quality['pre_commit']}[/{pc_color}]"
        )

        return Panel(content, title="Tests & Quality", border_style="yellow")

    def create_stats_panel(self) -> Panel:
        """Create the code statistics panel."""
        stats = self.get_code_statistics()

        content = Table(show_header=False, box=None, padding=(0, 1))
        content.add_column("", style="cyan")
        content.add_column("", justify="right")

        # Largest files header
        content.add_row("[bold]Largest Files:[/bold]", "")

        # List largest files
        for i, (name, lines, _) in enumerate(stats["largest_files"], 1):
            content.add_row(f"  {i}. {name}", f"[yellow]{lines:,} lines[/yellow]")

        # Add separator
        content.add_row("", "")

        # Summary stats - clearer labeling
        content.add_row("[bold]Code Summary:[/bold]", "")
        content.add_row("  Total lines of code:", f"[green]{stats['total_lines']:,}[/green]")
        content.add_row("  Python files (project):", f"[green]{stats['total_py_files']}[/green]")

        # File type counts
        if stats["file_counts"]:
            content.add_row("", "")
            content.add_row("[bold]Project Files by Type:[/bold]", "")
            for ext, count in sorted(stats["file_counts"].items()):
                content.add_row(f"  {ext} files:", f"[dim]{count}[/dim]")

        return Panel(content, title="Code Statistics", border_style="magenta")

    def create_pr_panel(self) -> Panel:
        """Create the pull request status panel."""
        pr_status = self.get_pr_status()

        # Handle errors
        if pr_status.get("error"):
            error_text = Text(f"Error: {pr_status['error']}", style="red")
            return Panel(error_text, title="Pull Requests (Error)", border_style="red")

        # Handle no PRs
        if not pr_status["prs"]:
            no_prs_text = Text("No open pull requests", style="dim", justify="center")
            return Panel(no_prs_text, title="Pull Requests (0)", border_style="dim")

        # Create table with PR data
        table = Table(show_header=True, box=None)
        table.add_column("PR#", justify="right", style="cyan", width=5)
        table.add_column("Title", style="white", min_width=30)  # More space for title
        table.add_column("💬", justify="center", width=3)  # Comments
        table.add_column("+/-", justify="right", width=14)  # Wider to show full +1454/-234 etc
        table.add_column("📝", justify="right", width=3)  # Commits
        table.add_column("CI", justify="center", width=4)  # CI status
        table.add_column("Review", justify="center", width=6)  # Review status

        for pr in pr_status["prs"]:
            # Format title with truncation - allow longer titles now
            title = pr["title"]
            if len(title) > 35:
                title = title[:32] + "..."

            # Format comments
            comment_count = pr["comments"]
            comments_text = str(comment_count) if comment_count > 0 else "-"
            comments_style = "yellow" if comment_count > 0 else "dim"

            # Format line changes
            additions = pr["additions"]
            deletions = pr["deletions"]
            if additions > 0 or deletions > 0:
                lines_text = f"+{additions}/-{deletions}" if deletions > 0 else f"+{additions}"
                lines_style = (
                    "green"
                    if additions > deletions
                    else "red"
                    if deletions > additions
                    else "yellow"
                )
            else:
                lines_text = "-"
                lines_style = "dim"

            # Format commits
            commits = pr["commits_ahead"]
            commits_text = str(commits) if commits > 0 else "-"
            commits_style = "green" if commits > 0 else "dim"

            # Format CI status
            ci_status = pr.get("ci_status", "unknown")
            if ci_status == "passing":
                ci_text = "✅"
                ci_style = "green"
            elif ci_status == "failing":
                ci_text = "❌"
                ci_style = "red"
            elif ci_status == "pending":
                ci_text = "⏳"
                ci_style = "yellow"
            elif ci_status == "mixed":
                ci_text = "⚠️"
                ci_style = "yellow"
            elif ci_status == "none":
                ci_text = "-"
                ci_style = "dim"
            else:
                ci_text = "?"
                ci_style = "dim"

            # Format review status
            review_status = pr.get("review_status", "pending")
            if review_status == "approved":
                review_text = "✅"
                review_style = "green"
            elif review_status == "changes_requested":
                review_text = "❌"
                review_style = "red"
            elif review_status == "commented":
                review_text = "💬"
                review_style = "yellow"
            else:  # pending
                review_text = "⏳"
                review_style = "dim"

            table.add_row(
                f"#{pr['number']}",
                title,
                f"[{comments_style}]{comments_text}[/{comments_style}]",
                f"[{lines_style}]{lines_text}[/{lines_style}]",
                f"[{commits_style}]{commits_text}[/{commits_style}]",
                f"[{ci_style}]{ci_text}[/{ci_style}]",
                f"[{review_style}]{review_text}[/{review_style}]",
            )

        title_text = f"Pull Requests ({pr_status['total_open']})"
        return Panel(table, title=title_text, border_style="purple")

    def create_header(self) -> Panel:
        """Create the header panel."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text = Text(
            f"📊 CI Dashboard | Updated: {now} | Refresh: {self.refresh_interval}s", justify="center"
        )
        return Panel(header_text, style="bold blue")

    def create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()

        # Main vertical split
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
        )

        # Body split into two columns - make right side wider for PRs
        layout["body"].split_row(
            Layout(name="left", ratio=2),  # 40% width
            Layout(name="right", ratio=3),  # 60% width
        )

        # Left column splits - now includes stats at bottom
        layout["left"].split_column(
            Layout(name="git", size=12),
            Layout(name="tests", size=10),
            Layout(name="stats"),  # Moved from right to left
        )

        # Right column splits into 2 sections (removed stats)
        layout["right"].split_column(
            Layout(name="worktrees"),
            Layout(name="prs"),  # Gets more space now
        )

        return layout

    async def update_display(self) -> Layout:
        """Update all panels in the layout."""
        layout = self.create_layout()

        # Update all panels
        layout["header"].update(self.create_header())
        layout["git"].update(self.create_git_panel())
        layout["worktrees"].update(self.create_worktree_panel())
        layout["prs"].update(self.create_pr_panel())
        layout["tests"].update(self.create_test_panel())
        layout["stats"].update(self.create_stats_panel())

        return layout

    async def run(self) -> None:
        """Main dashboard loop."""
        with Live(
            self.create_layout(), console=self.console, refresh_per_second=1, screen=True
        ) as live:
            while True:
                try:
                    layout = await self.update_display()
                    live.update(layout)
                    await asyncio.sleep(self.refresh_interval)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    await asyncio.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="CI Dashboard for claudelearnspokemon")
    parser.add_argument(
        "--refresh", type=int, default=30, help="Refresh interval in seconds (default: 30)"
    )
    parser.add_argument("--watch", action="store_true", help="Watch mode (no screen clear)")

    args = parser.parse_args()

    # Check if we're in the right environment
    project_root = Path("/home/sd/claudelearnspokemon")
    if not project_root.exists():
        print("Error: Project directory not found at /home/sd/claudelearnspokemon")
        sys.exit(1)

    # Create and run dashboard
    dashboard = CIDashboard(refresh_interval=args.refresh)

    try:
        asyncio.run(dashboard.run())
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
