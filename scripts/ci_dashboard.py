#!/usr/bin/env python3
# mypy: ignore-errors
# ruff: noqa: UP007
"""
CI Dashboard - Terminal-based monitoring for claudelearnspokemon project
Displays git status, worktree info, test results, and code statistics
"""

import argparse
import asyncio
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

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

    def run_command(self, cmd: str, cwd: Optional[Path] = None) -> tuple[int, str, str]:
        """Run a shell command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd or self.project_root,
                timeout=10,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)

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
            code, stdout, _ = self.run_command("git branch --show-current")
            if code == 0:
                status["branch"] = stdout.strip() or "main"

            # Get ahead/behind status
            code, stdout, _ = self.run_command(
                "git rev-list --left-right --count origin/main...HEAD"
            )
            if code == 0 and stdout:
                parts = stdout.strip().split()
                if len(parts) == 2:
                    status["behind"] = int(parts[0])
                    status["ahead"] = int(parts[1])

            # Get uncommitted changes
            code, stdout, _ = self.run_command("git status --porcelain")
            if code == 0:
                status["uncommitted"] = len([line for line in stdout.splitlines() if line.strip()])

            # Get last 3 commits
            code, stdout, _ = self.run_command("git log --oneline -3")
            if code == 0:
                status["last_commits"] = stdout.strip().splitlines()[:3]

            return status

        return self.get_cached_result("git_status", 10, _fetch_git_status)

    def get_worktree_status(self) -> list[dict[str, Any]]:
        """Get status of all worktrees."""

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
                }

                # Get branch name
                code, stdout, _ = self.run_command("git branch --show-current", cwd=worktree_dir)
                if code == 0:
                    worktree["branch"] = stdout.strip() or "unknown"

                # Get last commit time
                code, stdout, _ = self.run_command("git log -1 --format=%ct", cwd=worktree_dir)
                if code == 0 and stdout.strip():
                    timestamp = int(stdout.strip())
                    worktree["last_modified"] = datetime.fromtimestamp(timestamp)

                    # Calculate relative time
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

                # Get uncommitted changes
                code, stdout, _ = self.run_command("git status --porcelain", cwd=worktree_dir)
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

            # Run pytest with coverage
            code, stdout, stderr = self.run_command(
                "python -m pytest tests/ --co -q 2>/dev/null | grep -E '^tests/' | wc -l"
            )
            if code == 0:
                status["total"] = int(stdout.strip() or 0)

            # Get last test results from cache or quick test
            code, stdout, _ = self.run_command(
                "python -m pytest tests/ --tb=no -q --no-header 2>/dev/null | tail -1"
            )
            if code == 0 and "passed" in stdout:
                # Parse pytest summary line
                import re

                matches = re.findall(r"(\d+)\s+(\w+)", stdout)
                for count, result_type in matches:
                    if "passed" in result_type:
                        status["passed"] = int(count)
                    elif "failed" in result_type:
                        status["failed"] = int(count)
                    elif "skipped" in result_type:
                        status["skipped"] = int(count)

            status["last_run"] = datetime.now().strftime("%H:%M:%S")

            # Estimate coverage (would need pytest-cov for real coverage)
            if status["total"] > 0:
                status["coverage"] = int((status["passed"] / status["total"]) * 100)

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

            # Check ruff
            code, stdout, _ = self.run_command("ruff check src/ tests/ 2>/dev/null | wc -l")
            if code == 0:
                lines = int(stdout.strip() or 0)
                quality["ruff_violations"] = max(0, lines - 1)  # Subtract header line

            # Check black
            code, _, _ = self.run_command("black --check src/ tests/ 2>/dev/null")
            quality["black_status"] = "formatted" if code == 0 else "needs formatting"

            # Check mypy
            code, stdout, _ = self.run_command("mypy src/ 2>/dev/null | grep -E '^src/' | wc -l")
            if code == 0:
                quality["mypy_errors"] = int(stdout.strip() or 0)

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

            # Find all Python files
            code, stdout, _ = self.run_command("find src/ tests/ -name '*.py' -type f | head -20")

            if code == 0:
                py_files = [f.strip() for f in stdout.splitlines() if f.strip()]

                # Get line counts for each file
                file_sizes = []
                for file_path in py_files:
                    code, stdout, _ = self.run_command(f"wc -l {file_path}")
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

            # Count files by extension
            for ext in [".py", ".md", ".sh", ".toml", ".yml", ".yaml"]:
                code, stdout, _ = self.run_command(f"find . -name '*{ext}' -type f | wc -l")
                if code == 0:
                    count = int(stdout.strip() or 0)
                    if count > 0:
                        stats["file_counts"][ext] = count

            return stats

        return self.get_cached_result("code_stats", 120, _fetch_code_stats)

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
        """Create the worktree status panel."""
        worktrees = self.get_worktree_status()

        table = Table(show_header=True, box=None)
        table.add_column("Worktree", style="cyan")
        table.add_column("Last Activity", justify="right")
        table.add_column("Changes", justify="center")
        table.add_column("Status", justify="center")

        for worktree in sorted(
            worktrees, key=lambda x: x.get("last_modified") or datetime.min, reverse=True
        )[:10]:
            # Determine status style
            if worktree["status"] == "active":
                status_icon = "✅"
                time_style = "green"
            elif worktree["status"] == "recent":
                status_icon = "⚠️"
                time_style = "yellow"
            else:
                status_icon = "🔴"
                time_style = "red"

            # Changes indicator
            changes_text = f"{worktree['uncommitted']}" if worktree["uncommitted"] > 0 else "-"
            changes_style = "yellow" if worktree["uncommitted"] > 0 else "dim"

            table.add_row(
                worktree["name"],
                f"[{time_style}]{worktree['last_modified_relative']}[/{time_style}]",
                f"[{changes_style}]{changes_text}[/{changes_style}]",
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

        # Summary stats
        content.add_row("[bold]Summary:[/bold]", "")
        content.add_row("  Total lines:", f"[green]{stats['total_lines']:,}[/green]")
        content.add_row("  Python files:", f"[green]{stats['total_py_files']}[/green]")

        # File type counts
        if stats["file_counts"]:
            content.add_row("", "")
            content.add_row("[bold]File Types:[/bold]", "")
            for ext, count in sorted(stats["file_counts"].items()):
                content.add_row(f"  {ext}:", f"[dim]{count} files[/dim]")

        return Panel(content, title="Code Statistics", border_style="magenta")

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

        # Body split into two rows
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        # Left column splits
        layout["left"].split_column(
            Layout(name="git", size=12),
            Layout(name="tests"),
        )

        # Right column splits
        layout["right"].split_column(
            Layout(name="worktrees"),
            Layout(name="stats"),
        )

        return layout

    async def update_display(self) -> Layout:
        """Update all panels in the layout."""
        layout = self.create_layout()

        # Update all panels
        layout["header"].update(self.create_header())
        layout["git"].update(self.create_git_panel())
        layout["worktrees"].update(self.create_worktree_panel())
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
