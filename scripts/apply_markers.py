#!/usr/bin/env python3
"""
Script to apply pytest markers to test files based on profiling results.

Reads profile_results.json and adds @pytest.mark.fast, @pytest.mark.medium,
or @pytest.mark.slow markers to the appropriate test files.
"""

import json
import re
from pathlib import Path


class TestMarkerApplier:
    """Apply pytest markers to test files based on profiling."""

    def __init__(self, profile_results_file: str = "profile_results.json"):
        self.project_root = Path(__file__).parent.parent
        self.profile_file = self.project_root / profile_results_file
        self.results = self.load_profile_results()

    def load_profile_results(self) -> dict:
        """Load profiling results from JSON file."""
        if not self.profile_file.exists():
            raise FileNotFoundError(f"Profile results not found: {self.profile_file}")

        with open(self.profile_file) as f:
            data = json.load(f)

        return data["results"]

    def has_pytest_mark_import(self, content: str) -> bool:
        """Check if file already imports pytest."""
        # Look for pytest import
        return bool(
            re.search(r"^import pytest", content, re.MULTILINE)
            or re.search(r"^from.*pytest.*import", content, re.MULTILINE)
        )

    def add_pytest_import(self, content: str) -> str:
        """Add pytest import if not present."""
        lines = content.split("\n")

        # Find the best place to add import (after docstring, before first class/def)
        insert_idx = 0
        in_docstring = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip module docstring
            if i == 0 and stripped.startswith('"""'):
                if stripped.count('"""') == 1:
                    in_docstring = True
                continue
            elif in_docstring and '"""' in stripped:
                in_docstring = False
                insert_idx = i + 1
                continue
            elif in_docstring:
                continue

            # Look for imports section
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_idx = i + 1
            elif stripped and not stripped.startswith("#"):
                # Found first non-import, non-comment line
                break

        # Insert import
        lines.insert(insert_idx, "import pytest")
        if insert_idx < len(lines) - 1:
            lines.insert(insert_idx + 1, "")  # Add blank line

        return "\n".join(lines)

    def add_marker_to_class(self, content: str, marker: str) -> str:
        """Add marker to all test classes in the file."""
        lines = content.split("\n")
        new_lines = []

        for line in lines:
            stripped_line = line.strip()
            # Look for test class definitions (both pytest and unittest styles)
            if (
                re.match(r"^class Test\w*:", stripped_line)  # pytest style: class TestFoo:
                or re.match(
                    r"^class Test\w*\(.*TestCase.*\):", stripped_line
                )  # unittest style: class TestFoo(unittest.TestCase):
                or re.match(r"^class Test\w*\(.*\):", stripped_line)
            ):  # any other test class: class TestFoo(SomeBase):
                new_lines.append(f"@pytest.mark.{marker}")
                new_lines.append(line)
            else:
                new_lines.append(line)

        return "\n".join(new_lines)

    def apply_marker_to_file(self, test_file: Path, category: str) -> None:
        """Apply the appropriate marker to a test file."""
        print(f"Applying @pytest.mark.{category} to {test_file.name}")

        # Read current content
        content = test_file.read_text()

        # Check if pytest import is needed
        if not self.has_pytest_mark_import(content):
            content = self.add_pytest_import(content)

        # Check if marker already exists
        if f"@pytest.mark.{category}" in content:
            print("  ⚠️  Marker already exists, skipping")
            return

        # Add marker to classes
        updated_content = self.add_marker_to_class(content, category)

        # Write back to file
        test_file.write_text(updated_content)
        print(f"  ✓ Added @pytest.mark.{category}")

    def apply_all_markers(self) -> None:
        """Apply markers to all test files based on profiling results."""
        print("🏷️  Applying test markers based on profiling results...")

        for test_path, result in self.results.items():
            test_file = self.project_root / test_path
            category = result["category"]

            if not test_file.exists():
                print(f"❌ Test file not found: {test_file}")
                continue

            self.apply_marker_to_file(test_file, category)

        print("\n✅ Marker application complete!")

    def show_summary(self) -> None:
        """Show summary of marker application."""
        categories = {}
        for result in self.results.values():
            cat = result["category"]
            categories[cat] = categories.get(cat, 0) + 1

        print("\n📊 MARKER SUMMARY:")
        for category, count in sorted(categories.items()):
            print(f"   @pytest.mark.{category}: {count} test files")


def main():
    """Main marker application function."""
    try:
        applier = TestMarkerApplier()
        applier.show_summary()
        applier.apply_all_markers()

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Run 'python scripts/profile_tests.py' first to generate profiling data")
    except Exception as e:
        print(f"❌ Failed to apply markers: {e}")


if __name__ == "__main__":
    main()
