#!/usr/bin/env python3
"""
Script to mark remaining tests that don't have speed markers with a default marker.
Since most tests should be fast by default, this marks all remaining tests as @pytest.mark.fast.
"""

import re
import subprocess
from pathlib import Path


def get_unmarked_test_files():
    """Get list of test files that contain tests without markers."""
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

    if result.returncode != 0:
        print("❌ Error running pytest to find unmarked tests")
        return []

    # Parse output to extract test file names
    test_files = set()
    lines = result.stdout.split("\n")

    for line in lines:
        # Look for lines like "<Module test_something.py>"
        if "<Module test_" in line and ".py>" in line:
            # Extract filename from line like "    <Module test_something.py>"
            parts = line.split()
            for part in parts:
                if part.startswith("test_") and part.endswith(".py>"):
                    filename = part.rstrip(">")
                    test_files.add(filename)

    return list(test_files)


def add_fast_marker_to_file(test_file_path: Path):
    """Add @pytest.mark.fast to all test classes in the file."""
    content = test_file_path.read_text()

    # Check if pytest import exists
    if not re.search(r"^import pytest", content, re.MULTILINE):
        # Add pytest import
        lines = content.split("\n")
        insert_idx = 0

        # Find good place to insert import
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ")):
                insert_idx = i + 1
            elif line.strip() and not line.strip().startswith("#"):
                break

        lines.insert(insert_idx, "import pytest")
        lines.insert(insert_idx + 1, "")
        content = "\n".join(lines)

    # Add markers to test classes AND test functions
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        stripped_line = line.strip()

        # Look for test class definitions (both pytest and unittest styles)
        if (
            re.match(r"^class Test\w*:", stripped_line)  # pytest style
            or re.match(r"^class Test\w*\(.*TestCase.*\):", stripped_line)  # unittest style
            or re.match(r"^class Test\w*\(.*\):", stripped_line)
        ):  # any other test class

            # Check if this line already has a marker above it
            if new_lines and "@pytest.mark." in new_lines[-1]:
                # Already has a marker, just add the line
                new_lines.append(line)
            else:
                # Add fast marker
                new_lines.append("@pytest.mark.fast")
                new_lines.append(line)

        # Also look for standalone test functions
        elif re.match(r"^def test_\w*\(", stripped_line):  # test functions
            # Check if this line already has a marker above it
            if new_lines and "@pytest.mark." in new_lines[-1]:
                # Already has a marker, just add the line
                new_lines.append(line)
            else:
                # Add fast marker
                new_lines.append("@pytest.mark.fast")
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Write back to file
    updated_content = "\n".join(new_lines)
    if updated_content != content:
        test_file_path.write_text(updated_content)
        print(f"  ✓ Added @pytest.mark.fast to {test_file_path.name}")
        return True
    else:
        print(f"  ⚠️  No changes needed for {test_file_path.name}")
        return False


def main():
    print("🔍 Finding test files with unmarked tests...")

    unmarked_files = get_unmarked_test_files()
    if not unmarked_files:
        print("✅ No unmarked test files found!")
        return

    print(f"📝 Found {len(unmarked_files)} test files with unmarked tests:")
    for filename in sorted(unmarked_files):
        print(f"  - {filename}")

    print(f"\n🏷️  Adding @pytest.mark.fast to unmarked tests in {len(unmarked_files)} files...")

    project_root = Path(__file__).parent.parent
    updated_files = 0

    for filename in unmarked_files:
        test_file = project_root / "tests" / filename
        if test_file.exists():
            if add_fast_marker_to_file(test_file):
                updated_files += 1
        else:
            print(f"  ❌ File not found: {test_file}")

    print(f"\n✅ Updated {updated_files} files with fast markers!")

    # Verify the fix
    print("\n🔍 Verifying all tests now have markers...")
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

    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        summary_line = [line for line in lines if "tests collected" in line]

        if summary_line and summary_line[0].startswith("="):
            parts = summary_line[0].split()
            if len(parts) > 1:
                count_part = parts[1]  # Should be "X/403"
                unmarked_count = int(count_part.split("/")[0])

                if unmarked_count == 0:
                    print("🎉 SUCCESS: All tests now have speed markers!")
                else:
                    print(f"⚠️  Still have {unmarked_count} tests without markers")
            else:
                print("✅ No unmarked tests found!")


if __name__ == "__main__":
    main()
