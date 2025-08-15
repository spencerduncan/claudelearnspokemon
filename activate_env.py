#!/usr/bin/env python3
"""Auto-activation helper for Python scripts in this project."""

import os
import sys
from pathlib import Path


def ensure_venv() -> None:
    """Ensure we're running in the project's virtual environment."""
    project_root = Path(__file__).parent
    venv_python = project_root / "venv" / "bin" / "python"

    # Check if we're already in the right venv
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        # We're in a venv, check if it's the right one
        if str(project_root / "venv") in sys.prefix:
            return  # Already in the correct venv

    # Not in the right venv, try to activate it
    if venv_python.exists():
        print(f"🔄 Switching to project venv: {venv_python}")
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)
    else:
        print("❌ Virtual environment not found. Please run setup.sh first.")
        sys.exit(1)


# Call this at the start of any Python script
if __name__ != "__main__":
    ensure_venv()
