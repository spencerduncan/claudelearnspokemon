#!/usr/bin/env python3
"""Verify that the development environment is properly configured."""

import sys
from pathlib import Path


def check_venv() -> bool:
    """Check if we're running in the project's virtual environment."""
    venv_path = Path(__file__).parent / "venv"
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        and str(venv_path) in sys.prefix
    )


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    try:
        import black  # noqa: F401
        import docker  # noqa: F401
        import mypy  # noqa: F401
        import numpy  # noqa: F401
        import pytest  # noqa: F401
        import ruff  # noqa: F401

        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e.name}")
        return False


def check_precommit() -> bool:
    """Check if pre-commit hooks are installed."""
    git_hooks = Path(__file__).parent / ".git" / "hooks" / "pre-commit"
    return git_hooks.exists()


def main() -> int:
    """Run all verification checks."""
    print("🔍 Verifying claudelearnspokemon development environment...")

    checks = {
        "Virtual environment": check_venv(),
        "Dependencies": check_dependencies(),
        "Pre-commit hooks": check_precommit(),
    }

    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}: {'OK' if passed else 'FAILED'}")
        all_passed = all_passed and passed

    if not all_passed:
        print("\n⚠️ Some checks failed. Run './setup.sh' to fix the environment.")
        return 1

    print("\n✅ All checks passed! Environment is properly configured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
