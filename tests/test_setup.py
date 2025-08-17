"""Basic test to verify project setup."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.fast
def test_project_imports() -> None:
    """Test that the main package can be imported."""
    import claudelearnspokemon

    assert claudelearnspokemon.__version__ == "0.1.0"


@pytest.mark.fast
def test_basic_assertion() -> None:
    """Simple test to verify pytest is working."""
    assert 1 + 1 == 2
