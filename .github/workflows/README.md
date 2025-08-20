# No CI Setup

Don't setup CI here, we aren't paying GitHub for this.

## Local Testing Instructions

Run these commands locally before submitting PRs:

```bash
# Set up environment
./setup.sh

# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=claudelearnspokemon tests/

# Run specific test files
python -m pytest tests/test_memory_graph.py -v

# Run linting
python -m ruff check src/ tests/
python -m mypy src/

# Run formatting
python -m ruff format src/ tests/

# Use the smart test runner (recommended)
./scripts/run_tests.sh tests/test_memory_graph.py -v
```

## Quality Control

All code quality checks must pass locally before PR submission:
- All tests passing
- No linting errors (ruff)
- No type errors (mypy)
- Proper formatting (ruff format)
