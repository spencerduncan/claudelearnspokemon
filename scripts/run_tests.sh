#!/bin/bash
# Smart test runner that handles environment setup automatically
# Usage: ./scripts/run_tests.sh [test-path] [pytest-args]
#
# Examples:
#   ./scripts/run_tests.sh                                    # Run all tests
#   ./scripts/run_tests.sh tests/test_emulator_pool.py       # Run specific test file
#   ./scripts/run_tests.sh tests/test_emulator_pool.py -v    # With verbose output
#   ./scripts/run_tests.sh tests/test_checkpoint_manager.py::TestCheckpointManagerBasics

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_msg() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Detect current directory and find project root
CURRENT_DIR="$(pwd)"
PROJECT_ROOT=""
WORKTREE_NAME=""

# Check if we're in a worktree
if [[ "$CURRENT_DIR" == */worktrees/* ]]; then
    # Extract worktree name
    WORKTREE_NAME=$(echo "$CURRENT_DIR" | sed -n 's|.*/worktrees/\([^/]*\).*|\1|p')
    # Find worktree root
    if [ -n "$WORKTREE_NAME" ]; then
        PROJECT_ROOT="/home/sd/worktrees/$WORKTREE_NAME"
    fi
elif [[ "$CURRENT_DIR" == */claudelearnspokemon* ]]; then
    # We're in the main repo
    PROJECT_ROOT="/home/sd/claudelearnspokemon"
else
    print_error "Not in a recognized project directory"
    print_info "Please run from within claudelearnspokemon or a worktree"
    exit 1
fi

# Navigate to project root if not already there
if [ "$CURRENT_DIR" != "$PROJECT_ROOT" ]; then
    print_msg "Navigating to project root: $PROJECT_ROOT"
    cd "$PROJECT_ROOT"
fi

# Check for virtual environment
if [ ! -d "venv" ]; then
    print_warn "Virtual environment not found"
    print_msg "Running setup.sh to create environment..."
    if [ -f "setup.sh" ]; then
        ./setup.sh
    else
        print_error "setup.sh not found. Cannot set up environment."
        exit 1
    fi
fi

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_msg "Activating virtual environment..."
    source venv/bin/activate
else
    print_info "Virtual environment already active: $VIRTUAL_ENV"
fi

# Set PYTHONPATH to include src directory
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
print_info "PYTHONPATH set to: $PYTHONPATH"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_warn "pytest not found, installing..."
    pip install --quiet pytest pytest-asyncio pytest-mock
fi

# Parse test arguments
TEST_PATH="${1:-tests}"  # Default to all tests
shift || true  # Remove first argument
PYTEST_ARGS="$@"  # Remaining arguments are pytest flags

# Add default pytest options if none provided
if [ -z "$PYTEST_ARGS" ]; then
    PYTEST_ARGS="-v"  # Default to verbose
fi

# Check if test path exists
if [ ! -e "$TEST_PATH" ]; then
    print_error "Test path not found: $TEST_PATH"
    print_info "Available test files:"
    if [ -d "tests" ]; then
        find tests -name "*.py" -type f | sed 's/^/  - /'
    else
        print_warn "No tests directory found"
    fi
    exit 1
fi

# Run the tests
print_msg "Running tests: $TEST_PATH"
print_info "Arguments: $PYTEST_ARGS"
echo "────────────────────────────────────────────────────────"

# Run pytest with proper error handling
if python -m pytest "$TEST_PATH" $PYTEST_ARGS; then
    echo "────────────────────────────────────────────────────────"
    print_msg "✅ Tests passed successfully!"
else
    EXIT_CODE=$?
    echo "────────────────────────────────────────────────────────"
    print_error "❌ Tests failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi
