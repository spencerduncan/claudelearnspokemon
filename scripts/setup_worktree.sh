#!/bin/bash
# Helper script for setting up worktrees with proper environment
# Usage: ./scripts/setup_worktree.sh [worktree-name] [--fresh]
#
# Examples:
#   ./scripts/setup_worktree.sh issue-123 --fresh  # Create new worktree for issue-123
#   ./scripts/setup_worktree.sh issue-91           # Setup existing worktree

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the base directory
BASE_DIR="/home/sd/claudelearnspokemon"
WORKTREE_BASE="/home/sd/worktrees"

# Parse arguments
WORKTREE_NAME="$1"
FRESH_FLAG="$2"

# Function to print colored messages
print_msg() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage if no arguments
if [ -z "$WORKTREE_NAME" ]; then
    echo "Usage: $0 <worktree-name> [--fresh]"
    echo ""
    echo "Examples:"
    echo "  $0 issue-123 --fresh  # Create new worktree for issue-123"
    echo "  $0 issue-91           # Setup existing worktree"
    echo ""
    echo "Available worktrees:"
    if [ -d "$WORKTREE_BASE" ]; then
        ls -1 "$WORKTREE_BASE" 2>/dev/null | sed 's/^/  - /'
    fi
    exit 1
fi

WORKTREE_PATH="$WORKTREE_BASE/$WORKTREE_NAME"

# Handle fresh worktree creation
if [ "$FRESH_FLAG" == "--fresh" ]; then
    if [ -d "$WORKTREE_PATH" ]; then
        print_error "Worktree $WORKTREE_NAME already exists at $WORKTREE_PATH"
        exit 1
    fi

    print_msg "Creating fresh worktree: $WORKTREE_NAME"

    # Ensure we're in the main repo
    cd "$BASE_DIR"

    # Create the worktree
    git worktree add "$WORKTREE_PATH" -b "$WORKTREE_NAME"

    print_msg "Worktree created at $WORKTREE_PATH"
fi

# Check if worktree exists
if [ ! -d "$WORKTREE_PATH" ]; then
    print_error "Worktree $WORKTREE_NAME does not exist at $WORKTREE_PATH"
    print_msg "Use --fresh flag to create a new worktree"
    exit 1
fi

# Navigate to worktree
cd "$WORKTREE_PATH"
print_msg "Working in: $WORKTREE_PATH"

# Copy essential files if they don't exist
if [ ! -f "pyproject.toml" ] && [ -f "$BASE_DIR/pyproject.toml" ]; then
    print_msg "Copying pyproject.toml from main repo"
    cp "$BASE_DIR/pyproject.toml" .
fi

if [ ! -f "setup.sh" ] && [ -f "$BASE_DIR/setup.sh" ]; then
    print_msg "Copying setup.sh from main repo"
    cp "$BASE_DIR/setup.sh" .
    chmod +x setup.sh
fi

if [ ! -f "CLAUDE.md" ] && [ -f "$BASE_DIR/CLAUDE.md" ]; then
    print_msg "Copying CLAUDE.md from main repo"
    cp "$BASE_DIR/CLAUDE.md" .
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    print_msg "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
print_msg "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip quietly
print_msg "Upgrading pip..."
pip install --quiet --upgrade pip

# Install dependencies
print_msg "Installing dependencies..."
if [ -f "pyproject.toml" ]; then
    pip install --quiet -e ".[dev]"
else
    print_warn "No pyproject.toml found, skipping dependency installation"
fi

# Set up pre-commit hooks if available
if command -v pre-commit &> /dev/null && [ -f ".pre-commit-config.yaml" ]; then
    print_msg "Installing pre-commit hooks..."
    pre-commit install --install-hooks
fi

# Create tests directory if it doesn't exist
if [ ! -d "tests" ]; then
    print_msg "Creating tests directory..."
    mkdir -p tests
fi

# Create src directory structure if it doesn't exist
if [ ! -d "src/claudelearnspokemon" ]; then
    print_msg "Creating src/claudelearnspokemon directory structure..."
    mkdir -p src/claudelearnspokemon

    # Create __init__.py if it doesn't exist
    if [ ! -f "src/claudelearnspokemon/__init__.py" ]; then
        touch src/claudelearnspokemon/__init__.py
    fi
fi

# Export PYTHONPATH
export PYTHONPATH="$WORKTREE_PATH/src:$PYTHONPATH"

print_msg "✅ Worktree setup complete!"
echo ""
echo "Environment ready with:"
echo "  - Virtual environment: activated"
echo "  - Dependencies: installed"
echo "  - PYTHONPATH: $PYTHONPATH"
echo "  - Working directory: $WORKTREE_PATH"
echo ""
echo "You can now run tests with:"
echo "  python -m pytest tests/"
echo ""
echo "To activate this environment later:"
echo "  cd $WORKTREE_PATH && source venv/bin/activate"
