#!/bin/bash
# Automatic environment setup script for Claude instances

set -e

echo "🔧 Setting up Python environment for claudelearnspokemon..."

# Check if we're in a worktree
CURRENT_DIR=$(pwd)
if [[ "$CURRENT_DIR" != /home/sd/worktrees/* ]]; then
    echo "❌ Error: setup.sh must be run from within a worktree."
    echo "   Navigate to a worktree first: cd /home/sd/worktrees/issue-X"
    echo "   Or create a new worktree: /home/sd/claudelearnspokemon/scripts/setup_worktree.sh issue-X --fresh"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Make sure you're in the worktree root."
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Set PYTHONPATH for worktree
export PYTHONPATH="$CURRENT_DIR:$PYTHONPATH"
echo "🔧 Set PYTHONPATH to include worktree: $CURRENT_DIR"

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --quiet --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install --quiet -e ".[dev]"

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install --install-hooks

echo "✅ Worktree environment setup complete!"
echo "🎮 Virtual environment activated with proper PYTHONPATH. You're ready to work on this issue!"
echo "📁 Working in: $CURRENT_DIR"
