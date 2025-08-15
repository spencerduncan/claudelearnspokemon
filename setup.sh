#!/bin/bash
# Automatic environment setup script for Claude instances

set -e

echo "🔧 Setting up Python environment for claudelearnspokemon..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Make sure you're in the project root."
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --quiet --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install --quiet -e ".[dev]"

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install --install-hooks

echo "✅ Environment setup complete!"
echo "🎮 Virtual environment activated. You're ready to work on the Pokemon speedrun agent!"
