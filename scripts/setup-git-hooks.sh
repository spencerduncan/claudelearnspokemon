#!/bin/bash
# Setup git hooks for auto-fixing code issues
set -e

echo "🔧 Setting up git hooks for auto-fixing..."

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "❌ pre-commit not found. Installing..."
    pip install pre-commit
fi

# Install pre-commit hooks
echo "📦 Installing pre-commit hooks..."
pre-commit install

# Configure git for better auto-fixing workflow
echo "⚙️ Configuring git settings..."

# Make git automatically stage changes made by pre-commit hooks
git config --local core.autocrlf false
git config --local hooks.autoStageFixedFiles true

# Configure git to be more helpful with pre-commit
git config --local advice.addIgnoredFile false

echo "✅ Git hooks configured successfully!"
echo ""
echo "📋 What this setup does:"
echo "   • Auto-fixes formatting (Black, Ruff) on commit"
echo "   • Auto-applies test markers when needed"
echo "   • Runs fast tests before each commit"
echo "   • Only blocks commits on real failures (not style issues)"
echo ""
echo "🚀 Usage:"
echo "   • Make your changes and commit normally"
echo "   • Pre-commit hooks will auto-fix style issues"
echo "   • Fixed files are automatically staged"
echo "   • Only real failures (tests, type errors) will block commits"
echo ""
echo "🔍 To run hooks manually:"
echo "   pre-commit run --all-files         # Run all hooks"
echo "   pre-commit run ruff --all-files    # Run just ruff"
echo "   pre-commit run pytest-fast         # Run just fast tests"
echo ""
echo "🎯 Pro tip: The CI will validate the same checks, so if hooks"
echo "   pass locally, CI should pass too!"
