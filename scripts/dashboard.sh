#!/bin/bash
# CI Dashboard launcher script

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_msg() {
    echo -e "${GREEN}[DASHBOARD]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Detect project root
PROJECT_ROOT="/home/sd/claudelearnspokemon"

if [ ! -d "$PROJECT_ROOT" ]; then
    print_error "Project directory not found at $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT"

# Check for virtual environment
if [ ! -d "venv" ]; then
    print_msg "Virtual environment not found. Setting up..."
    if [ -f "setup.sh" ]; then
        ./setup.sh
    else
        print_error "setup.sh not found. Cannot set up environment."
        exit 1
    fi
fi

# Activate virtual environment
print_msg "Activating virtual environment..."
source venv/bin/activate

# Check if rich is installed
if ! python -c "import rich" 2>/dev/null; then
    print_msg "Installing required dependencies..."
    pip install --quiet rich
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Parse command line arguments
REFRESH_INTERVAL=30
WATCH_MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --refresh)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        --watch)
            WATCH_MODE="--watch"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --refresh SECONDS    Set refresh interval (default: 30)"
            echo "  --watch             Run in watch mode (no screen clear)"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Clear screen for better display
clear

# Print startup message
print_msg "Starting CI Dashboard"
print_info "Refresh interval: ${REFRESH_INTERVAL}s"
print_info "Press Ctrl+C to stop"
echo ""

# Run the dashboard
exec python scripts/ci_dashboard.py --refresh "$REFRESH_INTERVAL" $WATCH_MODE
