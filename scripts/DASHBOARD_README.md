# CI Dashboard

A dead-simple terminal dashboard for monitoring your claudelearnspokemon project.

## Quick Start

```bash
# Run the dashboard
./scripts/dashboard.sh

# Custom refresh interval (10 seconds)
./scripts/dashboard.sh --refresh 10

# Watch mode (no screen clear)
./scripts/dashboard.sh --watch
```

## Features

The dashboard displays 4 main panels:

1. **Git Status**
   - Current branch and sync status with main
   - Uncommitted changes count
   - Recent commits

2. **Worktree Status**
   - All worktrees in `/home/sd/worktrees/`
   - Last modified time with color coding:
     - ✅ Green: Active (< 1 hour)
     - ⚠️ Yellow: Recent (< 1 day)
     - 🔴 Red: Stale (> 1 day)
   - Uncommitted changes per worktree

3. **Tests & Code Quality**
   - Test results (pass/fail counts)
   - Coverage percentage with visual bar
   - Ruff violations count
   - Black formatting status
   - Mypy type errors
   - Pre-commit hooks status

4. **Code Statistics**
   - Top 5 largest source files
   - Total lines of code
   - File counts by type

## Controls

- **Ctrl+C**: Stop the dashboard
- Auto-refreshes every 30 seconds (configurable)

## Requirements

- Python 3.10+
- `rich` library (automatically installed)

## Troubleshooting

If the dashboard doesn't start:
1. Ensure you're in the project directory
2. Run `./setup.sh` to set up the environment
3. Check that `rich` is installed: `pip install rich`

## Customization

Edit `scripts/ci_dashboard.py` to:
- Change refresh interval default
- Add/remove panels
- Modify cache durations
- Adjust colors and styles
