#!/bin/bash

# Installation script for Production Health Monitor
# Sets up systemd service, logging, and email configuration
# Author: Generated for Issue #1 - Health Check Automation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="$SCRIPT_DIR/health_monitor_production.sh"
SERVICE_NAME="routing-health-monitor"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LOG_DIR="/var/log"
LOGROTATE_FILE="/etc/logrotate.d/${SERVICE_NAME}"
CONFIG_DIR="/etc/${SERVICE_NAME}"
CONFIG_FILE="$CONFIG_DIR/config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Validate dependencies
check_dependencies() {
    info "Checking system dependencies..."
    
    local missing_deps=()
    
    # Check required commands
    for cmd in curl jq systemctl; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_deps+=("$cmd")
        fi
    done
    
    # Check if systemd is available
    if ! systemctl --version >/dev/null 2>&1; then
        error "systemd is required but not available"
        exit 1
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        error "Missing required dependencies: ${missing_deps[*]}"
        info "Please install missing dependencies:"
        info "  Ubuntu/Debian: apt-get install ${missing_deps[*]}"
        info "  RHEL/CentOS: yum install ${missing_deps[*]}"
        exit 1
    fi
    
    success "All dependencies are available"
}

# Create systemd service file
create_service_file() {
    info "Creating systemd service file..."
    
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Message Routing Engine Health Monitor
Documentation=file://$SCRIPT_DIR/health_monitor_production.sh
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
Group=root
ExecStart=$MONITOR_SCRIPT monitor
ExecStop=$MONITOR_SCRIPT stop
Restart=always
RestartSec=10
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=30

# Environment
Environment=ALERT_EMAIL=ops@company.com
EnvironmentFile=-$CONFIG_FILE

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/var/log /var/run /tmp
ProtectHome=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF

    success "Created systemd service file: $SERVICE_FILE"
}

# Create configuration directory and file
create_config() {
    info "Creating configuration directory..."
    
    mkdir -p "$CONFIG_DIR"
    
    # Create default configuration
    cat > "$CONFIG_FILE" << EOF
# Health Monitor Configuration
# This file is sourced by the systemd service

# Email configuration
ALERT_EMAIL=ops@company.com

# Optional: Custom health check URL (default: http://localhost:8080/health)
# HEALTH_URL=http://localhost:8080/health

# Optional: Custom check interval in seconds (default: 30)
# CHECK_INTERVAL=30

# Optional: Custom response time threshold in milliseconds (default: 50)
# RESPONSE_THRESHOLD=50

# Optional: Custom alert cooldown in seconds (default: 300)
# ALERT_COOLDOWN=300
EOF

    chmod 600 "$CONFIG_FILE"
    success "Created configuration file: $CONFIG_FILE"
}

# Setup log rotation
setup_logrotate() {
    info "Setting up log rotation..."
    
    cat > "$LOGROTATE_FILE" << 'EOF'
/var/log/routing_health_monitor.log /var/log/routing_health_metrics.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    postrotate
        /bin/systemctl reload-or-restart routing-health-monitor.service >/dev/null 2>&1 || true
    endscript
}
EOF

    success "Created logrotate configuration: $LOGROTATE_FILE"
}

# Configure email system
configure_email() {
    info "Checking email system configuration..."
    
    if command -v mail >/dev/null 2>&1; then
        success "System mail command is available"
    elif command -v sendmail >/dev/null 2>&1; then
        success "Sendmail is available"
    else
        warn "No mail system detected"
        info "To enable email alerts, install one of:"
        info "  Ubuntu/Debian: apt-get install mailutils"
        info "  RHEL/CentOS: yum install mailx"
        info "Or configure postfix/sendmail for local mail delivery"
    fi
}

# Test the health monitor script
test_monitor() {
    info "Testing health monitor script..."
    
    if [[ ! -f "$MONITOR_SCRIPT" ]]; then
        error "Health monitor script not found: $MONITOR_SCRIPT"
        exit 1
    fi
    
    if [[ ! -x "$MONITOR_SCRIPT" ]]; then
        error "Health monitor script is not executable: $MONITOR_SCRIPT"
        exit 1
    fi
    
    # Test basic functionality
    if "$MONITOR_SCRIPT" check >/dev/null 2>&1; then
        success "Health monitor script test passed"
    else
        warn "Health monitor script test failed (service may not be running)"
        info "This is normal if the routing service isn't running yet"
    fi
}

# Install and enable service
install_service() {
    info "Installing and enabling systemd service..."
    
    # Reload systemd to recognize new service
    systemctl daemon-reload
    
    # Enable service to start on boot
    systemctl enable "$SERVICE_NAME"
    
    success "Service enabled: $SERVICE_NAME"
}

# Start service
start_service() {
    info "Starting health monitor service..."
    
    if systemctl start "$SERVICE_NAME"; then
        success "Service started successfully"
        
        # Show status
        sleep 2
        systemctl status "$SERVICE_NAME" --no-pager -l
    else
        error "Failed to start service"
        info "Check service logs: journalctl -u $SERVICE_NAME -f"
        exit 1
    fi
}

# Show final instructions
show_instructions() {
    success "Health monitor installation completed!"
    echo ""
    info "Service Management Commands:"
    echo "  Start:   sudo systemctl start $SERVICE_NAME"
    echo "  Stop:    sudo systemctl stop $SERVICE_NAME"
    echo "  Restart: sudo systemctl restart $SERVICE_NAME"
    echo "  Status:  sudo systemctl status $SERVICE_NAME"
    echo ""
    info "Monitoring Commands:"
    echo "  Check:   $MONITOR_SCRIPT check"
    echo "  Status:  $MONITOR_SCRIPT status"
    echo "  Stop:    $MONITOR_SCRIPT stop"
    echo ""
    info "Log Files:"
    echo "  Monitor: /var/log/routing_health_monitor.log"
    echo "  Metrics: /var/log/routing_health_metrics.log"
    echo "  Service: journalctl -u $SERVICE_NAME -f"
    echo ""
    info "Configuration:"
    echo "  Config:  $CONFIG_FILE"
    echo "  Edit email address and restart service to apply changes"
    echo ""
    warn "Remember to configure your email system for alerts to work!"
}

# Uninstall function
uninstall_service() {
    info "Uninstalling health monitor service..."
    
    # Stop service if running
    if systemctl is-active "$SERVICE_NAME" >/dev/null 2>&1; then
        systemctl stop "$SERVICE_NAME"
        info "Stopped service"
    fi
    
    # Disable service
    if systemctl is-enabled "$SERVICE_NAME" >/dev/null 2>&1; then
        systemctl disable "$SERVICE_NAME"
        info "Disabled service"
    fi
    
    # Remove service file
    if [[ -f "$SERVICE_FILE" ]]; then
        rm -f "$SERVICE_FILE"
        info "Removed service file"
    fi
    
    # Remove logrotate config
    if [[ -f "$LOGROTATE_FILE" ]]; then
        rm -f "$LOGROTATE_FILE"
        info "Removed logrotate configuration"
    fi
    
    # Reload systemd
    systemctl daemon-reload
    
    success "Service uninstalled"
    
    warn "Configuration directory preserved: $CONFIG_DIR"
    warn "Log files preserved: /var/log/routing_health_*"
}

# Main execution
main() {
    case "${1:-install}" in
        "install")
            info "Installing Message Routing Engine Health Monitor..."
            check_root
            check_dependencies
            test_monitor
            create_config
            create_service_file
            setup_logrotate
            configure_email
            install_service
            start_service
            show_instructions
            ;;
        "uninstall")
            info "Uninstalling Message Routing Engine Health Monitor..."
            check_root
            uninstall_service
            ;;
        "test")
            info "Testing health monitor without installation..."
            check_dependencies
            test_monitor
            success "Health monitor script is ready for installation"
            ;;
        "config")
            info "Current configuration:"
            echo ""
            if [[ -f "$CONFIG_FILE" ]]; then
                cat "$CONFIG_FILE"
            else
                warn "Configuration file not found: $CONFIG_FILE"
                info "Run '$0 install' first"
            fi
            ;;
        *)
            echo "Usage: $0 {install|uninstall|test|config}"
            echo ""
            echo "Commands:"
            echo "  install    - Install and start health monitor service (default)"
            echo "  uninstall  - Stop and remove health monitor service"
            echo "  test       - Test health monitor script without installing"
            echo "  config     - Show current configuration"
            echo ""
            echo "Installation creates:"
            echo "  - Systemd service: $SERVICE_NAME"
            echo "  - Configuration: $CONFIG_FILE"
            echo "  - Log rotation: $LOGROTATE_FILE"
            echo ""
            echo "Run as root: sudo $0 install"
            exit 1
            ;;
    esac
}

main "$@"