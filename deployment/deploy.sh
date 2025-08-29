#!/bin/bash

# Message Routing Engine Deployment Orchestrator
# Executes the accelerated shadow-to-production rollout strategy
# 
# Usage: ./deploy.sh [phase] [action]
#   phase: shadow|partial|full|rollback
#   action: start|stop|status|validate

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
COMPOSE_PROJECT="pokemon-routing-engine"
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$SCRIPT_DIR/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] $*" | tee -a "$LOG_DIR/deploy.log"
}

log_info() { log "${BLUE}INFO${NC}" "$@"; }
log_warn() { log "${YELLOW}WARN${NC}" "$@"; }
log_error() { log "${RED}ERROR${NC}" "$@"; }
log_success() { log "${GREEN}SUCCESS${NC}" "$@"; }

# Initialize directories
init_environment() {
    mkdir -p "$LOG_DIR" "$RESULTS_DIR"
    
    # Clear old logs (keep last 5 files)
    find "$LOG_DIR" -name "deploy.log.*" -type f | sort -r | tail -n +6 | xargs rm -f
    
    # Rotate current log
    if [[ -f "$LOG_DIR/deploy.log" ]]; then
        mv "$LOG_DIR/deploy.log" "$LOG_DIR/deploy.log.$(date +%Y%m%d_%H%M%S)"
    fi
    
    log_info "Deployment environment initialized"
}

# Health check function
check_health() {
    local service=$1
    local endpoint=$2
    local max_attempts=${3:-30}
    local wait_time=${4:-5}
    
    log_info "Checking health of $service..."
    
    for ((i=1; i<=max_attempts; i++)); do
        if curl -sf "$endpoint" > /dev/null 2>&1; then
            log_success "$service is healthy"
            return 0
        fi
        
        if [[ $i -eq $max_attempts ]]; then
            log_error "$service failed health check after $max_attempts attempts"
            return 1
        fi
        
        log_info "Attempt $i/$max_attempts failed, waiting ${wait_time}s..."
        sleep $wait_time
    done
}

# Validate SLA compliance
validate_sla() {
    local phase=$1
    log_info "Validating SLA compliance for $phase phase..."
    
    # Get metrics from Prometheus
    local prometheus_url="http://localhost:9090"
    
    # Query P95 latency
    local p95_query="histogram_quantile(0.95, routing_duration_seconds_bucket{system=\"new_routing_engine\"})"
    local p95_result=$(curl -s "${prometheus_url}/api/v1/query?query=${p95_query}" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
    
    # Query success rate
    local success_query="rate(routing_requests_total{status=\"success\",system=\"new_routing_engine\"}[5m]) / rate(routing_requests_total{system=\"new_routing_engine\"}[5m]) * 100"
    local success_result=$(curl -s "${prometheus_url}/api/v1/query?query=${success_query}" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
    
    # Convert to milliseconds and get thresholds based on phase
    local p95_ms=$(echo "$p95_result * 1000" | bc -l 2>/dev/null || echo "0")
    
    case $phase in
        "shadow")
            local p95_threshold=45.0
            local success_threshold=98.5
            ;;
        "partial")
            local p95_threshold=48.0
            local success_threshold=99.0
            ;;
        "full")
            local p95_threshold=50.0
            local success_threshold=99.13
            ;;
        *)
            log_error "Unknown phase: $phase"
            return 1
            ;;
    esac
    
    log_info "Current metrics: P95=${p95_ms}ms (threshold: ${p95_threshold}ms), Success=${success_result}% (threshold: ${success_threshold}%)"
    
    # Check SLA compliance
    local sla_passed=true
    
    if (( $(echo "$p95_ms > $p95_threshold" | bc -l) )); then
        log_error "P95 latency ${p95_ms}ms exceeds threshold ${p95_threshold}ms"
        sla_passed=false
    fi
    
    if (( $(echo "$success_result < $success_threshold" | bc -l) )); then
        log_error "Success rate ${success_result}% below threshold ${success_threshold}%"
        sla_passed=false
    fi
    
    if [[ $sla_passed == true ]]; then
        log_success "SLA compliance validated for $phase phase"
        return 0
    else
        log_error "SLA validation failed for $phase phase"
        return 1
    fi
}

# Phase 1: Shadow Deployment
deploy_shadow() {
    log_info "Starting Phase 1: Shadow Deployment (48 hours)"
    log_info "Configuration: 10% mirrored traffic, zero production impact"
    
    cd "$SCRIPT_DIR/phase1-shadow"
    
    # Stop any existing deployment
    docker-compose -p "$COMPOSE_PROJECT" down --remove-orphans || true
    
    # Build and start shadow deployment
    log_info "Building routing engine image..."
    docker-compose -p "$COMPOSE_PROJECT" build
    
    log_info "Starting shadow deployment stack..."
    docker-compose -p "$COMPOSE_PROJECT" up -d
    
    # Wait for services to be healthy
    sleep 30
    
    # Health checks
    check_health "Traffic Splitter" "http://localhost/health" || return 1
    check_health "Routing Engine" "http://localhost/health/routing" || return 1
    check_health "Prometheus" "http://localhost:9090/-/healthy" || return 1
    check_health "Grafana" "http://localhost:3000/api/health" || return 1
    
    # Start performance testing
    log_info "Starting shadow testing validation..."
    docker-compose -p "$COMPOSE_PROJECT" run --rm performance-tester &
    
    log_success "Phase 1 shadow deployment started successfully"
    log_info "Monitor at: http://localhost:3000 (Grafana) | http://localhost:9090 (Prometheus)"
    log_info "Performance testing will run for 48 hours"
    
    # Save deployment state
    echo "shadow" > "$RESULTS_DIR/current_phase"
    echo "$(date -Iseconds)" > "$RESULTS_DIR/shadow_start_time"
}

# Phase 2: Partial Rollout
deploy_partial() {
    local percentage=${1:-25}
    log_info "Starting Phase 2: Partial Rollout ($percentage% live traffic)"
    
    # Validate shadow phase completion
    if [[ ! -f "$RESULTS_DIR/shadow_validation_passed" ]]; then
        log_error "Shadow validation not passed. Cannot proceed to partial rollout."
        return 1
    fi
    
    cd "$SCRIPT_DIR"
    
    # Update configuration for partial rollout
    log_info "Updating configuration for $percentage% traffic..."
    
    # Create partial rollout docker-compose from template
    envsubst < phase2-partial/docker-compose.partial.template.yml > phase2-partial/docker-compose.partial.yml \
        <<< "TRAFFIC_PERCENTAGE=$percentage"
    
    # Deploy partial configuration
    cd phase2-partial
    docker-compose -p "$COMPOSE_PROJECT" up -d --scale message-routing-engine=2
    
    # Health checks
    check_health "Partial Routing" "http://localhost/health/routing" || return 1
    
    # Validate SLA compliance for 1 hour before proceeding
    log_info "Monitoring SLA compliance for partial rollout..."
    for i in {1..12}; do  # 12 * 5 minutes = 1 hour
        sleep 300  # Wait 5 minutes
        if validate_sla "partial"; then
            log_info "SLA check $i/12 passed"
        else
            log_error "SLA check failed. Initiating rollback..."
            rollback_deployment
            return 1
        fi
    done
    
    log_success "Phase 2 partial rollout ($percentage%) completed successfully"
    echo "partial_$percentage" > "$RESULTS_DIR/current_phase"
}

# Phase 3: Full Production Rollout
deploy_full() {
    log_info "Starting Phase 3: Full Production Rollout (100% traffic)"
    
    # Validate partial phase completion
    if [[ ! -f "$RESULTS_DIR/partial_validation_passed" ]]; then
        log_error "Partial rollout validation not passed. Cannot proceed to full rollout."
        return 1
    fi
    
    cd "$SCRIPT_DIR"
    
    # Deploy full production configuration
    log_info "Deploying full production configuration..."
    cd phase3-full
    docker-compose -p "$COMPOSE_PROJECT" up -d --scale message-routing-engine=3
    
    # Health checks
    check_health "Full Production" "http://localhost/health/routing" || return 1
    
    # Monitor for 2 hours before declaring success
    log_info "Monitoring full production deployment..."
    for i in {1..24}; do  # 24 * 5 minutes = 2 hours
        sleep 300
        if validate_sla "full"; then
            log_info "Full production SLA check $i/24 passed"
        else
            log_error "Full production SLA check failed. Initiating rollback..."
            rollback_deployment
            return 1
        fi
    done
    
    # Decommission legacy components
    log_info "Decommissioning legacy routing components..."
    docker stop pokemon-gym-legacy || true
    
    log_success "Phase 3 full production rollout completed successfully!"
    log_success "Message Routing Engine deployment COMPLETED"
    
    echo "full" > "$RESULTS_DIR/current_phase"
    echo "$(date -Iseconds)" > "$RESULTS_DIR/deployment_completed"
}

# Rollback function
rollback_deployment() {
    local current_phase=$(cat "$RESULTS_DIR/current_phase" 2>/dev/null || echo "unknown")
    log_warn "Initiating emergency rollback from phase: $current_phase"
    
    # Stop current deployment
    docker-compose -p "$COMPOSE_PROJECT" down || true
    
    # Restart legacy system
    log_info "Restarting legacy Pokemon Gym service..."
    cd "$PROJECT_ROOT/docker/pokemon-gym"
    docker-compose up -d
    
    # Verify legacy system health
    check_health "Legacy System" "http://localhost:8080/health" || {
        log_error "CRITICAL: Legacy system failed to start during rollback!"
        return 1
    }
    
    log_success "Rollback completed - Legacy system restored"
    echo "rollback" > "$RESULTS_DIR/current_phase"
    echo "$(date -Iseconds)" > "$RESULTS_DIR/rollback_time"
}

# Status function
show_status() {
    log_info "=== Deployment Status ==="
    
    local current_phase=$(cat "$RESULTS_DIR/current_phase" 2>/dev/null || echo "not-deployed")
    log_info "Current Phase: $current_phase"
    
    if [[ -f "$RESULTS_DIR/shadow_start_time" ]]; then
        local shadow_start=$(cat "$RESULTS_DIR/shadow_start_time")
        log_info "Shadow deployment started: $shadow_start"
    fi
    
    # Show running containers
    log_info "Running containers:"
    docker ps --filter "label=com.docker.compose.project=$COMPOSE_PROJECT" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    # Show recent metrics if available
    if command -v curl >/dev/null && curl -sf "http://localhost:9090/-/healthy" >/dev/null 2>&1; then
        log_info "Recent routing metrics available at: http://localhost:3000"
        validate_sla "$current_phase" || log_warn "Current deployment not meeting SLA requirements"
    fi
}

# Main execution
main() {
    local phase=${1:-""}
    local action=${2:-"start"}
    
    init_environment
    
    case "$phase" in
        "shadow")
            case "$action" in
                "start") deploy_shadow ;;
                "status") show_status ;;
                "validate") validate_sla "shadow" ;;
                *) log_error "Unknown action: $action" ;;
            esac
            ;;
        "partial")
            local percentage=${3:-25}
            case "$action" in
                "start") deploy_partial "$percentage" ;;
                "status") show_status ;;
                "validate") validate_sla "partial" ;;
                *) log_error "Unknown action: $action" ;;
            esac
            ;;
        "full")
            case "$action" in
                "start") deploy_full ;;
                "status") show_status ;;
                "validate") validate_sla "full" ;;
                *) log_error "Unknown action: $action" ;;
            esac
            ;;
        "rollback")
            rollback_deployment
            ;;
        "status")
            show_status
            ;;
        *)
            echo "Usage: $0 [phase] [action] [options]"
            echo ""
            echo "Phases:"
            echo "  shadow          - Deploy shadow mode (10% mirrored traffic)"
            echo "  partial [%]     - Deploy partial rollout (default 25%)"
            echo "  full            - Deploy full production (100% traffic)"
            echo "  rollback        - Emergency rollback to legacy system"
            echo "  status          - Show current deployment status"
            echo ""
            echo "Actions:"
            echo "  start           - Start the specified phase"
            echo "  status          - Show status for the phase"
            echo "  validate        - Validate SLA compliance"
            echo ""
            echo "Examples:"
            echo "  $0 shadow start                # Start shadow deployment"
            echo "  $0 partial start 50            # Start 50% partial rollout"
            echo "  $0 full start                  # Start full production"
            echo "  $0 rollback                    # Emergency rollback"
            echo "  $0 status                      # Show current status"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"