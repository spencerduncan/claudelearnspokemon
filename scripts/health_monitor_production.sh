#!/bin/bash

# Production Health Monitor for Message Routing Engine
# Monitors http://localhost:8080/health every 30 seconds
# Alerts on response time > 50ms or health check failures
# Author: Generated for Issue #1 - Health Check Automation

set -euo pipefail

# Configuration
HEALTH_URL="http://localhost:8080/health"
ROUTING_HEALTH_URL="http://localhost:8080/health/routing"
CHECK_INTERVAL=30
RESPONSE_THRESHOLD=50  # milliseconds
LOG_FILE="/var/log/routing_health_monitor.log"
METRICS_FILE="/var/log/routing_health_metrics.log"
PID_FILE="/var/run/routing_health_monitor.pid"
ALERT_EMAIL="${ALERT_EMAIL:-ops@company.com}"
ALERT_COOLDOWN=300  # 5 minutes between duplicate alerts
LAST_ALERT_FILE="/tmp/routing_health_last_alert"

# Alert tracking
declare -A alert_states
alert_states["health"]=0
alert_states["routing"]=0
alert_states["response_time"]=0

# Logging function
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# Metrics logging function
log_metrics() {
    local endpoint="$1"
    local response_time_ms="$2"
    local status_code="$3"
    local health_status="$4"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "$timestamp,$endpoint,$response_time_ms,$status_code,$health_status" >> "$METRICS_FILE"
}

# Email alert function
send_alert() {
    local subject="$1"
    local message="$2"
    local alert_key="$3"
    
    # Check cooldown period
    local current_time=$(date +%s)
    local last_alert_time=0
    
    if [[ -f "$LAST_ALERT_FILE.$alert_key" ]]; then
        last_alert_time=$(cat "$LAST_ALERT_FILE.$alert_key" 2>/dev/null || echo 0)
    fi
    
    local time_diff=$((current_time - last_alert_time))
    
    if [[ $time_diff -lt $ALERT_COOLDOWN ]]; then
        log "INFO" "Alert suppressed for $alert_key (cooldown: ${time_diff}s/${ALERT_COOLDOWN}s)"
        return 0
    fi
    
    # Send email alert using system mail or sendmail
    if command -v mail >/dev/null 2>&1; then
        echo -e "$message\n\nTimestamp: $(date)\nHost: $(hostname)" | mail -s "$subject" "$ALERT_EMAIL"
        log "INFO" "Email alert sent to $ALERT_EMAIL: $subject"
        echo "$current_time" > "$LAST_ALERT_FILE.$alert_key"
    elif command -v sendmail >/dev/null 2>&1; then
        {
            echo "To: $ALERT_EMAIL"
            echo "Subject: $subject"
            echo ""
            echo "$message"
            echo ""
            echo "Timestamp: $(date)"
            echo "Host: $(hostname)"
        } | sendmail "$ALERT_EMAIL"
        log "INFO" "Sendmail alert sent to $ALERT_EMAIL: $subject"
        echo "$current_time" > "$LAST_ALERT_FILE.$alert_key"
    else
        log "ERROR" "No mail system available for alerts. Install 'mail' or 'sendmail'"
        log "ERROR" "ALERT: $subject - $message"
    fi
}

# Health check function
check_endpoint() {
    local url="$1"
    local endpoint_name="$2"
    
    log "DEBUG" "Checking $endpoint_name endpoint: $url"
    
    # Perform health check with timing
    local start_time=$(date +%s%3N)
    local http_code
    local response
    
    if response=$(curl -s -w "%{http_code}" --connect-timeout 10 --max-time 15 "$url" 2>/dev/null); then
        local end_time=$(date +%s%3N)
        local response_time_ms=$((end_time - start_time))
        
        # Extract HTTP status code (last 3 characters)
        http_code="${response: -3}"
        local body="${response%???}"
        
        # Log metrics
        local health_status="unknown"
        if [[ "$http_code" == "200" ]]; then
            if echo "$body" | jq -e '.status == "healthy"' >/dev/null 2>&1; then
                health_status="healthy"
            else
                health_status="degraded"
            fi
        else
            health_status="unhealthy"
        fi
        
        log_metrics "$endpoint_name" "$response_time_ms" "$http_code" "$health_status"
        
        # Check response time threshold
        if [[ $response_time_ms -gt $RESPONSE_THRESHOLD ]]; then
            log "WARN" "$endpoint_name response time ${response_time_ms}ms > ${RESPONSE_THRESHOLD}ms threshold"
            
            if [[ ${alert_states["response_time"]} -eq 0 ]]; then
                send_alert "[PERF] Routing Engine Slow Response" \
                    "Response time for $endpoint_name is ${response_time_ms}ms (threshold: ${RESPONSE_THRESHOLD}ms)\nURL: $url\nStatus: $health_status" \
                    "response_time"
                alert_states["response_time"]=1
            fi
        else
            # Response time is good - clear alert state
            alert_states["response_time"]=0
        fi
        
        # Check HTTP status and health
        if [[ "$http_code" == "200" && "$health_status" == "healthy" ]]; then
            log "INFO" "$endpoint_name OK (${response_time_ms}ms, status: $health_status)"
            alert_states["$endpoint_name"]=0  # Clear alert state
            return 0
        else
            log "ERROR" "$endpoint_name FAILED (${response_time_ms}ms, HTTP: $http_code, status: $health_status)"
            
            if [[ ${alert_states["$endpoint_name"]} -eq 0 ]]; then
                send_alert "[CRITICAL] Routing Engine Health Check Failed" \
                    "$endpoint_name health check failed\nURL: $url\nHTTP Status: $http_code\nHealth Status: $health_status\nResponse Time: ${response_time_ms}ms\nResponse Body: $body" \
                    "$endpoint_name"
                alert_states["$endpoint_name"]=1
            fi
            return 1
        fi
    else
        local end_time=$(date +%s%3N)
        local response_time_ms=$((end_time - start_time))
        
        log_metrics "$endpoint_name" "$response_time_ms" "000" "connection_failed"
        log "ERROR" "$endpoint_name CONNECTION FAILED (${response_time_ms}ms)"
        
        if [[ ${alert_states["$endpoint_name"]} -eq 0 ]]; then
            send_alert "[CRITICAL] Routing Engine Connection Failed" \
                "$endpoint_name connection failed\nURL: $url\nConnection timeout or network error\nResponse Time: ${response_time_ms}ms" \
                "$endpoint_name"
            alert_states["$endpoint_name"]=1
        fi
        return 1
    fi
}

# Main monitoring loop
monitor_health() {
    log "INFO" "Starting health monitoring (PID: $$)"
    log "INFO" "Monitor config: interval=${CHECK_INTERVAL}s, threshold=${RESPONSE_THRESHOLD}ms, email=$ALERT_EMAIL"
    
    # Create metrics file header if it doesn't exist
    if [[ ! -f "$METRICS_FILE" ]]; then
        echo "timestamp,endpoint,response_time_ms,status_code,health_status" > "$METRICS_FILE"
        log "INFO" "Created metrics file: $METRICS_FILE"
    fi
    
    local consecutive_failures=0
    
    while true; do
        local overall_health=0
        
        # Check basic health endpoint
        if check_endpoint "$HEALTH_URL" "health"; then
            overall_health=$((overall_health + 1))
        fi
        
        # Check detailed routing health endpoint
        if check_endpoint "$ROUTING_HEALTH_URL" "routing"; then
            overall_health=$((overall_health + 1))
        fi
        
        # Track consecutive failures for escalation
        if [[ $overall_health -eq 0 ]]; then
            consecutive_failures=$((consecutive_failures + 1))
            log "ERROR" "All health checks failed (consecutive failures: $consecutive_failures)"
            
            # Escalate after 3 consecutive failures (1.5 minutes)
            if [[ $consecutive_failures -eq 3 ]]; then
                send_alert "[CRITICAL] Routing Engine Complete Outage" \
                    "All health endpoints have failed for $consecutive_failures consecutive checks\nThis indicates a complete service outage\nImmediate investigation required" \
                    "complete_outage"
            fi
        else
            if [[ $consecutive_failures -gt 0 ]]; then
                log "INFO" "Health checks recovered after $consecutive_failures failures"
            fi
            consecutive_failures=0
        fi
        
        log "DEBUG" "Health check cycle complete, sleeping ${CHECK_INTERVAL}s"
        sleep "$CHECK_INTERVAL"
    done
}

# Signal handlers for graceful shutdown
cleanup() {
    log "INFO" "Received shutdown signal, cleaning up..."
    rm -f "$PID_FILE"
    log "INFO" "Health monitoring stopped"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main execution
main() {
    case "${1:-monitor}" in
        "monitor")
            # Create PID file
            echo $$ > "$PID_FILE"
            
            # Ensure log files exist and are writable
            touch "$LOG_FILE" "$METRICS_FILE"
            chmod 644 "$LOG_FILE" "$METRICS_FILE"
            
            monitor_health
            ;;
        "check")
            # One-time health check for testing
            echo "Performing one-time health check..."
            check_endpoint "$HEALTH_URL" "health"
            check_endpoint "$ROUTING_HEALTH_URL" "routing"
            ;;
        "status")
            # Show current status
            if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
                echo "Health monitor is running (PID: $(cat "$PID_FILE"))"
                echo "Log file: $LOG_FILE"
                echo "Metrics file: $METRICS_FILE"
            else
                echo "Health monitor is not running"
            fi
            ;;
        "stop")
            # Stop the monitor
            if [[ -f "$PID_FILE" ]]; then
                local pid=$(cat "$PID_FILE")
                if kill -TERM "$pid" 2>/dev/null; then
                    echo "Sent shutdown signal to PID $pid"
                    # Wait for graceful shutdown
                    local timeout=10
                    while [[ $timeout -gt 0 ]] && kill -0 "$pid" 2>/dev/null; do
                        sleep 1
                        timeout=$((timeout - 1))
                    done
                    if kill -0 "$pid" 2>/dev/null; then
                        echo "Process did not shut down gracefully, forcing termination"
                        kill -KILL "$pid"
                    fi
                    rm -f "$PID_FILE"
                else
                    echo "Process not running"
                    rm -f "$PID_FILE"
                fi
            else
                echo "PID file not found, health monitor not running"
            fi
            ;;
        *)
            echo "Usage: $0 {monitor|check|status|stop}"
            echo ""
            echo "Commands:"
            echo "  monitor  - Start continuous health monitoring (default)"
            echo "  check    - Perform one-time health check"
            echo "  status   - Show current monitor status"
            echo "  stop     - Stop the health monitor"
            echo ""
            echo "Environment variables:"
            echo "  ALERT_EMAIL - Email address for alerts (default: ops@company.com)"
            exit 1
            ;;
    esac
}

main "$@"