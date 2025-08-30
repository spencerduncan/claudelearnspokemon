# Message Routing Engine - Monitoring Setup

## Quick Monitoring Setup

### 1. Basic Health Monitoring Script
Create a simple monitoring script for immediate use:

```bash
#!/bin/bash
# monitor_routing.sh - Simple health monitor

while true; do
    echo "=== $(date) ==="
    
    # Health check
    if curl -sf http://localhost:8000/health > /dev/null; then
        echo "✅ Routing server healthy"
    else
        echo "❌ Routing server down - checking..."
        ps aux | grep routing_server | grep -v grep
    fi
    
    # Performance check
    response_time=$(curl -w "%{time_total}" -s http://localhost:8000/health -o /dev/null)
    echo "⏱️ Response time: ${response_time}s"
    
    # Memory check
    memory_usage=$(ps -o pid,vsz,rss,comm -p $(pgrep -f routing_server) 2>/dev/null | tail -1 | awk '{print $3}')
    if [[ -n "$memory_usage" ]]; then
        echo "💾 Memory usage: ${memory_usage}KB"
    fi
    
    sleep 30
done
```

### 2. Quick Start Commands

```bash
# Make monitoring script executable
chmod +x monitor_routing.sh

# Run basic monitoring
./monitor_routing.sh

# Background monitoring with logging
nohup ./monitor_routing.sh > routing_health.log 2>&1 &
```

### 3. Health Check Endpoints

The routing server provides these endpoints:

```bash
# Basic health check
curl http://localhost:8000/health
# Returns: {"status": "healthy", "timestamp": "..."}

# Detailed status (if available)
curl http://localhost:8000/status
# Returns: system component status

# Prometheus metrics
curl http://localhost:8000/metrics
# Returns: Prometheus-formatted metrics
```

### 4. Log Monitoring

```bash
# Monitor real-time logs
tail -f routing_health.log

# Check for errors
grep -i error routing_health.log

# Check performance issues
grep -i "slow\|timeout\|circuit" routing_health.log
```

## Advanced Monitoring (Optional)

### Using Deployment Scripts
If the full deployment infrastructure is needed:

```bash
cd deployment

# Start shadow monitoring (safe test environment)
python3 health_monitor.py --phase=shadow

# Start circuit breaker monitoring
python3 circuit_breaker_monitor.py

# Full deployment monitoring
./deploy.sh shadow start  # Only if needed for testing
```

### Docker-based Monitoring (If Available)
```bash
# Check if docker-compose monitoring is available
ls deployment/phase*/*.yml

# If available, start minimal monitoring stack
cd deployment/phase1-shadow
docker-compose -f docker-compose.shadow.yml up prometheus grafana -d
```

## Key Metrics to Watch

### Critical Alerts
- **Health check fails**: Server down or unresponsive
- **Response time > 50ms**: Performance degradation  
- **Memory > 20MB**: Potential memory leak
- **CPU > 80%**: Resource exhaustion

### Performance Metrics
- **Classification time**: Should be < 5ms
- **Total routing time**: Should be < 25ms
- **Request success rate**: Should be > 99%

### System Metrics
- **Process CPU usage**: `top -p $(pgrep -f routing_server)`
- **Memory usage**: `ps -o pid,vsz,rss,comm -p $(pgrep -f routing_server)`
- **File descriptors**: `lsof -p $(pgrep -f routing_server) | wc -l`

## Alerting Setup

### Simple Email Alerts (Basic Setup)
```bash
#!/bin/bash
# alert_if_down.sh

if ! curl -sf http://localhost:8000/health > /dev/null; then
    echo "Routing server is down at $(date)" | mail -s "ALERT: Routing Server Down" admin@example.com
fi
```

### Cron Job Setup
```bash
# Add to crontab (crontab -e)
# Check every 2 minutes
*/2 * * * * /path/to/alert_if_down.sh

# Daily health summary
0 9 * * * tail -100 /path/to/routing_health.log | mail -s "Daily Routing Summary" admin@example.com
```

## Troubleshooting Monitoring

### Monitor Not Working
1. Check if routing server is running: `pgrep -f routing_server`
2. Verify port 8000 is accessible: `netstat -tlnp | grep 8000`  
3. Check firewall: `sudo ufw status`

### False Alerts
1. Adjust health check timeout in monitoring script
2. Add retry logic before alerting
3. Check network connectivity issues

### Performance Monitoring Issues
1. Ensure sufficient system resources
2. Monitor during low traffic periods first
3. Adjust monitoring frequency if needed

## Integration with Existing Systems

### Nagios/PRTG Integration
```bash
# Custom check command for Nagios
define command{
    command_name    check_routing_health
    command_line    curl -sf http://localhost:8000/health
}
```

### Zabbix Agent
```bash
# Add to zabbix_agentd.conf
UserParameter=routing.health,curl -sf http://localhost:8000/health >/dev/null; echo $?
UserParameter=routing.response_time,curl -w "%{time_total}" -s http://localhost:8000/health -o /dev/null
```

---

**Priority**: Focus on basic health monitoring first, add advanced features only if needed.  
**Status**: Ready for immediate deployment  
**Last Updated**: 2025-08-30