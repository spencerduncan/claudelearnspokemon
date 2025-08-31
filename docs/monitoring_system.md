# Pokemon Speedrun Monitoring System

Comprehensive monitoring and metrics collection system for the Pokemon speedrun learning agent, providing operational visibility, performance monitoring, and alerting capabilities.

## Overview

The monitoring system consists of five main components:

1. **Prometheus Metrics Export** - Standard format metrics export for Grafana/Prometheus
2. **System Metrics Collection** - OS-level CPU, memory, disk monitoring  
3. **HTTP Monitoring Middleware** - Request/response instrumentation
4. **Application Metrics** - Pokemon speedrun specific metrics (experiments, patterns)
5. **Alert Management** - Configurable alerting with SLA monitoring

## Quick Start

### Basic Setup

```python
from claudelearnspokemon.prometheus_metrics import PrometheusMetricsExporter
from claudelearnspokemon.system_metrics import SystemMetricsCollector
from claudelearnspokemon.speedrun_metrics import SpeedrunMetricsCollector

# Initialize components
system_metrics = SystemMetricsCollector()
speedrun_metrics = SpeedrunMetricsCollector()
prometheus_exporter = PrometheusMetricsExporter()

# Start Prometheus HTTP server
prometheus_exporter.start_http_server(port=8000)
```

### Recording Metrics

```python
# System metrics (automatic)
system_data = system_metrics.get_metrics()
print(f"CPU: {system_data.cpu_percent}%, Memory: {system_data.memory_percent}%")

# Speedrun experiment
from claudelearnspokemon.speedrun_metrics import ExperimentResult, ExperimentStatus

experiment = ExperimentResult(
    experiment_id="exp_001",
    status=ExperimentStatus.SUCCESS,
    duration_seconds=125.5,
    script_compilation_time_ms=87.3,
    checkpoint_loading_time_ms=425.8,
    ai_strategy="genetic_algorithm"
)
speedrun_metrics.record_experiment(experiment)
```

### HTTP Monitoring

```python
from claudelearnspokemon.monitoring_middleware import HTTPMonitoringMiddleware

middleware = HTTPMonitoringMiddleware()

# Monitor requests
with middleware.monitor_request("GET", "http://localhost:8081/health") as metrics:
    response = requests.get("http://localhost:8081/health")
    metrics.status_code = response.status_code
    metrics.response_size_bytes = len(response.content)
```

## Performance Requirements

The monitoring system is designed with strict performance requirements:

- **Prometheus metrics collection**: <5ms per scrape
- **System metrics gathering**: <3ms (with caching)
- **HTTP middleware overhead**: <1ms per request
- **Application metrics**: <2ms per measurement
- **Total monitoring overhead**: <2% of application runtime

## Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  System Metrics     │    │   HTTP Middleware    │    │  Speedrun Metrics   │
│  (CPU/Memory/Disk)  │    │  (Request/Response)  │    │ (Experiments/SLA)   │
└─────────┬───────────┘    └──────────┬───────────┘    └─────────┬───────────┘
          │                           │                          │
          └───────────────────────────┼──────────────────────────┘
                                      │
                             ┌────────▼────────┐
                             │ Prometheus      │
                             │ Metrics Export  │
                             └────────┬────────┘
                                      │
                             ┌────────▼────────┐
                             │ Grafana         │
                             │ Dashboard       │
                             └─────────────────┘
```

## Configuration Files

### Prometheus Configuration

Use the provided `monitoring/prometheus_config.yml`:

```bash
# Start Prometheus
prometheus --config.file=monitoring/prometheus_config.yml
```

### Grafana Dashboard

Import `monitoring/grafana_dashboard.json` into Grafana for pre-configured visualizations.

### Alert Rules

Alert rules are defined in `monitoring/alert_rules.yml` and include:

- Performance regression alerts (compilation >150ms, checkpoint >750ms)
- Resource exhaustion alerts (CPU >85%, memory >90%) 
- SLA violations (success rate <95%)
- System health alerts (insufficient healthy processes)

## Metrics Reference

### System Metrics

| Metric | Description | Type |
|--------|-------------|------|
| `pokemon_speedrun_system_cpu_percent` | CPU usage percentage | Gauge |
| `pokemon_speedrun_system_memory_bytes` | Memory usage in bytes | Gauge |
| `pokemon_speedrun_system_disk_percent` | Disk usage percentage | Gauge |

### Process Metrics

| Metric | Description | Type |
|--------|-------------|------|
| `pokemon_speedrun_healthy_processes` | Number of healthy processes | Gauge |
| `pokemon_speedrun_process_startup_seconds` | Process startup time | Histogram |
| `pokemon_speedrun_process_failures_total` | Total process failures | Counter |

### Application Metrics

| Metric | Description | Type |
|--------|-------------|------|
| `pokemon_speedrun_experiments_total` | Total experiments run | Counter |
| `pokemon_speedrun_experiment_success_rate` | Experiment success rate percentage | Gauge |
| `pokemon_speedrun_pattern_discoveries_total` | Total patterns discovered | Counter |
| `pokemon_speedrun_sla_compliance` | SLA compliance boolean | Gauge |

## Alerting

### Default Alert Rules

1. **Script Compilation Performance** - Triggers when compilation >150ms for 5 minutes
2. **High CPU Usage** - Triggers when CPU >85% for 2 minutes  
3. **High Memory Usage** - Triggers when memory >90% for 2 minutes
4. **Low Experiment Success Rate** - Triggers when success rate <95% for 15 minutes
5. **Insufficient Healthy Processes** - Triggers when <3 healthy processes for 1 minute

### Custom Alerts

```python
from claudelearnspokemon.alert_manager import AlertManager, AlertRule, AlertSeverity

alert_manager = AlertManager()

# Add custom alert rule
custom_rule = AlertRule(
    rule_id="custom_performance",
    name="Custom Performance Alert",
    description="Custom performance threshold exceeded",
    metric_name="speedrun.average_script_compilation_ms",
    threshold_value=120.0,  # 120ms threshold
    operator=">",
    severity=AlertSeverity.WARNING
)

alert_manager.add_alert_rule(custom_rule)
alert_manager.start_monitoring()
```

## SLA Monitoring

The system tracks SLA compliance against these targets:

- **Experiment Success Rate**: ≥95%
- **Script Compilation**: <100ms (target), <150ms (alert threshold)
- **Checkpoint Loading**: <500ms (target), <750ms (alert threshold)  
- **Pattern Quality**: ≥70% average quality score
- **System Availability**: ≥99% healthy process ratio

```python
# Check SLA compliance
sla_status = speedrun_metrics.get_sla_compliance()
if sla_status["overall_sla_compliant"]:
    print("✅ Meeting SLA requirements")
else:
    print("❌ SLA violation detected")
    for requirement, compliant in sla_status["individual_compliance"].items():
        if not compliant:
            print(f"  - {requirement}: FAILING")
```

## Integration with Existing Code

### EmulatorPool Integration

```python
from claudelearnspokemon.emulator_pool import EmulatorPool
from claudelearnspokemon.monitoring_middleware import monitor_requests_session

# Create instrumented session
pool = EmulatorPool()
instrumented_session = monitor_requests_session(pool.session)
```

### Process Metrics Integration

```python
from claudelearnspokemon.process_metrics_collector import ProcessMetricsCollector

# Existing process metrics collectors automatically work
# with the new Prometheus export system
process_collector = ProcessMetricsCollector(process_id=12345)
prometheus_exporter.add_process_collector(process_collector)
```

## Troubleshooting

### High Monitoring Overhead

If monitoring overhead exceeds 2% of runtime:

1. Increase cache durations:
   ```python
   system_metrics = SystemMetricsCollector(cache_duration=10.0)  # 10 second cache
   ```

2. Reduce HTTP monitoring detail:
   ```python
   middleware = HTTPMonitoringMiddleware(
       enable_detailed_logging=False,
       max_recorded_requests=500  # Reduce storage
   )
   ```

3. Adjust Prometheus scrape interval to 30s+ in `prometheus_config.yml`

### Missing Metrics

1. Verify components are initialized and recording data
2. Check Prometheus server is running: `curl http://localhost:8000/metrics`
3. Validate metric names in Grafana queries match exported metrics
4. Check logs for collection errors

### Alert Issues

1. Verify alert rules syntax in `monitoring/alert_rules.yml`
2. Check metric sources are registered with AlertManager
3. Validate thresholds are appropriate for your system
4. Test alerting with intentionally triggered conditions

## Performance Validation

Run the monitoring system benchmarks:

```bash
# Run performance tests
python -m pytest tests/test_monitoring_integration.py::TestMonitoringSystemIntegration::test_performance_under_concurrent_load -v

# Check individual component performance  
python -m pytest tests/test_prometheus_metrics.py::TestPrometheusMetricsExporter::test_update_metrics_performance -v
```

Expected results:
- Prometheus updates: <5ms
- System metrics collection: <3ms (fresh) / <1ms (cached)
- HTTP middleware: <1ms overhead per request
- Speedrun metrics: <2ms per recording