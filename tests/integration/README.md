# Docker Integration Tests

Comprehensive Docker-based integration testing infrastructure for Pokemon-gym server instances using real containers.

**Author**: John Botmack - Performance Engineering
**Performance Standards**: <100ms client creation, <100ms action execution, <50ms status checks

## Overview

This integration test suite addresses the critical gap in testing real Pokemon-gym server instances. While unit tests verify individual components with mocks, these integration tests validate:

- Real network communication with Docker containers
- Performance characteristics under realistic load
- Server version compatibility across Pokemon-gym releases
- Error handling with actual network failures
- Concurrent client behavior with real servers

## Architecture

```
tests/integration/
├── conftest.py                      # Docker fixtures & environment config
├── test_server_integration.py       # Basic real server integration tests
├── test_performance_integration.py  # Performance benchmarks & baselines
├── test_compatibility_integration.py # Multi-version compatibility tests
└── README.md                       # This documentation
```

### Core Components

**Docker Fixtures** (`conftest.py`):
- `pokemon_gym_container`: Single container for basic testing
- `multiple_pokemon_containers`: 4 containers for concurrent testing
- `slow_pokemon_container`: Timeout testing container
- `version_compatibility_containers`: Multiple versions for compatibility

**Performance Collector** (`test_performance_integration.py`):
- High-precision timing with `time.perf_counter()`
- Statistical analysis (mean, median, P95, P99)
- Regression detection and baseline export
- John Botmack performance validation

## Setup Requirements

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_INTEGRATION_TESTS` | `""` | Set to `1` to enable integration tests |
| `DOCKER_TIMEOUT` | `30` | Container startup timeout (seconds) |
| `POKEMON_GYM_IMAGE` | `pokemon-gym:latest` | Docker image for testing |

### Docker Images

The integration tests require Pokemon-gym Docker images:

```bash
# Pull required images
docker pull pokemon-gym:latest
docker pull pokemon-gym:v1.2  # Optional for compatibility testing
docker pull pokemon-gym:v1.1  # Optional for compatibility testing
docker pull pokemon-gym:v1.0  # Optional for compatibility testing
```

### Dependencies

```bash
# Install integration test dependencies
pip install testcontainers>=3.7.0
pip install pytest-timeout pytest-xdist
```

## Running Tests

### Local Execution

```bash
# Enable integration tests
export RUN_INTEGRATION_TESTS=1

# Run all integration tests
pytest tests/integration/ -v

# Run specific test suites
pytest tests/integration/test_server_integration.py -v      # Basic tests
pytest tests/integration/test_performance_integration.py -v # Performance
pytest tests/integration/test_compatibility_integration.py -v # Compatibility

# Run with parallel execution (faster)
pytest tests/integration/ -v -n 4

# Run with timeout protection
pytest tests/integration/ -v --timeout=300
```

### CI/CD Execution

Integration tests run automatically in GitHub Actions:

```yaml
# .github/workflows/integration-tests.yml
- Basic Integration Tests: Real server communication
- Performance Tests: Benchmark validation
- Compatibility Tests: Multi-version support
```

**Test Matrix**:
- Python 3.10, 3.11
- Test suites: basic, performance, compatibility
- Performance regression detection
- Baseline collection from main branch

## Performance Standards

### John Botmack Performance Requirements

| Operation | Requirement | Validation |
|-----------|-------------|------------|
| Client Creation | <100ms | P95 percentile |
| Action Execution | <100ms | P95 percentile |
| Status Checks | <50ms | P95 percentile |
| Concurrent Overhead | <20% | Vs single-client baseline |

### Performance Testing Features

**Comprehensive Benchmarking**:
- Multi-iteration measurements (20+ per test)
- Statistical analysis with P95/P99 validation
- Memory usage correlation testing
- Network latency impact assessment

**Regression Detection**:
- Automatic baseline establishment
- Performance comparison across builds
- 20% regression threshold for failures
- Historical baseline storage

**Throughput Validation**:
- Sustained operations per second measurement
- Load testing with consistent performance
- Resource usage monitoring

## Test Categories

### Basic Integration (`test_server_integration.py`)

**Real Server Communication**:
- Client creation with Docker containers
- Session lifecycle (initialize → action → stop)
- Network error handling validation
- Concurrent client testing (4+ parallel)

**Docker Container Management**:
- Health monitoring and status validation
- Resource usage verification
- Container cleanup and isolation

### Performance Integration (`test_performance_integration.py`)

**Benchmark Suites**:
- Client creation performance across adapter types
- Action execution timing validation
- Status check frequency optimization
- Memory scaling impact analysis

**Performance Analysis**:
- Statistical metrics collection
- Baseline establishment for regression testing
- Throughput measurement under sustained load
- Network latency correlation

### Compatibility Integration (`test_compatibility_integration.py`)

**Version Compatibility Matrix**:
- Multi-version Pokemon-gym server testing
- Feature detection and validation
- Performance characteristic comparison
- Error handling consistency

**API Compatibility**:
- Response format validation across versions
- Backward compatibility verification
- Feature evolution tracking

## Configuration Options

### Environment-Based Configuration

```python
# Skip integration tests if Docker unavailable
SKIP_INTEGRATION_TESTS = not bool(os.getenv("RUN_INTEGRATION_TESTS", ""))

# Performance tuning
CLIENT_CREATION_TIMEOUT_MS = 100
CONCURRENT_CLIENT_COUNT = 4
CONTAINER_STARTUP_TIMEOUT = 30
```

### Test Markers

```python
@pytest.mark.integration  # All integration tests
@pytest.mark.slow         # Long-running tests
@pytest.mark.medium       # Medium duration tests
```

## Debugging and Troubleshooting

### Common Issues

**Docker Not Available**:
```bash
# Check Docker daemon
docker info
systemctl status docker

# Install Docker if missing
sudo apt-get install docker.io
```

**Container Startup Timeouts**:
```bash
# Check available resources
docker system df
docker system prune -f

# Monitor container logs
docker logs <container_id>
```

**Performance Test Failures**:
```bash
# Check system load
top
free -h

# Run single performance test for debugging
pytest tests/integration/test_performance_integration.py::TestPerformanceIntegration::test_client_creation_performance_benchmark -v -s
```

### Performance Debugging

**Enable Performance Logging**:
```python
import logging
logging.getLogger("claudelearnspokemon").setLevel(logging.DEBUG)
```

**Manual Timing Verification**:
```python
import time
start = time.perf_counter()
# Your operation here
duration_ms = (time.perf_counter() - start) * 1000
print(f"Operation took {duration_ms:.2f}ms")
```

## Integration with Existing Tests

### Test Structure Compatibility

The integration tests follow the same patterns as existing unit tests:

```python
# Same test class structure
class TestRealServerIntegration:
    def setup_method(self):
        # Test setup

    def test_specific_functionality(self):
        # Test implementation
```

### Performance Integration with Unit Tests

```bash
# Run both unit and integration tests
pytest tests/ -m "not slow"           # Fast unit tests only
pytest tests/integration/ -m "slow"   # Integration tests only
pytest tests/                         # All tests (long)
```

## Continuous Integration

### GitHub Actions Integration

**Automated Execution**:
- Pull request validation
- Main branch baseline collection
- Daily regression testing (3 AM UTC)
- Performance trend analysis

**Artifact Collection**:
- Performance baselines stored for 90 days
- Test results retained for 30 days
- Coverage reports uploaded to Codecov

### Performance Monitoring

**Baseline Collection**:
```bash
# Baselines automatically collected from main branch
find . -name "performance_baseline_*.json" -exec cat {} \;
```

**Regression Detection**:
- 20% performance degradation threshold
- Automatic comparison against previous baselines
- PR blocking for significant regressions

## Best Practices

### Writing Integration Tests

**Performance Focus**:
- Always measure and validate timing requirements
- Use statistical validation (P95/P99) not just averages
- Test concurrent scenarios representative of production

**Docker Best Practices**:
- Clean up containers in finally blocks
- Use appropriate timeout values
- Validate container health before testing

**Error Handling**:
- Test both success and failure scenarios
- Validate error messages and types
- Ensure graceful cleanup on test failures

### Maintenance

**Regular Tasks**:
- Review performance baselines monthly
- Update Docker images for new Pokemon-gym releases
- Monitor CI execution times and optimize as needed

**Performance Standards Review**:
- John Botmack standards should be reviewed quarterly
- Adjust thresholds based on production performance data
- Update documentation with new requirements

## References

- [Issue #193](https://github.com/spencerduncan/claudelearnspokemon/issues/193): Original Docker integration test requirement
- [testcontainers-python](https://testcontainers-python.readthedocs.io/): Docker container testing framework
- [Performance Engineering Principles](docs/performance_tuning.md): John Botmack performance standards

---

*"In real-time systems, every microsecond counts. This implementation pushes the hardware to its limits while maintaining correctness."* - John Botmack
