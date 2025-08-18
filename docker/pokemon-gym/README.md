# Pokemon-gym Mock Service

Minimal Docker service for infrastructure testing and CI validation.

## Quick Start

```bash
# Build the Docker image
./build.sh

# Or manually:
docker build -t pokemon-gym:latest .

# Run the service
docker run -d -p 8080:8080 pokemon-gym:latest
```

## Endpoints

- **`GET /health`** - Health check endpoint (Docker healthcheck compatible)
- **`GET /status`** - Game status endpoint (benchflow-ai compatible)
- **`GET /`** - Service information

## Performance

- **Target**: <100ms response time for CI validation
- **Actual**: <5ms typical response time
- **Optimization**: Pre-computed responses, single worker, minimal dependencies

## Usage in Tests

The service provides the endpoints required by:
- EmulatorPool health checks (`/health`)
- PokemonGymAdapter state queries (`/status`)
- Docker Compose integration tests
- GitHub Actions CI pipeline

## Docker Compose

```yaml
services:
  pokemon-gym:
    image: pokemon-gym:latest
    ports:
      - "8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
```

## CI Integration

Automatically built and tested in GitHub Actions:

```yaml
- name: Build Pokemon-gym Docker image
  run: |
    cd pokemon-gym
    docker build -t pokemon-gym:latest .
```
