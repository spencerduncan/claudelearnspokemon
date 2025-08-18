#!/bin/bash
set -e

echo "Building Pokemon-gym mock service Docker image..."

# Build the image with performance optimizations
docker build \
    --tag pokemon-gym:latest \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo "✅ Pokemon-gym:latest built successfully"

# Optional: Test the image quickly
echo "🧪 Running quick health check..."
CONTAINER_ID=$(docker run -d -p 8080:8080 pokemon-gym:latest)

# Wait briefly for startup
sleep 2

# Test health endpoint
if curl -f http://localhost:8080/health >/dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    docker logs $CONTAINER_ID
    docker stop $CONTAINER_ID
    exit 1
fi

# Cleanup
docker stop $CONTAINER_ID >/dev/null 2>&1 || true
docker rm $CONTAINER_ID >/dev/null 2>&1 || true

echo "🚀 Pokemon-gym mock service ready for testing"
