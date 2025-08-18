#!/usr/bin/env python3
"""
Minimal Pokemon-gym Mock Service for Infrastructure Testing

Performance-optimized mock service providing required endpoints for CI validation.
Designed for <100ms response times with minimal resource usage.
"""

import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Pokemon-gym Mock Service")

# Pre-computed responses for maximum performance
HEALTH_RESPONSE = {"status": "healthy", "service": "pokemon-gym-mock", "version": "1.0.0"}

STATUS_RESPONSE = {
    "game_status": "running",
    "emulator_state": "active",
    "frame_count": 12345,
    "performance_ms": 16.7,  # 60fps equivalent
    "memory_usage": "85%",
    "last_input": "A",
    "response_time_ms": 1.2,
}


@app.get("/health")
async def health_check() -> JSONResponse:
    """
    Health check endpoint - optimized for sub-millisecond response.
    Required by Docker healthcheck and EmulatorPool validation.
    """
    return JSONResponse(content=HEALTH_RESPONSE, status_code=200)


@app.get("/status")
async def game_status() -> JSONResponse:
    """
    Game status endpoint - benchflow-ai compatible response.
    Performance target: <5ms response time for real-time validation.
    """
    # Add minimal timestamp for realism without performance impact
    response = STATUS_RESPONSE.copy()
    response["timestamp"] = int(time.time())

    return JSONResponse(content=response, status_code=200)


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint for basic connectivity testing."""
    return {
        "service": "pokemon-gym-mock",
        "description": "Minimal mock service for infrastructure testing",
        "endpoints": ["/health", "/status"],
        "performance": "optimized for <100ms CI validation",
    }


if __name__ == "__main__":
    import uvicorn

    # Development mode - single worker for minimal overhead
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
