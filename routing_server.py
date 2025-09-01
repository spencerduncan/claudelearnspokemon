"""
Production FastAPI server for Message Routing Engine
Provides HTTP endpoints for routing requests and health monitoring with graceful shutdown support
"""

import asyncio
import json
import logging
import os
import signal
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.claudelearnspokemon.claude_code_manager import ClaudeCodeManager
from src.claudelearnspokemon.routing_integration import (
    IntegrationConfig, IntegrationMode, RoutingAdapter
)
from src.claudelearnspokemon.sonnet_worker_pool import SonnetWorkerPool
from src.claudelearnspokemon.priority_queue import MessagePriority

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
ROUTING_REQUESTS = Counter('routing_requests_total', 'Total routing requests', ['method', 'status'])
ROUTING_DURATION = Histogram('routing_duration_seconds', 'Routing request duration')
HEALTH_CHECKS = Counter('health_check_requests_total', 'Health check requests')

# Graceful shutdown infrastructure
shutdown_event = threading.Event()
active_requests = set()
active_requests_lock = threading.Lock()

# Global routing adapter
routing_adapter: RoutingAdapter = None


def register_request(request_id: str) -> None:
    """Register an active request for graceful shutdown tracking."""
    with active_requests_lock:
        active_requests.add(request_id)
        logger.debug(f"Registered active request: {request_id} (total: {len(active_requests)})")


def unregister_request(request_id: str) -> None:
    """Unregister a completed request."""
    with active_requests_lock:
        active_requests.discard(request_id)
        logger.debug(f"Unregistered request: {request_id} (remaining: {len(active_requests)})")


def get_active_request_count() -> int:
    """Get the current number of active requests."""
    with active_requests_lock:
        return len(active_requests)


def signal_handler(signum: int, frame) -> None:
    """Handle SIGTERM and SIGINT for graceful shutdown."""
    signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT" if signum == signal.SIGINT else f"SIG{signum}"
    logger.info(f"Received {signal_name}, initiating graceful shutdown...")
    
    active_count = get_active_request_count()
    if active_count > 0:
        logger.info(f"Waiting for {active_count} active requests to complete...")
    
    shutdown_event.set()


class RoutingRequest(BaseModel):
    """Request model for routing API."""
    content: str
    context: dict[str, Any] = {}
    priority: str = "NORMAL"
    force_strategic: bool = False
    force_tactical: bool = False


class RoutingResponse(BaseModel):
    """Response model for routing API."""
    success: bool
    worker_id: str | None = None
    worker_type: str | None = None
    routing_time_ms: float = 0.0
    error_message: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with graceful shutdown support."""
    global routing_adapter
    
    logger.info("Starting Message Routing Engine server...")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    logger.info("Registered signal handlers for SIGTERM and SIGINT")
    
    try:
        # Initialize Claude manager and worker pool
        claude_manager = ClaudeCodeManager()
        worker_pool = SonnetWorkerPool()
        
        # Determine integration mode from environment
        mode_str = os.getenv("ROUTING_MODE", "full").lower()
        shadow_percentage = float(os.getenv("SHADOW_PERCENTAGE", "10.0"))
        
        if mode_str == "shadow":
            integration_mode = IntegrationMode.SHADOW
        elif mode_str == "partial":
            integration_mode = IntegrationMode.PARTIAL
        elif mode_str == "disabled":
            integration_mode = IntegrationMode.DISABLED
        else:
            integration_mode = IntegrationMode.FULL
        
        # Configure integration
        config = IntegrationConfig(
            mode=integration_mode,
            shadow_percentage=shadow_percentage,
            enable_metrics=True,
            enable_tracing=True,
            fallback_on_error=True,
            max_routing_time_ms=50.0
        )
        
        # Initialize routing adapter
        routing_adapter = RoutingAdapter(claude_manager, worker_pool, config)
        
        logger.info(f"Routing engine started in {integration_mode.value} mode")
        logger.info("Server startup complete - ready to handle requests")
        
    except Exception as e:
        logger.error(f"Failed to initialize routing engine: {e}")
        routing_adapter = None
        raise  # Re-raise to prevent server start with broken routing
    
    yield
    
    # Graceful shutdown sequence
    logger.info("Beginning graceful shutdown sequence...")
    
    # Wait for active requests to complete (with timeout)
    shutdown_timeout = float(os.getenv("SHUTDOWN_TIMEOUT", "30.0"))
    start_time = time.time()
    
    while get_active_request_count() > 0 and (time.time() - start_time) < shutdown_timeout:
        active_count = get_active_request_count()
        remaining_time = shutdown_timeout - (time.time() - start_time)
        logger.info(f"Waiting for {active_count} active requests (timeout: {remaining_time:.1f}s)")
        await asyncio.sleep(0.5)
    
    final_active_count = get_active_request_count()
    if final_active_count > 0:
        logger.warning(f"Shutdown timeout reached with {final_active_count} requests still active")
    else:
        logger.info("All requests completed successfully")
    
    # Clean up resources and connections
    logger.info("Cleaning up routing engine resources...")
    if routing_adapter:
        try:
            routing_adapter.shutdown()
            logger.info("Routing adapter shutdown completed")
        except Exception as e:
            logger.error(f"Error during routing adapter shutdown: {e}")
    
    # Reset signal handlers
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    logger.info("Message Routing Engine server stopped gracefully")


# Initialize FastAPI app
app = FastAPI(
    title="Message Routing Engine",
    description="Intelligent message routing for Pokemon speedrun learning agent",
    version="2.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    """Middleware to track active requests for graceful shutdown."""
    # Only track routing requests, not health checks or metrics
    if request.url.path.startswith("/route/"):
        request_id = f"{request.method}:{request.url.path}:{id(request)}"
        register_request(request_id)
        
        try:
            response = await call_next(request)
            return response
        finally:
            unregister_request(request_id)
    else:
        # Fast path for health checks and metrics
        return await call_next(request)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    HEALTH_CHECKS.labels(endpoint="health").inc()
    
    # Return degraded status during shutdown to help load balancers drain traffic
    if shutdown_event.is_set():
        return JSONResponse(
            status_code=503,
            content={
                "status": "shutting_down",
                "timestamp": time.time(),
                "active_requests": get_active_request_count()
            }
        )
    
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/health/routing")
async def routing_health_check():
    """Detailed routing engine health check."""
    HEALTH_CHECKS.labels(endpoint="routing").inc()
    
    if not routing_adapter:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Routing adapter not initialized"}
        )
    
    try:
        health_status = routing_adapter.get_integration_health()
        
        if health_status.get("routing_engine_active"):
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "routing_engine": health_status
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "timestamp": time.time(),
                    "routing_engine": health_status
                }
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/health/classifier")
async def classifier_health_check():
    """Message classifier health check."""
    HEALTH_CHECKS.labels(endpoint="classifier").inc()
    
    if not routing_adapter or not routing_adapter.message_router:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Classifier not available"}
        )
    
    try:
        classifier_health = routing_adapter.message_router.classifier.get_health_status()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "classifier": classifier_health
        }
    except Exception as e:
        logger.error(f"Classifier health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/health/queues")
async def queues_health_check():
    """Priority queues health check."""
    HEALTH_CHECKS.labels(endpoint="queues").inc()
    
    if not routing_adapter or not routing_adapter.message_router:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Queues not available"}
        )
    
    try:
        health_status = routing_adapter.message_router.get_health_status()
        queue_status = health_status.get("queues", {})
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "queues": queue_status
        }
    except Exception as e:
        logger.error(f"Queue health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/route/strategic", response_model=RoutingResponse)
async def route_strategic(request: RoutingRequest):
    """Route strategic planning request."""
    if not routing_adapter:
        raise HTTPException(status_code=503, detail="Routing engine not available")
    
    with ROUTING_DURATION.time():
        start_time = time.time()
        
        try:
            # Convert priority string to enum
            priority = MessagePriority[request.priority.upper()]
            
            worker_id = routing_adapter.route_strategic_request(
                request_content=request.content,
                context=request.context,
                priority=priority
            )
            
            routing_time = (time.time() - start_time) * 1000
            
            if worker_id:
                ROUTING_REQUESTS.labels(method="strategic", status="success").inc()
                return RoutingResponse(
                    success=True,
                    worker_id=worker_id,
                    worker_type="strategic",
                    routing_time_ms=routing_time
                )
            else:
                ROUTING_REQUESTS.labels(method="strategic", status="failure").inc()
                return RoutingResponse(
                    success=False,
                    error_message="No strategic workers available",
                    routing_time_ms=routing_time
                )
        
        except Exception as e:
            ROUTING_REQUESTS.labels(method="strategic", status="error").inc()
            logger.error(f"Strategic routing failed: {e}")
            return RoutingResponse(
                success=False,
                error_message=str(e),
                routing_time_ms=(time.time() - start_time) * 1000
            )


@app.post("/route/tactical", response_model=RoutingResponse)
async def route_tactical(request: RoutingRequest):
    """Route tactical development request."""
    if not routing_adapter:
        raise HTTPException(status_code=503, detail="Routing engine not available")
    
    with ROUTING_DURATION.time():
        start_time = time.time()
        
        try:
            # Convert priority string to enum
            priority = MessagePriority[request.priority.upper()]
            
            worker_id = routing_adapter.route_tactical_request(
                request_content=request.content,
                context=request.context,
                priority=priority
            )
            
            routing_time = (time.time() - start_time) * 1000
            
            if worker_id:
                ROUTING_REQUESTS.labels(method="tactical", status="success").inc()
                return RoutingResponse(
                    success=True,
                    worker_id=worker_id,
                    worker_type="tactical",
                    routing_time_ms=routing_time
                )
            else:
                ROUTING_REQUESTS.labels(method="tactical", status="failure").inc()
                return RoutingResponse(
                    success=False,
                    error_message="No tactical workers available",
                    routing_time_ms=routing_time
                )
        
        except Exception as e:
            ROUTING_REQUESTS.labels(method="tactical", status="error").inc()
            logger.error(f"Tactical routing failed: {e}")
            return RoutingResponse(
                success=False,
                error_message=str(e),
                routing_time_ms=(time.time() - start_time) * 1000
            )


@app.post("/route/auto", response_model=RoutingResponse)
async def route_auto(request: RoutingRequest):
    """Route request with automatic classification."""
    if not routing_adapter:
        raise HTTPException(status_code=503, detail="Routing engine not available")
    
    with ROUTING_DURATION.time():
        start_time = time.time()
        
        try:
            # Convert priority string to enum
            priority = MessagePriority[request.priority.upper()]
            
            worker_id = routing_adapter.route_auto_request(
                request_content=request.content,
                context=request.context,
                priority=priority
            )
            
            routing_time = (time.time() - start_time) * 1000
            
            if worker_id:
                # Determine worker type from worker_id
                worker_type = "strategic" if "strategic" in worker_id else "tactical"
                
                ROUTING_REQUESTS.labels(method="auto", status="success").inc()
                return RoutingResponse(
                    success=True,
                    worker_id=worker_id,
                    worker_type=worker_type,
                    routing_time_ms=routing_time
                )
            else:
                ROUTING_REQUESTS.labels(method="auto", status="failure").inc()
                return RoutingResponse(
                    success=False,
                    error_message="No workers available",
                    routing_time_ms=routing_time
                )
        
        except Exception as e:
            ROUTING_REQUESTS.labels(method="auto", status="error").inc()
            logger.error(f"Auto routing failed: {e}")
            return RoutingResponse(
                success=False,
                error_message=str(e),
                routing_time_ms=(time.time() - start_time) * 1000
            )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/admin/status")
async def admin_status():
    """Administrative status endpoint."""
    if not routing_adapter:
        return {"status": "routing_adapter_not_initialized"}
    
    try:
        return routing_adapter.get_integration_health()
    except Exception as e:
        logger.error(f"Failed to get admin status: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/admin/shutdown")
async def shutdown_status():
    """Shutdown status endpoint for monitoring graceful shutdown progress."""
    return {
        "shutdown_initiated": shutdown_event.is_set(),
        "active_requests": get_active_request_count(),
        "timestamp": time.time(),
        "shutdown_timeout": float(os.getenv("SHUTDOWN_TIMEOUT", "30.0"))
    }


@app.post("/admin/config")
async def update_config(config_data: dict[str, Any]):
    """Update routing configuration dynamically."""
    if not routing_adapter:
        raise HTTPException(status_code=503, detail="Routing engine not available")
    
    try:
        # Parse new configuration
        mode = IntegrationMode(config_data.get("mode", "full"))
        shadow_percentage = float(config_data.get("shadow_percentage", 10.0))
        partial_percentage = float(config_data.get("partial_percentage", 50.0))
        
        new_config = IntegrationConfig(
            mode=mode,
            shadow_percentage=shadow_percentage,
            partial_percentage=partial_percentage,
            enable_metrics=config_data.get("enable_metrics", True),
            enable_tracing=config_data.get("enable_tracing", True),
            fallback_on_error=config_data.get("fallback_on_error", True),
            max_routing_time_ms=float(config_data.get("max_routing_time_ms", 50.0))
        )
        
        success = routing_adapter.update_integration_config(new_config)
        
        if success:
            return {"status": "success", "config": config_data}
        else:
            raise HTTPException(status_code=400, detail="Failed to update configuration")
    
    except Exception as e:
        logger.error(f"Config update failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Production-grade uvicorn configuration with graceful shutdown support
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True,
        # Enable graceful shutdown with timeout
        timeout_graceful_shutdown=int(os.getenv("SHUTDOWN_TIMEOUT", "30")),
        # Optimize for production
        workers=1,  # Single worker for signal handling
        backlog=2048,
        max_requests=10000,
        max_requests_jitter=1000,
    )
    
    server = uvicorn.Server(config)
    
    try:
        logger.info("Starting Message Routing Engine server with graceful shutdown support...")
        server.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested via KeyboardInterrupt")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("Server shutdown complete")