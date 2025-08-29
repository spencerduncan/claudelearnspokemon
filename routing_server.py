"""
Production FastAPI server for Message Routing Engine
Provides HTTP endpoints for routing requests and health monitoring
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
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

# Global routing adapter
routing_adapter: RoutingAdapter = None


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
    """Application lifespan manager."""
    global routing_adapter
    
    logger.info("Starting Message Routing Engine server...")
    
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
        
    except Exception as e:
        logger.error(f"Failed to initialize routing engine: {e}")
        routing_adapter = None
    
    yield
    
    # Shutdown
    if routing_adapter:
        routing_adapter.shutdown()
    logger.info("Message Routing Engine server stopped")


# Initialize FastAPI app
app = FastAPI(
    title="Message Routing Engine",
    description="Intelligent message routing for Pokemon speedrun learning agent",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    HEALTH_CHECKS.labels(endpoint="health").inc()
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
    uvicorn.run(app, host="0.0.0.0", port=8080)