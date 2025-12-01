#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application entry point
"""

import time
import json
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core import openai
from app.api import admin
from app.utils.reload_config import RELOAD_CONFIG
from app.utils.helpers import debug_log

try:
    from granian import Granian
    HAS_GRANIAN = True
except ImportError:
    HAS_GRANIAN = False


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request/Response logging middleware"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        if settings.DEBUG_LOGGING:
            body = b""
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                # Create new request with body for downstream
                async def receive():
                    return {"type": "http.request", "body": body}
                request = Request(request.scope, receive)

            debug_log(f"[REQUEST] {request.method} {request.url.path}")
            if request.query_params:
                debug_log(f"[REQUEST] Query: {dict(request.query_params)}")
            if body:
                try:
                    body_json = json.loads(body.decode())
                    # Truncate messages content for readability
                    if "messages" in body_json:
                        body_preview = {**body_json, "messages": f"[{len(body_json['messages'])} messages]"}
                    else:
                        body_preview = body_json
                    debug_log(f"[REQUEST] Body: {json.dumps(body_preview, ensure_ascii=False)}")
                except:
                    debug_log(f"[REQUEST] Body: {body[:500]}...")

        # Process request
        response = await call_next(request)

        # Log response
        if settings.DEBUG_LOGGING:
            duration = time.time() - start_time
            debug_log(f"[RESPONSE] {request.method} {request.url.path} -> {response.status_code} ({duration:.3f}s)")

        return response


# Create FastAPI app
app = FastAPI(
    title="OpenAI Compatible API Server",
    description="An OpenAI-compatible API server for Z.AI chat service",
    version="1.0.0",
)

# Add request logging middleware (first, so it wraps everything)
app.add_middleware(RequestLoggingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Include API routers
app.include_router(openai.router)
app.include_router(admin.router)


@app.options("/")
async def handle_options():
    """Handle OPTIONS requests"""
    return Response(status_code=200)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "OpenAI Compatible API Server"}


def run_server():
    if HAS_GRANIAN:
        Granian(
            "main:app",
            interface="asgi",
            address="0.0.0.0",
            port=settings.LISTEN_PORT,
            reload=False,
            **RELOAD_CONFIG,
        ).serve()
    else:
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=settings.LISTEN_PORT,
            reload=False,
        )


if __name__ == "__main__":
    run_server()
