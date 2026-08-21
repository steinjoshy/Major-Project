"""
FastAPI application entry point.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.api.routes import api_router
from backend.app.core.config import Settings, get_settings
from backend.app.schemas.common import ErrorDetail, ErrorResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()

    # Create upload directory
    os.makedirs(settings.upload_dir, exist_ok=True)

    yield

    # Shutdown
    pass


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="AI Demand Forecasting & Inventory Optimization API",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=f"HTTP_{exc.status_code}",
                    message=exc.detail,
                )
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="Request validation failed",
                    details={"errors": exc.errors()},
                )
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        # Log the actual exception server-side
        import logging
        logging.exception("Unhandled exception")

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="INTERNAL_ERROR",
                    message="An unexpected error occurred",
                )
            ).model_dump(),
        )

    # Include API router
    app.include_router(api_router, prefix="/api")

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "demand-forecasting-api",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
