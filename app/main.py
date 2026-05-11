"""FastAPI application entrypoint for learning backend."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, jd, mock_interview, resume
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook used for startup/shutdown logs."""
    settings = get_settings()
    print(f"Learning backend starting on {settings.app_host}:{settings.app_port}")
    yield
    print("Learning backend shutting down")


settings = get_settings()
app = FastAPI(
    title="FaceTomato Learning API",
    description="A minimal backend skeleton for learning and step-by-step rebuild",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(resume.router, prefix="/api")
app.include_router(jd.router, prefix="/api")
app.include_router(mock_interview.router, prefix="/api")


@app.get("/")
async def root():
    """Root metadata endpoint."""
    return {"name": "FaceTomato Learning API", "version": "0.1.0", "docs": "/docs"}
