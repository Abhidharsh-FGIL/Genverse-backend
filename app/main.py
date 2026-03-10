from contextlib import asynccontextmanager
import asyncio
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import close_db, run_migrations
from app.routers import register_routers
from app.services.renewal_scheduler import run_renewal_scheduler
from app.services.notification_scheduler import run_notification_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.STORAGE_ROOT, exist_ok=True)
    await run_migrations()
    # Start background schedulers
    renewal_task = asyncio.create_task(run_renewal_scheduler())
    notification_task = asyncio.create_task(run_notification_scheduler())
    yield
    renewal_task.cancel()
    notification_task.cancel()
    await close_db()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    redirect_slashes=False,
    description=(
        "Genverse.ai Backend API — AI-first EdTech platform providing "
        "multi-tenant educational management, AI-driven content generation, "
        "personalized learning paths, and assessment validation."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def strip_trailing_slash(request: Request, call_next):
    """Strip trailing slashes so /folders/ hits the /folders route."""
    if request.url.path != "/" and request.url.path.endswith("/"):
        request.scope["path"] = request.url.path.rstrip("/")
    return await call_next(request)

if os.path.exists(settings.STORAGE_ROOT):
    app.mount("/uploads", StaticFiles(directory=settings.STORAGE_ROOT), name="uploads")

# Register all API routers (defined in app/routers/__init__.py)
register_routers(app)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
