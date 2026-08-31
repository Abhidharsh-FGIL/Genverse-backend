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
from app.services.stale_attempt_scheduler import run_stale_attempt_scheduler
from app.services.payment_reconciliation import run_payment_reconciliation


async def _backfill_study_time_if_needed():
    """One-time backfill of study_time_daily from historical data. Runs on startup if table is empty."""
    from sqlalchemy import text
    from app.database import AsyncSessionLocal
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(text("SELECT 1 FROM study_time_daily LIMIT 1"))
            if result.scalar() is not None:
                print("[StudyTimeBackfill] Table already has data, skipping backfill", flush=True)
                return
        # Table is empty — run backfill
        async with AsyncSessionLocal() as db:
            from app.services.study_time_aggregator import backfill_study_time
            count = await backfill_study_time(db, days_back=90)
            await db.commit()
            print(f"[StudyTimeBackfill] Done: {count} user-days processed", flush=True)
    except Exception as e:
        print(f"[StudyTimeBackfill] Error: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.STORAGE_ROOT, exist_ok=True)
    await run_migrations()
    # Start background schedulers
    renewal_task = asyncio.create_task(run_renewal_scheduler())
    notification_task = asyncio.create_task(run_notification_scheduler())
    stale_attempt_task = asyncio.create_task(run_stale_attempt_scheduler())
    payment_reconciliation_task = asyncio.create_task(run_payment_reconciliation())
    # One-time backfill of study time data (runs in background, doesn't block startup)
    backfill_task = asyncio.create_task(_backfill_study_time_if_needed())
    yield
    renewal_task.cancel()
    notification_task.cancel()
    stale_attempt_task.cancel()
    payment_reconciliation_task.cancel()
    backfill_task.cancel()
    await close_db()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
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
async def normalize_path(request: Request, call_next):
    """Fix double /api prefix from proxy setup (e.g. /api/api/v1/... → /api/v1/...)."""
    path: str = request.scope.get("path", "")
    if "/api/api/" in path:
        request.scope["path"] = path.replace("/api/api/", "/api/", 1)
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
