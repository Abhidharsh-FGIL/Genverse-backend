"""
Celery application instance.

Start the worker:
    celery -A app.celery_app worker --loglevel=info
"""
from celery import Celery
from app.config import settings

celery = Celery(
    "eduverse",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    result_expires=3600,  # results expire after 1 hour
    task_time_limit=900,  # hard kill after 15 minutes
    task_soft_time_limit=600,  # soft limit (SoftTimeLimitExceeded) after 10 minutes
    worker_prefetch_multiplier=1,  # don't prefetch tasks — ebook generation is long-running
    broker_connection_retry_on_startup=True,  # silence Celery 6.0 deprecation warning
)

# Auto-discover tasks in these modules
celery.autodiscover_tasks(["app.tasks"])
