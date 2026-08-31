"""
Celery task for AI assessment generation.

Offloads the LLM call (question generation) to a Celery worker process.
Progress events are published to a Redis pub/sub channel so the FastAPI
SSE endpoint can forward them to the client in real time.
"""
import asyncio
import json
import uuid

import redis as sync_redis

from app.celery_app import celery
from app.config import settings


def _publish(r: sync_redis.Redis, channel: str, event: dict):
    """Publish a progress event to the Redis pub/sub channel."""
    r.publish(channel, json.dumps(event))


def _run_async(coro):
    """Run an async coroutine from sync Celery context."""
    return asyncio.run(coro)


@celery.task(bind=True, name="generate_assessment_task", max_retries=0)
def generate_assessment_task(self, params: dict):
    """
    Generate practice assessment questions in a Celery worker.

    params dict keys:
        subject, topics, grade, board, difficulty, question_count,
        question_types, mode, blooms_level, mcq_subtypes, type_weightage,
        topic_weightage, negative_marking, source_text, exam_type,
        allowed_types, user_id, org_id
    """
    task_id = self.request.id
    channel = f"assessment:progress:{task_id}"
    r = sync_redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    try:
        _run_async(_generate_assessment_async(params, channel, r))
    except Exception as exc:
        _publish(r, channel, {
            "stage": "error",
            "progress": 0,
            "message": str(exc),
        })
    finally:
        _publish(r, channel, {"stage": "__DONE__"})
        r.close()

    return {"status": "dispatched", "task_id": task_id}


async def _generate_assessment_async(params: dict, channel: str, r: sync_redis.Redis):
    """Async assessment generation — runs inside asyncio.run() in the Celery worker."""
    from app.services.ai_service import AIService

    ai = AIService()
    try:
        await _do_generate(ai, params, channel, r)
    finally:
        if ai._openai_client:
            await ai._openai_client.close()
            ai._openai_client = None


async def _do_generate(ai, params: dict, channel: str, r: sync_redis.Redis):
    """Inner generation logic."""
    import logging

    _log = logging.getLogger(__name__)
    question_count = params["question_count"]
    source_text = params.get("source_text")
    _log.info(
        "[Assessment-Celery] Starting: %d questions, source_text=%s chars, types=%s",
        question_count,
        len(source_text) if source_text else 0,
        params.get("allowed_types"),
    )

    # Stage 1: Preparing prompt
    _publish(r, channel, {
        "stage": "preparing",
        "progress": 20,
        "message": "Preparing assessment configuration...",
    })

    # Stage 2: Generating questions via LLM
    _publish(r, channel, {
        "stage": "generating",
        "progress": 30,
        "message": f"Generating {question_count} questions with AI...",
    })

    # The LLM call below is the longest step by far (often the majority of
    # total generation time) and previously had NO progress signal between
    # the 30% "generating" tick and the 80% "processing" tick — the frontend
    # progress bar just sat still for the whole duration, which read as
    # "stuck". Running it as a task and polling it lets us publish real
    # incrementing progress while it's still in flight, instead of a single
    # long silent gap.
    gen_task = asyncio.ensure_future(ai.generate_practice_assessment(
        subject=params["subject"],
        topics=params.get("topics"),
        grade=params.get("grade"),
        board=params.get("board"),
        difficulty=params.get("difficulty", "medium"),
        question_count=question_count,
        question_types=params.get("question_types"),
        mode=params.get("mode", "practice"),
        blooms_level=params.get("blooms_level") or "mixed",
        mcq_subtypes=params.get("mcq_subtypes"),
        type_weightage=params.get("type_weightage"),
        topic_weightage=params.get("topic_weightage"),
        negative_marking=params.get("negative_marking", False),
        source_text=params.get("source_text"),
        language=params.get("language"),
        exam_type=params.get("exam_type"),
    ))

    heartbeat_progress = 30
    while True:
        done, _pending = await asyncio.wait({gen_task}, timeout=2.5)
        if gen_task in done:
            break
        heartbeat_progress = min(heartbeat_progress + 6, 75)
        _publish(r, channel, {
            "stage": "generating",
            "progress": heartbeat_progress,
            "message": f"Generating {question_count} questions with AI...",
        })

    raw = gen_task.result()  # re-raises if the task itself raised

    _log.info("[Assessment-Celery] LLM returned %d raw questions", len(raw))

    _publish(r, channel, {
        "stage": "processing",
        "progress": 80,
        "message": "Processing and validating questions...",
    })

    # Post-process: filter by allowed types, build question_json + answer_key_json
    # (shared with the in-process SSE fallback in ai_assistant.py so the two paths
    # can't drift apart — see AIService.finalize_generated_questions).
    allowed_types = set(params.get("allowed_types", ["mcq"]))
    _log.info("[Assessment-Celery] Filtering by allowed_types=%s", allowed_types)

    question_json, answer_key_json = ai.finalize_generated_questions(raw, allowed_types)

    _log.info("[Assessment-Celery] After filtering: %d questions passed (from %d raw)", len(question_json), len(raw))

    # Stage 3: Complete
    _publish(r, channel, {
        "stage": "complete",
        "progress": 100,
        "message": f"{len(question_json)} questions generated successfully!",
        "question_json": question_json,
        "answer_key_json": answer_key_json,
    })
