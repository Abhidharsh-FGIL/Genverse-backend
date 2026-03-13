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
        topic_weightage, negative_marking, source_text,
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
    import uuid as _uuid

    question_count = params["question_count"]

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

    raw = await ai.generate_practice_assessment(
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
    )

    _publish(r, channel, {
        "stage": "processing",
        "progress": 80,
        "message": "Processing and validating questions...",
    })

    # Post-process: filter by allowed types, build question_json + answer_key_json
    allowed_types = set(params.get("allowed_types", ["mcq"]))

    question_json = []
    answer_key_json = []
    for q in raw:
        qid = q.get("id") or str(_uuid.uuid4())
        q_type = (q.get("type") or "mcq").lower()

        if q_type not in allowed_types:
            continue

        opts = q.get("options")
        if isinstance(opts, dict):
            opts = list(opts.values())

        question_json.append({
            "id": qid,
            "type": q_type,
            "subtype": q.get("subtype"),
            "text": q.get("text") or q.get("question", ""),
            "options": opts,
            "pairs": q.get("pairs"),
            "points": q.get("marks") or q.get("points") or 1,
            "blooms_level": q.get("blooms_level"),
        })
        answer_key_json.append({
            "id": qid,
            "correctAnswer": q.get("correct_answer") or q.get("correctAnswer", ""),
            "explanation": q.get("explanation", ""),
        })

    # Stage 3: Complete
    _publish(r, channel, {
        "stage": "complete",
        "progress": 100,
        "message": f"{len(question_json)} questions generated successfully!",
        "question_json": question_json,
        "answer_key_json": answer_key_json,
    })
