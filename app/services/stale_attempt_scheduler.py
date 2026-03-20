"""
Background scheduler to auto-submit stale "in_progress" evaluation attempts.

Runs every 5 minutes to find attempts that have exceeded their assessment's
time_limit (plus a 2-minute grace buffer) and auto-submits them with scoring.
This handles cases where the browser beacon failed (crash, network loss, etc.).
"""

import asyncio
from datetime import datetime, timezone, timedelta

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.attributes import flag_modified

from app.database import AsyncSessionLocal
from app.models.evaluation import (
    EvaluationAttempt,
    EvaluationAssessment,
    EvaluationQuestion,
    EvaluationInvitation,
)


# Grace period beyond time_limit before force-submitting (seconds)
GRACE_BUFFER_SECONDS = 120  # 2 minutes
# Fallback max duration for assessments with no time_limit (hours)
MAX_IN_PROGRESS_HOURS = 24


def _score_attempt_standalone(attempt, questions, assessment):
    """Score an attempt's responses against correct answers. Mutates attempt in-place."""
    responses = attempt.responses or {}
    score = 0
    max_score = sum(q.marks for q in questions)
    for q in questions:
        student_answer = responses.get(str(q.id))
        if student_answer and str(student_answer).strip().lower() == str(q.correct_answer or "").strip().lower():
            score += q.marks
        elif student_answer and assessment.negative_marking:
            score -= q.negative_marks
    attempt.score = max(0, score)
    attempt.max_score = max_score
    attempt.percentage = (attempt.score / max_score * 100) if max_score else 0


async def _cleanup_stale_attempts():
    """Find and auto-submit all stale in_progress attempts."""
    print("[StaleAttemptScheduler] Running stale attempt cleanup...", flush=True)

    async with AsyncSessionLocal() as db:
        now = datetime.now(timezone.utc)

        # Fetch all in_progress attempts
        result = await db.execute(
            select(EvaluationAttempt).where(
                EvaluationAttempt.status == "in_progress",
            )
        )
        in_progress_attempts = result.scalars().all()

        if not in_progress_attempts:
            print("[StaleAttemptScheduler] No stale attempts found.", flush=True)
            return

        submitted_count = 0

        for attempt in in_progress_attempts:
            # Fetch the assessment for time_limit
            assess_result = await db.execute(
                select(EvaluationAssessment).where(
                    EvaluationAssessment.id == attempt.assessment_id
                )
            )
            assessment = assess_result.scalar_one_or_none()
            if not assessment:
                continue

            # Determine if attempt is stale
            if assessment.time_limit:
                # time_limit is in seconds
                deadline = attempt.started_at + timedelta(seconds=assessment.time_limit + GRACE_BUFFER_SECONDS)
            else:
                # No time limit — use fallback max duration
                deadline = attempt.started_at + timedelta(hours=MAX_IN_PROGRESS_HOURS)

            if now <= deadline:
                continue  # Not yet expired, skip

            # Fetch questions and score
            if assessment.question_ids:
                q_result = await db.execute(
                    select(EvaluationQuestion).where(
                        EvaluationQuestion.id.in_(assessment.question_ids)
                    )
                )
            elif assessment.paper_id:
                q_result = await db.execute(
                    select(EvaluationQuestion).where(
                        EvaluationQuestion.paper_id == assessment.paper_id
                    )
                )
            else:
                continue

            questions = q_result.scalars().all()
            _score_attempt_standalone(attempt, questions, assessment)

            attempt.submitted_at = now
            attempt.status = "submitted"

            # Set metadata
            meta = dict(attempt.attempt_metadata or {})
            meta["auto_submitted"] = True
            meta["submit_reason"] = "server_timeout"
            attempt.attempt_metadata = meta
            flag_modified(attempt, "attempt_metadata")

            # Update invitation status if linked
            if attempt.invitation_id:
                inv_result = await db.execute(
                    select(EvaluationInvitation).where(
                        EvaluationInvitation.id == attempt.invitation_id
                    )
                )
                inv = inv_result.scalar_one_or_none()
                if inv and inv.status != "completed":
                    inv.status = "completed"

            submitted_count += 1
            print(
                f"[StaleAttemptScheduler] Auto-submitted attempt {attempt.id} "
                f"(started: {attempt.started_at}, score: {attempt.score}/{attempt.max_score})",
                flush=True,
            )

        await db.commit()
        print(f"[StaleAttemptScheduler] Done. Auto-submitted {submitted_count} stale attempts.", flush=True)


async def run_stale_attempt_scheduler():
    """Background loop that checks for stale attempts every 5 minutes."""
    print("[StaleAttemptScheduler] Started background stale attempt scheduler", flush=True)
    while True:
        try:
            await _cleanup_stale_attempts()
        except Exception as e:
            print(f"[StaleAttemptScheduler] Error: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # Run every 5 minutes
        await asyncio.sleep(5 * 60)
