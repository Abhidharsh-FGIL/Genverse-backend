import uuid
import json
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Query
from pydantic import BaseModel
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.classes import Submission, Assignment
from app.schemas.classes import (
    SubmissionCreate, SubmissionGradeRequest, SubmissionResponse, AutoGradeRequest
)
from app.core.exceptions import NotFoundException, ForbiddenException, ConflictException
from app.services.storage_service import StorageService
from app.services.ai_service import AIService

router = APIRouter()


class SubmissionPatchRequest(BaseModel):
    grade: Optional[Any] = None
    status: Optional[str] = None
    remediation_plan: Optional[Any] = None


@router.post("/", response_model=SubmissionResponse, status_code=status.HTTP_201_CREATED)
async def create_submission(payload: SubmissionCreate, current_user: CurrentUser, db: DBSession):
    assignment_result = await db.execute(
        select(Assignment).where(Assignment.id == uuid.UUID(payload.assignment_id))
    )
    assignment = assignment_result.scalar_one_or_none()
    if not assignment:
        raise NotFoundException("Assignment not found")

    existing = await db.execute(
        select(Submission).where(
            Submission.assignment_id == assignment.id,
            Submission.student_id == current_user.id,
        )
    )
    if existing.scalar_one_or_none():
        raise ConflictException("Submission already exists for this assignment")

    now = datetime.now(timezone.utc)
    is_late = assignment.due_date and now > assignment.due_date

    # Store answers as JSON in text_response if provided separately
    text_response = payload.text_response
    if text_response is None and payload.answers is not None:
        text_response = json.dumps(payload.answers)

    submission = Submission(
        assignment_id=assignment.id,
        student_id=current_user.id,
        submitted_at=now,
        status="late" if is_late else "submitted",
        text_response=text_response,
        files=payload.files or [],
    )
    db.add(submission)
    await db.commit()
    await db.refresh(submission)
    return submission


@router.post("/{submission_id}/files")
async def upload_submission_file(
    submission_id: uuid.UUID,
    file: UploadFile = File(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise NotFoundException("Submission not found")
    if submission.student_id != current_user.id:
        raise ForbiddenException("Not your submission")

    storage = StorageService()
    file_info = await storage.upload_file(
        file=file,
        bucket="assignment-files",
        prefix=str(submission_id),
    )
    current_files = submission.files or []
    current_files.append(file_info)
    submission.files = current_files
    await db.commit()
    return {"file": file_info, "message": "File uploaded"}


@router.get("/", response_model=list[SubmissionResponse])
async def list_submissions(
    current_user: CurrentUser,
    db: DBSession,
    assignment_id: str | None = Query(None),
    student_id: str | None = Query(None),
    status: str | None = Query(None),
):
    from app.models.user import User
    q = (
        select(Submission, User)
        .join(User, Submission.student_id == User.id)
    )
    if assignment_id:
        q = q.where(Submission.assignment_id == uuid.UUID(assignment_id))
    if student_id:
        q = q.where(Submission.student_id == uuid.UUID(student_id))
    if status:
        q = q.where(Submission.status == status)
    q = q.order_by(Submission.submitted_at.desc())
    result = await db.execute(q)
    rows = result.all()
    submissions = []
    for sub, user in rows:
        r = SubmissionResponse.model_validate(sub)
        r.student_name = user.name
        submissions.append(r)
    return submissions


@router.get("/{submission_id}", response_model=SubmissionResponse)
async def get_submission(submission_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    from app.models.user import User
    result = await db.execute(
        select(Submission, User).join(User, Submission.student_id == User.id).where(Submission.id == submission_id)
    )
    row = result.one_or_none()
    if not row:
        raise NotFoundException("Submission not found")
    sub, user = row
    r = SubmissionResponse.model_validate(sub)
    r.student_name = user.name
    return r


@router.patch("/{submission_id}", response_model=SubmissionResponse)
async def patch_submission(
    submission_id: uuid.UUID,
    payload: SubmissionPatchRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Update a submission's grade and/or status (used by the grading page)."""
    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise NotFoundException("Submission not found")

    if payload.grade is not None:
        submission.grade = payload.grade
        submission.graded_by = current_user.id
        submission.graded_at = datetime.now(timezone.utc)
    if payload.status is not None:
        submission.status = payload.status
    if payload.remediation_plan is not None:
        submission.remediation_plan = payload.remediation_plan

    await db.commit()
    await db.refresh(submission)
    return submission


@router.post("/{submission_id}/grade", response_model=SubmissionResponse)
async def grade_submission(
    submission_id: uuid.UUID,
    payload: SubmissionGradeRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise NotFoundException("Submission not found")

    submission.grade = payload.grade
    submission.remediation_plan = payload.remediation_plan
    submission.graded_by = current_user.id
    submission.graded_at = datetime.now(timezone.utc)
    submission.status = "returned" if payload.return_to_student else "graded"
    await db.commit()
    await db.refresh(submission)
    return submission


@router.post("/auto-grade")
async def auto_grade_submission(
    payload: AutoGradeRequest, current_user: CurrentUser, db: DBSession
):
    """Use AI to suggest a grade for a submission based on a rubric."""
    result = await db.execute(select(Submission).where(Submission.id == uuid.UUID(payload.submission_id)))
    submission = result.scalar_one_or_none()
    if not submission:
        raise NotFoundException("Submission not found")

    ai = AIService()
    suggestion = await ai.auto_grade(
        submission_id=payload.submission_id,
        rubric_id=payload.rubric_id,
        db=db,
    )
    submission.ai_grade_suggestion = suggestion
    await db.commit()
    return {"suggestion": suggestion, "message": "AI grade suggestion generated"}
