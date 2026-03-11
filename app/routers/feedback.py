from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field
from app.dependencies import CurrentUser, DBSession
from app.models.feedback import Feedback
from app.services.email_service import send_feedback_notification_email

router = APIRouter()


class FeedbackCreate(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = None
    page: str | None = None


@router.post("", status_code=201)
async def submit_feedback(
    body: FeedbackCreate,
    user: CurrentUser,
    db: DBSession,
    bg: BackgroundTasks,
):
    """Submit user feedback (1-5 star rating with optional comment).
    Saves to DB and sends email notification to the support team."""
    fb = Feedback(
        user_id=user.id,
        rating=body.rating,
        comment=body.comment,
        page=body.page,
    )
    db.add(fb)
    await db.commit()

    # Send email notification in background
    bg.add_task(
        send_feedback_notification_email,
        user_name=user.name or "Unknown",
        user_email=user.email or "",
        rating=body.rating,
        comment=body.comment or "",
        page=body.page or "",
    )

    return {"status": "ok", "message": "Feedback submitted successfully"}
