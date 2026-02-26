import uuid
from fastapi import APIRouter, status, Query
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.classes import Announcement, AnnouncementComment
from app.models.user import User
from app.schemas.classes import (
    AnnouncementCreate, AnnouncementResponse, CommentCreate, CommentResponse
)
from app.core.exceptions import NotFoundException, ForbiddenException

router = APIRouter()


@router.post("/", response_model=AnnouncementResponse, status_code=status.HTTP_201_CREATED)
async def create_announcement(payload: AnnouncementCreate, current_user: CurrentUser, db: DBSession):
    announcement = Announcement(
        class_id=uuid.UUID(payload.class_id),
        author_id=current_user.id,
        content=payload.content,
        allow_comments=payload.allow_comments,
    )
    db.add(announcement)
    await db.commit()
    await db.refresh(announcement)

    r = AnnouncementResponse.model_validate(announcement)
    r.author_name = current_user.name
    r.comment_count = 0
    return r


@router.get("/", response_model=list[AnnouncementResponse])
async def list_announcements(
    current_user: CurrentUser,
    db: DBSession,
    class_id: str | None = Query(None),
    limit: int = Query(20, le=100),
):
    q = select(Announcement, User).join(User, Announcement.author_id == User.id)
    if class_id:
        q = q.where(Announcement.class_id == uuid.UUID(class_id))
    q = q.order_by(Announcement.created_at.desc()).limit(limit)
    result = await db.execute(q)
    rows = result.all()

    responses = []
    for ann, user in rows:
        comment_result = await db.execute(
            select(AnnouncementComment).where(AnnouncementComment.announcement_id == ann.id)
        )
        comments = comment_result.scalars().all()
        r = AnnouncementResponse.model_validate(ann)
        r.author_name = user.name
        r.comment_count = len(comments)
        responses.append(r)
    return responses


@router.get("/{announcement_id}", response_model=AnnouncementResponse)
async def get_announcement(announcement_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(Announcement, User)
        .join(User, Announcement.author_id == User.id)
        .where(Announcement.id == announcement_id)
    )
    row = result.one_or_none()
    if not row:
        raise NotFoundException("Announcement not found")
    ann, user = row

    comment_result = await db.execute(
        select(AnnouncementComment).where(AnnouncementComment.announcement_id == ann.id)
    )
    r = AnnouncementResponse.model_validate(ann)
    r.author_name = user.name
    r.comment_count = len(comment_result.scalars().all())
    return r


@router.delete("/{announcement_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_announcement(announcement_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(Announcement).where(Announcement.id == announcement_id))
    ann = result.scalar_one_or_none()
    if not ann:
        raise NotFoundException("Announcement not found")
    if ann.author_id != current_user.id:
        raise ForbiddenException("Not your announcement")
    await db.delete(ann)
    await db.commit()


@router.post("/{announcement_id}/comments", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
async def add_comment(
    announcement_id: uuid.UUID,
    payload: CommentCreate,
    current_user: CurrentUser,
    db: DBSession,
):
    ann_result = await db.execute(select(Announcement).where(Announcement.id == announcement_id))
    ann = ann_result.scalar_one_or_none()
    if not ann:
        raise NotFoundException("Announcement not found")
    if not ann.allow_comments:
        raise ForbiddenException("Comments are disabled for this announcement")

    comment = AnnouncementComment(
        announcement_id=announcement_id,
        author_id=current_user.id,
        content=payload.content,
    )
    db.add(comment)
    await db.commit()
    await db.refresh(comment)

    r = CommentResponse.model_validate(comment)
    r.author_name = current_user.name
    return r


@router.get("/{announcement_id}/comments", response_model=list[CommentResponse])
async def list_comments(announcement_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(AnnouncementComment, User)
        .join(User, AnnouncementComment.author_id == User.id)
        .where(AnnouncementComment.announcement_id == announcement_id)
        .order_by(AnnouncementComment.created_at.asc())
    )
    rows = result.all()
    return [
        CommentResponse(
            id=c.id, announcement_id=c.announcement_id, author_id=c.author_id,
            content=c.content, created_at=c.created_at, author_name=u.name,
        )
        for c, u in rows
    ]
