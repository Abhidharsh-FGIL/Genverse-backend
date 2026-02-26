import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, status, UploadFile, File, Form, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import DBSession, CurrentUser
from app.models.communication import GroupChat, GroupChatMessage, ChatReadReceipt
from app.schemas.communication import (
    GroupChatCreate, GroupChatResponse, MessageCreate, MessageResponse, ReadReceiptUpdate
)
from app.core.exceptions import NotFoundException, ForbiddenException
from app.services.storage_service import StorageService

router = APIRouter()


async def _check_class_membership(class_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession) -> bool:
    """Return True if user is the class teacher, a co-teacher, or an enrolled student."""
    from app.models.classes import ClassStudent, ClassTeacher, Class
    teacher_result = await db.execute(
        select(Class).where(Class.id == class_id, Class.teacher_id == user_id, Class.is_active == True)
    )
    if teacher_result.scalar_one_or_none():
        return True
    co_result = await db.execute(
        select(ClassTeacher).where(ClassTeacher.class_id == class_id, ClassTeacher.teacher_id == user_id)
    )
    if co_result.scalar_one_or_none():
        return True
    student_result = await db.execute(
        select(ClassStudent).where(ClassStudent.class_id == class_id, ClassStudent.student_id == user_id)
    )
    if student_result.scalar_one_or_none():
        return True
    return False


async def _check_is_class_teacher(class_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession) -> bool:
    """Return True if user is the main teacher or a co-teacher of the class."""
    from app.models.classes import ClassTeacher, Class
    teacher_result = await db.execute(
        select(Class).where(Class.id == class_id, Class.teacher_id == user_id, Class.is_active == True)
    )
    if teacher_result.scalar_one_or_none():
        return True
    co_result = await db.execute(
        select(ClassTeacher).where(ClassTeacher.class_id == class_id, ClassTeacher.teacher_id == user_id)
    )
    if co_result.scalar_one_or_none():
        return True
    return False


@router.post("/", response_model=GroupChatResponse, status_code=status.HTTP_201_CREATED)
async def create_group_chat(payload: GroupChatCreate, current_user: CurrentUser, db: DBSession):
    class_id = uuid.UUID(payload.class_id) if payload.class_id else None
    if class_id and not await _check_is_class_teacher(class_id, current_user.id, db):
        raise ForbiddenException("Only teachers can create a class chat")
    chat = GroupChat(
        name=payload.name,
        description=payload.description,
        class_id=class_id,
        org_id=uuid.UUID(payload.org_id) if payload.org_id else None,
        creator_id=current_user.id,
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return chat


@router.get("/", response_model=list[GroupChatResponse])
async def list_chats(
    current_user: CurrentUser,
    db: DBSession,
    class_id: str | None = Query(None),
    org_id: str | None = Query(None),
):
    q = select(GroupChat).where(GroupChat.is_active == True)
    if class_id:
        q = q.where(GroupChat.class_id == uuid.UUID(class_id))
    if org_id:
        q = q.where(GroupChat.org_id == uuid.UUID(org_id))
    result = await db.execute(q)
    chats = result.scalars().all()

    responses = []
    for chat in chats:
        # Get unread count
        read_result = await db.execute(
            select(ChatReadReceipt).where(
                ChatReadReceipt.chat_id == chat.id,
                ChatReadReceipt.user_id == current_user.id,
            )
        )
        read_receipt = read_result.scalars().first()
        last_read = read_receipt.last_read_at if read_receipt else None

        if last_read:
            unread_result = await db.execute(
                select(func.count(GroupChatMessage.id)).where(
                    GroupChatMessage.chat_id == chat.id,
                    GroupChatMessage.created_at > last_read,
                )
            )
        else:
            unread_result = await db.execute(
                select(func.count(GroupChatMessage.id)).where(GroupChatMessage.chat_id == chat.id)
            )
        unread = unread_result.scalar_one()

        r = GroupChatResponse.model_validate(chat)
        r.unread_count = unread
        responses.append(r)
    return responses


@router.get("/unread-count")
async def get_unread_count(current_user: CurrentUser, db: DBSession):
    """Return total and per-chat unread message counts for all chats the user is part of."""
    from app.models.classes import ClassStudent, ClassTeacher, Class
    from app.models.organization import OrgMember

    # Collect class IDs the user is a student, teacher, or co-teacher of
    student_result = await db.execute(
        select(ClassStudent.class_id).where(ClassStudent.student_id == current_user.id)
    )
    teacher_result = await db.execute(
        select(Class.id).where(Class.teacher_id == current_user.id, Class.is_active == True)
    )
    co_teacher_result = await db.execute(
        select(ClassTeacher.class_id).where(ClassTeacher.teacher_id == current_user.id)
    )
    class_ids = list({
        r[0] for r in (
            student_result.all() + teacher_result.all() + co_teacher_result.all()
        )
    })

    # Collect org IDs the user belongs to
    org_result = await db.execute(
        select(OrgMember.org_id).where(
            OrgMember.user_id == current_user.id,
            OrgMember.status == "active",
        )
    )
    org_ids = [r[0] for r in org_result.all()]

    if not class_ids and not org_ids:
        return {"total": 0, "per_chat": {}}

    # Fetch relevant active chats
    from sqlalchemy import or_
    conditions = []
    if class_ids:
        conditions.append(GroupChat.class_id.in_(class_ids))
    if org_ids:
        conditions.append(GroupChat.org_id.in_(org_ids))
    chats_result = await db.execute(
        select(GroupChat).where(GroupChat.is_active == True, or_(*conditions))
    )
    chats = chats_result.scalars().all()

    per_chat: dict[str, int] = {}
    total = 0
    for chat in chats:
        read_result = await db.execute(
            select(ChatReadReceipt).where(
                ChatReadReceipt.chat_id == chat.id,
                ChatReadReceipt.user_id == current_user.id,
            )
        )
        receipt = read_result.scalars().first()
        last_read = receipt.last_read_at if receipt else None

        if last_read:
            count_q = select(func.count(GroupChatMessage.id)).where(
                GroupChatMessage.chat_id == chat.id,
                GroupChatMessage.created_at > last_read,
                GroupChatMessage.is_deleted == False,
            )
        else:
            count_q = select(func.count(GroupChatMessage.id)).where(
                GroupChatMessage.chat_id == chat.id,
                GroupChatMessage.is_deleted == False,
            )
        unread = (await db.execute(count_q)).scalar_one()
        if unread > 0:
            per_chat[str(chat.id)] = unread
            total += unread

    return {"total": total, "per_chat": per_chat}


@router.get("/group/by-class/{class_id}")
async def get_chat_by_class(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Return the first active group chat for a class, or null if none exists."""
    if not await _check_class_membership(class_id, current_user.id, db):
        raise ForbiddenException("You are not a member of this class")
    result = await db.execute(
        select(GroupChat).where(
            GroupChat.class_id == class_id,
            GroupChat.is_active == True,
        )
    )
    chat = result.scalars().first()
    if not chat:
        return None
    return {"id": str(chat.id), "name": chat.name}


@router.get("/{chat_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    chat_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
    before: str | None = Query(None),
    limit: int = Query(50, le=100),
):
    chat_result = await db.execute(select(GroupChat).where(GroupChat.id == chat_id, GroupChat.is_active == True))
    chat = chat_result.scalar_one_or_none()
    if not chat:
        raise NotFoundException("Chat not found")
    if chat.class_id and not await _check_class_membership(chat.class_id, current_user.id, db):
        raise ForbiddenException("You are not a member of this class")

    from app.models.user import User
    q = (
        select(GroupChatMessage, User)
        .join(User, GroupChatMessage.user_id == User.id)
        .where(
            GroupChatMessage.chat_id == chat_id,
            GroupChatMessage.is_deleted == False,
        )
    )
    if before:
        q = q.where(GroupChatMessage.created_at < datetime.fromisoformat(before))
    q = q.order_by(GroupChatMessage.created_at.desc()).limit(limit)
    result = await db.execute(q)
    rows = result.all()
    messages = []
    for msg, user in reversed(rows):
        r = MessageResponse.model_validate(msg)
        r.sender_name = user.name
        r.sender_avatar = user.avatar_url
        messages.append(r)
    return messages


@router.post("/{chat_id}/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def send_message(
    chat_id: uuid.UUID,
    payload: MessageCreate,
    current_user: CurrentUser,
    db: DBSession,
):
    chat_result = await db.execute(select(GroupChat).where(GroupChat.id == chat_id, GroupChat.is_active == True))
    chat = chat_result.scalar_one_or_none()
    if not chat:
        raise NotFoundException("Chat not found")
    if chat.class_id and not await _check_class_membership(chat.class_id, current_user.id, db):
        raise ForbiddenException("You are not a member of this class")

    msg = GroupChatMessage(
        chat_id=chat_id,
        user_id=current_user.id,
        content=payload.content,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)

    r = MessageResponse.model_validate(msg)
    r.sender_name = current_user.name
    r.sender_avatar = current_user.avatar_url
    return r


@router.post("/{chat_id}/messages/with-attachment", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def send_message_with_attachment(
    chat_id: uuid.UUID,
    content: str = Form(""),
    file: UploadFile = File(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    chat_result = await db.execute(select(GroupChat).where(GroupChat.id == chat_id, GroupChat.is_active == True))
    chat = chat_result.scalar_one_or_none()
    if not chat:
        raise NotFoundException("Chat not found")
    if chat.class_id and not await _check_is_class_teacher(chat.class_id, current_user.id, db):
        raise ForbiddenException("Only teachers can upload attachments")

    storage = StorageService()
    file_info = await storage.upload_file(
        file=file,
        bucket="chat-attachments",
        prefix=str(chat_id),
    )

    msg = GroupChatMessage(
        chat_id=chat_id,
        user_id=current_user.id,
        content=content,
        attachments=[file_info],
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)

    r = MessageResponse.model_validate(msg)
    r.sender_name = current_user.name
    r.sender_avatar = current_user.avatar_url
    return r


@router.delete("/{chat_id}/messages/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    chat_id: uuid.UUID, message_id: uuid.UUID, current_user: CurrentUser, db: DBSession
):
    result = await db.execute(
        select(GroupChatMessage).where(
            GroupChatMessage.id == message_id,
            GroupChatMessage.chat_id == chat_id,
            GroupChatMessage.user_id == current_user.id,
        )
    )
    msg = result.scalar_one_or_none()
    if not msg:
        raise NotFoundException("Message not found")
    msg.is_deleted = True
    await db.commit()


@router.post("/{chat_id}/read")
async def mark_as_read(chat_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(ChatReadReceipt).where(
            ChatReadReceipt.chat_id == chat_id,
            ChatReadReceipt.user_id == current_user.id,
        )
    )
    receipt = result.scalars().first()
    now = datetime.now(timezone.utc)
    if not receipt:
        receipt = ChatReadReceipt(chat_id=chat_id, user_id=current_user.id, last_read_at=now)
        db.add(receipt)
    else:
        receipt.last_read_at = now
    await db.commit()
    return {"message": "Marked as read"}
