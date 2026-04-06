import uuid
import random
import string
from fastapi import APIRouter, BackgroundTasks, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Optional
from sqlalchemy import select, func, or_

from app.dependencies import DBSession, CurrentUser
from app.models.classes import Class, ClassStudent, ClassTeacher, Assignment, Submission, PendingClassEnrollment, GradeSectionTeacher, ClassGroup, ClassGroupMember
from app.models.user import User
from app.models.organization import Organization, OrgMember
from app.schemas.classes import (
    ClassCreate, ClassUpdate, ClassResponse, ClassStudentResponse, JoinClassRequest,
    AssignmentResponse, SubmissionResponse,
    ClassGroupCreate, ClassGroupUpdate, ClassGroupSetMembers, ClassGroupResponse, ClassGroupMemberInfo,
)
from app.core.exceptions import NotFoundException, ForbiddenException, ConflictException
from app.models.gamification import WorkspaceGamification, StudentBadge


class AddStudentByEmailRequest(BaseModel):
    email: EmailStr
    roll_no: Optional[str] = None
    org_id: Optional[str] = None  # fallback when the class itself has no org_id


class AddCoTeacherRequest(BaseModel):
    teacher_id: str

router = APIRouter()


def _generate_join_code(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


async def _build_class_response(class_: Class, db, student_count: int | None = None) -> ClassResponse:
    """Build a ClassResponse with teacher_name and student_count populated."""
    if student_count is None:
        count_result = await db.execute(
            select(func.count(ClassStudent.id)).where(ClassStudent.class_id == class_.id)
        )
        student_count = count_result.scalar_one()

    teacher_result = await db.execute(select(User.name).where(User.id == class_.teacher_id))
    teacher_name = teacher_result.scalar_one_or_none()

    r = ClassResponse.model_validate(class_)
    r.student_count = student_count
    r.teacher_name = teacher_name
    return r


@router.post("/", response_model=ClassResponse, status_code=status.HTTP_201_CREATED)
async def create_class(payload: ClassCreate, current_user: CurrentUser, db: DBSession):
    join_code = _generate_join_code()
    # Ensure unique join code
    while True:
        result = await db.execute(select(Class).where(Class.join_code == join_code))
        if not result.scalar_one_or_none():
            break
        join_code = _generate_join_code()

    # Allow org admin to assign a different teacher; otherwise default to creator
    assigned_teacher_id = current_user.id
    if payload.teacher_id:
        try:
            tid = uuid.UUID(payload.teacher_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid teacher_id format",
            )
        # Verify the teacher exists
        teacher_check = await db.execute(select(User.id).where(User.id == tid))
        if not teacher_check.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Selected teacher not found",
            )
        assigned_teacher_id = tid

    # Resolve org and auto-set academic year
    resolved_org_id = uuid.UUID(payload.org_id) if payload.org_id else None
    academic_year = None
    if resolved_org_id:
        org_result = await db.execute(select(Organization.current_academic_year).where(Organization.id == resolved_org_id))
        academic_year = org_result.scalar_one_or_none()

    class_ = Class(
        name=payload.name,
        board=payload.board,
        grade=payload.grade,
        subject=payload.subject,
        section=payload.section,
        join_code=join_code,
        teacher_id=assigned_teacher_id,
        color=payload.color,
        description=payload.description,
        org_id=resolved_org_id,
        academic_year=academic_year,
    )
    db.add(class_)
    await db.commit()
    await db.refresh(class_)

    return await _build_class_response(class_, db)


@router.get("/", response_model=list[ClassResponse])
async def list_classes(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    from app.models.organization import OrgMember

    # Org admin: return all classes in the org
    if org_id:
        admin_check = await db.execute(
            select(OrgMember).where(
                OrgMember.org_id == uuid.UUID(org_id),
                OrgMember.user_id == current_user.id,
                OrgMember.role == "org_admin",
                OrgMember.status == "active",
            )
        )
        if admin_check.scalar_one_or_none():
            parsed_org_id = uuid.UUID(org_id)

            # Get all active teacher/admin member IDs in this org
            members_result = await db.execute(
                select(OrgMember.user_id).where(
                    OrgMember.org_id == parsed_org_id,
                    OrgMember.role.in_(["teacher", "org_admin"]),
                    OrgMember.status == "active",
                )
            )
            org_teacher_ids = [row[0] for row in members_result.all()]

            result = await db.execute(
                select(Class).where(
                    Class.is_active == True,
                    or_(
                        Class.org_id == parsed_org_id,
                        # Also include classes from org teachers that were created without org_id
                        (Class.org_id.is_(None) & Class.teacher_id.in_(org_teacher_ids))
                        if org_teacher_ids else False,
                    ),
                )
            )
            all_classes = {c.id: c for c in result.scalars().all()}
            responses = []
            for c in all_classes.values():
                responses.append(await _build_class_response(c, db))
            return responses

    # Teacher / co-teacher: return only classes they are part of
    teacher_q = select(Class).where(Class.teacher_id == current_user.id, Class.is_active == True)
    if org_id:
        # Include classes that match org_id OR were created without org_id (personal workspace)
        parsed_org = uuid.UUID(org_id)
        teacher_q = teacher_q.where(
            or_(Class.org_id == parsed_org, Class.org_id.is_(None))
        )
    result = await db.execute(teacher_q)
    classes = result.scalars().all()

    # Also check co-teacher
    co_result = await db.execute(
        select(Class).join(ClassTeacher, ClassTeacher.class_id == Class.id).where(
            ClassTeacher.teacher_id == current_user.id, Class.is_active == True
        )
    )
    co_classes = co_result.scalars().all()
    all_classes = {c.id: c for c in list(classes) + list(co_classes)}

    # Also include classes where the user is a class teacher for matching grade+section
    class_teacher_view_ids = set()
    gst_result = await db.execute(
        select(GradeSectionTeacher).where(GradeSectionTeacher.teacher_id == current_user.id)
    )
    gst_assignments = gst_result.scalars().all()
    for gst in gst_assignments:
        gst_classes_q = select(Class).where(
            Class.org_id == gst.org_id,
            Class.grade == gst.grade,
            Class.section == gst.section,
            Class.is_active == True,
        )
        if gst.academic_year:
            gst_classes_q = gst_classes_q.where(Class.academic_year == gst.academic_year)
        gst_classes_result = await db.execute(gst_classes_q)
        for c in gst_classes_result.scalars().all():
            if c.id not in all_classes:
                all_classes[c.id] = c
                class_teacher_view_ids.add(c.id)

    responses = []
    for c in all_classes.values():
        r = await _build_class_response(c, db)
        if c.id in class_teacher_view_ids:
            r.is_class_teacher_view = True
        responses.append(r)
    return responses


@router.get("/enrolled")
async def get_enrolled_classes(
    current_user: CurrentUser,
    db: DBSession,
    org_id: Optional[str] = Query(None),
):
    """Get classes the current student is enrolled in, optionally filtered by org."""
    query = (
        select(ClassStudent, Class)
        .join(Class, ClassStudent.class_id == Class.id)
        .where(
            ClassStudent.student_id == current_user.id,
        )
    )
    if org_id:
        try:
            query = query.where(Class.org_id == uuid.UUID(org_id))
        except ValueError:
            pass

    result = await db.execute(query)
    rows = result.all()
    responses = []
    for enrollment, c in rows:
        r = await _build_class_response(c, db)
        d = r.model_dump()
        d["roll_no"] = enrollment.roll_no
        d["joined_at_enrollment"] = enrollment.joined_at.isoformat() if enrollment.joined_at else None
        responses.append(d)
    return responses


@router.get("/student/enrolled", response_model=list[ClassResponse])
async def get_enrolled_classes_legacy(current_user: CurrentUser, db: DBSession):
    """Legacy alias kept for backwards compatibility."""
    result = await db.execute(
        select(Class).join(ClassStudent, ClassStudent.class_id == Class.id).where(
            ClassStudent.student_id == current_user.id,
        )
    )
    classes = result.scalars().all()
    responses = []
    for c in classes:
        responses.append(await _build_class_response(c, db))
    return responses


@router.get("/students")
async def list_students_for_classes(
    classIds: str = Query(..., description="Comma-separated class UUIDs"),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Get students across multiple classes (for assessment distribution)."""
    class_id_list = [uuid.UUID(cid.strip()) for cid in classIds.split(",") if cid.strip()]
    if not class_id_list:
        return []
    result = await db.execute(
        select(ClassStudent, User)
        .join(User, ClassStudent.student_id == User.id)
        .where(ClassStudent.class_id.in_(class_id_list))
    )
    rows = result.all()
    return [
        ClassStudentResponse(
            id=cs.id, class_id=cs.class_id, student_id=cs.student_id,
            roll_no=cs.roll_no or user.roll_number, joined_at=cs.joined_at,
            student_name=user.name, student_email=user.email,
            student_avatar=user.avatar_url,
        )
        for cs, user in rows
    ]


@router.get("/archived", response_model=list[ClassResponse])
async def list_archived_classes(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """List archived (soft-deleted) classes. Org admins see all org archived classes; teachers see their own."""
    from app.models.organization import OrgMember

    if org_id:
        admin_check = await db.execute(
            select(OrgMember).where(
                OrgMember.org_id == uuid.UUID(org_id),
                OrgMember.user_id == current_user.id,
                OrgMember.role == "org_admin",
                OrgMember.status == "active",
            )
        )
        if admin_check.scalar_one_or_none():
            parsed_org_id = uuid.UUID(org_id)
            members_result = await db.execute(
                select(OrgMember.user_id).where(
                    OrgMember.org_id == parsed_org_id,
                    OrgMember.role.in_(["teacher", "org_admin"]),
                    OrgMember.status == "active",
                )
            )
            org_teacher_ids = [row[0] for row in members_result.all()]
            result = await db.execute(
                select(Class).where(
                    Class.is_active == False,
                    or_(
                        Class.org_id == parsed_org_id,
                        (Class.org_id.is_(None) & Class.teacher_id.in_(org_teacher_ids))
                        if org_teacher_ids else False,
                    ),
                )
            )
            classes = result.scalars().all()
            return [await _build_class_response(c, db) for c in classes]

    # Fallback: teacher's own archived classes (active AND null-org-id)
    if org_id:
        parsed_org = uuid.UUID(org_id)
        result = await db.execute(
            select(Class).where(
                Class.teacher_id == current_user.id,
                Class.is_active == False,
                or_(Class.org_id == parsed_org, Class.org_id.is_(None))
            )
        )
    else:
        result = await db.execute(
            select(Class).where(Class.teacher_id == current_user.id, Class.is_active == False)
        )
    classes = result.scalars().all()
    return [await _build_class_response(c, db) for c in classes]


@router.get("/{class_id}", response_model=ClassResponse)
async def get_class(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")

    return await _build_class_response(class_, db)


@router.patch("/{class_id}", response_model=ClassResponse)
async def update_class(
    class_id: uuid.UUID, payload: ClassUpdate, current_user: CurrentUser, db: DBSession
):
    from app.models.organization import OrgMember
    from app.models.user import UserRole

    result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")

    # Allow class teacher, co-teacher, grade-section teacher, or org admin
    is_owner = class_.teacher_id == current_user.id
    if not is_owner:
        co_result = await db.execute(
            select(ClassTeacher).where(ClassTeacher.class_id == class_id, ClassTeacher.teacher_id == current_user.id)
        )
        if co_result.scalar_one_or_none():
            is_owner = True
    if not is_owner and class_.org_id and class_.grade and class_.section:
        gst_q = select(GradeSectionTeacher).where(
            GradeSectionTeacher.org_id == class_.org_id,
            GradeSectionTeacher.teacher_id == current_user.id,
            GradeSectionTeacher.grade == class_.grade,
            GradeSectionTeacher.section == class_.section,
        )
        if class_.academic_year:
            gst_q = gst_q.where(GradeSectionTeacher.academic_year == class_.academic_year)
        gst_result = await db.execute(gst_q)
        if gst_result.scalar_one_or_none():
            is_owner = True
    is_org_admin = False
    if class_.org_id:
        # Check OrgMember table
        admin_result = await db.execute(
            select(OrgMember).where(
                OrgMember.org_id == class_.org_id,
                OrgMember.user_id == current_user.id,
                OrgMember.role == "org_admin",
                OrgMember.status == "active",
            )
        )
        is_org_admin = admin_result.scalar_one_or_none() is not None

        # Fallback: also check UserRole table for org_admin role
        if not is_org_admin:
            role_result = await db.execute(
                select(UserRole).where(
                    UserRole.user_id == current_user.id,
                    UserRole.role == "org_admin",
                )
            )
            is_org_admin = role_result.scalar_one_or_none() is not None
    if not is_owner and not is_org_admin:
        raise ForbiddenException("Only the class teacher or org admin can update this class")

    # Restoring (unarchiving) a class is restricted to org admins only
    updates = payload.model_dump(exclude_unset=True)
    if updates.get("is_active") is True and not is_org_admin:
        raise ForbiddenException("Only an org admin can restore an archived class")

    for key, value in updates.items():
        if key == "teacher_id" and value:
            value = uuid.UUID(value)
        setattr(class_, key, value)
    await db.commit()
    await db.refresh(class_)

    return await _build_class_response(class_, db)


@router.delete("/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_class(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Soft-delete (archive) a class by setting is_active = False."""
    from app.models.organization import OrgMember
    from app.models.user import UserRole

    result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")

    is_owner = class_.teacher_id == current_user.id
    if not is_owner:
        co_result = await db.execute(
            select(ClassTeacher).where(ClassTeacher.class_id == class_id, ClassTeacher.teacher_id == current_user.id)
        )
        if co_result.scalar_one_or_none():
            is_owner = True
    if not is_owner and class_.org_id and class_.grade and class_.section:
        gst_q = select(GradeSectionTeacher).where(
            GradeSectionTeacher.org_id == class_.org_id,
            GradeSectionTeacher.teacher_id == current_user.id,
            GradeSectionTeacher.grade == class_.grade,
            GradeSectionTeacher.section == class_.section,
        )
        if class_.academic_year:
            gst_q = gst_q.where(GradeSectionTeacher.academic_year == class_.academic_year)
        gst_result = await db.execute(gst_q)
        if gst_result.scalar_one_or_none():
            is_owner = True
    is_org_admin = False
    if class_.org_id:
        admin_result = await db.execute(
            select(OrgMember).where(
                OrgMember.org_id == class_.org_id,
                OrgMember.user_id == current_user.id,
                OrgMember.role == "org_admin",
                OrgMember.status == "active",
            )
        )
        is_org_admin = admin_result.scalar_one_or_none() is not None

        # Fallback: also check UserRole table for org_admin role
        if not is_org_admin:
            role_result = await db.execute(
                select(UserRole).where(
                    UserRole.user_id == current_user.id,
                    UserRole.role == "org_admin",
                )
            )
            is_org_admin = role_result.scalar_one_or_none() is not None
    if not is_owner and not is_org_admin:
        raise ForbiddenException("Only the class teacher or org admin can archive this class")

    class_.is_active = False
    await db.commit()


@router.post("/join", response_model=ClassResponse)
async def join_class(payload: JoinClassRequest, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(Class).where(Class.join_code == payload.join_code, Class.is_active == True))
    class_ = result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Invalid join code")

    existing = await db.execute(
        select(ClassStudent).where(
            ClassStudent.class_id == class_.id,
            ClassStudent.student_id == current_user.id,
        )
    )
    if existing.scalar_one_or_none():
        raise ConflictException("You are already enrolled in this class")

    enrollment = ClassStudent(class_id=class_.id, student_id=current_user.id)
    db.add(enrollment)

    # If org class, ensure the student has an active org membership
    if class_.org_id:
        from app.models.organization import OrgMember
        existing_member = await db.execute(
            select(OrgMember).where(
                OrgMember.org_id == class_.org_id,
                OrgMember.user_id == current_user.id,
                OrgMember.role == "student",
            )
        )
        org_member = existing_member.scalar_one_or_none()
        if org_member:
            org_member.status = "active"
        else:
            db.add(OrgMember(
                org_id=class_.org_id,
                user_id=current_user.id,
                role="student",
                status="active",
            ))

    await db.commit()

    return await _build_class_response(class_, db)


@router.get("/{class_id}/students", response_model=list[ClassStudentResponse])
async def list_class_students(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(ClassStudent, User)
        .join(User, ClassStudent.student_id == User.id)
        .where(ClassStudent.class_id == class_id)
    )
    rows = result.all()
    return [
        ClassStudentResponse(
            id=cs.id, class_id=cs.class_id, student_id=cs.student_id,
            roll_no=cs.roll_no or user.roll_number, joined_at=cs.joined_at,
            student_name=user.name, student_email=user.email,
            student_avatar=user.avatar_url,
        )
        for cs, user in rows
    ]


@router.get("/{class_id}/leaderboard")
async def get_class_leaderboard(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Get class leaderboard with XP, streak, and badge count for each student."""
    class_result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = class_result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")

    students_result = await db.execute(
        select(ClassStudent, User)
        .join(User, ClassStudent.student_id == User.id)
        .where(ClassStudent.class_id == class_id)
    )
    rows = students_result.all()

    leaderboard = []
    for enrollment, user in rows:
        if class_.org_id:
            ws_result = await db.execute(
                select(WorkspaceGamification).where(
                    WorkspaceGamification.user_id == user.id,
                    WorkspaceGamification.org_id == class_.org_id,
                )
            )
        else:
            ws_result = await db.execute(
                select(WorkspaceGamification).where(
                    WorkspaceGamification.user_id == user.id,
                    WorkspaceGamification.org_id.is_(None),
                )
            )
        ws_gam = ws_result.scalars().first()

        badge_count_result = await db.execute(
            select(func.count(StudentBadge.id)).where(StudentBadge.student_id == user.id)
        )
        badge_count = badge_count_result.scalar_one()

        leaderboard.append({
            "student_id": str(user.id),
            "name": user.name or "Student",
            "xp": ws_gam.xp if ws_gam else 0,
            "streak": ws_gam.streak if ws_gam else 0,
            "badge_count": badge_count,
            "is_current_user": user.id == current_user.id,
        })

    leaderboard.sort(key=lambda x: x["xp"], reverse=True)
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1

    return leaderboard


@router.get("/{class_id}/pending-invites")
async def list_pending_invites(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """List pending email invitations for students who haven't signed up yet."""
    result = await db.execute(
        select(PendingClassEnrollment).where(PendingClassEnrollment.class_id == class_id)
    )
    rows = result.scalars().all()
    return [
        {
            "id": str(p.id),
            "email": p.email,
            "roll_no": p.roll_no,
            "invited_by": str(p.invited_by),
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in rows
    ]


@router.delete("/{class_id}/pending-invites/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_pending_invite(class_id: uuid.UUID, invite_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Cancel a pending invitation."""
    result = await db.execute(
        select(PendingClassEnrollment).where(
            PendingClassEnrollment.id == invite_id,
            PendingClassEnrollment.class_id == class_id,
        )
    )
    invite = result.scalar_one_or_none()
    if not invite:
        raise NotFoundException("Pending invitation not found")
    await db.delete(invite)
    await db.commit()


@router.delete("/{class_id}/students/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_student(
    class_id: uuid.UUID, student_id: uuid.UUID, current_user: CurrentUser, db: DBSession
):
    result = await db.execute(
        select(ClassStudent).where(
            ClassStudent.class_id == class_id,
            ClassStudent.student_id == student_id,
        )
    )
    enrollment = result.scalar_one_or_none()
    if not enrollment:
        raise NotFoundException("Student not enrolled in this class")

    # Look up class org before deleting enrollment
    class_result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = class_result.scalar_one_or_none()

    await db.delete(enrollment)
    await db.flush()  # flush so the deleted row is excluded in the next query

    # If no other classes in the same org still enroll this student, deactivate their org membership
    if class_ and class_.org_id:
        from app.models.organization import OrgMember
        remaining = await db.execute(
            select(func.count(ClassStudent.id))
            .join(Class, ClassStudent.class_id == Class.id)
            .where(
                ClassStudent.student_id == student_id,
                Class.org_id == class_.org_id,
            )
        )
        if remaining.scalar_one() == 0:
            member_result = await db.execute(
                select(OrgMember).where(
                    OrgMember.org_id == class_.org_id,
                    OrgMember.user_id == student_id,
                    OrgMember.role == "student",
                )
            )
            org_member = member_result.scalar_one_or_none()
            if org_member:
                org_member.status = "inactive"

    await db.commit()


@router.post("/{class_id}/students", response_model=ClassStudentResponse, status_code=status.HTTP_201_CREATED)
async def add_student_by_email(
    class_id: uuid.UUID, payload: AddStudentByEmailRequest, current_user: CurrentUser, db: DBSession,
    background_tasks: BackgroundTasks,
):
    """Teacher adds a student to the class by email. Creates a pending enrollment if the user doesn't exist yet."""
    class_result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = class_result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")
    if not class_.is_active:
        raise HTTPException(status_code=400, detail="Cannot add students to an archived class")

    # Determine effective org_id FIRST: use class's own org_id, or fall back to
    # the one sent by the frontend (covers classes created before org_id was enforced).
    effective_org_id = class_.org_id
    if not effective_org_id and payload.org_id:
        try:
            effective_org_id = uuid.UUID(payload.org_id)
            # Also persist the org_id on the class so future operations are consistent
            class_.org_id = effective_org_id
        except ValueError:
            pass

    user_result = await db.execute(select(User).where(User.email == payload.email))
    student = user_result.scalar_one_or_none()
    if not student:
        # Student doesn't have an account yet — store a pending enrollment.
        # They will be auto-enrolled when they sign up with this email.
        dup_check = await db.execute(
            select(PendingClassEnrollment).where(
                PendingClassEnrollment.email == payload.email,
                PendingClassEnrollment.class_id == class_id,
            )
        )
        if not dup_check.scalar_one_or_none():
            pending = PendingClassEnrollment(
                email=str(payload.email),
                class_id=class_id,
                org_id=effective_org_id,
                roll_no=payload.roll_no,
                invited_by=current_user.id,
            )
            db.add(pending)
            await db.commit()

        # Send invitation email in background (don't block the response)
        from app.services.email_service import send_class_invitation_email
        # Get org name if available
        org_name = None
        if effective_org_id:
            from app.models.organization import Organization
            org_result = await db.execute(select(Organization.name).where(Organization.id == effective_org_id))
            org_name = org_result.scalar_one_or_none()
        background_tasks.add_task(
            send_class_invitation_email,
            str(payload.email),
            current_user.name or "Your Teacher",
            class_.name,
            org_name,
            class_.join_code,
        )

        return JSONResponse(
            status_code=202,
            content={
                "status": "pending",
                "message": f"No account found for {payload.email}. An invitation email has been sent — they will be automatically enrolled when they sign up.",
            },
        )

    existing = await db.execute(
        select(ClassStudent).where(
            ClassStudent.class_id == class_id,
            ClassStudent.student_id == student.id,
        )
    )
    if existing.scalar_one_or_none():
        raise ConflictException("Student is already enrolled in this class")

    enrollment = ClassStudent(
        class_id=class_id,
        student_id=student.id,
        roll_no=payload.roll_no,
    )
    db.add(enrollment)

    # Ensure the student has an active OrgMember record so they can see the org workspace.
    if effective_org_id:
        from app.models.organization import OrgMember
        existing_member = await db.execute(
            select(OrgMember).where(
                OrgMember.org_id == effective_org_id,
                OrgMember.user_id == student.id,
                OrgMember.role == "student",
            )
        )
        org_member = existing_member.scalar_one_or_none()
        if org_member:
            # Always ensure status is active when added/re-added to a class
            org_member.status = "active"
        else:
            db.add(OrgMember(
                org_id=effective_org_id,
                user_id=student.id,
                role="student",
                status="active",
            ))

    await db.commit()
    await db.refresh(enrollment)
    return ClassStudentResponse(
        id=enrollment.id, class_id=enrollment.class_id, student_id=enrollment.student_id,
        roll_no=enrollment.roll_no or student.roll_number, joined_at=enrollment.joined_at,
        student_name=student.name, student_email=student.email,
    )


@router.get("/{class_id}/co-teachers")
async def list_co_teachers(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """List co-teachers for a class."""
    result = await db.execute(
        select(ClassTeacher, User)
        .join(User, ClassTeacher.teacher_id == User.id)
        .where(ClassTeacher.class_id == class_id)
    )
    rows = result.all()
    return [
        {"id": str(ct.id), "teacher_id": str(ct.teacher_id), "name": user.name,
         "email": user.email, "role": ct.role, "added_at": ct.added_at.isoformat()}
        for ct, user in rows
    ]


@router.post("/{class_id}/co-teachers", status_code=status.HTTP_201_CREATED)
async def add_co_teacher(
    class_id: uuid.UUID, payload: AddCoTeacherRequest, current_user: CurrentUser, db: DBSession
):
    """Add a co-teacher to a class."""
    from app.models.organization import OrgMember
    class_result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = class_result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")

    # Allow class owner, co-teacher, grade-section teacher, or org admin to add co-teachers
    is_class_owner = class_.teacher_id == current_user.id
    if not is_class_owner:
        co_result = await db.execute(
            select(ClassTeacher).where(ClassTeacher.class_id == class_id, ClassTeacher.teacher_id == current_user.id)
        )
        if co_result.scalar_one_or_none():
            is_class_owner = True
    if not is_class_owner and class_.org_id and class_.grade and class_.section:
        gst_q = select(GradeSectionTeacher).where(
            GradeSectionTeacher.org_id == class_.org_id,
            GradeSectionTeacher.teacher_id == current_user.id,
            GradeSectionTeacher.grade == class_.grade,
            GradeSectionTeacher.section == class_.section,
        )
        if class_.academic_year:
            gst_q = gst_q.where(GradeSectionTeacher.academic_year == class_.academic_year)
        gst_result = await db.execute(gst_q)
        if gst_result.scalar_one_or_none():
            is_class_owner = True
    is_org_admin = False
    if class_.org_id:
        admin_result = await db.execute(
            select(OrgMember).where(
                OrgMember.org_id == class_.org_id,
                OrgMember.user_id == current_user.id,
                OrgMember.role == "org_admin",
                OrgMember.status == "active",
            )
        )
        is_org_admin = admin_result.scalar_one_or_none() is not None
    if not is_class_owner and not is_org_admin:
        raise ForbiddenException("Only the class owner or org admin can add co-teachers")

    teacher_id = uuid.UUID(payload.teacher_id)
    existing = await db.execute(
        select(ClassTeacher).where(
            ClassTeacher.class_id == class_id,
            ClassTeacher.teacher_id == teacher_id,
        )
    )
    if existing.scalar_one_or_none():
        raise ConflictException("Teacher is already a co-teacher of this class")

    co_teacher = ClassTeacher(class_id=class_id, teacher_id=teacher_id, role="co_teacher")
    db.add(co_teacher)
    await db.commit()
    await db.refresh(co_teacher)

    user_result = await db.execute(select(User).where(User.id == teacher_id))
    user = user_result.scalar_one_or_none()
    return {
        "id": str(co_teacher.id), "teacher_id": str(co_teacher.teacher_id),
        "name": user.name if user else None, "email": user.email if user else None,
        "role": co_teacher.role, "added_at": co_teacher.added_at.isoformat(),
    }


@router.delete("/{class_id}/co-teachers/{class_teacher_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_co_teacher(
    class_id: uuid.UUID, class_teacher_id: uuid.UUID, current_user: CurrentUser, db: DBSession
):
    """Remove a co-teacher from a class."""
    result = await db.execute(
        select(ClassTeacher).where(
            ClassTeacher.id == class_teacher_id,
            ClassTeacher.class_id == class_id,
        )
    )
    co_teacher = result.scalar_one_or_none()
    if not co_teacher:
        raise NotFoundException("Co-teacher record not found")
    await db.delete(co_teacher)
    await db.commit()


@router.get("/{class_id}/org-teachers")
async def list_org_teachers_for_class(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """List teachers from the same org who can be invited as co-teachers."""
    from app.models.organization import OrgMember

    class_result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = class_result.scalar_one_or_none()
    if not class_ or not class_.org_id:
        return []

    result = await db.execute(
        select(OrgMember, User)
        .join(User, OrgMember.user_id == User.id)
        .where(
            OrgMember.org_id == class_.org_id,
            OrgMember.role.in_(["teacher", "org_admin"]),
            OrgMember.status == "active",
            OrgMember.user_id != current_user.id,
        )
    )
    rows = result.all()
    return [
        {"id": str(user.id), "name": user.name, "email": user.email, "role": member.role}
        for member, user in rows
    ]


@router.get("/{class_id}/assignments", response_model=list[AssignmentResponse])
async def get_class_assignments(
    class_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
    status_filter: str | None = Query(None, alias="status"),
    type_filter: str | None = Query(None, alias="type"),
):
    """Get all assignments for a specific class. Optionally filter by type."""
    q = select(Assignment).where(Assignment.class_id == class_id)
    if status_filter:
        q = q.where(Assignment.status == status_filter)
    if type_filter:
        q = q.where(Assignment.assignment_type == type_filter)
    q = q.order_by(Assignment.created_at.desc())
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{class_id}/submissions")
async def get_class_submissions(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Get all submissions for a class's assignments, joined with student name."""
    assignment_result = await db.execute(
        select(Assignment.id).where(Assignment.class_id == class_id)
    )
    assignment_ids = [row[0] for row in assignment_result.all()]
    if not assignment_ids:
        return []

    result = await db.execute(
        select(Submission, User)
        .join(User, Submission.student_id == User.id)
        .where(Submission.assignment_id.in_(assignment_ids))
    )
    rows = result.all()
    submissions = []
    for sub, user in rows:
        submissions.append({
            "id": str(sub.id),
            "assignment_id": str(sub.assignment_id),
            "student_id": str(sub.student_id),
            "student_name": user.name,
            "student_email": user.email,
            "submitted_at": sub.submitted_at.isoformat() if sub.submitted_at else None,
            "status": sub.status,
            "files": sub.files,
            "text_response": sub.text_response,
            "grade": sub.grade,
            "remediation_plan": sub.remediation_plan,
        })
    return submissions


# ---- Auto-map: Suggest students matching class grade/section/board ----

@router.get("/{class_id}/suggested-students")
async def get_suggested_students(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Return org students whose grade, section, and board match the class, excluding already-enrolled."""
    class_result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = class_result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")
    if not class_.org_id:
        return []

    # Get already-enrolled student IDs
    enrolled_result = await db.execute(
        select(ClassStudent.student_id).where(ClassStudent.class_id == class_id)
    )
    enrolled_ids = {r[0] for r in enrolled_result.all()}

    # Query org students matching grade + section + board
    q = (
        select(User)
        .join(OrgMember, OrgMember.user_id == User.id)
        .where(
            OrgMember.org_id == class_.org_id,
            OrgMember.role == "student",
            OrgMember.status == "active",
            User.grade == class_.grade,
            User.is_active == True,
        )
    )
    if class_.section:
        q = q.where(User.section == class_.section)
    if class_.board:
        q = q.where(User.board_preference == class_.board)

    result = await db.execute(q)
    students = result.scalars().all()

    return [
        {
            "id": str(s.id),
            "name": s.name,
            "email": s.email,
            "roll_number": s.roll_number,
            "grade": s.grade,
            "section": s.section,
            "board_preference": s.board_preference,
        }
        for s in students
        if s.id not in enrolled_ids
    ]


class BulkAddStudentsRequest(BaseModel):
    student_ids: list[str]


@router.post("/{class_id}/students/bulk", status_code=status.HTTP_201_CREATED)
async def bulk_add_students(
    class_id: uuid.UUID, payload: BulkAddStudentsRequest, current_user: CurrentUser, db: DBSession
):
    """Bulk-add students to a class by their user IDs."""
    class_result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = class_result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")
    if not class_.is_active:
        raise HTTPException(status_code=400, detail="Cannot add students to an archived class")

    # Get already-enrolled student IDs
    enrolled_result = await db.execute(
        select(ClassStudent.student_id).where(ClassStudent.class_id == class_id)
    )
    enrolled_ids = {r[0] for r in enrolled_result.all()}

    added = []
    skipped = []
    for sid_str in payload.student_ids:
        try:
            sid = uuid.UUID(sid_str)
        except ValueError:
            skipped.append({"id": sid_str, "reason": "invalid UUID"})
            continue

        if sid in enrolled_ids:
            skipped.append({"id": sid_str, "reason": "already enrolled"})
            continue

        # Verify user exists
        user_result = await db.execute(select(User).where(User.id == sid))
        user = user_result.scalar_one_or_none()
        if not user:
            skipped.append({"id": sid_str, "reason": "user not found"})
            continue

        enrollment = ClassStudent(
            class_id=class_id,
            student_id=sid,
            roll_no=user.roll_number,
        )
        db.add(enrollment)
        enrolled_ids.add(sid)

        # Ensure OrgMember record exists
        if class_.org_id:
            existing_member = await db.execute(
                select(OrgMember).where(
                    OrgMember.org_id == class_.org_id,
                    OrgMember.user_id == sid,
                )
            )
            member = existing_member.scalar_one_or_none()
            if member:
                member.status = "active"
            else:
                db.add(OrgMember(org_id=class_.org_id, user_id=sid, role="student", status="active"))

        added.append({"id": sid_str, "name": user.name, "email": user.email})

    await db.commit()
    return {"added": len(added), "skipped": len(skipped), "added_students": added, "skipped_students": skipped}


# ---------------------------------------------------------------------------
# Class Groups (student grouping within a class)
# ---------------------------------------------------------------------------

async def _can_manage_class(class_: Class, user_id: uuid.UUID, db) -> bool:
    """Class owner, co-teacher, grade-section teacher, or org admin can manage."""
    if class_.teacher_id == user_id:
        return True
    co = await db.execute(
        select(ClassTeacher).where(ClassTeacher.class_id == class_.id, ClassTeacher.teacher_id == user_id)
    )
    if co.scalar_one_or_none():
        return True
    if class_.org_id and class_.grade and class_.section:
        gst_q = select(GradeSectionTeacher).where(
            GradeSectionTeacher.org_id == class_.org_id,
            GradeSectionTeacher.teacher_id == user_id,
            GradeSectionTeacher.grade == class_.grade,
            GradeSectionTeacher.section == class_.section,
        )
        if class_.academic_year:
            gst_q = gst_q.where(GradeSectionTeacher.academic_year == class_.academic_year)
        if (await db.execute(gst_q)).scalar_one_or_none():
            return True
    if class_.org_id:
        admin = await db.execute(
            select(OrgMember).where(
                OrgMember.org_id == class_.org_id,
                OrgMember.user_id == user_id,
                OrgMember.role == "org_admin",
                OrgMember.status == "active",
            )
        )
        if admin.scalar_one_or_none():
            return True
    return False


async def _serialize_group(group: ClassGroup, db) -> ClassGroupResponse:
    member_rows = await db.execute(
        select(ClassGroupMember, User)
        .join(User, ClassGroupMember.student_id == User.id)
        .where(ClassGroupMember.group_id == group.id)
    )
    members = [
        ClassGroupMemberInfo(
            student_id=m.student_id,
            name=u.name,
            email=u.email,
            roll_no=getattr(u, "roll_number", None),
            avatar=getattr(u, "avatar_url", None),
        )
        for m, u in member_rows.all()
    ]
    return ClassGroupResponse(
        id=group.id,
        class_id=group.class_id,
        name=group.name,
        description=group.description,
        color=group.color,
        created_by=group.created_by,
        created_at=group.created_at,
        members=members,
        member_count=len(members),
    )


async def _validate_member_ids(class_id: uuid.UUID, member_ids: list[str], db) -> list[uuid.UUID]:
    if not member_ids:
        return []
    try:
        ids = [uuid.UUID(m) for m in member_ids]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid member id format")
    result = await db.execute(
        select(ClassStudent.student_id).where(
            ClassStudent.class_id == class_id,
            ClassStudent.student_id.in_(ids),
        )
    )
    enrolled = {row[0] for row in result.all()}
    invalid = [str(i) for i in ids if i not in enrolled]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Students not in class: {invalid}")
    return list(enrolled)


@router.get("/{class_id}/groups", response_model=list[ClassGroupResponse])
async def list_class_groups(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    class_ = (await db.execute(select(Class).where(Class.id == class_id))).scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")
    groups = (await db.execute(
        select(ClassGroup).where(ClassGroup.class_id == class_id).order_by(ClassGroup.created_at)
    )).scalars().all()
    return [await _serialize_group(g, db) for g in groups]


@router.post("/{class_id}/groups", response_model=ClassGroupResponse, status_code=status.HTTP_201_CREATED)
async def create_class_group(
    class_id: uuid.UUID, payload: ClassGroupCreate, current_user: CurrentUser, db: DBSession
):
    class_ = (await db.execute(select(Class).where(Class.id == class_id))).scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")
    if not await _can_manage_class(class_, current_user.id, db):
        raise ForbiddenException("Not allowed to manage groups for this class")

    member_uuids = await _validate_member_ids(class_id, payload.member_ids, db)

    group = ClassGroup(
        class_id=class_id,
        name=payload.name,
        description=payload.description,
        color=payload.color,
        created_by=current_user.id,
    )
    db.add(group)
    await db.flush()
    for sid in member_uuids:
        db.add(ClassGroupMember(group_id=group.id, student_id=sid))
    await db.commit()
    await db.refresh(group)
    return await _serialize_group(group, db)


@router.patch("/{class_id}/groups/{group_id}", response_model=ClassGroupResponse)
async def update_class_group(
    class_id: uuid.UUID, group_id: uuid.UUID, payload: ClassGroupUpdate,
    current_user: CurrentUser, db: DBSession,
):
    class_ = (await db.execute(select(Class).where(Class.id == class_id))).scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")
    if not await _can_manage_class(class_, current_user.id, db):
        raise ForbiddenException("Not allowed to manage groups for this class")
    group = (await db.execute(
        select(ClassGroup).where(ClassGroup.id == group_id, ClassGroup.class_id == class_id)
    )).scalar_one_or_none()
    if not group:
        raise NotFoundException("Group not found")
    if payload.name is not None:
        group.name = payload.name
    if payload.description is not None:
        group.description = payload.description
    if payload.color is not None:
        group.color = payload.color
    await db.commit()
    await db.refresh(group)
    return await _serialize_group(group, db)


@router.put("/{class_id}/groups/{group_id}/members", response_model=ClassGroupResponse)
async def set_class_group_members(
    class_id: uuid.UUID, group_id: uuid.UUID, payload: ClassGroupSetMembers,
    current_user: CurrentUser, db: DBSession,
):
    class_ = (await db.execute(select(Class).where(Class.id == class_id))).scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")
    if not await _can_manage_class(class_, current_user.id, db):
        raise ForbiddenException("Not allowed to manage groups for this class")
    group = (await db.execute(
        select(ClassGroup).where(ClassGroup.id == group_id, ClassGroup.class_id == class_id)
    )).scalar_one_or_none()
    if not group:
        raise NotFoundException("Group not found")

    member_uuids = await _validate_member_ids(class_id, payload.member_ids, db)

    existing = (await db.execute(
        select(ClassGroupMember).where(ClassGroupMember.group_id == group_id)
    )).scalars().all()
    existing_ids = {m.student_id for m in existing}
    new_ids = set(member_uuids)

    for m in existing:
        if m.student_id not in new_ids:
            await db.delete(m)
    for sid in new_ids - existing_ids:
        db.add(ClassGroupMember(group_id=group_id, student_id=sid))
    await db.commit()
    await db.refresh(group)
    return await _serialize_group(group, db)


@router.delete("/{class_id}/groups/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_class_group(
    class_id: uuid.UUID, group_id: uuid.UUID, current_user: CurrentUser, db: DBSession
):
    class_ = (await db.execute(select(Class).where(Class.id == class_id))).scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")
    if not await _can_manage_class(class_, current_user.id, db):
        raise ForbiddenException("Not allowed to manage groups for this class")
    group = (await db.execute(
        select(ClassGroup).where(ClassGroup.id == group_id, ClassGroup.class_id == class_id)
    )).scalar_one_or_none()
    if not group:
        raise NotFoundException("Group not found")
    await db.delete(group)
    await db.commit()
