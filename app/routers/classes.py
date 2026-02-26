import uuid
import random
import string
from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Optional
from sqlalchemy import select, func

from app.dependencies import DBSession, CurrentUser
from app.models.classes import Class, ClassStudent, ClassTeacher, Assignment, Submission, PendingClassEnrollment
from app.models.user import User
from app.schemas.classes import (
    ClassCreate, ClassUpdate, ClassResponse, ClassStudentResponse, JoinClassRequest,
    AssignmentResponse, SubmissionResponse,
)
from app.core.exceptions import NotFoundException, ForbiddenException, ConflictException


class AddStudentByEmailRequest(BaseModel):
    email: EmailStr
    roll_no: Optional[str] = None
    org_id: Optional[str] = None  # fallback when the class itself has no org_id


class AddCoTeacherRequest(BaseModel):
    teacher_id: str

router = APIRouter()


def _generate_join_code(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


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
            assigned_teacher_id = uuid.UUID(payload.teacher_id)
        except ValueError:
            pass

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
        org_id=uuid.UUID(payload.org_id) if payload.org_id else None,
    )
    db.add(class_)
    await db.commit()
    await db.refresh(class_)

    # Count students
    count_result = await db.execute(
        select(func.count(ClassStudent.id)).where(ClassStudent.class_id == class_.id)
    )
    student_count = count_result.scalar_one()

    response = ClassResponse.model_validate(class_)
    response.student_count = student_count
    return response


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
            result = await db.execute(
                select(Class).where(
                    Class.org_id == uuid.UUID(org_id),
                    Class.is_active == True,
                )
            )
            all_classes = {c.id: c for c in result.scalars().all()}
            responses = []
            for c in all_classes.values():
                count_result = await db.execute(
                    select(func.count(ClassStudent.id)).where(ClassStudent.class_id == c.id)
                )
                r = ClassResponse.model_validate(c)
                r.student_count = count_result.scalar_one()
                responses.append(r)
            return responses

    # Teacher / co-teacher: return only classes they are part of
    teacher_q = select(Class).where(Class.teacher_id == current_user.id, Class.is_active == True)
    if org_id:
        teacher_q = teacher_q.where(Class.org_id == uuid.UUID(org_id))
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

    responses = []
    for c in all_classes.values():
        count_result = await db.execute(
            select(func.count(ClassStudent.id)).where(ClassStudent.class_id == c.id)
        )
        student_count = count_result.scalar_one()
        r = ClassResponse.model_validate(c)
        r.student_count = student_count
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
            Class.is_active == True,
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
        count_result = await db.execute(
            select(func.count(ClassStudent.id)).where(ClassStudent.class_id == c.id)
        )
        student_count = count_result.scalar_one()
        r = ClassResponse.model_validate(c)
        r.student_count = student_count
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
            Class.is_active == True,
        )
    )
    classes = result.scalars().all()
    responses = []
    for c in classes:
        count_result = await db.execute(
            select(func.count(ClassStudent.id)).where(ClassStudent.class_id == c.id)
        )
        student_count = count_result.scalar_one()
        r = ClassResponse.model_validate(c)
        r.student_count = student_count
        responses.append(r)
    return responses


@router.get("/{class_id}", response_model=ClassResponse)
async def get_class(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")

    count_result = await db.execute(
        select(func.count(ClassStudent.id)).where(ClassStudent.class_id == class_id)
    )
    student_count = count_result.scalar_one()
    r = ClassResponse.model_validate(class_)
    r.student_count = student_count
    return r


@router.patch("/{class_id}", response_model=ClassResponse)
async def update_class(
    class_id: uuid.UUID, payload: ClassUpdate, current_user: CurrentUser, db: DBSession
):
    result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")
    if class_.teacher_id != current_user.id:
        raise ForbiddenException("Only the class teacher can update this class")

    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(class_, key, value)
    await db.commit()
    await db.refresh(class_)

    count_result = await db.execute(
        select(func.count(ClassStudent.id)).where(ClassStudent.class_id == class_id)
    )
    student_count = count_result.scalar_one()
    r = ClassResponse.model_validate(class_)
    r.student_count = student_count
    return r


@router.delete("/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_class(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")
    if class_.teacher_id != current_user.id:
        raise ForbiddenException("Only the class teacher can delete this class")
    await db.delete(class_)
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

    count_result = await db.execute(
        select(func.count(ClassStudent.id)).where(ClassStudent.class_id == class_.id)
    )
    student_count = count_result.scalar_one()
    r = ClassResponse.model_validate(class_)
    r.student_count = student_count
    return r


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
            roll_no=cs.roll_no, joined_at=cs.joined_at,
            student_name=user.name, student_email=user.email,
        )
        for cs, user in rows
    ]


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
    class_id: uuid.UUID, payload: AddStudentByEmailRequest, current_user: CurrentUser, db: DBSession
):
    """Teacher adds a student to the class by email. Creates a pending enrollment if the user doesn't exist yet."""
    class_result = await db.execute(select(Class).where(Class.id == class_id))
    class_ = class_result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")

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
        return JSONResponse(
            status_code=202,
            content={
                "status": "pending",
                "message": f"No account found for {payload.email}. An invitation has been recorded — they will be automatically enrolled when they sign up.",
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
        roll_no=enrollment.roll_no, joined_at=enrollment.joined_at,
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

    # Allow class owner OR org admin to add co-teachers
    is_class_owner = class_.teacher_id == current_user.id
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
):
    """Get all assignments for a specific class."""
    q = select(Assignment).where(Assignment.class_id == class_id)
    if status_filter:
        q = q.where(Assignment.status == status_filter)
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
            "text_response": sub.text_response,
            "grade": sub.grade,
            "remediation_plan": sub.remediation_plan,
        })
    return submissions
