import uuid
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from typing import Optional

from app.dependencies import DBSession, CurrentUser
from app.models.classes import (
    GradeSectionTeacher, Class, ClassStudent, Assignment, Submission,
    Quiz, QuizAttempt,
)
from app.models.organization import Organization, OrgMember
from app.models.user import User
from app.core.exceptions import NotFoundException, ForbiddenException

router = APIRouter()


async def _require_org_admin(user_id: uuid.UUID, org_id: uuid.UUID, db):
    result = await db.execute(
        select(OrgMember).where(
            OrgMember.org_id == org_id,
            OrgMember.user_id == user_id,
            OrgMember.role == "org_admin",
            OrgMember.status == "active",
        )
    )
    if not result.scalar_one_or_none():
        raise ForbiddenException("Organization admin access required")


class AssignClassTeacherRequest(BaseModel):
    org_id: str
    teacher_id: str
    grade: int
    section: str
    board: Optional[str] = None


@router.get("/section-comparison")
async def get_section_comparison(
    org_id: str = Query(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Compare sections within the same grade+board for admin overview.
    Uses GradeSectionTeacher assignments to determine sections, then finds
    matching classes for each grade+section+board combination."""
    parsed_oid = uuid.UUID(org_id)
    await _require_org_admin(current_user.id, parsed_oid, db)

    # Get all class teacher assignments for this org
    gst_result = await db.execute(
        select(GradeSectionTeacher, User.name).join(
            User, GradeSectionTeacher.teacher_id == User.id
        ).where(GradeSectionTeacher.org_id == parsed_oid)
    )
    gst_rows = gst_result.all()

    # Get all active classes for this org
    classes_result = await db.execute(
        select(Class).where(Class.org_id == parsed_oid, Class.is_active == True)
    )
    all_classes = classes_result.scalars().all()

    # Group GST assignments by grade+board
    grade_board_map: dict[tuple, list] = {}
    for gst, teacher_name in gst_rows:
        # Determine board from matching classes
        board = gst.board if hasattr(gst, 'board') and gst.board else None
        if not board:
            matching = [c for c in all_classes if c.grade == gst.grade and c.section == gst.section]
            if matching:
                board = matching[0].board
            else:
                # Try without section match
                matching = [c for c in all_classes if c.grade == gst.grade]
                board = matching[0].board if matching else "Unknown"
        key = (gst.grade, board)
        if key not in grade_board_map:
            grade_board_map[key] = []
        grade_board_map[key].append((gst, teacher_name))

    comparisons = []
    for (grade, board), gst_list in sorted(grade_board_map.items()):
        sections_data = []
        for gst, teacher_name in sorted(gst_list, key=lambda x: x[0].section):
            section = gst.section
            # Find classes matching this grade+section (or grade only if section doesn't match)
            section_classes = [c for c in all_classes if c.grade == grade and c.section == section]
            if not section_classes:
                section_classes = [c for c in all_classes if c.grade == grade and (c.board or "") == board]

            class_ids = [c.id for c in section_classes]

            # Count students enrolled in these classes
            student_count = 0
            if class_ids:
                sc_result = await db.execute(
                    select(func.count(func.distinct(ClassStudent.student_id))).where(
                        ClassStudent.class_id.in_(class_ids),
                    )
                )
                student_count = sc_result.scalar() or 0

            # Also count students from GST student list
            gst_students_result = await db.execute(
                select(func.count(User.id)).where(
                    User.grade == grade,
                    User.section == section,
                )
            )

            # Get assignments
            assignment_ids = []
            if class_ids:
                a_result = await db.execute(
                    select(Assignment.id, Assignment.assignment_type).where(Assignment.class_id.in_(class_ids))
                )
                a_rows = a_result.all()
                assignment_ids = [r[0] for r in a_rows]
                exam_count = sum(1 for r in a_rows if r[1] == "manual_exam")
                assignment_count_only = len(a_rows) - exam_count
            else:
                exam_count = 0
                assignment_count_only = 0

            # Get submissions stats
            total_submissions = 0
            graded_count = 0
            pending_count = 0
            total_score = 0
            total_max = 0
            scores_list = []

            if assignment_ids:
                subs_result = await db.execute(
                    select(Submission).where(Submission.assignment_id.in_(assignment_ids))
                )
                subs = subs_result.scalars().all()
                total_submissions = len(subs)
                for sub in subs:
                    if sub.status == "graded" and sub.grade and isinstance(sub.grade, dict):
                        graded_count += 1
                        sc = sub.grade.get("totalScore", 0)
                        mx = sub.grade.get("maxScore", 100)
                        total_score += sc
                        total_max += mx
                        if mx > 0:
                            scores_list.append(sc / mx * 100)
                    elif sub.status == "pending":
                        pending_count += 1

            avg_score = round(total_score / total_max * 100, 1) if total_max > 0 else None
            highest = round(max(scores_list), 1) if scores_list else None
            lowest = round(min(scores_list), 1) if scores_list else None
            pass_rate = round(sum(1 for s in scores_list if s >= 40) / len(scores_list) * 100, 1) if scores_list else None

            # Subject-wise breakdown
            subjects = []
            for c in section_classes:
                c_a_result = await db.execute(
                    select(Assignment.id, Assignment.assignment_type).where(Assignment.class_id == c.id)
                )
                c_a_rows = c_a_result.all()
                c_aids = [r[0] for r in c_a_rows]
                c_graded = 0
                c_total_score = 0
                c_total_max = 0
                c_scores = []
                c_teacher_result = await db.execute(select(User.name).where(User.id == c.teacher_id))
                c_teacher = c_teacher_result.scalar_one_or_none()
                if c_aids:
                    c_subs_result = await db.execute(
                        select(Submission).where(Submission.assignment_id.in_(c_aids))
                    )
                    for s in c_subs_result.scalars().all():
                        if s.status == "graded" and s.grade and isinstance(s.grade, dict):
                            c_graded += 1
                            sc = s.grade.get("totalScore", 0)
                            mx = s.grade.get("maxScore", 100)
                            c_total_score += sc
                            c_total_max += mx
                            if mx > 0:
                                c_scores.append(sc / mx * 100)
                subjects.append({
                    "subject": c.subject,
                    "teacher": c_teacher,
                    "assignments": len(c_aids),
                    "exams": sum(1 for r in c_a_rows if r[1] == "manual_exam"),
                    "graded": c_graded,
                    "avg_score": round(c_total_score / c_total_max * 100, 1) if c_total_max > 0 else None,
                    "highest": round(max(c_scores), 1) if c_scores else None,
                    "lowest": round(min(c_scores), 1) if c_scores else None,
                })

            sections_data.append({
                "section": section,
                "class_teacher": teacher_name,
                "students": student_count,
                "subjects": len(section_classes),
                "total_assignments": len(assignment_ids),
                "exams": exam_count,
                "assignments_only": assignment_count_only,
                "total_submissions": total_submissions,
                "graded": graded_count,
                "pending": pending_count,
                "avg_score": avg_score,
                "highest": highest,
                "lowest": lowest,
                "pass_rate": pass_rate,
                "subject_breakdown": subjects,
            })

        comparisons.append({
            "grade": grade,
            "board": board,
            "sections": sections_data,
        })

    return comparisons


@router.post("/", status_code=status.HTTP_201_CREATED)
async def assign_class_teacher(
    payload: AssignClassTeacherRequest, current_user: CurrentUser, db: DBSession
):
    """Assign a teacher as class teacher for a grade+section. Org admin only."""
    org_id = uuid.UUID(payload.org_id)
    await _require_org_admin(current_user.id, org_id, db)

    teacher_id = uuid.UUID(payload.teacher_id)

    # Verify teacher exists and is a member of the org
    member_result = await db.execute(
        select(OrgMember).where(
            OrgMember.org_id == org_id,
            OrgMember.user_id == teacher_id,
            OrgMember.role.in_(["teacher", "org_admin"]),
            OrgMember.status == "active",
        )
    )
    if not member_result.scalar_one_or_none():
        raise NotFoundException("Teacher not found in this organization")

    # Get org academic year
    org_result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = org_result.scalar_one_or_none()
    if not org:
        raise NotFoundException("Organization not found")

    academic_year = org.current_academic_year or ""

    # Check for existing assignment (unique constraint)
    existing = await db.execute(
        select(GradeSectionTeacher).where(
            GradeSectionTeacher.org_id == org_id,
            GradeSectionTeacher.grade == payload.grade,
            GradeSectionTeacher.section == payload.section,
            GradeSectionTeacher.academic_year == academic_year,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A class teacher is already assigned for Grade {payload.grade} Section {payload.section}",
        )

    gst = GradeSectionTeacher(
        org_id=org_id,
        teacher_id=teacher_id,
        grade=payload.grade,
        section=payload.section,
        board=payload.board,
        academic_year=academic_year,
    )
    db.add(gst)
    await db.commit()
    await db.refresh(gst)

    teacher_result = await db.execute(select(User).where(User.id == teacher_id))
    teacher = teacher_result.scalar_one_or_none()

    return {
        "id": str(gst.id),
        "org_id": str(gst.org_id),
        "teacher_id": str(gst.teacher_id),
        "teacher_name": teacher.name if teacher else None,
        "teacher_email": teacher.email if teacher else None,
        "grade": gst.grade,
        "section": gst.section,
        "board": gst.board,
        "academic_year": gst.academic_year,
    }


@router.get("/")
async def list_class_teachers(
    current_user: CurrentUser, db: DBSession,
    org_id: str = Query(...),
    academic_year: str | None = Query(None),
):
    """List all class teacher assignments in an org, filtered by academic year."""
    parsed_org_id = uuid.UUID(org_id)

    # Use provided academic_year or fall back to org's current
    filter_year = academic_year
    if not filter_year:
        org_result = await db.execute(select(Organization).where(Organization.id == parsed_org_id))
        org = org_result.scalar_one_or_none()
        if org and org.current_academic_year:
            filter_year = org.current_academic_year

    q = (
        select(GradeSectionTeacher, User)
        .join(User, GradeSectionTeacher.teacher_id == User.id)
        .where(GradeSectionTeacher.org_id == parsed_org_id)
    )
    if filter_year:
        q = q.where(GradeSectionTeacher.academic_year == filter_year)

    result = await db.execute(q)
    rows = result.all()

    return [
        {
            "id": str(gst.id),
            "org_id": str(gst.org_id),
            "teacher_id": str(gst.teacher_id),
            "teacher_name": user.name,
            "teacher_email": user.email,
            "grade": gst.grade,
            "section": gst.section,
            "board": gst.board,
            "academic_year": gst.academic_year,
        }
        for gst, user in rows
    ]


@router.get("/my")
async def get_my_class_teacher_assignments(
    current_user: CurrentUser, db: DBSession,
    org_id: Optional[str] = Query(None),
    academic_year: Optional[str] = Query(None),
):
    """Get current user's class teacher assignments."""
    q = select(GradeSectionTeacher).where(GradeSectionTeacher.teacher_id == current_user.id)
    if org_id:
        parsed_org_id = uuid.UUID(org_id)
        q = q.where(GradeSectionTeacher.org_id == parsed_org_id)

        # Use provided year or fall back to org's current
        filter_year = academic_year
        if not filter_year:
            org_result = await db.execute(select(Organization).where(Organization.id == parsed_org_id))
            org = org_result.scalar_one_or_none()
            if org and org.current_academic_year:
                filter_year = org.current_academic_year
        if filter_year:
            q = q.where(GradeSectionTeacher.academic_year == filter_year)

    result = await db.execute(q)
    assignments = result.scalars().all()

    return [
        {
            "id": str(a.id),
            "org_id": str(a.org_id),
            "grade": a.grade,
            "section": a.section,
            "board": a.board,
            "academic_year": a.academic_year,
        }
        for a in assignments
    ]


@router.delete("/{assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_class_teacher(
    assignment_id: uuid.UUID, current_user: CurrentUser, db: DBSession
):
    """Remove a class teacher assignment."""
    result = await db.execute(
        select(GradeSectionTeacher).where(GradeSectionTeacher.id == assignment_id)
    )
    gst = result.scalar_one_or_none()
    if not gst:
        raise NotFoundException("Class teacher assignment not found")
    await _require_org_admin(current_user.id, gst.org_id, db)
    await db.delete(gst)
    await db.commit()


@router.get("/{assignment_id}/students")
async def get_class_teacher_students(
    assignment_id: uuid.UUID, current_user: CurrentUser, db: DBSession
):
    """Get all students in the grade+section for this class teacher assignment.
    Uses class enrollments for historical years, profile matching for current year."""
    result = await db.execute(
        select(GradeSectionTeacher).where(GradeSectionTeacher.id == assignment_id)
    )
    gst = result.scalar_one_or_none()
    if not gst:
        raise NotFoundException("Class teacher assignment not found")

    # Verify access: must be the assigned teacher or org admin
    if gst.teacher_id != current_user.id:
        await _require_org_admin(current_user.id, gst.org_id, db)

    # Match students by profile grade+section
    q = (
        select(User)
        .join(OrgMember, OrgMember.user_id == User.id)
        .where(
            OrgMember.org_id == gst.org_id,
            OrgMember.role == "student",
            OrgMember.status == "active",
            User.grade == gst.grade,
            User.section == gst.section,
            User.is_active == True,
        )
    )

    students_result = await db.execute(q)
    students = students_result.scalars().all()

    return [
        {
            "id": str(s.id),
            "name": s.name,
            "email": s.email,
            "roll_number": s.roll_number,
            "grade": s.grade,
            "section": s.section,
            "board_preference": s.board_preference,
            "phone": s.phone,
            "parent_name": s.parent_name,
            "parent_phone": s.parent_phone,
            "emergency_contact_name": s.emergency_contact_name,
            "emergency_contact_phone": s.emergency_contact_phone,
            "emergency_contact_relation": s.emergency_contact_relation,
            "gender": s.gender,
            "date_of_birth": str(s.date_of_birth) if s.date_of_birth else None,
            "profile_complete": all([
                s.grade, s.section, s.board_preference, s.roll_number,
                s.emergency_contact_name, s.emergency_contact_phone,
                s.emergency_contact_relation,
            ]),
        }
        for s in students
    ]


@router.get("/{assignment_id}/classes")
async def get_class_teacher_classes(
    assignment_id: uuid.UUID, current_user: CurrentUser, db: DBSession
):
    """Get all subject classes for this grade+section in the same academic year."""
    result = await db.execute(
        select(GradeSectionTeacher).where(GradeSectionTeacher.id == assignment_id)
    )
    gst = result.scalar_one_or_none()
    if not gst:
        raise NotFoundException("Class teacher assignment not found")

    if gst.teacher_id != current_user.id:
        await _require_org_admin(current_user.id, gst.org_id, db)

    q = select(Class).where(
        Class.org_id == gst.org_id,
        Class.grade == gst.grade,
        Class.section == gst.section,
        Class.is_active == True,
    )

    classes_result = await db.execute(q)
    classes = classes_result.scalars().all()

    response = []
    for c in classes:
        # Get student count
        count_result = await db.execute(
            select(func.count(ClassStudent.id)).where(ClassStudent.class_id == c.id)
        )
        student_count = count_result.scalar_one()

        # Get assignment count
        assignment_count_result = await db.execute(
            select(func.count(Assignment.id)).where(Assignment.class_id == c.id)
        )
        assignment_count = assignment_count_result.scalar_one()

        # Get teacher name
        teacher_result = await db.execute(select(User.name).where(User.id == c.teacher_id))
        teacher_name = teacher_result.scalar_one_or_none()

        response.append({
            "id": str(c.id),
            "name": c.name,
            "subject": c.subject,
            "board": c.board,
            "grade": c.grade,
            "section": c.section,
            "teacher_id": str(c.teacher_id),
            "teacher_name": teacher_name,
            "student_count": student_count,
            "assignment_count": assignment_count,
            "academic_year": c.academic_year,
        })

    return response


@router.get("/{assignment_id}/reports")
async def get_class_teacher_reports(
    assignment_id: uuid.UUID, current_user: CurrentUser, db: DBSession
):
    """Aggregated reports across all subject classes for a class teacher's grade+section."""
    result = await db.execute(
        select(GradeSectionTeacher).where(GradeSectionTeacher.id == assignment_id)
    )
    gst = result.scalar_one_or_none()
    if not gst:
        raise NotFoundException("Class teacher assignment not found")

    if gst.teacher_id != current_user.id:
        await _require_org_admin(current_user.id, gst.org_id, db)

    # Get all classes for this grade+section (no academic year filter while feature is coming soon)
    classes_q = select(Class).where(
        Class.org_id == gst.org_id,
        Class.grade == gst.grade,
        Class.section == gst.section,
        Class.is_active == True,
    )
    classes_result = await db.execute(classes_q)
    classes = classes_result.scalars().all()
    class_ids = [c.id for c in classes]

    if not class_ids:
        return {"students": [], "subjects": [], "overall": {"total_students": 0, "total_classes": 0}}

    # Get all students in this grade+section
    students_result = await db.execute(
        select(User)
        .join(OrgMember, OrgMember.user_id == User.id)
        .where(
            OrgMember.org_id == gst.org_id,
            OrgMember.role == "student",
            OrgMember.status == "active",
            User.grade == gst.grade,
            User.section == gst.section,
            User.is_active == True,
        )
    )
    students = students_result.scalars().all()
    student_ids = [s.id for s in students]

    # Get all assignments across these classes
    assignments_result = await db.execute(
        select(Assignment).where(Assignment.class_id.in_(class_ids))
    )
    all_assignments = assignments_result.scalars().all()
    assignment_map = {a.id: a for a in all_assignments}

    # Get all submissions for these assignments
    if all_assignments:
        subs_result = await db.execute(
            select(Submission).where(
                Submission.assignment_id.in_([a.id for a in all_assignments]),
                Submission.student_id.in_(student_ids) if student_ids else True,
            )
        )
        all_submissions = subs_result.scalars().all()
    else:
        all_submissions = []

    # Build per-student performance across subjects
    student_performance = {}
    for s in students:
        student_performance[str(s.id)] = {
            "id": str(s.id),
            "name": s.name,
            "email": s.email,
            "roll_number": s.roll_number,
            "subjects": {},
        }

    # Build per-subject stats
    subject_stats = {}
    for c in classes:
        subject_stats[c.subject] = {
            "subject": c.subject,
            "class_id": str(c.id),
            "class_name": c.name,
            "total_assignments": 0,
            "total_submissions": 0,
            "graded_count": 0,
            "average_score": None,
            "scores": [],
        }

    # Count assignments per subject
    for a in all_assignments:
        class_ = next((c for c in classes if c.id == a.class_id), None)
        if class_ and class_.subject in subject_stats:
            subject_stats[class_.subject]["total_assignments"] += 1

    # Process submissions
    for sub in all_submissions:
        assignment = assignment_map.get(sub.assignment_id)
        if not assignment:
            continue
        class_ = next((c for c in classes if c.id == assignment.class_id), None)
        if not class_:
            continue

        subject = class_.subject
        sid = str(sub.student_id)

        # Update subject stats
        if subject in subject_stats:
            subject_stats[subject]["total_submissions"] += 1

        # Update student performance
        if sid in student_performance:
            if subject not in student_performance[sid]["subjects"]:
                student_performance[sid]["subjects"][subject] = {
                    "submissions": 0, "graded": 0, "total_score": 0, "max_score": 0,
                }
            student_performance[sid]["subjects"][subject]["submissions"] += 1

            if sub.grade and isinstance(sub.grade, dict) and "totalScore" in sub.grade:
                score = sub.grade.get("totalScore", 0)
                max_score = sub.grade.get("maxScore", 100)
                student_performance[sid]["subjects"][subject]["graded"] += 1
                student_performance[sid]["subjects"][subject]["total_score"] += score
                student_performance[sid]["subjects"][subject]["max_score"] += max_score

                if subject in subject_stats:
                    subject_stats[subject]["graded_count"] += 1
                    subject_stats[subject]["scores"].append(score / max_score * 100 if max_score else 0)

    # Calculate averages
    for subject, stats in subject_stats.items():
        if stats["scores"]:
            stats["average_score"] = round(sum(stats["scores"]) / len(stats["scores"]), 1)
        del stats["scores"]

    # Calculate per-student subject averages
    for sid, perf in student_performance.items():
        for subject, data in perf["subjects"].items():
            if data["max_score"] > 0:
                data["average_percentage"] = round(data["total_score"] / data["max_score"] * 100, 1)
            else:
                data["average_percentage"] = None

    return {
        "students": list(student_performance.values()),
        "subjects": list(subject_stats.values()),
        "overall": {
            "total_students": len(students),
            "total_classes": len(classes),
            "total_assignments": len(all_assignments),
            "total_submissions": len(all_submissions),
        },
    }


@router.get("/{assignment_id}/student-report/{student_id}")
async def get_student_detailed_report(
    assignment_id: uuid.UUID, student_id: uuid.UUID, current_user: CurrentUser, db: DBSession
):
    """Detailed per-student report across all subject classes for PTA-style review."""
    result = await db.execute(
        select(GradeSectionTeacher).where(GradeSectionTeacher.id == assignment_id)
    )
    gst = result.scalar_one_or_none()
    if not gst:
        raise NotFoundException("Class teacher assignment not found")

    if gst.teacher_id != current_user.id:
        await _require_org_admin(current_user.id, gst.org_id, db)

    # Get student profile
    student_result = await db.execute(select(User).where(User.id == student_id))
    student = student_result.scalar_one_or_none()
    if not student:
        raise NotFoundException("Student not found")

    # Get all classes for this grade+section (no academic year filter while feature is coming soon)
    classes_q = select(Class).where(
        Class.org_id == gst.org_id,
        Class.grade == gst.grade,
        Class.section == gst.section,
        Class.is_active == True,
    )
    classes_result = await db.execute(classes_q)
    classes = classes_result.scalars().all()

    # Build per-subject detailed report
    subject_reports = []
    for c in classes:
        # Get teacher name
        teacher_result = await db.execute(select(User.name).where(User.id == c.teacher_id))
        teacher_name = teacher_result.scalar_one_or_none()

        # Get all assignments for this class
        assignments_result = await db.execute(
            select(Assignment).where(Assignment.class_id == c.id).order_by(Assignment.created_at.desc())
        )
        class_assignments = assignments_result.scalars().all()

        assignment_details = []
        total_score = 0
        total_max = 0
        graded_count = 0

        for a in class_assignments:
            # Get this student's submission for this assignment
            sub_result = await db.execute(
                select(Submission).where(
                    Submission.assignment_id == a.id,
                    Submission.student_id == student_id,
                )
            )
            sub = sub_result.scalar_one_or_none()

            detail = {
                "assignment_id": str(a.id),
                "title": a.title,
                "topic": a.topic,
                "points": a.points,
                "assignment_type": a.assignment_type or "assignment",
                "due_date": a.due_date.isoformat() if a.due_date else None,
                "status": "not_submitted",
                "score": None,
                "max_score": None,
                "percentage": None,
                "comment": None,
                "submitted_at": None,
                "graded_at": None,
            }

            if sub:
                detail["status"] = sub.status
                detail["submitted_at"] = sub.submitted_at.isoformat() if sub.submitted_at else None
                detail["graded_at"] = sub.graded_at.isoformat() if sub.graded_at else None

                if sub.grade and isinstance(sub.grade, dict):
                    score = sub.grade.get("totalScore", 0)
                    max_score = sub.grade.get("maxScore", a.points or 100)
                    detail["score"] = score
                    detail["max_score"] = max_score
                    detail["percentage"] = round(score / max_score * 100, 1) if max_score else 0
                    detail["comment"] = sub.grade.get("overallComment", "")
                    total_score += score
                    total_max += max_score
                    graded_count += 1

            assignment_details.append(detail)

        subject_reports.append({
            "class_id": str(c.id),
            "subject": c.subject,
            "class_name": c.name,
            "teacher_name": teacher_name,
            "total_assignments": len(class_assignments),
            "submitted": sum(1 for d in assignment_details if d["status"] != "not_submitted"),
            "graded": graded_count,
            "average_percentage": round(total_score / total_max * 100, 1) if total_max > 0 else None,
            "total_score": total_score,
            "total_max": total_max,
            "assignments": assignment_details,
        })

    # Calculate overall
    all_total_score = sum(s["total_score"] for s in subject_reports)
    all_total_max = sum(s["total_max"] for s in subject_reports)

    return {
        "student": {
            "id": str(student.id),
            "name": student.name,
            "email": student.email,
            "roll_number": student.roll_number,
            "grade": student.grade,
            "section": student.section,
            "board_preference": student.board_preference,
            "gender": student.gender,
            "date_of_birth": str(student.date_of_birth) if student.date_of_birth else None,
            "parent_name": student.parent_name,
            "parent_phone": student.parent_phone,
            "emergency_contact_name": student.emergency_contact_name,
            "emergency_contact_phone": student.emergency_contact_phone,
            "emergency_contact_relation": student.emergency_contact_relation,
            "phone": student.phone,
            "address": student.address,
            "city": student.city,
            "state": student.state,
        },
        "subjects": subject_reports,
        "overall": {
            "total_subjects": len(subject_reports),
            "overall_percentage": round(all_total_score / all_total_max * 100, 1) if all_total_max > 0 else None,
            "total_score": all_total_score,
            "total_max": all_total_max,
        },
    }


@router.get("/comprehensive-report")
async def get_comprehensive_report(
    org_id: str = Query(...),
    grade: int = Query(None),
    board: str = Query(None),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Generate comprehensive report data for all sections in a grade+board.
    Returns all students, their profiles, all assignments, all scores — everything
    needed for a full downloadable report."""
    parsed_oid = uuid.UUID(org_id)
    await _require_org_admin(current_user.id, parsed_oid, db)

    # Get class teacher assignments (optionally filtered)
    q = select(GradeSectionTeacher, User.name, User.email).join(
        User, GradeSectionTeacher.teacher_id == User.id
    ).where(GradeSectionTeacher.org_id == parsed_oid)
    if grade:
        q = q.where(GradeSectionTeacher.grade == grade)

    gst_result = await db.execute(q)
    gst_rows = gst_result.all()

    # Get all active classes
    class_q = select(Class).where(Class.org_id == parsed_oid, Class.is_active == True)
    if grade:
        class_q = class_q.where(Class.grade == grade)
    if board:
        class_q = class_q.where(Class.board == board)
    classes_result = await db.execute(class_q)
    all_classes = classes_result.scalars().all()

    sections = []
    for gst, teacher_name, teacher_email in gst_rows:
        section = gst.section
        # Find matching classes
        section_classes = [c for c in all_classes if c.grade == gst.grade and c.section == section]
        if not section_classes:
            section_classes = [c for c in all_classes if c.grade == gst.grade]

        # Get the board from classes
        section_board = section_classes[0].board if section_classes else (board or "")
        if board and section_board != board:
            continue

        class_ids = [c.id for c in section_classes]

        # Get all students in these classes
        students_map = {}
        if class_ids:
            cs_result = await db.execute(
                select(ClassStudent, User).join(User, ClassStudent.student_id == User.id).where(
                    ClassStudent.class_id.in_(class_ids)
                )
            )
            for cs, user in cs_result.all():
                if str(user.id) not in students_map:
                    students_map[str(user.id)] = {
                        "id": str(user.id),
                        "name": user.name,
                        "email": user.email,
                        "roll_number": user.roll_number,
                        "gender": user.gender,
                        "date_of_birth": str(user.date_of_birth) if user.date_of_birth else None,
                        "parent_name": user.parent_name,
                        "parent_phone": user.parent_phone,
                        "phone": user.phone,
                        "subjects": {},
                    }

        # Get all assignments and submissions per class
        subject_data = []
        for c in section_classes:
            teacher_result = await db.execute(select(User.name).where(User.id == c.teacher_id))
            subj_teacher = teacher_result.scalar_one_or_none()

            a_result = await db.execute(
                select(Assignment).where(Assignment.class_id == c.id).order_by(Assignment.created_at.desc())
            )
            class_assignments = a_result.scalars().all()

            assignment_list = []
            for a in class_assignments:
                # Get all submissions for this assignment
                s_result = await db.execute(
                    select(Submission).where(Submission.assignment_id == a.id)
                )
                subs = s_result.scalars().all()

                sub_map = {}
                for sub in subs:
                    score = None
                    max_score = None
                    percentage = None
                    comment = None
                    if sub.grade and isinstance(sub.grade, dict):
                        score = sub.grade.get("totalScore", 0)
                        max_score = sub.grade.get("maxScore", a.points or 100)
                        percentage = round(score / max_score * 100, 1) if max_score else 0
                        comment = sub.grade.get("overallComment", "")
                    sub_map[str(sub.student_id)] = {
                        "status": sub.status,
                        "score": score,
                        "max_score": max_score,
                        "percentage": percentage,
                        "comment": comment,
                        "submitted_at": sub.submitted_at.isoformat() if sub.submitted_at else None,
                        "graded_at": sub.graded_at.isoformat() if sub.graded_at else None,
                    }

                assignment_list.append({
                    "id": str(a.id),
                    "title": a.title,
                    "topic": a.topic,
                    "assignment_type": a.assignment_type or "assignment",
                    "points": a.points,
                    "due_date": a.due_date.isoformat() if a.due_date else None,
                    "submissions": sub_map,
                })

                # Update student subject data
                for sid, sub_data in sub_map.items():
                    if sid in students_map:
                        if c.subject not in students_map[sid]["subjects"]:
                            students_map[sid]["subjects"][c.subject] = {"total_score": 0, "total_max": 0, "graded": 0}
                        if sub_data["score"] is not None:
                            students_map[sid]["subjects"][c.subject]["total_score"] += sub_data["score"]
                            students_map[sid]["subjects"][c.subject]["total_max"] += sub_data["max_score"]
                            students_map[sid]["subjects"][c.subject]["graded"] += 1

            subject_data.append({
                "class_id": str(c.id),
                "subject": c.subject,
                "teacher": subj_teacher,
                "assignments": assignment_list,
            })

        # Compute student averages
        students_list = list(students_map.values())
        for st in students_list:
            total_s = sum(s["total_score"] for s in st["subjects"].values())
            total_m = sum(s["total_max"] for s in st["subjects"].values())
            st["overall_percentage"] = round(total_s / total_m * 100, 1) if total_m > 0 else None
            for subj_name, subj_info in st["subjects"].items():
                subj_info["percentage"] = round(subj_info["total_score"] / subj_info["total_max"] * 100, 1) if subj_info["total_max"] > 0 else None

        sections.append({
            "grade": gst.grade,
            "section": section,
            "board": section_board,
            "class_teacher": teacher_name,
            "class_teacher_email": teacher_email,
            "students": sorted(students_list, key=lambda x: x.get("roll_number") or x["name"]),
            "subjects": subject_data,
        })

    return {"sections": sections}
