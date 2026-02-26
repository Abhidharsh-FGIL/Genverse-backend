import uuid
from fastapi import APIRouter, Query
from sqlalchemy import select, func

from app.dependencies import DBSession, CurrentUser
from app.models.classes import Class, Assignment, Submission, ClassStudent
from app.models.assessment import AssessmentAttempt, TopicMastery
from app.models.organization import Organization, OrgMember
from app.models.user import User

router = APIRouter()


@router.get("/teacher/class/{class_id}")
async def get_class_analytics(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Teacher analytics for a specific class."""
    # Total students
    student_count_result = await db.execute(
        select(func.count(ClassStudent.id)).where(ClassStudent.class_id == class_id)
    )
    student_count = student_count_result.scalar_one()

    # Assignment completion rates
    assignments_result = await db.execute(
        select(Assignment).where(Assignment.class_id == class_id)
    )
    assignments = assignments_result.scalars().all()

    assignment_stats = []
    for assignment in assignments:
        total_submissions = await db.execute(
            select(func.count(Submission.id)).where(Submission.assignment_id == assignment.id)
        )
        graded_submissions = await db.execute(
            select(func.count(Submission.id)).where(
                Submission.assignment_id == assignment.id,
                Submission.status.in_(["graded", "returned"]),
            )
        )
        assignment_stats.append({
            "id": str(assignment.id),
            "title": assignment.title,
            "total_submissions": total_submissions.scalar_one(),
            "graded_submissions": graded_submissions.scalar_one(),
            "completion_rate": (total_submissions.scalar_one() / student_count * 100) if student_count else 0,
        })

    return {
        "class_id": str(class_id),
        "student_count": student_count,
        "assignment_count": len(assignments),
        "assignment_stats": assignment_stats,
    }


@router.get("/teacher/gradebook/{class_id}")
async def get_gradebook(class_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Full gradebook for a class."""
    students_result = await db.execute(
        select(ClassStudent, User)
        .join(User, ClassStudent.student_id == User.id)
        .where(ClassStudent.class_id == class_id)
    )
    students = students_result.all()

    assignments_result = await db.execute(
        select(Assignment)
        .where(Assignment.class_id == class_id, Assignment.status == "published")
    )
    assignments = assignments_result.scalars().all()

    gradebook = []
    for cs, student in students:
        student_grades = []
        total_score = 0
        total_possible = 0
        for assignment in assignments:
            sub_result = await db.execute(
                select(Submission).where(
                    Submission.assignment_id == assignment.id,
                    Submission.student_id == student.id,
                )
            )
            sub = sub_result.scalar_one_or_none()
            if sub and sub.grade:
                score = sub.grade.get("totalScore", 0)
                max_score = sub.grade.get("maxScore", assignment.points)
                total_score += score
                total_possible += max_score
                student_grades.append({
                    "assignment_id": str(assignment.id),
                    "assignment_title": assignment.title,
                    "score": score,
                    "max_score": max_score,
                    "percentage": (score / max_score * 100) if max_score else 0,
                })
            else:
                total_possible += assignment.points
                student_grades.append({
                    "assignment_id": str(assignment.id),
                    "assignment_title": assignment.title,
                    "score": None,
                    "max_score": assignment.points,
                    "status": sub.status if sub else "not_submitted",
                })

        gradebook.append({
            "student_id": str(student.id),
            "student_name": student.name,
            "roll_no": cs.roll_no,
            "grades": student_grades,
            "average_percentage": (total_score / total_possible * 100) if total_possible else 0,
        })

    return {
        "class_id": str(class_id),
        "assignments": [{"id": str(a.id), "title": a.title, "points": a.points} for a in assignments],
        "students": gradebook,
    }


@router.get("/org/{org_id}")
async def get_org_analytics(org_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Organization-wide analytics for org admin."""
    # Member counts
    member_counts_result = await db.execute(
        select(OrgMember.role, func.count(OrgMember.id))
        .where(OrgMember.org_id == org_id, OrgMember.status == "active")
        .group_by(OrgMember.role)
    )
    member_counts = dict(member_counts_result.all())

    # Class count
    class_count_result = await db.execute(
        select(func.count(Class.id)).where(Class.org_id == org_id, Class.is_active == True)
    )
    class_count = class_count_result.scalar_one()

    return {
        "org_id": str(org_id),
        "member_counts": member_counts,
        "class_count": class_count,
        "total_members": sum(member_counts.values()),
    }


@router.get("/user/progress")
async def get_user_progress(current_user: CurrentUser, db: DBSession):
    """Individual user's personal progress summary."""
    from app.models.content import Ebook, MindMap, UserLibraryItem

    assessments_result = await db.execute(
        select(func.count(AssessmentAttempt.id)).where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
        )
    )
    total_assessments = assessments_result.scalar_one()

    avg_score_result = await db.execute(
        select(func.avg(AssessmentAttempt.percentage)).where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
        )
    )
    avg_score = avg_score_result.scalar_one() or 0

    ebook_count_result = await db.execute(
        select(func.count(Ebook.id)).where(Ebook.user_id == current_user.id)
    )
    ebook_count = ebook_count_result.scalar_one()

    mindmap_count_result = await db.execute(
        select(func.count(MindMap.id)).where(MindMap.user_id == current_user.id)
    )
    mindmap_count = mindmap_count_result.scalar_one()

    library_count_result = await db.execute(
        select(func.count(UserLibraryItem.id)).where(UserLibraryItem.user_id == current_user.id)
    )
    library_count = library_count_result.scalar_one()

    return {
        "user_id": str(current_user.id),
        "xp": current_user.xp,
        "streak": current_user.streak,
        "total_assessments_completed": total_assessments,
        "average_assessment_score": round(avg_score, 2),
        "ebooks_created": ebook_count,
        "mindmaps_created": mindmap_count,
        "library_documents": library_count,
    }
