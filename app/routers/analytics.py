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
    from app.models.ai import AiChat

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

    chat_count_result = await db.execute(
        select(func.count(AiChat.id)).where(AiChat.user_id == current_user.id)
    )
    chat_count = chat_count_result.scalar_one()

    return {
        "user_id": str(current_user.id),
        "xp": current_user.xp,
        "streak": current_user.streak,
        "total_assessments_completed": total_assessments,
        "average_assessment_score": round(avg_score, 2),
        "ebooks_created": ebook_count,
        "mindmaps_created": mindmap_count,
        "library_documents": library_count,
        "total_chats": chat_count,
    }


@router.get("/user/monthly-comparison")
async def get_monthly_comparison(current_user: CurrentUser, db: DBSession):
    """This-month vs last-month counts for assessments, documents, chats, and avg score."""
    from datetime import datetime, timezone, timedelta
    from app.models.content import UserLibraryItem
    from app.models.ai import AiChat

    now = datetime.now(timezone.utc)
    first_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    first_last = (first_this - timedelta(days=1)).replace(day=1)

    async def count_rows(model, date_col, start, end):
        r = await db.execute(
            select(func.count(model.id))
            .where(model.user_id == current_user.id, date_col >= start, date_col < end)
        )
        return r.scalar_one()

    async def avg_pct(start, end):
        r = await db.execute(
            select(func.avg(AssessmentAttempt.percentage))
            .where(
                AssessmentAttempt.user_id == current_user.id,
                AssessmentAttempt.status == "evaluated",
                AssessmentAttempt.submitted_at >= start,
                AssessmentAttempt.submitted_at < end,
            )
        )
        return round(r.scalar_one() or 0, 1)

    return {
        "thisMonth": {
            "assessments": await count_rows(AssessmentAttempt, AssessmentAttempt.submitted_at, first_this, now),
            "documents":   await count_rows(UserLibraryItem, UserLibraryItem.created_at, first_this, now),
            "chats":       await count_rows(AiChat, AiChat.created_at, first_this, now),
            "avgScore":    await avg_pct(first_this, now),
        },
        "lastMonth": {
            "assessments": await count_rows(AssessmentAttempt, AssessmentAttempt.submitted_at, first_last, first_this),
            "documents":   await count_rows(UserLibraryItem, UserLibraryItem.created_at, first_last, first_this),
            "chats":       await count_rows(AiChat, AiChat.created_at, first_last, first_this),
            "avgScore":    await avg_pct(first_last, first_this),
        },
    }


@router.get("/user/score-trend")
async def get_score_trend(
    current_user: CurrentUser,
    db: DBSession,
    limit: int = Query(12, le=30),
):
    """Last N assessed scores with real submission dates, sorted ascending."""
    from app.models.assessment import PracticeAssessment

    rows = (await db.execute(
        select(AssessmentAttempt, PracticeAssessment.subject)
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
            AssessmentAttempt.submitted_at.isnot(None),
        )
        .order_by(AssessmentAttempt.submitted_at.desc())
        .limit(limit)
    )).all()

    return [
        {
            "date":    attempt.submitted_at.strftime("%b %d"),
            "score":   round(attempt.percentage or 0, 1),
            "subject": subject or "General",
        }
        for attempt, subject in reversed(rows)
    ]


@router.get("/user/study-time")
async def get_study_time(current_user: CurrentUser, db: DBSession):
    """AI interactions + assessments grouped by subject — proxy for study activity distribution."""
    from app.models.ai import AiInteractionHistory
    from app.models.assessment import PracticeAssessment

    ai_rows = (await db.execute(
        select(AiInteractionHistory)
        .where(AiInteractionHistory.user_id == current_user.id)
        .order_by(AiInteractionHistory.created_at.desc())
        .limit(200)
    )).scalars().all()

    subject_counts: dict[str, int] = {}
    for row in ai_rows:
        ctx = row.context_snapshot or {}
        subject = ctx.get("subject") or "General"
        subject_counts[subject] = subject_counts.get(subject, 0) + 1

    attempt_rows = (await db.execute(
        select(PracticeAssessment.subject, func.count(AssessmentAttempt.id))
        .join(AssessmentAttempt, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(AssessmentAttempt.user_id == current_user.id)
        .group_by(PracticeAssessment.subject)
    )).all()
    for subject, cnt in attempt_rows:
        s = subject or "General"
        subject_counts[s] = subject_counts.get(s, 0) + cnt

    return [
        {"subject": s, "interactions": c}
        for s, c in sorted(subject_counts.items(), key=lambda x: -x[1])
        if s and c > 0
    ][:10]


@router.get("/user/activity-heatmap")
async def get_activity_heatmap(
    current_user: CurrentUser,
    db: DBSession,
    days: int = Query(30, le=90),
):
    """Daily activity count (AI chats + interactions + assessment attempts) for the last N days."""
    from datetime import datetime, timezone, timedelta, date
    from app.models.ai import AiInteractionHistory, AiChat

    since = datetime.now(timezone.utc) - timedelta(days=days)

    ai_rows = (await db.execute(
        select(
            func.date(AiInteractionHistory.created_at).label("day"),
            func.count(AiInteractionHistory.id).label("cnt"),
        )
        .where(
            AiInteractionHistory.user_id == current_user.id,
            AiInteractionHistory.created_at >= since,
        )
        .group_by(func.date(AiInteractionHistory.created_at))
    )).all()

    attempt_rows = (await db.execute(
        select(
            func.date(AssessmentAttempt.submitted_at).label("day"),
            func.count(AssessmentAttempt.id).label("cnt"),
        )
        .where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.submitted_at >= since,
        )
        .group_by(func.date(AssessmentAttempt.submitted_at))
    )).all()

    chat_rows = (await db.execute(
        select(
            func.date(AiChat.created_at).label("day"),
            func.count(AiChat.id).label("cnt"),
        )
        .where(
            AiChat.user_id == current_user.id,
            AiChat.created_at >= since,
        )
        .group_by(func.date(AiChat.created_at))
    )).all()

    daily: dict[str, int] = {}
    for row in ai_rows:
        daily[str(row.day)] = daily.get(str(row.day), 0) + row.cnt
    for row in attempt_rows:
        daily[str(row.day)] = daily.get(str(row.day), 0) + row.cnt
    for row in chat_rows:
        daily[str(row.day)] = daily.get(str(row.day), 0) + row.cnt

    today = date.today()
    result = []
    for i in range(days - 1, -1, -1):
        d = str(today - timedelta(days=i))
        result.append({"date": d, "count": daily.get(d, 0)})

    return {"dailyActivity": result}


@router.get("/user/recent-activity")
async def get_recent_activity(current_user: CurrentUser, db: DBSession):
    """Last 5 user actions merged across assessments, library, chats, and ebooks."""
    from datetime import datetime, timezone
    from app.models.assessment import AssessmentAttempt, PracticeAssessment
    from app.models.content import UserLibraryItem, Ebook
    from app.models.ai import AiChat

    events = []

    rows = (await db.execute(
        select(AssessmentAttempt, PracticeAssessment.title.label("ptitle"))
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.submitted_at.isnot(None),
        )
        .order_by(AssessmentAttempt.submitted_at.desc())
        .limit(5)
    )).all()
    for attempt, title in rows:
        events.append({"type": "assessment", "label": f"Completed: {title or 'Assessment'}",
                       "icon": "📝", "ts": attempt.submitted_at})

    lib_rows = (await db.execute(
        select(UserLibraryItem).where(UserLibraryItem.user_id == current_user.id)
        .order_by(UserLibraryItem.created_at.desc()).limit(5)
    )).scalars().all()
    for item in lib_rows:
        events.append({"type": "document", "label": f"Uploaded: {item.title or 'Document'}",
                       "icon": "📄", "ts": item.created_at})

    chat_rows = (await db.execute(
        select(AiChat).where(AiChat.user_id == current_user.id)
        .order_by(AiChat.created_at.desc()).limit(5)
    )).scalars().all()
    for chat in chat_rows:
        events.append({"type": "chat", "label": f"AI Chat: {chat.title or 'New Chat'}",
                       "icon": "💬", "ts": chat.created_at})

    ebook_rows = (await db.execute(
        select(Ebook).where(Ebook.user_id == current_user.id)
        .order_by(Ebook.created_at.desc()).limit(5)
    )).scalars().all()
    for ebook in ebook_rows:
        events.append({"type": "ebook", "label": f"Created eBook: {ebook.title or 'eBook'}",
                       "icon": "📚", "ts": ebook.created_at})

    now = datetime.now(timezone.utc)

    def normalize_ts(ts):
        if ts is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts

    events.sort(key=lambda e: normalize_ts(e["ts"]), reverse=True)

    def human_time(ts):
        if not ts:
            return ""
        t = normalize_ts(ts)
        s = int((now - t).total_seconds())
        if s < 3600:
            return f"{max(1, s // 60)} min ago"
        if s < 86400:
            return f"{s // 3600} hr{'s' if s // 3600 > 1 else ''} ago"
        return f"{s // 86400} day{'s' if s // 86400 > 1 else ''} ago"

    return [
        {"type": e["type"], "label": e["label"], "icon": e["icon"], "time": human_time(e["ts"])}
        for e in events[:5]
    ]
