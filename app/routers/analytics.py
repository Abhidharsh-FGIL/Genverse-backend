import uuid
from fastapi import APIRouter, Query
from sqlalchemy import select, func, case, or_

from app.dependencies import DBSession, CurrentUser
from app.models.classes import Class, Assignment, Submission, ClassStudent
from app.models.assessment import AssessmentAttempt, TopicMastery, PracticeAssessment
from app.models.organization import Organization, OrgMember
from app.models.user import User

router = APIRouter()


def _parse_org_id(org_id: str | None):
    if not org_id or org_id == "personal":
        return None
    try:
        return uuid.UUID(org_id)
    except ValueError:
        return None


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

    # Batch query: get submission counts per assignment in a single query
    assignment_ids = [a.id for a in assignments]
    sub_stats = {}
    if assignment_ids:
        stats_result = await db.execute(
            select(
                Submission.assignment_id,
                func.count(Submission.id).label("total"),
                func.count(case((Submission.status.in_(["graded", "returned"]), 1))).label("graded"),
            )
            .where(Submission.assignment_id.in_(assignment_ids))
            .group_by(Submission.assignment_id)
        )
        for row in stats_result.all():
            sub_stats[row.assignment_id] = {"total": row.total, "graded": row.graded}

    assignment_stats = []
    for assignment in assignments:
        stats = sub_stats.get(assignment.id, {"total": 0, "graded": 0})
        assignment_stats.append({
            "id": str(assignment.id),
            "title": assignment.title,
            "total_submissions": stats["total"],
            "graded_submissions": stats["graded"],
            "completion_rate": (stats["total"] / student_count * 100) if student_count else 0,
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

    # Bulk fetch all submissions for this class's published assignments in one query
    assignment_ids = [a.id for a in assignments]
    sub_lookup: dict[tuple, Submission] = {}
    if assignment_ids:
        all_subs_result = await db.execute(
            select(Submission).where(Submission.assignment_id.in_(assignment_ids))
        )
        for sub in all_subs_result.scalars().all():
            sub_lookup[(sub.assignment_id, sub.student_id)] = sub

    gradebook = []
    for cs, student in students:
        student_grades = []
        total_score = 0
        total_possible = 0
        for assignment in assignments:
            sub = sub_lookup.get((assignment.id, student.id))
            has_grade = (
                sub is not None
                and sub.grade
                and isinstance(sub.grade, dict)
                and ("totalScore" in sub.grade or "score" in sub.grade)
            )
            if has_grade:
                score = sub.grade.get("totalScore", sub.grade.get("score", 0))
                max_score = sub.grade.get("maxScore", sub.grade.get("max_score", assignment.points))
                total_score += score
                total_possible += max_score
                student_grades.append({
                    "assignment_id": str(assignment.id),
                    "assignment_title": assignment.title,
                    "score": score,
                    "max_score": max_score,
                    "percentage": (score / max_score * 100) if max_score else 0,
                    "status": sub.status,
                })
            elif sub is not None and sub.status in ("graded", "returned"):
                # Grade was published but grade dict might be missing totalScore key
                score = sub.grade.get("totalScore", 0) if isinstance(sub.grade, dict) else 0
                max_score = assignment.points
                total_score += score
                total_possible += max_score
                student_grades.append({
                    "assignment_id": str(assignment.id),
                    "assignment_title": assignment.title,
                    "score": score,
                    "max_score": max_score,
                    "percentage": (score / max_score * 100) if max_score else 0,
                    "status": sub.status,
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
    """Comprehensive organization-wide analytics for org admin dashboard."""
    from datetime import datetime, timezone, timedelta
    from app.models.content import Ebook, UserLibraryItem
    from app.models.ai import AiChat, AiInteractionHistory
    from app.models.subscription import Subscription, PointTransaction
    from app.models.evaluation import EvaluationAssessment, EvaluationAttempt

    now = datetime.now(timezone.utc)
    thirty_days_ago = now - timedelta(days=30)
    seven_days_ago = now - timedelta(days=7)

    # --- Member counts by role ---
    member_counts_result = await db.execute(
        select(OrgMember.role, func.count(OrgMember.id))
        .where(OrgMember.org_id == org_id, OrgMember.status == "active")
        .group_by(OrgMember.role)
    )
    member_counts = dict(member_counts_result.all())
    total_members = sum(member_counts.values())

    # --- All member user_ids for sub-queries ---
    member_ids_result = await db.execute(
        select(OrgMember.user_id).where(OrgMember.org_id == org_id, OrgMember.status == "active")
    )
    member_ids = [r[0] for r in member_ids_result.all()]

    # --- Classes (org-scoped + null-org-id classes from org teachers) ---
    org_teacher_ids = [
        r for r in member_ids  # member_ids already fetched above
    ]
    classes_result = await db.execute(
        select(Class.id, Class.name, Class.subject, Class.grade, Class.teacher_id)
        .where(
            Class.is_active == True,
            or_(
                Class.org_id == org_id,
                (Class.org_id.is_(None) & Class.teacher_id.in_(org_teacher_ids))
                if org_teacher_ids else False,
            ),
        )
    )
    classes = classes_result.all()
    class_ids = [c.id for c in classes]

    # --- Total students across all classes ---
    total_students = 0
    if class_ids:
        ts_result = await db.execute(
            select(func.count(func.distinct(ClassStudent.student_id)))
            .where(ClassStudent.class_id.in_(class_ids))
        )
        total_students = ts_result.scalar_one()

    # --- Students per class ---
    students_per_class = {}
    if class_ids:
        spc_result = await db.execute(
            select(ClassStudent.class_id, func.count(ClassStudent.id))
            .where(ClassStudent.class_id.in_(class_ids))
            .group_by(ClassStudent.class_id)
        )
        students_per_class = dict(spc_result.all())

    # --- Assignments & Submissions ---
    total_assignments = 0
    total_submissions = 0
    graded_submissions = 0
    if class_ids:
        assign_result = await db.execute(
            select(func.count(Assignment.id))
            .where(Assignment.class_id.in_(class_ids))
        )
        total_assignments = assign_result.scalar_one()

        sub_result = await db.execute(
            select(
                func.count(Submission.id),
                func.count(case((Submission.status.in_(["graded", "returned"]), 1))),
            )
            .join(Assignment, Submission.assignment_id == Assignment.id)
            .where(Assignment.class_id.in_(class_ids))
        )
        row = sub_result.one()
        total_submissions = row[0]
        graded_submissions = row[1]

    # --- Submission status distribution ---
    submission_statuses = {}
    if class_ids:
        ss_result = await db.execute(
            select(Submission.status, func.count(Submission.id))
            .join(Assignment, Submission.assignment_id == Assignment.id)
            .where(Assignment.class_id.in_(class_ids))
            .group_by(Submission.status)
        )
        submission_statuses = dict(ss_result.all())

    # --- Practice Assessments (org-scoped) ---
    practice_count_result = await db.execute(
        select(func.count(PracticeAssessment.id))
        .where(PracticeAssessment.org_id == org_id)
    )
    practice_count = practice_count_result.scalar_one()

    # Practice attempts & avg score
    practice_stats_result = await db.execute(
        select(
            func.count(AssessmentAttempt.id),
            func.avg(AssessmentAttempt.percentage),
        )
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            PracticeAssessment.org_id == org_id,
            AssessmentAttempt.status == "evaluated",
        )
    )
    ps_row = practice_stats_result.one()
    practice_attempts = ps_row[0] or 0
    practice_avg_score = round(ps_row[1] or 0, 1)

    # --- Evaluation Assessments ---
    eval_count_result = await db.execute(
        select(func.count(EvaluationAssessment.id))
        .where(EvaluationAssessment.org_id == org_id)
    )
    eval_count = eval_count_result.scalar_one()

    eval_stats_result = await db.execute(
        select(
            func.count(EvaluationAttempt.id),
            func.avg(EvaluationAttempt.percentage),
        )
        .join(EvaluationAssessment, EvaluationAttempt.assessment_id == EvaluationAssessment.id)
        .where(
            EvaluationAssessment.org_id == org_id,
            EvaluationAttempt.status == "evaluated",
        )
    )
    es_row = eval_stats_result.one()
    eval_attempts = es_row[0] or 0
    eval_avg_score = round(es_row[1] or 0, 1)

    # --- Points usage: derive from subscription (quota - balance) for monthly ---
    points_balance = 0
    points_quota = 0
    if member_ids:
        pass  # member_ids needed for top-users below

    # Org subscription
    sub_result2 = await db.execute(
        select(Subscription.points_balance, Subscription.points_monthly_quota, Subscription.plan)
        .where(Subscription.org_id == org_id, Subscription.status.in_(["active", "trialing"]))
        .limit(1)
    )
    sub_row = sub_result2.first()
    if sub_row:
        points_balance = sub_row.points_balance or 0
        points_quota = sub_row.points_monthly_quota or 0

    total_points_used = max(0, points_quota - points_balance)

    # --- Top points users ---
    top_users_data = []
    if member_ids:
        tu_result = await db.execute(
            select(
                PointTransaction.user_id,
                func.sum(PointTransaction.points_used).label("total_pts"),
            )
            .where(PointTransaction.user_id.in_(member_ids), PointTransaction.points_used > 0)
            .group_by(PointTransaction.user_id)
            .order_by(func.sum(PointTransaction.points_used).desc())
            .limit(10)
        )
        top_user_rows = tu_result.all()
        if top_user_rows:
            top_uids = [r.user_id for r in top_user_rows]
            names_result = await db.execute(
                select(User.id, User.name).where(User.id.in_(top_uids))
            )
            name_map = {r.id: r.name for r in names_result.all()}
            top_users_data = [
                {"name": name_map.get(r.user_id, "Unknown"), "points": r.total_pts}
                for r in top_user_rows
            ]

    # --- Class performance (avg submission scores per class) ---
    class_performance = []
    if class_ids:
        cp_result = await db.execute(
            select(
                Assignment.class_id,
                func.avg(
                    case(
                        (Submission.grade.isnot(None), Submission.grade["totalScore"].as_float()),
                        else_=None,
                    )
                ).label("avg_score"),
                func.count(Submission.id).label("sub_count"),
            )
            .join(Submission, Submission.assignment_id == Assignment.id)
            .where(
                Assignment.class_id.in_(class_ids),
                Submission.status.in_(["graded", "returned"]),
            )
            .group_by(Assignment.class_id)
        )
        cp_rows = cp_result.all()
        class_name_map = {c.id: c.name for c in classes}
        for row in cp_rows:
            class_performance.append({
                "class_name": class_name_map.get(row.class_id, "Unknown"),
                "avg_score": round(row.avg_score or 0, 1),
                "submissions": row.sub_count,
            })

    # --- Recent activity (last 30 days counts) ---
    recent_submissions = 0
    if class_ids:
        rs_result = await db.execute(
            select(func.count(Submission.id))
            .join(Assignment, Submission.assignment_id == Assignment.id)
            .where(Assignment.class_id.in_(class_ids), Submission.submitted_at >= thirty_days_ago)
        )
        recent_submissions = rs_result.scalar_one()

    recent_practice = 0
    rp_result = await db.execute(
        select(func.count(AssessmentAttempt.id))
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(PracticeAssessment.org_id == org_id, AssessmentAttempt.submitted_at >= thirty_days_ago)
    )
    recent_practice = rp_result.scalar_one()

    # --- AI usage (org members) ---
    total_chats = 0
    total_ai_interactions = 0
    if member_ids:
        tc_result = await db.execute(
            select(func.count(AiChat.id)).where(AiChat.user_id.in_(member_ids))
        )
        total_chats = tc_result.scalar_one()

        tai_result = await db.execute(
            select(func.count(AiInteractionHistory.id)).where(AiInteractionHistory.user_id.in_(member_ids))
        )
        total_ai_interactions = tai_result.scalar_one()

    # --- Content created by org members ---
    total_ebooks = 0
    total_documents = 0
    if member_ids:
        eb_result = await db.execute(
            select(func.count(Ebook.id)).where(Ebook.user_id.in_(member_ids))
        )
        total_ebooks = eb_result.scalar_one()

        doc_result = await db.execute(
            select(func.count(UserLibraryItem.id)).where(UserLibraryItem.user_id.in_(member_ids))
        )
        total_documents = doc_result.scalar_one()

    # --- Weekly activity trend (last 4 weeks) ---
    weekly_activity = []
    for i in range(3, -1, -1):
        week_start = now - timedelta(weeks=i + 1)
        week_end = now - timedelta(weeks=i)
        wk_count = 0
        if member_ids:
            wk_result = await db.execute(
                select(func.count(AiInteractionHistory.id))
                .where(
                    AiInteractionHistory.user_id.in_(member_ids),
                    AiInteractionHistory.created_at >= week_start,
                    AiInteractionHistory.created_at < week_end,
                )
            )
            wk_count = wk_result.scalar_one()
        weekly_activity.append({
            "week": f"Week {4 - i}",
            "interactions": wk_count,
        })

    # --- Assessment score distribution ---
    score_distribution = {"90-100": 0, "75-89": 0, "60-74": 0, "40-59": 0, "0-39": 0}
    dist_result = await db.execute(
        select(AssessmentAttempt.percentage)
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            PracticeAssessment.org_id == org_id,
            AssessmentAttempt.status == "evaluated",
            AssessmentAttempt.percentage.isnot(None),
        )
    )
    for (pct,) in dist_result.all():
        if pct >= 90:
            score_distribution["90-100"] += 1
        elif pct >= 75:
            score_distribution["75-89"] += 1
        elif pct >= 60:
            score_distribution["60-74"] += 1
        elif pct >= 40:
            score_distribution["40-59"] += 1
        else:
            score_distribution["0-39"] += 1

    # Also add evaluation attempt scores
    eval_dist_result = await db.execute(
        select(EvaluationAttempt.percentage)
        .join(EvaluationAssessment, EvaluationAttempt.assessment_id == EvaluationAssessment.id)
        .where(
            EvaluationAssessment.org_id == org_id,
            EvaluationAttempt.status == "evaluated",
            EvaluationAttempt.percentage.isnot(None),
        )
    )
    for (pct,) in eval_dist_result.all():
        if pct >= 90:
            score_distribution["90-100"] += 1
        elif pct >= 75:
            score_distribution["75-89"] += 1
        elif pct >= 60:
            score_distribution["60-74"] += 1
        elif pct >= 40:
            score_distribution["40-59"] += 1
        else:
            score_distribution["0-39"] += 1

    return {
        "org_id": str(org_id),
        # Summary
        "total_members": total_members,
        "member_counts": {k: v for k, v in member_counts.items()},
        "class_count": len(classes),
        "total_students": total_students,
        "total_assignments": total_assignments,
        "total_submissions": total_submissions,
        "graded_submissions": graded_submissions,
        "submission_statuses": submission_statuses,
        # Assessments
        "practice_assessments": practice_count,
        "evaluation_assessments": eval_count,
        "total_assessments": practice_count + eval_count,
        "practice_attempts": practice_attempts,
        "practice_avg_score": practice_avg_score,
        "eval_attempts": eval_attempts,
        "eval_avg_score": eval_avg_score,
        # Points
        "total_points_used": total_points_used,
        "points_balance": points_balance,
        "points_quota": points_quota,
        "top_users": top_users_data,
        # Content
        "total_ebooks": total_ebooks,
        "total_documents": total_documents,
        "total_chats": total_chats,
        "total_ai_interactions": total_ai_interactions,
        # Class details
        "classes": [
            {
                "id": str(c.id),
                "name": c.name,
                "subject": c.subject,
                "grade": c.grade,
                "students": students_per_class.get(c.id, 0),
            }
            for c in classes
        ],
        "class_performance": class_performance,
        # Trends
        "weekly_activity": weekly_activity,
        "recent_submissions_30d": recent_submissions,
        "recent_practice_30d": recent_practice,
        "score_distribution": score_distribution,
    }


@router.get("/org/{org_id}/member-usage")
async def get_org_member_usage(org_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Per-member points usage for the current billing cycle."""
    from datetime import datetime, timezone
    from app.models.subscription import Subscription, PointTransaction

    # Get org member IDs
    member_result = await db.execute(
        select(OrgMember.user_id, OrgMember.role)
        .where(OrgMember.org_id == org_id, OrgMember.status == "active")
    )
    members = member_result.all()
    member_ids = [m.user_id for m in members]
    role_map = {m.user_id: m.role for m in members}

    if not member_ids:
        return []

    # Get names
    names_result = await db.execute(
        select(User.id, User.name, User.email).where(User.id.in_(member_ids))
    )
    name_map = {r.id: r.name or r.email or "Unknown" for r in names_result.all()}

    # Get org subscription ID and billing period
    sub_result = await db.execute(
        select(Subscription.id, Subscription.current_period_start)
        .where(Subscription.org_id == org_id, Subscription.status.in_(["active", "trialing"]))
        .limit(1)
    )
    sub_row = sub_result.first()
    if not sub_row:
        # No org subscription — return members with 0 usage
        return [
            {"user_id": str(uid), "name": name_map.get(uid, "Unknown"), "role": role_map.get(uid, "student"), "points_used": 0, "last_active": None}
            for uid in member_ids
        ]

    org_sub_id = sub_row.id
    if sub_row.current_period_start:
        period_start = sub_row.current_period_start
    else:
        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # Sum positive point transactions per member — only from the ORG subscription
    usage_result = await db.execute(
        select(
            PointTransaction.user_id,
            func.coalesce(func.sum(PointTransaction.points_used), 0).label("total"),
            func.max(PointTransaction.created_at).label("last_active"),
        )
        .where(
            PointTransaction.subscription_id == org_sub_id,
            PointTransaction.user_id.in_(member_ids),
            PointTransaction.points_used > 0,
            PointTransaction.created_at >= period_start,
        )
        .group_by(PointTransaction.user_id)
    )
    usage_map = {r.user_id: {"total": r.total, "last_active": r.last_active} for r in usage_result.all()}

    return [
        {
            "user_id": str(uid),
            "name": name_map.get(uid, "Unknown"),
            "role": role_map.get(uid, "student"),
            "points_used": usage_map.get(uid, {}).get("total", 0),
            "last_active": usage_map.get(uid, {}).get("last_active"),
        }
        for uid in member_ids
    ]


@router.get("/user/progress")
async def get_user_progress(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """Individual user's personal progress summary."""
    from app.models.content import Ebook, MindMap, UserLibraryItem
    from app.models.ai import AiChat

    parsed_oid = _parse_org_id(org_id)

    base_attempt_q = (
        select(AssessmentAttempt.id)
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
        )
    )
    if org_id is not None:
        base_attempt_q = base_attempt_q.where(
            PracticeAssessment.org_id == parsed_oid if parsed_oid else PracticeAssessment.org_id.is_(None)
        )

    assessments_result = await db.execute(select(func.count()).select_from(base_attempt_q.subquery()))
    total_assessments = assessments_result.scalar_one()

    avg_q = (
        select(func.avg(AssessmentAttempt.percentage))
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
        )
    )
    if org_id is not None:
        avg_q = avg_q.where(
            PracticeAssessment.org_id == parsed_oid if parsed_oid else PracticeAssessment.org_id.is_(None)
        )
    avg_score_result = await db.execute(avg_q)
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
async def get_monthly_comparison(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """This-month vs last-month counts for assessments, documents, chats, and avg score."""
    from datetime import datetime, timezone, timedelta
    from app.models.content import UserLibraryItem
    from app.models.ai import AiChat

    now = datetime.now(timezone.utc)
    first_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    first_last = (first_this - timedelta(days=1)).replace(day=1)
    parsed_oid = _parse_org_id(org_id)

    async def count_rows(model, date_col, start, end):
        r = await db.execute(
            select(func.count(model.id))
            .where(model.user_id == current_user.id, date_col >= start, date_col < end)
        )
        return r.scalar_one()

    async def count_attempts(start, end):
        q = (
            select(func.count(AssessmentAttempt.id))
            .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
            .where(
                AssessmentAttempt.user_id == current_user.id,
                AssessmentAttempt.submitted_at >= start,
                AssessmentAttempt.submitted_at < end,
            )
        )
        if org_id is not None:
            q = q.where(
                PracticeAssessment.org_id == parsed_oid if parsed_oid else PracticeAssessment.org_id.is_(None)
            )
        return (await db.execute(q)).scalar_one()

    async def avg_pct(start, end):
        q = (
            select(func.avg(AssessmentAttempt.percentage))
            .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
            .where(
                AssessmentAttempt.user_id == current_user.id,
                AssessmentAttempt.status == "evaluated",
                AssessmentAttempt.submitted_at >= start,
                AssessmentAttempt.submitted_at < end,
            )
        )
        if org_id is not None:
            q = q.where(
                PracticeAssessment.org_id == parsed_oid if parsed_oid else PracticeAssessment.org_id.is_(None)
            )
        return round((await db.execute(q)).scalar_one() or 0, 1)

    # Count graded class submissions for org workspace
    async def count_submissions(start, end):
        if not parsed_oid:
            return 0
        try:
            r = await db.execute(
                select(func.count(Submission.id))
                .join(Assignment, Submission.assignment_id == Assignment.id)
                .join(Class, Assignment.class_id == Class.id)
                .where(
                    Submission.student_id == current_user.id,
                    Submission.status.in_(["graded", "returned"]),
                    Class.org_id == parsed_oid,
                    Submission.submitted_at >= start,
                    Submission.submitted_at < end,
                )
            )
            return r.scalar_one()
        except Exception:
            return 0

    this_assessments = await count_attempts(first_this, now)
    last_assessments = await count_attempts(first_last, first_this)
    this_subs = await count_submissions(first_this, now)
    last_subs = await count_submissions(first_last, first_this)

    return {
        "thisMonth": {
            "assessments": this_assessments + this_subs,
            "documents":   await count_rows(UserLibraryItem, UserLibraryItem.created_at, first_this, now),
            "chats":       await count_rows(AiChat, AiChat.created_at, first_this, now),
            "avgScore":    await avg_pct(first_this, now),
        },
        "lastMonth": {
            "assessments": last_assessments + last_subs,
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
    org_id: str | None = Query(None),
):
    """Last N assessed scores with real submission dates, sorted ascending."""
    parsed_oid = _parse_org_id(org_id)

    q = (
        select(AssessmentAttempt, PracticeAssessment.subject)
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
            AssessmentAttempt.submitted_at.isnot(None),
        )
    )
    if org_id is not None:
        q = q.where(
            PracticeAssessment.org_id == parsed_oid if parsed_oid else PracticeAssessment.org_id.is_(None)
        )
    q = q.order_by(AssessmentAttempt.submitted_at.desc()).limit(limit)
    rows = (await db.execute(q)).all()

    entries = [
        {
            "date":    attempt.submitted_at.strftime("%b %d"),
            "score":   round(attempt.percentage or 0, 1),
            "subject": subject or "General",
            "source":  "practice",
            "_sort":   attempt.submitted_at,
        }
        for attempt, subject in rows
    ]

    # Include graded class submissions for org workspace
    if parsed_oid:
        try:
            sub_q = (
                select(Submission, Assignment, Class)
                .join(Assignment, Submission.assignment_id == Assignment.id)
                .join(Class, Assignment.class_id == Class.id)
                .where(
                    Submission.student_id == current_user.id,
                    Submission.status.in_(["graded", "returned"]),
                    Class.org_id == parsed_oid,
                    Submission.submitted_at.isnot(None),
                )
                .order_by(Submission.submitted_at.desc())
                .limit(limit)
            )
            sub_rows = (await db.execute(sub_q)).all()
            for sub, assign, cls in sub_rows:
                grade_data = sub.grade or {}
                total = grade_data.get("totalScore", 0)
                max_s = grade_data.get("maxScore", assign.points or 1)
                pct = round((total / max_s * 100) if max_s else 0, 1)
                entries.append({
                    "date":    sub.submitted_at.strftime("%b %d"),
                    "score":   pct,
                    "subject": cls.subject or "General",
                    "source":  "assignment",
                    "_sort":   sub.submitted_at,
                })
        except Exception:
            pass

    # Sort by date ascending, take last N, remove internal sort key
    entries.sort(key=lambda e: e["_sort"])
    entries = entries[-limit:]
    for e in entries:
        del e["_sort"]

    return entries


@router.get("/user/study-time")
async def get_study_time(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """AI interactions + assessments grouped by subject — proxy for study activity distribution."""
    from app.models.ai import AiInteractionHistory

    parsed_oid = _parse_org_id(org_id)

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

    attempt_q = (
        select(PracticeAssessment.subject, func.count(AssessmentAttempt.id))
        .join(AssessmentAttempt, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(AssessmentAttempt.user_id == current_user.id)
        .group_by(PracticeAssessment.subject)
    )
    if org_id is not None:
        attempt_q = attempt_q.where(
            PracticeAssessment.org_id == parsed_oid if parsed_oid else PracticeAssessment.org_id.is_(None)
        )
    attempt_rows = (await db.execute(attempt_q)).all()
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
    org_id: str | None = Query(None),
):
    """Daily activity count (AI chats + interactions + assessment attempts) for the last N days."""
    from datetime import datetime, timezone, timedelta, date
    from app.models.ai import AiInteractionHistory, AiChat

    since = datetime.now(timezone.utc) - timedelta(days=days)
    parsed_oid = _parse_org_id(org_id)

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

    attempt_q = (
        select(
            func.date(AssessmentAttempt.submitted_at).label("day"),
            func.count(AssessmentAttempt.id).label("cnt"),
        )
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.submitted_at >= since,
        )
        .group_by(func.date(AssessmentAttempt.submitted_at))
    )
    if org_id is not None:
        attempt_q = attempt_q.where(
            PracticeAssessment.org_id == parsed_oid if parsed_oid else PracticeAssessment.org_id.is_(None)
        )
    attempt_rows = (await db.execute(attempt_q)).all()

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
async def get_recent_activity(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
    limit: int = Query(5, ge=1, le=50),
):
    """Recent user actions merged across assessments, library, chats, ebooks,
    submissions, and assignments.

    When the caller is an org admin and org_id is provided, returns org-wide
    activity from all members instead of only the caller's own activity.
    Includes the acting user's name for org-wide views.
    """
    from datetime import datetime, timezone
    from app.models.content import UserLibraryItem, Ebook
    from app.models.ai import AiChat

    parsed_oid = _parse_org_id(org_id)
    events: list[dict] = []

    # Determine if the current user is an org admin for org-wide activity
    is_org_admin = False
    org_member_ids: list | None = None
    if parsed_oid:
        membership = (await db.execute(
            select(OrgMember).where(
                OrgMember.org_id == parsed_oid,
                OrgMember.user_id == current_user.id,
                OrgMember.role == "org_admin",
            )
        )).scalar_one_or_none()
        if membership:
            is_org_admin = True
            org_member_ids = [
                row[0] for row in (await db.execute(
                    select(OrgMember.user_id).where(
                        OrgMember.org_id == parsed_oid,
                        OrgMember.status == "active",
                    )
                )).all()
            ]

    show_user = is_org_admin and bool(org_member_ids)

    # Helper: user filter — all org members for admins, else just the caller
    def _user_filter(user_col):
        if show_user:
            return user_col.in_(org_member_ids)
        return user_col == current_user.id

    # Pre-load user names for org-wide view
    user_names: dict = {}
    if show_user:
        name_rows = (await db.execute(
            select(User.id, User.name).where(User.id.in_(org_member_ids))
        )).all()
        user_names = {uid: name for uid, name in name_rows}

    def _uname(uid) -> str:
        if not show_user:
            return ""
        return user_names.get(uid, "Member")

    # --- Assessments ---
    attempt_q = (
        select(AssessmentAttempt, PracticeAssessment.title.label("ptitle"))
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            _user_filter(AssessmentAttempt.user_id),
            AssessmentAttempt.submitted_at.isnot(None),
        )
    )
    if parsed_oid:
        attempt_q = attempt_q.where(PracticeAssessment.org_id == parsed_oid)
    elif org_id is not None:
        attempt_q = attempt_q.where(PracticeAssessment.org_id.is_(None))
    attempt_q = attempt_q.order_by(AssessmentAttempt.submitted_at.desc()).limit(limit)
    rows = (await db.execute(attempt_q)).all()
    for attempt, title in rows:
        uname = _uname(attempt.user_id)
        label = f"{uname} completed {title or 'Assessment'}" if uname else f"Completed: {title or 'Assessment'}"
        events.append({"type": "assessment", "label": label, "icon": "📝",
                       "ts": attempt.submitted_at, "user_name": uname})

    # --- Library uploads ---
    lib_q = select(UserLibraryItem).where(_user_filter(UserLibraryItem.user_id))
    if parsed_oid and not show_user:
        lib_q = lib_q.where(UserLibraryItem.org_id == parsed_oid)
    elif not parsed_oid and org_id is not None:
        lib_q = lib_q.where(UserLibraryItem.org_id.is_(None))
    lib_rows = (await db.execute(
        lib_q.order_by(UserLibraryItem.created_at.desc()).limit(limit)
    )).scalars().all()
    for item in lib_rows:
        uname = _uname(item.user_id)
        label = f"{uname} uploaded {item.title or 'Document'}" if uname else f"Uploaded: {item.title or 'Document'}"
        events.append({"type": "document", "label": label, "icon": "📄",
                       "ts": item.created_at, "user_name": uname})

    # --- AI Chats ---
    chat_q = select(AiChat).where(_user_filter(AiChat.user_id))
    if parsed_oid and not show_user:
        # Personal view within an org — filter by org_id
        chat_q = chat_q.where(AiChat.org_id == parsed_oid)
    elif not parsed_oid and org_id is not None:
        chat_q = chat_q.where(AiChat.org_id.is_(None))
    # For org admin view, _user_filter already restricts to org members
    chat_rows = (await db.execute(
        chat_q.order_by(AiChat.created_at.desc()).limit(limit)
    )).scalars().all()
    for chat in chat_rows:
        uname = _uname(chat.user_id)
        label = f"{uname} started AI Chat: {chat.title or 'New Chat'}" if uname else f"AI Chat: {chat.title or 'New Chat'}"
        events.append({"type": "chat", "label": label, "icon": "💬",
                       "ts": chat.created_at, "user_name": uname})

    # --- eBooks ---
    ebook_q = select(Ebook).where(_user_filter(Ebook.user_id))
    if parsed_oid and not show_user:
        ebook_q = ebook_q.where(Ebook.org_id == parsed_oid)
    elif not parsed_oid and org_id is not None:
        ebook_q = ebook_q.where(Ebook.org_id.is_(None))
    ebook_rows = (await db.execute(
        ebook_q.order_by(Ebook.created_at.desc()).limit(limit)
    )).scalars().all()
    for ebook in ebook_rows:
        uname = _uname(ebook.user_id)
        label = f"{uname} created eBook: {ebook.title or 'eBook'}" if uname else f"Created eBook: {ebook.title or 'eBook'}"
        events.append({"type": "ebook", "label": label, "icon": "📚",
                       "ts": ebook.created_at, "user_name": uname})

    # --- Submissions (student submitted assignment work) ---
    if parsed_oid:
        sub_q = (
            select(Submission, Assignment.title.label("atitle"))
            .join(Assignment, Submission.assignment_id == Assignment.id)
            .join(Class, Assignment.class_id == Class.id)
            .where(
                Class.org_id == parsed_oid,
                _user_filter(Submission.student_id),
                Submission.submitted_at.isnot(None),
            )
            .order_by(Submission.submitted_at.desc()).limit(limit)
        )
        sub_rows = (await db.execute(sub_q)).all()
        for sub, atitle in sub_rows:
            uname = _uname(sub.student_id)
            label = f"{uname} submitted {atitle or 'Assignment'}" if uname else f"Submitted: {atitle or 'Assignment'}"
            events.append({"type": "submission", "label": label, "icon": "📥",
                           "ts": sub.submitted_at, "user_name": uname})

        # --- Graded submissions ---
        graded_q = (
            select(Submission, Assignment.title.label("atitle"))
            .join(Assignment, Submission.assignment_id == Assignment.id)
            .join(Class, Assignment.class_id == Class.id)
            .where(
                Class.org_id == parsed_oid,
                Submission.graded_at.isnot(None),
                Submission.status == "graded",
            )
        )
        if show_user:
            graded_q = graded_q.where(Submission.graded_by.in_(org_member_ids))
        else:
            graded_q = graded_q.where(Submission.graded_by == current_user.id)
        graded_q = graded_q.order_by(Submission.graded_at.desc()).limit(limit)
        graded_rows = (await db.execute(graded_q)).all()
        for sub, atitle in graded_rows:
            uname = _uname(sub.graded_by) if sub.graded_by else ""
            label = f"{uname} graded {atitle or 'Assignment'}" if uname else f"Graded: {atitle or 'Assignment'}"
            events.append({"type": "grading", "label": label, "icon": "✅",
                           "ts": sub.graded_at, "user_name": uname})

        # --- Assignments posted ---
        assign_q = (
            select(Assignment, Class.name.label("cname"))
            .join(Class, Assignment.class_id == Class.id)
            .where(
                Class.org_id == parsed_oid,
                Assignment.status == "published",
            )
            .order_by(Assignment.created_at.desc()).limit(limit)
        )
        assign_rows = (await db.execute(assign_q)).all()
        for asgn, cname in assign_rows:
            # Assignment doesn't have created_by; use class teacher
            uname = ""
            events.append({"type": "assignment", "label": f"Assignment posted: {asgn.title} ({cname})",
                           "icon": "📋", "ts": asgn.created_at, "user_name": uname})

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
        {"type": e["type"], "label": e["label"], "icon": e["icon"],
         "time": human_time(e["ts"]), "user_name": e.get("user_name", "")}
        for e in events[:limit]
    ]


# ── Study Time Detailed Endpoints ──────────────────────────────────────────


@router.post("/study-time/backfill")
async def trigger_study_time_backfill(
    current_user: CurrentUser,
    db: DBSession,
    days: int = Query(90, le=365),
):
    """Backfill study_time_daily from historical AI interactions and assessment attempts.
    Run once after deploying the study time feature to populate existing data."""
    from app.services.study_time_aggregator import backfill_study_time
    count = await backfill_study_time(db, days_back=days)
    await db.commit()
    return {"status": "ok", "user_days_processed": count, "days_back": days}


@router.get("/user/study-time-detailed")
async def get_study_time_detailed(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
    period: str = Query("30d", pattern="^(7d|30d|90d)$"),
):
    """Detailed study time analytics for the current user."""
    from datetime import datetime, timezone, timedelta, date
    from app.models.study_time import StudyTimeDaily
    from app.services.study_time_aggregator import compute_realtime_study_minutes

    days = {"7d": 7, "30d": 30, "90d": 90}[period]
    today = date.today()
    start_date = today - timedelta(days=days)
    prev_start = start_date - timedelta(days=days)
    parsed_oid = _parse_org_id(org_id)

    # Base filter
    def _base_filter():
        filters = [
            StudyTimeDaily.user_id == current_user.id,
            StudyTimeDaily.date >= start_date,
            StudyTimeDaily.date <= today,
        ]
        if parsed_oid:
            filters.append(StudyTimeDaily.org_id == parsed_oid)
        return filters

    # Current period totals from pre-aggregated data
    rows = (await db.execute(
        select(StudyTimeDaily).where(*_base_filter())
    )).scalars().all()

    # Add today's real-time data (not yet aggregated)
    today_realtime = await compute_realtime_study_minutes(current_user.id, today, db)
    today_stored = sum(r.total_minutes for r in rows if r.date == today)
    # Use the larger of stored vs realtime for today (avoid double counting)
    today_extra = max(0, today_realtime["total_minutes"] - today_stored)

    total_minutes = sum(r.total_minutes for r in rows) + today_extra
    daily_average = round(total_minutes / max(days, 1))

    # Previous period for trend
    prev_filters = [
        StudyTimeDaily.user_id == current_user.id,
        StudyTimeDaily.date >= prev_start,
        StudyTimeDaily.date < start_date,
    ]
    if parsed_oid:
        prev_filters.append(StudyTimeDaily.org_id == parsed_oid)

    prev_total = (await db.execute(
        select(func.coalesce(func.sum(StudyTimeDaily.total_minutes), 0)).where(*prev_filters)
    )).scalar_one()
    trend_percent = round(((total_minutes - prev_total) / max(prev_total, 1)) * 100) if prev_total > 0 else 0

    # By date
    date_map: dict[str, int] = {}
    for r in rows:
        d = str(r.date)
        date_map[d] = date_map.get(d, 0) + r.total_minutes
    # Override today with realtime if higher
    today_str = str(today)
    if today_realtime["total_minutes"] > date_map.get(today_str, 0):
        date_map[today_str] = today_realtime["total_minutes"]
    by_date = [
        {"date": str(today - timedelta(days=i)), "minutes": date_map.get(str(today - timedelta(days=i)), 0)}
        for i in range(days - 1, -1, -1)
    ]

    # By subject (from stored rows, excluding today if we have realtime)
    subject_map: dict[str, int] = {}
    for r in rows:
        if r.date == today and today_extra > 0:
            continue  # skip stored today data, use realtime instead
        subject_map[r.subject] = subject_map.get(r.subject, 0) + r.total_minutes
    # Add today's realtime subjects
    for subj, mins in today_realtime.get("subjects", {}).items():
        subject_map[subj] = subject_map.get(subj, 0) + mins
    by_subject = sorted(
        [{"subject": s, "minutes": m} for s, m in subject_map.items()],
        key=lambda x: -x["minutes"],
    )[:10]

    # Peak hours
    hour_map: dict[int, int] = {}
    for r in rows:
        if r.peak_hour is not None:
            hour_map[r.peak_hour] = hour_map.get(r.peak_hour, 0) + r.total_minutes
    peak_hours = sorted(
        [{"hour": h, "minutes": m} for h, m in hour_map.items()],
        key=lambda x: -x["minutes"],
    )[:24]

    # Consistency score: % of days in period with >= 15 min study
    active_dates = {str(r.date) for r in rows if r.total_minutes >= 15}
    if today_realtime["total_minutes"] >= 15:
        active_dates.add(today_str)
    consistency_score = round((len(active_dates) / max(days, 1)) * 100)

    # Class average (if in org context)
    class_average_minutes = None
    if parsed_oid:
        org_total = (await db.execute(
            select(
                func.coalesce(func.sum(StudyTimeDaily.total_minutes), 0),
                func.count(func.distinct(StudyTimeDaily.user_id)),
            ).where(
                StudyTimeDaily.org_id == parsed_oid,
                StudyTimeDaily.date >= start_date,
                StudyTimeDaily.date <= today,
            )
        )).first()
        if org_total and org_total[1] > 0:
            class_average_minutes = round(org_total[0] / org_total[1] / max(days, 1))

    return {
        "total_minutes": total_minutes,
        "daily_average": daily_average,
        "trend_percent": trend_percent,
        "by_date": by_date,
        "by_subject": by_subject,
        "peak_hours": peak_hours,
        "consistency_score": consistency_score,
        "class_average_minutes": class_average_minutes,
    }


@router.get("/org/{org_id}/study-time")
async def get_org_study_time(
    org_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
    period: str = Query("30d", pattern="^(7d|30d|90d)$"),
    class_id: uuid.UUID | None = Query(None),
):
    """Organization-wide study time analytics for admins/teachers."""
    from datetime import datetime, timezone, timedelta, date
    from app.models.study_time import StudyTimeDaily

    days = {"7d": 7, "30d": 30, "90d": 90}[period]
    today = date.today()
    start_date = today - timedelta(days=days)

    # Get org member IDs (optionally filtered by class)
    if class_id:
        member_ids_result = await db.execute(
            select(ClassStudent.student_id).where(ClassStudent.class_id == class_id)
        )
    else:
        member_ids_result = await db.execute(
            select(OrgMember.user_id).where(OrgMember.org_id == org_id, OrgMember.status == "active", OrgMember.role == "student")
        )
    member_ids = [r[0] for r in member_ids_result.all()]

    if not member_ids:
        return {
            "org_total_minutes": 0, "avg_per_student_daily": 0,
            "active_students": 0, "by_date": [], "by_subject": [],
            "top_students": [], "at_risk_students": [],
        }

    # Fetch all study time rows for these members in the period
    rows = (await db.execute(
        select(StudyTimeDaily).where(
            StudyTimeDaily.user_id.in_(member_ids),
            StudyTimeDaily.org_id == org_id,
            StudyTimeDaily.date >= start_date,
            StudyTimeDaily.date <= today,
        )
    )).scalars().all()

    org_total = sum(r.total_minutes for r in rows)
    active_user_ids = {r.user_id for r in rows if r.total_minutes > 0}
    active_students = len(active_user_ids)
    avg_per_student_daily = round(org_total / max(len(member_ids), 1) / max(days, 1))

    # By date
    date_map: dict[str, int] = {}
    date_active: dict[str, set] = {}
    for r in rows:
        d = str(r.date)
        date_map[d] = date_map.get(d, 0) + r.total_minutes
        date_active.setdefault(d, set()).add(r.user_id)
    by_date = [
        {
            "date": str(today - timedelta(days=i)),
            "total_minutes": date_map.get(str(today - timedelta(days=i)), 0),
            "active_students": len(date_active.get(str(today - timedelta(days=i)), set())),
        }
        for i in range(days - 1, -1, -1)
    ]

    # By subject
    subject_map: dict[str, int] = {}
    for r in rows:
        subject_map[r.subject] = subject_map.get(r.subject, 0) + r.total_minutes
    by_subject = sorted(
        [{"subject": s, "total_minutes": m} for s, m in subject_map.items()],
        key=lambda x: -x["total_minutes"],
    )[:10]

    # Per-student totals
    student_totals: dict[uuid.UUID, int] = {}
    for r in rows:
        student_totals[r.user_id] = student_totals.get(r.user_id, 0) + r.total_minutes

    # Get names
    name_result = await db.execute(
        select(User.id, User.name, User.email).where(User.id.in_(member_ids))
    )
    name_map = {r.id: r.name or r.email or "Unknown" for r in name_result.all()}

    # Top students
    top_students = sorted(
        [
            {"user_id": str(uid), "name": name_map.get(uid, "Unknown"), "total_minutes": mins}
            for uid, mins in student_totals.items()
        ],
        key=lambda x: -x["total_minutes"],
    )[:10]

    # At-risk students: < 15 min/day average or no activity in last 3 days
    three_days_ago = today - timedelta(days=3)
    recent_active = set()
    for r in rows:
        if r.date >= three_days_ago:
            recent_active.add(r.user_id)

    at_risk_students = []
    for uid in member_ids:
        total = student_totals.get(uid, 0)
        daily_avg = round(total / max(days, 1))
        inactive_recently = uid not in recent_active

        if daily_avg < 15 or inactive_recently:
            days_inactive = 0
            if inactive_recently:
                # Calculate actual days inactive
                user_dates = sorted([r.date for r in rows if r.user_id == uid], reverse=True)
                if user_dates:
                    days_inactive = (today - user_dates[0]).days
                else:
                    days_inactive = days  # no activity in entire period

            at_risk_students.append({
                "user_id": str(uid),
                "name": name_map.get(uid, "Unknown"),
                "total_minutes": total,
                "daily_avg": daily_avg,
                "days_inactive": days_inactive,
            })

    at_risk_students.sort(key=lambda x: x["total_minutes"])

    return {
        "org_total_minutes": org_total,
        "avg_per_student_daily": avg_per_student_daily,
        "active_students": active_students,
        "by_date": by_date,
        "by_subject": by_subject,
        "top_students": top_students,
        "at_risk_students": at_risk_students[:20],
    }


@router.get("/org/{org_id}/student/{student_id}/study-time")
async def get_student_study_time_for_admin(
    org_id: uuid.UUID,
    student_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
    period: str = Query("30d", pattern="^(7d|30d|90d)$"),
):
    """Individual student study time — accessed by teacher or org admin."""
    from datetime import datetime, timezone, timedelta, date
    from app.models.study_time import StudyTimeDaily

    days = {"7d": 7, "30d": 30, "90d": 90}[period]
    today = date.today()
    start_date = today - timedelta(days=days)
    prev_start = start_date - timedelta(days=days)

    rows = (await db.execute(
        select(StudyTimeDaily).where(
            StudyTimeDaily.user_id == student_id,
            StudyTimeDaily.org_id == org_id,
            StudyTimeDaily.date >= start_date,
            StudyTimeDaily.date <= today,
        )
    )).scalars().all()

    total_minutes = sum(r.total_minutes for r in rows)
    daily_average = round(total_minutes / max(days, 1))

    # Trend
    prev_total = (await db.execute(
        select(func.coalesce(func.sum(StudyTimeDaily.total_minutes), 0)).where(
            StudyTimeDaily.user_id == student_id,
            StudyTimeDaily.org_id == org_id,
            StudyTimeDaily.date >= prev_start,
            StudyTimeDaily.date < start_date,
        )
    )).scalar_one()
    trend_percent = round(((total_minutes - prev_total) / max(prev_total, 1)) * 100) if prev_total > 0 else 0

    # By date
    date_map: dict[str, int] = {}
    for r in rows:
        d = str(r.date)
        date_map[d] = date_map.get(d, 0) + r.total_minutes
    by_date = [
        {"date": str(today - timedelta(days=i)), "minutes": date_map.get(str(today - timedelta(days=i)), 0)}
        for i in range(days - 1, -1, -1)
    ]

    # By subject
    subject_map: dict[str, int] = {}
    for r in rows:
        subject_map[r.subject] = subject_map.get(r.subject, 0) + r.total_minutes
    by_subject = sorted(
        [{"subject": s, "minutes": m} for s, m in subject_map.items()],
        key=lambda x: -x["minutes"],
    )[:10]

    # Peak hours
    hour_map: dict[int, int] = {}
    for r in rows:
        if r.peak_hour is not None:
            hour_map[r.peak_hour] = hour_map.get(r.peak_hour, 0) + r.total_minutes
    peak_hours = sorted(
        [{"hour": h, "minutes": m} for h, m in hour_map.items()],
        key=lambda x: -x["minutes"],
    )[:24]

    # Consistency
    active_days = len({str(r.date) for r in rows if r.total_minutes >= 15})
    consistency_score = round((active_days / max(days, 1)) * 100)

    # Student name
    student = (await db.execute(
        select(User.name, User.email).where(User.id == student_id)
    )).first()
    student_name = (student.name or student.email or "Unknown") if student else "Unknown"

    return {
        "student_name": student_name,
        "total_minutes": total_minutes,
        "daily_average": daily_average,
        "trend_percent": trend_percent,
        "by_date": by_date,
        "by_subject": by_subject,
        "peak_hours": peak_hours,
        "consistency_score": consistency_score,
    }
