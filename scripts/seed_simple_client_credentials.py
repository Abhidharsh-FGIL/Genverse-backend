"""
Seed script: creates client demo credentials.

Two organizations:
  1. Org Pro (GenVerse only)   – 1 admin, 2 teachers, 5 students
  2. Org Pro + Evaluation Hub  – 1 admin, 2 teachers, 5 students

Run from the project root:
    python -m scripts.seed_simple_client_credentials
"""
import asyncio
import uuid
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from app.database import AsyncSessionLocal
from app.models.user import User, UserRole
from app.models.subscription import Subscription
from app.models.organization import Organization, OrgMember
from app.core.security import hash_password


# -- Organization definitions ------------------------------------------------─

ORGS = [
    {
        "admin_email": "admin1@gmail.com",
        "admin_password": "admin1",
        "admin_name": "Admin 1",
        "org_name": "GenVerse Pro Demo",
        "plan": "org_pro",
        "product_type": "genverse",
        "has_genverse": True,
        "has_evaluation": False,
        "points_balance": 20000,
        "points_monthly_quota": 20000,
        "storage_limit_mb": 51200,
        "max_seats": 1000,
        "teacher_start": 1,   # teacher1, teacher2
        "student_start": 1,   # student1 .. student5
    },
    {
        "admin_email": "admin2@gmail.com",
        "admin_password": "admin2",
        "admin_name": "Admin 2",
        "org_name": "GenVerse Pro+Eval Demo",
        "plan": "org_pro",
        "product_type": "genverse_evaluation",
        "has_genverse": True,
        "has_evaluation": True,
        "points_balance": 20000,
        "points_monthly_quota": 20000,
        "storage_limit_mb": 51200,
        "max_seats": 1000,
        "teacher_start": 3,   # teacher3, teacher4
        "student_start": 6,   # student6 .. student10
    },
]


async def _create_user(db, email, password, name, now, period_end):
    """Create a user with a free personal subscription. Returns user_id."""
    user_id = uuid.uuid4()
    db.add(User(
        id=user_id,
        email=email,
        hashed_password=hash_password(password),
        name=name,
        language="en",
        auth_provider="email",
        onboarding_completed=True,
        xp=0,
        streak=0,
        is_active=True,
    ))
    await db.flush()

    db.add(UserRole(id=uuid.uuid4(), user_id=user_id, role="normal_user"))

    # Personal workspace free subscription
    db.add(Subscription(
        id=uuid.uuid4(),
        user_id=user_id,
        org_id=None,
        plan="free",
        status="active",
        workspace_type="individual",
        points_balance=100,
        points_monthly_quota=100,
        storage_limit_mb=100,
        max_seats=None,
        current_period_start=now,
        current_period_end=period_end,
        auto_renew=False,
    ))
    return user_id


async def _ensure_org_member(db, org_id, email, password, name, role, now, period_end):
    """Create a user (if needed) and add as org member."""
    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        user_id = existing.id
        print(f"    [skip] user already exists: {email}")
    else:
        user_id = await _create_user(db, email, password, name, now, period_end)
        print(f"    [insert] user: {email}")

    # Add to org if not already a member
    existing_member = await db.scalar(
        select(OrgMember).where(
            OrgMember.org_id == org_id,
            OrgMember.user_id == user_id,
        )
    )
    if existing_member:
        print(f"    [skip] already org member: {email}")
    else:
        db.add(OrgMember(
            id=uuid.uuid4(),
            org_id=org_id,
            user_id=user_id,
            role=role,
            status="active",
        ))
        print(f"    [insert] org member: {email} -> {role}")


async def _seed_org(db, data, now, period_end):
    """Create an org with admin, 2 teachers, and 5 students."""
    t_start = data["teacher_start"]
    s_start = data["student_start"]

    # Check if admin already exists
    existing = await db.scalar(select(User).where(User.email == data["admin_email"]))
    if existing:
        user_id = existing.id
        print(f"  [skip] admin already exists: {data['admin_email']}")

        # Find existing org
        member = await db.scalar(
            select(OrgMember).where(
                OrgMember.user_id == user_id,
                OrgMember.role == "org_admin",
            )
        )
        if member:
            org_id = member.org_id

            # Update org flags
            org = await db.scalar(select(Organization).where(Organization.id == org_id))
            if org:
                org.product_type = data["product_type"]
                org.has_genverse = data["has_genverse"]
                org.has_evaluation = data["has_evaluation"]

            # Update org subscription
            sub = await db.scalar(
                select(Subscription).where(
                    Subscription.org_id == org_id,
                    Subscription.workspace_type == "organization",
                )
            )
            if sub:
                sub.plan = data["plan"]
                sub.status = "active"
                sub.points_balance = data["points_balance"]
                sub.points_monthly_quota = data["points_monthly_quota"]
                sub.storage_limit_mb = data["storage_limit_mb"]
                sub.max_seats = data["max_seats"]
                print(f"  [update] org subscription -> {data['plan']}")
        else:
            org_id = None
    else:
        user_id = await _create_user(db, data["admin_email"], data["admin_password"], data["admin_name"], now, period_end)
        print(f"  [insert] admin: {data['admin_email']}")
        org_id = None

    # Create org if needed
    if org_id is None:
        org_id = uuid.uuid4()
        db.add(Organization(
            id=org_id,
            name=data["org_name"],
            product_type=data["product_type"],
            has_genverse=data["has_genverse"],
            has_evaluation=data["has_evaluation"],
        ))
        await db.flush()

        # Add admin membership
        db.add(OrgMember(
            id=uuid.uuid4(),
            org_id=org_id,
            user_id=user_id,
            role="org_admin",
            status="active",
        ))

        # Create org subscription
        db.add(Subscription(
            id=uuid.uuid4(),
            user_id=user_id,
            org_id=org_id,
            plan=data["plan"],
            status="active",
            workspace_type="organization",
            points_balance=data["points_balance"],
            points_monthly_quota=data["points_monthly_quota"],
            storage_limit_mb=data["storage_limit_mb"],
            max_seats=data["max_seats"],
            current_period_start=now,
            current_period_end=period_end,
            auto_renew=False,
        ))
        print(f"  [insert] org '{data['org_name']}' + subscription ({data['plan']})")

    # -- 2 Teachers --
    for i in range(t_start, t_start + 2):
        await _ensure_org_member(
            db, org_id,
            f"teacher{i}@gmail.com",
            f"teacher{i}",
            f"Teacher {i}",
            "teacher", now, period_end,
        )

    # -- 5 Students --
    for i in range(s_start, s_start + 5):
        await _ensure_org_member(
            db, org_id,
            f"student{i}@gmail.com",
            f"student{i}",
            f"Student {i}",
            "student", now, period_end,
        )


async def seed():
    async with AsyncSessionLocal() as db:
        now = datetime.now(timezone.utc)
        period_end = now + timedelta(days=365)  # 1 year for demo

        for data in ORGS:
            print(f"\n-- {data['org_name']} ({data['plan']}) --")
            await _seed_org(db, data, now, period_end)

        await db.commit()

        print("\nDone. Client demo users ready:")

        for data in ORGS:
            t_start = data["teacher_start"]
            s_start = data["student_start"]
            plan_label = "Org Pro" if not data["has_evaluation"] else "Org Pro + Evaluation Hub"
            pts = data["points_balance"]
            print(f"\n-- {data['org_name']} ({plan_label}) --")
            print(f"  {data['admin_email']:<35} / {data['admin_password']:<12} (Admin, {pts} pts)")
            for i in range(t_start, t_start + 2):
                email = f"teacher{i}@gmail.com"
                print(f"  {email:<35} / {'teacher' + str(i):<12} (Teacher)")
            for i in range(s_start, s_start + 5):
                email = f"student{i}@gmail.com"
                print(f"  {email:<35} / {'student' + str(i):<12} (Student)")


if __name__ == "__main__":
    asyncio.run(seed())
