"""
Seed script: creates a GT org with 1 admin, 2 teachers, and 1 student,
all mapped to the org_pro + evaluation plan with 20,000 points.
Run from the project root:
    python -m scripts.seed_gt_users
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


ORG_NAME = "GT Academy"
ORG_SLUG = "gt"
PASSWORD = "Test@123"

ADMIN = {
    "email": "gtadmin@genverse.dev",
    "name": "GT Admin",
    "role": "org_admin",
}

TEACHERS = [
    {"email": "gtteacher1@genverse.dev", "name": "GT Teacher 1"},
    {"email": "gtteacher2@genverse.dev", "name": "GT Teacher 2"},
]

STUDENT = {
    "email": "gtstudent@genverse.dev",
    "name": "GT Student",
}


async def _get_or_create_user(db, email, name, now, period_end):
    """Return existing user_id or create a new user with a free personal subscription."""
    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        print(f"  [skip] user already exists: {email}")
        return existing.id

    user_id = uuid.uuid4()
    db.add(User(
        id=user_id,
        email=email,
        hashed_password=hash_password(PASSWORD),
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
    print(f"  [insert] user: {email} ({name})")
    return user_id


async def _ensure_org_member(db, org_id, user_id, email, role):
    existing = await db.scalar(
        select(OrgMember).where(
            OrgMember.org_id == org_id,
            OrgMember.user_id == user_id,
        )
    )
    if existing:
        if existing.role != role:
            existing.role = role
            existing.status = "active"
            print(f"  [update] org member role -> {role}: {email}")
        else:
            print(f"  [skip] already org member: {email}")
    else:
        db.add(OrgMember(
            id=uuid.uuid4(),
            org_id=org_id,
            user_id=user_id,
            role=role,
            status="active",
        ))
        print(f"  [insert] org member: {email} -> {role}")


async def seed():
    async with AsyncSessionLocal() as db:
        now = datetime.now(timezone.utc)
        period_end = now + timedelta(days=30)

        # ── Find or create the GT org ──
        admin_user = await db.scalar(select(User).where(User.email == ADMIN["email"]))

        org = None
        admin_id = None

        if admin_user:
            admin_id = admin_user.id
            print(f"  [skip] admin already exists: {ADMIN['email']}")

            # Find the org this admin belongs to
            member = await db.scalar(
                select(OrgMember).where(
                    OrgMember.user_id == admin_id,
                    OrgMember.role == "org_admin",
                )
            )
            if member:
                org = await db.scalar(
                    select(Organization).where(Organization.id == member.org_id)
                )

        if org is None:
            # Create admin user
            admin_id = await _get_or_create_user(db, ADMIN["email"], ADMIN["name"], now, period_end)

            # Create org
            org_id = uuid.uuid4()
            org = Organization(
                id=org_id,
                name=ORG_NAME,
                product_type="genverse_evaluation",
                has_genverse=True,
                has_evaluation=True,
            )
            db.add(org)
            await db.flush()
            print(f"  [insert] org: {ORG_NAME}")

        org_id = org.id

        # ── Upsert org subscription (admin, 20,000 pts) ──
        org_sub = await db.scalar(
            select(Subscription).where(
                Subscription.org_id == org_id,
                Subscription.workspace_type == "organization",
            )
        )
        if org_sub:
            org_sub.plan = "org_pro"
            org_sub.status = "active"
            org_sub.points_balance = 20000
            org_sub.points_monthly_quota = 20000
            org_sub.storage_limit_mb = 51200
            org_sub.max_seats = 1000
            org_sub.current_period_start = now
            org_sub.current_period_end = period_end
            print("  [update] org subscription -> org_pro (20,000 pts)")
        else:
            db.add(Subscription(
                id=uuid.uuid4(),
                user_id=admin_id,
                org_id=org_id,
                plan="org_pro",
                status="active",
                workspace_type="organization",
                points_balance=20000,
                points_monthly_quota=20000,
                storage_limit_mb=51200,
                max_seats=1000,
                current_period_start=now,
                current_period_end=period_end,
                auto_renew=False,
            ))
            print("  [insert] org subscription -> org_pro (20,000 pts)")

        # ── Admin org membership ──
        await _ensure_org_member(db, org_id, admin_id, ADMIN["email"], "org_admin")

        # ── Teachers ──
        for t in TEACHERS:
            t_id = await _get_or_create_user(db, t["email"], t["name"], now, period_end)
            await _ensure_org_member(db, org_id, t_id, t["email"], "teacher")

        # ── Student ──
        s_id = await _get_or_create_user(db, STUDENT["email"], STUDENT["name"], now, period_end)
        await _ensure_org_member(db, org_id, s_id, STUDENT["email"], "student")

        await db.commit()

        print("\nDone. GT users ready:")
        print(f"  gtadmin@genverse.dev      / Test@123  (Org Admin, 20,000 pts — org_pro + eval)")
        print(f"  gtteacher1@genverse.dev   / Test@123  (Teacher)")
        print(f"  gtteacher2@genverse.dev   / Test@123  (Teacher)")
        print(f"  gtstudent@genverse.dev    / Test@123  (Student)")


if __name__ == "__main__":
    asyncio.run(seed())
