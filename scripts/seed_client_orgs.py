"""
Seed script: creates two client organizations for testing.

Both organizations are on the Org Pro + Evaluation Hub plan.
Each has: 1 admin, 2 teachers, 5 students.

Run from the project root:
    python -m scripts.seed_client_orgs
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


PASSWORD = "Client@123"

# ── Organization definitions ─────────────────────────────────────────────────

ORGS = [
    {
        "admin_email": "client-org1-admin@genverse.dev",
        "admin_name": "Client Org 1 Admin",
        "org_name": "Client Organization 1",
        "plan": "org_pro",
        "product_type": "genverse_evaluation",
        "has_genverse": True,
        "has_evaluation": True,
        "points_balance": 20000,
        "points_monthly_quota": 20000,
        "storage_limit_mb": 51200,
        "max_seats": 1000,
        "slug": "client-org1",
    },
    {
        "admin_email": "client-org2-admin@genverse.dev",
        "admin_name": "Client Org 2 Admin",
        "org_name": "Client Organization 2",
        "plan": "org_pro",
        "product_type": "genverse_evaluation",
        "has_genverse": True,
        "has_evaluation": True,
        "points_balance": 20000,
        "points_monthly_quota": 20000,
        "storage_limit_mb": 51200,
        "max_seats": 1000,
        "slug": "client-org2",
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


async def _ensure_org_member(db, org_id, email, name, role, now, period_end):
    """Create a user (if needed) and add as org member."""
    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        user_id = existing.id
        print(f"    [skip] user already exists: {email}")
    else:
        user_id = await _create_user(db, email, PASSWORD, name, now, period_end)
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
    slug = data["slug"]

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
        user_id = await _create_user(db, data["admin_email"], PASSWORD, data["admin_name"], now, period_end)
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

    # ── 2 Teachers ──
    for i in range(1, 3):
        await _ensure_org_member(
            db, org_id,
            f"teacher{i}-{slug}@genverse.dev",
            f"Teacher {i} ({data['org_name']})",
            "teacher", now, period_end,
        )

    # ── 5 Students ──
    for i in range(1, 6):
        await _ensure_org_member(
            db, org_id,
            f"student{i}-{slug}@genverse.dev",
            f"Student {i} ({data['org_name']})",
            "student", now, period_end,
        )


async def seed():
    async with AsyncSessionLocal() as db:
        now = datetime.now(timezone.utc)
        period_end = now + timedelta(days=365)

        for data in ORGS:
            print(f"\n── {data['org_name']} ({data['plan']}) ──")
            await _seed_org(db, data, now, period_end)

        await db.commit()

        print("\n")
        print("=" * 80)
        print("  CLIENT ORGANIZATION CREDENTIALS — COPY BELOW FOR CLIENT")
        print("=" * 80)

        for data in ORGS:
            slug = data["slug"]
            pts = data["points_balance"]
            print()
            print(f"  Organization : {data['org_name']}")
            print(f"  Plan         : Org Pro + Evaluation Hub")
            print(f"  Points       : {pts}")
            print(f"  {'─' * 70}")
            print(f"  {'Role':<12} {'Email':<50} {'Password'}")
            print(f"  {'─' * 70}")
            print(f"  {'Admin':<12} {data['admin_email']:<50} {PASSWORD}")
            for i in range(1, 3):
                email = f"teacher{i}-{slug}@genverse.dev"
                print(f"  {'Teacher':<12} {email:<50} {PASSWORD}")
            for i in range(1, 6):
                email = f"student{i}-{slug}@genverse.dev"
                print(f"  {'Student':<12} {email:<50} {PASSWORD}")
            print()

        print("=" * 80)
        print(f"  Total: {len(ORGS)} orgs × (1 admin + 2 teachers + 5 students) = {len(ORGS) * 8} logins")
        print(f"  Default password for all: {PASSWORD}")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(seed())
