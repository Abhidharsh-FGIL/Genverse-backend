"""
Seed script: creates a test organization "GenVerse Demo School"
with an admin, a teacher, and a student account.

Run from the backend root:
    python -m scripts.seed_test_org
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

ORG_NAME = "GenVerse Demo School"
PASSWORD = "Test@123"

USERS = [
    {"email": "demoadmin@genverse.dev",   "name": "Demo Admin",   "org_role": "org_admin", "app_role": "normal_user"},
    {"email": "demoteacher@genverse.dev", "name": "Demo Teacher", "org_role": "teacher",   "app_role": "normal_user"},
    {"email": "demostudent@genverse.dev", "name": "Demo Student", "org_role": "student",   "app_role": "normal_user"},
]


async def seed():
    async with AsyncSessionLocal() as db:
        now = datetime.now(timezone.utc)
        period_end = now + timedelta(days=30)

        # ── Check if org already exists ──
        existing_org = await db.scalar(
            select(Organization).where(Organization.name == ORG_NAME)
        )
        if existing_org:
            print(f"Organization '{ORG_NAME}' already exists — skipping org creation.")
            org_id = existing_org.id
        else:
            org_id = uuid.uuid4()
            db.add(Organization(
                id=org_id,
                name=ORG_NAME,
                product_type="genverse",
                has_genverse=True,
                has_evaluation=True,
            ))
            await db.flush()
            print(f"[insert] Organization: {ORG_NAME}")

        # ── Create users and add to org ──
        admin_user_id = None
        for u in USERS:
            existing = await db.scalar(
                select(User).where(User.email == u["email"])
            )
            if existing:
                user_id = existing.id
                print(f"  [skip] user already exists: {u['email']}")
            else:
                user_id = uuid.uuid4()
                db.add(User(
                    id=user_id,
                    email=u["email"],
                    hashed_password=hash_password(PASSWORD),
                    name=u["name"],
                    language="en",
                    is_active=True,
                ))
                await db.flush()

                db.add(UserRole(id=uuid.uuid4(), user_id=user_id, role=u["app_role"]))

                # Individual free subscription
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
                ))
                print(f"  [insert] user: {u['email']} ({u['name']})")

            if u["org_role"] == "org_admin":
                admin_user_id = user_id

            # Add to org if not already a member
            existing_member = await db.scalar(
                select(OrgMember).where(
                    OrgMember.org_id == org_id,
                    OrgMember.user_id == user_id,
                )
            )
            if existing_member:
                print(f"  [skip] already org member: {u['email']}")
            else:
                db.add(OrgMember(
                    id=uuid.uuid4(),
                    org_id=org_id,
                    user_id=user_id,
                    role=u["org_role"],
                    status="active",
                ))
                print(f"  [insert] org member: {u['email']} -> {u['org_role']}")

        # ── Org subscription (org_pro plan) ──
        existing_sub = await db.scalar(
            select(Subscription).where(
                Subscription.org_id == org_id,
                Subscription.workspace_type == "organization",
            )
        )
        if existing_sub:
            existing_sub.plan = "org_pro"
            existing_sub.status = "active"
            existing_sub.points_balance = 20000
            existing_sub.points_monthly_quota = 20000
            existing_sub.storage_limit_mb = 51200
            existing_sub.max_seats = 1000
            print("  [update] org subscription -> org_pro")
        else:
            db.add(Subscription(
                id=uuid.uuid4(),
                user_id=admin_user_id,
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
            ))
            print("  [insert] org subscription -> org_pro (20000 pts, 1000 seats)")

        await db.commit()

        print("\n" + "=" * 55)
        print(f"  Organization: {ORG_NAME}")
        print(f"  Plan: org_pro (20,000 points, 1000 seats)")
        print("=" * 55)
        print(f"  Admin:   demoadmin@genverse.dev   / {PASSWORD}")
        print(f"  Teacher: demoteacher@genverse.dev / {PASSWORD}")
        print(f"  Student: demostudent@genverse.dev / {PASSWORD}")
        print("=" * 55)


if __name__ == "__main__":
    asyncio.run(seed())
