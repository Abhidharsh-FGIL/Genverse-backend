"""
Seed script: creates test users (Free / Plus / Pro) with matching subscriptions,
plus organization admins (Org Basic / Org Pro / Org Evaluation) with their orgs.
Run from the project root:
    python -m scripts.seed_test_users
"""
import asyncio
import uuid
import sys
import os
from datetime import datetime, timezone, timedelta

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from app.database import AsyncSessionLocal
from app.models.user import User, UserRole
from app.models.subscription import Subscription
from app.models.organization import Organization, OrgMember
from app.core.security import hash_password


# ── Individual test users ────────────────────────────────────────────────────

TEST_USERS = [
    {
        "email": "freeuser@genverse.dev",
        "password": "Test@123",
        "name": "Free Test User",
        "plan": "free",
        "points_balance": 100,
        "points_monthly_quota": 100,
        "storage_limit_mb": 100,
    },
    {
        "email": "plususer@genverse.dev",
        "password": "Test@123",
        "name": "Plus Test User",
        "plan": "individual_pro",
        "points_balance": 800,
        "points_monthly_quota": 800,
        "storage_limit_mb": 200,
    },
    {
        "email": "prouser@genverse.dev",
        "password": "Test@123",
        "name": "Pro Test User",
        "plan": "individual_power",
        "points_balance": 2000,
        "points_monthly_quota": 2000,
        "storage_limit_mb": 500,
    },
]


# ── Organization admin test users ────────────────────────────────────────────

ORG_ADMIN_USERS = [
    {
        "email": "orgbasic@genverse.dev",
        "password": "Test@123",
        "name": "Org Basic Admin",
        "org_name": "Basic Test School",
        "plan": "org_basic",
        "points_balance": 5000,
        "points_monthly_quota": 5000,
        "storage_limit_mb": 5120,
        "max_seats": 50,
    },
    {
        "email": "orgpro@genverse.dev",
        "password": "Test@123",
        "name": "Org Pro Admin",
        "org_name": "Pro Test Academy",
        "plan": "org_pro",
        "points_balance": 20000,
        "points_monthly_quota": 20000,
        "storage_limit_mb": 51200,
        "max_seats": 1000,
    },
    {
        "email": "orgeval@genverse.dev",
        "password": "Test@123",
        "name": "Org Eval Admin",
        "org_name": "Evaluation Test Institute",
        "plan": "org_evaluation",
        "points_balance": 3000,
        "points_monthly_quota": 3000,
        "storage_limit_mb": 2048,
        "max_seats": 100,
    },
]


async def _seed_individual_user(db, data, now, period_end):
    """Create or update an individual test user + subscription."""
    existing = await db.scalar(
        select(User).where(User.email == data["email"])
    )
    if existing:
        print(f"  [skip] user already exists: {data['email']}")
        sub = await db.scalar(
            select(Subscription).where(
                Subscription.user_id == existing.id,
                Subscription.workspace_type == "individual",
            )
        )
        if sub:
            sub.plan = data["plan"]
            sub.status = "active"
            sub.points_balance = data["points_balance"]
            sub.points_monthly_quota = data["points_monthly_quota"]
            sub.storage_limit_mb = data["storage_limit_mb"]
            print(f"  [update] subscription -> {data['plan']}")
        else:
            db.add(Subscription(
                id=uuid.uuid4(),
                user_id=existing.id,
                org_id=None,
                plan=data["plan"],
                status="active",
                workspace_type="individual",
                points_balance=data["points_balance"],
                points_monthly_quota=data["points_monthly_quota"],
                storage_limit_mb=data["storage_limit_mb"],
                max_seats=None,
                current_period_start=now,
                current_period_end=period_end,
            ))
            print(f"  [insert] subscription -> {data['plan']}")
        return

    user_id = uuid.uuid4()
    db.add(User(
        id=user_id,
        email=data["email"],
        hashed_password=hash_password(data["password"]),
        name=data["name"],
        language="en",
        is_active=True,
    ))
    await db.flush()

    db.add(UserRole(id=uuid.uuid4(), user_id=user_id, role="normal_user"))

    db.add(Subscription(
        id=uuid.uuid4(),
        user_id=user_id,
        org_id=None,
        plan=data["plan"],
        status="active",
        workspace_type="individual",
        points_balance=data["points_balance"],
        points_monthly_quota=data["points_monthly_quota"],
        storage_limit_mb=data["storage_limit_mb"],
        max_seats=None,
        current_period_start=now,
        current_period_end=period_end,
    ))
    print(f"  [insert] user + subscription: {data['email']} ({data['plan']})")


async def _seed_org_admin(db, data, now, period_end):
    """Create or update an org admin user + organization + org subscription."""
    existing = await db.scalar(
        select(User).where(User.email == data["email"])
    )

    if existing:
        user_id = existing.id
        print(f"  [skip] user already exists: {data['email']}")

        # Ensure org exists for this user
        member = await db.scalar(
            select(OrgMember).where(
                OrgMember.user_id == user_id,
                OrgMember.role == "org_admin",
            )
        )
        if member:
            org_id = member.org_id
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
                ))
                print(f"  [insert] org subscription -> {data['plan']}")
            return

        # User exists but no org — create org + membership below
    else:
        # Create user
        user_id = uuid.uuid4()
        db.add(User(
            id=user_id,
            email=data["email"],
            hashed_password=hash_password(data["password"]),
            name=data["name"],
            language="en",
            is_active=True,
        ))
        await db.flush()

        db.add(UserRole(id=uuid.uuid4(), user_id=user_id, role="normal_user"))

        # Also give individual free subscription (personal workspace)
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
        print(f"  [insert] user: {data['email']}")

    # Create organization
    org_id = uuid.uuid4()
    db.add(Organization(
        id=org_id,
        name=data["org_name"],
        product_type="genverse",
        has_genverse=True,
        has_evaluation=False,
    ))
    await db.flush()

    # Add user as org_admin
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
    ))
    print(f"  [insert] org '{data['org_name']}' + admin + subscription: {data['email']} ({data['plan']})")


async def seed():
    async with AsyncSessionLocal() as db:
        now = datetime.now(timezone.utc)
        period_end = now + timedelta(days=30)

        print("── Individual users ──")
        for data in TEST_USERS:
            await _seed_individual_user(db, data, now, period_end)

        print("\n── Organization admins ──")
        for data in ORG_ADMIN_USERS:
            await _seed_org_admin(db, data, now, period_end)

        await db.commit()

        print("\nDone. Test users ready:")
        print("  freeuser@genverse.dev   / Test@123  (Free plan, 100 pts)")
        print("  plususer@genverse.dev   / Test@123  (Plus plan, 800 pts)")
        print("  prouser@genverse.dev    / Test@123  (Pro plan, 2000 pts)")
        print("  orgbasic@genverse.dev   / Test@123  (Org Basic admin, 5000 pts)")
        print("  orgpro@genverse.dev     / Test@123  (Org Pro admin, 20000 pts)")
        print("  orgeval@genverse.dev    / Test@123  (Org Eval admin, 3000 pts)")


if __name__ == "__main__":
    asyncio.run(seed())
