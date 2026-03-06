"""
Seed script: creates 3 test users (Free / Plus / Pro) with matching subscriptions.
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
from app.core.security import hash_password


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


async def seed():
    async with AsyncSessionLocal() as db:
        now = datetime.now(timezone.utc)
        period_end = now + timedelta(days=30)

        for data in TEST_USERS:
            # Check if user already exists
            existing = await db.scalar(
                select(User).where(User.email == data["email"])
            )
            if existing:
                print(f"  [skip] user already exists: {data['email']}")
                # Ensure subscription exists and is updated
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
                continue

            # Create user
            user_id = uuid.uuid4()
            user = User(
                id=user_id,
                email=data["email"],
                hashed_password=hash_password(data["password"]),
                name=data["name"],
                language="en",
                is_active=True,
            )
            db.add(user)
            await db.flush()

            # Add normal_user role
            db.add(UserRole(
                id=uuid.uuid4(),
                user_id=user_id,
                role="normal_user",
            ))

            # Create subscription
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

        await db.commit()
        print("\nDone. Test users ready:")
        print("  freeuser@genverse.dev  / Test@123  (Free plan, 100 pts)")
        print("  plususer@genverse.dev  / Test@123  (Plus plan, 800 pts)")
        print("  prouser@genverse.dev   / Test@123  (Pro plan, 2000 pts)")


if __name__ == "__main__":
    asyncio.run(seed())
