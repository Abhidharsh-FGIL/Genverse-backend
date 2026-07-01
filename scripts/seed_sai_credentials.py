"""
Seed script: creates SAI org credentials — 1 admin, 10 teachers, 20 students.
All email addresses contain "sai".

Run from the project root:
    python -m scripts.seed_sai_credentials
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


ORG = {
    "admin_email": "admin.sai@genverse.dev",
    "admin_password": "Sai@Admin1",
    "admin_name": "SAI Admin",
    "org_name": "SAI Institute",
    "plan": "org_pro",
    "product_type": "genverse",
    "has_genverse": True,
    "has_evaluation": False,
    "points_balance": 20000,
    "points_monthly_quota": 20000,
    "storage_limit_mb": 51200,
    "max_seats": 1000,
}

_TEACHER_NAMES = [
    "Sanjana", "Reeba", "Gowri", "Sowthaman", "Aravindh",
    "Abhi", "Sakthi", "Siva", "Priya", "Nisha",
]

_STUDENT_NAMES = [
    "Arjun", "Karthik", "Deepa", "Meena", "Ramya",
    "Vishnu", "Ajith", "Divya", "Harish", "Kavya",
    "Manoj", "Pooja", "Rahul", "Sneha", "Surya",
    "Vidya", "Anand", "Bhavya", "Dinesh", "Geetha",
]

TEACHERS = [
    {
        "email": f"{name.lower()}@genverse.dev",
        "password": f"{name}@{i}",
        "name": name,
    }
    for i, name in enumerate(_TEACHER_NAMES, start=1)
]

STUDENTS = [
    {
        "email": f"{name.lower()}.student@genverse.dev",
        "password": f"{name}@{i}",
        "name": name,
    }
    for i, name in enumerate(_STUDENT_NAMES, start=1)
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
    """Create a user (if needed) and add them as an org member."""
    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        user_id = existing.id
        print(f"    [skip] user already exists: {email}")
    else:
        user_id = await _create_user(db, email, password, name, now, period_end)
        print(f"    [insert] user: {email}")

    existing_member = await db.scalar(
        select(OrgMember).where(
            OrgMember.org_id == org_id,
            OrgMember.user_id == user_id,
        )
    )
    if existing_member:
        print(f"    [skip] already org member: {email} ({existing_member.role})")
    else:
        db.add(OrgMember(
            id=uuid.uuid4(),
            org_id=org_id,
            user_id=user_id,
            role=role,
            status="active",
        ))
        print(f"    [insert] org member: {email} -> {role}")


async def seed():
    async with AsyncSessionLocal() as db:
        now = datetime.now(timezone.utc)
        period_end = now + timedelta(days=365)

        print(f"\n-- {ORG['org_name']} ({ORG['plan']}) --")

        # Admin
        existing_admin = await db.scalar(select(User).where(User.email == ORG["admin_email"]))
        if existing_admin:
            user_id = existing_admin.id
            print(f"  [skip] admin already exists: {ORG['admin_email']}")

            member = await db.scalar(
                select(OrgMember).where(
                    OrgMember.user_id == user_id,
                    OrgMember.role == "org_admin",
                )
            )
            org_id = member.org_id if member else None

            if org_id:
                org = await db.scalar(select(Organization).where(Organization.id == org_id))
                if org:
                    org.product_type = ORG["product_type"]
                    org.has_genverse = ORG["has_genverse"]
                    org.has_evaluation = ORG["has_evaluation"]

                sub = await db.scalar(
                    select(Subscription).where(
                        Subscription.org_id == org_id,
                        Subscription.workspace_type == "organization",
                    )
                )
                if sub:
                    sub.plan = ORG["plan"]
                    sub.status = "active"
                    sub.points_balance = ORG["points_balance"]
                    sub.points_monthly_quota = ORG["points_monthly_quota"]
                    sub.storage_limit_mb = ORG["storage_limit_mb"]
                    sub.max_seats = ORG["max_seats"]
                    print(f"  [update] org subscription -> {ORG['plan']}")
        else:
            user_id = await _create_user(
                db, ORG["admin_email"], ORG["admin_password"], ORG["admin_name"], now, period_end
            )
            print(f"  [insert] admin: {ORG['admin_email']}")
            org_id = None

        # Create org if it doesn't exist yet
        if not existing_admin or org_id is None:
            org_id = uuid.uuid4()
            db.add(Organization(
                id=org_id,
                name=ORG["org_name"],
                product_type=ORG["product_type"],
                has_genverse=ORG["has_genverse"],
                has_evaluation=ORG["has_evaluation"],
            ))
            await db.flush()

            db.add(OrgMember(
                id=uuid.uuid4(),
                org_id=org_id,
                user_id=user_id,
                role="org_admin",
                status="active",
            ))

            db.add(Subscription(
                id=uuid.uuid4(),
                user_id=user_id,
                org_id=org_id,
                plan=ORG["plan"],
                status="active",
                workspace_type="organization",
                points_balance=ORG["points_balance"],
                points_monthly_quota=ORG["points_monthly_quota"],
                storage_limit_mb=ORG["storage_limit_mb"],
                max_seats=ORG["max_seats"],
                current_period_start=now,
                current_period_end=period_end,
                auto_renew=False,
            ))
            print(f"  [insert] org '{ORG['org_name']}' + subscription ({ORG['plan']})")

        # Teachers
        print("\n  -- Teachers --")
        for t in TEACHERS:
            await _ensure_org_member(db, org_id, t["email"], t["password"], t["name"], "teacher", now, period_end)

        # Students
        print("\n  -- Students --")
        for s in STUDENTS:
            await _ensure_org_member(db, org_id, s["email"], s["password"], s["name"], "student", now, period_end)

        await db.commit()

        print("\nDone. SAI credentials ready:")
        print(f"\n-- {ORG['org_name']} (Org Pro) --")
        print(f"  {'admin.sai@genverse.dev':<35} / {'Sai@Admin1':<14} (Admin,   20000 pts)")
        for t in TEACHERS:
            print(f"  {t['email']:<35} / {t['password']:<14} (Teacher)")
        for s in STUDENTS:
            print(f"  {s['email']:<35} / {s['password']:<14} (Student)")


if __name__ == "__main__":
    asyncio.run(seed())
