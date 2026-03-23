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

from sqlalchemy import select, delete
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
        "product_type": "genverse",
        "has_genverse": True,
        "has_evaluation": False,
        "points_balance": 5000,
        "points_monthly_quota": 5000,
        "storage_limit_mb": 5120,
        "max_seats": 50,
        "add_teacher_student": True,
        "slug": "basic",
    },
    {
        "email": "orgpro@genverse.dev",
        "password": "Test@123",
        "name": "Org Pro Admin",
        "org_name": "Pro Test Academy",
        "plan": "org_pro",
        "product_type": "genverse",
        "has_genverse": True,
        "has_evaluation": False,
        "points_balance": 20000,
        "points_monthly_quota": 20000,
        "storage_limit_mb": 51200,
        "max_seats": 1000,
        "add_teacher_student": True,
        "slug": "pro",
    },
    {
        "email": "orgbasiceval@genverse.dev",
        "password": "Test@123",
        "name": "Org Basic+Eval Admin",
        "org_name": "Basic+Eval Test School",
        "plan": "org_basic",
        "product_type": "genverse_evaluation",
        "has_genverse": True,
        "has_evaluation": True,
        "points_balance": 5000,
        "points_monthly_quota": 5000,
        "storage_limit_mb": 5120,
        "max_seats": 50,
        "add_teacher_student": True,
        "slug": "basiceval",
    },
    {
        "email": "orgproeval@genverse.dev",
        "password": "Test@123",
        "name": "Org Pro+Eval Admin",
        "org_name": "Pro+Eval Test Academy",
        "plan": "org_pro",
        "product_type": "genverse_evaluation",
        "has_genverse": True,
        "has_evaluation": True,
        "points_balance": 20000,
        "points_monthly_quota": 20000,
        "storage_limit_mb": 51200,
        "max_seats": 1000,
        "add_teacher_student": True,
        "slug": "proeval",
    },
    {
        "email": "orgeval@genverse.dev",
        "password": "Test@123",
        "name": "Org Eval Admin",
        "org_name": "Evaluation Test Institute",
        "plan": "org_evaluation",
        "product_type": "evaluation",
        "has_genverse": False,
        "has_evaluation": True,
        "points_balance": 3000,
        "points_monthly_quota": 3000,
        "storage_limit_mb": 2048,
        "max_seats": 100,
        "add_teacher_student": False,
        "slug": "eval",
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
    ))
    print(f"  [insert] user: {email} ({name})")
    return user_id


async def _ensure_org_member(db, org_id, email, password, name, role, now, period_end):
    """Create a user (if needed) and add as org member."""
    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        user_id = existing.id
        print(f"  [skip] user already exists: {email}")
    else:
        user_id = await _create_user(db, email, password, name, now, period_end)

    # Add to org if not already a member
    existing_member = await db.scalar(
        select(OrgMember).where(
            OrgMember.org_id == org_id,
            OrgMember.user_id == user_id,
        )
    )
    if existing_member:
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


async def _seed_org_admin(db, data, now, period_end):
    """Create or update an org admin user + organization + org subscription + teacher/student."""
    existing = await db.scalar(
        select(User).where(User.email == data["email"])
    )

    if existing:
        user_id = existing.id
        print(f"  [skip] user already exists: {data['email']}")

        # Ensure org exists for this user — first try org_admin, then any role (may have been corrupted)
        member = await db.scalar(
            select(OrgMember).where(
                OrgMember.user_id == user_id,
                OrgMember.role == "org_admin",
            )
        )
        if not member:
            # Role may have been corrupted (e.g. bulk upload changed org_admin to student)
            member = await db.scalar(
                select(OrgMember).where(OrgMember.user_id == user_id)
            )
            if member:
                print(f"  [fix] restoring org_admin role (was '{member.role}') for {data['email']}")
                member.role = "org_admin"
                member.status = "active"

        if member:
            org_id = member.org_id

            # Clean up ALL other OrgMember entries for this user (duplicate orgs, stale roles)
            all_memberships = await db.execute(
                select(OrgMember).where(
                    OrgMember.user_id == user_id,
                    OrgMember.id != member.id,
                )
            )
            for stale in all_memberships.scalars().all():
                # If it's a duplicate org created by the seed script, also remove that org
                if stale.role == "org_admin" and stale.org_id != org_id:
                    dup_org = await db.scalar(select(Organization).where(Organization.id == stale.org_id))
                    if dup_org:
                        # Remove subscription for duplicate org
                        dup_sub = await db.scalar(
                            select(Subscription).where(Subscription.org_id == stale.org_id)
                        )
                        if dup_sub:
                            await db.delete(dup_sub)
                        # Remove all members of duplicate org
                        dup_members = await db.execute(
                            select(OrgMember).where(OrgMember.org_id == stale.org_id)
                        )
                        for dm in dup_members.scalars().all():
                            await db.delete(dm)
                        await db.delete(dup_org)
                        print(f"  [cleanup] removed duplicate org '{dup_org.name}' for {data['email']}")
                    continue
                print(f"  [cleanup] removing stale '{stale.role}' membership for {data['email']}")
                await db.delete(stale)

            # Clean up stale UserRole entries (e.g. student/teacher roles added by bad bulk uploads)
            stale_roles = await db.execute(
                select(UserRole).where(
                    UserRole.user_id == user_id,
                    UserRole.role.notin_(["normal_user", "org_admin", "teacher"]),
                )
            )
            for stale in stale_roles.scalars().all():
                print(f"  [cleanup] removing stale UserRole '{stale.role}' for {data['email']}")
                await db.delete(stale)

            # Update org product_type flags
            org = await db.scalar(
                select(Organization).where(Organization.id == org_id)
            )
            if org:
                org.product_type = data["product_type"]
                org.has_genverse = data["has_genverse"]
                org.has_evaluation = data["has_evaluation"]
                print(f"  [update] org flags -> product_type={data['product_type']}, genverse={data['has_genverse']}, eval={data['has_evaluation']}")

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

            # Add teacher/student if needed
            if data["add_teacher_student"]:
                slug = data["slug"]
                await _ensure_org_member(
                    db, org_id,
                    f"teacher-{slug}@genverse.dev", data["password"],
                    f"Teacher ({data['org_name']})", "teacher", now, period_end,
                )
                await _ensure_org_member(
                    db, org_id,
                    f"student-{slug}@genverse.dev", data["password"],
                    f"Student ({data['org_name']})", "student", now, period_end,
                )
            return

        # User exists but no org — create org + membership below
    else:
        # Create user
        user_id = await _create_user(db, data["email"], data["password"], data["name"], now, period_end)

    # Create organization
    org_id = uuid.uuid4()
    db.add(Organization(
        id=org_id,
        name=data["org_name"],
        product_type=data["product_type"],
        has_genverse=data["has_genverse"],
        has_evaluation=data["has_evaluation"],
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

    # Add teacher/student members if needed
    if data["add_teacher_student"]:
        slug = data["slug"]
        await _ensure_org_member(
            db, org_id,
            f"teacher-{slug}@genverse.dev", data["password"],
            f"Teacher ({data['org_name']})", "teacher", now, period_end,
        )
        await _ensure_org_member(
            db, org_id,
            f"student-{slug}@genverse.dev", data["password"],
            f"Student ({data['org_name']})", "student", now, period_end,
        )


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
        print("\n── Personal workspace ──")
        print("  freeuser@genverse.dev          / Test@123  (Free plan, 100 pts)")
        print("  plususer@genverse.dev          / Test@123  (Plus plan, 800 pts)")
        print("  prouser@genverse.dev           / Test@123  (Pro plan, 2000 pts)")
        print("\n── Org Basic (GenVerse only) ──")
        print("  orgbasic@genverse.dev          / Test@123  (Admin, 5000 pts)")
        print("  teacher-basic@genverse.dev     / Test@123  (Teacher)")
        print("  student-basic@genverse.dev     / Test@123  (Student)")
        print("\n── Org Pro (GenVerse only) ──")
        print("  orgpro@genverse.dev            / Test@123  (Admin, 20000 pts)")
        print("  teacher-pro@genverse.dev       / Test@123  (Teacher)")
        print("  student-pro@genverse.dev       / Test@123  (Student)")
        print("\n── Org Basic + Evaluation Hub ──")
        print("  orgbasiceval@genverse.dev      / Test@123  (Admin, 5000 pts)")
        print("  teacher-basiceval@genverse.dev / Test@123  (Teacher)")
        print("  student-basiceval@genverse.dev / Test@123  (Student)")
        print("\n── Org Pro + Evaluation Hub ──")
        print("  orgproeval@genverse.dev        / Test@123  (Admin, 20000 pts)")
        print("  teacher-proeval@genverse.dev   / Test@123  (Teacher)")
        print("  student-proeval@genverse.dev   / Test@123  (Student)")
        print("\n── Evaluation Hub Only (no teacher/student) ──")
        print("  orgeval@genverse.dev           / Test@123  (Admin, 3000 pts)")


if __name__ == "__main__":
    asyncio.run(seed())
