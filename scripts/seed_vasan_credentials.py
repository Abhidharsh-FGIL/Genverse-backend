"""
Seed script: creates Vasan Matriculation HR. Sec. School org credentials —
1 admin, 1 teacher, 2 students — with Org Pro plan.

Idempotent: re-running updates existing records instead of skipping them.

Run from the project root:
    python -m scripts.seed_vasan_credentials
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


# ── Org / credential config ───────────────────────────────────────────────────
ORG = {
    "admin_email":           "admin.vasan@genverse.dev",
    "admin_password":        "Vasan@Admin1",
    "admin_name":            "Kannan Ramasamy",
    "org_name":              "Vasan Matriculation HR. Sec. School",
    "plan":                  "org_pro",
    "product_type":          "genverse",
    "has_genverse":          True,
    "has_evaluation":        False,
    "points_balance":        20000,
    "points_monthly_quota":  20000,
    "storage_limit_mb":      51200,
    "max_seats":             1000,
}

TEACHERS = [
    {
        "email":    "meenakshi.vasan@genverse.dev",
        "password": "Meenakshi@1",
        "name":     "Meenakshi Sundaram",
    },
]

STUDENTS = [
    {
        "email":    "karthikeyan.vasan@genverse.dev",
        "password": "Karthikeyan@1",
        "name":     "Karthikeyan Murugan",
    },
    {
        "email":    "priyanka.vasan@genverse.dev",
        "password": "Priyanka@1",
        "name":     "Priyanka Selvam",
    },
]


# ── DB helpers ────────────────────────────────────────────────────────────────
async def _upsert_user(db, email, password, name, now, period_end):
    """Create user if not exists, update name + password if they do. Returns user_id."""
    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        existing.name            = name
        existing.hashed_password = hash_password(password)
        existing.is_active       = True
        existing.onboarding_completed = True
        print(f"    [update] user: {email}")
        return existing.id

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
    print(f"    [insert] user: {email}")
    return user_id


async def _ensure_org_member(db, org_id, email, password, name, role, now, period_end):
    """Upsert user and ensure they are an active org member with the given role."""
    user_id = await _upsert_user(db, email, password, name, now, period_end)

    existing_member = await db.scalar(
        select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
    )
    if existing_member:
        existing_member.role   = role
        existing_member.status = "active"
        print(f"    [update] org member: {email} → {role}")
    else:
        db.add(OrgMember(
            id=uuid.uuid4(),
            org_id=org_id,
            user_id=user_id,
            role=role,
            status="active",
        ))
        print(f"    [insert] org member: {email} → {role}")


# ── Main seed ─────────────────────────────────────────────────────────────────
async def seed():
    async with AsyncSessionLocal() as db:
        now        = datetime.now(timezone.utc)
        period_end = now + timedelta(days=365)

        print(f"\n-- {ORG['org_name']} ({ORG['plan']}) --")

        # ── Admin user ───────────────────────────────────────────────────────
        user_id = await _upsert_user(
            db, ORG["admin_email"], ORG["admin_password"], ORG["admin_name"], now, period_end
        )

        # ── Resolve or create the org ────────────────────────────────────────
        member = await db.scalar(
            select(OrgMember).where(
                OrgMember.user_id == user_id,
                OrgMember.role    == "org_admin",
            )
        )
        org_id = member.org_id if member else None

        if org_id:
            # Org already exists — update name + config
            org = await db.scalar(select(Organization).where(Organization.id == org_id))
            if org:
                org.name             = ORG["org_name"]
                org.product_type     = ORG["product_type"]
                org.has_genverse     = ORG["has_genverse"]
                org.has_evaluation   = ORG["has_evaluation"]
                org.branding_enabled = True
                print(f"  [update] org name → '{ORG['org_name']}'")

            sub = await db.scalar(
                select(Subscription).where(
                    Subscription.org_id         == org_id,
                    Subscription.workspace_type == "organization",
                )
            )
            if sub:
                sub.plan                 = ORG["plan"]
                sub.status               = "active"
                sub.points_balance       = ORG["points_balance"]
                sub.points_monthly_quota = ORG["points_monthly_quota"]
                sub.storage_limit_mb     = ORG["storage_limit_mb"]
                sub.max_seats            = ORG["max_seats"]
                print(f"  [update] org subscription → {ORG['plan']}")
            else:
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
                print(f"  [insert] org subscription → {ORG['plan']}")
        else:
            # Brand-new org
            org_id = uuid.uuid4()
            db.add(Organization(
                id=org_id,
                name=ORG["org_name"],
                product_type=ORG["product_type"],
                has_genverse=ORG["has_genverse"],
                has_evaluation=ORG["has_evaluation"],
                branding_enabled=True,
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

        # ── Teachers ─────────────────────────────────────────────────────────
        print("\n  -- Teachers --")
        for t in TEACHERS:
            await _ensure_org_member(
                db, org_id, t["email"], t["password"], t["name"], "teacher", now, period_end
            )

        # ── Students ─────────────────────────────────────────────────────────
        print("\n  -- Students --")
        for s in STUDENTS:
            await _ensure_org_member(
                db, org_id, s["email"], s["password"], s["name"], "student", now, period_end
            )

        await db.commit()

        # ── Summary ──────────────────────────────────────────────────────────
        W = 42
        print(f"\n{'─' * 62}")
        print(f"  Vasan Matriculation HR. Sec. School  —  Org Pro")
        print(f"{'─' * 62}")
        print(f"  {'admin.vasan@genverse.dev':{W}} / Vasan@Admin1     (Admin)")
        for t in TEACHERS:
            print(f"  {t['email']:{W}} / {t['password']:<16} (Teacher)")
        for s in STUDENTS:
            print(f"  {s['email']:{W}} / {s['password']:<16} (Student)")
        print(f"{'─' * 62}\n")


if __name__ == "__main__":
    asyncio.run(seed())
