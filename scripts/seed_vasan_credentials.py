"""
Seed script: creates Vasan Matriculation HR. Sec. School org credentials —
1 admin, 1 teacher, 2 students — with Org Pro plan.

Run from the project root:
    python -m scripts.seed_vasan_credentials

Logo: a placeholder badge PNG is written to uploads/logos/vasan_logo.png on
first run.  Replace that file with the actual school logo PNG at any time and
re-run to update the stored URL (the path stays the same).
"""
import asyncio
import struct
import zlib
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


# ── Logo ──────────────────────────────────────────────────────────────────────
LOGO_REL   = "logos/vasan_logo.png"
LOGO_URL   = f"/uploads/{LOGO_REL}"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    payload = tag + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def _make_badge_png(size: int = 120) -> bytes:
    """
    Pure-stdlib 24-bit PNG — dark-green ring, cream fill.
    No external dependencies required.
    """
    r = size // 2
    outer_r  = r
    inner_r  = int(r * 0.86)

    DARK_GREEN = (26,  92,  26)
    CREAM      = (240, 232, 176)
    WHITE      = (255, 255, 255)

    raw = bytearray()
    for y in range(size):
        raw += b"\x00"           # PNG filter byte: None
        cy = y - r
        for x in range(size):
            cx = x - r
            d  = (cx * cx + cy * cy) ** 0.5
            raw += bytes(CREAM if d <= inner_r else DARK_GREEN if d <= outer_r else WHITE)

    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw)))
        + _png_chunk(b"IEND", b"")
    )


def _ensure_logo(uploads_root: str) -> None:
    logo_path = os.path.join(uploads_root, LOGO_REL)
    if os.path.exists(logo_path):
        print(f"  [logo]  found at {logo_path}")
        return
    os.makedirs(os.path.dirname(logo_path), exist_ok=True)
    with open(logo_path, "wb") as fh:
        fh.write(_make_badge_png())
    print(f"  [logo]  placeholder badge saved → {logo_path}")
    print("          Replace this file with the actual school logo PNG to update all profiles.")


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
async def _create_user(db, email, password, name, now, period_end):
    user_id = uuid.uuid4()
    db.add(User(
        id=user_id,
        email=email,
        hashed_password=hash_password(password),
        name=name,
        avatar_url=LOGO_URL,
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
    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        user_id = existing.id
        print(f"    [skip]   user exists: {email}")
    else:
        user_id = await _create_user(db, email, password, name, now, period_end)
        print(f"    [insert] user: {email}")

    existing_member = await db.scalar(
        select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
    )
    if existing_member:
        print(f"    [skip]   already member: {email} ({existing_member.role})")
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
    script_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uploads_root = os.path.join(script_dir, "uploads")
    _ensure_logo(uploads_root)

    async with AsyncSessionLocal() as db:
        now        = datetime.now(timezone.utc)
        period_end = now + timedelta(days=365)

        print(f"\n-- {ORG['org_name']} ({ORG['plan']}) --")

        # Admin
        existing_admin = await db.scalar(select(User).where(User.email == ORG["admin_email"]))
        if existing_admin:
            user_id = existing_admin.id
            print(f"  [skip]   admin exists: {ORG['admin_email']}")

            member = await db.scalar(
                select(OrgMember).where(
                    OrgMember.user_id == user_id,
                    OrgMember.role    == "org_admin",
                )
            )
            org_id = member.org_id if member else None

            if org_id:
                org = await db.scalar(select(Organization).where(Organization.id == org_id))
                if org:
                    org.product_type   = ORG["product_type"]
                    org.has_genverse   = ORG["has_genverse"]
                    org.has_evaluation = ORG["has_evaluation"]
                    org.logo_url       = LOGO_URL

                sub = await db.scalar(
                    select(Subscription).where(
                        Subscription.org_id         == org_id,
                        Subscription.workspace_type == "organization",
                    )
                )
                if sub:
                    sub.plan                  = ORG["plan"]
                    sub.status                = "active"
                    sub.points_balance        = ORG["points_balance"]
                    sub.points_monthly_quota  = ORG["points_monthly_quota"]
                    sub.storage_limit_mb      = ORG["storage_limit_mb"]
                    sub.max_seats             = ORG["max_seats"]
                    print(f"  [update] org subscription → {ORG['plan']}")
        else:
            user_id = await _create_user(
                db, ORG["admin_email"], ORG["admin_password"], ORG["admin_name"], now, period_end
            )
            print(f"  [insert] admin: {ORG['admin_email']}")
            org_id = None

        # Create org if needed
        if not existing_admin or org_id is None:
            org_id = uuid.uuid4()
            db.add(Organization(
                id=org_id,
                name=ORG["org_name"],
                logo_url=LOGO_URL,
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
            print(f"  [insert] org '{ORG['org_name']}' + Org Pro subscription")

        # Teachers
        print("\n  -- Teachers --")
        for t in TEACHERS:
            await _ensure_org_member(
                db, org_id, t["email"], t["password"], t["name"], "teacher", now, period_end
            )

        # Students
        print("\n  -- Students --")
        for s in STUDENTS:
            await _ensure_org_member(
                db, org_id, s["email"], s["password"], s["name"], "student", now, period_end
            )

        await db.commit()

        # Summary
        W = 42
        print(f"\n{'─' * 62}")
        print(f"  Vasan Matriculation HR. Sec. School  —  Org Pro")
        print(f"{'─' * 62}")
        print(f"  {'admin.vasan@genverse.dev':{W}} / Vasan@Admin1     (Admin, 20 000 pts)")
        for t in TEACHERS:
            print(f"  {t['email']:{W}} / {t['password']:<16} (Teacher)")
        for s in STUDENTS:
            print(f"  {s['email']:{W}} / {s['password']:<16} (Student)")
        print(f"{'─' * 62}")
        print(f"  Logo → {LOGO_URL}")
        print(f"  File → {os.path.join(uploads_root, LOGO_REL)}")
        print(f"  Replace the file with the actual school logo PNG to update")
        print(f"  all profile avatars and the org logo simultaneously.\n")


if __name__ == "__main__":
    asyncio.run(seed())
