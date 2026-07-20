"""
Targeted points top-up for the Pro+Eval Test Academy org.

Accounts covered:
  orgproeval@genverse.dev      (org admin)
  teacher-proeval@genverse.dev (teacher)
  student-proeval@genverse.dev (student)

What it does:
  1. Locates the organisation via the admin email.
  2. Resets the org subscription's points_balance and points_monthly_quota
     to POINTS_BALANCE (default 20 000).
  3. Ensures the subscription is active and the period hasn't lapsed.
  4. Does NOT touch any other org or individual subscription.

Run from the project root:
    python -m scripts.topup_proeval_points

Optional override (set POINTS env var before running):
    POINTS=50000 python -m scripts.topup_proeval_points
"""
import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from app.database import AsyncSessionLocal
from app.models.user import User
from app.models.subscription import Subscription
from app.models.organization import OrgMember

# ── Config ────────────────────────────────────────────────────────────────────

ADMIN_EMAIL    = "orgproeval@genverse.dev"
POINTS_BALANCE = int(os.environ.get("POINTS", 20_000))


async def main() -> None:
    async with AsyncSessionLocal() as db:
        now        = datetime.now(timezone.utc)
        period_end = now + timedelta(days=365)

        # 1. Resolve admin user
        admin = await db.scalar(select(User).where(User.email == ADMIN_EMAIL))
        if not admin:
            print(f"[ERROR] Admin user not found: {ADMIN_EMAIL}")
            print("        Run  python -m scripts.seed_test_users  first to create the org.")
            return

        # 2. Find org via admin membership
        membership = await db.scalar(
            select(OrgMember).where(
                OrgMember.user_id == admin.id,
                OrgMember.role    == "org_admin",
            )
        )
        if not membership:
            print(f"[ERROR] {ADMIN_EMAIL} exists but has no org_admin membership.")
            print("        Run  python -m scripts.seed_test_users  to set up the org.")
            return

        org_id = membership.org_id
        print(f"[INFO]  Found org  id={org_id}")

        # 3. Locate org subscription
        sub = await db.scalar(
            select(Subscription).where(
                Subscription.org_id         == org_id,
                Subscription.workspace_type == "organization",
            )
        )
        if not sub:
            print("[ERROR] No organisation-level subscription found.")
            print("        Run  python -m scripts.seed_test_users  to create it.")
            return

        old_balance = sub.points_balance
        sub.points_balance       = POINTS_BALANCE
        sub.points_monthly_quota = POINTS_BALANCE
        sub.status               = "active"
        sub.current_period_start = now
        sub.current_period_end   = period_end

        await db.commit()

        print()
        print("=" * 60)
        print("  POINTS TOP-UP COMPLETE")
        print("=" * 60)
        print(f"  Org ID          : {org_id}")
        print(f"  Admin email     : {ADMIN_EMAIL}")
        print(f"  Points before   : {old_balance}")
        print(f"  Points after    : {POINTS_BALANCE}")
        print(f"  Monthly quota   : {POINTS_BALANCE}")
        print(f"  Period end      : {period_end.strftime('%Y-%m-%d')}")
        print()
        print("  Accounts in this org:")
        print(f"    orgproeval@genverse.dev       (Admin)")
        print(f"    teacher-proeval@genverse.dev  (Teacher)")
        print(f"    student-proeval@genverse.dev  (Student)")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
