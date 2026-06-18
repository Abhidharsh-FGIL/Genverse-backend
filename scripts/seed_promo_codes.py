"""
Seed script: creates high-discount promo codes (99%, 95%, 90% off).
Run from the project root:
    python -m scripts.seed_promo_codes
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from app.database import AsyncSessionLocal
from app.models.promo import PromoCode


PROMO_CODES = [
    {
        "code": "GENVERSE99",
        "description": "99% off — internal / partner use only",
        "applies_to": "plan",
        "discount_type": "percentage",
        "discount_value": 99.0,
        "max_uses": 500,
        "per_user_limit": 1,
        "is_active": True,
    },
    {
        "code": "GENVERSE95",
        "description": "95% off — internal / partner use only",
        "applies_to": "plan",
        "discount_type": "percentage",
        "discount_value": 95.0,
        "max_uses": 500,
        "per_user_limit": 1,
        "is_active": True,
    },
    {
        "code": "GENVERSE90",
        "description": "90% off — internal / partner use only",
        "applies_to": "plan",
        "discount_type": "percentage",
        "discount_value": 90.0,
        "max_uses": 500,
        "per_user_limit": 1,
        "is_active": True,
    },
]


async def seed():
    async with AsyncSessionLocal() as session:
        print("\n" + "=" * 55)
        print("  Genverse Promo Code Seeder")
        print("=" * 55)

        for data in PROMO_CODES:
            result = await session.execute(
                select(PromoCode).where(PromoCode.code == data["code"])
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update fields in case discount values changed
                existing.description = data["description"]
                existing.applies_to = data["applies_to"]
                existing.discount_type = data["discount_type"]
                existing.discount_value = data["discount_value"]
                existing.max_uses = data["max_uses"]
                existing.per_user_limit = data["per_user_limit"]
                existing.is_active = data["is_active"]
                status = "UPDATED"
            else:
                promo = PromoCode(**data)
                session.add(promo)
                status = "CREATED"

            discount = int(data["discount_value"])
            print(
                f"  [{status:7s}]  code={data['code']:<14}  "
                f"discount={discount}%  "
                f"max_uses={data['max_uses']}  "
                f"applies_to={data['applies_to']}"
            )

        await session.commit()
        print("=" * 55)
        print("  Done. All promo codes committed to the database.")
        print("=" * 55 + "\n")


if __name__ == "__main__":
    asyncio.run(seed())
