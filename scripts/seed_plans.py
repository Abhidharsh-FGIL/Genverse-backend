"""
Seed script: populates plan_definitions, point_costs, and feature_limits tables.
Run from the project root:
    python -m scripts.seed_plans
"""
import asyncio
import uuid
import sys
import os

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, text
from app.database import AsyncSessionLocal
from app.models.subscription import PlanDefinition, PointCost, FeatureLimit


# ── Plan definitions ──────────────────────────────────────────────────────────

PLAN_DEFINITIONS = [
    {
        "plan": "free",
        "display_name": "Free",
        "price_inr": 0.0,
        "workspace_type": "individual",
        "monthly_points": 100,
        "storage_mb": 100,
        "max_seats": None,
        "description": "Get started for free. Includes basic AI tools and limited usage.",
        "is_active": True,
    },
    {
        "plan": "individual_pro",
        "display_name": "Individual Pro",
        "price_inr": 499.0,
        "workspace_type": "individual",
        "monthly_points": 2000,
        "storage_mb": 2048,
        "max_seats": None,
        "description": "For individual learners and educators who need more power.",
        "is_active": True,
    },
    {
        "plan": "individual_power",
        "display_name": "Individual Power",
        "price_inr": 999.0,
        "workspace_type": "individual",
        "monthly_points": 5000,
        "storage_mb": 10240,
        "max_seats": None,
        "description": "Maximum power for individuals — unlimited creativity.",
        "is_active": True,
    },
    {
        "plan": "org_basic",
        "display_name": "Organization Basic",
        "price_inr": 4999.0,
        "workspace_type": "organization",
        "monthly_points": 5000,
        "storage_mb": 5120,
        "max_seats": 50,
        "description": "For small organizations getting started with AI-powered education.",
        "is_active": True,
    },
    {
        "plan": "org_pro",
        "display_name": "Organization Pro",
        "price_inr": 14999.0,
        "workspace_type": "organization",
        "monthly_points": 20000,
        "storage_mb": 51200,
        "max_seats": 1000,
        "description": "Full-featured plan for growing organizations and institutions.",
        "is_active": True,
    },
    {
        "plan": "org_evaluation",
        "display_name": "Organization Evaluation",
        "price_inr": 0.0,
        "workspace_type": "organization",
        "monthly_points": 3000,
        "storage_mb": 2048,
        "max_seats": 100,
        "description": "Trial plan for organizations evaluating the platform.",
        "is_active": True,
    },
]


# ── Point costs per action ────────────────────────────────────────────────────

POINT_COSTS = [
    {"action": "generate_content",       "cost": 5,  "description": "Generate AI content (lesson, notes, etc.)"},
    {"action": "create_assessment",      "cost": 10, "description": "Create a full assessment or quiz"},
    {"action": "generate_quiz",          "cost": 5,  "description": "Generate a quick quiz"},
    {"action": "chat_message",           "cost": 1,  "description": "Single AI assistant chat message"},
    {"action": "playground_story",       "cost": 5,  "description": "Generate an interactive story"},
    {"action": "playground_quiz",        "cost": 5,  "description": "Generate a playground quiz battle"},
    {"action": "playground_roleplay",    "cost": 5,  "description": "Generate a roleplay scenario"},
    {"action": "generate_ebook",         "cost": 20, "description": "Generate a full eBook"},
    {"action": "upload_file",            "cost": 2,  "description": "Upload and process a file"},
    {"action": "summarize_document",     "cost": 5,  "description": "Summarize an uploaded document"},
    {"action": "translate_content",      "cost": 3,  "description": "Translate content to another language"},
    {"action": "generate_flashcards",    "cost": 5,  "description": "Generate flashcard set"},
    {"action": "export_pdf",             "cost": 2,  "description": "Export content as PDF"},
]


# ── Feature limits per plan ───────────────────────────────────────────────────
# daily_limit / monthly_limit = None means unlimited for that plan

FEATURE_LIMITS = [
    # ── free ──
    {"plan": "free", "feature_key": "ai_chat",            "enabled": True,  "daily_limit": 10,  "monthly_limit": 50},
    {"plan": "free", "feature_key": "generate_content",   "enabled": True,  "daily_limit": 3,   "monthly_limit": 20},
    {"plan": "free", "feature_key": "create_assessment",  "enabled": True,  "daily_limit": 2,   "monthly_limit": 10},
    {"plan": "free", "feature_key": "playground",         "enabled": True,  "daily_limit": 3,   "monthly_limit": 15},
    {"plan": "free", "feature_key": "ebook",              "enabled": False, "daily_limit": 0,   "monthly_limit": 0},
    {"plan": "free", "feature_key": "file_upload",        "enabled": True,  "daily_limit": 2,   "monthly_limit": 10},
    {"plan": "free", "feature_key": "export_pdf",         "enabled": True,  "daily_limit": 2,   "monthly_limit": 10},

    # ── individual_pro ──
    {"plan": "individual_pro", "feature_key": "ai_chat",            "enabled": True, "daily_limit": 50,  "monthly_limit": None},
    {"plan": "individual_pro", "feature_key": "generate_content",   "enabled": True, "daily_limit": 20,  "monthly_limit": None},
    {"plan": "individual_pro", "feature_key": "create_assessment",  "enabled": True, "daily_limit": 10,  "monthly_limit": None},
    {"plan": "individual_pro", "feature_key": "playground",         "enabled": True, "daily_limit": 10,  "monthly_limit": None},
    {"plan": "individual_pro", "feature_key": "ebook",              "enabled": True, "daily_limit": 2,   "monthly_limit": 20},
    {"plan": "individual_pro", "feature_key": "file_upload",        "enabled": True, "daily_limit": 10,  "monthly_limit": None},
    {"plan": "individual_pro", "feature_key": "export_pdf",         "enabled": True, "daily_limit": None,"monthly_limit": None},

    # ── individual_power ──
    {"plan": "individual_power", "feature_key": "ai_chat",            "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "individual_power", "feature_key": "generate_content",   "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "individual_power", "feature_key": "create_assessment",  "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "individual_power", "feature_key": "playground",         "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "individual_power", "feature_key": "ebook",              "enabled": True, "daily_limit": 5,    "monthly_limit": None},
    {"plan": "individual_power", "feature_key": "file_upload",        "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "individual_power", "feature_key": "export_pdf",         "enabled": True, "daily_limit": None, "monthly_limit": None},

    # ── org_basic ──
    {"plan": "org_basic", "feature_key": "ai_chat",            "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_basic", "feature_key": "generate_content",   "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_basic", "feature_key": "create_assessment",  "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_basic", "feature_key": "playground",         "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_basic", "feature_key": "ebook",              "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_basic", "feature_key": "file_upload",        "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_basic", "feature_key": "export_pdf",         "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_basic", "feature_key": "org_management",     "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_basic", "feature_key": "member_invites",     "enabled": True, "daily_limit": None, "monthly_limit": None},

    # ── org_pro ──
    {"plan": "org_pro", "feature_key": "ai_chat",            "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_pro", "feature_key": "generate_content",   "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_pro", "feature_key": "create_assessment",  "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_pro", "feature_key": "playground",         "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_pro", "feature_key": "ebook",              "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_pro", "feature_key": "file_upload",        "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_pro", "feature_key": "export_pdf",         "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_pro", "feature_key": "org_management",     "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_pro", "feature_key": "member_invites",     "enabled": True, "daily_limit": None, "monthly_limit": None},
    {"plan": "org_pro", "feature_key": "analytics",          "enabled": True, "daily_limit": None, "monthly_limit": None},

    # ── org_evaluation ──
    {"plan": "org_evaluation", "feature_key": "ai_chat",            "enabled": True, "daily_limit": 20,  "monthly_limit": None},
    {"plan": "org_evaluation", "feature_key": "generate_content",   "enabled": True, "daily_limit": 10,  "monthly_limit": None},
    {"plan": "org_evaluation", "feature_key": "create_assessment",  "enabled": True, "daily_limit": 5,   "monthly_limit": None},
    {"plan": "org_evaluation", "feature_key": "playground",         "enabled": True, "daily_limit": 5,   "monthly_limit": None},
    {"plan": "org_evaluation", "feature_key": "ebook",              "enabled": True, "daily_limit": 1,   "monthly_limit": 5},
    {"plan": "org_evaluation", "feature_key": "file_upload",        "enabled": True, "daily_limit": 5,   "monthly_limit": None},
    {"plan": "org_evaluation", "feature_key": "export_pdf",         "enabled": True, "daily_limit": 5,   "monthly_limit": None},
    {"plan": "org_evaluation", "feature_key": "org_management",     "enabled": True, "daily_limit": None,"monthly_limit": None},
    {"plan": "org_evaluation", "feature_key": "member_invites",     "enabled": True, "daily_limit": None,"monthly_limit": None},
]


# ── Seed logic ────────────────────────────────────────────────────────────────

async def seed():
    async with AsyncSessionLocal() as db:
        inserted = {"plan_definitions": 0, "point_costs": 0, "feature_limits": 0}

        # ── plan_definitions ──
        for data in PLAN_DEFINITIONS:
            existing = await db.scalar(
                select(PlanDefinition).where(PlanDefinition.plan == data["plan"])
            )
            if existing:
                print(f"  [skip]   plan_definitions -> {data['plan']} (already exists)")
            else:
                db.add(PlanDefinition(id=uuid.uuid4(), **data))
                inserted["plan_definitions"] += 1
                print(f"  [insert] plan_definitions -> {data['plan']}")

        # ── point_costs ──
        for data in POINT_COSTS:
            existing = await db.scalar(
                select(PointCost).where(PointCost.action == data["action"])
            )
            if existing:
                print(f"  [skip]   point_costs -> {data['action']} (already exists)")
            else:
                db.add(PointCost(id=uuid.uuid4(), **data))
                inserted["point_costs"] += 1
                print(f"  [insert] point_costs -> {data['action']}")

        # ── feature_limits ──
        for data in FEATURE_LIMITS:
            existing = await db.scalar(
                select(FeatureLimit).where(
                    FeatureLimit.plan == data["plan"],
                    FeatureLimit.feature_key == data["feature_key"],
                )
            )
            if existing:
                print(f"  [skip]   feature_limits -> {data['plan']}/{data['feature_key']} (already exists)")
            else:
                db.add(FeatureLimit(id=uuid.uuid4(), **data))
                inserted["feature_limits"] += 1
                print(f"  [insert] feature_limits -> {data['plan']}/{data['feature_key']}")

        await db.commit()

        print("\nDone.")
        print(f"  plan_definitions : {inserted['plan_definitions']} inserted")
        print(f"  point_costs      : {inserted['point_costs']} inserted")
        print(f"  feature_limits   : {inserted['feature_limits']} inserted")


if __name__ == "__main__":
    asyncio.run(seed())
