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

from sqlalchemy import select, delete, text
from app.database import AsyncSessionLocal
from app.models.subscription import PlanDefinition, PointCost, FeatureLimit


# ── Plan definitions ──────────────────────────────────────────────────────────
# Only individual plans updated. Org plans remain unchanged.

PLAN_DEFINITIONS = [
    {
        "plan": "free",
        "display_name": "Free",
        "price_inr": 0.0,
        "workspace_type": "individual",
        "monthly_points": 100,
        "storage_mb": 100,
        "max_file_size_mb": 5,
        "max_seats": None,
        "description": "Get started for free with basic AI tools and limited usage.",
        "is_active": True,
    },
    {
        "plan": "individual_pro",
        "display_name": "Plus",
        "price_inr": 299.0,
        "workspace_type": "individual",
        "monthly_points": 800,
        "storage_mb": 200,
        "max_file_size_mb": 10,
        "max_seats": None,
        "description": "For serious learners who want to accelerate their growth.",
        "is_active": True,
    },
    {
        "plan": "individual_power",
        "display_name": "Pro",
        "price_inr": 999.0,
        "workspace_type": "individual",
        "monthly_points": 2000,
        "storage_mb": 500,
        "max_file_size_mb": 20,
        "max_seats": None,
        "description": "Maximum capability. Every tool, no limits.",
        "is_active": True,
    },
    # ── Org plans (unchanged) ──
    {
        "plan": "org_basic",
        "display_name": "Organization Basic",
        "price_inr": 4999.0,
        "workspace_type": "organization",
        "monthly_points": 5000,
        "storage_mb": 5120,
        "max_file_size_mb": 20,
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
        "max_file_size_mb": 50,
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
        "max_file_size_mb": 20,
        "max_seats": 100,
        "description": "Trial plan for organizations evaluating the platform.",
        "is_active": True,
    },
]


# ── Point costs per action ────────────────────────────────────────────────────
# Action keys MUST match what backend routers pass to PointsService.deduct()

POINT_COSTS = [
    # AI Assistant — base + per-enhancement sub-features
    {"action": "basic_chat",             "cost": 1,  "xp_reward": 2,  "description": "AI chat message"},
    {"action": "chat_followup",          "cost": 0,  "xp_reward": 1,  "description": "AI chat follow-up questions"},
    {"action": "chat_next_steps",        "cost": 0,  "xp_reward": 1,  "description": "AI chat next step suggestions"},
    {"action": "chat_video_refs",        "cost": 1,  "xp_reward": 1,  "description": "AI chat video references"},
    {"action": "chat_mindmap",           "cost": 1,  "xp_reward": 2,  "description": "AI chat mind map generation"},
    {"action": "chat_infographic",       "cost": 1,  "xp_reward": 2,  "description": "AI chat infographic generation"},
    {"action": "chat_practice",          "cost": 1,  "xp_reward": 2,  "description": "AI chat practice exercises"},
    # Playground
    {"action": "playground_explore",     "cost": 2,  "xp_reward": 5,  "description": "Playground session"},
    {"action": "playground_harder",      "cost": 1,  "xp_reward": 2,  "description": "Playground harder mode bonus"},
    # OCR
    {"action": "ocr_extraction",         "cost": 3,  "xp_reward": 3,  "description": "OCR text extraction"},
    # Assessment
    {"action": "generate_assessment",    "cost": 3,  "xp_reward": 5,  "description": "Assessment generation"},
    # eBook
    {"action": "generate_ebook",         "cost": 1,  "xp_reward": 10, "description": "eBook generation (base, multiplied by pages)"},
    {"action": "generate_audiobook",     "cost": 6,  "xp_reward": 5,  "description": "Audio eBook generation"},
    {"action": "ebook_download_pdf",     "cost": 1,  "xp_reward": 0,  "description": "eBook PDF download"},
    {"action": "ebook_download_docx",    "cost": 1,  "xp_reward": 0,  "description": "eBook DOCX download"},
    # Career
    {"action": "career_guidance",        "cost": 2,  "xp_reward": 5,  "description": "Career path generation"},
    # Insights
    {"action": "generate_insights",      "cost": 3,  "xp_reward": 3,  "description": "Recommendation / insight refresh"},
    # News Feed
    {"action": "news_feed",              "cost": 1,  "xp_reward": 1,  "description": "News feed section view"},
    # Recommendations
    {"action": "view_recommendations",   "cost": 1,  "xp_reward": 0,  "description": "View recommendations page"},
    # Career Profile
    {"action": "view_career_profile",    "cost": 1,  "xp_reward": 0,  "description": "View career guidance profile"},
    # Knowledge Vault
    {"action": "rag_query",              "cost": 1,  "xp_reward": 3,  "description": "Knowledge Vault AI query"},
    # Mind Map (standalone)
    {"action": "generate_mindmap",       "cost": 2,  "xp_reward": 10, "description": "Mind map generation"},
    # Video Studio
    {"action": "generate_video_script",  "cost": 5,  "xp_reward": 5,  "description": "Video script generation"},
    {"action": "generate_video_visuals", "cost": 5,  "xp_reward": 5,  "description": "Video visuals generation"},

    # ── Organization workspace actions ──
    {"action": "org_auto_grade",             "cost": 2,  "xp_reward": 8,  "description": "Org: auto-grade student submission", "workspace_type": "org"},
    {"action": "org_ai_feedback",            "cost": 2,  "xp_reward": 5,  "description": "Org: AI feedback on submission", "workspace_type": "org"},
    {"action": "org_lesson_plan_gen",        "cost": 3,  "xp_reward": 5,  "description": "Org: AI lesson plan generation", "workspace_type": "org"},
    {"action": "org_ai_suggest_questions",   "cost": 2,  "xp_reward": 5,  "description": "Org: AI question suggestions", "workspace_type": "org"},
    {"action": "org_rubric_ai_generate",     "cost": 2,  "xp_reward": 5,  "description": "Org: AI rubric generation", "workspace_type": "org"},
    {"action": "org_auto_evaluate_attempt",  "cost": 2,  "xp_reward": 8,  "description": "Org: auto-evaluate assessment attempt", "workspace_type": "org"},
    {"action": "org_class_recommendations",  "cost": 3,  "xp_reward": 3,  "description": "Org: AI class recommendations", "workspace_type": "org"},
]


# ── Feature limits per plan ───────────────────────────────────────────────────
# daily_limit / monthly_limit = None means unlimited for that plan
# Only individual plans get sub-feature restrictions.
# Org plans keep the existing simple feature limits (unchanged below).

_FL = True   # enabled
_FX = False  # disabled
_U = None    # unlimited

FEATURE_LIMITS = [
    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  FREE plan                                                      ║
    # ╚══════════════════════════════════════════════════════════════════╝

    # AI Assistant
    {"plan": "free", "feature_key": "ai_chat",                "enabled": _FL, "daily_limit": 10,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "ai_vault_context",       "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "ai_long_context",        "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    # Knowledge Vault
    {"plan": "free", "feature_key": "vault_ai_chat",          "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "vault_download_notes",   "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "vault_batch_download",   "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    # eBook Creator
    {"plan": "free", "feature_key": "ebook_medium",           "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "ebook_large",            "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "ebook_multi_chapter",    "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "ebook_audio",            "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "ebook_pdf",              "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "ebook_docx",             "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    # Extract OCR
    {"plan": "free", "feature_key": "ocr_pdf",                "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "ocr_multi_page",         "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "ocr_extraction",         "enabled": _FL, "daily_limit": _U,  "monthly_limit": 3},
    # Assessment Hub
    {"plan": "free", "feature_key": "create_assessment",      "enabled": _FL, "daily_limit": _U,  "monthly_limit": 3},
    {"plan": "free", "feature_key": "personal_multi_topic",  "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "personal_blooms",      "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "assessment_assertion",   "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "assessment_hot",         "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    # Playground
    {"plan": "free", "feature_key": "playground_rapid_mcq",       "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "playground_concept_connect", "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "playground_roleplay",        "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "playground_imagine",         "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    # Recommendations
    {"plan": "free", "feature_key": "rec_focus_areas",        "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "ai_action_cards",      "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "weekly_goals",           "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    # Career Guidance
    {"plan": "free", "feature_key": "career_guidance",        "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "career_paths",           "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "career_skill_gap",       "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    # Analytics
    {"plan": "free", "feature_key": "bloom_radar", "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "free", "feature_key": "predictive_growth",     "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    # News Feed
    {"plan": "free", "feature_key": "news_personal_insights", "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  PLUS plan  (individual_pro)                                    ║
    # ╚══════════════════════════════════════════════════════════════════╝

    # AI Assistant — all unlocked
    {"plan": "individual_pro", "feature_key": "ai_chat",                "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "ai_vault_context",       "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "ai_long_context",        "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    # Knowledge Vault — all unlocked
    {"plan": "individual_pro", "feature_key": "vault_ai_chat",          "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "vault_download_notes",   "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "vault_batch_download",   "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    # eBook Creator — medium & multi-chapter yes, large/audio/docx no
    {"plan": "individual_pro", "feature_key": "ebook_medium",           "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "ebook_large",            "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "ebook_multi_chapter",    "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "ebook_audio",            "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "ebook_pdf",              "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "ebook_docx",             "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    # Extract OCR — all unlocked
    {"plan": "individual_pro", "feature_key": "ocr_pdf",                "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "ocr_multi_page",         "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "ocr_extraction",         "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    # Assessment Hub — multi-topic, blooms, assertion yes; HOT no
    {"plan": "individual_pro", "feature_key": "create_assessment",      "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "personal_multi_topic",  "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "personal_blooms",      "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "assessment_assertion",   "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "assessment_hot",         "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    # Playground — rapid MCQ, concept connect yes; roleplay, imagine no
    {"plan": "individual_pro", "feature_key": "playground_rapid_mcq",       "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "playground_concept_connect", "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "playground_roleplay",        "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "playground_imagine",         "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    # Recommendations — all unlocked
    {"plan": "individual_pro", "feature_key": "rec_focus_areas",        "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "ai_action_cards",      "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "weekly_goals",           "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    # Career Guidance — explorer, paths, skill gap all enabled
    {"plan": "individual_pro", "feature_key": "career_guidance",        "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "career_paths",           "enabled": _FX, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "career_skill_gap",       "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    # Analytics — all unlocked
    {"plan": "individual_pro", "feature_key": "bloom_radar", "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_pro", "feature_key": "predictive_growth",     "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    # News Feed — all unlocked
    {"plan": "individual_pro", "feature_key": "news_personal_insights", "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  PRO plan  (individual_power) — everything enabled              ║
    # ╚══════════════════════════════════════════════════════════════════╝

    {"plan": "individual_power", "feature_key": "ai_chat",                "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "ai_vault_context",       "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "ai_long_context",        "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "vault_ai_chat",          "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "vault_download_notes",   "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "vault_batch_download",   "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "ebook_medium",           "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "ebook_large",            "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "ebook_multi_chapter",    "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "ebook_audio",            "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "ebook_pdf",              "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "ebook_docx",             "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "ocr_pdf",                "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "ocr_multi_page",         "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "ocr_extraction",         "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "create_assessment",      "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "personal_multi_topic",  "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "personal_blooms",      "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "assessment_assertion",   "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "assessment_hot",         "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "playground_rapid_mcq",       "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "playground_concept_connect", "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "playground_roleplay",        "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "playground_imagine",         "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "rec_focus_areas",        "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "ai_action_cards",      "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "weekly_goals",           "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "career_guidance",        "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "career_paths",           "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "career_skill_gap",       "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "bloom_radar", "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "predictive_growth",     "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "individual_power", "feature_key": "news_personal_insights", "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  Org plans (unchanged)                                          ║
    # ╚══════════════════════════════════════════════════════════════════╝

    # ── org_basic ──
    {"plan": "org_basic", "feature_key": "ai_chat",            "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_basic", "feature_key": "generate_content",   "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_basic", "feature_key": "create_assessment",  "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_basic", "feature_key": "playground",         "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_basic", "feature_key": "ebook",              "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_basic", "feature_key": "file_upload",        "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_basic", "feature_key": "export_pdf",         "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_basic", "feature_key": "org_management",     "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_basic", "feature_key": "member_invites",     "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},

    # ── org_pro ──
    {"plan": "org_pro", "feature_key": "ai_chat",            "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_pro", "feature_key": "generate_content",   "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_pro", "feature_key": "create_assessment",  "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_pro", "feature_key": "playground",         "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_pro", "feature_key": "ebook",              "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_pro", "feature_key": "file_upload",        "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_pro", "feature_key": "export_pdf",         "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_pro", "feature_key": "org_management",     "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_pro", "feature_key": "member_invites",     "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},
    {"plan": "org_pro", "feature_key": "analytics",          "enabled": _FL, "daily_limit": _U, "monthly_limit": _U},

    # ── org_evaluation ──
    {"plan": "org_evaluation", "feature_key": "ai_chat",            "enabled": _FL, "daily_limit": 20,  "monthly_limit": _U},
    {"plan": "org_evaluation", "feature_key": "generate_content",   "enabled": _FL, "daily_limit": 10,  "monthly_limit": _U},
    {"plan": "org_evaluation", "feature_key": "create_assessment",  "enabled": _FL, "daily_limit": 5,   "monthly_limit": _U},
    {"plan": "org_evaluation", "feature_key": "playground",         "enabled": _FL, "daily_limit": 5,   "monthly_limit": _U},
    {"plan": "org_evaluation", "feature_key": "ebook",              "enabled": _FL, "daily_limit": 1,   "monthly_limit": 5},
    {"plan": "org_evaluation", "feature_key": "file_upload",        "enabled": _FL, "daily_limit": 5,   "monthly_limit": _U},
    {"plan": "org_evaluation", "feature_key": "export_pdf",         "enabled": _FL, "daily_limit": 5,   "monthly_limit": _U},
    {"plan": "org_evaluation", "feature_key": "org_management",     "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
    {"plan": "org_evaluation", "feature_key": "member_invites",     "enabled": _FL, "daily_limit": _U,  "monthly_limit": _U},
]


# ── Seed logic (upsert) ─────────────────────────────────────────────────────

async def seed():
    async with AsyncSessionLocal() as db:
        # Ensure xp_reward column exists (migration helper)
        await db.execute(text("ALTER TABLE point_costs ADD COLUMN IF NOT EXISTS xp_reward INTEGER DEFAULT 0"))
        await db.commit()

        counts = {"plan_definitions": 0, "point_costs": 0, "feature_limits": 0}

        # ── plan_definitions (upsert) ──
        for data in PLAN_DEFINITIONS:
            extra = {}
            if "max_file_size_mb" in data:
                extra["max_file_size_mb"] = data.pop("max_file_size_mb")
            existing = await db.scalar(
                select(PlanDefinition).where(PlanDefinition.plan == data["plan"])
            )
            if existing:
                for k, v in data.items():
                    if k != "plan":
                        setattr(existing, k, v)
                for k, v in extra.items():
                    if hasattr(existing, k):
                        setattr(existing, k, v)
                counts["plan_definitions"] += 1
                print(f"  [update] plan_definitions -> {data['plan']}")
            else:
                obj_data = {**data}
                if extra and hasattr(PlanDefinition, "max_file_size_mb"):
                    obj_data.update(extra)
                db.add(PlanDefinition(id=uuid.uuid4(), **obj_data))
                counts["plan_definitions"] += 1
                print(f"  [insert] plan_definitions -> {data['plan']}")

        # ── point_costs (delete all + re-insert for clean slate) ──
        await db.execute(delete(PointCost))
        for data in POINT_COSTS:
            db.add(PointCost(id=uuid.uuid4(), **data))
            counts["point_costs"] += 1
            print(f"  [insert] point_costs -> {data['action']} ({data['cost']} pts)")

        # ── feature_limits (delete all + re-insert for clean slate) ──
        await db.execute(delete(FeatureLimit))
        for data in FEATURE_LIMITS:
            db.add(FeatureLimit(id=uuid.uuid4(), **data))
            counts["feature_limits"] += 1
            print(f"  [insert] feature_limits -> {data['plan']}/{data['feature_key']}")

        await db.commit()

        print("\nDone.")
        print(f"  plan_definitions : {counts['plan_definitions']} upserted")
        print(f"  point_costs      : {counts['point_costs']} inserted")
        print(f"  feature_limits   : {counts['feature_limits']} inserted")


if __name__ == "__main__":
    asyncio.run(seed())
