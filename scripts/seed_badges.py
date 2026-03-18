"""Seed badge definitions into the database.

Run: python -m scripts.seed_badges
"""
import asyncio
from sqlalchemy import select
from app.database import async_session
from app.models.gamification import Badge

BADGES = [
    # ── Streak badges ──
    {
        "name": "First Step",
        "description": "Log in for the first time",
        "icon": "footprints",
        "category": "streak",
        "rarity": "common",
        "xp_reward": 5,
        "criteria": {"type": "streak", "threshold": 1},
    },
    {
        "name": "On a Roll",
        "description": "Maintain a 3-day login streak",
        "icon": "flame",
        "category": "streak",
        "rarity": "common",
        "xp_reward": 10,
        "criteria": {"type": "streak", "threshold": 3},
    },
    {
        "name": "Week Warrior",
        "description": "Maintain a 7-day login streak",
        "icon": "calendar-check",
        "category": "streak",
        "rarity": "rare",
        "xp_reward": 25,
        "criteria": {"type": "streak", "threshold": 7},
    },
    {
        "name": "Dedicated Learner",
        "description": "Maintain a 30-day login streak",
        "icon": "trophy",
        "category": "streak",
        "rarity": "epic",
        "xp_reward": 100,
        "criteria": {"type": "streak", "threshold": 30},
    },
    # ── XP badges ──
    {
        "name": "Rising Star",
        "description": "Earn your first 50 XP",
        "icon": "star",
        "category": "xp",
        "rarity": "common",
        "xp_reward": 10,
        "criteria": {"type": "xp", "threshold": 50},
    },
    {
        "name": "XP Hunter",
        "description": "Earn 200 XP",
        "icon": "zap",
        "category": "xp",
        "rarity": "rare",
        "xp_reward": 25,
        "criteria": {"type": "xp", "threshold": 200},
    },
    {
        "name": "XP Master",
        "description": "Earn 1000 XP",
        "icon": "crown",
        "category": "xp",
        "rarity": "epic",
        "xp_reward": 50,
        "criteria": {"type": "xp", "threshold": 1000},
    },
    # ── Submission badges ──
    {
        "name": "First Submission",
        "description": "Submit your first assignment",
        "icon": "file-check",
        "category": "submissions",
        "rarity": "common",
        "xp_reward": 10,
        "criteria": {"type": "submissions", "threshold": 1},
    },
    {
        "name": "Prolific Student",
        "description": "Submit 10 assignments",
        "icon": "files",
        "category": "submissions",
        "rarity": "rare",
        "xp_reward": 30,
        "criteria": {"type": "submissions", "threshold": 10},
    },
    {
        "name": "Assignment Machine",
        "description": "Submit 50 assignments",
        "icon": "rocket",
        "category": "submissions",
        "rarity": "epic",
        "xp_reward": 75,
        "criteria": {"type": "submissions", "threshold": 50},
    },
    # ── Score badges ──
    {
        "name": "Perfect Score",
        "description": "Get 100% on any assignment",
        "icon": "check-circle",
        "category": "scores",
        "rarity": "rare",
        "xp_reward": 30,
        "criteria": {"type": "perfect_score", "threshold": 1},
    },
    {
        "name": "High Achiever",
        "description": "Score above 90% on 5 assignments",
        "icon": "award",
        "category": "scores",
        "rarity": "epic",
        "xp_reward": 50,
        "criteria": {"type": "high_scores", "threshold": 5},
    },
    # ── Class badges ──
    {
        "name": "Class Explorer",
        "description": "Join your first class",
        "icon": "book-open",
        "category": "classes",
        "rarity": "common",
        "xp_reward": 5,
        "criteria": {"type": "classes_joined", "threshold": 1},
    },
    {
        "name": "Social Learner",
        "description": "Join 3 classes",
        "icon": "users",
        "category": "classes",
        "rarity": "rare",
        "xp_reward": 20,
        "criteria": {"type": "classes_joined", "threshold": 3},
    },
    # ── Quiz badges ──
    {
        "name": "Quiz Starter",
        "description": "Complete your first quiz",
        "icon": "brain",
        "category": "quizzes",
        "rarity": "common",
        "xp_reward": 10,
        "criteria": {"type": "quizzes_completed", "threshold": 1},
    },
    {
        "name": "Quiz Champion",
        "description": "Complete 10 quizzes",
        "icon": "medal",
        "category": "quizzes",
        "rarity": "rare",
        "xp_reward": 30,
        "criteria": {"type": "quizzes_completed", "threshold": 10},
    },
]


async def seed():
    async with async_session() as db:
        for badge_data in BADGES:
            existing = await db.execute(
                select(Badge).where(Badge.name == badge_data["name"])
            )
            if existing.scalar_one_or_none():
                print(f"  Badge '{badge_data['name']}' already exists, skipping.")
                continue
            db.add(Badge(**badge_data))
            print(f"  + Created badge: {badge_data['name']} ({badge_data['rarity']})")
        await db.commit()
    print(f"\nDone! {len(BADGES)} badge definitions seeded.")


if __name__ == "__main__":
    asyncio.run(seed())
