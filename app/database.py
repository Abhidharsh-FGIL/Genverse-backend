from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData

from app.config import settings

# Naming conventions for Alembic auto-generation
naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=naming_convention)


class Base(DeclarativeBase):
    metadata = metadata


engine = create_async_engine(
    settings.database_url,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def run_migrations() -> None:
    """Apply lightweight column-level migrations for workspace isolation."""
    from sqlalchemy import text

    _fk = "REFERENCES organizations(id) ON DELETE SET NULL"
    migrations = [
        ("practice_assessments",    "org_id", f"ALTER TABLE practice_assessments ADD COLUMN org_id UUID {_fk}",    "CREATE INDEX IF NOT EXISTS idx_practice_assessments_org_id ON practice_assessments(org_id)"),
        ("topic_mastery",           "org_id", f"ALTER TABLE topic_mastery ADD COLUMN org_id UUID {_fk}",           "CREATE INDEX IF NOT EXISTS idx_topic_mastery_org_id ON topic_mastery(org_id)"),
        ("user_library_items",      "org_id", f"ALTER TABLE user_library_items ADD COLUMN org_id UUID {_fk}",      "CREATE INDEX IF NOT EXISTS idx_user_library_items_org_id ON user_library_items(org_id)"),
        ("ebooks",                  "org_id", f"ALTER TABLE ebooks ADD COLUMN org_id UUID {_fk}",                  "CREATE INDEX IF NOT EXISTS idx_ebooks_org_id ON ebooks(org_id)"),
        ("mindmaps",                "org_id", f"ALTER TABLE mindmaps ADD COLUMN org_id UUID {_fk}",                "CREATE INDEX IF NOT EXISTS idx_mindmaps_org_id ON mindmaps(org_id)"),
        ("video_projects",          "org_id", f"ALTER TABLE video_projects ADD COLUMN org_id UUID {_fk}",          "CREATE INDEX IF NOT EXISTS idx_video_projects_org_id ON video_projects(org_id)"),
        ("ai_chats",                "org_id", f"ALTER TABLE ai_chats ADD COLUMN org_id UUID {_fk}",                "CREATE INDEX IF NOT EXISTS idx_ai_chats_org_id ON ai_chats(org_id)"),
        ("user_insights",           "org_id", f"ALTER TABLE user_insights ADD COLUMN org_id UUID {_fk}",           "CREATE INDEX IF NOT EXISTS idx_user_insights_org_id ON user_insights(org_id)"),
        ("insight_articles",        "org_id", f"ALTER TABLE insight_articles ADD COLUMN org_id UUID {_fk}",        "CREATE INDEX IF NOT EXISTS idx_insight_articles_org_id ON insight_articles(org_id)"),
        ("career_guidance_sessions","org_id", f"ALTER TABLE career_guidance_sessions ADD COLUMN org_id UUID {_fk}", "CREATE INDEX IF NOT EXISTS idx_career_guidance_sessions_org_id ON career_guidance_sessions(org_id)"),
    ]

    async with engine.begin() as conn:
        # Create new tables if they don't exist
        ws_gam_exists = await conn.execute(text(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'workspace_gamification'"
        ))
        if not ws_gam_exists.scalar():
            await conn.execute(text("""
                CREATE TABLE workspace_gamification (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                    org_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
                    xp INTEGER NOT NULL DEFAULT 0,
                    streak INTEGER NOT NULL DEFAULT 0,
                    last_activity_date TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
            """))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ws_gam_user ON workspace_gamification(user_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ws_gam_org ON workspace_gamification(org_id)"))

        # Add unique constraint on ai_context_sessions (user_id, workspace_id) to prevent duplicate rows
        uq_exists = await conn.execute(text(
            "SELECT 1 FROM information_schema.table_constraints "
            "WHERE constraint_name = 'uq_ai_context_user_workspace' AND table_name = 'ai_context_sessions'"
        ))
        if not uq_exists.scalar():
            # First, deduplicate any existing rows (keep the most recently updated one)
            await conn.execute(text("""
                DELETE FROM ai_context_sessions a
                USING ai_context_sessions b
                WHERE a.user_id = b.user_id
                  AND a.workspace_id = b.workspace_id
                  AND a.id != b.id
                  AND a.updated_at < b.updated_at
            """))
            await conn.execute(text(
                "ALTER TABLE ai_context_sessions "
                "ADD CONSTRAINT uq_ai_context_user_workspace UNIQUE (user_id, workspace_id)"
            ))

        # Add missing columns to existing tables
        for table, column, alter_sql, index_sql in migrations:
            exists = await conn.execute(text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = :table AND column_name = :column"
            ), {"table": table, "column": column})
            if not exists.scalar():
                await conn.execute(text(alter_sql))
                await conn.execute(text(index_sql))

        # Add max_file_size_mb to plan_definitions if missing
        mfs_exists = await conn.execute(text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'plan_definitions' AND column_name = 'max_file_size_mb'"
        ))
        if not mfs_exists.scalar():
            await conn.execute(text(
                "ALTER TABLE plan_definitions ADD COLUMN max_file_size_mb INTEGER DEFAULT 5"
            ))


async def close_db() -> None:
    await engine.dispose()
