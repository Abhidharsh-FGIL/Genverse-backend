from fastapi import FastAPI

from app.routers import (
    auth,
    users,
    organizations,
    subscriptions,
    classes,
    assignments,
    submissions,
    teacher,
    rubrics,
    lesson_plans,
    assessments,
    evaluation,
    ai_assistant,
    library,
    ebooks,
    video_studio,
    mindmaps,
    playground,
    career,
    past_papers,
    ocr,
    insights,
    gamification,
    group_chats,
    announcements,
    analytics,
    audio,
)

_API = "/api/v1"

_ROUTES = [
    # (router,               path-suffix,        tags)
    (auth.router,            "/auth",             ["Authentication"]),
    (users.router,           "/users",            ["Users"]),
    (organizations.router,   "/organizations",    ["Organizations"]),
    (subscriptions.router,   "/subscriptions",    ["Subscriptions"]),
    (classes.router,         "/classes",          ["Classes"]),
    (assignments.router,     "/assignments",      ["Assignments"]),
    (submissions.router,     "/submissions",      ["Submissions"]),
    (teacher.router,         "/teacher",          ["Teacher"]),
    (rubrics.router,         "/rubrics",          ["Rubrics"]),
    (lesson_plans.router,    "/lesson-plans",     ["Lesson Plans"]),
    (assessments.router,     "/assessments",      ["Practice Assessments"]),
    (evaluation.router,      "/evaluation",       ["Evaluation Hub"]),
    (ai_assistant.router,    "/ai",               ["AI Assistant"]),
    (library.router,         "/library",          ["Knowledge Vault"]),
    (ebooks.router,          "/ebooks",           ["eBooks"]),
    (video_studio.router,    "/video-studio",     ["Video Studio"]),
    (mindmaps.router,        "/mindmaps",         ["Mind Maps"]),
    (playground.router,      "/playground",       ["Playground"]),
    (career.router,          "/career",           ["Career Guidance"]),
    (past_papers.router,     "/past-papers",      ["Past Papers"]),
    (ocr.router,             "/ocr",              ["OCR & Document Extraction"]),
    (insights.router,        "/insights",         ["Insights & Intelligence"]),
    (gamification.router,    "/gamification",     ["Gamification"]),
    (group_chats.router,     "/chats",            ["Group Chats"]),
    (announcements.router,   "/announcements",    ["Announcements"]),
    (analytics.router,       "/analytics",        ["Analytics"]),
    (audio.router,           "/audio",            ["Audio QA"]),
]


def register_routers(app: FastAPI) -> None:
    """Attach every API router to the FastAPI application."""
    for router, suffix, tags in _ROUTES:
        app.include_router(router, prefix=f"{_API}{suffix}", tags=tags)
