"""
Content Safety Pipeline for the AI Assistant.

Four-stage guardrail:
  1. Keyword filter   (sync, regex, ~0 ms)
  2. LLM classifier   (async, Gemini Flash, ~200-400 ms)
     — also assesses grade relevance when student_mode is ON
  3. Grade-level gate  (sync, lookup, ~0 ms)  — soft: injects context, not hard-block
  4. Output check      (sync, regex, ~0 ms)   — post-generation
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("content_guard")

# Cached google-genai client — created once and reused across all guard calls
# so the underlying HTTP session is not torn down between requests.
_guard_gemini_client = None


def _get_guard_client():
    global _guard_gemini_client
    if _guard_gemini_client is None:
        from app.config import settings
        from google import genai as genai_new
        _guard_gemini_client = genai_new.Client(
            api_key=settings.GOOGLE_GEMINI_API_KEY or settings.GEMINI_API_KEY
        )
    return _guard_gemini_client

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class GuardAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REDIRECT = "redirect"


@dataclass
class GuardResult:
    action: GuardAction
    category: str                # educational | harmful | sensitive | non_educational | output_unsafe
    message: Optional[str]       # friendly text returned instead of AI answer (None when ALLOW)
    confidence: float = 1.0
    trigger: Optional[str] = None
    grade_relevance: Optional[str] = None       # at_grade | below_grade | above_grade | way_above_grade
    suggested_grade: Optional[int] = None        # which grade this topic typically belongs to
    grade_context: Optional[str] = None          # extra instruction to inject into system prompt


# ---------------------------------------------------------------------------
# Constants — keyword patterns  (compiled once at module load)
# ---------------------------------------------------------------------------

_FLAGS = re.IGNORECASE

# Stage 1 — harmful patterns (always blocked)
HARMFUL_PATTERNS: list[re.Pattern] = [
    re.compile(p, _FLAGS) for p in [
        # Violence / weapons instructions
        r"\b(?:how\s+to\s+(?:make|build|create|assemble)\s+(?:a\s+)?(?:bomb|explosive|weapon|gun|knife|poison))",
        r"\b(?:instructions?\s+for\s+(?:killing|murder|assault|attack))",
        r"\b(?:how\s+to\s+(?:hurt|harm|injure|kill)\s+(?:someone|myself|a\s+person|people))",
        # Self-harm
        r"\b(?:how\s+to\s+(?:commit\s+)?suicide)",
        r"\b(?:ways?\s+to\s+(?:end\s+my\s+life|die|kill\s+myself))",
        r"\b(?:self[\s-]?harm\s+(?:methods?|techniques?|ways?))",
        # Illegal activities
        r"\b(?:how\s+to\s+(?:hack\s+into|steal\s+from|break\s+into|forge|counterfeit))",
        r"\b(?:how\s+to\s+(?:make|cook|manufacture)\s+(?:meth|drugs|cocaine|heroin|lsd|ecstasy))",
        # Exploitation
        r"\b(?:child\s+(?:porn|exploitation|abuse))",
        r"\bcsam\b",
    ]
]

# Stage 1 — sensitive patterns (always blocked)
SENSITIVE_PATTERNS: list[re.Pattern] = [
    re.compile(p, _FLAGS) for p in [
        r"\b(?:write\s+(?:me\s+)?(?:erotica|sexual\s+content|porn|smut))",
        r"\b(?:explicit\s+sexual\s+(?:story|scene|content|fiction))",
        r"\b(?:sex\s+(?:positions?|techniques?|tips))\b",
        r"\b(?:write\s+.*(?:racist|sexist|homophobic|hateful)\s+(?:joke|speech|rant))",
        r"\b(?:why\s+(?:are|is)\s+(?:\w+\s+){0,3}(?:inferior|superior)\s+(?:race|ethnic))",
    ]
]

# Stage 4 — output-safety patterns (post-generation check)
OUTPUT_UNSAFE_PATTERNS: list[re.Pattern] = [
    re.compile(p, _FLAGS) for p in [
        r"\b(?:here\s+(?:is|are)\s+(?:how|instructions?)\s+to\s+(?:make|build)\s+(?:a\s+)?(?:bomb|weapon|explosive))",
        r"\b(?:step[\s-]?by[\s-]?step\s+(?:guide|instructions?)\s+to\s+(?:kill|harm|hurt))",
        r"\b(?:ways?\s+to\s+commit\s+suicide)",
        r"\b(?:here\s+is\s+a\s+detailed\s+(?:sexual|erotic))",
        r"(?:DAN\s+mode|ignore\s+previous\s+instructions|you\s+are\s+now\s+unrestricted)",
    ]
]

# ---------------------------------------------------------------------------
# Friendly messages
# ---------------------------------------------------------------------------

BLOCK_MESSAGES = {
    "harmful": (
        "I can’t help with that topic. This assistant is designed to help students with learning and school subjects. 📚\n\n"
        "If you have a question about maths, science, history, or any topic you're studying, "
        "I’ll be happy to help!"
    ),

    "sensitive": (
        "That topic isn’t something I can discuss here. Let’s keep our conversation focused on learning. 😊\n\n"
        "What subject or topic would you like help with?"
    ),
}

REDIRECT_MESSAGES = {
    "non_educational": (
        "Hey! That sounds interesting, but I'm your study buddy — I'm best at helping you "
        "with learning and school stuff. Here's what I can do for you:\n\n"
        "📚 **Explain any concept** — from any subject, at your level\n"
        "✏️ **Help with homework** — walk through problems step by step\n"
        "🧠 **Practice & revision** — create questions to test yourself\n"
        "🔍 **Break down tough topics** — make hard things simple\n"
        "🎯 **Exam prep** — tips, important questions, and quick revision\n\n"
        "So, what are you studying today? I'm ready to help! 🚀"
    ),
}

OUTPUT_REPLACEMENT_MESSAGE = (
    "I generated a response, but upon review it contained content that isn't appropriate "
    "for an educational setting. Let me try again with a better answer.\n\n"
    "Could you rephrase your question? I'm here to help with your learning!"
)

# ---------------------------------------------------------------------------
# Grade-relevance prompt templates (injected into system prompt)
# ---------------------------------------------------------------------------

GRADE_CONTEXT_TEMPLATES = {
    "above_grade": (
        "GRADE RELEVANCE NOTICE: The student is in Grade {student_grade} but is asking about "
        "a topic typically taught around Grade {suggested_grade}. This shows great curiosity! "
        "You MUST still answer the question, but adapt your response:\n"
        "- Start by acknowledging their curiosity positively (e.g., 'Great question! This is "
        "something you'll study in more detail in higher grades, but let me break it down for you.')\n"
        "- Simplify the explanation to match their current grade level.\n"
        "- Use analogies and examples a Grade {student_grade} student would relate to.\n"
        "- Skip advanced math/notation they haven't learned yet — use intuitive descriptions instead.\n"
        "- At the end, briefly mention what they'll learn about this topic in future grades."
    ),
    "way_above_grade": (
        "GRADE RELEVANCE NOTICE: The student is in Grade {student_grade} but is asking about "
        "a topic typically introduced around Grade {suggested_grade} or later. "
        "This shows impressive intellectual curiosity! You MUST still answer, but carefully:\n"
        "- Start with warm encouragement: 'Wow, that's an advanced topic! I love that you're curious about it.'\n"
        "- Give a very simplified, age-appropriate overview — think of it as a 'sneak peek'.\n"
        "- Use fun, concrete analogies a Grade {student_grade} student can picture.\n"
        "- Avoid any formulas, notation, or terminology they haven't encountered.\n"
        "- Keep it shorter than usual — a taste of the concept, not the full lesson.\n"
        "- End by connecting it to something they ARE learning now, if possible."
    ),
    "below_grade": (
        "GRADE RELEVANCE NOTICE: The student is in Grade {student_grade} and is asking about "
        "a topic typically covered around Grade {suggested_grade} — so they should already know the basics. "
        "Respond at their actual Grade {student_grade} level:\n"
        "- Don't oversimplify — they've likely learned the fundamentals already.\n"
        "- If it seems like a review question, reinforce key points and add deeper insights.\n"
        "- If they're struggling with a basic topic, be patient and encouraging — "
        "everyone learns at their own pace."
    ),
    "at_grade": None,  # No extra instructions needed — normal grade-calibrated response
}

# ---------------------------------------------------------------------------
# LLM classifier prompt (with grade-relevance assessment)
# ---------------------------------------------------------------------------

CLASSIFIER_PROMPT_BASE = """You are a content safety classifier for an educational platform used by K-12 students (ages 6-17).

Classify the following student query into EXACTLY ONE category:

Categories:
1. "educational" — The query is about academics, learning, school subjects, homework, exam prep, general knowledge, science, history, math, languages, arts, career guidance, study skills, or any topic that has legitimate educational value.
2. "non_educational" — The query is clearly unrelated to education or learning (e.g., casual chat, entertainment, gossip, social media, gaming, dating advice). Note: general knowledge questions ARE educational even if not about a specific school subject.
3. "sensitive" — The query asks for explicit sexual content, promotes hate speech, asks about illegal drug use/manufacture, or requests content that would be inappropriate for a school setting.
4. "harmful" — The query requests instructions for violence, self-harm, suicide, weapons creation, exploitation of minors, or any content that could cause real-world physical harm.

IMPORTANT GUIDELINES:
- Questions about sensitive ACADEMIC topics (reproduction, historical atrocities, chemical reactions, nuclear physics) are "educational" — they are part of legitimate curriculum.
- Questions like "what is alcohol" or "what are drugs" are "educational" (health education). Only classify as "sensitive" if the query asks HOW TO USE/MAKE them recreationally.
- Casual greetings like "hello", "how are you" are "educational" (rapport building is part of learning).
- Creative writing prompts are "educational" unless they explicitly request harmful or sexual content.
- When in doubt, classify as "educational" — false positives are worse than false negatives for student trust.

Query: "{query}"

Respond with ONLY this JSON (no markdown, no explanation):
{{"category": "<one of: educational, non_educational, sensitive, harmful>", "confidence": <float 0.0 to 1.0>, "reason": "<one brief sentence>"}}"""

CLASSIFIER_PROMPT_WITH_GRADE = """You are a content safety classifier AND grade-relevance assessor for an educational platform used by K-12 students.

## Task 1: Safety Classification
Classify the student query into EXACTLY ONE safety category:
1. "educational" — academics, learning, school subjects, homework, general knowledge, or any topic with educational value.
2. "non_educational" — clearly unrelated to education (entertainment, gossip, gaming, dating advice). General knowledge IS educational.
3. "sensitive" — explicit sexual content, hate speech, illegal drug use/manufacture.
4. "harmful" — instructions for violence, self-harm, weapons, exploitation.

GUIDELINES:
- Academic topics about sensitive subjects (reproduction, historical atrocities, nuclear physics) are "educational".
- "What is alcohol/drugs" = educational (health ed). "How to use/make drugs recreationally" = sensitive.
- Greetings like "hello" = educational. Creative writing = educational unless harmful/sexual.
- When in doubt, classify as "educational".

## Task 2: Grade Relevance Assessment
The student is currently in **Grade {grade}** (Indian K-12 curriculum).
Assess whether this query's topic is appropriate for their grade level:
- "at_grade" — the topic is part of or close to Grade {grade} curriculum (within ±1 grade)
- "below_grade" — the topic is typically taught in earlier grades (the student should already know this)
- "above_grade" — the topic is typically taught 2-4 grades higher
- "way_above_grade" — the topic is taught 5+ grades higher or at college level
- "general" — the topic is general knowledge not tied to a specific grade (e.g., "what is rain", greetings)

Also estimate the **suggested_grade**: the grade level where this topic is typically first introduced in Indian curriculum (1-12). For general knowledge, use 0.

Query: "{query}"

Respond with ONLY this JSON (no markdown, no explanation):
{{"category": "<educational|non_educational|sensitive|harmful>", "confidence": <float 0.0-1.0>, "reason": "<brief sentence>", "grade_relevance": "<at_grade|below_grade|above_grade|way_above_grade|general>", "suggested_grade": <int 0-12>}}"""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ContentGuardService:
    """Multi-stage content safety pipeline for the AI assistant."""

    def __init__(self, student_mode: bool = False, grade: int | None = None):
        self.student_mode = student_mode
        self.grade = grade
        self.grade_band = self._get_grade_band(grade) if grade else None

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _get_grade_band(grade: int | None) -> str | None:
        if grade is None:
            return None
        if grade <= 3:
            return "early-primary"
        if grade <= 6:
            return "upper-primary"
        if grade <= 9:
            return "middle-school"
        return "high-school"

    # -- Stage 1: keyword filter -------------------------------------------

    def check_keywords(self, query: str) -> GuardResult:
        """Fast regex scan — catches obviously harmful / sensitive queries."""
        for pat in HARMFUL_PATTERNS:
            m = pat.search(query)
            if m:
                return GuardResult(
                    action=GuardAction.BLOCK,
                    category="harmful",
                    message=BLOCK_MESSAGES["harmful"],
                    trigger=f"keyword: {m.group(0)[:60]}",
                )
        for pat in SENSITIVE_PATTERNS:
            m = pat.search(query)
            if m:
                return GuardResult(
                    action=GuardAction.BLOCK,
                    category="sensitive",
                    message=BLOCK_MESSAGES["sensitive"],
                    trigger=f"keyword: {m.group(0)[:60]}",
                )
        return GuardResult(action=GuardAction.ALLOW, category="educational", message=None)

    # -- Stage 2: LLM classifier + grade relevance -------------------------

    async def classify_query(self, query: str) -> GuardResult:
        """Lightweight Gemini Flash call to classify query intent and grade relevance."""
        from app.config import settings

        if not settings.CONTENT_GUARD_LLM_CLASSIFIER:
            return GuardResult(action=GuardAction.ALLOW, category="educational", message=None)

        # Very short queries (greetings etc.) — skip
        if len(query.strip()) < 3:
            return GuardResult(action=GuardAction.ALLOW, category="educational", message=None)

        try:
            client = _get_guard_client()

            # Use grade-aware prompt when student_mode + grade is set
            if self.student_mode and self.grade:
                prompt = CLASSIFIER_PROMPT_WITH_GRADE.format(
                    query=query[:500], grade=self.grade,
                )
            else:
                prompt = CLASSIFIER_PROMPT_BASE.format(query=query[:500])

            response = await client.aio.models.generate_content(
                model=settings.AI_PRIMARY_MODEL,
                contents=prompt,
            )
            text = response.text.strip()

            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            category = data.get("category", "educational")
            confidence = float(data.get("confidence", 0.5))
            reason = data.get("reason", "")

            # Extract grade relevance info (only present when student_mode + grade)
            grade_relevance = data.get("grade_relevance")
            suggested_grade = data.get("suggested_grade")
            if suggested_grade is not None:
                suggested_grade = int(suggested_grade)

            # Build grade context instruction for the system prompt
            grade_context = None
            if grade_relevance and grade_relevance in GRADE_CONTEXT_TEMPLATES:
                template = GRADE_CONTEXT_TEMPLATES[grade_relevance]
                if template:
                    grade_context = template.format(
                        student_grade=self.grade,
                        suggested_grade=suggested_grade or "higher",
                    )

            # Safety classification decisions
            if category == "harmful":
                return GuardResult(
                    action=GuardAction.BLOCK, category="harmful",
                    message=BLOCK_MESSAGES["harmful"],
                    confidence=confidence,
                    trigger=f"llm_classifier: {reason}",
                )
            if category == "sensitive":
                return GuardResult(
                    action=GuardAction.BLOCK, category="sensitive",
                    message=BLOCK_MESSAGES["sensitive"],
                    confidence=confidence,
                    trigger=f"llm_classifier: {reason}",
                )
            if category == "non_educational":
                if self.student_mode:
                    # Strict: hard redirect back to learning
                    return GuardResult(
                        action=GuardAction.REDIRECT, category="non_educational",
                        message=REDIRECT_MESSAGES["non_educational"],
                        confidence=confidence,
                        trigger=f"llm_classifier: {reason}",
                    )
                # Light mode: still a student platform — soft redirect via
                # system prompt rule 7 (the LLM will gently steer back).
                # We ALLOW so the LLM can respond, but tag it so the system
                # prompt's "STAY EDUCATIONAL" rule handles it naturally.
                return GuardResult(
                    action=GuardAction.ALLOW, category="non_educational",
                    message=None, confidence=confidence,
                )

            # Educational — ALLOW, but carry grade relevance info
            return GuardResult(
                action=GuardAction.ALLOW, category="educational",
                message=None, confidence=confidence,
                grade_relevance=grade_relevance,
                suggested_grade=suggested_grade,
                grade_context=grade_context,
            )

        except Exception as e:
            # FAIL OPEN — never block a legitimate query due to classifier error
            logger.warning("ContentGuard classifier error (failing open): %s", e)
            return GuardResult(action=GuardAction.ALLOW, category="educational", message=None, confidence=0.0)

    # -- Stage 3: output safety check --------------------------------------

    def check_output(self, text: str) -> GuardResult:
        """Lightweight keyword scan on the AI-generated response."""
        for pat in OUTPUT_UNSAFE_PATTERNS:
            m = pat.search(text)
            if m:
                return GuardResult(
                    action=GuardAction.BLOCK,
                    category="output_unsafe",
                    message=OUTPUT_REPLACEMENT_MESSAGE,
                    trigger=f"output_filter: {m.group(0)[:50]}",
                )
        return GuardResult(action=GuardAction.ALLOW, category="educational", message=None)

    # -- Orchestrator ------------------------------------------------------

    async def run_input_pipeline(self, query: str) -> GuardResult:
        """Run stages 1-2 sequentially. Returns first non-ALLOW result.

        If ALLOW, the result may carry grade_relevance / grade_context
        metadata that the caller should inject into the system prompt.
        """
        from app.config import settings

        if not settings.CONTENT_GUARD_ENABLED:
            return GuardResult(action=GuardAction.ALLOW, category="educational", message=None)

        # Stage 1: keyword filter (instant)
        result = self.check_keywords(query)
        if result.action != GuardAction.ALLOW:
            return result

        # Stage 2: LLM classifier + grade relevance (async, ~200-400ms)
        result = await self.classify_query(query)
        return result

    # -- Logging -----------------------------------------------------------

    @staticmethod
    def log_guard_event(query: str, result: GuardResult, user_id: str | None = None):
        """Log non-ALLOW guard decisions for analytics/tuning."""
        if result.action == GuardAction.ALLOW:
            return
        logger.warning(
            "[ContentGuard] %s | user=%s | category=%s | confidence=%.2f | trigger=%s | query=%.200s",
            result.action.value, user_id or "unknown",
            result.category, result.confidence,
            result.trigger, query,
        )
