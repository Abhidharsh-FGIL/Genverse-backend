"""
AI Service - Unified AI routing for all operations.

Model routing (configurable via .env):
  Direct questions  → AI_PRIMARY_MODEL  (Gemini)
  File-based Q&A    → AI_DOCUMENT_MODEL (Claude)
  Fallback          → AI_FALLBACK_MODEL (OpenAI)

API keys:
  GOOGLE_GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
"""
import json
import re
import asyncio
from typing import AsyncIterator, List, Optional, Any, Dict
from pathlib import Path

from app.config import settings

_SENTINEL = object()


def _build_genai_config(gemini_gen_config: dict | None):
    """Convert a legacy gemini_gen_config dict to a google-genai GenerateContentConfig."""
    if not gemini_gen_config:
        return None
    try:
        from google.genai import types as _gt
        kwargs = {}
        if gemini_gen_config.get("max_output_tokens"):
            kwargs["max_output_tokens"] = gemini_gen_config["max_output_tokens"]
        if gemini_gen_config.get("response_mime_type"):
            kwargs["response_mime_type"] = gemini_gen_config["response_mime_type"]
        if gemini_gen_config.get("system_instruction"):
            kwargs["system_instruction"] = gemini_gen_config["system_instruction"]
        return _gt.GenerateContentConfig(**kwargs) if kwargs else None
    except Exception:
        return None


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient Gemini errors that warrant a retry."""
    msg = str(exc)
    return any(k in msg for k in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "quota"))


async def _gemini_generate_with_retry(client, model: str, contents, config=None, max_retries: int = 2):
    """Call client.aio.models.generate_content with exponential-backoff retry on 503/429.

    Retry order: primary model × 2, then backup model × 1, then raise for caller's fallback.
    """
    models_to_try = [model]
    backup = settings.AI_GEMINI_BACKUP_MODEL
    if backup and backup != model:
        models_to_try.append(backup)

    last_exc: Exception | None = None
    for m in models_to_try:
        for attempt in range(max_retries if m == model else 1):
            try:
                return await client.aio.models.generate_content(
                    model=m, contents=contents, config=config,
                )
            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc) and attempt < max_retries - 1:
                    wait = 0.5 * (2 ** attempt)
                    print(f"[Gemini] 503/429 ({m}) attempt {attempt + 1}, retry in {wait}s…", flush=True)
                    await asyncio.sleep(wait)
                    continue
                if _is_retryable(exc):
                    print(f"[Gemini] 503/429 ({m}) exhausted, trying backup model…", flush=True)
                    break
                raise
    raise last_exc  # type: ignore[misc]


class AIService:
    """Unified AI service wrapping Gemini, Anthropic Claude, and OpenAI."""

    def __init__(self):
        self._gemini_client = None
        self._gemini_async_client = None
        self._openai_client = None
        self._anthropic_client = None

    def _get_gemini(self):
        if not self._gemini_client and settings.GOOGLE_GEMINI_API_KEY:
            import google.generativeai as genai
            genai.configure(api_key=settings.GOOGLE_GEMINI_API_KEY)
            self._gemini_client = genai.GenerativeModel(settings.AI_PRIMARY_MODEL)
        return self._gemini_client

    def _get_gemini_async(self):
        """Return the google-genai async client (natively async, zero thread overhead)."""
        if not self._gemini_async_client and settings.GOOGLE_GEMINI_API_KEY:
            from google import genai as genai_new
            self._gemini_async_client = genai_new.Client(api_key=settings.GOOGLE_GEMINI_API_KEY)
        return self._gemini_async_client

    def _get_openai(self):
        if not self._openai_client and settings.OPENAI_API_KEY:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai_client

    def _get_anthropic(self):
        if not self._anthropic_client and settings.ANTHROPIC_API_KEY:
            import anthropic
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        return self._anthropic_client

    async def _ebook_llm_call(self, prompt: str, max_tokens: int = 8192) -> str:
        """Dedicated LLM call for ebook text generation.

        Uses AI_EBOOK_MODEL (default: gpt-4o) via OpenAI with JSON mode.
        Falls back to Gemini if OpenAI is unavailable.
        """
        model = settings.AI_EBOOK_MODEL or "gpt-4o"

        # Try OpenAI first (primary for ebook text)
        openai = self._get_openai()
        if openai:
            try:
                response = await openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    temperature=0.7,
                )
                text = response.choices[0].message.content
                if text:
                    return text
            except Exception as e:
                print(f"[Ebook] OpenAI ({model}) failed: {e}", flush=True)

        # Fallback to Gemini with JSON mode
        gemini = self._get_gemini()
        if gemini:
            try:
                response = gemini.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "response_mime_type": "application/json",
                    },
                )
                return response.text
            except Exception as e:
                print(f"[Ebook] Gemini fallback failed: {e}", flush=True)

        return ""

    def _get_grade_band(self, grade: int | None) -> str:
        """Return a descriptive grade band for calibrating responses."""
        if not grade:
            return ""
        if grade <= 3:
            return "early-primary"
        elif grade <= 6:
            return "upper-primary"
        elif grade <= 9:
            return "middle-school"
        else:
            return "high-school"

    # ------------------------------------------------------------------
    # Grade × Board fused profile builders
    # ------------------------------------------------------------------

    def _build_grade_board_profile(self, grade: int, band: str, board_key: str) -> str:
        """Build a single fused instruction that merges grade AND board so they
        create measurably different outputs.  The board modifies vocabulary
        ceiling, depth, structure, example style, and response length relative
        to the grade baseline."""

        # --- Age-range label ---
        age_map = {
            "early-primary": "ages 6-8",
            "upper-primary": "ages 9-11",
            "middle-school": "ages 12-14",
            "high-school": "ages 15-17",
        }
        age = age_map.get(band, "")

        # --- Per-board OUTPUT RULES that create visible differences ---
        # Each tuple: (vocab_rule, structure_rule, depth_rule, example_rule,
        #              length_rule, engagement_rule, curriculum_note)
        board_rules = {
            "STATE": {
                "label": "State Board",
                "vocab": (
                    "Use ONLY the simplest everyday words. Prefer one-syllable or two-syllable words. "
                    "If a technical term is unavoidable, immediately follow it with '(that means …)' in the simplest possible words. "
                    "Write as if the student thinks in their regional language and reads English as a second language."
                ),
                "structure": (
                    "Use VERY short bullet points — max 8-10 words per bullet. "
                    "One idea per bullet. NO paragraphs. NO long sentences. "
                    "Use numbered lists for steps. Keep total response SHORT (aim for 40-60% the length of a CBSE answer)."
                ),
                "depth": (
                    "Cover ONLY the basic fact or definition. Do NOT go deeper unless asked. "
                    "One example is enough. Skip edge cases, history, or 'why' explanations unless the student asks."
                ),
                "examples": (
                    "Use local, everyday Indian examples: village life, markets, farming, cooking, festivals, family. "
                    "Avoid Western or unfamiliar references. Make examples feel like home."
                ),
                "engagement": (
                    "Keep it warm and encouraging but brief. A simple 'Good question!' is enough. "
                    "Ask ONE simple check question at the end, nothing more."
                ),
                "curriculum": (
                    "Follow State Board textbook patterns. Answers should look like what their textbook says — "
                    "plain, direct, definition-style. State Board rewards brevity and correctness."
                ),
            },
            "CBSE": {
                "label": "CBSE",
                "vocab": (
                    "Use standard textbook English matching NCERT language level. "
                    "Introduce key terms with **bold** and explain them clearly. "
                    "Language should feel like reading an NCERT chapter — neither too simple nor too complex."
                ),
                "structure": (
                    "Structure answers POINT-WISE with key terms **bolded** — this is the CBSE exam-answer format. "
                    "Use numbered points for processes/steps. Use bullet points for lists of features/types. "
                    "Include clear headings or sub-sections for longer answers. Medium-length response."
                ),
                "depth": (
                    "Cover the concept thoroughly as NCERT does: definition + explanation + example. "
                    "Include ONE worked example for Maths/Science. Mention diagram descriptions where relevant. "
                    "Cover the standard scope — not too shallow, not too deep."
                ),
                "examples": (
                    "Use examples from NCERT textbooks when possible. Otherwise use Indian context: "
                    "Indian geography, Indian history, Indian daily life. Mix of rural and urban references."
                ),
                "engagement": (
                    "Be structured and clear. Include a 'Think about it' prompt or a check question. "
                    "For higher grades, include exam tips: 'In board exams, this is often asked as…'"
                ),
                "curriculum": (
                    "Follow NCERT terminology and definitions EXACTLY — students are tested on specific wording. "
                    "Use the same concept ordering as NCERT chapters. For definitions, prefer the textbook phrasing."
                ),
            },
            "ICSE": {
                "label": "ICSE",
                "vocab": (
                    "Use richer vocabulary than CBSE — ICSE students are exposed to more advanced English. "
                    "Introduce technical terms confidently and use them throughout. "
                    "Sentences can be longer and more nuanced. Use academic but accessible language."
                ),
                "structure": (
                    "Use PARAGRAPH-form answers — ICSE expects flowing, well-written responses, not just bullet points. "
                    "Structure with clear topic sentences. Use bullets only for lists of items. "
                    "Answers should be MORE DETAILED than CBSE-style — about 30-50% longer."
                ),
                "depth": (
                    "Go deeper than the basic definition — explain WHY, not just WHAT. "
                    "Include the reasoning behind concepts. Show cause-and-effect chains. "
                    "For Science, include experiment details, observations, and inferences. "
                    "For Maths, show alternative methods when they exist."
                ),
                "examples": (
                    "Use a mix of Indian and international examples. ICSE values broader exposure. "
                    "Include real-world applications: 'This is used in…' or 'Scientists discovered this when…' "
                    "Multiple examples are welcome to show different facets of a concept."
                ),
                "engagement": (
                    "Encourage analytical thinking: 'Why do you think this happens?' "
                    "Pose application questions: 'How would you use this in…?' "
                    "For older students, include comparison questions: 'How is this different from…?'"
                ),
                "curriculum": (
                    "Follow ICSE/ISC syllabus depth. Use textbook approaches from Selina, Concise, Frank publishers. "
                    "ICSE tests understanding over memorisation — answers should demonstrate comprehension."
                ),
            },
            "IB": {
                "label": "IB (International Baccalaureate)",
                "vocab": (
                    "Use sophisticated, globally-aware vocabulary. Even for young students, introduce precise terms "
                    "and build vocabulary actively. Use words like 'investigate', 'explore', 'consider'. "
                    "Language should feel international — no regional bias."
                ),
                "structure": (
                    "Use an INQUIRY-BASED structure: pose a question → explore → discover → reflect. "
                    "Weave questions INTO the explanation, not just at the end. "
                    "Use IB command terms naturally (describe, explain, analyse, evaluate). "
                    "Responses should be the MOST DETAILED of all boards — thoroughness is valued."
                ),
                "depth": (
                    "Go DEEP — explore multiple angles, perspectives, and connections. "
                    "Connect to other subjects (interdisciplinary links). "
                    "Include 'big picture' thinking: how does this connect to global issues, ethics, or other cultures? "
                    "Encourage the student to form their own conclusions."
                ),
                "examples": (
                    "Use GLOBAL, DIVERSE examples from different countries and cultures. "
                    "Include current events, real-world problems, and sustainability connections. "
                    "Avoid culturally narrow examples — IB values international-mindedness."
                ),
                "engagement": (
                    "Make it highly interactive and inquiry-driven. Ask open-ended questions throughout. "
                    "Include 'What do you think?' and 'How might this be different in another country?' "
                    "Connect to TOK (Theory of Knowledge) concepts where natural. "
                    "End with a reflection or exploration prompt."
                ),
                "curriculum": (
                    "Follow IB curriculum philosophy. Reference IB assessment criteria and command terms. "
                    "IB values process over product — show thinking, not just answers."
                ),
            },
            "CAMBRIDGE": {
                "label": "Cambridge (IGCSE/A-Level)",
                "vocab": (
                    "Use precise, formal academic English. Cambridge values exactness in language. "
                    "Technical terms should be used correctly from the start with clear definitions. "
                    "Avoid colloquial language — maintain an academic register throughout."
                ),
                "structure": (
                    "Use a LOGICAL ARGUMENT structure: claim → evidence → reasoning → conclusion. "
                    "Answers should be well-structured with clear progression. "
                    "Use subheadings for longer responses. Include units, significant figures for Science/Maths. "
                    "Response length should be DETAILED — Cambridge rewards thorough, precise answers."
                ),
                "depth": (
                    "Be THOROUGH and PRECISE. Cover definitions, explanations, worked examples, and edge cases. "
                    "For Science: include proper working with units at every step. "
                    "For Maths: show full algebraic working. "
                    "Include exam technique tips: mark allocation awareness, command word interpretation."
                ),
                "examples": (
                    "Use international, scientifically rigorous examples. "
                    "Include real-world applications with proper data/figures where relevant. "
                    "Cambridge values evidence-based reasoning — examples should support arguments."
                ),
                "engagement": (
                    "Be intellectually rigorous but supportive. Pose analytical questions. "
                    "Include 'Examiner tip' callouts for exam-relevant advice. "
                    "For higher grades, include past-paper style practice questions."
                ),
                "curriculum": (
                    "Follow Cambridge International syllabus learning objectives. "
                    "Use Cambridge-prescribed conventions and terminology. "
                    "Cambridge emphasises structured, evidence-based reasoning throughout."
                ),
            },
        }

        rules = board_rules[board_key]

        # --- Grade-specific engagement style ---
        engagement_styles = {
            "early-primary": (
                "TONE: Warm, playful, like telling a short bedtime story to a curious child.\n"
                "- Make it fun and vivid, as if painting a magical picture in their mind.\n"
                "- Start with a fun hook: a tiny story, 'imagine this…', or a surprising fact.\n"
                "- Use everyday objects, animals, food as analogies.\n"
                "- Use 'First… Then… Finally…' patterns for steps.\n"
                "- Celebrate curiosity: 'Great question!' or 'You're thinking like a little scientist!'\n"
                "- Suggest a fun try-at-home activity when the topic allows.\n"
                "RESPONSE LENGTH (strictly enforced):\n"
                "- Short answers → single short paragraph, 8-9 lines max.\n"
                "- Detailed answers → storytelling with humor and emojis, 10-12 lines max. Do NOT exceed 12 lines."
            ),
            "upper-primary": (
                "TONE: Friendly and encouraging, like a cool older sibling who knows stuff.\n"
                "- Open with 'Did you know?' fun facts or relatable scenarios.\n"
                "- Use daily life, sports, nature examples they'd connect with.\n"
                "- Include 'Think about it' moments and quick challenges.\n"
                "- Break complex topics into numbered steps.\n"
                "- End with 'What would happen if…?' scenarios.\n"
                "RESPONSE LENGTH (strictly enforced):\n"
                "- Short answers → concise summary in 4-5 lines. Do NOT exceed 5 lines.\n"
                "- Detailed answers → step-by-step with real-world applications, 10-15 lines max."
            ),
            "middle-school": (
                "TONE: Clear and structured, balancing depth with accessibility.\n"
                "- Start with thought-provoking real-world connections.\n"
                "- Encourage critical thinking: 'What would happen if we changed X?'\n"
                "- Walk through reasoning step-by-step.\n"
                "- Connect concepts across subjects.\n"
                "- Present common misconceptions and ask them to spot the error.\n"
                "RESPONSE LENGTH (strictly enforced):\n"
                "- Short answers → concise summary in 4-5 lines. Do NOT exceed 5 lines.\n"
                "- Detailed answers → structured breakdown with applications, 15-20 lines max."
            ),
            "high-school": (
                "TONE: Precise and academic but still supportive.\n"
                "- Open with a conceptual question or real-world problem.\n"
                "- Use Socratic prompts: 'Think about this before reading the answer…'\n"
                "- Include 'Exam corner' tips and common pitfalls.\n"
                "- Connect to competitive exams (JEE, NEET, CUET) where relevant.\n"
                "- End with a practice problem or discussion question.\n"
                "RESPONSE LENGTH (strictly enforced):\n"
                "- Short answers → concise summary in 4-5 lines with exam relevance. Do NOT exceed 5 lines.\n"
                "- Detailed answers → structured, competitive answer with principles and reasoning, 20-25 lines max."
            ),
        }

        return (
            f"STUDENT PROFILE: Grade {grade} ({age}) — {rules['label']}\n\n"
            f"VOCABULARY RULES (follow strictly):\n{rules['vocab']}\n\n"
            f"RESPONSE STRUCTURE (follow strictly):\n{rules['structure']}\n\n"
            f"DEPTH & DETAIL:\n{rules['depth']}\n\n"
            f"EXAMPLES & ANALOGIES:\n{rules['examples']}\n\n"
            f"INTERACTIVE ENGAGEMENT:\n{rules['engagement']}\n"
            f"{engagement_styles.get(band, '')}\n\n"
            f"CURRICULUM ALIGNMENT:\n{rules['curriculum']}"
        )

    def _build_grade_only_profile(self, grade: int, band: str) -> str:
        """Grade profile when no board is selected — the original behaviour."""
        age_map = {
            "early-primary": "ages 6-8",
            "upper-primary": "ages 9-11",
            "middle-school": "ages 12-14",
            "high-school": "ages 15-17",
        }
        age = age_map.get(band, "")
        profiles = {
            "early-primary": (
                f"The student is in Grade {grade} ({age}).\n"
                "LANGUAGE & TONE:\n"
                "- Use very simple words and short sentences a young child can understand.\n"
                "- Use a warm, playful, encouraging tone — like telling a short bedtime story to a curious child.\n"
                "- Make it fun and vivid, as if painting a magical picture in their mind.\n"
                "- Avoid abstract concepts, technical terms, dialogue format, or any formal teaching tone.\n"
                "RESPONSE LENGTH (strictly enforced):\n"
                "- Short/simple questions → single short paragraph, 8-9 lines max. Keep words simple and imaginative.\n"
                "- Detailed explanations → storytelling-based, back-and-forth between characters, "
                "humor, excitement, emojis. 10-12 lines max. Do NOT exceed 12 lines.\n"
                "INTERACTIVE ENGAGEMENT (use these naturally, not all at once):\n"
                "- Start explanations with a fun hook: a tiny story, a 'imagine this…' scenario, or a surprising fact.\n"
                "- Use everyday objects, animals, colours, and food as analogies.\n"
                "- Sprinkle in mini-challenges: 'Can you guess what happens next?' or 'Let's count together!'\n"
                "- After explaining, ask a simple check question.\n"
                "- Celebrate their curiosity: 'What a great question!'\n"
                "- Use patterns like 'First… Then… Finally…' to make steps easy to follow.\n"
                "- If the topic allows, suggest a fun activity: 'Try this at home: …'"
            ),
            "upper-primary": (
                f"The student is in Grade {grade} ({age}).\n"
                "LANGUAGE & TONE:\n"
                "- Use clear, simple language with age-appropriate vocabulary.\n"
                "- Introduce technical terms but always explain them in plain words right after.\n"
                "- Keep explanations structured but friendly — not too formal.\n"
                "RESPONSE LENGTH (strictly enforced):\n"
                "- Short/simple questions → concise summary in 4-5 lines with structured explanations "
                "and real-world applications. Do NOT exceed 5 lines for short answers.\n"
                "- Detailed explanations → step-by-step breakdown with real-world applications, "
                "keep within 10-15 lines.\n"
                "INTERACTIVE ENGAGEMENT (use these naturally, not all at once):\n"
                "- Open with a 'Did you know?' fun fact or a relatable scenario.\n"
                "- Use examples from daily life, school, sports, nature.\n"
                "- Include 'Think about it' moments and quick challenges.\n"
                "- Break complex topics into numbered steps with clear transitions.\n"
                "- End with a thought-provoking 'What would happen if…?' scenario."
            ),
            "middle-school": (
                f"The student is in Grade {grade} ({age}).\n"
                "LANGUAGE & TONE:\n"
                "- Use moderately detailed explanations with proper terminology.\n"
                "- Define key terms when first introduced, then use them naturally.\n"
                "- Be clear and structured, balancing depth with accessibility.\n"
                "RESPONSE LENGTH (strictly enforced):\n"
                "- Short/simple questions → concise summary in 4-5 lines with structured explanations "
                "and real-world applications. Do NOT exceed 5 lines for short answers.\n"
                "- Detailed explanations → detailed step-by-step breakdown with real-world applications, "
                "keep within 15-20 lines.\n"
                "INTERACTIVE ENGAGEMENT (use these naturally, not all at once):\n"
                "- Start with a thought-provoking question or real-world connection.\n"
                "- Encourage critical thinking: 'What do you think would happen if we changed X?'\n"
                "- Walk through logic step-by-step.\n"
                "- Connect concepts across subjects.\n"
                "- Include quick self-check questions."
            ),
            "high-school": (
                f"The student is in Grade {grade} ({age}).\n"
                "LANGUAGE & TONE:\n"
                "- Use precise academic language appropriate for senior students.\n"
                "- Students are preparing for board exams and competitive entrances.\n"
                "- Be thorough, cover edge cases, include exam-relevant tips.\n"
                "RESPONSE LENGTH (strictly enforced):\n"
                "- Short/simple questions → concise summary in 4-5 lines with technical concepts "
                "and exam relevance. Do NOT exceed 5 lines for short answers.\n"
                "- Detailed explanations → structured, competitive answer with principles and reasoning, "
                "keep within 20-25 lines.\n"
                "INTERACTIVE ENGAGEMENT (use these naturally, not all at once):\n"
                "- Open with a conceptual question or real-world problem.\n"
                "- Use Socratic-style prompts.\n"
                "- Include 'Exam corner' tips and 'Common pitfall' callouts.\n"
                "- Connect to competitive exam relevance (JEE, NEET, CUET).\n"
                "- End with a practice problem or discussion question."
            ),
        }
        return profiles.get(band, f"The student is in Grade {grade}.")

    def _build_board_only_profile(self, board_key: str) -> str:
        """Board profile when no grade is selected."""
        board_profiles = {
            "STATE": (
                "BOARD: State Board\n"
                "Use the simplest possible language. Short sentences, basic vocabulary. "
                "Bullet points over paragraphs. Local everyday examples. "
                "Keep answers brief and direct — State Board rewards correctness and brevity."
            ),
            "CBSE": (
                "BOARD: CBSE\n"
                "Follow NCERT terminology and conventions. Structure answers point-wise with **bold** key terms. "
                "Use textbook-standard English. Include worked examples for Maths/Science."
            ),
            "ICSE": (
                "BOARD: ICSE\n"
                "Use richer vocabulary and paragraph-form answers. Explain 'why' not just 'what'. "
                "Go deeper than surface definitions. Include real-world applications."
            ),
            "IB": (
                "BOARD: IB\n"
                "Use inquiry-based approach. Sophisticated vocabulary, global examples, multiple perspectives. "
                "Encourage the student to form their own conclusions. Reference IB command terms."
            ),
            "CAMBRIDGE": (
                "BOARD: Cambridge\n"
                "Use precise formal academic English. Structure: claim -> evidence -> reasoning -> conclusion. "
                "Include full working with units. Exam technique tips where relevant."
            ),
        }
        return board_profiles.get(board_key, f"Follow {board_key} curriculum standards.")

    def _build_context_prompt(self, context: dict | None) -> str:
        if not context:
            return ""

        grade = context.get("grade")
        board = context.get("board")
        subject = context.get("subject")
        language = context.get("language")
        student_mode = context.get("student_mode")

        parts = []

        # --- Combined grade × board profile ---
        # Grade and board are merged into ONE instruction block so they interact
        # properly. The board modifies vocabulary ceiling, response depth, structure,
        # and example style *relative to* the grade baseline — creating visible
        # differences even at the same grade level.

        band = self._get_grade_band(grade) if grade else None
        board_key = None
        if board:
            board_upper = board.upper()
            for k in ("CBSE", "ICSE", "STATE", "IB", "CAMBRIDGE"):
                if k in board_upper:
                    board_key = k
                    break

        if grade and board_key:
            # Build a single fused instruction
            parts.append(self._build_grade_board_profile(grade, band, board_key))
        elif grade:
            # Grade only (no board selected) — use default grade profile
            parts.append(self._build_grade_only_profile(grade, band))
        elif board_key:
            # Board only (no grade) — just board instructions
            parts.append(self._build_board_only_profile(board_key))
        elif board:
            parts.append(f"Follow {board} curriculum standards and conventions.")

        if subject:
            parts.append(f"Subject focus: {subject}. Keep your response relevant to this subject area.")
        if language:
            # Resolve language code (e.g. "hi", "ta") to full name ("Hindi", "Tamil")
            lang_name = self._LANGUAGE_NAMES.get(language.lower(), language)
            _base_name = lang_name.split("(")[0].strip().lower()
            if _base_name != "english":
                script = self._SCRIPT_MAP.get(_base_name, lang_name)
                parts.append(
                    f"RESPONSE LANGUAGE: {lang_name}. Write your ENTIRE response using ONLY {script} script. "
                    f"Do NOT mix characters from other Indian/foreign scripts. "
                    f"For example, if writing in Tamil, never use Devanagari (Hindi) characters — use only Tamil script. "
                    f"Do NOT put English translations in parentheses like '(Gravity)' or '(Mass)' — "
                    f"write the term directly in {lang_name} or use the English term as-is without parenthetical translation. "
                    f"Do NOT use emojis or special Unicode symbols (🌟, ⚡, 🍔, etc.) in the response. "
                    f"SCRIPT EXCEPTIONS (keep these in their standard form — do NOT transliterate or translate): "
                    f"LaTeX math ($...$, $$...$$), chemical equations ($\\ce{{...}}$), code blocks, "
                    f"programming keywords, variable names, and standard scientific symbols. "
                    f"SPELLING & GRAMMAR ACCURACY: Use correct {lang_name} orthography throughout. "
                    f"Every word must use proper character combinations, vowel signs, and conjuncts for {lang_name}. "
                    f"Do NOT produce transliterated English written in {lang_name} script — use genuine {lang_name} vocabulary."
                )
            else:
                # Explicit English instruction — prevents the model from continuing
                # in a non-English language carried over from prior conversation turns
                # or from topic-based associations (e.g. Indian history topics).
                parts.append(
                    "RESPONSE LANGUAGE: English. You MUST respond entirely in English. "
                    "Do NOT use Hindi, Tamil, or any other language in your response — "
                    "even if the topic relates to Indian history, culture, or a non-English subject, "
                    "and even if earlier messages in this conversation were in another language."
                )

        return "\n".join(parts) if parts else ""

    def _build_settings_prompt(self, chat_settings: dict | None) -> str:
        """Build a system prompt section from per-chat AI settings."""
        if not chat_settings:
            return ""
        parts = []

        # Personality — how the AI should behave
        personality_map = {
            "mentor": (
                "Act as a supportive mentor. Guide with encouragement and wisdom. "
                "Ask thought-provoking questions to help the student discover answers themselves. "
                "Celebrate effort and progress, not just correct answers. "
                "When correcting mistakes, be gentle — frame it as learning."
            ),
            "coach": (
                "Act as a focused, results-driven coach. Be direct and action-oriented. "
                "Push the student to think critically — don't give answers immediately, "
                "guide them with leading questions. Set clear expectations and challenge them to improve. "
                "Praise effort but always raise the bar."
            ),
            "tutor": (
                "Act as a patient, methodical tutor. Break down every concept into clear steps. "
                "After each explanation, verify understanding with a quick check question. "
                "If the student seems confused, try a different approach or analogy. "
                "Never rush — understanding matters more than coverage."
            ),
            "friend": (
                "Act as a knowledgeable, relatable friend. Explain things casually, "
                "using everyday language, pop culture references, and humour where appropriate. "
                "Avoid sounding like a textbook — be conversational and approachable."
            ),
            "professor": (
                "Act as an authoritative professor. Deliver comprehensive, academically rigorous responses. "
                "Use precise terminology and formal structure. Cite concepts, theories, and frameworks. "
                "Maintain an intellectual but respectful tone."
            ),
            "technical-expert": (
                "Act as a senior technical expert. Use precise, industry-standard terminology. "
                "Dive into implementation details, cover edge cases, and reference best practices. "
                "Assume the student wants depth, not surface-level explanations."
            ),
            "helpful": "Be helpful, clear, and informative in your responses.",
        }
        personality = chat_settings.get("personality", "helpful")
        parts.append(personality_map.get(personality, personality_map["helpful"]))

        # Difficulty — how deep and complex the response should be
        difficulty_map = {
            "easy": (
                "Difficulty: EASY. Use the simplest possible language. "
                "No jargon, no technical terms unless you explain them immediately. "
                "Prefer everyday analogies and short sentences. "
                "If explaining a formula, show it step-by-step with numbers, not just symbols."
            ),
            "medium": (
                "Difficulty: MEDIUM. Use clear explanations with moderate technical depth. "
                "Define key terms when first introduced. Balance simplicity with accuracy. "
                "Include one or two examples to reinforce understanding."
            ),
            "hard": (
                "Difficulty: HARD. Use advanced concepts and proper technical terminology. "
                "Do not over-explain basics — the student knows the foundations. "
                "Focus on deeper understanding, exceptions, and interconnections between topics."
            ),
            "expert": (
                "Difficulty: EXPERT. Assume expert-level knowledge. "
                "Skip introductory definitions entirely. Focus on nuance, edge cases, "
                "trade-offs, proofs, derivations, and cutting-edge aspects. "
                "Use precise notation and formal reasoning."
            ),
        }
        difficulty = chat_settings.get("difficulty", "medium")
        parts.append(difficulty_map.get(difficulty, difficulty_map["medium"]))

        # Content length — with explicit line limits to prevent over-generation
        length_map = {
            "small": "LENGTH: Be extremely brief — 1-3 sentences maximum (4-5 lines). Just the core answer, nothing extra.",
            "brief": "LENGTH: Keep it concise — 1-2 short paragraphs (6-8 lines max). Get to the point fast.",
            "summary": "LENGTH: Give a focused summary — 2-3 paragraphs (10-12 lines max) covering key points without excessive detail.",
            "medium": "LENGTH: Provide a moderately detailed response (12-15 lines) covering key points with some examples.",
            "detailed": "LENGTH: Provide a comprehensive, in-depth response (18-25 lines) with thorough explanations, examples, and structure.",
            "deep-dive": (
                "LENGTH: Provide an exhaustive deep-dive (30+ lines allowed). Cover every important aspect, "
                "edge case, and example. Use headings and sub-sections to organise."
            ),
        }
        content_length = chat_settings.get("content_length", "medium")
        parts.append(length_map.get(content_length, length_map["medium"]))

        if chat_settings.get("explain_3ways"):
            parts.append(
                "IMPORTANT — Explain in 3 Ways (this is SEPARATE from the content length rule above):\n"
                "First, answer the user's question normally following the LENGTH rule above.\n"
                "Then, AFTER your main answer, add a divider line (---) and provide these three clearly labelled sections:\n"
                "**Analogy:** A simple, relatable analogy or metaphor that makes the concept easy to grasp.\n"
                "**Technical:** A precise, formal definition or technical explanation.\n"
                "**Real-World Example:** A concrete, real-world application or example of the concept in action.\n"
                "Each of these three sections should always be 3-5 lines regardless of the content length setting. "
                "The content length setting applies ONLY to the main answer above the divider, NOT to the 3-way explanation."
            )
        else:
            parts.append(
                "IMPORTANT: Do NOT use the 'Explain in 3 Ways' format. Do NOT include Analogy/Technical/Real-World Example sections. "
                "Even if previous messages in this conversation used that format, it is now TURNED OFF. "
                "Give a direct, normal response without splitting into those three sections."
            )

        if chat_settings.get("examples"):
            parts.append("Always include concrete, real-world examples to illustrate every concept you explain.")
        else:
            parts.append(
                "Do NOT add extra examples unless the user specifically asks for them. "
                "Even if previous messages included examples, the examples setting is currently OFF."
            )

        output_mode = chat_settings.get("output_mode", "text")
        if output_mode == "structured":
            parts.append("Structure your response with clear headings (##) and logical sections.")
        elif output_mode == "bullets":
            parts.append("Present information primarily using bullet points and numbered lists.")
        else:
            parts.append("Use normal flowing text format. Do NOT force headings or bullet-point-only structure unless it naturally fits.")

        if chat_settings.get("student_mode"):
            parts.append(
                "STUDENT MODE is ON. You are talking directly to a student, not a professional or teacher.\n"
                "CONTENT GUARDRAILS (strictly enforced in student mode):\n"
                "- You are an EDUCATIONAL assistant ONLY. Always answer questions that are relevant to learning, "
                "academics, school subjects, homework, general knowledge, or personal development.\n"
                "- If a student asks about something clearly non-educational (entertainment gossip, dating, social media drama), "
                "gently redirect them: 'That's interesting! But as your study buddy, I'm best at helping with learning. "
                "What subject are you working on?'\n"
                "- NEVER provide content involving: explicit violence, self-harm, sexual content, drug use instructions, "
                "weapons creation, or anything harmful to a minor.\n"
                "- If a student asks about a sensitive topic that IS part of curriculum (e.g., reproduction in biology, "
                "wars in history, chemical hazards in chemistry), answer it factually and age-appropriately for their grade. "
                "These are legitimate academic topics — do not refuse them.\n"
                "- Do NOT provide personal opinions on politics, religion, or controversial social issues. "
                "Present multiple perspectives factually and let the student form their own views.\n"
                "- If unsure whether a topic is appropriate, err on the side of answering educationally — "
                "curious students should never feel shamed for asking questions.\n"
                "CORE RULES:\n"
                "- Use encouraging, supportive language throughout — be the teacher every student wishes they had.\n"
                "- When correcting mistakes, be gentle: 'Almost! Let's look at it this way…' or 'Good thinking! Just one small thing…'\n"
                "- When they get something right, celebrate it: 'Exactly right!' or 'You nailed it!'\n"
                "- Never be condescending or patronising.\n"
                "MAKE IT INTERACTIVE:\n"
                "- Don't just lecture — engage. Weave in questions, prompts, and challenges naturally throughout.\n"
                "- Use 'Let's think about this together…' to walk through reasoning collaboratively.\n"
                "- After explaining a concept, include a quick check: 'Does that make sense? Here's a quick one to test it: …'\n"
                "- Use scaffolded hints when they're stuck: give a nudge first, then more detail if needed.\n"
                "- Relate the topic to things students care about — school life, future careers, or everyday situations.\n"
                "- When appropriate, present the concept as a mini-story or scenario before the formal explanation.\n"
                "- End responses with something forward-looking: a question to ponder, a challenge to try, or a 'next time you see X, notice how…' moment.\n"
                "- Keep the energy positive and curious — learning should feel like discovery, not a chore."
            )

        # Note: followup, next_steps, and practice are handled as separate UI cards
        # after the response — do NOT include them inside the response body.

        return "\n".join(parts) if parts else ""

    async def chat(
        self, messages: List[dict], context: dict | None = None,
        chat_settings: dict | None = None, has_files: bool = False,
        max_output_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Non-streaming chat with AI.

        Routing:
          - has_files=True  → OpenAI (better at document Q&A), Gemini fallback
          - has_files=False → Gemini Pro (direct questions), OpenAI fallback
        """
        # --- Content Guard: pre-generation check ---
        # Only run for user-facing chat calls (context/chat_settings present).
        # Internal service calls (mindmap, infographic, lesson plan, etc.) pass
        # context=None & chat_settings=None — skip the guard for those.
        grade_context_instruction = None
        guard = None
        is_user_chat = context is not None or chat_settings is not None

        if is_user_chat:
            from app.services.content_guard import ContentGuardService, GuardAction

            student_mode = bool(
                (chat_settings or {}).get("student_mode")
                or (context or {}).get("student_mode")
            )
            grade = (context or {}).get("grade")
            guard = ContentGuardService(student_mode=student_mode, grade=grade)

            user_query = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_query = m.get("content", "")
                    break

            if user_query:
                guard_result = await guard.run_input_pipeline(user_query)
                guard.log_guard_event(user_query, guard_result)
                if guard_result.action != GuardAction.ALLOW:
                    return guard_result.message
                grade_context_instruction = guard_result.grade_context
        # --- End pre-generation check ---

        context_str = self._build_context_prompt(context)
        settings_str = self._build_settings_prompt(chat_settings)
        system_prompt = (
            "You are Genverse.ai, an AI-powered educational assistant built for students. "
            "You are part of a learning platform used by school students from Grade 1 to Grade 12. "
            "Every response you give should be helpful, age-appropriate, and focused on learning.\n\n"
            "IMPORTANT RULES — follow these strictly:\n"
            "1. NEVER mention your training data cutoff, knowledge cutoff date, or any date-based limitation "
            "(e.g. never say 'As of late 2023', 'As of my last update', 'my training data goes up to', "
            "'I don't have information after', etc.). Present information confidently without disclaimers "
            "about when you were trained. If you are unsure about something, say so without referencing dates.\n"
            "2. Math: use $...$ for inline math and $$...$$ for display/block math. "
            "Never use \\(...\\) or \\[...\\] notation. "
            "CRITICAL — NEVER wrap a sentence or example that contains math in backtick code spans. "
            "BAD: `If $f(x) = x^3$, then $f'(x) = 3x^2$.` "
            "GOOD: If $f(x) = x^3$, then $f'(x) = 3x^2$. "
            "Backticks are ONLY for computer code, terminal commands, or variable names — never for math examples.\n"
            "3. Chemical equations: use $\\ce{...}$ notation (e.g. $\\ce{H2O}$, $\\ce{2H2 + O2 -> 2H2O}$).\n"
            "4. Tables: use standard Markdown pipe table syntax (| col | col | with a header separator row).\n"
            "5. Lists, headings, bold, italic, code blocks: use standard Markdown syntax.\n"
            "6. STUDENT SAFETY — this platform is used by minors (ages 6-17). You MUST:\n"
            "   - NEVER provide instructions for violence, self-harm, weapons, illegal activities, or explicit sexual content.\n"
            "   - NEVER use profanity, slang that is inappropriate for minors, or sexually suggestive language.\n"
            "   - If asked about harmful topics, politely decline and redirect to educational alternatives.\n"
            "   - For sensitive curriculum topics (biology reproduction, history of wars, chemical hazards), "
            "answer factually and age-appropriately — these are valid educational topics.\n"
            "   - Do NOT provide personal opinions on politics, religion, or controversial social issues. "
            "Present multiple perspectives factually.\n"
            "7. STAY EDUCATIONAL — You are a study buddy. Your purpose is to help students learn. "
            "If a query is clearly non-educational (gossip, dating, social media drama), "
            "briefly acknowledge it and gently steer back: 'I'm best at helping with learning! What subject are you working on?'\n"
            "8. PROPORTIONAL RESPONSES — Always match your response length and depth to the input:\n"
            "   - Greetings, small talk, thank you, ok, sure, etc. → 1-2 sentences max. "
            "Example: 'Hey! I'm Genverse, your study buddy. What would you like to learn today?'\n"
            "   - Simple/short questions → Brief, focused answer (1 short paragraph).\n"
            "   - Substantive academic questions → Detailed, structured response with examples.\n"
            "   NEVER pad short inputs with unsolicited examples, subject lists, exam tips, or motivational filler. "
            "If the student hasn't asked a specific question, just warmly invite them to ask one. Be concise by default.\n"
            "9. NEVER mention the student's grade level or curriculum board in your response. "
            "Do NOT say things like 'As a Grade 5 student…', 'For your grade level…', "
            "'Since you are in Grade 8…', 'According to CBSE…', 'In the ICSE curriculum…', "
            "'Based on your board…'. Just answer naturally — the grade and board context shapes "
            "your tone, depth, and curriculum focus internally, but must never appear in the output."
        )
        if context_str:
            system_prompt += f"\n{context_str}"
        if settings_str:
            system_prompt += f"\n\n{settings_str}"
        if grade_context_instruction:
            system_prompt += f"\n\n{grade_context_instruction}"

        _conv = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        _lang_tag = ""
        _rlang = (context or {}).get("language")
        if _rlang:
            _rlang_name = self._LANGUAGE_NAMES.get(_rlang.lower(), _rlang)
            if _rlang_name.lower() != "english":
                _lang_tag = (
                    f"\n\n[SYSTEM: Respond entirely in {_rlang_name}. "
                    f"Use ONLY {_rlang_name} script — do NOT mix other scripts like Devanagari in Tamil or Tamil in Hindi. "
                    f"CRITICAL EXCEPTIONS — keep these exactly as-is in standard notation: "
                    f"LaTeX math ($...$, $$...$$), chemical equations ($\\ce{{...}}$), code blocks, and scientific symbols. "
                    f"Use correct {_rlang_name} orthography — no spelling mistakes, no transliterated English.]"
                )
            else:
                _lang_tag = (
                    "\n\n[SYSTEM: Respond entirely in English. "
                    "Do NOT use Hindi, Tamil, or any other language — "
                    "regardless of the topic or previous conversation language.]"
                )
        full_prompt = system_prompt + "\n\n" + _conv + _lang_tag

        response_text: str | None = None

        # Build Gemini generation_config when custom options are requested
        gemini_gen_config = {}
        if max_output_tokens:
            gemini_gen_config["max_output_tokens"] = max_output_tokens
        if json_mode:
            gemini_gen_config["response_mime_type"] = "application/json"
        gemini_gen_config = gemini_gen_config or None

        if has_files:
            # File-based queries → Anthropic Claude primary, OpenAI fallback, Gemini fallback
            try:
                anthropic = self._get_anthropic()
                if anthropic and response_text is None:
                    claude_messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in messages if m["role"] != "system"
                    ]
                    if not claude_messages or claude_messages[0]["role"] != "user":
                        claude_messages.insert(0, {"role": "user", "content": "Hello"})
                    response = await anthropic.messages.create(
                        model=settings.AI_DOCUMENT_MODEL,
                        max_tokens=max_output_tokens or 4096,
                        system=system_prompt,
                        messages=claude_messages,
                    )
                    response_text = response.content[0].text
            except Exception as e:
                print(f"[AIService] Anthropic chat failed: {e}", flush=True)
            try:
                openai = self._get_openai()
                if openai and response_text is None:
                    openai_kwargs: dict = {
                        "model": settings.AI_FALLBACK_MODEL,
                        "messages": [{"role": "system", "content": system_prompt}] + messages,
                        "max_tokens": max_output_tokens or 4096,
                    }
                    if json_mode:
                        openai_kwargs["response_format"] = {"type": "json_object"}
                    response = await openai.chat.completions.create(**openai_kwargs)
                    response_text = response.choices[0].message.content
            except Exception as e:
                print(f"[AIService] OpenAI chat fallback failed: {e}", flush=True)
            try:
                _gc = self._get_gemini_async()
                if _gc and response_text is None:
                    _cfg = _build_genai_config(gemini_gen_config)
                    _r = await _gemini_generate_with_retry(_gc, settings.AI_PRIMARY_MODEL, full_prompt, _cfg)
                    response_text = _r.text
            except Exception as e:
                print(f"[AIService] Gemini chat fallback failed: {e}", flush=True)
        else:
            # Direct questions → Gemini async primary, OpenAI fallback
            try:
                _gc = self._get_gemini_async()
                if _gc and response_text is None:
                    _cfg = _build_genai_config(gemini_gen_config)
                    _r = await _gemini_generate_with_retry(_gc, settings.AI_PRIMARY_MODEL, full_prompt, _cfg)
                    response_text = _r.text
            except Exception as e:
                print(f"[AIService] Gemini chat failed: {e}", flush=True)
            try:
                openai = self._get_openai()
                if openai and response_text is None:
                    openai_kwargs2: dict = {
                        "model": settings.AI_FALLBACK_MODEL,
                        "messages": [{"role": "system", "content": system_prompt}] + messages,
                    }
                    if json_mode:
                        openai_kwargs2["response_format"] = {"type": "json_object"}
                    response = await openai.chat.completions.create(**openai_kwargs2)
                    response_text = response.choices[0].message.content
            except Exception as e:
                print(f"[AIService] OpenAI chat fallback failed: {e}", flush=True)

        if response_text is None:
            return "AI service is not configured or all providers failed. Please check your API keys."

        # --- Content Guard: post-generation output check ---
        if guard is not None:
            from app.services.content_guard import GuardAction
            output_check = guard.check_output(response_text)
            if output_check.action != GuardAction.ALLOW:
                guard.log_guard_event(user_query, output_check)
                return output_check.message
        # --- End output check ---

        return response_text

    async def _stream_gemini(self, full_prompt: str) -> AsyncIterator[str]:
        """Stream from Gemini using the native async client (zero thread overhead).

        Two optimisations vs. vanilla Gemini streaming:
        1. thinking_budget=0 — disables gemini-2.5-flash's extended reasoning mode.
           Without this the model silently "thinks" for 3-7 s before the first token
           appears, giving GPT-4o-mini a huge perceived-speed advantage.
        2. Chunk sub-division — Gemini natively returns 50-200 char blobs while
           OpenAI streams ~10-30 chars. We split each blob into ≤15-char word-boundary
           pieces and yield control (asyncio.sleep(0)) between them so uvicorn flushes
           each SSE event before the next arrives, giving the user a smooth
           per-word typewriter effect instead of whole-paragraph jumps.

        Retries the initial connection up to 2 times on 503/429 before raising
        so the caller can fall through to the next provider.
        """
        client = self._get_gemini_async()
        if not client:
            return

        primary = settings.AI_PRIMARY_MODEL
        backup = settings.AI_GEMINI_BACKUP_MODEL
        models_to_try = [primary] + ([backup] if backup and backup != primary else [])

        # Build a config that disables extended thinking (reduces TTFT from ~7s → ~1s).
        # Gracefully skip if the SDK version doesn't expose ThinkingConfig yet.
        _stream_config = None
        try:
            from google.genai import types as _gt
            _stream_config = _gt.GenerateContentConfig(
                thinking_config=_gt.ThinkingConfig(thinking_budget=0)
            )
        except Exception:
            pass  # older SDK or model doesn't support it — fall through with no config

        last_exc: Exception | None = None
        for model_name in models_to_try:
            for attempt in range(2):
                try:
                    # generate_content_stream is an async def (coroutine) that returns
                    # an AsyncIterator — it MUST be awaited before iterating.
                    stream = await client.aio.models.generate_content_stream(
                        model=model_name,
                        contents=full_prompt,
                        config=_stream_config,
                    )
                    async for chunk in stream:
                        try:
                            text = chunk.text
                            if not text:
                                continue
                            # Sub-divide large chunks into ≤15-char word-boundary pieces
                            # so SSE events look like per-token output (smooth typewriter).
                            # asyncio.sleep(0) hands control back to the event loop so
                            # uvicorn can flush each piece as a separate HTTP chunk.
                            words = text.split(' ')
                            buf = ''
                            for j, word in enumerate(words):
                                buf += ('' if j == 0 else ' ') + word
                                if len(buf) >= 15 or j == len(words) - 1:
                                    yield buf
                                    buf = ''
                                    await asyncio.sleep(0)
                        except (ValueError, AttributeError):
                            continue
                    return  # stream completed successfully
                except Exception as exc:
                    last_exc = exc
                    if _is_retryable(exc) and attempt == 0:
                        wait = 0.5
                        print(f"[Gemini Stream] 503/429 ({model_name}), retrying in {wait}s…", flush=True)
                        await asyncio.sleep(wait)
                        continue
                    if _is_retryable(exc):
                        print(f"[Gemini Stream] 503/429 ({model_name}) exhausted → trying backup", flush=True)
                        break  # next model
                    raise  # non-retryable error → let _prefetch fall to OpenAI
        if last_exc:
            raise last_exc

    async def _stream_openai(self, system_prompt: str, messages: List[dict]) -> AsyncIterator[str]:
        """Stream from OpenAI (natively async)."""
        openai = self._get_openai()
        if not openai:
            return

        stream = await openai.chat.completions.create(
            model=settings.AI_FALLBACK_MODEL,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def _stream_anthropic(self, system_prompt: str, messages: List[dict]) -> AsyncIterator[str]:
        """Stream from Anthropic Claude (natively async)."""
        client = self._get_anthropic()
        if not client:
            return

        # Convert messages: Anthropic expects role=user/assistant only, system is separate
        claude_messages = []
        for m in messages:
            role = m["role"]
            if role == "system":
                continue  # system goes in the system parameter
            claude_messages.append({"role": role, "content": m["content"]})

        # Ensure messages alternate user/assistant — merge consecutive same-role
        if not claude_messages or claude_messages[0]["role"] != "user":
            claude_messages.insert(0, {"role": "user", "content": "Hello"})

        async with client.messages.stream(
            model=settings.AI_DOCUMENT_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=claude_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    # ------------------------------------------------------------------
    # Document summarization (context-aware, map-reduce when needed)
    # ------------------------------------------------------------------

    # ~200 words/chunk × 1.3 tokens/word ≈ 260 tokens per chunk.
    # Single-pass limits are kept LOW so the final streaming call has
    # small context → fast time-to-first-token.  Map-reduce batch limits
    # are LARGE so fewer concurrent calls are needed.
    _TOKENS_PER_CHUNK = 260
    _SINGLE_PASS_LIMITS = {          # max chunks for single-pass (no map-reduce)
        "gemini":  150,              # ~39K tokens — fast TTFT in final call
        "claude":  150,              # same — keeps final call responsive
        "openai":  150,              # same
    }
    _MAP_BATCH_LIMITS = {            # max chunks per MAP batch
        "gemini":  500,              # 500 chunks ≈ 130K tokens — Gemini handles easily
        "claude":  400,
        "openai":  300,
    }

    def _best_available_provider(self) -> str:
        """Return the best available provider name for summarization."""
        if self._get_gemini():
            return "gemini"
        if self._get_anthropic():
            return "claude"
        if self._get_openai():
            return "openai"
        return "gemini"  # fallback default

    async def _llm_call(self, prompt: str, max_tokens: int = 2048, tag: str = "") -> str:
        """Single LLM call with Gemini → Claude → OpenAI fallback chain.

        Gemini is tried first because it has the largest context window
        and is fastest for large-document workloads.
        """
        # 1. Gemini (1M context, fastest)
        gemini = self._get_gemini()
        if gemini:
            try:
                resp = await asyncio.to_thread(gemini.generate_content, prompt)
                return resp.text or ""
            except Exception as e:
                print(f"[Summary] Gemini {tag} failed: {e}", flush=True)
        # 2. Claude
        client = self._get_anthropic()
        if client:
            try:
                resp = await client.messages.create(
                    model=settings.AI_DOCUMENT_MODEL,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            except Exception as e:
                print(f"[Summary] Claude {tag} failed: {e}", flush=True)
        # 3. OpenAI
        openai = self._get_openai()
        if openai:
            try:
                resp = await openai.chat.completions.create(
                    model=settings.AI_FALLBACK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                print(f"[Summary] OpenAI {tag} failed: {e}", flush=True)
        return ""

    async def summarize_document_chunks(
        self,
        chunks: list[str],
        user_question: str,
    ) -> str:
        """Summarize a document's chunks with context-aware strategy.

        1. Small docs (fits in single LLM call) → return raw text, let the
           final chat call handle it.  Zero extra API calls.
        2. Medium docs → single-pass summary via one LLM call.
        3. Large docs → MAP-REDUCE: batch chunks, summarize each batch
           concurrently, then combine.
        """
        provider = self._best_available_provider()
        single_limit = self._SINGLE_PASS_LIMITS.get(provider, 550)
        batch_limit = self._MAP_BATCH_LIMITS.get(provider, 400)
        n = len(chunks)

        print(
            f"[Summary] {n} chunks, provider={provider}, "
            f"single_limit={single_limit}, batch_limit={batch_limit}",
            flush=True,
        )

        # --- SINGLE PASS: fits in the final chat call's context ---
        if n <= single_limit:
            print(f"[Summary] Single-pass: returning {n} raw chunks for final chat call", flush=True)
            return "\n\n".join(chunks)

        # --- MAP PHASE: split into batches & summarize concurrently ---
        batches: list[list[str]] = []
        for i in range(0, n, batch_limit):
            batches.append(chunks[i : i + batch_limit])

        print(f"[Summary] Map-reduce: {n} chunks → {len(batches)} batches of ~{batch_limit}", flush=True)

        batch_prompts = []
        for idx, batch in enumerate(batches):
            batch_text = "\n\n".join(batch)
            batch_prompts.append(
                f"You are creating a detailed summary of PART {idx + 1} of {len(batches)} "
                "from an educational document. Your summary will be combined with "
                "summaries of other parts to create a complete document summary.\n\n"
                "--- DOCUMENT SECTION ---\n"
                f"{batch_text}\n"
                "--- END ---\n\n"
                "INSTRUCTIONS:\n"
                "- Summarize EVERY chapter, section, or topic present in this section.\n"
                "- Use the document's own headings/chapter titles as your headings.\n"
                "- Include ALL key definitions, formulas, theorems, equations, and important facts verbatim.\n"
                "- Preserve specific details: names, numbers, dates, examples, diagrams described.\n"
                "- For each topic, explain the core concept — not just mention it.\n"
                "- Use bullet points under each heading for key points.\n"
                "- Do NOT skip any section or chapter — cover everything.\n"
                "- Do NOT add any information that is not in the text.\n"
                "- Be thorough — this summary is the ONLY thing the reader will see."
            )

        # Use higher token budget so batch summaries retain detail
        partial_summaries = await asyncio.gather(
            *(self._llm_call(p, max_tokens=4096, tag=f"map-{i+1}/{len(batches)}")
              for i, p in enumerate(batch_prompts))
        )

        partial_summaries = [s for s in partial_summaries if s.strip()]
        if not partial_summaries:
            print("[Summary] All MAP batches failed — returning first batch raw", flush=True)
            return "\n\n".join(chunks[:batch_limit])

        combined = "\n\n---\n\n".join(partial_summaries)

        # --- REDUCE PHASE (only if combined partials are very large) ---
        combined_word_count = len(combined.split())
        if combined_word_count > 15_000:
            print(f"[Summary] Reduce pass: {combined_word_count} words in partials", flush=True)
            reduce_prompt = (
                "Below are detailed summaries of consecutive sections of a document, "
                "presented in order. Merge them into ONE unified, comprehensive summary.\n\n"
                f"{combined}\n\n"
                "INSTRUCTIONS:\n"
                "- Combine all section summaries into a single coherent document summary.\n"
                "- Preserve ALL chapter/section headings from the partial summaries.\n"
                "- Keep all key definitions, formulas, and specific details.\n"
                "- Remove redundancy between sections but do NOT drop any unique content.\n"
                "- Maintain the original document's order and flow.\n\n"
                f"User's original request: {user_question}\n\n"
                "Produce the final unified summary:"
            )
            result = await self._llm_call(reduce_prompt, max_tokens=8192, tag="reduce")
            if result.strip():
                return result

        return combined

    async def stream_chat(
        self, messages: List[dict], context: dict | None = None,
        chat_settings: dict | None = None, has_files: bool = False,
    ) -> AsyncIterator[str]:
        """SSE streaming chat with AI.

        Guard check and provider stream start concurrently so that by the
        time the guard LLM call finishes (~300ms), the first tokens from
        the provider are already queued. TTFB = max(guard, provider_ttfb)
        instead of guard + provider_ttfb.
        """
        from app.services.content_guard import ContentGuardService, GuardAction

        student_mode = bool(
            (chat_settings or {}).get("student_mode")
            or (context or {}).get("student_mode")
        )
        grade = (context or {}).get("grade")
        guard = ContentGuardService(student_mode=student_mode, grade=grade)

        user_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_query = m.get("content", "")
                break

        # Build system prompt immediately (sync) — grade_context is intentionally
        # omitted here so the provider stream can start before the guard finishes.
        context_str = self._build_context_prompt(context)
        settings_str = self._build_settings_prompt(chat_settings)
        system_prompt = (
            "You are Genverse.ai, an AI-powered educational assistant built for students. "
            "You are part of a learning platform used by school students from Grade 1 to Grade 12. "
            "Every response you give should be helpful, age-appropriate, and focused on learning.\n\n"
            "IMPORTANT RULES — follow these strictly:\n"
            "1. NEVER mention your training data cutoff, knowledge cutoff date, or any date-based limitation "
            "(e.g. never say 'As of late 2023', 'As of my last update', 'my training data goes up to', "
            "'I don't have information after', etc.). Present information confidently without disclaimers "
            "about when you were trained. If you are unsure about something, say so without referencing dates.\n"
            "2. Math: use $...$ for inline math and $$...$$ for display/block math. "
            "Never use \\(...\\) or \\[...\\] notation. "
            "CRITICAL — NEVER wrap a sentence or example that contains math in backtick code spans. "
            "BAD: `If $f(x) = x^3$, then $f'(x) = 3x^2$.` "
            "GOOD: If $f(x) = x^3$, then $f'(x) = 3x^2$. "
            "Backticks are ONLY for computer code, terminal commands, or variable names — never for math examples.\n"
            "3. Chemical equations: use $\\ce{...}$ notation (e.g. $\\ce{H2O}$, $\\ce{2H2 + O2 -> 2H2O}$).\n"
            "4. Tables: use standard Markdown pipe table syntax (| col | col | with a header separator row).\n"
            "5. Lists, headings, bold, italic, code blocks: use standard Markdown syntax.\n"
            "6. STUDENT SAFETY — this platform is used by minors (ages 6-17). You MUST:\n"
            "   - NEVER provide instructions for violence, self-harm, weapons, illegal activities, or explicit sexual content.\n"
            "   - NEVER use profanity, slang that is inappropriate for minors, or sexually suggestive language.\n"
            "   - If asked about harmful topics, politely decline and redirect to educational alternatives.\n"
            "   - For sensitive curriculum topics (biology reproduction, history of wars, chemical hazards), "
            "answer factually and age-appropriately — these are valid educational topics.\n"
            "   - Do NOT provide personal opinions on politics, religion, or controversial social issues. "
            "Present multiple perspectives factually.\n"
            "7. STAY EDUCATIONAL — You are a study buddy. Your purpose is to help students learn. "
            "If a query is clearly non-educational (gossip, dating, social media drama), "
            "briefly acknowledge it and gently steer back: 'I'm best at helping with learning! What subject are you working on?'\n"
            "8. PROPORTIONAL RESPONSES — Always match your response length and depth to the input:\n"
            "   - Greetings, small talk, thank you, ok, sure, etc. → 1-2 sentences max. "
            "Example: 'Hey! I'm Genverse, your study buddy. What would you like to learn today?'\n"
            "   - Simple/short questions → Brief, focused answer (1 short paragraph).\n"
            "   - Substantive academic questions → Detailed, structured response with examples.\n"
            "   NEVER pad short inputs with unsolicited examples, subject lists, exam tips, or motivational filler. "
            "If the student hasn't asked a specific question, just warmly invite them to ask one. Be concise by default.\n"
            "9. NEVER mention the student's grade level or curriculum board in your response. "
            "Do NOT say things like 'As a Grade 5 student…', 'For your grade level…', "
            "'Since you are in Grade 8…', 'According to CBSE…', 'In the ICSE curriculum…', "
            "'Based on your board…'. Just answer naturally — the grade and board context shapes "
            "your tone, depth, and curriculum focus internally, but must never appear in the output."
        )
        if context_str:
            system_prompt += f"\n{context_str}"
        if settings_str:
            system_prompt += f"\n\n{settings_str}"

        _conv = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        _lang_tag = ""
        _rlang = (context or {}).get("language")
        if _rlang:
            _rlang_name = self._LANGUAGE_NAMES.get(_rlang.lower(), _rlang)
            if _rlang_name.lower() != "english":
                _lang_tag = (
                    f"\n\n[SYSTEM: Respond entirely in {_rlang_name}. "
                    f"Use ONLY {_rlang_name} script — do NOT mix other scripts like Devanagari in Tamil or Tamil in Hindi. "
                    f"CRITICAL EXCEPTIONS — keep these exactly as-is in standard notation: "
                    f"LaTeX math ($...$, $$...$$), chemical equations ($\\ce{{...}}$), code blocks, and scientific symbols. "
                    f"Use correct {_rlang_name} orthography — no spelling mistakes, no transliterated English.]"
                )
            else:
                _lang_tag = (
                    "\n\n[SYSTEM: Respond entirely in English. "
                    "Do NOT use Hindi, Tamil, or any other language — "
                    "regardless of the topic or previous conversation language.]"
                )
        full_prompt = system_prompt + "\n\n" + _conv + _lang_tag

        # --- Concurrent guard + provider startup ---
        # Guard runs as a background task while the provider stream is already fetching.
        guard_task = (
            asyncio.create_task(guard.run_input_pipeline(user_query))
            if user_query else None
        )

        _queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        _stream_error: list = []

        async def _prefetch():
            try:
                if has_files:
                    try:
                        client = self._get_anthropic()
                        if client:
                            print(f"[MODEL] Anthropic (document Q&A): {settings.AI_DOCUMENT_MODEL}", flush=True)
                            async for tok in self._stream_anthropic(system_prompt, messages):
                                await _queue.put(tok)
                            return
                    except Exception as e:
                        print(f"[AIService] Anthropic stream failed: {e}", flush=True)
                    try:
                        client = self._get_openai()
                        if client:
                            print(f"[MODEL] OpenAI (document Q&A fallback): {settings.AI_FALLBACK_MODEL}", flush=True)
                            async for tok in self._stream_openai(system_prompt, messages):
                                await _queue.put(tok)
                            return
                    except Exception as e:
                        print(f"[AIService] OpenAI stream fallback failed: {e}", flush=True)
                    try:
                        if self._get_gemini_async():
                            print(f"[MODEL] Gemini (document Q&A fallback): {settings.AI_PRIMARY_MODEL}", flush=True)
                            async for tok in self._stream_gemini(full_prompt):
                                await _queue.put(tok)
                            return
                    except Exception as e:
                        print(f"[AIService] Gemini stream fallback failed: {e}", flush=True)
                else:
                    try:
                        if self._get_gemini_async():
                            print(f"[MODEL] Gemini (direct Q&A): {settings.AI_PRIMARY_MODEL}", flush=True)
                            async for tok in self._stream_gemini(full_prompt):
                                await _queue.put(tok)
                            return
                    except Exception as e:
                        print(f"[AIService] Gemini stream failed: {e}", flush=True)
                    try:
                        client = self._get_openai()
                        if client:
                            print(f"[MODEL] OpenAI (direct Q&A fallback): {settings.AI_FALLBACK_MODEL}", flush=True)
                            async for tok in self._stream_openai(system_prompt, messages):
                                await _queue.put(tok)
                            return
                    except Exception as e:
                        print(f"[AIService] OpenAI fallback failed: {e}", flush=True)
                await _queue.put("AI service is temporarily unavailable. Please try again in a moment.")
            except asyncio.CancelledError:
                pass
            except Exception as e:
                _stream_error.append(e)
            finally:
                await _queue.put(None)  # sentinel — always signal completion

        stream_task = asyncio.create_task(_prefetch())

        # Await guard (stream has been running concurrently for its duration).
        # Hard timeout of 500ms: if the guard LLM call hasn't responded by then,
        # fail open so a slow/unavailable classifier never stalls the stream.
        if guard_task:
            try:
                guard_result = await asyncio.wait_for(guard_task, timeout=0.5)
            except (asyncio.TimeoutError, Exception):
                guard_result = None  # fail open — never block legitimate queries
            if guard_result and guard_result.action != GuardAction.ALLOW:
                guard.log_guard_event(user_query, guard_result)
                stream_task.cancel()
                try:
                    await asyncio.shield(stream_task)
                except (asyncio.CancelledError, Exception):
                    pass
                yield guard_result.message
                return
            if guard_result:
                guard.log_guard_event(user_query, guard_result)

        # Drain the queue — tokens were already buffering while guard ran
        accumulated = ""
        next_check_at = 500
        try:
            while True:
                token = await _queue.get()
                if token is None:
                    break
                if _stream_error:
                    raise _stream_error[0]
                accumulated += token
                yield token

                if len(accumulated) >= next_check_at:
                    next_check_at += 500
                    output_check = guard.check_output(accumulated)
                    if output_check.action != GuardAction.ALLOW:
                        guard.log_guard_event(user_query, output_check)
                        stream_task.cancel()
                        yield "\n\n" + output_check.message
                        return
        finally:
            if not stream_task.done():
                stream_task.cancel()
                try:
                    await asyncio.shield(stream_task)
                except (asyncio.CancelledError, Exception):
                    pass

        # Final output check on complete text
        output_check = guard.check_output(accumulated)
        if output_check.action != GuardAction.ALLOW:
            guard.log_guard_event(user_query, output_check)
            yield "\n\n" + output_check.message

    async def ask_document(self, query: str, context: str, ai_context: dict | None = None) -> str:
        """RAG query against extracted document text."""
        _DOC_CONTEXT_LIMIT = 60000
        prompt = f"""You are a helpful study assistant. The user has uploaded a file and is asking questions about it.

RULES — you MUST follow these without exception:
1. Answer ONLY using information explicitly present in the file content below.
2. Do NOT add any facts, explanations, or details from your general knowledge.
3. If the answer is not found in the file, respond: "This information is not available in the uploaded file."
4. Do NOT speculate, infer beyond what the text states, or fill in gaps with outside knowledge.
5. When quoting or referencing, stay faithful to the file's wording.
6. If the question is partially answerable, answer only the part supported by the file and clearly state what is not covered.
7. Respond naturally and directly — do NOT say "based on the document", "according to the context", "from the excerpts", or similar phrases. Just answer as if you read the file.

FILE CONTENT:
---
{context[:_DOC_CONTEXT_LIMIT]}
---

User's question: {query}

Answer:"""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, ai_context)

    @staticmethod
    def _distribute_questions(total: int, weights: dict) -> dict:
        """Distribute `total` questions proportionally by percentage weights."""
        if not weights:
            return {}
        total_weight = sum(weights.values()) or 1
        items = list(weights.items())
        counts: dict = {}
        allocated = 0
        for key, w in items[:-1]:
            c = max(1, round(total * w / total_weight))
            counts[key] = c
            allocated += c
        # Last item absorbs any rounding remainder
        last_key = items[-1][0]
        counts[last_key] = max(1, total - allocated)
        return counts

    @staticmethod
    def _fix_json_escapes(s: str) -> str:
        """Repair invalid JSON backslash escapes produced by LLM-generated LaTeX.
        JSON only allows: \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX
        LaTeX commands like \\frac, \\text, \\alpha inside JSON strings need doubling.
        """
        out = []
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch == '\\' and i + 1 < n:
                nxt = s[i + 1]
                if nxt == '\\':
                    out.append('\\\\')
                    i += 2
                elif nxt == 'u':
                    if i + 5 < n and all(c in '0123456789abcdefABCDEF' for c in s[i+2:i+6]):
                        out.append(s[i:i+6])
                        i += 6
                    else:
                        out.append('\\\\')
                        i += 1
                elif nxt in ('"', '/'):
                    out.append(s[i:i+2])
                    i += 2
                elif nxt in ('t', 'b', 'f', 'n', 'r'):
                    # Could be a valid JSON escape OR the start of a LaTeX command
                    # (\text, \beta, \frac, \nu, \rho). If followed by a letter → LaTeX.
                    if i + 2 < n and s[i + 2].isalpha():
                        out.append('\\\\')
                        i += 1
                    else:
                        out.append(s[i:i+2])
                        i += 2
                else:
                    out.append('\\\\')
                    i += 1
            else:
                out.append(ch)
                i += 1
        return ''.join(out)

    async def generate_practice_assessment(
        self,
        subject: str,
        topics: List[str] | None,
        grade: int | None,
        board: str | None,
        difficulty: str,
        question_count: int,
        question_types: List[str] | None,
        mode: str,
        blooms_level: str = "mixed",
        mcq_subtypes: List[str] | None = None,
        type_weightage: dict | None = None,
        topic_weightage: dict | None = None,
        negative_marking: bool = False,
        source_text: str | None = None,
        language: str | None = None,
    ) -> List[dict]:
        """Generate practice assessment questions as JSON — respects all config options."""
        print(f"[Assessment] language={language}, grade={grade}, board={board}, subject={subject}, subtypes={mcq_subtypes}, question_types={question_types}", flush=True)
        types = question_types or ["mcq"]

        # ── Compute exact counts per question type ──────────────────────────
        if type_weightage and len(types) > 1:
            filtered_weights = {t: type_weightage.get(t, 0) for t in types}
            type_counts = self._distribute_questions(question_count, filtered_weights)
        else:
            type_counts = {types[0]: question_count} if len(types) == 1 else {
                t: max(1, question_count // len(types)) for t in types
            }
            # Fix rounding on last item
            diff = question_count - sum(type_counts.values())
            if diff:
                type_counts[types[-1]] = type_counts.get(types[-1], 1) + diff

        # ── MCQ subtype distribution ────────────────────────────────────────
        subtypes = mcq_subtypes or ["standard"]
        mcq_count = type_counts.get("mcq", 0)
        mcq_subtype_counts: dict = {}
        if mcq_count > 0:
            # Equal distribution across all requested subtypes
            base_per_subtype = mcq_count // len(subtypes)
            remainder = mcq_count % len(subtypes)
            for i, s in enumerate(subtypes):
                mcq_subtype_counts[s] = base_per_subtype + (1 if i < remainder else 0)

        # ── Build distribution section for the prompt ───────────────────────
        type_labels = {
            "mcq": "MCQ", "fill": "Fill in the Blank", "short": "Short Answer",
            "long": "Long Answer", "true_false": "True / False", "match": "Match the Following",
        }
        subtype_labels = {
            "standard": "Standard MCQ", "case": "Case-based MCQ",
            "assertion_reason": "Assertion-Reason MCQ", "higher_order": "Higher Order Thinking MCQ",
        }
        dist_lines = []
        for t, cnt in type_counts.items():
            label = type_labels.get(t, t)
            dist_lines.append(f"  - {label}: {cnt} question(s)")
            if t == "mcq" and mcq_subtype_counts:
                for s, sc in mcq_subtype_counts.items():
                    dist_lines.append(f"      • {subtype_labels.get(s, s)}: EXACTLY {sc} question(s)")

        # ── Topic / chapter distribution ────────────────────────────────────
        topics_str = ", ".join(topics) if topics else subject
        topic_section = ""
        if topic_weightage and topics and len(topics) > 1:
            t_counts = self._distribute_questions(question_count, {
                t: topic_weightage.get(t, 0) for t in topics
            })
            topic_section = "\nTOPIC DISTRIBUTION (spread questions across topics as shown):\n" + \
                "\n".join(f"  - {t}: {c} question(s)" for t, c in t_counts.items())

        # ── Source instruction ───────────────────────────────────────────────
        _SOURCE_CHAR_LIMIT = 60000  # ~15k tokens — fits well within Gemini/GPT context
        if source_text and source_text.strip():
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Assessment prompt: using source_text (%d chars, limit %d)",
                len(source_text), _SOURCE_CHAR_LIMIT,
            )
            truncated = source_text[:_SOURCE_CHAR_LIMIT]
            source_section = (
                "⚠️ SOURCE-ONLY MODE — CRITICAL REQUIREMENT:\n"
                "You MUST generate ALL questions STRICTLY and EXCLUSIVELY from the source text below.\n"
                "RULES:\n"
                "- Every question must be directly answerable from the source text.\n"
                "- Every correct answer must be explicitly stated or directly derivable from the source text.\n"
                "- Every explanation must reference content from the source text.\n"
                "- Do NOT add any fact, concept, or detail that is not present in the source text.\n"
                "- Do NOT use your general knowledge — treat the source text as the ONLY truth.\n"
                "- Cover different sections/topics from the source text to ensure variety.\n\n"
                f"SOURCE TEXT:\n---\n{truncated}\n---"
            )
            if len(source_text) > _SOURCE_CHAR_LIMIT:
                source_section += f"\n(Source truncated from {len(source_text)} to {_SOURCE_CHAR_LIMIT} chars)"
        else:
            source_section = f"Generate questions based on your educational knowledge of: {topics_str}"

        # ── Bloom's taxonomy instruction ────────────────────────────────────
        blooms_map = {
            "remember": "Recall / recognition of facts",
            "understand": "Interpretation and explanation of concepts",
            "apply": "Use of knowledge in new practical situations",
            "analyze": "Break down information, find patterns and relationships",
            "evaluate": "Justify decisions, critique, judge quality",
            "create": "Design, produce, or construct new ideas",
        }
        if blooms_level and blooms_level != "mixed":
            blooms_section = f"BLOOM'S LEVEL: All questions must target '{blooms_level.capitalize()}' — {blooms_map.get(blooms_level, '')}."
        else:
            blooms_section = "BLOOM'S LEVEL: Use a balanced mix across Remember, Understand, Apply, and higher levels."

        # ── Negative marking note ────────────────────────────────────────────
        neg_section = (
            "NEGATIVE MARKING: This is a negative-marking assessment. Every question MUST have "
            "one clearly unambiguous correct answer with no trick or confusable options."
            if negative_marking else ""
        )

        allowed_types_str = " | ".join(f'"{t}"' for t in types)

        # ── Language instruction ─────────────────────────────────────────────
        lang_code = (language or "en").strip().lower()
        lang_name = self._LANGUAGE_NAMES.get(lang_code, language or "English")
        is_non_english = lang_code not in ("en", "english", "")
        if is_non_english:
            lang_section = (
                f"\nLANGUAGE: You MUST write ALL question text, options, explanations, "
                f"and correct answers ENTIRELY in {lang_name}. "
                f"Use the native script of {lang_name} throughout — do NOT mix in English "
                f"words or translations unless they are technical terms with no equivalent. "
                f"The JSON keys (\"id\", \"type\", \"text\", etc.) must remain in English."
            )
        else:
            lang_section = ""

        # ── Build MCQ subtype constraint for prompt ──────────────────────────
        mcq_subtype_constraint = ""
        if mcq_subtype_counts and len(mcq_subtype_counts) > 0:
            allowed_subtypes_str = " | ".join(f'"{s}"' for s in mcq_subtype_counts.keys())
            mcq_subtype_constraint = f"""
MCQ SUBTYPE DISTRIBUTION — THIS IS MANDATORY, NOT OPTIONAL:
You MUST distribute MCQ questions across subtypes EXACTLY as specified below.
The "subtype" field for every MCQ question MUST be one of: {allowed_subtypes_str}
""" + "\n".join(
                f"  - \"{s}\" subtype: EXACTLY {sc} MCQ question(s)"
                for s, sc in mcq_subtype_counts.items()
            )

        prompt = f"""You are an expert question paper setter. Generate exactly {question_count} questions for a {mode} assessment.

SUBJECT: {subject or topics_str}
TOPICS: {topics_str}
GRADE: {f'Grade {grade}' if grade else 'General'}{f' ({board})' if board else ''}
DIFFICULTY: {difficulty}
MODE: {mode}{lang_section}
{blooms_section}
{neg_section}

ALLOWED QUESTION TYPES — STRICTLY: {allowed_types_str}
You MUST NOT generate any question with a "type" outside this list. Every single question must use only these types.

EXACT QUESTION DISTRIBUTION (generate exactly this many of each type — no more, no less, no substitutions):
{chr(10).join(dist_lines)}
{mcq_subtype_constraint}
{topic_section}

{source_section}

QUESTION FORMAT RULES — follow exactly:
1. MCQ (standard): 4 distinct options as a list. Exactly one correct.
   "subtype": "standard"
   "options": ["option1", "option2", "option3", "option4"]
   "correct_answer": the exact correct option string.

2. MCQ (case): Include a brief scenario/passage (2-4 sentences) in "text" above the question.
   Then ask a question about it. 4 options as above.
   "subtype": "case"

3. MCQ (assertion_reason): Two statements.
   "subtype": "assertion_reason"
   "text": "Assertion (A): [statement A]\\nReason (R): [statement R]\\nChoose the correct option:"
   "options": [
     "Both A and R are true, and R is the correct explanation of A",
     "Both A and R are true, but R is not the correct explanation of A",
     "A is true but R is false",
     "A is false but R is true"
   ]
   "correct_answer": one of those four strings exactly.

4. MCQ (higher_order): Requires analysis, application, or evaluation — NOT simple recall.
   Scenario-based or multi-step reasoning. 4 options.
   "subtype": "higher_order"

5. Fill in the Blank: "text" has ___ for the missing word/phrase.
   "correct_answer": the exact word/phrase that fills the blank. "options": null.

6. Short Answer: Question needing 2-4 sentence answer.
   "correct_answer": concise model answer. "options": null.

7. Long Answer: Descriptive/essay question.
   "correct_answer": key points and model answer outline. "options": null.

8. True / False: A clear factual statement.
   "options": ["True", "False"]. "correct_answer": "True" or "False".

9. Match the Following: Two columns to match.
   "text": "Match the items in Column A with Column B."
   "options": right-column items as an array (e.g. ["Paris", "Berlin", "Tokyo", "Cairo"])
   "pairs": [{{"left": "Capital of France", "right": "Paris"}}, ...]
   "correct_answer": matching in "A-1, B-2, C-3" notation.

Return a JSON array of EXACTLY {question_count} objects. Each object MUST have ALL these fields:
- "id": "q1", "q2", "q3" ... (sequential, no gaps)
- "type": MUST be one of the ALLOWED TYPES ONLY: {allowed_types_str}
- "subtype": for MCQ — MUST match the exact subtype distribution above; for all others — null
- "text": the full question text (string)
- "options": array of strings for mcq/true_false/match; null for fill/short/long
- "pairs": array of {{"left":..., "right":...}} objects for match; null for all others
- "correct_answer": string (required for all types)
- "explanation": 1-2 sentence explanation of why the answer is correct
- "marks": 1 for mcq/fill/true_false; 2 for short/match; 4 for long
- "blooms_level": one of "remember" | "understand" | "apply" | "analyze" | "evaluate" | "create"

MATH NOTATION — CRITICAL:
1. Every mathematical expression, formula, matrix, fraction, trig function, Greek letter, or operator MUST be wrapped in LaTeX math delimiters: $...$ for inline (e.g. $1 + \\tan^2\\theta = \\sec^2\\theta$, $\\frac{{1}}{{2}}$) or $$...$$ for display/block math (e.g. $$\\begin{{pmatrix}}1&2\\\\3&4\\end{{pmatrix}}$$).
2. NEVER output bare LaTeX commands (\\frac, \\begin, \\sqrt, \\tan, \\sin, \\theta, etc.) outside $ or $$ delimiters.
3. NEVER use Unicode math characters (², ³, θ, α, β, π, ∑, →, −, ·, ∞, etc.) in option values or formulas — use LaTeX inside $...$ instead (e.g. write $\\theta$ not θ, write $x^2$ not x², write $\\tan^2\\theta - 1 = \\sec^2\\theta$ not tan²θ−1=sec²θ).
4. Plain English words in question text do NOT need delimiters; ONLY mathematical expressions get $...$.

⚠️ FINAL CHECKS BEFORE OUTPUT:
1. Verify every "type" field is one of {allowed_types_str}.
2. Verify every MCQ "subtype" field matches the required distribution above.
3. If source text was provided, verify every question is answerable ONLY from that text — remove any question that requires outside knowledge.
{f"""
⚠️ CRITICAL LANGUAGE REQUIREMENT: Every single "text", "options" array element, "correct_answer", and "explanation" value MUST be written in {lang_name} using its native script. NOT in English. If any value is in English, rewrite it in {lang_name} before returning. Only JSON keys remain in English.""" if is_non_english else ""}
Return ONLY the raw JSON array. No markdown fences, no explanation text outside the array."""

        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()

            # Extract JSON from markdown code fences
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("["):
                        cleaned = part
                        break

            # Fallback: find JSON array anywhere in the response
            if not cleaned.startswith("["):
                import re as _re
                match = _re.search(r'\[[\s\S]*\]', cleaned)
                if match:
                    cleaned = match.group(0)

            try:
                result = json.loads(cleaned, strict=False)
            except json.JSONDecodeError:
                cleaned = AIService._fix_json_escapes(cleaned)
                result = json.loads(cleaned, strict=False)

            if isinstance(result, list):
                return result
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).exception(
                "[generate_practice_assessment] Failed to parse LLM response"
            )
        return []

    async def auto_evaluate_attempt(self, questions: List[dict], responses: dict) -> dict:
        """Auto-score an assessment attempt."""
        if not questions:
            return {"score": 0, "max_score": 0, "percentage": 0, "feedback": {}}

        score = 0
        max_score = 0
        feedback = {}
        for q in questions:
            q_id = str(q.get("id", ""))
            # Support both camelCase (stored format) and snake_case (AI generation format)
            marks = q.get("points") or q.get("marks") or 1
            max_score += marks
            student_answer = responses.get(q_id, "")
            correct = q.get("correctAnswer") or q.get("correct_answer") or ""
            is_correct = str(student_answer).strip().lower() == str(correct).strip().lower()
            if is_correct:
                score += marks
            feedback[q_id] = {
                "correct": is_correct,
                "student_answer": student_answer,
                "correct_answer": correct,
                "explanation": q.get("explanation", ""),
            }

        percentage = (score / max_score * 100) if max_score else 0
        return {
            "score": score,
            "max_score": max_score,
            "percentage": round(percentage, 2),
            "feedback": feedback,
        }

    async def evaluate_assignment_attempt(
        self,
        responses_json: list,
        answer_key_json: list,
        questions_json: list,
        subject: str = "",
    ) -> dict:
        """Per-question AI evaluation for assignment/quiz attempts.
        Returns feedback_json compatible with GradeSubmissionPage."""

        answer_map = {r["questionId"]: r.get("answer", "") for r in responses_json}
        key_map = {k["id"]: k for k in answer_key_json}
        max_score = sum(q.get("points", 1) for q in questions_json)

        # Build per-question payload for the prompt
        items = []
        for q in questions_json:
            qid = q["id"]
            key = key_map.get(qid, {})
            items.append({
                "questionId": qid,
                "type": q.get("type", "mcq"),
                "text": q.get("text", ""),
                "points": q.get("points", 1),
                "options": q.get("options"),
                "studentAnswer": answer_map.get(qid, ""),
                "correctAnswer": key.get("correctAnswer", ""),
                "explanation": key.get("explanation", ""),
            })

        prompt = f"""You are an expert teacher grading a student's assignment attempt{f' for "{subject}"' if subject else ''}.

Evaluate each question and assign a score. For each question provide the correct answer and a clear explanation of why it is correct.

Questions:
{json.dumps(items, indent=2)}

Scoring rules:
- MCQ: exact match between studentAnswer and correctAnswer (compare as strings) → full points or 0
- fill-blank / true-false: case-insensitive string match → full points or 0
- short-answer / essay: judge quality against correctAnswer → partial credit allowed
- matching: evaluate pair accuracy proportionally
- "score" must be 0 to the question's "points" value (decimals allowed for partial)
- "correctAnswer" must state the correct answer clearly (do NOT comment on the student's response)
- "explanation" must explain why that answer is correct

Return ONLY valid JSON, no markdown:
{{
  "feedback": [
    {{
      "questionId": "<id>",
      "score": <number>,
      "correctAnswer": "<the correct answer stated clearly and concisely>",
      "explanation": "<explanation of why this is the correct answer>"
    }}
  ]
}}"""

        try:
            response = await self.chat([{"role": "user", "content": prompt}])
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            data = json.loads(cleaned)
            feedback_list = data.get("feedback", [])

            total_score = sum(f.get("score", 0) for f in feedback_list)
            percentage = round((total_score / max_score) * 100, 1) if max_score > 0 else 0

            return {
                "feedback_json": feedback_list,
                "score": total_score,
                "max_score": max_score,
                "percentage": percentage,
            }
        except Exception:
            # Fallback: simple objective scoring without AI feedback
            fallback = []
            total_score = 0.0
            obj_types = {"mcq", "fill", "fill-blank", "true-false", "truefalse"}
            for q in questions_json:
                qid = q["id"]
                points = q.get("points", 1)
                qtype = (q.get("type") or "mcq").lower()
                student_ans = str(answer_map.get(qid, "")).strip()
                correct_ans = str(key_map.get(qid, {}).get("correctAnswer", "")).strip()
                if qtype in obj_types:
                    is_correct = student_ans.lower() == correct_ans.lower()
                    score = points if is_correct else 0
                    ca = correct_ans if correct_ans else "See answer key"
                    expl = "This is correct." if is_correct else f"The correct answer is: {correct_ans}."
                else:
                    score = 0
                    ca = correct_ans if correct_ans else "See answer key"
                    expl = "Manual review required."
                total_score += score
                fallback.append({"questionId": qid, "score": score, "correctAnswer": ca, "explanation": expl})

            percentage = round((total_score / max_score) * 100, 1) if max_score > 0 else 0
            return {
                "feedback_json": fallback,
                "score": total_score,
                "max_score": max_score,
                "percentage": percentage,
            }

    async def generate_ebook_outline(
        self,
        title: str,
        topic: str,
        subject: str | None,
        language: str,
        chapter_range: tuple,
        tone: str,
        grade: int | None = None,
        board: str | None = None,
    ) -> List[dict]:
        """Generate chapter titles and descriptions only — no full content."""
        min_ch, max_ch = chapter_range

        tone_context = {
            "academic": "formal and scholarly",
            "simple": "beginner-friendly and easy to follow",
            "story_based": "narrative-driven with engaging storytelling",
            "exam_oriented": "focused on exam-relevant topics and key facts",
        }.get(tone, "educational")

        # Build grade/board context lines only when provided
        grade_line = f"Grade level: Grade {grade}" if grade else ""
        board_line = f"Education board: {board}" if board else ""
        grade_board_section = "\n".join(filter(None, [grade_line, board_line]))

        grade_board_instructions = ""
        if grade or board:
            parts = []
            if grade:
                parts.append(f"grade {grade} students")
            if board:
                parts.append(f"the {board} curriculum/syllabus")
            target = " following ".join(parts) if board and grade else parts[0]
            grade_board_instructions = (
                f"\n- Content depth and vocabulary must be appropriate for {target}"
                f"\n- Chapter topics should align with what {target} would study"
            )

        language_name = self._LANGUAGE_NAMES.get(language.lower(), language)
        is_non_english_outline = language_name.lower() != "english"

        lang_instruction = ""
        if is_non_english_outline:
            lang_instruction = (
                f"\n\nLANGUAGE REQUIREMENT (MANDATORY):"
                f"\n- Write ALL chapter titles and descriptions in {language_name}."
                f"\n- Do NOT use English for titles or descriptions."
                f"\n- Every single word in the output must be in {language_name}."
                f"\n- Use correct {language_name} orthography — proper character combinations, vowel signs, and grammar."
                f"\n- Do NOT transliterate English words into {language_name} script."
                f"\n- Standard scientific symbols, formulas, and numbers may remain in their universal form."
            )

        prompt = f"""You are an expert educational author. Create a chapter outline for an eBook.

Title: {title}
Topic: {topic}
Subject: {subject or "General"}
{grade_board_section}
Language: {language_name}
Number of chapters: between {min_ch} and {max_ch}
Writing style: {tone_context}

Generate a logical, well-structured chapter outline where:
- The FIRST chapter MUST be an Introduction that sets the context and previews the book
- The LAST chapter MUST be a Conclusion that summarizes key takeaways
- Chapter titles are concise and clear (4-8 words)
- Descriptions are 1-2 sentences explaining what the chapter covers
- Chapters flow naturally from foundational concepts to advanced ones
- The tone/style "{tone}" is reflected in how chapters are framed{grade_board_instructions}{lang_instruction}

Return ONLY valid JSON in this exact structure:
{{
  "chapters": [
    {{
      "title": "Chapter title here",
      "description": "Brief description of what this chapter covers."
    }}
  ]
}}"""
        for attempt in range(3):
            try:
                response = await self._ebook_llm_call(prompt, max_tokens=2048)
                parsed = self._parse_json_response(response)
                if parsed and isinstance(parsed.get("chapters"), list) and len(parsed["chapters"]) > 0:
                    return parsed["chapters"]
                print(f"[Outline] attempt {attempt+1}: empty or missing chapters. Raw: {response[:300]}", flush=True)
            except Exception as e:
                print(f"[Outline] attempt {attempt+1} error: {type(e).__name__}: {e}", flush=True)

        # All attempts failed — build a safe fallback outline
        print(f"[Outline] all attempts failed, returning fallback for '{title}'", flush=True)
        num_chapters = (min_ch + max_ch) // 2
        fallback = []
        for i in range(num_chapters):
            if i == 0:
                fallback.append({"title": f"Introduction to {title}", "description": f"An overview of {title} and what this book covers."})
            elif i == num_chapters - 1:
                fallback.append({"title": "Conclusion and Key Takeaways", "description": "A summary of everything covered in this book."})
            else:
                fallback.append({"title": f"Chapter {i + 1}: {title}", "description": f"Core concepts and topics for chapter {i + 1}."})
        return fallback

    async def generate_ebook_images(
        self,
        title: str,
        chapters: List[dict],
        image_density: str,
        image_types: List[str] | None,
        subject: str | None = None,
        grade: int | None = None,
        tone: str = "academic",
    ) -> dict:
        """Generate infographic images using Gemini image generation for ebook cover and chapters."""
        import asyncio
        import base64
        from google import genai
        from google.genai import types

        api_key = settings.GEMINI_API_KEY or settings.GOOGLE_GEMINI_API_KEY
        print(f"[EbookImage] Starting image generation. API key set: {bool(api_key)}, density: {image_density}, chapters: {len(chapters)}")
        client = genai.Client(api_key=api_key)

        grade_str = f"Grade {grade}" if grade else "General"
        subj_str = subject or "General"

        images_per_chapter = {"minimal": 0, "standard": 1, "visual_heavy": 2}.get(image_density, 1)
        result: dict = {"cover_image": None, "chapter_images": {}}

        def _generate_image(prompt: str) -> str | None:
            """Generate an infographic image using Gemini and return as base64 data URL."""
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                    ),
                )

                # Safely access candidates
                if not response.candidates:
                    print(f"[EbookImage] No candidates in response. Prompt feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                    return None

                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    print(f"[EbookImage] No content parts. Finish reason: {getattr(candidate, 'finish_reason', 'N/A')}")
                    return None

                for part in candidate.content.parts:
                    if part.inline_data is not None:
                        img_bytes = part.inline_data.data
                        mime = part.inline_data.mime_type or "image/png"
                        b64 = base64.b64encode(img_bytes).decode("utf-8")
                        print(f"[EbookImage] Image generated ({len(img_bytes)} bytes)")
                        return f"data:{mime};base64,{b64}"

                print("[EbookImage] Response had parts but no image data found")
                return None
            except Exception as e:
                print(f"[EbookImage] Generation failed: {type(e).__name__}: {e}")
                return None

        def _build_cover_prompt() -> str:
            return (
                f"Create a professional, visually stunning book cover image for an educational eBook.\n\n"
                f"Book Title: {title}\n"
                f"Subject: {subj_str}\n"
                f"Grade Level: {grade_str}\n\n"
                "Design requirements:\n"
                "- Clean, modern book cover design\n"
                "- Bold, prominent title text at the center\n"
                "- Use vibrant, appealing colors related to the subject\n"
                "- Include relevant icons or illustrations related to the topic\n"
                "- Professional educational style suitable for students\n"
                "- No blank or empty spaces — fill with relevant visual elements\n"
                "- The image should look like a real book cover"
            )

        def _build_chapter_prompt(ch: dict, img_idx: int) -> str:
            ch_title = ch.get("title", "")
            key_pts = ch.get("key_points", []) or []
            key_pts_str = "\n".join(f"- {kp}" for kp in key_pts[:5]) if key_pts else ""
            summary = ch.get("summary", "")

            variant = ""
            if img_idx > 0:
                variant = "\nMake this a DIFFERENT style from the previous image — use a different layout, color scheme, and visual approach."

            return (
                f"Create a beautiful, colorful educational infographic image about: {ch_title}\n\n"
                f"Subject: {subj_str}\n"
                f"Grade Level: {grade_str}\n"
                f"Chapter Summary: {summary}\n\n"
                f"Key information to visualize:\n{key_pts_str}\n\n"
                "Design requirements:\n"
                "- Bold heading at the top with the chapter title\n"
                "- Use vibrant colors, icons, and visual hierarchy\n"
                "- Organize information into clear visual sections\n"
                "- Include key facts and concepts as visual callouts\n"
                "- Use arrows, connectors, or flow lines between related concepts\n"
                "- Make text readable and well-spaced\n"
                "- Professional infographic style suitable for students\n"
                "- Colored background with contrasting text\n"
                "- Do NOT leave large empty spaces — fill with relevant visual elements"
                f"{variant}"
            )

        loop = asyncio.get_running_loop()

        # Build all tasks: list of (key, chapter_index, prompt)
        tasks: list[tuple[str, int | None, str]] = []
        tasks.append(("cover", None, _build_cover_prompt()))

        if images_per_chapter > 0:
            max_chapters = 10
            for i, ch in enumerate(chapters[:max_chapters]):
                for img_idx in range(images_per_chapter):
                    tasks.append((f"ch_{i}_{img_idx}", i, _build_chapter_prompt(ch, img_idx)))

        # Run ALL image generations in parallel
        print(f"[EbookImage] Generating {len(tasks)} images in parallel...")
        data_urls = await asyncio.gather(
            *[loop.run_in_executor(None, _generate_image, t[2]) for t in tasks]
        )

        for (key, ch_idx, _), data_url in zip(tasks, data_urls):
            if key == "cover":
                result["cover_image"] = data_url
                if data_url:
                    print("[EbookImage] Cover image generated successfully")
            elif data_url and ch_idx is not None:
                ch_key = str(ch_idx)
                result["chapter_images"].setdefault(ch_key, [])
                result["chapter_images"][ch_key].append(data_url)
                print(f"[EbookImage] Chapter {ch_idx+1} image generated successfully")

        print(f"[EbookImage] Done. Cover: {bool(result['cover_image'])}, Chapters with images: {len(result['chapter_images'])}")
        return result

    # ── Shared helpers for ebook generation ────────────────────────────────

    _TONE_INSTRUCTIONS = {
        "academic": (
            "Write in formal, scholarly language with precise terminology. "
            "Define key terms when introduced. Use evidence-based arguments, "
            "structured sub-sections with clear headings, and rigorous explanations. "
            "Each chapter should read like a well-researched textbook section."
        ),
        "simple": (
            "Write in plain, easy-to-understand language suitable for beginners. "
            "Avoid jargon; explain technical terms immediately in simple words. "
            "Use short sentences, bullet points, relatable everyday analogies, "
            "and friendly examples that a student new to the topic can follow."
        ),
        "story_based": (
            "Open every chapter with a short engaging story, scenario, or character dialogue "
            "that naturally introduces the topic. Narrate concepts through the story, "
            "weaving educational content into the narrative. Use vivid descriptions, "
            "relatable characters, and real-world situations to make learning immersive."
        ),
        "exam_oriented": (
            "Focus strictly on exam-relevant facts, formulas, definitions, and concepts. "
            "Use callout markers like 'Remember:', 'Key Formula:', and 'Exam Tip:' "
            "to highlight critical information. End every chapter with 3-5 practice "
            "questions (with answers) covering the chapter's most testable content."
        ),
    }

    _SIZE_CONTENT_GUIDES = {
        "short": {
            "total_pages": 15,
            "content_pages": "pages 5–15 (11 content pages)",
            "paragraphs": "4-5 substantial paragraphs",
            "depth": (
                "Cover the concept with a clear introduction, 2-3 detailed body sections with examples, "
                "and a concise conclusion. Each chapter must feel complete and informative on its own."
            ),
            "key_points": "4-5 key points per chapter",
            "words_hint": "~1000-1200 words per chapter",
        },
        "medium": {
            "total_pages": 30,
            "content_pages": "pages 5–30 (26 content pages)",
            "paragraphs": "6-8 detailed paragraphs",
            "depth": (
                "Cover the topic with solid depth. Include an introduction, 3-5 well-developed sections "
                "with examples and explanations, connections to related ideas, and a conclusion paragraph."
            ),
            "key_points": "5-7 key points per chapter",
            "words_hint": "~1200-1500 words per chapter",
        },
        "large": {
            "total_pages": 60,
            "content_pages": "pages 5–60 (56 content pages)",
            "paragraphs": "9-12 comprehensive paragraphs with internal sub-headings",
            "depth": (
                "Cover the topic exhaustively. Use sub-headings to structure major ideas. Include an introduction, "
                "multiple in-depth sections with worked examples or case studies, real-world applications, "
                "and a thorough conclusion."
            ),
            "key_points": "6-8 key points per chapter",
            "words_hint": "~1400-1800 words per chapter",
        },
    }

    _LANGUAGE_NAMES = {
        "en": "English",
        "hi": "Hindi (हिन्दी)",
        "ta": "Tamil (தமிழ்)",
        "te": "Telugu (తెలుగు)",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "zh": "Chinese (中文)",
        "ar": "Arabic (عربي)",
        "pt": "Portuguese",
    }

    # Maps language name → its writing script (used for strong script-enforcement instructions)
    _SCRIPT_MAP = {
        "hindi": "Devanagari (हिन्दी)",
        "tamil": "Tamil (தமிழ்)",
        "telugu": "Telugu (తెలుగు)",
        "kannada": "Kannada (ಕನ್ನಡ)",
        "malayalam": "Malayalam (മലയാളം)",
        "bengali": "Bengali (বাংলা)",
        "marathi": "Devanagari (मराठी)",
        "gujarati": "Gujarati (ગુજરાતી)",
        "punjabi": "Gurmukhi (ਪੰਜਾਬੀ)",
        "urdu": "Urdu (اردو)",
        "arabic": "Arabic (العربية)",
        "chinese": "Chinese characters (中文)",
    }

    _HTML_FORMAT_RULES = (
        "FORMATTING: Use ONLY these HTML tags for formatting — no markdown syntax at all (no **, no #, no *, no _):\n"
        "  - Sub-headings: <h3>Sub-heading Text</h3> on its own line\n"
        "  - Bold/emphasis: <b>important term</b>\n"
        "  - Italic: <i>emphasized text</i>\n"
        "  - Paragraphs: separate paragraphs with blank lines (double newline)\n"
        "  - Lists: <ul><li>item</li></ul> or <ol><li>item</li></ol>\n"
        "  - Code blocks: <pre><code class=\"language-python\">code here</code></pre> (use the correct language class).\n"
        "    IMPORTANT: Preserve proper indentation and newlines inside code blocks.\n"
        "  - Inline code: <code>variable_name</code>\n"
        "  - Math/equations: Use LaTeX — inline: $E = mc^2$, block: $$\\\\int_0^\\\\infty e^{-x}\\\\,dx = 1$$\n"
        "  - Trig/log operators: use \\\\sin, \\\\cos, \\\\tan, \\\\cot, \\\\sec, \\\\csc, \\\\log, \\\\ln, \\\\exp, \\\\lim — NEVER \\\\text{sin} or \\\\operatorname{sin}\n"
        "  - Fractions: \\\\frac{numerator}{denominator}  — e.g. $\\\\frac{\\\\pi}{2}$\n"
        "  - Greek letters: \\\\pi, \\\\theta, \\\\alpha, \\\\beta, \\\\gamma, \\\\delta, \\\\omega, \\\\Sigma, etc.\n"
        "  - CRITICAL — JSON ESCAPING: ALL LaTeX backslashes in the JSON string MUST be double-escaped.\n"
        "    Write \\\\\\\\sin  not  \\\\sin  (the JSON decoder halves every double-backslash).\n"
        "    Example correct: \"$y = \\\\\\\\sin(x)$\"   → after JSON decode → $y = \\\\sin(x)$ → KaTeX renders correctly.\n"
        "    Example WRONG:   \"$y = \\\\sin(x)$\"       → \\\\s is invalid JSON / \\\\t becomes a tab character.\n"
        "  - Do NOT wrap paragraphs in <p> tags — just use blank lines between them.\n"
        "  - NEVER use markdown: no **bold**, no *italic*, no # headings, no - bullet lists, no ```code fences```."
    )

    # LaTeX commands that start with a JSON-valid escape character.
    # \t (tab) starts: \text, \textbf, \textit, \textrm, \texttt, \textsf, \top, \tilde, \to
    # \f (form-feed) starts: \frac, \forall, \fbox
    # \b (backspace) starts: \begin, \bar, \binom, \boldsymbol, \beta
    # These must be double-escaped in JSON strings (\\text{} → \text{} after decode).
    # We fix single-backslash occurrences here so json.loads always succeeds.
    _LATEX_JSON_FIX = [
        # \text, \textbf, \textit, \textrm, \texttt, \textsf, \textsc, \textsl
        (re.compile(r'(?<!\\)\\t(?=ext(?:bf|it|rm|sf|tt|sc|sl)?\{|ext\b)'), r'\\\\t'),
        # \top, \tilde, \to
        (re.compile(r'(?<!\\)\\t(?=op\b|ilde\b|o\b)'), r'\\\\t'),
        # \frac, \forall, \fbox
        (re.compile(r'(?<!\\)\\f(?=rac\{|orall\b|box\{)'), r'\\\\f'),
        # \begin, \bar, \binom, \boldsymbol, \beta, \bf
        (re.compile(r'(?<!\\)\\b(?=egin\{|ar\{|inom\{|oldsymbol\{|eta\b|f\b)'), r'\\\\b'),
    ]

    @staticmethod
    def _parse_json_response(response: str) -> dict | None:
        """Try to parse a JSON response, stripping markdown fences if present."""
        cleaned = response.strip()
        # Strip markdown code fences: ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        # If still not valid JSON, try to extract the first { ... } block
        if not cleaned.startswith("{"):
            brace_start = cleaned.find("{")
            if brace_start != -1:
                cleaned = cleaned[brace_start:]
        # Trim anything after the last }
        if cleaned:
            brace_end = cleaned.rfind("}")
            if brace_end != -1:
                cleaned = cleaned[: brace_end + 1]

        # Fix LaTeX backslashes that conflict with JSON escape sequences before parsing.
        # e.g. \text{sin} → \\text{sin} so json.loads decodes it as \text{sin} not [TAB]ext{sin}
        for _pat, _rep in AIService._LATEX_JSON_FIX:
            cleaned = _pat.sub(_rep, cleaned)

        try:
            return json.loads(cleaned)
        except Exception:
            return None

    @staticmethod
    def _convert_markdown_fences_to_html(text: str) -> str:
        """Convert markdown code fences (```lang ... ```) to HTML <pre><code> blocks."""
        def _replace_fence(m: re.Match) -> str:
            lang = m.group(1) or ""
            code = m.group(2)
            # HTML-escape the code content
            code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if lang:
                return f'<pre><code class="language-{lang}">{code}</code></pre>'
            return f"<pre><code>{code}</code></pre>"
        # Match ```lang\ncode``` or ```\ncode```
        text = re.sub(r"```(\w+)?\s*\n(.*?)```", _replace_fence, text, flags=re.DOTALL)
        # Also convert inline backticks: `code` → <code>code</code>
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        return text

    # ── Step 1: Generate book metadata (title page, summary, TOC, thank you) ─

    async def _generate_ebook_metadata(
        self,
        title: str,
        author: str,
        subject: str | None,
        grade: int | None,
        language: str,
        book_size: str,
        tone: str,
        chapter_list: list[dict],
    ) -> dict:
        """Generate book metadata: title_page, book_summary, table_of_contents, thank_you_message."""
        language_name = self._LANGUAGE_NAMES.get(language.lower(), language)
        toc_list = "\n".join(f"  {i+1}. {ch.get('title', f'Chapter {i+1}')}" for i, ch in enumerate(chapter_list))
        is_non_english = language_name.lower() != "english"
        if is_non_english:
            lang_enforcement = (
                f"\n\n⚠️ CRITICAL LANGUAGE REQUIREMENT — {language_name.upper()} ONLY:\n"
                f"- The 'title' in title_page MUST be a fluent {language_name} translation of: \"{title}\".\n"
                f"- ALL text fields (subtitle, description, book_summary, thank_you_message, ui_labels) MUST be in {language_name}.\n"
                f"- Do NOT write any of these fields in English — use {language_name} script exclusively.\n"
                f"- Use correct {language_name} orthography: proper character combinations, vowel signs, and conjuncts. No spelling mistakes.\n"
                f"- Use genuine {language_name} vocabulary — do NOT transliterate English words into {language_name} script.\n"
                f"- The 'author' field is kept as-is from the input.\n"
                f"- Chapter titles in 'table_of_contents' are already set — copy them exactly as provided.\n"
                f"- LaTeX math, chemical formulas, and scientific symbols stay in standard notation."
            )
            ui_labels_template = (
                f'"ui_labels": {{\n'
                f'    "about_this_book": "<translate \'About This Book\' to {language_name}>",\n'
                f'    "table_of_contents": "<translate \'Table of Contents\' to {language_name}>",\n'
                f'    "chapter": "<translate \'Chapter\' to {language_name}>",\n'
                f'    "key_points": "<translate \'Key Points\' to {language_name}>",\n'
                f'    "assessment_questions": "<translate \'Assessment Questions\' to {language_name}>",\n'
                f'    "multiple_choice_questions": "<translate \'Multiple Choice Questions\' to {language_name}>",\n'
                f'    "fill_in_the_blanks": "<translate \'Fill in the Blanks\' to {language_name}>",\n'
                f'    "short_answer_questions": "<translate \'Short Answer Questions\' to {language_name}>",\n'
                f'    "long_answer_questions": "<translate \'Long Answer Questions\' to {language_name}>",\n'
                f'    "thank_you": "<translate \'Thank You\' to {language_name}>",\n'
                f'    "answer": "<translate \'Answer\' to {language_name}>",\n'
                f'    "questions": "<translate \'Questions\' to {language_name}>",\n'
                f'    "figure": "<translate \'Figure\' to {language_name}>",\n'
                f'    "by": "<translate \'by\' (as in \'written by\') to {language_name}>"\n'
                f'  }}'
            )
            title_field = f'"title": "<fluent {language_name} translation of: {title}>"'
        else:
            lang_enforcement = "\n\nLANGUAGE: Write all generated content in English."
            ui_labels_template = (
                '"ui_labels": {'
                '"about_this_book": "About This Book",'
                '"table_of_contents": "Table of Contents",'
                '"chapter": "Chapter",'
                '"key_points": "Key Points",'
                '"assessment_questions": "Assessment Questions",'
                '"multiple_choice_questions": "Multiple Choice Questions",'
                '"fill_in_the_blanks": "Fill in the Blanks",'
                '"short_answer_questions": "Short Answer Questions",'
                '"long_answer_questions": "Long Answer Questions",'
                '"thank_you": "Thank You",'
                '"answer": "Answer",'
                '"questions": "Questions",'
                '"figure": "Figure",'
                '"by": "by"'
                '}'
            )
            title_field = f'"title": "{title}"'

        prompt = f"""Generate the metadata for an educational eBook. Return ONLY valid JSON (no markdown fences).

Book details:
- Title: {title}
- Author: {author}
- Subject: {subject or "General"}
- Grade: {grade or "General"}
- Language: {language_name}
- Book Size: {book_size}
- Tone: {tone.replace("_", " ").title()}

Chapters in this book:
{toc_list}
{lang_enforcement}

Return this exact JSON structure (fill every string field with {language_name} content):
{{
  "title_page": {{
    {title_field},
    "author": "{author}",
    "subtitle": "<write a compelling subtitle in {language_name}>",
    "description": "<write 2-3 sentence overview of the entire book in {language_name}>"
  }},
  "book_summary": "<write 4-10 sentences giving a comprehensive overview of the ENTIRE book in {language_name} — its scope, key themes, and what the reader will learn>",
  "table_of_contents": [
    {", ".join(f'{{"chapter_number": {i+1}, "title": "{ch.get("title", f"Chapter {i+1}")}"}}' for i, ch in enumerate(chapter_list))}
  ],
  "thank_you_message": "<write 2-3 warm, encouraging sentences in {language_name} wishing the reader well after completing the book>",
  {ui_labels_template}
}}"""
        print(f"[Ebook] Generating metadata for '{title}'...", flush=True)
        response = await self._ebook_llm_call(prompt, max_tokens=2048)
        parsed = self._parse_json_response(response)
        if parsed:
            return parsed
        # Fallback
        return {
            "title_page": {"title": title, "author": author, "subtitle": "", "description": ""},
            "book_summary": "",
            "table_of_contents": [
                {"chapter_number": i + 1, "title": ch.get("title", f"Chapter {i + 1}")}
                for i, ch in enumerate(chapter_list)
            ],
            "thank_you_message": f"Thank you for reading {title}. We hope this book has been a valuable and enriching experience for you.",
            "ui_labels": {},
        }

    # ── Step 2: Generate a single chapter (called in parallel) ───────────

    async def _generate_single_chapter(
        self,
        chapter_number: int,
        chapter_title: str,
        chapter_description: str,
        title: str,
        subject: str | None,
        grade: int | None,
        language: str,
        book_size: str,
        tone: str,
        total_chapters: int,
        all_chapter_titles: list[str],
        board: str | None = None,
    ) -> dict:
        """Generate content for a single chapter via its own LLM call."""
        language_name = self._LANGUAGE_NAMES.get(language, language.upper())
        size_guide = self._SIZE_CONTENT_GUIDES.get(book_size, self._SIZE_CONTENT_GUIDES["short"])
        tone_guide = self._TONE_INSTRUCTIONS.get(tone, self._TONE_INSTRUCTIONS["academic"])

        # Give context about the full book so chapters feel cohesive
        other_chapters = "\n".join(
            f"  {i+1}. {t}" for i, t in enumerate(all_chapter_titles)
        )

        # Build grade/board context
        grade_line = f"Grade: {grade}" if grade else "Grade: General"
        board_line = f"\nBoard/Curriculum: {board}" if board else ""

        grade_board_instructions = ""
        if grade or board:
            parts = []
            if grade:
                parts.append(
                    f"- Content depth, vocabulary, and complexity must be appropriate for grade {grade} students."
                )
            if board:
                parts.append(
                    f"- Align content with the {board} curriculum standards and syllabus expectations."
                )
            grade_board_instructions = "\n" + "\n".join(parts)

        prompt = f"""You are writing Chapter {chapter_number} of a {total_chapters}-chapter educational eBook.

Book: "{title}"
Subject: {subject or "General"}
{grade_line}{board_line}
Language: {language_name}

Full book outline (for context — you are writing ONLY chapter {chapter_number}):
{other_chapters}

CHAPTER TO WRITE:
- Chapter Number: {chapter_number}
- Title: {chapter_title}
{f'- Description/Scope: {chapter_description}' if chapter_description else ''}

TONE: {tone.replace("_", " ").title()}
{tone_guide}

CONTENT REQUIREMENTS:
- Length: {size_guide["paragraphs"]} — {size_guide["words_hint"]}
- Structure: {size_guide["depth"]}
- The content must be the FULL chapter body — not a placeholder, stub, or summary.
- Do NOT include the chapter title in the content — it will be added separately.{grade_board_instructions}

{self._HTML_FORMAT_RULES}

LANGUAGE — STRICT: Write ALL prose, headings, lists, and explanations in {language_name}.
{
    f"Use correct {language_name} orthography throughout — proper character combinations, vowel signs, and conjuncts. "
    f"Do NOT produce transliterated English in {language_name} script. "
    f"Use genuine {language_name} vocabulary and grammatically correct sentences."
    if language_name.lower() != "english" else ""
}
LATEX / MATH EXCEPTION: All LaTeX math ($...$, $$...$$), chemical equations ($\\ce{{...}}$), code blocks, variable names, and standard scientific symbols MUST remain in their standard notation — do NOT convert them to {language_name} script.

QUALITY STANDARDS (mandatory):
- Every paragraph must be substantive, complete, and factually accurate — no filler, no placeholders.
- Sentences must be well-formed and flow naturally in {language_name}.
- Technical terms should use correct {language_name} equivalents; retain the English term only if no natural equivalent exists.
- Do NOT repeat the same point across multiple paragraphs.

Return ONLY valid JSON (no markdown fences):
{{
  "chapter_number": {chapter_number},
  "title": "{chapter_title}",
  "content": "Full chapter content here with HTML formatting...",
  "key_points": ["point 1", "point 2", "..."],
  "summary": "1-2 sentence recap of this chapter"
}}"""
        # Retry up to 2 times for reliable JSON output
        max_retries = 2
        for attempt in range(max_retries + 1):
            response = await self._ebook_llm_call(prompt, max_tokens=12000)
            parsed = self._parse_json_response(response)
            if parsed:
                parsed["chapter_number"] = chapter_number
                # Post-process: convert any markdown code fences to HTML
                if "content" in parsed and isinstance(parsed["content"], str):
                    parsed["content"] = self._convert_markdown_fences_to_html(parsed["content"])
                print(f"[Ebook] Chapter {chapter_number}/{total_chapters} generated: {chapter_title}", flush=True)
                return parsed
            if attempt < max_retries:
                print(f"[Ebook] Chapter {chapter_number} attempt {attempt+1} failed, retrying...", flush=True)
            else:
                print(f"[Ebook] Chapter {chapter_number} all attempts failed. First 300 chars: {response[:300]}", flush=True)

        # Final fallback — try to extract just the content field from the raw response
        fallback_content = response
        try:
            # Attempt partial extraction: find "content" field value in the raw JSON string
            content_match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
            if content_match:
                fallback_content = content_match.group(1).encode().decode('unicode_escape')
        except Exception:
            pass  # keep the raw response if extraction fails
        # Convert any remaining markdown code fences to HTML
        fallback_content = self._convert_markdown_fences_to_html(fallback_content)
        return {
            "chapter_number": chapter_number,
            "title": chapter_title,
            "content": fallback_content,
            "key_points": [],
            "summary": "",
        }

    # ── Step 3: Generate assessments (separate call) ─────────────────────

    async def _generate_ebook_assessment(
        self,
        title: str,
        language: str,
        chapters_data: list[dict],
        assessment_config: dict,
    ) -> dict | None:
        """Generate final_assessment section across all chapters."""
        language_name = self._LANGUAGE_NAMES.get(language, language.upper())
        difficulty = assessment_config.get("difficulty", "medium")
        q_types = assessment_config.get("questionTypes", ["MCQ"])
        blooms = assessment_config.get("bloomsLevel", "understand")
        q_types_str = ", ".join(q_types)

        # Normalise to lowercase for matching (frontend may send "mcq" or "MCQ")
        q_types_lower = [q.lower().replace(" ", "") for q in q_types]

        type_instructions = []
        json_fields = []
        if any(q in q_types_lower for q in ["mcq", "multiplechoice"]):
            type_instructions.append('- For MCQ: include in "mcq_questions" with "chapter_number", "question", "options" (4 choices), "answer" (correct option text).')
            json_fields.append('    "mcq_questions": [{ "chapter_number": 1, "question": "...", "options": ["A", "B", "C", "D"], "answer": "..." }]')
        if any(q in q_types_lower for q in ["fillinblank", "fill_in_blank", "fillintheblank"]):
            type_instructions.append('- For Fill in Blank: include in "fill_in_blank_questions" with "chapter_number", "question" (with ___), "answer".')
            json_fields.append('    "fill_in_blank_questions": [{ "chapter_number": 1, "question": "The ___ process...", "answer": "..." }]')
        if any(q in q_types_lower for q in ["shortanswer", "short_answer", "short"]):
            type_instructions.append('- For Short Answer: include in "short_answer_questions" with "chapter_number", "question", "answer" (2-3 sentences).')
            json_fields.append('    "short_answer_questions": [{ "chapter_number": 1, "question": "...", "answer": "..." }]')
        if any(q in q_types_lower for q in ["longanswer", "long_answer", "long"]):
            type_instructions.append('- For Long Answer: include in "long_answer_questions" with "chapter_number", "question", "answer" (detailed).')
            json_fields.append('    "long_answer_questions": [{ "chapter_number": 1, "question": "...", "answer": "..." }]')

        # If no types matched (e.g. empty or unknown), default to MCQ so the page is never blank
        if not json_fields:
            type_instructions.append('- For MCQ: include in "mcq_questions" with "chapter_number", "question", "options" (4 choices), "answer" (correct option text).')
            json_fields.append('    "mcq_questions": [{ "chapter_number": 1, "question": "...", "options": ["A", "B", "C", "D"], "answer": "..." }]')

        # Build chapter summaries for context
        chapter_context = "\n".join(
            f"Chapter {ch['chapter_number']}: {ch['title']}\n  Summary: {ch.get('summary', 'N/A')}\n  Key Points: {', '.join(ch.get('key_points', []))}"
            for ch in chapters_data
        )

        is_non_english_assess = language_name.lower() != "english"
        lang_assess_note = (
            f" Use correct {language_name} orthography throughout — proper character combinations and grammar. "
            f"EXCEPTION: LaTeX math ($...$, $$...$$), chemical formulas ($\\ce{{...}}$), "
            f"code, and standard scientific symbols must stay in their standard notation."
            if is_non_english_assess else ""
        )
        prompt = f"""Generate assessment questions for an educational eBook titled "{title}".

LANGUAGE — STRICT: Write ALL questions, options, and answers ENTIRELY in {language_name}.{lang_assess_note}

MATH FORMATTING (mandatory):
- Wrap ALL mathematical expressions in LaTeX delimiters: inline $...$ or block $$...$$
- Use proper LaTeX operators: \\\\sin, \\\\cos, \\\\tan, \\\\log, \\\\frac{{a}}{{b}}, \\\\sqrt{{x}}
- Greek letters: \\\\theta, \\\\pi, \\\\alpha, \\\\beta — NOT Unicode θ, π, α
- Superscripts/subscripts: $x^{{2}}$, $x_{{n}}$ (always with braces in LaTeX)
- Example question: "Which identity is correct? A) $\\\\sin^{{2}}(\\\\theta) + \\\\cos^{{2}}(\\\\theta) = 1$"
- ALL backslashes in JSON strings MUST be double-escaped (\\\\\\\\sin not \\\\sin)

Chapter summaries:
{chapter_context}

REQUIREMENTS:
- Generate 3-5 questions PER CHAPTER, distributed across ALL {len(chapters_data)} chapters.
- Question types: {q_types_str}
- Difficulty: {difficulty}
- Bloom's Taxonomy level: {blooms}
{chr(10).join(type_instructions)}
- Only include JSON keys for the selected question types above — omit others.

Return ONLY valid JSON (no markdown fences):
{{
{("," + chr(10)).join(json_fields)}
}}"""
        print(f"[Ebook] Generating assessments for {len(chapters_data)} chapters...", flush=True)
        for attempt in range(3):
            response = await self._ebook_llm_call(prompt, max_tokens=12000)
            parsed = self._parse_json_response(response)
            if parsed:
                print(f"[Ebook] Assessments generated successfully", flush=True)
                return parsed
            if attempt < 2:
                print(f"[Ebook] Assessment attempt {attempt+1} failed, retrying...", flush=True)
        print(f"[Ebook] Assessment all attempts failed", flush=True)
        return None

    # ── TOC page-number calculation ───────────────────────────────────────

    @staticmethod
    def _calculate_toc_page_numbers(
        toc: list[dict],
        chapters: list[dict],
        has_summary: bool = True,
    ) -> list[dict]:
        """Assign content-aware page numbers to TOC entries.

        Front-matter layout (fixed):
          Page 1  — Title Page
          Page 2  — About This Book  (only if has_summary)
          Page 3  — Table of Contents
          Page 4+ — Chapters

        Page count per chapter is estimated from actual word count
        (stripped of HTML) at ~280 words per page — a standard
        educational textbook page — giving realistic, non-uniform gaps
        rather than the old equal-distribution formula.
        """
        _TAG_RE = re.compile(r"<[^>]+>")
        _WORDS_PER_PAGE = 280

        def _word_count(html: str) -> int:
            return len(_TAG_RE.sub(" ", html or "").split())

        # front: title(1) + summary(1 if present) + toc(1)
        front_pages = 2 + (1 if has_summary else 0)
        ch_map = {ch.get("chapter_number", i + 1): ch for i, ch in enumerate(chapters)}

        current_page = front_pages + 1   # first chapter always starts here
        updated: list[dict] = []
        for entry in toc:
            new_entry = {**entry, "page_number": current_page}
            ch = ch_map.get(entry.get("chapter_number"))
            words = _word_count(ch.get("content", "")) if ch else 0
            pages = max(1, round(words / _WORDS_PER_PAGE))
            current_page += pages
            updated.append(new_entry)
        return updated

    # ── Main entry point: parallel ebook generation ──────────────────────

    async def generate_ebook(
        self,
        title: str,
        subject: str | None,
        grade: int | None,
        language: str,
        source_type: str,
        outline: List[str] | None,
        page_count: int,
        chapter_range: tuple = (3, 5),
        tone: str = "academic",
        book_size: str = "short",
        chapters: List[dict] | None = None,
        image_density: str = "standard",
        image_types: List[str] | None = None,
        author: str = "",
        assessment_config: dict | None = None,
        on_chapter_done: Any = None,
        board: str | None = None,
    ) -> dict:
        """Generate structured eBook content using parallel LLM calls per chapter, then images.

        Args:
            on_chapter_done: Optional async callback(chapter_number, total) called when each chapter finishes.
        """
        author = author or "Anonymous"
        assessment_enabled = bool(assessment_config and assessment_config.get("enabled"))

        # Build the chapter list from provided chapters or outline
        if chapters and len(chapters) > 0:
            chapter_list = [
                {"title": ch.get("title", f"Chapter {i+1}"), "description": ch.get("description", "")}
                for i, ch in enumerate(chapters)
            ]
        elif outline:
            chapter_list = [{"title": item, "description": ""} for item in outline]
        else:
            # No chapters provided — generate outline first
            min_ch, max_ch = chapter_range
            outline_data = await self.generate_ebook_outline(
                title=title, topic=subject, subject=subject,
                language=language, chapter_range=chapter_range, tone=tone,
            )
            chapter_list = [{"title": ch["title"], "description": ch.get("description", "")} for ch in outline_data]

        total_chapters = len(chapter_list)
        all_chapter_titles = [ch["title"] for ch in chapter_list]
        print(f"[Ebook] Starting parallel generation: {total_chapters} {book_size} chapters", flush=True)

        # ── Step 1 + Step 2 in parallel: metadata + all chapters concurrently ──
        async def _gen_chapter(idx: int, ch: dict) -> dict:
            result = await self._generate_single_chapter(
                chapter_number=idx + 1,
                chapter_title=ch["title"],
                chapter_description=ch.get("description", ""),
                title=title,
                subject=subject,
                grade=grade,
                board=board,
                language=language,
                book_size=book_size,
                tone=tone,
                total_chapters=total_chapters,
                all_chapter_titles=all_chapter_titles,
            )
            if on_chapter_done:
                await on_chapter_done(idx + 1, total_chapters)
            return result

        # Fire metadata + all chapters concurrently
        tasks = [self._generate_ebook_metadata(
            title=title, author=author, subject=subject, grade=grade,
            language=language, book_size=book_size, tone=tone, chapter_list=chapter_list,
        )]
        for idx, ch in enumerate(chapter_list):
            tasks.append(_gen_chapter(idx, ch))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unpack results
        metadata = results[0] if not isinstance(results[0], Exception) else {}
        if isinstance(metadata, Exception):
            print(f"[Ebook] Metadata generation failed: {metadata}", flush=True)
            metadata = {}

        generated_chapters = []
        for i, result in enumerate(results[1:], start=1):
            if isinstance(result, Exception):
                print(f"[Ebook] Chapter {i} failed: {result}", flush=True)
                generated_chapters.append({
                    "chapter_number": i,
                    "title": chapter_list[i - 1]["title"],
                    "content": f"<i>Content generation failed for this chapter. Error: {result}</i>",
                    "key_points": [],
                    "summary": "",
                })
            else:
                generated_chapters.append(result)

        # Sort by chapter_number to maintain order
        generated_chapters.sort(key=lambda c: c.get("chapter_number", 0))

        # ── Step 3: Assessments (after chapters are done, needs their content) ──
        final_assessment = None
        if assessment_enabled:
            try:
                final_assessment = await self._generate_ebook_assessment(
                    title=title, language=language,
                    chapters_data=generated_chapters, assessment_config=assessment_config,
                )
            except Exception as e:
                print(f"[Ebook] Assessment generation failed: {e}", flush=True)

        # ── Assemble final ebook_data ──
        raw_toc = metadata.get("table_of_contents", [
            {"chapter_number": i + 1, "title": ch["title"]} for i, ch in enumerate(chapter_list)
        ])
        has_summary = bool(metadata.get("book_summary"))
        toc_with_pages = self._calculate_toc_page_numbers(raw_toc, generated_chapters, has_summary)

        ebook_data = {
            "title": title,
            "author": author,
            "language": language,
            "book_size": book_size,
            "tone": tone,
            "title_page": metadata.get("title_page", {"title": title, "author": author, "subtitle": "", "description": ""}),
            "book_summary": metadata.get("book_summary", ""),
            "table_of_contents": toc_with_pages,
            "chapters": generated_chapters,
            "final_assessment": final_assessment,
            "thank_you_message": metadata.get("thank_you_message",
                f"Thank you for reading {title}. We hope this book has been a valuable and enriching experience for you."),
            "ui_labels": metadata.get("ui_labels", {}),
        }

        print(f"[Ebook] All {total_chapters} chapters assembled successfully", flush=True)

        # ── Generate images ──
        if image_density != "minimal":
            try:
                images = await self.generate_ebook_images(
                    title=title,
                    chapters=generated_chapters,
                    image_density=image_density,
                    image_types=image_types,
                    subject=subject,
                    grade=grade,
                    tone=tone,
                )
                ebook_data["images"] = images
            except Exception as e:
                print(f"[EbookImage] Image generation error: {e}")

        return ebook_data

    async def generate_ebook_content_only(
        self,
        title: str,
        subject: str | None,
        grade: int | None,
        language: str,
        source_type: str,
        outline: list[str] | None,
        page_count: int,
        chapter_range: tuple = (3, 5),
        tone: str = "academic",
        book_size: str = "short",
        chapters: list[dict] | None = None,
        author: str = "",
        assessment_config: dict | None = None,
        on_chapter_done: Any = None,
        board: str | None = None,
    ) -> dict:
        """Generate structured eBook content as JSON — NO image generation.
        Used by the SSE endpoint which handles images as a separate step."""
        return await self.generate_ebook(
            title=title,
            subject=subject,
            grade=grade,
            board=board,
            language=language,
            source_type=source_type,
            outline=outline,
            page_count=page_count,
            chapter_range=chapter_range,
            tone=tone,
            book_size=book_size,
            chapters=chapters,
            author=author,
            image_density="minimal",
            image_types=None,
            assessment_config=assessment_config,
            on_chapter_done=on_chapter_done,
        )

    async def generate_mindmap(
        self,
        topic: str,
        subject: str | None,
        grade: int | None,
        board: str | None,
        depth: int,
        language: str = "en",
    ) -> dict:
        """Generate a mind map structure as JSON."""
        lang_name = self._LANGUAGE_NAMES.get(language.lower(), language)
        lang_rule = (
            f"- LANGUAGE: ALL node labels MUST be written in {lang_name}. "
            + (
                f"Use {lang_name} script exclusively — do NOT write labels in English or any other script. "
                "Proper nouns, formulas, and standard scientific symbols may remain in their original form."
                if lang_name.lower() != "english"
                else "Write all node labels in English."
            )
        )
        prompt = f"""Create a mind map for:
Topic: {topic}
Subject: {subject or "General"}
Grade: {grade or "General"}
Board: {board or "General"}
Depth: {depth} levels
Language: {lang_name}

ACCURACY & SPELLING (CRITICAL):
- Every word in every node label MUST be spelled correctly.
- Scientific terms, proper nouns, formulas, and dates must be 100% accurate.
- Double-check all technical vocabulary and terminology before including it.
- Use proper capitalisation for proper nouns and sentence-case for other labels.
- Do NOT abbreviate in a way that changes meaning or creates ambiguity.
{lang_rule}

Return JSON with this structure:
{{
  "root": {{
    "id": "root",
    "label": "{topic}",
    "children": [
      {{
        "id": "node1",
        "label": "Subtopic 1",
        "children": [...]
      }}
    ]
  }}
}}

Return ONLY valid JSON.
"""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return {"root": {"id": "root", "label": topic, "children": []}}

    async def generate_video_script(
        self,
        topic: str,
        subject: str | None,
        grade: int | None,
        duration_minutes: int,
        style: str,
    ) -> dict:
        """Generate a structured video script."""
        prompt = f"""Create a {style} educational video script:
Topic: {topic}
Subject: {subject or "General"}
Grade: {grade or "General"}
Duration: ~{duration_minutes} minutes

Return JSON:
{{
  "title": "...",
  "duration_minutes": {duration_minutes},
  "scenes": [
    {{
      "scene_number": 1,
      "title": "...",
      "narration": "...",
      "visual_description": "...",
      "duration_seconds": 30
    }}
  ]
}}

Return ONLY valid JSON.
"""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return {"title": topic, "scenes": [{"scene_number": 1, "narration": response}]}

    async def generate_video_visuals(self, script_json: dict | None) -> dict:
        """Generate visual references for video scenes."""
        if not script_json:
            return {}
        prompt = f"Based on this video script, suggest visual elements for each scene:\n{json.dumps(script_json, indent=2)[:3000]}"
        response = await self.chat([{"role": "user", "content": prompt}])
        return {"visuals": response}

    async def generate_lesson_plan(
        self,
        class_id: str,
        topic: str,
        board: str,
        grade: int,
        subject: str,
        additional_context: str | None = None,
        class_name: str | None = None,
        class_section: str | None = None,
        class_description: str | None = None,
        language: str = "en",
    ) -> dict:
        """Generate a structured, grade-aware lesson plan."""

        # Build a rich grade context so the AI calibrates language and complexity
        grade_label = f"Grade {grade}"
        board_label = board or "General Curriculum"

        class_context_parts = []
        if class_name:
            class_context_parts.append(f"Class name: {class_name}")
        if class_section:
            class_context_parts.append(f"Section: {class_section}")
        if class_description:
            class_context_parts.append(f"Class notes: {class_description}")
        class_context_str = "\n".join(class_context_parts)

        lang_name = self._LANGUAGE_NAMES.get(language.lower(), language)
        _base_lang = lang_name.split("(")[0].strip().lower()
        _script = self._SCRIPT_MAP.get(_base_lang, lang_name)
        if _base_lang == "english":
            _lang_block = ""
        else:
            _lang_block = (
                f"LANGUAGE REQUIREMENT (mandatory — follow exactly)\n"
                f"Write ALL content values in: {lang_name}\n"
                f"Use {_script} script for all text values — titles, objectives, step descriptions, tasks, homework, differentiation text.\n"
                f"JSON field names (keys) MUST stay in English exactly as shown in the schema below.\n"
                f"Do NOT transliterate English words into {lang_name.split('(')[0].strip()} script.\n"
                f"EXCEPTIONS — keep in original form: LaTeX math ($...$), code, proper nouns with no {lang_name.split('(')[0].strip()} equivalent.\n"
            )

        prompt = f"""You are an expert teacher creating a lesson plan. Use every detail below to calibrate the plan.

{_lang_block}
CLASS DETAILS
─────────────
Subject      : {subject}
Board        : {board_label}
Grade        : {grade_label}
{class_context_str}

TOPIC
─────
{topic}

{f'TEACHER NOTES{chr(10)}─────────────{chr(10)}{additional_context}' if additional_context else ''}

GRADE-CALIBRATION RULES (follow strictly)
──────────────────────────────────────────
• Language & vocabulary must match the cognitive level of {grade_label} students.
  – Grades 1-3: very simple sentences, visual/hands-on activities, concrete examples.
  – Grades 4-6: simple explanations, real-world links, semi-concrete examples.
  – Grades 7-9: moderate terminology, structured reasoning, some abstraction.
  – Grades 10-12: subject-specific terminology, abstract reasoning, analytical tasks.
• Align learning objectives to {board_label} curriculum standards for {grade_label} {subject}.
• Time estimate should be realistic for a {grade_label} class period (typically 35-60 min).
• Steps must progress from activate-prior-knowledge → introduce → model → guided practice → independent practice → closure.
• Practice tasks must be solvable by an average {grade_label} student without extra resources.
• Formative check must be a single focused question or quick activity appropriate for {grade_label}.
• Homework must be achievable in 15-30 minutes by a {grade_label} student.
• Differentiation: easy = scaffolded/simplified for below-grade learners; standard = grade-level; advanced = extension/enrichment for above-grade learners.

MATH FORMATTING (JSON-safe LaTeX)
──────────────────────────────────
• Wrap every mathematical expression in $...$ (inline) or $$...$$ (display/block).
• Use proper LaTeX operator names: \\sin \\cos \\tan \\log \\ln — NOT \\text{{sin}}.
• Greek letters: \\alpha \\beta \\theta \\pi etc.
• Fractions: \\frac{{a}}{{b}} — double every backslash inside JSON strings.
• Do NOT write math as plain text like "sin^2(x)"; write "$\\sin^2 x$" instead.

OUTPUT
──────
Return ONLY valid JSON matching this schema exactly:
{{
  "title": "descriptive lesson title",
  "objectives": ["By the end of this lesson, students will be able to ...", "..."],
  "timeEstimate": 45,
  "steps": [
    {{"step": 1, "title": "step title", "description": "detailed teacher instructions", "duration": 10}},
    ...
  ],
  "practiceTasks": ["specific task 1", "specific task 2"],
  "formativeCheck": "one focused exit-ticket question or activity",
  "homework": "clear homework instruction",
  "differentiation": {{
    "easy": "what to simplify for struggling learners",
    "standard": "standard grade-level approach",
    "advanced": "extension challenge for advanced learners"
  }}
}}
"""
        _cfg = _build_genai_config({
            "max_output_tokens": 4096,
            "response_mime_type": "application/json",
        })
        response = ""
        try:
            _gc = self._get_gemini_async()
            if _gc:
                _r = await _gemini_generate_with_retry(_gc, settings.AI_PRIMARY_MODEL, prompt, _cfg)
                response = _r.text or ""
        except Exception as _e:
            print(f"[LessonPlan] Gemini failed: {type(_e).__name__}: {_e}", flush=True)
        parsed = self._parse_json_response(response)
        if parsed:
            return parsed
        return {"title": f"Lesson Plan: {topic}", "objectives": [topic], "timeEstimate": 45, "steps": []}

    async def generate_rubric(
        self, board: str, grade: int, subject: str, topic: str, criteria_count: int,
        difficulty_level: str = 'medium'
    ) -> List[dict]:
        """Generate grading rubric criteria aligned to board, grade, subject and difficulty."""
        board_context = {
            'CBSE': 'Follow CBSE/NCERT curriculum framework and learning outcomes. Use terminology aligned with CBSE competency-based evaluation.',
            'ICSE': 'Follow ICSE/CISCE curriculum standards. Use detailed analytical descriptors matching ICSE application-based assessment.',
            'IGCSE': 'Follow Cambridge IGCSE assessment objectives. Use command words (describe, explain, evaluate, analyse) consistent with Cambridge grading.',
            'IB': 'Follow IB curriculum framework with ATL skills integration. Use criterion-referenced descriptors aligned with IB rubric conventions.',
            'Cambridge': 'Follow Cambridge International assessment objectives. Use descriptors consistent with Cambridge mark schemes.',
            'State Board': 'Follow state board curriculum standards. Use clear, accessible descriptors appropriate for regional assessment style.',
        }.get(board, 'Follow the relevant curriculum standards for this education board.')

        difficulty_guidance = {
            'simple': 'Focus on recall, basic comprehension, and foundational skills. Use clear, simple language. Assess whether the student can identify, list, define, or demonstrate basic understanding.',
            'medium': 'Focus on application and analysis. Assess whether the student can explain, compare, apply concepts to new situations, and demonstrate working knowledge.',
            'complex': 'Focus on higher-order thinking: evaluation, synthesis, critical analysis. Assess depth of reasoning, originality, ability to justify arguments, and mastery of advanced concepts.',
        }.get(difficulty_level, 'Use moderately detailed descriptors.')

        if grade <= 5:
            grade_lang = 'Use age-appropriate, simple language suitable for primary school students.'
        elif grade <= 8:
            grade_lang = 'Use clear academic language suitable for middle school students.'
        else:
            grade_lang = 'Use formal academic language and rigorous expectations suitable for senior secondary students.'

        prompt = f"""You are an expert educator and curriculum designer. Create a detailed, classroom-ready grading rubric.

CONTEXT:
- Title/Topic: {topic}
- Subject: {subject}
- Board: {board}
- Grade: {grade}
- Difficulty: {difficulty_level}
- Number of criteria: {criteria_count}

BOARD-SPECIFIC GUIDANCE:
{board_context}

DIFFICULTY GUIDANCE:
{difficulty_guidance}

GRADE LEVEL:
{grade_lang}

INSTRUCTIONS:
1. Create exactly {criteria_count} evaluation criteria relevant to "{topic}" in {subject}.
2. Each criterion title must reflect a specific skill or learning outcome from the {board} Grade {grade} {subject} curriculum.
3. Weights across all criteria MUST total exactly 100%. Distribute based on importance.
4. Each criterion must have exactly 4 performance levels with clear, distinct, observable descriptors.
5. Descriptors must be specific — not vague (avoid generic phrases like "good work" without detail).
6. Each level must clearly differentiate from adjacent levels so teachers can objectively assess.
7. Align criteria to {board} curriculum expectations for Grade {grade} {subject}.

Return a JSON array in this exact format:
[
  {{
    "id": "criterion_1",
    "title": "Criterion Name",
    "weight": 25,
    "levels": [
      {{"level": "Excellent", "score": 4, "description": "Detailed descriptor..."}},
      {{"level": "Good", "score": 3, "description": "Detailed descriptor..."}},
      {{"level": "Satisfactory", "score": 2, "description": "Detailed descriptor..."}},
      {{"level": "Needs Improvement", "score": 1, "description": "Detailed descriptor..."}}
    ]
  }}
]

Return ONLY valid JSON array. No markdown, no explanation.
"""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return []

    async def auto_grade(self, submission_id: str, rubric_id: str, db) -> dict:
        """Generate an AI grade suggestion for a submission based on a rubric."""
        from sqlalchemy import select
        from app.models.classes import Submission, Rubric

        submission_result = await db.execute(
            select(Submission).where(Submission.id == submission_id)
        )
        submission = submission_result.scalar_one_or_none()
        rubric_result = await db.execute(
            select(Rubric).where(Rubric.id == rubric_id)
        )
        rubric = rubric_result.scalar_one_or_none()
        if not submission or not rubric:
            return {}

        prompt = f"""Grade this student submission based on the rubric.

Student response:
{submission.text_response or '(No text response - file submission)'}

Rubric criteria:
{json.dumps(rubric.criteria, indent=2)[:3000]}

Return JSON:
{{
  "totalScore": 0-100,
  "maxScore": 100,
  "criterionScores": [
    {{"criterionId": "...", "score": 0, "level": "...", "comment": "..."}}
  ],
  "overallComment": "..."
}}

Return ONLY valid JSON.
"""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return {"totalScore": 0, "maxScore": 100, "criterionScores": [], "overallComment": response}

    async def auto_grade_direct(
        self,
        submission_text: Optional[str],
        rubric: Optional[dict],
        questions: Optional[List[dict]],
        answers: Optional[dict],
        student_name: Optional[str],
        feedback_only: bool = False,
    ) -> dict:
        """Grade a submission directly from payload data (no DB lookups needed)."""
        student_label = student_name or "the student"

        # Build context about questions and answers
        qa_context = ""
        if questions and answers:
            qa_lines = []
            for q in questions:
                qid = q.get("id", "")
                qtext = q.get("text", q.get("question", ""))
                qtype = q.get("type", "")
                correct = q.get("correctAnswer", q.get("correct_answer", ""))
                student_ans = answers.get(qid, "(no answer)")
                qa_lines.append(
                    f"Q ({qtype}): {qtext}\n  Student answered: {student_ans}\n  Correct answer: {correct}"
                )
            qa_context = "\n".join(qa_lines)

        rubric_context = ""
        if rubric and rubric.get("criteria"):
            rubric_context = json.dumps(rubric["criteria"], indent=2)[:4000]

        if feedback_only or not rubric:
            prompt = f"""You are an expert teacher. Analyze this student submission and provide detailed feedback.

Student: {student_label}
{"Questions & Answers:" + chr(10) + qa_context if qa_context else ""}
{"Submission text:" + chr(10) + submission_text if submission_text else ""}

Return ONLY valid JSON in this exact format:
{{
  "overallComment": "2-3 sentence overall assessment",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "areasForImprovement": ["area 1", "area 2"],
  "remediationTopics": [
    {{"criterionTitle": "topic name", "recommendation": "specific advice", "resources": ["resource 1"]}}
  ]
}}"""
        else:
            # Extract exact criterion titles for the prompt
            criterion_titles = []
            if rubric and rubric.get("criteria"):
                criterion_titles = [c.get("title", "") for c in rubric["criteria"] if c.get("title")]
            titles_list = "\n".join(f'  - "{t}"' for t in criterion_titles) if criterion_titles else ""

            prompt = f"""You are an expert teacher. Grade this student submission using the rubric criteria.

Student: {student_label}
{"Questions & Answers:" + chr(10) + qa_context if qa_context else ""}
{"Submission text:" + chr(10) + submission_text if submission_text else ""}

Rubric criteria:
{rubric_context}

You MUST provide a score for EVERY criterion. The exact criterion titles you must use are:
{titles_list}

Return ONLY valid JSON in this exact format:
{{
  "criterionScores": [
    {{"criterionTitle": "exact criterion title from rubric", "points": <integer>, "comment": "brief comment"}}
  ],
  "overallComment": "2-3 sentence overall assessment",
  "strengths": ["strength 1", "strength 2"],
  "areasForImprovement": ["area 1", "area 2"],
  "remediationTopics": [
    {{"criterionTitle": "topic name", "recommendation": "specific advice", "resources": ["resource 1"]}}
  ]
}}

CRITICAL: criterionScores MUST contain exactly {len(criterion_titles)} entries, one for each criterion listed above. Copy each criterion title exactly as written — do not rephrase, abbreviate, or skip any. Points must be a valid integer within that criterion's level range."""

        response = ""
        try:
            response = await self.chat(
                [{"role": "user", "content": prompt}],
                json_mode=True,
            )
            cleaned = response.strip()
            # Strip markdown code fences if present
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[AIService] auto_grade_direct JSON parse failed: {e}\nRaw response: {response[:500]}", flush=True)
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except Exception:
                    pass
            return {
                "overallComment": "AI grading completed. Please review manually.",
                "strengths": [],
                "areasForImprovement": [],
                "remediationTopics": [],
            }
        except Exception as e:
            print(f"[AIService] auto_grade_direct failed: {e}", flush=True)
            return {
                "overallComment": "AI grading completed. Please review manually.",
                "strengths": [],
                "areasForImprovement": [],
                "remediationTopics": [],
            }

    async def suggest_questions(
        self, class_id: str, topic: str, question_types: List[str] | None, count: int, db
    ) -> List[dict]:
        """Suggest assignment questions for a topic."""
        types_str = ", ".join(question_types or ["Short Answer", "Essay", "MCQ"])
        prompt = f"""Suggest {count} creative assignment questions for:
Topic: {topic}
Question types: {types_str}

Return JSON array:
[
  {{
    "type": "...",
    "question": "...",
    "difficulty": "easy|medium|hard",
    "marks": 5
  }}
]

Return ONLY valid JSON.
"""
        try:
            response = await self.chat([{"role": "user", "content": prompt}])
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return []

    async def generate_assignment_questions(
        self,
        topic: str,
        subject: str,
        grade: int,
        mcq_count: int = 0,
        fib_count: int = 0,
        short_answer_count: int = 0,
        long_answer_count: int = 0,
        true_false_count: int = 0,
        match_count: int = 0,
        difficulty: str = "medium",
        blooms_level: str | None = None,
        mcq_subtypes: list | None = None,
        lesson_plan_context: dict | None = None,
        rubric_criteria: list | None = None,
        source_text: str | None = None,
        topic_weightage: dict | None = None,
        language: str = "en",
        language_label: str = "English",
        language_instruction: str | None = None,
    ) -> List[dict]:
        """Generate structured assignment questions for an AssignmentEditor.

        When a lesson plan is provided, questions are derived from its objectives and steps.
        When rubric criteria are provided, questions are aligned to each criterion so the
        assessment can be evaluated against the rubric.
        """
        parts = []
        if mcq_count:
            mcq_label = f"{mcq_count} MCQ (multiple choice)"
            if mcq_subtypes and len(mcq_subtypes) > 0:
                subtype_labels = {
                    "standard": "Standard MCQ",
                    "case": "Case-based MCQ (passage/scenario followed by questions)",
                    "assertion_reason": "Assertion-Reason MCQ (Assertion + Reason with options like Both A and R are true and R is the correct explanation, etc.)",
                    "higher_order": "Higher Order Thinking MCQ (requires analysis, evaluation, or application — not simple recall)",
                }
                sub_parts = [subtype_labels.get(s, s) for s in mcq_subtypes]
                mcq_label += f" — include these subtypes: {', '.join(sub_parts)}"
            parts.append(mcq_label)
        if fib_count:
            parts.append(f"{fib_count} Fill-in-the-blank")
        if short_answer_count:
            parts.append(f"{short_answer_count} Short answer")
        if long_answer_count:
            parts.append(f"{long_answer_count} Long answer")
        if true_false_count:
            parts.append(f"{true_false_count} True/False")
        if match_count:
            parts.append(f"{match_count} Match-the-following")
        types_str = ", ".join(parts) or "5 Short answer"

        # Build difficulty & Bloom's level instruction
        difficulty_instruction = f"Difficulty: {difficulty}"
        if difficulty == "mixed":
            difficulty_instruction = "Difficulty: Mixed — distribute questions across easy, medium, and hard levels"

        blooms_section = ""
        if blooms_level and blooms_level != "mixed":
            blooms_map = {
                "remember": "Remember (recall facts, terms, concepts)",
                "understand": "Understand (explain ideas, interpret, summarize)",
                "apply": "Apply (use knowledge in new situations, solve problems)",
                "analyze": "Analyze (break down information, identify patterns, compare/contrast)",
                "evaluate": "Evaluate (justify decisions, critique, assess)",
                "create": "Create (design, construct, produce original work)",
            }
            blooms_desc = blooms_map.get(blooms_level, blooms_level.capitalize())
            blooms_section = f"\nBloom's Taxonomy Level: {blooms_desc}\nAll questions MUST target the '{blooms_level}' cognitive level. Frame questions that require students to {blooms_level} rather than simply recall.\n"
        elif blooms_level == "mixed" or not blooms_level:
            blooms_section = "\nBloom's Taxonomy: Mixed — distribute questions across Remember, Understand, Apply, Analyze, Evaluate, and Create levels.\n"

        # Build optional context sections
        lesson_plan_section = ""
        if lesson_plan_context:
            objectives = lesson_plan_context.get("objectives") or []
            steps = lesson_plan_context.get("steps") or []
            practice = lesson_plan_context.get("practice_tasks") or []
            formative = lesson_plan_context.get("formative_check") or ""

            obj_text = "\n".join(f"  - {o}" for o in objectives) if objectives else "  (none listed)"
            step_text = "\n".join(
                f"  {i+1}. {s.get('title', s) if isinstance(s, dict) else s}"
                for i, s in enumerate(steps)
            ) if steps else "  (none listed)"
            practice_text = "\n".join(f"  - {p}" for p in practice) if practice else "  (none listed)"

            lesson_plan_section = f"""
Teaching Plan (use this to derive relevant questions):
  Plan Title: {lesson_plan_context.get('title', '')}
  Topic: {lesson_plan_context.get('topic', topic)}
  Learning Objectives:
{obj_text}
  Lesson Steps:
{step_text}
  Practice Tasks:
{practice_text}
  Formative Check: {formative}

Questions MUST be rooted in the above lesson plan content, objectives, and activities.
"""

        rubric_section = ""
        if rubric_criteria:
            criteria_lines = []
            for c in rubric_criteria:
                title = c.get("title", "")
                outcome = c.get("linkedOutcome", "")
                line = f"  - {title}" + (f" (outcome: {outcome})" if outcome else "")
                criteria_lines.append(line)
            criteria_text = "\n".join(criteria_lines)
            rubric_section = f"""
Rubric Assessment Criteria (questions must be aligned to these criteria so the assessment can be evaluated against the rubric):
{criteria_text}

Distribute questions across these criteria. Each question should clearly target one of the above criteria.
"""

        # Build source document section (vault file content)
        source_section = ""
        if source_text:
            # Truncate to ~6000 words to stay within context limits
            words = source_text.split()
            truncated = " ".join(words[:6000])
            if len(words) > 6000:
                truncated += " [... content truncated ...]"
            source_section = f"""
Source Document (generate questions ONLY from the content below — do not invent information outside this document):
---
{truncated}
---

"""

        # Build topic weightage section
        weightage_section = ""
        if topic_weightage and len(topic_weightage) > 1:
            weightage_lines = [f"  - {t}: {w}%" for t, w in topic_weightage.items()]
            weightage_section = f"""
Topic Weightage (distribute questions proportionally based on these weights):
{chr(10).join(weightage_lines)}

You MUST distribute the total number of questions across topics according to these percentages.
For example, if there are 10 questions total and Topic A has 60% weight, generate ~6 questions about Topic A.
"""

        lang_block = (
            language_instruction
            or f"Generate ALL question text, options, explanations, model answers, match pairs and any other content strictly in {language_label}. Do not mix languages. Even fixed/templated option sets (True/False, Assertion-Reason choices) MUST be translated into {language_label}. For Tamil specifically, use correct natural Tamil script — do NOT transliterate, do NOT mix English words. Proper nouns and standard scientific/mathematical symbols may remain in English."
        )

        prompt = f"""Generate assignment questions for a Grade {grade} {subject} class.

OUTPUT LANGUAGE — STRICT: {language_label}
{lang_block}

Topic: {topic}
{difficulty_instruction}
{blooms_section}Question breakdown: {types_str}
{source_section}{lesson_plan_section}{rubric_section}{weightage_section}
Return a JSON object with a "questions" array. Each question MUST follow this schema exactly:
- type: one of "mcq", "fill-blank", "short-answer", "long-answer", "true-false", "match"
- text: the question text (REQUIRED for ALL types including match-the-following)
- points: integer (2 for fill-blank/true-false, 5 for mcq/short-answer, 10 for match/long-answer)

REQUIRED fields per type:
- MCQ: "options" (array of 4 strings), "correctAnswer" (index 0-3 as integer), "explanation" (1-2 sentences)
- Fill-in-the-blank: "correctAnswer" (string — the word/phrase that fills the blank), "explanation" (1-2 sentences). The question text MUST contain ___ to mark the blank.
- True/False: "correctAnswer" (string: "True" or "False"), "explanation" (1-2 sentences explaining why it is true or false)
- Short answer: "correctAnswer" (string — the expected model answer, 1-3 sentences), "explanation" (1-2 sentences)
- Long answer: "correctAnswer" (string — a detailed model answer, 3-6 sentences), "explanation" (brief grading guidance)
- Match the following: "text" (a question/instruction like "Match the following items"), "matchPairs" (array of 4-5 objects with "left" and "right" keys), "explanation" (brief explanation of the correct matches)

CRITICAL: Every question MUST have "correctAnswer" (or "matchPairs" for match type), "explanation", and "text" fields. Do NOT omit any of these.

Return ONLY valid JSON, no markdown fences.
Example: {{"questions": [{{"type": "mcq", "text": "What is ...?", "options": ["A","B","C","D"], "correctAnswer": 0, "points": 5, "explanation": "Because ..."}}, {{"type": "true-false", "text": "The earth is flat.", "correctAnswer": "False", "points": 2, "explanation": "The earth is an oblate spheroid."}}, {{"type": "short-answer", "text": "Explain photosynthesis.", "correctAnswer": "Photosynthesis is the process by which plants convert sunlight into energy.", "points": 5, "explanation": "Key concepts: light energy, chlorophyll, glucose production."}}, {{"type": "long-answer", "text": "Discuss the causes of World War I.", "correctAnswer": "World War I was caused by a combination of factors including militarism, alliances, imperialism, and nationalism...", "points": 10, "explanation": "Should cover at least 3 major causes with examples."}}, {{"type": "match", "text": "Match the following terms with their definitions.", "matchPairs": [{{"left": "Mitosis", "right": "Cell division"}}, {{"left": "Meiosis", "right": "Reproductive cell division"}}], "points": 10, "explanation": "Mitosis produces identical cells while meiosis produces gametes."}}]}}
"""
        try:
            response = await self.chat([{"role": "user", "content": prompt}])
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            parsed = json.loads(cleaned)
            return parsed.get("questions", parsed) if isinstance(parsed, dict) else parsed
        except Exception:
            return []

    async def stream_playground(
        self,
        topic: str,
        mode: str,
        messages: List[dict],
        grade: int | None,
        harder_mode: bool,
        context: dict | None,
    ) -> AsyncIterator[str]:
        """Stream playground exploration responses."""
        mode_prompts = {
            "experiment": f"Let's run a thought experiment about '{topic}'. Guide the student through interactive hypotheses and observations.",
            "play": f"Let's play and explore '{topic}' in a fun, engaging way. Make it interactive and creative.",
            "challenge": f"Give a challenging problem about '{topic}' that requires deep thinking. {'Make it harder than usual.' if harder_mode else ''}",
            "imagine": f"Imagine and create a scenario involving '{topic}'. Encourage creative storytelling and speculation.",
        }
        system = mode_prompts.get(mode, f"Explore the topic: {topic}")

        ctx = context or {}
        student_mode = bool(ctx.get("student_mode", True))
        board = ctx.get("board")
        if student_mode:
            if grade:
                system += f"\nAdapt for Grade {grade} students."
            if board:
                system += (
                    f"\nAlign examples and terminology with the {board} curriculum."
                )

        full_messages = [{"role": "system", "content": system}] + messages + [
            {"role": "user", "content": f"Continue our {mode} session about {topic}."}
        ]
        async for chunk in self.stream_chat(full_messages, context):
            yield chunk

    async def playground_explore(
        self,
        topic: str,
        mode: str,
        messages: List[dict],
        grade: int | None,
        harder_mode: bool,
        context: dict | None,
    ) -> str:
        chunks = []
        async for chunk in self.stream_playground(topic, mode, messages, grade, harder_mode, context):
            chunks.append(chunk)
        return "".join(chunks)

    async def _fetch_org_enrichment_data(self, user_id: str, org_id: str | None, db) -> dict:
        """
        Fetch org-specific data (class submissions, evaluation attempts, engagement)
        to enrich AI analytics. Returns empty dict for personal workspace.
        """
        import uuid as _uuid
        if not org_id or org_id == "personal":
            return {}

        try:
            parsed_org = _uuid.UUID(org_id)
        except (ValueError, TypeError):
            return {}

        from sqlalchemy import select, func as sqlfunc
        from app.models.classes import Submission, Assignment, Class
        from app.models.evaluation import EvaluationAttempt, EvaluationAssessment
        from app.models.gamification import WorkspaceGamification, StudentBadge
        from app.models.content import UserLibraryItem, Ebook

        result = {}

        # ── 1. Graded class submissions ──────────────────────────────────────
        try:
            sub_result = await db.execute(
                select(Submission, Assignment, Class)
                .join(Assignment, Submission.assignment_id == Assignment.id)
                .join(Class, Assignment.class_id == Class.id)
                .where(
                    Submission.student_id == user_id,
                    Submission.status.in_(["graded", "returned"]),
                    Class.org_id == parsed_org,
                )
                .order_by(Submission.submitted_at.desc())
                .limit(30)
            )
            sub_rows = sub_result.all()

            submissions_text_lines = []
            sub_subject_scores = {}  # subject -> list of percentages
            for sub, assign, cls in sub_rows:
                grade_data = sub.grade or {}
                total = grade_data.get("totalScore", 0)
                max_s = grade_data.get("maxScore", assign.points or 1)
                pct = round((total / max_s * 100) if max_s else 0)
                feedback = grade_data.get("overallComment", "")
                subject = cls.subject or "General"

                submissions_text_lines.append(
                    f"- {assign.title} ({subject}): {total}/{max_s} ({pct}%)"
                    + (f" — Feedback: {feedback[:120]}" if feedback else "")
                )
                sub_subject_scores.setdefault(subject, []).append(pct)

            if submissions_text_lines:
                result["submissions_text"] = "\n".join(submissions_text_lines)
                result["submissions_count"] = len(sub_rows)
                # Per-subject aggregates from submissions
                sub_subject_lines = []
                for subj, scores in sub_subject_scores.items():
                    avg_s = round(sum(scores) / len(scores))
                    best_s = max(scores)
                    sub_subject_lines.append(
                        f"- {subj}: {len(scores)} assignments, avg {avg_s}%, best {best_s}%"
                    )
                result["submissions_subject_text"] = "\n".join(sub_subject_lines)
                # For combining stats
                all_pcts = [p for scores in sub_subject_scores.values() for p in scores]
                result["submissions_avg"] = round(sum(all_pcts) / len(all_pcts)) if all_pcts else 0
                result["submissions_best"] = max(all_pcts) if all_pcts else 0
        except Exception:
            pass

        # ── 2. Evaluation attempts ───────────────────────────────────────────
        try:
            eval_result = await db.execute(
                select(EvaluationAttempt, EvaluationAssessment)
                .join(EvaluationAssessment, EvaluationAttempt.assessment_id == EvaluationAssessment.id)
                .where(
                    EvaluationAttempt.student_id == user_id,
                    EvaluationAttempt.status == "evaluated",
                    EvaluationAssessment.org_id == parsed_org,
                )
                .order_by(EvaluationAttempt.submitted_at.desc())
                .limit(20)
            )
            eval_rows = eval_result.all()
            if eval_rows:
                eval_lines = []
                for attempt, assessment in eval_rows:
                    pct = round(attempt.percentage or 0)
                    eval_lines.append(
                        f"- {assessment.title} (Difficulty: {assessment.difficulty or 'N/A'}): "
                        f"{attempt.score or 0}/{attempt.max_score or 0} ({pct}%)"
                    )
                result["evaluations_text"] = "\n".join(eval_lines)
                result["evaluations_count"] = len(eval_rows)
                eval_pcts = [round(a.percentage or 0) for a, _ in eval_rows]
                result["evaluations_avg"] = round(sum(eval_pcts) / len(eval_pcts)) if eval_pcts else 0
        except Exception:
            pass

        # ── 3. Org engagement data ───────────────────────────────────────────
        try:
            ws_gam_result = await db.execute(
                select(WorkspaceGamification).where(
                    WorkspaceGamification.user_id == user_id,
                    WorkspaceGamification.org_id == parsed_org,
                )
            )
            ws_gam = ws_gam_result.scalar_one_or_none()
            if ws_gam:
                result["org_xp"] = ws_gam.xp
                result["org_streak"] = ws_gam.streak
        except Exception:
            pass

        try:
            doc_result = await db.execute(
                select(sqlfunc.count(UserLibraryItem.id)).where(
                    UserLibraryItem.user_id == user_id,
                    UserLibraryItem.org_id == parsed_org,
                )
            )
            result["doc_count"] = doc_result.scalar() or 0
        except Exception:
            pass

        try:
            ebook_result = await db.execute(
                select(sqlfunc.count(Ebook.id)).where(
                    Ebook.user_id == user_id,
                    Ebook.org_id == parsed_org,
                )
            )
            result["ebook_count"] = ebook_result.scalar() or 0
        except Exception:
            pass

        try:
            badge_result = await db.execute(
                select(sqlfunc.count(StudentBadge.id)).where(
                    StudentBadge.student_id == user_id,
                )
            )
            result["badge_count"] = badge_result.scalar() or 0
        except Exception:
            pass

        return result

    def _build_org_context_text(self, org_data: dict) -> str:
        """Build prompt text from org enrichment data."""
        if not org_data:
            return ""

        sections = []

        if org_data.get("submissions_text"):
            sections.append(
                f"CLASS ASSIGNMENT PERFORMANCE ({org_data.get('submissions_count', 0)} graded assignments):\n"
                f"{org_data['submissions_text']}"
            )
        if org_data.get("submissions_subject_text"):
            sections.append(
                f"CLASS SUBJECT AVERAGES:\n{org_data['submissions_subject_text']}"
            )
        if org_data.get("evaluations_text"):
            sections.append(
                f"ORGANIZATION EXAM RESULTS ({org_data.get('evaluations_count', 0)} evaluations):\n"
                f"{org_data['evaluations_text']}"
            )

        engagement_parts = []
        if org_data.get("org_xp"):
            engagement_parts.append(f"Organization XP: {org_data['org_xp']}")
        if org_data.get("org_streak"):
            engagement_parts.append(f"Current streak: {org_data['org_streak']} days")
        if org_data.get("doc_count"):
            engagement_parts.append(f"Documents uploaded: {org_data['doc_count']}")
        if org_data.get("ebook_count"):
            engagement_parts.append(f"eBooks created: {org_data['ebook_count']}")
        if org_data.get("badge_count"):
            engagement_parts.append(f"Badges earned: {org_data['badge_count']}")
        if engagement_parts:
            sections.append(f"ORGANIZATION ENGAGEMENT:\n" + "\n".join(f"- {p}" for p in engagement_parts))

        return "\n\n".join(sections)

    async def generate_career_profile(self, user_id: str, db, org_id: str | None = None) -> dict:
        """
        Agentic career profile builder.
        Reads assessment scores, topic mastery, recent AI chat messages, and past career
        guidance sessions to produce a fully personalised career profile — no user input required.
        The profile grows richer as the user takes more assessments and chats more.
        """
        from sqlalchemy import select, func as sqlfunc
        from app.models.assessment import AssessmentAttempt, TopicMastery, PracticeAssessment
        from app.models.ai import AiChat, AiChatMessage
        from app.models.insights import CareerGuidanceSession

        # ── 1. Assessment data ────────────────────────────────────────────────
        attempts_result = await db.execute(
            select(AssessmentAttempt, PracticeAssessment)
            .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
            .where(AssessmentAttempt.user_id == user_id, AssessmentAttempt.status == "evaluated")
            .order_by(AssessmentAttempt.submitted_at.desc())
            .limit(30)
        )
        attempt_rows = attempts_result.all()

        # ── 2. Topic mastery ──────────────────────────────────────────────────
        mastery_result = await db.execute(
            select(TopicMastery)
            .where(TopicMastery.user_id == user_id)
            .order_by(TopicMastery.mastery_level.desc())
            .limit(20)
        )
        mastery_data = mastery_result.scalars().all()

        # ── 3. Per-subject aggregated stats ───────────────────────────────────
        stats_result = await db.execute(
            select(
                PracticeAssessment.subject,
                sqlfunc.count(AssessmentAttempt.id).label("attempt_count"),
                sqlfunc.avg(AssessmentAttempt.percentage).label("avg_pct"),
                sqlfunc.max(AssessmentAttempt.percentage).label("best_pct"),
            )
            .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
            .where(AssessmentAttempt.user_id == user_id, AssessmentAttempt.status == "evaluated")
            .group_by(PracticeAssessment.subject)
            .order_by(sqlfunc.avg(AssessmentAttempt.percentage).desc())
        )
        subject_stats = stats_result.all()

        # ── 4. Recent user messages from AI chat (infer interests from topics) ─
        chat_msgs_result = await db.execute(
            select(AiChatMessage.content)
            .join(AiChat, AiChatMessage.chat_id == AiChat.id)
            .where(
                AiChat.user_id == user_id,
                AiChatMessage.role == "user",
            )
            .order_by(AiChatMessage.created_at.desc())
            .limit(30)
        )
        user_messages = [row[0] for row in chat_msgs_result.all()]

        # ── 5. Past career guidance sessions ─────────────────────────────────
        sessions_result = await db.execute(
            select(CareerGuidanceSession)
            .where(CareerGuidanceSession.user_id == user_id)
            .order_by(CareerGuidanceSession.created_at.desc())
            .limit(3)
        )
        past_sessions = sessions_result.scalars().all()

        # ── 6. Org-specific data (class submissions, evaluations, engagement) ─
        org_data = await self._fetch_org_enrichment_data(user_id, org_id, db)
        org_context = self._build_org_context_text(org_data)

        # ── Build prompt context ──────────────────────────────────────────────
        has_data = bool(attempt_rows or mastery_data or user_messages or org_data.get("submissions_count"))

        subject_text = "\n".join(
            f"- {row.subject or 'General'}: {int(row.attempt_count)} attempts, "
            f"avg {round(row.avg_pct or 0)}%, best {round(row.best_pct or 0)}%"
            for row in subject_stats
        ) or "No subject data yet."

        mastery_text = "\n".join(
            f"- {m.topic} ({m.subject}): {m.mastery_level:.0f}% mastery, trend: {m.trend}"
            for m in mastery_data
        ) or "No mastery data yet."

        chat_context = "\n".join(
            f"- {msg[:120]}" for msg in user_messages[:20]
        ) or "No chat history yet."

        past_sessions_text = "\n".join(
            f"- Interests: {session.interests} | Target: {session.target_careers}"
            for session in past_sessions
        ) or "No past career sessions."

        if not has_data:
            return {
                "summary": "You haven't used the platform enough yet for a personalised career profile. Start by taking assessments in subjects you enjoy and chatting with the AI Assistant about topics that interest you.",
                "inferred_interests": [],
                "subject_strengths": [],
                "top_careers": [],
                "skill_gaps": [],
                "next_steps": [
                    "Take assessments in subjects you enjoy to build your academic profile",
                    "Chat with the AI Assistant about topics you're curious about",
                    "Use the Generate Paths tab to explore careers manually",
                ],
                "data_richness": "none",
            }

        prompt = f"""You are an expert AI career counsellor. Analyse this student's platform usage data and generate a comprehensive, personalised career profile. Everything must be grounded in the specific data provided.

SUBJECT PERFORMANCE (from practice assessments):
{subject_text}

TOPIC MASTERY:
{mastery_text}

RECENT AI CHAT TOPICS (user's questions — use to infer interests):
{chat_context}

PAST CAREER GUIDANCE SESSIONS:
{past_sessions_text}
{f'{chr(10)}{org_context}' if org_context else ''}

Generate a career profile with this exact JSON structure:

{{
  "summary": "3-4 sentence personalised career readiness overview. Reference actual subjects and scores. Mention inferred interests from their chats.",
  "inferred_interests": ["keyword1", "keyword2"],
  "subject_strengths": [
    {{"subject": "Subject Name", "score": 78, "trend": "improving|steady|declining", "detail": "78% avg, 3 attempts"}}
  ],
  "top_careers": [
    {{
      "title": "Career Title",
      "compatibility": 85,
      "description": "2-sentence description referencing why it fits this specific student",
      "skills": ["Skill1", "Skill2", "Skill3", "Skill4"],
      "education": "Recommended education path",
      "reasons": ["Specific reason 1 from their data", "Specific reason 2"]
    }}
  ],
  "skill_gaps": [
    {{
      "skill": "Skill Name",
      "career": "Career Title",
      "current": 55,
      "required": 85,
      "note": "Short explanation"
    }}
  ],
  "next_steps": [
    "Specific actionable next step referencing actual data"
  ],
  "data_richness": "rich|moderate|sparse"
}}

Rules:
- subject_strengths: all subjects from assessment data, ordered by score descending
- top_careers: 4-6 careers ranked by compatibility, calculated from actual subject scores and inferred interests
- skill_gaps: 4-6 items for the top 2-3 careers only
- inferred_interests: 5-8 keywords extracted from chat topics
- data_richness: "rich" if >=10 assessment attempts, "moderate" if 3-9, "sparse" if <3
- Return ONLY valid JSON. No markdown fences."""

        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return {
                "summary": f"You've taken assessments across {len(subject_stats)} subject(s). Keep going to unlock a full AI career profile!",
                "inferred_interests": [],
                "subject_strengths": [
                    {"subject": row.subject or "General", "score": round(row.avg_pct or 0), "trend": "steady", "detail": f"{int(row.attempt_count)} attempts"}
                    for row in subject_stats
                ],
                "top_careers": [],
                "skill_gaps": [],
                "next_steps": ["Take more assessments to improve your career profile"],
                "data_richness": "sparse",
            }

    async def analyze_career(
        self,
        interests: List[str],
        strengths: List[str],
        target_careers: List[str] | None,
        grade: int | None,
        context: dict | None,
    ) -> dict:
        """Perform career compatibility analysis."""
        stream_text = (context or {}).get("stream", "Not specified")
        prompt = f"""Analyze career compatibility for a student:
Interests: {', '.join(interests)}
Strengths: {', '.join(strengths)}
Target careers: {', '.join(target_careers or ['Not specified'])}
Grade: {grade or 'Not specified'}
Preferred stream: {stream_text}

Return JSON with this exact structure:
{{
  "top_careers": [
    {{
      "title": "Career Title",
      "compatibility": 85,
      "description": "2-sentence description referencing why it fits this student",
      "skills": ["Skill1", "Skill2", "Skill3", "Skill4"],
      "education": "Recommended education path",
      "reasons": ["Specific reason 1 from their data", "Specific reason 2"]
    }}
  ],
  "compatibility_scores": {{"career_name": 0-100}},
  "strengths_analysis": "...",
  "recommended_paths": ["..."],
  "skills_to_develop": ["..."],
  "roadmap": "..."
}}

Rules:
- top_careers: 4-6 careers ranked by compatibility (0-100), each with title, compatibility score, description, skills array, education path, and reasons array
- compatibility_scores: map each career title to its compatibility number
- Return ONLY valid JSON. No markdown fences.
"""
        response = await self.chat([{"role": "user", "content": prompt}], context)
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return {"analysis": response, "compatibility_scores": {}}

    async def generate_insights(self, user_id: str, db) -> List[dict]:
        """Generate personalized learning insights for a user based on assessment data."""
        from sqlalchemy import select
        from app.models.assessment import AssessmentAttempt, TopicMastery, PracticeAssessment

        mastery_result = await db.execute(
            select(TopicMastery)
            .where(TopicMastery.user_id == user_id)
            .order_by(TopicMastery.mastery_level.asc())
            .limit(10)
        )
        mastery_data = mastery_result.scalars().all()

        attempts_result = await db.execute(
            select(AssessmentAttempt, PracticeAssessment)
            .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
            .where(AssessmentAttempt.user_id == user_id, AssessmentAttempt.status == "evaluated")
            .order_by(AssessmentAttempt.submitted_at.desc())
            .limit(10)
        )
        rows = attempts_result.all()

        mastery_summary = "\n".join(
            f"- {m.topic} ({m.subject}): {m.mastery_level:.0f}% mastery, trend: {m.trend}, {m.attempts_count} attempts"
            for m in mastery_data
        ) if mastery_data else "No topic mastery data yet."

        attempts_summary = "\n".join(
            f"- {assessment.subject} ({', '.join(assessment.topics or [])}): {attempt.percentage:.0f}% — difficulty: {assessment.difficulty}"
            for attempt, assessment in rows
        ) if rows else "No assessment attempts yet."

        prompt = f"""Generate 5 personalized learning insights for a student based on their assessment performance.

Topic mastery data (weakest first):
{mastery_summary}

Recent assessment results:
{attempts_summary}

Generate 5 insights using EXACTLY these type values (one of each):
- "weak_topic": A specific topic with low mastery (< 60%) that needs attention
- "retry_suggestion": Suggest retrying an assessment where they scored poorly
- "difficulty_upgrade": A topic/subject where they're excelling and should try harder difficulty
- "content_recommendation": A new topic or concept to explore next based on their progress
- "improvement": A positive trend, improvement, or strength worth celebrating

If no data is available, generate motivational getting-started insights with the correct types.

Return a JSON array of exactly 5 objects:
[
  {{
    "type": "weak_topic|retry_suggestion|difficulty_upgrade|content_recommendation|improvement",
    "title": "Short insight title (max 8 words)",
    "content": "Specific, actionable insight in 1-2 sentences referencing the data",
    "data": {{}}
  }}
]

Return ONLY valid JSON array. No markdown fences."""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return [{"type": "content_recommendation", "title": "Start Your Journey", "content": response, "data": {}}]

    async def generate_assessment_recommendations(self, user_id: str, db, org_id: str | None = None) -> List[dict]:
        """Generate actionable recommendations based on the user's assessment history."""
        from sqlalchemy import select
        from app.models.assessment import AssessmentAttempt, TopicMastery, PracticeAssessment

        # Fetch recent evaluated attempts joined with assessment metadata
        attempts_result = await db.execute(
            select(AssessmentAttempt, PracticeAssessment)
            .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
            .where(AssessmentAttempt.user_id == user_id, AssessmentAttempt.status == "evaluated")
            .order_by(AssessmentAttempt.submitted_at.desc())
            .limit(20)
        )
        rows = attempts_result.all()

        # Fetch topic mastery sorted weakest first
        mastery_result = await db.execute(
            select(TopicMastery)
            .where(TopicMastery.user_id == user_id)
            .order_by(TopicMastery.mastery_level.asc())
            .limit(15)
        )
        mastery_data = mastery_result.scalars().all()

        # Org-specific data
        org_data = await self._fetch_org_enrichment_data(user_id, org_id, db)
        org_context = self._build_org_context_text(org_data)

        attempts_summary = "\n".join(
            f"- Subject: {assessment.subject} | Topics: {', '.join(assessment.topics or [])} | "
            f"Difficulty: {assessment.difficulty} | Score: {attempt.percentage:.0f}% | "
            f"Date: {attempt.submitted_at.strftime('%Y-%m-%d') if attempt.submitted_at else 'N/A'}"
            for attempt, assessment in rows
        ) if rows else "No assessments taken yet."

        mastery_summary = "\n".join(
            f"- {m.topic} ({m.subject}): {m.mastery_level:.0f}% mastery | {m.attempts_count} attempts | trend: {m.trend}"
            for m in mastery_data
        ) if mastery_data else "No topic mastery data yet."

        prompt = f"""You are an AI learning coach. Analyze this student's assessment data and generate 6 specific, actionable recommendations.

RECENT PRACTICE ASSESSMENT ATTEMPTS (most recent first):
{attempts_summary}

TOPIC MASTERY (weakest first):
{mastery_summary}
{f'{chr(10)}{org_context}' if org_context else ''}

Generate exactly 6 recommendations using these types:
- "retry": Student scored < 60% on a topic/assessment — suggest retrying it
- "weak_topic": Topic mastery < 50% — needs focused practice
- "difficulty_upgrade": Consistently scoring > 80% — ready to move to harder difficulty
- "practice_more": Very few attempts (1-2) in a subject — needs more practice
- "strength": Topic mastery >= 80% — celebrate and suggest building on it
- "content": Suggest a related new topic to explore next

Priority scoring (higher = more urgent):
- Failed assessment / very weak topic: 85-95
- Weak topic needing attention: 70-84
- Difficulty upgrade ready: 55-70
- More practice needed: 45-60
- Strength / new content: 30-50

If no data exists, generate 6 helpful getting-started recommendations with priority 20-40.

Return a JSON array of exactly 6 objects:
[
  {{
    "type": "retry|weak_topic|difficulty_upgrade|practice_more|strength|content",
    "title": "Clear action title, max 8 words",
    "description": "One specific sentence explaining the recommendation",
    "reason": "The exact data point that triggered this (e.g. 'Scored 42% on Newton Laws')",
    "subject": "The subject name or null",
    "topic": "Specific topic name or null",
    "priority_score": 85
  }}
]

Return ONLY the JSON array. No markdown fences, no extra text."""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return []

    async def generate_assessment_summary(self, user_id: str, db, org_id: str | None = None) -> dict:
        """
        Agentic AI method — analyses the user's full Assessment Hub history
        and returns a structured coach-style summary with strengths, weak areas, goals and momentum.
        """
        from sqlalchemy import select, func as sqlfunc
        from app.models.assessment import AssessmentAttempt, TopicMastery, PracticeAssessment

        # ── Fetch all evaluated attempts + assessment metadata ──────────────
        attempts_result = await db.execute(
            select(AssessmentAttempt, PracticeAssessment)
            .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
            .where(AssessmentAttempt.user_id == user_id, AssessmentAttempt.status == "evaluated")
            .order_by(AssessmentAttempt.submitted_at.desc())
            .limit(30)
        )
        rows = attempts_result.all()

        # ── Topic mastery ───────────────────────────────────────────────────
        mastery_result = await db.execute(
            select(TopicMastery)
            .where(TopicMastery.user_id == user_id)
            .order_by(TopicMastery.mastery_level.desc())
            .limit(20)
        )
        mastery_data = mastery_result.scalars().all()

        # ── Aggregate stats per subject ──────────────────────────────────────
        stats_result = await db.execute(
            select(
                PracticeAssessment.subject,
                sqlfunc.count(AssessmentAttempt.id).label("attempt_count"),
                sqlfunc.avg(AssessmentAttempt.percentage).label("avg_pct"),
                sqlfunc.max(AssessmentAttempt.percentage).label("best_pct"),
            )
            .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
            .where(AssessmentAttempt.user_id == user_id, AssessmentAttempt.status == "evaluated")
            .group_by(PracticeAssessment.subject)
        )
        subject_stats = stats_result.all()

        # ── Org-specific data ────────────────────────────────────────────────
        org_data = await self._fetch_org_enrichment_data(user_id, org_id, db)
        org_context = self._build_org_context_text(org_data)

        total_attempts = len(rows) + org_data.get("submissions_count", 0) + org_data.get("evaluations_count", 0)
        all_pcts = [r[0].percentage or 0 for r in rows]
        if org_data.get("submissions_avg"):
            all_pcts.append(org_data["submissions_avg"])
        if org_data.get("evaluations_avg"):
            all_pcts.append(org_data["evaluations_avg"])
        overall_avg = round(sum(all_pcts) / len(all_pcts), 1) if all_pcts else 0
        best_overall = round(max(all_pcts, default=0), 1)

        # ── Build text context for the AI prompt ────────────────────────────
        attempts_text = "\n".join(
            f"- [{a.submitted_at.strftime('%b %d') if a.submitted_at else 'N/A'}] "
            f"{p.subject or 'General'} | Topics: {', '.join(p.topics or ['N/A'])} | "
            f"Difficulty: {p.difficulty} | Score: {a.percentage:.0f}%"
            for a, p in rows
        ) or "No assessment attempts yet."

        mastery_text = "\n".join(
            f"- {m.topic} ({m.subject}): {m.mastery_level:.0f}% mastery | "
            f"{m.attempts_count} attempts | trend: {m.trend}"
            for m in mastery_data
        ) or "No mastery data yet."

        subject_text = "\n".join(
            f"- {row.subject or 'General'}: {int(row.attempt_count)} attempts, "
            f"avg {round(row.avg_pct or 0)}%, best {round(row.best_pct or 0)}%"
            for row in subject_stats
        ) or "No subject data."

        if total_attempts == 0:
            return {
                "summary": "You haven't taken any assessments yet. Head to the Assessment Hub, create or take a quiz, and come back here for your personalised AI coaching review.",
                "momentum": "new",
                "strengths": [],
                "weak_areas": [],
                "goals": [
                    {"title": "Take your first assessment", "type": "explore", "priority": 90, "subject": None, "topic": None, "action_href": "/u/assessments"},
                    {"title": "Create a topic-based quiz", "type": "explore", "priority": 75, "subject": None, "topic": None, "action_href": "/u/assessments"},
                    {"title": "Upload study material to vault", "type": "explore", "priority": 50, "subject": None, "topic": None, "action_href": "/u/library"},
                ],
                "total_attempts": 0,
                "overall_avg": 0,
                "best_score": 0,
            }

        prompt = f"""You are an expert AI learning coach. Analyse this student's complete assessment and learning data and produce a personalised coaching summary.

TOTAL ASSESSMENTS & ASSIGNMENTS: {total_attempts}
OVERALL AVERAGE SCORE: {overall_avg}%
PERSONAL BEST SCORE: {best_overall}%

PER-SUBJECT STATS (practice assessments):
{subject_text}

RECENT PRACTICE ATTEMPTS (most recent first):
{attempts_text}

TOPIC MASTERY (strongest first):
{mastery_text}
{f'{chr(10)}{org_context}' if org_context else ''}

Based on ALL this data, generate a coaching summary with this exact JSON structure:

{{
  "summary": "2-3 sentence personalised coach narrative. Reference actual subjects/scores. Be specific and encouraging but honest about gaps.",
  "momentum": "improving|steady|declining",
  "strengths": [
    {{"label": "Subject or topic name", "detail": "Evidence: e.g. 78% avg, improving trend"}}
  ],
  "weak_areas": [
    {{"label": "Subject or topic name", "detail": "Evidence: e.g. 42% avg, needs more practice"}}
  ],
  "goals": [
    {{
      "title": "Specific action title, max 8 words",
      "type": "retry|practice|upgrade|explore|streak",
      "priority": 85,
      "subject": "the broad subject like Physics, Mathematics, Biology, etc. or null",
      "topic": "the SPECIFIC topic to practice like Gravity, Quadratic Equations, Photosynthesis, Newton's Laws, etc. MUST be a real topic name from the data above, NOT the subject name. This will be used to pre-fill an assessment creation form.",
      "action_href": "/u/assessments"
    }}
  ]
}}

Rules:
- strengths: 1-3 items with highest mastery/scores (>= 65%)
- weak_areas: 1-3 items that need attention (< 60% or low mastery)
- goals: exactly 3-5 specific goals ordered by priority (highest first)
- goal priority: retry failed (<60%) = 85-95, improve weak = 70-84, upgrade difficulty = 55-70, explore new = 30-55
- momentum: "improving" if recent scores are higher than older ones, "declining" if going down, "steady" otherwise
- CRITICAL for goals: "topic" must be an actual specific topic name extracted from the mastery or attempt data (e.g. "Gravity", "Thermodynamics", "Algebra"). Do NOT use the subject name (like "General" or "Physics") as the topic. If the data has topics like "Gravity (General)", the topic should be "Gravity" and subject should be "General Physics" or similar. If no specific topic exists, set topic to null.
- Return ONLY valid JSON. No markdown fences."""

        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            data = json.loads(cleaned)
            # Attach raw stats so frontend doesn't need to recompute
            data["total_attempts"] = total_attempts
            data["overall_avg"] = overall_avg
            data["best_score"] = best_overall
            return data
        except Exception:
            return {
                "summary": f"You've taken {total_attempts} assessment{'s' if total_attempts != 1 else ''} with an average score of {overall_avg}%. Keep going to unlock deeper insights!",
                "momentum": "steady",
                "strengths": [],
                "weak_areas": [],
                "goals": [{"title": "Keep taking assessments", "type": "practice", "priority": 70, "subject": None, "topic": None, "action_href": "/u/assessments"}],
                "total_attempts": total_attempts,
                "overall_avg": overall_avg,
                "best_score": best_overall,
            }

    async def generate_insight_feed(self, user_id: str, subject: str | None, db) -> List[dict]:
        """Generate curated insight articles for user's feed."""
        prompt = f"""Generate 5 educational insight articles{f' about {subject}' if subject else ''}.

Return JSON array:
[
  {{
    "title": "...",
    "summary": "...",
    "content": "...",
    "subject": "...",
    "tags": ["..."],
    "reading_time_minutes": 3
  }}
]

Return ONLY valid JSON.
"""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return []

    async def get_learning_intelligence(self, user_id: str, modules: List[str] | None, db) -> dict:
        """Get aggregated learning intelligence payload."""
        from sqlalchemy import select, func
        from app.models.assessment import AssessmentAttempt, TopicMastery

        avg_result = await db.execute(
            select(func.avg(AssessmentAttempt.percentage)).where(
                AssessmentAttempt.user_id == user_id,
                AssessmentAttempt.status == "evaluated",
            )
        )
        avg_score = avg_result.scalar_one() or 0

        mastery_result = await db.execute(
            select(TopicMastery).where(TopicMastery.user_id == user_id).limit(10)
        )
        mastery = mastery_result.scalars().all()

        return {
            "dashboard_snapshot": {
                "average_score": round(avg_score, 2),
                "topics_studied": len(mastery),
            },
            "bloom_profile": {
                "remember": min(avg_score, 100),
                "understand": min(avg_score * 0.9, 100),
                "apply": min(avg_score * 0.8, 100),
                "analyze": min(avg_score * 0.7, 100),
                "evaluate": min(avg_score * 0.6, 100),
                "create": min(avg_score * 0.5, 100),
            },
            "recommendations": [],
            "learning_trends": {"weekly": [], "monthly": []},
            "topic_strengths": [
                {"topic": m.topic, "subject": m.subject, "mastery": m.mastery_level}
                for m in mastery if m.mastery_level >= 70
            ],
            "topic_gaps": [
                {"topic": m.topic, "subject": m.subject, "mastery": m.mastery_level}
                for m in mastery if m.mastery_level < 50
            ],
            "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        }

    async def generate_evaluation_paper(
        self,
        subjects: List[dict],
        question_types: List[str] | None,
        difficulty: str = "medium",
        blooms_level: str = "mixed",
        question_count: int = 20,
        grade: int | None = None,
        board: str | None = None,
        mcq_subtypes: List[str] | None = None,
        type_weightage: dict | None = None,
        negative_marking: bool = False,
        language: str = "en",
        language_label: str = "English",
        language_instruction: str | None = None,
    ) -> tuple[List[dict], List[dict]]:
        """Generate institutional evaluation questions with full config support.

        Returns (question_json, answer_key_json) tuple.
        """
        types = question_types or ["mcq"]

        # ── Compute exact counts per question type ──────────────────────────
        if type_weightage and len(types) > 1:
            filtered_weights = {t: type_weightage.get(t, 0) for t in types}
            type_counts = self._distribute_questions(question_count, filtered_weights)
        else:
            type_counts = {types[0]: question_count} if len(types) == 1 else {
                t: max(1, question_count // len(types)) for t in types
            }
            diff = question_count - sum(type_counts.values())
            if diff:
                type_counts[types[-1]] = type_counts.get(types[-1], 1) + diff

        # ── MCQ subtype distribution ────────────────────────────────────────
        subtypes = mcq_subtypes or ["standard"]
        mcq_count = type_counts.get("mcq", 0)
        mcq_subtype_counts: dict = {}
        if mcq_count > 0 and len(subtypes) > 1:
            mcq_subtype_counts = self._distribute_questions(mcq_count, {s: 1 for s in subtypes})
        elif mcq_count > 0:
            mcq_subtype_counts = {subtypes[0]: mcq_count}

        # ── Distribution lines for prompt ───────────────────────────────────
        type_labels = {
            "mcq": "MCQ", "fill": "Fill in the Blank", "short": "Short Answer",
            "long": "Long Answer", "true_false": "True / False", "match": "Match the Following",
        }
        subtype_labels = {
            "standard": "Standard MCQ", "case": "Case-based MCQ",
            "assertion_reason": "Assertion-Reason MCQ", "higher_order": "Higher Order Thinking MCQ",
        }
        dist_lines = []
        for t, cnt in type_counts.items():
            label = type_labels.get(t, t)
            dist_lines.append(f"  - {label}: {cnt} question(s)")
            if t == "mcq" and mcq_subtype_counts:
                for s, sc in mcq_subtype_counts.items():
                    dist_lines.append(f"      • {subtype_labels.get(s, s)}: {sc}")

        # ── Build per-subject source sections ───────────────────────────────
        subject_sections = []
        all_subjects_str = ", ".join(s.get("subject", "") for s in subjects if s.get("subject"))
        for s in subjects:
            subj_name = s.get("subject", "General")
            chapters = s.get("chapters", [])
            chapters_str = ", ".join(c.get("name", "") for c in chapters) if chapters else "all topics"
            source_type = s.get("source_type", "online")
            source_text = s.get("source_text", "")

            chapter_names = [c.get("name", "") for c in chapters] if chapters else []
            section = f"\n### {subj_name}"
            if chapter_names:
                section += f"\nALLOWED CHAPTER NAMES (use ONLY these exact names in the 'chapter' field): {', '.join(chapter_names)}"
            if source_type in ("text", "file") and source_text and source_text.strip():
                section += (
                    f"\nSOURCE TEXT (generate questions for {subj_name} ONLY from this content):\n"
                    f"---\n{source_text[:5000]}\n---"
                )
            else:
                section += f"\nUse your educational knowledge of {subj_name} ({chapters_str})."
            subject_sections.append(section)

        # ── Bloom's taxonomy instruction ────────────────────────────────────
        blooms_map = {
            "remember": "Recall / recognition of facts",
            "understand": "Interpretation and explanation of concepts",
            "apply": "Use of knowledge in new practical situations",
            "analyze": "Break down information, find patterns and relationships",
            "evaluate": "Justify decisions, critique, judge quality",
            "create": "Design, produce, or construct new ideas",
        }
        if blooms_level and blooms_level != "mixed":
            blooms_section = f"BLOOM'S LEVEL: All questions must target '{blooms_level.capitalize()}' — {blooms_map.get(blooms_level, '')}."
        else:
            blooms_section = "BLOOM'S LEVEL: Use a balanced mix across Remember, Understand, Apply, and higher levels."

        neg_section = (
            "NEGATIVE MARKING: This is a negative-marking assessment. Every question MUST have "
            "one clearly unambiguous correct answer with no trick or confusable options."
            if negative_marking else ""
        )

        # ── Difficulty-specific instructions ──────────────────────────────
        difficulty_instructions = {
            "easy": (
                "DIFFICULTY CALIBRATION: EASY.\n"
                "- Questions should test basic recall and straightforward understanding.\n"
                "- MCQ distractors should be clearly different from the correct answer.\n"
                "- Avoid multi-step problems, tricky wording, or edge cases.\n"
                "- Answers should be directly findable in the textbook."
            ),
            "medium": (
                "DIFFICULTY CALIBRATION: MEDIUM.\n"
                "- Questions should test understanding and basic application.\n"
                "- Include a mix of direct recall and conceptual questions.\n"
                "- MCQ distractors should be plausible but distinguishable with knowledge.\n"
                "- Some questions should require 2-step reasoning."
            ),
            "hard": (
                "DIFFICULTY CALIBRATION: HARD.\n"
                "- Questions should test application, analysis, and higher-order thinking.\n"
                "- MCQ distractors should include common student misconceptions.\n"
                "- Include multi-step problems that require connecting concepts.\n"
                "- Questions should challenge students who have studied the material thoroughly."
            ),
        }
        difficulty_section = difficulty_instructions.get(difficulty, difficulty_instructions["medium"])

        allowed_types_str = " | ".join(f'"{t}"' for t in types)

        # ── Grade-specific and board-specific generation instructions ────
        grade_instruction = ""
        if grade:
            if grade <= 3:
                grade_instruction = (
                    f"GRADE CALIBRATION: Grade {grade} (ages 6-8, early primary).\n"
                    "- Use very simple, familiar language. Sentences should be short and clear.\n"
                    "- Questions should test basic recognition, recall, and simple understanding.\n"
                    "- Avoid abstract or multi-step reasoning. Focus on concrete, visual concepts.\n"
                    "- MCQ options should be clearly distinct — no tricky or confusable choices.\n"
                    "- Fill-in-the-blank answers should be single, common words.\n"
                    "- Use everyday examples (animals, food, colours, family, school)."
                )
            elif grade <= 6:
                grade_instruction = (
                    f"GRADE CALIBRATION: Grade {grade} (ages 9-11, upper primary).\n"
                    "- Use clear, age-appropriate language. Technical terms only if part of the syllabus.\n"
                    "- Questions can involve basic application and simple reasoning.\n"
                    "- MCQ distractors should be plausible but not confusingly similar.\n"
                    "- Short answers should expect 2-3 clear sentences.\n"
                    "- Include relatable examples from daily life, nature, and school."
                )
            elif grade <= 9:
                grade_instruction = (
                    f"GRADE CALIBRATION: Grade {grade} (ages 12-14, middle school).\n"
                    "- Use proper academic terminology as expected in the curriculum.\n"
                    "- Questions should include comprehension, application, and basic analysis.\n"
                    "- MCQ options can be closer in phrasing — test genuine understanding.\n"
                    "- Short/long answers should require structured responses with reasoning.\n"
                    "- Case-based and assertion-reason MCQs are appropriate at this level."
                )
            else:
                grade_instruction = (
                    f"GRADE CALIBRATION: Grade {grade} (ages 15-17, high school / board exam level).\n"
                    "- Use precise academic language matching board exam standards.\n"
                    "- Questions should span all Bloom's levels including analysis, evaluation, and application.\n"
                    "- MCQ options should include common misconceptions as distractors.\n"
                    "- Long answers should match the depth and format of actual board exam questions.\n"
                    "- Include numericals, derivations, or case studies where the subject demands it.\n"
                    "- Exam-pattern awareness: match the style students will face in actual exams."
                )

        board_instruction = ""
        if board:
            board_upper = (board or "").upper()
            if "CBSE" in board_upper:
                board_instruction = (
                    "BOARD ALIGNMENT: CBSE / NCERT.\n"
                    "- Follow NCERT textbook content, definitions, and terminology strictly.\n"
                    "- Question patterns should match CBSE board exam format.\n"
                    "- Use the exact phrasing and conventions from NCERT books.\n"
                    "- For science/math, follow the notation and methods used in NCERT."
                )
            elif "ICSE" in board_upper:
                board_instruction = (
                    "BOARD ALIGNMENT: ICSE / ISC.\n"
                    "- Follow ICSE curriculum standards which expect deeper conceptual understanding.\n"
                    "- Questions should be slightly more application-oriented than CBSE.\n"
                    "- Include questions that test comprehension and interpretation.\n"
                    "- Long answers should demonstrate structured analytical thinking."
                )
            elif "IB" in board_upper:
                board_instruction = (
                    "BOARD ALIGNMENT: IB (International Baccalaureate).\n"
                    "- Focus on inquiry-based, conceptual understanding.\n"
                    "- Questions should encourage critical thinking and global perspectives.\n"
                    "- Include questions that ask students to evaluate, compare, or construct arguments.\n"
                    "- Use international contexts and examples, not region-specific."
                )
            elif "CAMBRIDGE" in board_upper or "IGCSE" in board_upper:
                board_instruction = (
                    "BOARD ALIGNMENT: Cambridge (IGCSE / A-Level).\n"
                    "- Follow Cambridge exam format and marking scheme conventions.\n"
                    "- Questions should test precise factual knowledge and application.\n"
                    "- Use command words as defined by Cambridge (state, describe, explain, evaluate, etc.).\n"
                    "- Long answers should follow the structured response format expected in Cambridge exams."
                )
            elif "STATE" in board_upper:
                board_instruction = (
                    "BOARD ALIGNMENT: State Board.\n"
                    "- Use simple, direct question patterns matching state board exam style.\n"
                    "- Follow state textbook content and terminology.\n"
                    "- Questions should be straightforward — less tricky than CBSE/ICSE."
                )
            else:
                board_instruction = f"BOARD ALIGNMENT: {board}. Follow {board} curriculum standards and question patterns."

        lang_block = (
            language_instruction
            or f"Generate ALL question text, options, explanations and any other content strictly in {language_label}. Do not mix languages. Proper nouns and standard scientific/mathematical symbols may remain in English."
        )

        prompt = f"""You are an expert question paper setter for school students. Generate exactly {question_count} questions for an institutional evaluation paper.

OUTPUT LANGUAGE — STRICT: {language_label}
{lang_block}
This applies to EVERY field of EVERY question without exception: "text", every entry in "options", every "left"/"right" in "pairs", "correct_answer", and "explanation". Even fixed/templated option sets (e.g. True/False, Assertion-Reason choices) MUST be translated into {language_label}. For Tamil specifically, use correct, natural Tamil script — do NOT transliterate, do NOT mix English words, and ensure spelling/diacritics are accurate. The ONLY exceptions are proper nouns, chemical formulas, and standard mathematical symbols.

SUBJECTS: {all_subjects_str}
GRADE: {f'Grade {grade}' if grade else 'General'}{f' ({board})' if board else ''}

{difficulty_section}

{blooms_section}
{neg_section}

{grade_instruction}

{board_instruction}

ALLOWED QUESTION TYPES — STRICTLY: {allowed_types_str}
You MUST NOT generate any question with a "type" outside this list.

EXACT QUESTION DISTRIBUTION (generate exactly this many of each type — no more, no less):
{chr(10).join(dist_lines)}

SUBJECT-WISE SOURCE MATERIAL:
{chr(10).join(subject_sections)}

QUESTION FORMAT RULES — follow exactly:
1. MCQ (standard): 4 distinct options as a list. Exactly one correct.
   "options": ["option1", "option2", "option3", "option4"]
   "correct_answer": the exact correct option string.

2. MCQ (case): Include a brief scenario/passage (2-4 sentences) in "text".
   Then ask a question. 4 options as above.

3. MCQ (assertion_reason): Two statements.
   "text": labelled Assertion (A) and Reason (R), followed by "Choose the correct option:" — ALL written in {language_label}.
   "options": the four standard assertion-reason choices, written in {language_label} (translate the meanings below — do NOT output them in English):
     1) Both A and R are true, and R is the correct explanation of A
     2) Both A and R are true, but R is not the correct explanation of A
     3) A is true but R is false
     4) A is false but R is true
   "correct_answer": the exact {language_label} option string you generated.

4. MCQ (higher_order): Requires analysis/application — NOT simple recall. 4 options.

5. Fill in the Blank: "text" has ___ for the missing word/phrase.
   "correct_answer": the exact word/phrase. "options": null.

6. Short Answer: Question needing 2-4 sentence answer.
   "correct_answer": concise model answer. "options": null.

7. Long Answer: Descriptive/essay question.
   "correct_answer": key points and model answer outline. "options": null.

8. True / False: A clear factual statement.
   "options": ["True", "False"]. "correct_answer": "True" or "False".

9. Match the Following: Two columns to match.
   "options": right-column items as array. "pairs": [{{"left": "...", "right": "..."}}]
   "correct_answer": matching in "A-1, B-2" notation.

Return a JSON array of EXACTLY {question_count} objects. Each object MUST have ALL these fields:
- "id": "q1", "q2", "q3" ... (sequential)
- "type": one of the ALLOWED TYPES ONLY: {allowed_types_str}
- "subtype": for MCQ — "standard"|"case"|"assertion_reason"|"higher_order"; for others — null
- "text": the full question text
- "options": array of strings for mcq/true_false/match; null for fill/short/long
- "pairs": array of {{"left":..., "right":...}} for match; null for others
- "correct_answer": string (required)
- "explanation": 1-2 sentence explanation
- "marks": 1 for mcq/fill/true_false; 2 for short/match; 4 for long
- "subject": MUST be one of the subject names listed above — use the EXACT spelling provided
- "chapter": MUST be one of the chapter names listed above — use the EXACT spelling provided, do NOT invent your own chapter names
- "blooms_level": one of "remember"|"understand"|"apply"|"analyze"|"evaluate"|"create"

MATH NOTATION — CRITICAL:
1. Every mathematical expression, formula, matrix, fraction, trig function, Greek letter, or operator MUST be wrapped in LaTeX math delimiters: $...$ for inline (e.g. $1 + \\tan^2\\theta = \\sec^2\\theta$, $\\frac{{1}}{{2}}$) or $$...$$ for display/block math (e.g. $$\\begin{{pmatrix}}1&2\\\\3&4\\end{{pmatrix}}$$).
2. NEVER output bare LaTeX commands (\\frac, \\begin, \\sqrt, \\tan, \\sin, \\theta, etc.) outside $ or $$ delimiters.
3. NEVER use Unicode math characters (², ³, θ, α, β, π, ∑, →, −, ·, ∞, etc.) in option values or formulas — use LaTeX inside $...$ instead (e.g. write $\\theta$ not θ, write $x^2$ not x², write $\\tan^2\\theta - 1 = \\sec^2\\theta$ not tan²θ−1=sec²θ).
4. Plain English words in question text do NOT need delimiters; ONLY mathematical expressions get $...$.

Return ONLY the raw JSON array. No markdown fences, no explanation text outside the array."""

        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()

            # Try to extract JSON from markdown code fences
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("["):
                        cleaned = part
                        break

            # Fallback: find JSON array anywhere in the response via regex
            if not cleaned.startswith("["):
                match = re.search(r'\[[\s\S]*\]', cleaned)
                if match:
                    cleaned = match.group(0)

            try:
                questions = json.loads(cleaned, strict=False)
            except json.JSONDecodeError:
                cleaned = AIService._fix_json_escapes(cleaned)
                questions = json.loads(cleaned, strict=False)
            if not isinstance(questions, list):
                raise ValueError(f"Expected JSON array, got {type(questions).__name__}")

            # Build separate answer key
            answer_key = []
            for q in questions:
                answer_key.append({
                    "id": q.get("id"),
                    "correctAnswer": q.get("correct_answer", ""),
                    "explanation": q.get("explanation", ""),
                })
                # Map points field for frontend compatibility
                q["points"] = q.get("marks", 1)

            return questions, answer_key
        except Exception as e:
            print(f"[AIService] Evaluation paper generation failed. Error: {e}", flush=True)
            print(f"[AIService] Raw AI response (first 500 chars): {response[:500]}", flush=True)
            raise ValueError(f"Failed to generate evaluation paper: {e}")

    async def generate_follow_up_questions(
        self, user_message: str, ai_response: str, count: int = 4
    ) -> List[str]:
        """Generate questions that dive deeper into the same topic — clarifying or expanding the previous answer."""
        prompt = (
            f"You are helping a student go deeper into a topic. Based on this Q&A exchange, generate {count} "
            "follow-up questions that clarify, expand, or explore nuances of the SAME topic that was just discussed. "
            "These should be backward/vertical questions — digging deeper into what was just explained, "
            "NOT pivoting to a different topic or action.\n\n"
            "Good examples for 'What is the Big Bang theory?':\n"
            '  - "What evidence supports the Big Bang theory?"\n'
            '  - "What happened in the first few seconds after the Big Bang?"\n'
            '  - "Who first proposed the Big Bang theory and how was it discovered?"\n\n'
            f"Student asked: {user_message[:600]}\n\n"
            f"AI responded: {ai_response[:1200]}\n\n"
            f"Return ONLY a JSON array of {count} question strings that go deeper into this specific topic. "
            'Example: ["Why did X happen?", "How exactly does Y work?", "What is the difference between X and Z?"]'
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            questions = json.loads(cleaned)
            if isinstance(questions, list):
                return [str(q) for q in questions[:count]]
        except Exception:
            pass
        return []

    async def generate_next_steps(
        self, user_message: str, ai_response: str, count: int = 4
    ) -> list[str]:
        """Generate actionable prompts that move the student forward — broader topics or direct actions."""
        prompt = (
            f"You are helping a student decide what to do NEXT after reading an AI response. "
            f"Generate {count} next-step prompts that are forward/horizontal — they should either:\n"
            "  (a) Initiate an ACTION on this content: summarize, quiz, compare, create flashcards, practice problems, etc.\n"
            "  (b) Broaden to a RELATED topic or concept that naturally follows from what was discussed.\n\n"
            "These should NOT be questions that go deeper into the same topic (those are follow-up questions).\n\n"
            "Good examples for 'What is the Big Bang theory?':\n"
            '  - "Summarize the Big Bang theory in 5 bullet points"\n'
            '  - "Create a 5-question quiz on the Big Bang theory"\n'
            '  - "Compare the Big Bang theory vs the Steady State theory"\n'
            '  - "Explain the Big Bounce theory"\n\n'
            f"Student asked: {user_message[:600]}\n\n"
            f"AI responded: {ai_response[:1200]}\n\n"
            f"Return ONLY a JSON array of {count} short action-oriented strings the student can send as their next message. "
            'Example: ["Summarize this in simple terms", "Give me 5 practice questions on this", "Compare X and Y"]'
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            steps = json.loads(cleaned)
            if isinstance(steps, list):
                return [str(s) for s in steps[:count]]
        except Exception:
            pass
        return []

    async def generate_practice_exercises(
        self,
        user_message: str,
        ai_response: str,
        count: int = 3,
        grade: int | None = None,
        difficulty: str = "medium",
    ) -> list[dict]:
        """Generate quick practice exercises from a chat Q&A exchange.

        Returns a list of dicts, each with keys: question, answer, type, options (optional).
        """
        grade_hint = f" for grade {grade} students" if grade else ""
        prompt = (
            f"You are creating {count} quick practice exercises{grade_hint} based on the following Q&A exchange. "
            f"Difficulty: {difficulty}.\n\n"
            f"Student asked: {user_message[:600]}\n\n"
            f"AI responded: {ai_response[:1500]}\n\n"
            "Generate a MIX of question types. Return ONLY a JSON array with this exact format:\n"
            "[\n"
            '  {"question": "What is...?", "answer": "Concise answer here.", "type": "short"},\n'
            '  {"question": "Which of the following...?", "answer": "B", "type": "mcq", '
            '"options": ["option A", "correct option B", "option C", "option D"]},\n'
            '  {"question": "Statement to evaluate.", "answer": "True", "type": "true_false"}\n'
            "]\n\n"
            "Rules:\n"
            "- Use exactly these types: short, mcq, true_false\n"
            "- MCQ must have exactly 4 options and the answer must be the letter (A/B/C/D)\n"
            "- true_false answer must be exactly 'True' or 'False'\n"
            "- short answer should be 1-3 sentences\n"
            "- Questions must be directly based on the content discussed\n"
            f"- Generate exactly {count} exercises\n"
            "- Return ONLY the JSON array, no markdown fences or extra text"
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            exercises = json.loads(cleaned)
            if isinstance(exercises, list):
                result = []
                for ex in exercises[:count]:
                    if isinstance(ex, dict) and "question" in ex and "answer" in ex:
                        item = {
                            "question": str(ex["question"]),
                            "answer": str(ex["answer"]),
                            "type": ex.get("type", "short"),
                        }
                        if ex.get("options") and isinstance(ex["options"], list):
                            item["options"] = [str(o) for o in ex["options"][:4]]
                        result.append(item)
                return result
        except Exception:
            pass
        return []

    async def extract_video_search_query(
        self, user_message: str, ai_response: str,
        grade: int | None = None, student_mode: bool = False,
        language: str | None = None,
    ) -> str:
        """Extract the best YouTube search query from a Q&A exchange."""
        grade_instruction = ""
        if grade:
            grade_instruction = (
                f"\nThe student is in grade {grade}. The search query MUST include "
                f"'grade {grade}' or 'class {grade}' to find age-appropriate content.\n"
            )
        elif student_mode:
            grade_instruction = (
                "\nStudent mode is on. Prefer educational/tutorial-style videos "
                "suitable for students.\n"
            )

        lang_map = {"en": "English", "hi": "Hindi", "ta": "Tamil"}
        lang_name = lang_map.get((language or "").lower())
        language_instruction = ""
        if lang_name and lang_name.lower() != "english":
            language_instruction = (
                f"\nThe user wants videos in {lang_name}. Write the search query "
                f"in {lang_name} (use the {lang_name} script for the topic terms) "
                f"and append the word '{lang_name}' at the end so YouTube returns "
                f"videos in that language.\n"
            )

        prompt = (
            "Extract a concise, specific YouTube search query (5-10 words max) that would find "
            "the most relevant educational video for this topic.\n"
            f"{grade_instruction}"
            f"{language_instruction}\n"
            f"User asked: {user_message[:300]}\n"
            f"Topic summary: {ai_response[:400]}\n\n"
            "Return ONLY the search query string, nothing else."
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        return response.strip().strip('"').strip("'")[:100]

    # ── Chat Mind Map & Infographic ──────────────────────────────────────────

    async def generate_chat_mindmap(
        self,
        user_message: str,
        ai_response: str,
        grade: int | None = None,
        language: str = "English",
    ) -> dict:
        """Generate a structured mind map JSON from a chat Q&A exchange."""
        grade_str = f"grade {grade}" if grade else "general"
        lang_name = self._LANGUAGE_NAMES.get(language.lower(), language)
        lang_rule = (
            f"14. LANGUAGE — ALL node labels MUST be written in {lang_name}. "
            + (
                f"Use {lang_name} script exclusively — do NOT write labels in English or any other script. "
                "Proper nouns, chemical formulas, and standard scientific symbols may stay in their original form."
                if lang_name.lower() != "english"
                else "Write all node labels in English."
            )
        )
        prompt = f"""You are a mind-map extractor. Your ONLY job is to read the AI explanation below and
reorganise the SAME information into a hierarchical mind map. Do NOT add external knowledge.
Every single node label MUST come directly from a heading, key phrase, term, fact, or example
that appears in the explanation text.

Topic: {user_message[:500]}
Grade Level: {grade_str}
Language: {lang_name}

── AI EXPLANATION (this is your ONLY source — extract nodes from here) ──
{ai_response[:4000]}
── END OF EXPLANATION ──

STRICT Rules:
1. Root node "label" = the main topic (2-6 words).
2. First-level children = the major headings / sections from the explanation.
3. Deeper children = the key facts, terms, examples, dates, or details under each section.
4. Every node label must be a phrase that is clearly present in or directly derived from the explanation above.
5. Do NOT invent new facts, do NOT add information not in the explanation.
6. Keep labels concise (2-6 words). 3-4 depth levels minimum.
7. Aim for 4-8 first-level children, each with 2-5 sub-children.
8. No repetition. No markdown. No code fences. Raw JSON only.

ACCURACY & SPELLING RULES (CRITICAL — strictly follow):
9. Every word in every node label MUST be spelled correctly. Double-check all scientific terms, proper nouns, dates, and technical vocabulary.
10. Copy names, formulas, and technical terms EXACTLY as they appear in the explanation. Do NOT rephrase them in a way that introduces errors.
11. If the explanation mentions a person, place, formula, or date — verify it matches the source text character-by-character before including it.
12. Do NOT abbreviate words in a way that changes meaning or creates ambiguity.
13. Use proper capitalisation for proper nouns and sentence-case for other labels.
{lang_rule}

Return ONLY this JSON structure:
{{
  "label": "Main Topic",
  "children": [
    {{
      "label": "Section Heading",
      "children": [
        {{
          "label": "Key Fact A",
          "children": [
            {{ "label": "Detail" }}
          ]
        }}
      ]
    }}
  ]
}}"""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
            if fence_match:
                cleaned = fence_match.group(1).strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = cleaned[start:end]
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    json_str = json_str.replace("'", '"')
                    result = json.loads(json_str)
                if isinstance(result, dict) and "label" in result:
                    if "children" not in result:
                        result["children"] = []
                    return result
        except Exception:
            pass

        # Fallback: build a simple mind map from the AI response text
        sentences = re.split(r'(?<=[.!?])\s+', ai_response.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        children = []
        chunk_size = max(1, len(sentences) // 5) if len(sentences) > 5 else 1
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            if len(chunk) == 1:
                children.append({"label": chunk[0][:60], "children": []})
            else:
                sub_children = [{"label": s[:60]} for s in chunk[1:]]
                children.append({"label": chunk[0][:60], "children": sub_children})
        if not children:
            children = [{"label": ai_response[:60], "children": []}]
        return {"label": user_message[:50], "children": children[:8]}

    @staticmethod
    def _svg_escape(text: str) -> str:
        return (str(text)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))

    @staticmethod
    def _svg_wrap(text: str, max_chars: int = 52) -> list[str]:
        """Break text into lines at word boundaries."""
        words = text.split()
        lines, current = [], ""
        for word in words:
            if len(current) + len(word) + 1 > max_chars:
                if current:
                    lines.append(current)
                current = word
            else:
                current = (current + " " + word).strip()
        if current:
            lines.append(current)
        return lines or [""]

    @classmethod
    def _render_infographic_svg(cls, data: dict) -> str:
        """Render an infographic as a visually rich, magazine-quality SVG poster.

        Layout: bold header banner → 2-column section grid with icon + stat badge
        per card → highlighted key-takeaway ribbon at the bottom.
        """
        esc = cls._svg_escape
        wrap = cls._svg_wrap

        title    = esc((data.get("title") or "Infographic")[:60])
        subtitle = esc((data.get("subtitle") or "")[:90])
        sections = (data.get("sections") or [])[:4]
        kt       = (data.get("keyTakeaway") or "")[:200]

        # Per-section colour themes: (accent, card-bg, light-text, badge-bg)
        THEMES = [
            ("#3B82F6", "#0c1e3b", "#93c5fd", "#1d4ed8"),   # blue
            ("#10B981", "#052e1c", "#6ee7b7", "#065f46"),   # emerald
            ("#F59E0B", "#2d1a00", "#fcd34d", "#92400e"),   # amber
            ("#EF4444", "#2d0a0a", "#fca5a5", "#7f1d1d"),   # red
        ]
        # Emoji icons per section slot
        SEC_ICONS = ["🔹", "🟢", "🟡", "🔴"]

        W      = 960
        M      = 20          # outer margin
        GAP    = 16          # column gap
        CW     = (W - 2*M - GAP) // 2  # card width
        ICON_H = 70          # top icon zone height
        FACT_H = 26          # height per fact line
        BAD_H  = 38          # highlight badge height

        def card_height(sec: dict) -> int:
            facts = (sec.get("facts") or sec.get("points") or [])[:4]
            n_facts = sum(max(len(wrap(str(f), 44)), 1) for f in facts)
            return ICON_H + 8 + n_facts * FACT_H + (BAD_H + 12 if sec.get("highlight") else 0) + 16

        rows: list[list[dict]] = [sections[i:i+2] for i in range(0, len(sections), 2)]
        row_h = [max((card_height(s) for s in row), default=120) for row in rows]

        HEADER_H  = 96
        SECS_H    = sum(row_h) + max(len(rows)-1, 0)*GAP
        kt_lines  = wrap(kt, 100) if kt else []
        KT_H      = len(kt_lines)*20 + 40 if kt_lines else 0
        H = HEADER_H + M + SECS_H + (M + KT_H if KT_H else 0) + M

        o: list[str] = []
        o.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'viewBox="0 0 {W} {H}" width="{W}" height="{H}">')

        # ── Gradients & pattern ─────────────────────────────────────────────
        o.append("<defs>")
        o.append('<linearGradient id="gbg" x1="0" y1="0" x2="0" y2="1">'
                 '<stop offset="0%" stop-color="#071428"/>'
                 '<stop offset="100%" stop-color="#0b1220"/></linearGradient>')
        o.append('<linearGradient id="ghdr" x1="0" y1="0" x2="1" y2="0">'
                 '<stop offset="0%" stop-color="#0e2a5a"/>'
                 '<stop offset="50%" stop-color="#0a1628"/>'
                 '<stop offset="100%" stop-color="#130b2e"/></linearGradient>')
        for i, (acc, cbg, lt, bb) in enumerate(THEMES):
            o.append(f'<linearGradient id="gc{i}" x1="0" y1="0" x2="0" y2="1">'
                     f'<stop offset="0%" stop-color="{acc}" stop-opacity="0.22"/>'
                     f'<stop offset="100%" stop-color="{cbg}" stop-opacity="0.95"/>'
                     f'</linearGradient>')
        # Subtle dot-grid background pattern
        o.append('<pattern id="dots" width="36" height="36" patternUnits="userSpaceOnUse">'
                 '<circle cx="18" cy="18" r="1" fill="#1e3a5f" fill-opacity="0.55"/>'
                 '</pattern>')
        o.append("</defs>")

        # ── Background ──────────────────────────────────────────────────────
        o.append(f'<rect width="{W}" height="{H}" fill="url(#gbg)"/>')
        o.append(f'<rect width="{W}" height="{H}" fill="url(#dots)"/>')

        # ── Header ──────────────────────────────────────────────────────────
        o.append(f'<rect width="{W}" height="{HEADER_H}" fill="url(#ghdr)"/>')
        # Rainbow top stripe
        stripe_colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
        sw = W // len(stripe_colors)
        for si, sc in enumerate(stripe_colors):
            o.append(f'<rect x="{si*sw}" y="0" width="{sw}" height="5" fill="{sc}"/>')
        # Title
        o.append(f'<text x="{W//2}" y="48" text-anchor="middle" '
                 f'font-family="Arial Black,Arial,Helvetica,sans-serif" font-size="32" '
                 f'font-weight="900" fill="#ffffff" letter-spacing="1.2">{title}</text>')
        if subtitle:
            o.append(f'<text x="{W//2}" y="76" text-anchor="middle" '
                     f'font-family="Arial,Helvetica,sans-serif" font-size="14" '
                     f'font-style="italic" fill="#93c5fd">{subtitle}</text>')
        # Bottom border of header
        o.append(f'<line x1="0" y1="{HEADER_H}" x2="{W}" y2="{HEADER_H}" '
                 f'stroke="#1e3a5f" stroke-width="2"/>')

        # ── Section cards ───────────────────────────────────────────────────
        y_off = HEADER_H + M
        for ri, row in enumerate(rows):
            rh = row_h[ri]
            for ci, sec in enumerate(row):
                si = ri*2 + ci
                acc, cbg, lt, bb = THEMES[si % len(THEMES)]
                icon_em = SEC_ICONS[si % len(SEC_ICONS)]
                heading  = esc((sec.get("heading") or "")[:44])
                facts    = (sec.get("facts") or sec.get("points") or [])[:4]
                highlight = (sec.get("highlight") or "")[:30]

                cx = M + ci*(CW + GAP)
                cy = y_off

                # Card body
                o.append(f'<rect x="{cx}" y="{cy}" width="{CW}" height="{rh}" '
                         f'rx="14" fill="url(#gc{si % len(THEMES)})" '
                         f'stroke="{acc}" stroke-width="1.5" stroke-opacity="0.45"/>')
                # Top accent bar (full width)
                o.append(f'<rect x="{cx}" y="{cy}" width="{CW}" height="5" '
                         f'rx="2" fill="{acc}"/>')

                # Icon zone background
                o.append(f'<rect x="{cx}" y="{cy+5}" width="{CW}" height="{ICON_H-5}" '
                         f'fill="{acc}" fill-opacity="0.12"/>')
                # Large icon emoji
                o.append(f'<text x="{cx+36}" y="{cy+46}" '
                         f'font-family="Segoe UI Emoji,Apple Color Emoji,sans-serif" '
                         f'font-size="32" fill="{acc}" dominant-baseline="central">'
                         f'{icon_em}</text>')
                # Heading
                o.append(f'<text x="{cx+80}" y="{cy+38}" '
                         f'font-family="Arial,Helvetica,sans-serif" font-size="15" '
                         f'font-weight="bold" fill="{lt}">{heading}</text>')
                # Section number badge
                o.append(f'<circle cx="{cx+CW-20}" cy="{cy+25}" r="14" '
                         f'fill="{bb}" fill-opacity="0.7"/>')
                o.append(f'<text x="{cx+CW-20}" y="{cy+25}" text-anchor="middle" '
                         f'dominant-baseline="central" '
                         f'font-family="Arial,Helvetica,sans-serif" font-size="13" '
                         f'font-weight="bold" fill="#fff">{si+1}</text>')

                # Divider
                o.append(f'<line x1="{cx+12}" y1="{cy+ICON_H+4}" '
                         f'x2="{cx+CW-12}" y2="{cy+ICON_H+4}" '
                         f'stroke="{acc}" stroke-width="1" stroke-opacity="0.35"/>')

                # Facts — render with line-wrapping to match the card_height calculation
                fy = cy + ICON_H + 14
                for fact in facts:
                    fact_lines = wrap(str(fact), 44)
                    # Bullet dot aligned to first line
                    o.append(f'<circle cx="{cx+22}" cy="{fy+8}" r="3.5" fill="{acc}" fill-opacity="0.9"/>')
                    for li, line in enumerate(fact_lines):
                        o.append(f'<text x="{cx+34}" y="{fy + 8 + li*14}" '
                                 f'font-family="Arial,Helvetica,sans-serif" font-size="12" '
                                 f'dominant-baseline="central" fill="#e2e8f0">{esc(line)}</text>')
                    fy += max(len(fact_lines), 1) * FACT_H

                # Highlight / stat badge
                if highlight:
                    by = cy + rh - BAD_H - 10
                    bx = cx + 12
                    bw = CW - 24
                    o.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{BAD_H}" '
                             f'rx="{BAD_H//2}" fill="{bb}" fill-opacity="0.25" '
                             f'stroke="{acc}" stroke-width="1.5" stroke-opacity="0.7"/>')
                    o.append(f'<text x="{bx + bw//2}" y="{by + BAD_H//2}" '
                             f'text-anchor="middle" dominant-baseline="central" '
                             f'font-family="Arial,Helvetica,sans-serif" font-size="13" '
                             f'font-weight="bold" fill="{lt}">{esc(highlight)}</text>')

            y_off += rh + GAP

        # ── Key Takeaway ────────────────────────────────────────────────────
        if kt_lines:
            kty = y_off + M//2
            o.append(f'<rect x="{M}" y="{kty}" width="{W-2*M}" height="{KT_H}" '
                     f'rx="12" fill="#022c22" fill-opacity="0.85" '
                     f'stroke="#10B981" stroke-width="1.5" stroke-opacity="0.6"/>')
            o.append(f'<rect x="{M}" y="{kty}" width="6" height="{KT_H}" '
                     f'rx="3" fill="#10B981"/>')
            # Lightbulb icon
            o.append(f'<text x="{M+20}" y="{kty + 22}" '
                     f'font-family="Segoe UI Emoji,Apple Color Emoji,sans-serif" '
                     f'font-size="18" fill="#10B981">💡</text>')
            o.append(f'<text x="{M+46}" y="{kty + 22}" '
                     f'font-family="Arial,Helvetica,sans-serif" font-size="12" '
                     f'font-weight="bold" letter-spacing="1" fill="#10B981">KEY TAKEAWAY</text>')
            for li, line in enumerate(kt_lines[:3]):
                o.append(f'<text x="{M+20}" y="{kty + 38 + li*20}" '
                         f'font-family="Arial,Helvetica,sans-serif" font-size="12.5" '
                         f'fill="#6ee7b7">{esc(line)}</text>')

        o.append("</svg>")
        return "\n".join(o)

    async def _generate_infographic_svg_fallback(
        self, topic: str, content: str, grade: int | None, language: str = "English",
    ) -> dict | None:
        """Generate a structured infographic via the chat model when image generation fails."""
        grade_str = f"grade {grade}" if grade else "general"
        lang_name = self._LANGUAGE_NAMES.get(language.lower(), language)
        lang_req = (
            f"ALL text (title, subtitle, section headings, facts, keyTakeaway) MUST be written in {lang_name}. "
            + (
                f"Use {lang_name} script exclusively — do NOT mix English or other scripts. "
                "Proper nouns, numbers, and formulas may stay in standard form."
                if lang_name.lower() != "english"
                else "Write all text in English."
            )
        )
        prompt = f"""You are an infographic designer. Convert this into a structured infographic layout.

Topic: {topic}
Grade Level: {grade_str}
Language: {lang_name}

LANGUAGE REQUIREMENT (MANDATORY): {lang_req}

── AI EXPLANATION ──
{content}
── END ──

ACCURACY & SPELLING (CRITICAL):
- Every word, name, term, date, and formula in the output MUST be spelled correctly.
- Copy scientific terms, proper nouns, and formulas EXACTLY from the explanation above.
- Double-check ALL text strings before including them — spelling mistakes make the infographic unusable.
- Use proper capitalisation for headings and proper nouns.

Return ONLY raw JSON with: title, subtitle, sections (array of heading/icon/color/facts/highlight), keyTakeaway.
Icons: BookOpen, Globe, Clock, Lightbulb, Users, Star, Target, Zap, Award, Shield, Brain, Heart, TrendingUp, BarChart, Layers"""
        try:
            response = await self.chat([{"role": "user", "content": prompt}])
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            # Strip any trailing ``` that got captured
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
            data = json.loads(cleaned)
            # Ensure sections have the right structure
            for sec in data.get("sections", []):
                if "points" in sec and "facts" not in sec:
                    sec["facts"] = sec.pop("points")
            svg = self._render_infographic_svg(data)
            import base64
            svg_b64 = base64.b64encode(svg.encode()).decode()
            data_uri = f"data:image/svg+xml;base64,{svg_b64}"
            return {"image_base64": data_uri, "title": topic, "mode": "image"}
        except Exception as e:
            print(f"[Infographic SVG fallback] failed: {e}", flush=True)
            return None

    async def _prepare_infographic_labels(
        self,
        topic: str,
        explanation: str,
        language: str,
        grade: int | None,
    ) -> dict:
        """Use the text model to extract concise, spell-checked labels for the infographic.

        Returns a dict: {title, sections: [{heading, facts}], key_takeaway}.
        Keeping text generation separate from image generation prevents the image
        model from inventing or misspelling words.
        """
        lang_name = self._LANGUAGE_NAMES.get(language.lower(), language)
        grade_str = f"grade {grade}" if grade else "general"
        is_non_english = lang_name.lower() != "english"

        lang_rule = (
            f"Write ALL output text in {lang_name} using correct {lang_name} orthography. "
            f"Do NOT transliterate English into {lang_name} script."
            if is_non_english
            else "Write all output text in English."
        )

        prompt = f"""Extract concise, correctly-spelled text labels for an educational infographic.

Topic: {topic}
Grade Level: {grade_str}
Language: {lang_name}
{lang_rule}

── SOURCE TEXT ──
{explanation[:2500]}
── END ──

Rules:
- Title: 3-7 words, captures the main topic exactly.
- Sections: 3-4 sections maximum. Each has a 2-4 word heading and 2-3 short facts (≤12 words each).
- Key Takeaway: one sentence, ≤15 words.
- Every word must be spelled perfectly — double-check before including.
- Extract ONLY from the source text — do not invent content.

Return ONLY valid JSON (no markdown):
{{
  "title": "...",
  "sections": [
    {{"heading": "...", "facts": ["...", "...", "..."]}},
    {{"heading": "...", "facts": ["...", "..."]}}
  ],
  "key_takeaway": "..."
}}"""
        try:
            response = await self.chat([{"role": "user", "content": prompt}])
            parsed = self._parse_json_response(response)
            if parsed and parsed.get("title") and parsed.get("sections"):
                return parsed
        except Exception as e:
            print(f"[Infographic] label prep failed: {e}", flush=True)
        return {"title": topic[:60], "sections": [], "key_takeaway": ""}

    def _infer_topic_visual_hints(self, title: str, explanation: str) -> dict:
        combined = (title + " " + explanation[:600]).lower()

        math_kw = [
            "theorem", "equation", "formula", "triangle", "geometry", "algebra",
            "calculus", "proof", "angle", "circle", "polygon", "arithmetic",
            "derivative", "integral", "matrix", "vector", "probability", "statistics",
            "hypotenuse", "pythagoras", "quadratic", "fraction", "ratio", "exponent",
        ]
        science_kw = [
            "cell", "atom", "molecule", "organism", "photosynthesis", "evolution",
            "ecosystem", "chemical", "reaction", "force", "energy", "wave", "light",
            "gravity", "electric", "magnetic", "dna", "gene", "protein", "virus",
            "bacteria", "periodic", "element", "compound", "physics", "biology",
            "chemistry", "nucleus", "electron", "osmosis", "mitosis",
        ]
        history_kw = [
            "war", "revolution", "empire", "dynasty", "century", "ancient", "medieval",
            "civilization", "king", "queen", "battle", "treaty", "independence",
            "colonial", "historical", "period", "era", "decade", "event", "reign",
            "republic", "democracy", "constitution", "freedom", "liberation",
        ]
        geo_kw = [
            "continent", "country", "capital", "ocean", "mountain", "river", "climate",
            "population", "region", "map", "latitude", "longitude", "biome",
            "volcano", "earthquake", "weather", "atmosphere", "soil", "erosion",
            "desert", "rainfall", "tundra", "tropical",
        ]
        lang_kw = [
            "grammar", "vocabulary", "sentence", "verb", "noun", "adjective", "tense",
            "literature", "poetry", "writing", "reading", "comprehension", "metaphor",
            "simile", "narrative", "prose", "essay", "dialogue", "author", "clause",
            "punctuation", "phonics", "rhyme",
        ]
        econ_kw = [
            "supply", "demand", "market", "price", "economy", "trade", "gdp",
            "inflation", "tax", "budget", "profit", "cost", "investment", "bank",
            "currency", "stock", "entrepreneurship", "goods", "services", "scarcity",
        ]

        if any(k in combined for k in math_kw):
            return {
                "category": "mathematics",
                "color_scheme": "Deep navy (#0F172A) background with gold (#F59E0B), cyan (#06B6D4), and white text.",
                "layout_desc": (
                    "Large central diagram or formula occupying ~40% of the poster (e.g. labelled triangle, "
                    "graph, number line, or geometric figure). Surrounding info panels for definitions, "
                    "worked-example steps, and real-world applications."
                ),
                "visual_hint": (
                    "Draw mathematically accurate diagrams: labelled triangles with sides a/b/c, graphs with axes, "
                    "geometric shapes with angle marks. Color-code variables (e.g. hypotenuse in gold). "
                    "Render key formulas in large, clear typography — e.g. a² + b² = c². "
                    "Add small step-by-step worked example box. No generic clipart."
                ),
            }
        elif any(k in combined for k in science_kw):
            return {
                "category": "science",
                "color_scheme": "Dark (#111827) background with green (#10B981), blue (#3B82F6), and white text.",
                "layout_desc": (
                    "Central annotated scientific diagram (cell, atom, food chain, circuit, etc.) with "
                    "leader-line labels. Surrounding panels explain stages/parts, examples, and significance."
                ),
                "visual_hint": (
                    "Draw a detailed, realistic scientific illustration specific to the topic — e.g. a plant cell "
                    "with labelled organelles, an atom with electron shells, or a food web with arrows. "
                    "Use leader lines for all labels. Show cause-and-effect with directional arrows. "
                    "Include a process-flow diagram if the topic involves stages (e.g. mitosis, photosynthesis)."
                ),
            }
        elif any(k in combined for k in history_kw):
            return {
                "category": "history",
                "color_scheme": "Dark sepia (#1C1410) background with amber (#D97706), cream (#FEF3C7), and white text.",
                "layout_desc": (
                    "Horizontal or vertical timeline spine as the central visual. Event cards with dates branch "
                    "off the spine. Side panels provide context, key figures, causes, and consequences."
                ),
                "visual_hint": (
                    "Draw a clear dated timeline with event markers. Add small silhouette illustrations: "
                    "period maps, flags, monuments, or portrait silhouettes of key figures. "
                    "Use parchment-textured panels for event cards. Show cause → event → consequence flow."
                ),
            }
        elif any(k in combined for k in geo_kw):
            return {
                "category": "geography",
                "color_scheme": "Dark ocean (#0C1A2E) background with teal (#0D9488), earth-tone (#92400E), and white text.",
                "layout_desc": (
                    "Central map or topographic diagram with annotations. Surrounding panels for climate data, "
                    "key statistics, comparisons, and significance."
                ),
                "visual_hint": (
                    "Draw a simplified map or geographic cross-section relevant to the topic. "
                    "Add compass rose, scale bar, elevation shading, or climate zone color bands. "
                    "Use icons for key geographic features (mountains, rivers, volcanoes). "
                    "Include a small data chart (e.g. rainfall bar chart, population graph) if relevant."
                ),
            }
        elif any(k in combined for k in lang_kw):
            return {
                "category": "language",
                "color_scheme": "Deep purple (#1E1B4B) background with rose (#F43F5E), yellow (#FCD34D), and white text.",
                "layout_desc": (
                    "Typography-led layout. Large example sentences or text excerpts in highlighted boxes. "
                    "Sentence-structure diagrams or word-tree branching diagrams as the central visual."
                ),
                "visual_hint": (
                    "Use large, beautiful typography as a design element. Draw sentence structure diagrams "
                    "with labeled parts (subject/verb/object). Show example sentences in speech-bubble or "
                    "highlighted-quote style. Use word-web or tree diagram to show relationships. "
                    "Color-code grammar components (verbs in red, nouns in blue, etc.)."
                ),
            }
        elif any(k in combined for k in econ_kw):
            return {
                "category": "economics",
                "color_scheme": "Dark (#0F172A) background with emerald (#059669), orange (#EA580C), and white text.",
                "layout_desc": (
                    "Central economic diagram: supply-demand curve, flow cycle, or bar/pie chart. "
                    "Surrounding panels explain key concepts, real-world examples, and impacts."
                ),
                "visual_hint": (
                    "Draw accurate economic diagrams: supply-demand curves with labeled axes (Price vs Quantity), "
                    "circular flow of income diagram, or comparative bar charts. "
                    "Show economic relationships with directional arrows. "
                    "Use icons (coins, factories, shopping carts) to make abstract concepts concrete."
                ),
            }
        else:
            return {
                "category": "general",
                "color_scheme": "Dark (#111827) background with vibrant contrasting accent colours per section.",
                "layout_desc": (
                    "Strong title banner at top. 3-4 distinct content panels, each with a unique topic-relevant "
                    "icon or mini-diagram. Key takeaway bar at the bottom."
                ),
                "visual_hint": (
                    "Create specific, meaningful visuals tied to the topic — flowcharts for processes, "
                    "comparison tables for contrasting concepts, numbered steps for procedures, "
                    "or annotated diagrams for structures. Avoid generic clipart. "
                    "Each panel should contain a distinct visual element, not just text."
                ),
            }

    async def generate_chat_infographic(
        self,
        user_message: str,
        ai_response: str,
        grade: int | None = None,
        language: str = "English",
    ) -> dict:
        """Generate an infographic using a three-step approach that guarantees spelling accuracy.

        Step 1 — Text model extracts spell-checked labels (title, sections, takeaway).
        Step 2 — SVG is rendered immediately from those labels. SVG text is never
                  touched by an image model, so spelling is 100% guaranteed.
        Step 3 — Image model is attempted with an extremely tight spelling-enforcement
                  prompt. If it succeeds, the richer image is returned. If it fails or
                  times out, the accurate SVG from Step 2 is returned instead.
        """
        import base64
        import asyncio

        grade_str = f"grade {grade}" if grade else "general"
        topic = user_message[:200].strip()
        explanation = ai_response[:3000]
        lang_name = self._LANGUAGE_NAMES.get(language.lower(), language)
        is_non_english = lang_name.lower() != "english"

        # ── Step 1: spell-checked labels from text model ─────────────────────
        labels = await self._prepare_infographic_labels(topic, explanation, language, grade)
        title_text = labels.get("title") or topic[:60]
        key_takeaway = labels.get("key_takeaway", "")
        sections = labels.get("sections", [])[:4]

        # ── Step 2: build SVG immediately — guaranteed accurate text ──────────
        _panel_colors = ["#991B1B", "#065F46", "#1E3A5F", "#4C1D95"]
        svg_data = {
            "title": title_text,
            "subtitle": "",
            "sections": [
                {
                    "heading": s.get("heading", ""),
                    "icon": ["BookOpen", "Target", "Zap", "Award"][i % 4],
                    "color": _panel_colors[i % 4],
                    "facts": s.get("facts", [])[:3],
                    "highlight": "",
                }
                for i, s in enumerate(sections)
            ],
            "keyTakeaway": key_takeaway,
        }
        try:
            svg = self._render_infographic_svg(svg_data)
            svg_b64 = base64.b64encode(svg.encode()).decode()
            svg_result: dict = {
                "image_base64": f"data:image/svg+xml;base64,{svg_b64}",
                "title": title_text,
                "mode": "image",
            }
        except Exception as svg_err:
            print(f"[Infographic] SVG build failed: {svg_err}", flush=True)
            svg_result = {}

        # ── Step 3: attempt image model with tight spelling enforcement ───────
        def _spell_word(w: str) -> str:
            if len(w) > 5 and w[0].isupper():
                return f"{w} [{'-'.join(w.upper())}]"
            return w

        title_spelled = " ".join(_spell_word(w) for w in title_text.split())

        text_registry: list[str] = [f'TITLE: "{title_text}"']
        sections_detail = ""
        for i, s in enumerate(sections):
            heading = s.get("heading", "")
            facts = s.get("facts", [])[:3]
            text_registry.append(f'SECTION {i+1} HEADING: "{heading}"')
            text_registry.extend(f'  BULLET: "{f}"' for f in facts)
            sections_detail += (
                f"\nSECTION {i+1} — heading (copy exactly): {heading}\n"
                + "\n".join(f"  • {f}" for f in facts) + "\n"
            )
        if key_takeaway:
            text_registry.append(f'KEY TAKEAWAY: "{key_takeaway}"')

        lang_note = (
            f"All text is in {lang_name} — render every character exactly as given, "
            f"using correct {lang_name} script/orthography."
            if is_non_english
            else "All text is in English — render every character exactly as given."
        )

        visual_hints = self._infer_topic_visual_hints(title_text, explanation)

        image_prompt = (
            f"Create a high-quality educational infographic poster for {grade_str} students.\n"
            f"TOPIC: {title_text}\n"
            f"SUBJECT: {visual_hints['category'].upper()}\n\n"
            f"TOPIC CONTEXT (use this to draw accurate, content-specific visuals):\n{explanation[:500]}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "VISUAL DESIGN (follow precisely)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Layout: {visual_hints['layout_desc']}\n"
            f"Color scheme: {visual_hints['color_scheme']}\n"
            f"Illustrations: {visual_hints['visual_hint']}\n"
            "- Title banner at the very top — bold, large, exact title text\n"
            "- Key takeaway highlighted bar pinned at the very bottom\n"
            "- High information density — every panel must contain topic-specific visuals, not generic shapes\n"
            "- Academically accurate — diagrams, labels, and data must be factually correct\n"
            "- Student-friendly and visually engaging — designed for classroom use\n"
            f"- {lang_note}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️  MANDATORY TEXT — COPY VERBATIM, ZERO SPELLING ERRORS\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"TITLE (letter-for-letter): {title_spelled}\n"
            f"{sections_detail}\n"
            f"KEY TAKEAWAY (word-for-word): {key_takeaway}\n\n"
            "Complete text registry — every string must appear exactly as listed:\n"
            + "\n".join(f"  {t}" for t in text_registry) + "\n\n"
            "Text rules:\n"
            "1. Copy EVERY word from the registry above exactly — zero alterations.\n"
            "2. Do NOT paraphrase, abbreviate, reorder, or add/remove letters.\n"
            "3. Proper nouns: follow the letter-by-letter spelling shown in brackets.\n"
            "4. If unsure about a word, leave the space blank rather than guessing.\n"
        )

        client = self._get_gemini_async()
        if client:
            from google.genai import types
            image_model = settings.AI_IMAGE_MODEL
            configs_to_try = [
                types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
                types.GenerateContentConfig(response_modalities=["IMAGE"]),
                None,
            ]
            for attempt in range(3):
                cfg = configs_to_try[min(attempt, len(configs_to_try) - 1)]
                try:
                    call_kwargs: dict = dict(model=image_model, contents=[image_prompt])
                    if cfg is not None:
                        call_kwargs["config"] = cfg
                    resp = await asyncio.wait_for(
                        client.aio.models.generate_content(**call_kwargs),
                        timeout=60.0,
                    )
                    for part in resp.parts:
                        if getattr(part, "inline_data", None) is not None:
                            img_bytes = part.inline_data.data
                            mime = part.inline_data.mime_type or "image/png"
                            data_uri = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"
                            print(f"[Infographic] image generated ({len(img_bytes)} bytes)", flush=True)
                            return {"image_base64": data_uri, "title": title_text, "mode": "image"}
                    print(f"[Infographic] no image part (attempt={attempt+1})", flush=True)
                    if attempt < 2:
                        await asyncio.sleep(1.0)
                        continue
                    break
                except asyncio.TimeoutError:
                    print(f"[Infographic] timed out on attempt {attempt + 1}", flush=True)
                    if attempt < 2:
                        await asyncio.sleep(1.0)
                        continue
                    break
                except Exception as e:
                    if _is_retryable(e) and attempt < 2:
                        wait = 1.0 * (2 ** attempt)
                        await asyncio.sleep(wait)
                        continue
                    print(f"[Infographic] failed: {type(e).__name__}: {e}", flush=True)
                    break

        # Image model unavailable or all attempts failed — return the pre-built SVG
        if svg_result:
            print("[Infographic] returning accurate SVG", flush=True)
            return svg_result

        return {"title": title_text[:80], "subtitle": "", "sections": [], "keyTakeaway": "", "mode": "json"}

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping character-based chunks (legacy fallback)."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def semantic_chunk_text(
        self, text: str, max_words: int = 200, overlap_words: int = 30
    ) -> List[str]:
        """Semantic-aware chunking that respects paragraph and sentence boundaries.

        Strategy:
        1. Split text into paragraphs (double-newline boundaries).
        2. If a paragraph exceeds max_words, further split it by sentences.
        3. Accumulate words into a chunk; when the limit is reached save the chunk
           and carry the last `overlap_words` words into the next chunk for context.
        """
        import re

        # Normalise excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text.strip())
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

        chunks: List[str] = []
        current_words: List[str] = []
        current_count = 0

        def _flush():
            nonlocal current_words, current_count
            if current_words:
                chunks.append(' '.join(current_words))
            overlap = current_words[-overlap_words:] if len(current_words) > overlap_words else current_words[:]
            current_words = overlap[:]
            current_count = len(current_words)

        for para in paragraphs:
            para_words = para.split()

            if len(para_words) > max_words:
                # Long paragraph → split by sentence boundaries first
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    sent_words = sentence.split()
                    if current_count + len(sent_words) > max_words:
                        _flush()
                    current_words.extend(sent_words)
                    current_count += len(sent_words)
            else:
                if current_count + len(para_words) > max_words:
                    _flush()
                current_words.extend(para_words)
                current_count += len(para_words)

        # Flush the final chunk
        if current_words:
            chunks.append(' '.join(current_words))

        return chunks if chunks else [text]

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate a 768-dimensional embedding vector for document storage.

        Uses Google Gemini text-embedding-004 (primary) with OpenAI
        text-embedding-3-small at 768 dims as fallback.
        Returns None if both providers fail.
        """
        # Trim to avoid hitting API token limits
        text = text[:8000].strip()
        if not text:
            return None

        # --- Primary: Gemini text-embedding-004 (768 dims) ---
        try:
            import google.generativeai as genai
            if settings.GOOGLE_GEMINI_API_KEY:
                genai.configure(api_key=settings.GOOGLE_GEMINI_API_KEY)
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                )
                return result["embedding"]
        except Exception:
            pass

        # --- Fallback: OpenAI text-embedding-3-small at 768 dims ---
        try:
            openai_client = self._get_openai()
            if openai_client:
                resp = await openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text,
                    dimensions=768,
                )
                return resp.data[0].embedding
        except Exception:
            pass

        return None

    async def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate a 768-dimensional embedding for a search query.

        Uses the retrieval_query task type so Gemini optimises the vector
        for similarity search against retrieval_document embeddings.
        Falls back to generate_embedding if Gemini is unavailable.
        """
        query = query[:2000].strip()
        if not query:
            return None

        try:
            import google.generativeai as genai
            if settings.GOOGLE_GEMINI_API_KEY:
                genai.configure(api_key=settings.GOOGLE_GEMINI_API_KEY)
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="retrieval_query",
                )
                return result["embedding"]
        except Exception:
            pass

        return await self.generate_embedding(query)

    # ------------------------------------------------------------------
    # OCR prompt (ported from legacy GenVerse API)
    # ------------------------------------------------------------------
    _OCR_PROMPT = (
        "Analyze and extract all text from the uploaded image, "
        "including formulas, mathematical expressions, and physics/chemistry equations, "
        "while preserving their exact format as presented.\n\n"
        "### Key Extraction Guidelines:\n\n"
        "Mathematical & Chemical Accuracy:\n"
        "- Capture all equations, formulas, and notations as they appear.\n"
        "- Accurately preserve subscripts, superscripts, fractions, limits, summations, "
        "and integrals in their correct placement.\n"
        "- Prevent misinterpretation of mathematical numbers, symbols, and notations.\n\n"
        "Line Breaks & Formatting:\n"
        "- Maintain original line breaks and step-by-step formatting for clarity.\n"
        "- Ensure each step in equations or derivations appears on a separate line.\n\n"
        "Strikethrough Content Handling:\n"
        "- Identify and list all strikethrough content separately.\n"
        "- Do not include strikethrough text in the main transcription.\n\n"
        "Diagrams & Non-Text Elements:\n"
        "- If a diagram, graph, or image cannot be converted into text, "
        'include a note: [Refer to source file for diagram].\n\n'
        "Tabular Data Extraction:\n"
        "- Extract tables as structured text while maintaining alignment and readability.\n\n"
        "Language & Text Integrity:\n"
        "- Do not modify or interpret the text.\n"
        "- Do not add extra comments, headers, or additional explanations.\n"
        "- Correct only basic spelling and grammatical errors while preserving the original meaning.\n"
        "- Do not introduce new numbering, bullet points, or formatting unless present in the source.\n"
        "- Maintain the original line breaks, spacing, and order of text exactly as presented.\n\n"
        "### Return Format:\n"
        "1. Use # for Title or Headings if there is any, else don't add anything as heading.\n"
        "2. Use ## for Subsections.\n"
        "3. Use - for bullet points.\n"
        "4. If the text is part of a table, format it using markdown tables.\n"
        "5. If images or diagrams exist, describe them briefly; otherwise don't add remarks.\n"
    )

    _MAX_PDF_PAGES = 50
    _PDF_PARALLEL_WORKERS = 4

    # ------------------------------------------------------------------
    # Image preprocessing (OpenCV) — improves OCR on scanned/handwritten
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_image(image_path: str) -> bytes | None:
        """Resize, grayscale, denoise, threshold → return JPEG bytes."""
        try:
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                return None
            # Resize 2x for better OCR accuracy
            img = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Denoise
            img = cv2.bilateralFilter(img, 9, 75, 75)
            # Binary threshold (OTSU)
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # Convert back to RGB for Gemini
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # Encode to JPEG bytes
            _, buf = cv2.imencode(".jpg", img_rgb)
            return buf.tobytes()
        except Exception as e:
            print(f"[OCR] Preprocessing failed, using raw image: {e}")
            return None

    # ------------------------------------------------------------------
    # Language detection via Gemini vision
    # ------------------------------------------------------------------
    async def _detect_image_language(self, image_data: bytes) -> str:
        """Detect language in an image using Gemini vision. Returns ISO 639-1 code."""
        try:
            import google.generativeai as genai
            api_key = settings.GEMINI_API_KEY or settings.GOOGLE_GEMINI_API_KEY
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(settings.AI_PRIMARY_MODEL)
            prompt = (
                "Extract the text from this image and identify the language. "
                "Return the result in strict JSON format using double quotes only. "
                'Respond like this: {"language": "en"}\n'
                "Use ISO 639-1 codes: en, hi, ta, te, kn, ml, bn, gu, mr, etc."
            )
            resp = await asyncio.to_thread(
                model.generate_content,
                [prompt, {"mime_type": "image/jpeg", "data": image_data}],
            )
            text = (resp.text or "").strip()
            # Clean markdown code fences
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())
            data = json.loads(text)
            lang = data.get("language", "en")
            print(f"[OCR] Detected language: {lang}")
            return lang
        except Exception as e:
            print(f"[OCR] Language detection failed, defaulting to 'en': {e}")
            return "en"

    # ------------------------------------------------------------------
    # Single-page OCR via Gemini vision
    # ------------------------------------------------------------------
    # Ordered list of models to try for vision OCR — first one that succeeds wins.
    # gemini-2.0-flash is the guaranteed multimodal fallback.
    _OCR_MODEL_CANDIDATES = ["gemini-2.0-flash", "gemini-1.5-flash"]

    async def _ocr_single_image(
        self, image_data: bytes, language: str, page_label: str = "",
    ) -> str:
        """Run OCR on a single image using Gemini vision with the comprehensive prompt."""
        import google.generativeai as genai
        api_key = settings.GEMINI_API_KEY or settings.GOOGLE_GEMINI_API_KEY
        genai.configure(api_key=api_key)

        lang_hint = f"The text may be in {language} language. " if language != "en" else ""
        prompt = f"{lang_hint}{self._OCR_PROMPT}"

        # Try the configured primary model first; fall back to known vision-capable models.
        candidates = [settings.AI_PRIMARY_MODEL] + [
            m for m in self._OCR_MODEL_CANDIDATES if m != settings.AI_PRIMARY_MODEL
        ]
        for model_name in candidates:
            try:
                model = genai.GenerativeModel(model_name)
                resp = await asyncio.to_thread(
                    model.generate_content,
                    [prompt, {"mime_type": "image/jpeg", "data": image_data}],
                )
                text = resp.text or ""
                if page_label:
                    print(f"[OCR] {page_label} ({model_name}): extracted {len(text)} chars")
                return text
            except Exception as exc:
                print(f"[OCR] {page_label} model={model_name} failed: {exc}")
                continue

        print(f"[OCR] {page_label}: all vision models failed, returning empty")
        return ""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def extract_text_from_file(self, file_path: str, language: str = "en") -> str:
        """Extract text from a file (PDF, DOCX, image, etc.).

        Implements the legacy GenVerse OCR pipeline:
        - Image preprocessing (resize, grayscale, denoise, threshold)
        - Language auto-detection
        - Vision-based OCR via Gemini (works on scanned/handwritten docs)
        - Parallel page processing for multi-page PDFs
        """
        path = Path(file_path)
        if not path.exists():
            return ""

        ext = path.suffix.lower()
        try:
            # ── PDF: try fast text-layer extraction first, OCR only as fallback ──
            if ext == ".pdf":
                from PyPDF2 import PdfReader

                reader = PdfReader(file_path)
                page_count = len(reader.pages)

                if page_count > self._MAX_PDF_PAGES:
                    print(f"[OCR] PDF has {page_count} pages (limit {self._MAX_PDF_PAGES}), skipping")
                    return ""

                # Fast path: native text-layer extraction (works for digital PDFs)
                native_pages: list[str] = []
                try:
                    for i, page in enumerate(reader.pages):
                        try:
                            t = page.extract_text() or ""
                        except Exception:
                            t = ""
                        native_pages.append(t)
                    total_native_chars = sum(len(t.strip()) for t in native_pages)
                    print(f"[OCR] PDF native extraction: {total_native_chars} chars across {page_count} pages")
                except Exception as e:
                    print(f"[OCR] Native PDF extraction failed: {e}")
                    native_pages = ["" for _ in range(page_count)]

                # Pages with very little native text are likely scanned / image-heavy.
                # Threshold is intentionally high (300 chars ≈ 2-3 real sentences) so
                # that watermark-only text ("Downloaded from studiestoday.com" ≈ 40 chars)
                # never passes as "real content" and always triggers vision OCR.
                _PER_PAGE_TEXT_THRESHOLD = 300  # chars
                pages_needing_ocr = [
                    i for i, t in enumerate(native_pages) if len(t.strip()) < _PER_PAGE_TEXT_THRESHOLD
                ]

                # If every page has plenty of native text, return immediately (fastest path).
                # Also guard against the "watermarks only" case: if the entire document
                # yields fewer than 300 chars per page on average, force OCR regardless.
                avg_chars_per_page = total_native_chars / page_count if page_count else 0
                if not pages_needing_ocr and avg_chars_per_page >= _PER_PAGE_TEXT_THRESHOLD:
                    merged = "\n\n---\n\n".join(
                        f"**Page {i+1}:**\n\n{t}" for i, t in enumerate(native_pages) if t.strip()
                    )
                    print(f"[OCR] PDF fully covered by native text ({len(merged)} chars), skipping OCR")
                    return merged
                elif not pages_needing_ocr:
                    # Every page passed the per-page check individually, but overall
                    # average is suspiciously low — likely all watermarks.  Force all pages.
                    print(f"[OCR] Avg {avg_chars_per_page:.0f} chars/page — likely watermarks only, forcing full OCR")
                    pages_needing_ocr = list(range(page_count))

                print(f"[OCR] {len(pages_needing_ocr)}/{page_count} pages need vision OCR for images/diagrams")

                # Render pages using PyMuPDF (fitz) — no poppler needed, already installed.
                import fitz as _fitz  # PyMuPDF
                import io, tempfile, os
                ocr_pages_text: dict[int, str] = {}
                tmp_dir = tempfile.mkdtemp(prefix="ocr_pages_")
                try:
                    detected_lang = None
                    page_data_list: list[tuple[int, bytes]] = []
                    fitz_doc = _fitz.open(file_path)
                    for page_idx in pages_needing_ocr:
                        try:
                            fitz_page = fitz_doc[page_idx]
                            # 2× zoom → ~144 DPI, good balance of quality vs size
                            mat = _fitz.Matrix(2, 2)
                            pix = fitz_page.get_pixmap(matrix=mat)
                            jpeg_bytes = pix.tobytes("jpeg")
                        except Exception as e:
                            print(f"[OCR] Failed to render page {page_idx + 1} with PyMuPDF: {e}")
                            continue

                        page_path = os.path.join(tmp_dir, f"page_{page_idx}.jpg")
                        with open(page_path, "wb") as f:
                            f.write(jpeg_bytes)

                        # Detect language from first OCR'd page
                        if detected_lang is None:
                            detected_lang = await self._detect_image_language(jpeg_bytes)

                        preprocessed = await asyncio.to_thread(self._preprocess_image, page_path)
                        page_data_list.append((page_idx, preprocessed if preprocessed else jpeg_bytes))
                    fitz_doc.close()
                finally:
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)

                effective_lang = detected_lang if (detected_lang and language == "en") else language

                # Parallel vision OCR (in batches) — instructed to describe diagrams/figures too
                tasks = [
                    self._ocr_single_image(data, effective_lang, f"Page {idx + 1}")
                    for idx, data in page_data_list
                ]
                ocr_results: list[str] = []
                for batch_start in range(0, len(tasks), self._PDF_PARALLEL_WORKERS):
                    batch = tasks[batch_start : batch_start + self._PDF_PARALLEL_WORKERS]
                    batch_results = await asyncio.gather(*batch, return_exceptions=True)
                    for r in batch_results:
                        ocr_results.append(r if isinstance(r, str) else "")

                for (idx, _), text in zip(page_data_list, ocr_results):
                    if text and text.strip():
                        ocr_pages_text[idx] = text

                # Merge: prefer native text per page, fall back to OCR text where available
                merged_pages: list[str] = []
                for i in range(page_count):
                    native = native_pages[i].strip() if i < len(native_pages) else ""
                    ocr_text = ocr_pages_text.get(i, "").strip()
                    if native and ocr_text and i in pages_needing_ocr:
                        # Page had a little native text + images — combine both
                        merged_pages.append(f"**Page {i+1}:**\n\n{native}\n\n{ocr_text}")
                    elif ocr_text:
                        merged_pages.append(f"**Page {i+1}:**\n\n{ocr_text}")
                    elif native:
                        merged_pages.append(f"**Page {i+1}:**\n\n{native}")

                whole_content = "\n\n---\n\n".join(merged_pages)
                print(f"[OCR] PDF extraction complete: {len(whole_content)} chars total ({len(ocr_pages_text)} pages OCR'd)")
                return whole_content

            # ── DOCX: extract paragraphs ──
            elif ext in (".docx",):
                from docx import Document
                doc = Document(file_path)
                return "\n".join(para.text for para in doc.paragraphs)

            # ── Plain text ──
            elif ext in (".txt", ".md"):
                return path.read_text(encoding="utf-8", errors="ignore")

            # ── Images: preprocess → vision OCR ──
            elif ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
                # Preprocess image for better OCR
                preprocessed = await asyncio.to_thread(self._preprocess_image, file_path)
                if preprocessed:
                    image_data = preprocessed
                else:
                    with open(file_path, "rb") as f:
                        image_data = f.read()

                print(f"[OCR] Image ready: {len(image_data)} bytes")

                # Auto-detect language if not specified
                if language == "en":
                    language = await self._detect_image_language(image_data)

                text = await self._ocr_single_image(image_data, language, "Image")
                print(f"[OCR] Extracted {len(text)} chars from image")
                return text

        except Exception as e:
            print(f"[OCR] Error extracting text from {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return ""

    async def generate_class_recommendations(self, class_data: dict) -> List[dict]:
        """
        Generate actionable teaching recommendations for a teacher based on
        class-wide grading data (criterion averages, weak areas, student scores).
        """
        class_context = (
            f"Class: {class_data.get('class_name', 'Unknown')}"
            f" | Subject: {class_data.get('subject', 'N/A')}"
            f" | Grade: {class_data.get('grade', 'N/A')}"
            f" | Board: {class_data.get('board', 'N/A')}"
        )
        total_students = class_data.get("total_students", 0)
        graded_count = class_data.get("submissions_graded", 0)
        class_avg = class_data.get("class_average", 0)
        students_needing_help = class_data.get("students_needing_help", 0)

        criterion_lines = "\n".join(
            f"  - {c['name']}: {c['average']}% average"
            for c in class_data.get("criterion_averages", [])
        ) or "  No rubric criterion data available."

        weak_lines = "\n".join(
            f"  - {w['criterion']}: {w['average']}% (WEAK)"
            for w in class_data.get("weak_outcomes", [])
        ) or "  No weak areas identified."

        prompt = f"""You are an expert educational advisor helping a teacher improve their class performance.

Class context:
{class_context}
Total students: {total_students}
Submissions graded: {graded_count}
Overall class average: {class_avg}%
Students scoring below 60%: {students_needing_help}

Rubric criterion averages (all criteria):
{criterion_lines}

Weakest areas (bottom performers):
{weak_lines}

Based on this real grading data, generate 3 specific, actionable teaching recommendations.
Each recommendation must:
- Be directly tied to the data (reference specific criterion names and percentages)
- Suggest a concrete teaching strategy or activity
- Specify who it targets (whole class, struggling students, advanced students)
- Include a suggested action type

Return ONLY a JSON array of exactly 3 objects:
[
  {{
    "title": "Short recommendation title (max 8 words)",
    "description": "2-3 sentence specific recommendation referencing the actual data",
    "targets": "whole_class | struggling_students | advanced_students",
    "action_type": "remediation | enrichment | assessment | lesson_plan | activity",
    "priority": "high | medium | low"
  }}
]

Return ONLY valid JSON. No markdown fences."""

        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return [
                {
                    "title": "Review Weak Areas",
                    "description": response[:300] if response else "Focus extra time on the lowest-scoring criteria.",
                    "targets": "whole_class",
                    "action_type": "remediation",
                    "priority": "high",
                }
            ]

    async def generate_student_performance_summary(self, student_data: dict) -> dict:
        """Generate AI-powered detailed performance summary for a student across subjects."""
        subject_lines = []
        for subj in student_data.get("subjects", []):
            assignments_info = []
            for a in subj.get("assignments", []):
                atype = "Manual Exam" if a.get("assignment_type") == "manual_exam" else "Assignment"
                topic_str = f" [Topic: {a['topic']}]" if a.get("topic") else ""
                score_str = f"{a['score']}/{a['max_score']} ({a['percentage']}%)" if a.get("score") is not None else a.get("status", "pending")
                assignments_info.append(f"    - [{atype}]{topic_str} {a['title']}: {score_str}")
            subject_lines.append(
                f"  {subj['subject']} (Avg: {subj.get('average_percentage', 'N/A')}%, "
                f"{subj.get('graded', 0)}/{subj.get('total_assignments', 0)} graded):\n" +
                "\n".join(assignments_info)
            )
        subjects_text = "\n".join(subject_lines) or "No subject data available."

        # Topic mastery data
        mastery_lines = []
        for m in student_data.get("topic_mastery", []):
            mastery_lines.append(
                f"  {m['subject']} > {m['topic']}: mastery={m['mastery_level']}%, "
                f"attempts={m['attempts_count']}, correct={m['correct_count']}, trend={m['trend']}"
            )
        mastery_text = "\n".join(mastery_lines) if mastery_lines else "No topic mastery data available."

        prompt = f"""You are an expert educational analyst. Provide a DETAILED and COMPREHENSIVE analysis of this student's academic performance.

Student: {student_data.get('student_name', 'Unknown')}
Grade: {student_data.get('grade', 'N/A')}, Section: {student_data.get('section', 'N/A')}
Overall Percentage: {student_data.get('overall_percentage', 'N/A')}%

Subject-wise Performance:
{subjects_text}

Topic Mastery Data:
{mastery_text}

Return a JSON object with this EXACT structure:
{{
  "summary": "3-4 sentence overall assessment of the student",
  "momentum": "improving | steady | declining",
  "subject_analysis": [
    {{
      "subject": "Mathematics",
      "overall_grade": "A/B/C/D/F based on percentage",
      "percentage": 85,
      "summary": "2-3 sentence analysis of performance in this subject",
      "strong_topics": [
        {{"topic": "Algebra", "detail": "Why this is a strength"}}
      ],
      "weak_topics": [
        {{"topic": "Trigonometry", "detail": "Specific struggles observed", "suggestion": "What to do to improve"}}
      ],
      "assignment_insights": [
        {{"title": "Assessment 1", "observation": "Brief note on performance"}}
      ]
    }}
  ],
  "strengths": ["overall strength 1", "overall strength 2"],
  "areas_to_improve": ["area 1", "area 2"],
  "recommendations": [
    {{"title": "Short title", "detail": "Specific actionable recommendation with reasoning"}},
    {{"title": "Short title", "detail": "Another recommendation"}}
  ],
  "learning_style_note": "Brief observation about the student's learning patterns if detectable from the data"
}}

IMPORTANT RULES:
- For subject_analysis, include EVERY subject the student is enrolled in.
- Infer topics from assignment titles/topics when topic mastery data is unavailable.
- Be specific — reference actual assignment names, scores, and percentages.
- For weak_topics, always include a concrete suggestion.
- Recommendations should be actionable and specific to this student's data, not generic advice.
- If a subject has no graded work, still analyze it noting the lack of engagement.
Return ONLY valid JSON. No markdown fences."""

        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return {
                "summary": response[:500] if response else "Unable to generate summary.",
                "strengths": [],
                "areas_to_improve": [],
                "recommendations": [],
                "momentum": "steady",
                "subject_analysis": [],
            }

    async def generate_audiobook(
        self,
        ebook_json: dict | None,
        language: str,
        voice_profile: str | None,
        narration_style: str = "standard",
    ) -> dict:
        """Generate industry-grade audiobook from ebook content.

        Uses edge-tts (Microsoft Neural TTS) with chapter-aware narration,
        structured segments, silence gaps, and chapter timestamps.
        Falls back to gTTS when edge-tts is unavailable.
        """
        from app.services.audiobook_service import generate_audiobook as _generate

        return await _generate(
            ebook_json=ebook_json,
            language=language,
            voice_profile=voice_profile,
            narration_style=narration_style,
        )

    # ── Playground Gamification Modes ─────────────────────────────────────────

    async def playground_simulate(self, topic: str, variables: dict, controls: list | None = None) -> dict:
        """Sandbox mode: analyse scientific effects of dynamic slider values."""
        if controls:
            vars_str = "\n".join(
                f"  - {ctrl['label']} ({ctrl.get('unit', '')}): {variables.get(ctrl['id'], 50)}"
                for ctrl in controls
            )
        else:
            vars_str = "\n".join(f"  - {k}: {v}" for k, v in variables.items())

        prompt = f"""You are the GenVerse Simulation Engine — a dynamic science visualizer.

Topic: {topic}
Current parameter settings:
{vars_str}

Analyse the scientific effects of these settings on the topic.
Also determine how the visual scene should look right now.

Return ONLY valid JSON — no markdown, no extra text:
{{
  "headline": "A dramatic 1-sentence summary of what is happening right now",
  "analysis": "2-3 sentences of educational science behind these settings",
  "effects": ["Visible effect 1", "Visible effect 2", "Visible effect 3"],
  "verdict": "stable",
  "fun_fact": "A surprising related fact the student might not know",
  "visual_directives": {{
    "atmosphere_color": "#hexcolor",
    "actor_state": "healthy",
    "actor_emoji": "🌱",
    "particle_color": "#hexcolor",
    "particle_count": 20,
    "particle_type": "dot",
    "bg_intensity": 0.5,
    "special_effect": null
  }}
}}

Rules:
- verdict: exactly one of stable | interesting | unstable | dangerous | extreme
- visual_directives.actor_state: one of thriving | healthy | stressed | dying | extreme
- visual_directives.actor_emoji: pick the emoji that best represents the topic subject state
- visual_directives.atmosphere_color: hex color reflecting the overall state
- visual_directives.particle_color: hex matching the atmosphere
- visual_directives.particle_count: 5-45 based on activity level
- visual_directives.particle_type: one of bubble | sparkle | leaf | dot | smoke
- visual_directives.bg_intensity: 0.0-1.0
- visual_directives.special_effect: null OR one of: fire | ice | glow | storm | bloom | void
Return ONLY valid JSON."""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return {
                "headline": "Simulation running…",
                "analysis": response[:300],
                "effects": ["System processing…"],
                "verdict": "stable",
                "fun_fact": "",
                "visual_directives": None,
            }

    async def playground_setup(self, topic: str) -> dict:
        """Topic interpreter: returns dynamic controls + scene setup for the Sandbox mode."""
        prompt = f"""You are the GenVerse Simulation Engine. For the educational topic: '{topic}'

Return ONLY valid JSON with these exact keys:
{{
  "controls": [
    {{
      "id": "unique_snake_case_id",
      "label": "Human Label",
      "unit": "unit_symbol",
      "icon": "LucideIconName",
      "default": 50,
      "description": "Why this parameter matters for this topic"
    }}
  ],
  "scene_type": "scene_name",
  "actor_emoji": "🌱",
  "actor_label": "Short Name",
  "atmosphere": "Brief scene description"
}}

Rules:
- controls: exactly 3-4 TOPIC-SPECIFIC scientifically relevant sliders
- icon must be exactly one of: Sun, Droplets, Wind, Thermometer, Zap, Gauge, Microscope, FlaskConical, Atom, Mountain, Waves, Flame, Leaf, Activity, Cloud
- default: integer 0-100
- scene_type: exactly one of: garden, ocean, space, lab, forest, microscopic, geological, atmosphere, desert, arctic, urban, river
- actor_emoji: the single most representative emoji for this topic's main subject
- Return ONLY valid JSON, no markdown, no explanation."""

        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            return json.loads(cleaned[start:end])
        except Exception:
            return {
                "controls": [
                    {"id": "intensity", "label": "Intensity", "unit": "%", "icon": "Zap", "default": 50, "description": "Overall activity level"},
                    {"id": "temperature", "label": "Temperature", "unit": "%", "icon": "Thermometer", "default": 40, "description": "Thermal energy"},
                    {"id": "complexity", "label": "Complexity", "unit": "%", "icon": "Microscope", "default": 30, "description": "System complexity"},
                ],
                "scene_type": "lab",
                "actor_emoji": "⚗️",
                "actor_label": topic,
                "atmosphere": f"Simulating {topic}",
            }

    async def playground_flashcards(self, topic: str) -> list:
        """Flash-Duel: generate 5 gamified flashcards for a topic."""
        prompt = f"""You are an energetic quiz master generating flashcards for an educational battle game.

Topic: {topic}

Generate EXACTLY 5 flashcards. Each must test a distinct, interesting aspect of the topic.
Make questions progressively harder. Mix: 2 easy (10 pts), 2 medium (20 pts), 1 hard (30 pts).

Return ONLY a valid JSON array — no markdown fences, no extra text:
[
  {{
    "id": 1,
    "question": "An engaging, specific question about the topic",
    "answer": "Clear, concise answer (1-2 sentences)",
    "hint": "A subtle clue that does not give the answer away",
    "difficulty": "easy",
    "points": 10
  }}
]

difficulty must be exactly one of: easy | medium | hard
Return ONLY the JSON array."""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            data = json.loads(cleaned)
            return data if isinstance(data, list) else data.get("cards", [])
        except Exception:
            return [
                {"id": 1, "question": f"What is {topic}?", "answer": response[:200],
                 "hint": "Think about the basics.", "difficulty": "easy", "points": 10}
            ]

    async def playground_quest(self, topic: str, history: list, choice: str | None) -> dict:
        """Quest-Line: advance a branching narrative RPG that teaches the topic."""
        history_str = ""
        if history:
            history_str = "\nStory so far:\n" + "\n".join(
                f"[Scene {i + 1}] {h}" for i, h in enumerate(history)
            )
        chapter = len(history) + 1
        choice_str = f"\nThe player chose: {choice}" if choice else "\nThis is the very first scene — open with a dramatic hook."
        is_final = chapter >= 5
        final_note = (
            "IMPORTANT: This is the FINAL scene. Conclude the story with a satisfying ending "
            "that summarises what was learned. Set choices to an empty array []."
            if is_final
            else "Provide exactly 3 distinct choices that each reflect a different concept or approach."
        )
        prompt = f"""You are a master storyteller running an educational RPG adventure.

Topic: {topic}{history_str}{choice_str}

Write scene {chapter}. Teach real, accurate concepts about {topic} woven into the narrative.
High-energy, vivid, dramatic. Each scene: 3-4 sentences.
{final_note}

Return ONLY valid JSON — no markdown fences:
{{
  "scene": "Vivid 3-4 sentence scene description teaching a concept about {topic}",
  "narrator": "One atmospheric narrator line",
  "choices": ["Choice A: action tied to concept", "Choice B: different approach", "Choice C: creative/risky"],
  "chapter": {chapter},
  "is_final": {"true" if is_final else "false"},
  "xp_gained": {10 + chapter * 5}
}}
Return ONLY valid JSON."""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return {
                "scene": response[:400],
                "narrator": f"Your {topic} adventure continues…",
                "choices": ["Investigate further", "Take a different path", "Seek wisdom"],
                "chapter": chapter,
                "is_final": is_final,
                "xp_gained": 10 + chapter * 5,
            }

    async def playground_mirror_chat(self, topic: str, persona: str, messages: list) -> str:
        """The Mirror: LLM adopts an educational persona and chats in character."""
        system_prompt = (
            f'You are "{persona}", an ancient and wise entity who embodies the very essence of {topic}. '
            f"Speak in first person as this persona. Be educational, mystical, poetic, and high-energy. "
            f"Reveal knowledge about {topic} through your persona\u2019s perspective. "
            f"Keep responses 2-4 sentences. Never break character. "
            f"The student is visiting you to learn about {topic}."
        )
        gemini = self._get_gemini()
        if gemini:
            try:
                history_str = "\n".join(
                    f"{'STUDENT' if m['role'] == 'user' else 'YOU'}: {m['content']}"
                    for m in messages
                )
                full_prompt = system_prompt + "\n\n" + history_str
                resp = gemini.generate_content(full_prompt)
                return resp.text
            except Exception:
                pass
        openai = self._get_openai()
        if openai:
            try:
                oai_messages = [{"role": "system", "content": system_prompt}]
                for m in messages:
                    oai_messages.append({"role": m["role"], "content": m["content"]})
                resp = await openai.chat.completions.create(
                    model=settings.AI_FALLBACK_MODEL,
                    messages=oai_messages,
                )
                return resp.choices[0].message.content
            except Exception:
                pass
        return f"*{persona} gazes into the distance…* Speak, seeker. What do you wish to know about {topic}?"

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _audience_context(
        role: str = "student",
        grade: int | None = None,
        board: str | None = None,
        student_mode: bool = True,
    ) -> str:
        """Build an audience description string for AI prompts.

        When ``student_mode`` is False the prompt stays generic (no grade/board
        constraints) so power-users get unconstrained content. When True, the
        grade band shapes tone/difficulty and an optional ``board`` line anchors
        terminology to the student's curriculum (e.g. CBSE, ICSE, State Board).
        """
        board_line = ""
        if student_mode and board:
            board_line = (
                f" Align examples, terminology, and problem style with the "
                f"{board} curriculum so the content feels familiar to a "
                f"{board} student."
            )

        # Student mode off (or adult role with no grade) → generic adult audience
        if not student_mode or role in ("teacher", "org_admin", "guardian") or (grade is None and role == "normal_user"):
            return (
                "The audience is ADULTS (teachers, professionals, lifelong learners). "
                "Use sophisticated vocabulary, real-world scenarios, deeper analytical questions, "
                "industry jargon where appropriate, and a challenging tone. "
                "Include trick questions and nuanced distinctions. "
                "Roleplay: adopt a witty, competitive challenger persona — "
                "tease the player when they get things wrong, celebrate boldly when they're right."
            )
        if grade and grade <= 5:
            return (
                f"The audience is young children (grade {grade}). "
                "Use simple words, fun comparisons, colourful imagery, and an encouraging tone. "
                "Keep questions easy and concrete. "
                "Roleplay: be a friendly animal buddy (like a wise owl or clever fox) who cheers them on."
                f"{board_line}"
            )
        if grade and grade <= 8:
            return (
                f"The audience is middle-school students (grade {grade}). "
                "Use age-appropriate vocabulary, relatable examples, and moderate challenge. "
                "Roleplay: be an adventurous explorer character who discovers facts alongside the student."
                f"{board_line}"
            )
        return (
            f"The audience is high-school students (grade {grade or 'unknown'}). "
            "Use clear academic language, exam-style rigour, and a mix of easy to hard. "
            "Roleplay: be a cool mentor/coach character who pushes the student to excel."
            f"{board_line}"
        )

    def _parse_json_array(self, text: str) -> list:
        """Robustly extract a JSON array from LLM output."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        return json.loads(cleaned[start:end])

    def _parse_json_object(self, text: str) -> dict:
        """Robustly extract a JSON object from LLM output."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            return {}
        return json.loads(cleaned[start:end])

    async def playground_match_pairs(
        self, topic: str, role: str = "student", grade: int | None = None,
        board: str | None = None, student_mode: bool = True,
    ) -> list:
        """Match Mania / Concept Connect: generate 6 term-definition pairs."""
        audience = self._audience_context(role, grade, board, student_mode)
        prompt = (
            f"{audience}\n\n"
            f"Generate exactly 6 term-definition pairs for the topic: '{topic}'.\n"
            "Each pair must test a distinct, important concept within the topic.\n"
            "Terms should be concise (1-3 words). Definitions should be clear (5-12 words).\n\n"
            "Return ONLY a valid JSON array — no markdown, no explanation:\n"
            '[{"term": "Chloroplast", "definition": "Organelle where photosynthesis occurs in plant cells"}]'
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            data = self._parse_json_array(response)
            return data[:6] if isinstance(data, list) else []
        except Exception:
            return [
                {"term": f"Term {i+1}", "definition": f"Definition for concept {i+1} in {topic}"}
                for i in range(6)
            ]

    async def playground_swipe_facts(
        self, topic: str, role: str = "student", grade: int | None = None,
        board: str | None = None, student_mode: bool = True,
    ) -> list:
        """Swipe & Sort: generate 10 true/false statements."""
        audience = self._audience_context(role, grade, board, student_mode)
        prompt = (
            f"{audience}\n\n"
            f"Generate exactly 10 statements about '{topic}'.\n"
            "Mix TRUE and FALSE (approximately 5-6 true, 4-5 false).\n"
            "False statements must be plausible but scientifically/factually incorrect.\n\n"
            "Return ONLY a valid JSON array — no markdown, no explanation:\n"
            '[{"statement": "Plants absorb CO2 through their roots", "is_true": false, '
            '"explanation": "Plants absorb CO2 through stomata in their leaves"}]\n\n'
            "Every item: statement (string), is_true (boolean), explanation (1 sentence).\n"
            "Return ONLY the JSON array."
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            data = self._parse_json_array(response)
            return data[:10] if isinstance(data, list) else []
        except Exception:
            return []

    async def playground_speed_quiz(
        self, topic: str, role: str = "student", grade: int | None = None,
        board: str | None = None, student_mode: bool = True,
    ) -> dict:
        """Speed Blitz: generate 10 MCQ questions + a challenger persona."""
        audience = self._audience_context(role, grade, board, student_mode)
        is_adult = (
            not student_mode
            or role in ("teacher", "org_admin", "guardian")
            or (grade is None and role == "normal_user")
        )

        prompt = (
            f"{audience}\n\n"
            f"Generate a quiz challenge about '{topic}'.\n"
            "Return a JSON object with two keys:\n\n"
            '1. "challenger": an object describing the AI challenger character:\n'
            '   - "name": a fun character name\n'
            '   - "avatar": an emoji that represents this character\n'
            '   - "taunt": a short competitive opening line (10-15 words)\n'
            '   - "win_line": what they say if the player scores over 70%\n'
            '   - "lose_line": what they say if the player scores under 50%\n\n'
            '2. "questions": an array of exactly 10 MCQ questions. Each question:\n'
            '   - "question": the question text\n'
            '   - "options": array of exactly 4 options\n'
            '   - "correct": 0-based index of the correct answer\n'
            '   - "difficulty": "easy" | "medium" | "hard"\n'
            '   - "points": 10 (easy) | 20 (medium) | 30 (hard)\n'
            '   - "explanation": 1-sentence explanation of the correct answer\n'
            f'   Mix: {"3 easy, 4 medium, 3 hard" if is_adult else "4 easy, 4 medium, 2 hard"}.\n\n'
            "Return ONLY the JSON object — no markdown, no extra text."
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            data = self._parse_json_object(response)
            if "questions" in data:
                data["questions"] = data["questions"][:10]
            return data
        except Exception:
            try:
                questions = self._parse_json_array(response)
                return {
                    "questions": questions[:10] if isinstance(questions, list) else [],
                    "challenger": {
                        "name": "Quiz Bot",
                        "avatar": "\U0001F916",
                        "taunt": "Think you can beat me? Let's find out!",
                        "win_line": "Impressive! You actually beat me!",
                        "lose_line": "Better luck next time, challenger!",
                    },
                }
            except Exception:
                return {"questions": [], "challenger": {}}

    async def playground_roleplay(
        self, topic: str, role: str = "student", grade: int | None = None,
        board: str | None = None, student_mode: bool = True,
    ) -> dict:
        """Generate an immersive roleplay scenario with character, scenes, and choices."""
        audience = self._audience_context(role, grade, board, student_mode)
        is_adult = (
            not student_mode
            or role in ("teacher", "org_admin", "guardian")
            or (grade is None and role == "normal_user")
        )

        role_examples = '"seasoned professor", "industry expert", "detective"' if is_adult else '"friendly guide", "explorer buddy", "wise mentor"'
        prompt = (
            f"{audience}\n\n"
            f"Create an immersive roleplay scenario about '{topic}'.\n"
            "The user will play through the scenario by making choices.\n\n"
            "Return a JSON object with:\n"
            '1. "title": short scenario title (3-6 words)\n'
            '2. "setting": 1-sentence setting description\n'
            '3. "character": the AI character who guides the scenario:\n'
            '   - "name": a memorable character name\n'
            '   - "avatar": a single emoji representing this character\n'
            f'   - "role": their role (e.g. {role_examples})\n'
            '   - "personality": 1-sentence personality description\n\n'
            f'4. "scenes": array of exactly {5 if is_adult else 4} scenes. Each scene:\n'
            '   - "scene_number": 1-based index\n'
            '   - "narration": 1-2 sentences setting the scene\n'
            '   - "character_line": what the character says (15-30 words, in-character)\n'
            '   - "character_mood": one of "happy", "excited", "thinking", "concerned", "proud", "surprised", "encouraging"\n'
            '   - "choices": array of exactly 3 choices, each:\n'
            '     - "text": the action/response option (10-20 words)\n'
            '     - "quality": "best" | "good" | "poor"\n'
            '     - "feedback": 1-2 sentences explaining why this choice was good/bad\n'
            '     - "points": 30 (best) | 15 (good) | 5 (poor)\n\n'
            f'5. "total_scenes": {5 if is_adult else 4}\n\n'
            "IMPORTANT: Exactly ONE choice per scene must be 'best', ONE 'good', ONE 'poor'.\n"
            "Shuffle the order of best/good/poor in each scene.\n"
            "Each scene must teach something important about the topic.\n"
            "Return ONLY the JSON object — no markdown, no extra text."
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            return self._parse_json_object(response)
        except Exception:
            return {
                "title": f"Exploring {topic}",
                "setting": f"A journey through {topic}",
                "character": {"name": "Guide", "avatar": "\U0001F9D9", "role": "mentor", "personality": "Wise and encouraging"},
                "scenes": [],
                "total_scenes": 0,
            }

    async def playground_imagine(
        self, topic: str, role: str = "student", grade: int | None = None,
        board: str | None = None, student_mode: bool = True,
    ) -> dict:
        """Generate a creative 'What if?' scenario with open-ended prompts."""
        audience = self._audience_context(role, grade, board, student_mode)
        is_adult = (
            not student_mode
            or role in ("teacher", "org_admin", "guardian")
            or (grade is None and role == "normal_user")
        )

        prompt = (
            f"{audience}\n\n"
            f"Create a creative 'What if?' scenario about '{topic}' that sparks imagination.\n\n"
            "Return a JSON object with:\n"
            '1. "title": catchy scenario title (4-8 words)\n'
            '2. "hook": an imaginative premise or "What if?" question (2-3 sentences)\n'
            '3. "background_emoji": a single emoji representing the scenario mood\n'
            f'4. "prompts": array of exactly {4 if is_adult else 3} open-ended prompts. Each:\n'
            '   - "prompt_number": 1-based index\n'
            f'   - "question": a thought-provoking open-ended question ({"challenging, analytical, requires expertise" if is_adult else "creative, imaginative, age-appropriate"})\n'
            '   - "hint": a short hint or nudge (1 sentence)\n'
            f'   - "max_points": {25 if is_adult else 20}\n\n'
            "Questions should build on each other — start simpler, get more creative.\n"
            "Make the scenario fascinating and engaging.\n"
            "Return ONLY the JSON object — no markdown, no extra text."
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            return self._parse_json_object(response)
        except Exception:
            return {
                "title": f"Imagining {topic}",
                "hook": f"What if everything about {topic} was different?",
                "background_emoji": "\U0001F4A1",
                "prompts": [],
            }

    async def playground_imagine_evaluate(
        self, topic: str, question: str, answer: str,
        max_points: int = 20, role: str = "student", grade: int | None = None,
        board: str | None = None, student_mode: bool = True,
    ) -> dict:
        """Evaluate a creative open-ended answer."""
        audience = self._audience_context(role, grade, board, student_mode)
        prompt = (
            f"{audience}\n\n"
            f"Topic: {topic}\n"
            f"Question asked: {question}\n"
            f"Student's answer: {answer}\n\n"
            f"Evaluate this creative answer on a scale of 0 to {max_points}.\n"
            "Consider: creativity, depth of understanding, accuracy, and originality.\n"
            "Be encouraging but honest.\n\n"
            "Return a JSON object with:\n"
            f'- "score": integer 0-{max_points}\n'
            f'- "max_score": {max_points}\n'
            '- "feedback": 2-3 sentences of constructive feedback\n'
            '- "highlight": the best part of their answer (quote a phrase or summarize)\n\n'
            "Return ONLY the JSON object — no markdown."
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            return self._parse_json_object(response)
        except Exception:
            return {"score": max_points // 2, "max_score": max_points, "feedback": "Good effort!", "highlight": ""}

    # ── Myth Busters ─────────────────────────────────────────────────────────

    async def playground_mythbusters(
        self, topic: str, role: str = "student", grade: int | None = None,
        board: str | None = None, student_mode: bool = True,
    ) -> dict:
        """Generate a common myth/misconception + 8 evidence cards for sorting."""
        audience = self._audience_context(role, grade, board, student_mode)
        prompt = (
            f"{audience}\n\n"
            f"Pick a common myth or misconception related to '{topic}'.\n"
            "Return a JSON object with:\n"
            '1. "myth": the myth statement (1 sentence, stated as if it were true)\n'
            '2. "is_myth": boolean — true if it really IS a myth (false/incorrect), false if it is actually a fact\n'
            '3. "explanation": 2-3 sentence explanation of the truth\n'
            '4. "evidence": array of exactly 8 evidence cards. Each card:\n'
            '   - "id": integer 1-8\n'
            '   - "text": a piece of evidence or claim (1-2 sentences)\n'
            '   - "category": one of "busts" | "supports" | "red_herring"\n'
            '   - "explanation": why this evidence belongs in that category (1 sentence)\n\n'
            "Distribution: approximately 3 busts, 3 supports, 2 red_herring.\n"
            '"busts" = evidence that disproves the myth.\n'
            '"supports" = evidence that seems to confirm the myth.\n'
            '"red_herring" = sounds relevant but is actually irrelevant or misleading.\n\n'
            "Make evidence cards plausible and thought-provoking.\n"
            "Return ONLY the JSON object — no markdown, no extra text."
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            data = self._parse_json_object(response)
            if "evidence" in data:
                data["evidence"] = data["evidence"][:8]
            return data
        except Exception:
            return {
                "myth": f"Common belief about {topic}",
                "is_myth": True,
                "explanation": "This is a common misconception.",
                "evidence": [],
            }

    # ── Cascade Quiz ─────────────────────────────────────────────────────────

    async def playground_cascade_quiz(
        self, topic: str, role: str = "student", grade: int | None = None,
        board: str | None = None, student_mode: bool = True,
    ) -> dict:
        """Generate a branching quiz tree (depth 3, 7 questions total)."""
        audience = self._audience_context(role, grade, board, student_mode)
        prompt = (
            f"{audience}\n\n"
            f"Create a branching quiz tree about '{topic}'.\n"
            "The tree has 3 levels:\n"
            "- Level 0: 1 root question\n"
            "- Level 1: 2 questions (children of root)\n"
            "- Level 2: 4 questions (2 children per level-1 node)\n"
            "Total: 7 questions forming a binary tree.\n\n"
            "Each question should relate to its parent — when a student answers the parent,\n"
            "the children dive deeper into related sub-topics.\n\n"
            "Return a JSON object with a single key \"tree\" containing the root node.\n"
            "Each node has:\n"
            '- "id": unique string (e.g. "root", "l1a", "l1b", "l2a", "l2b", "l2c", "l2d")\n'
            '- "question": the question text\n'
            '- "options": array of exactly 4 answer options\n'
            '- "correct": 0-based index of the correct option\n'
            '- "explanation": 1-sentence explanation of the correct answer\n'
            '- "children": array of exactly 2 child nodes (omit for leaf nodes at level 2)\n\n'
            "Start broad at the root, get more specific at each level.\n"
            "Return ONLY the JSON object — no markdown, no extra text."
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            data = self._parse_json_object(response)
            return data
        except Exception:
            return {"tree": None}

    # ── Time Warp (Timeline) ─────────────────────────────────────────────────

    async def playground_timeline(
        self, topic: str, role: str = "student", grade: int | None = None,
        board: str | None = None, student_mode: bool = True,
    ) -> dict:
        """Generate a timeline of events + ordering/prediction challenges."""
        audience = self._audience_context(role, grade, board, student_mode)
        prompt = (
            f"{audience}\n\n"
            f"Create an educational timeline about '{topic}'.\n\n"
            "Return a JSON object with:\n"
            '1. "title": timeline title (3-6 words)\n'
            '2. "events": array of 6-8 events in chronological order. Each event:\n'
            '   - "id": integer starting from 1\n'
            '   - "date": the date/period (e.g. "1687", "350 BC", "1900s")\n'
            '   - "title": event title (3-8 words)\n'
            '   - "description": 2-3 sentence description of the event and its significance\n'
            '   - "emoji": a single emoji representing this event\n\n'
            '3. "challenges": array of 2-3 challenges:\n'
            '   For ordering challenges (type "order"):\n'
            '   - "type": "order"\n'
            '   - "scrambled": array of 4 event titles in scrambled order\n'
            '   - "correct_order": the same 4 titles in correct chronological order\n\n'
            '   For prediction challenges (type "predict"):\n'
            '   - "type": "predict"\n'
            '   - "after_event": title of the event this question follows\n'
            '   - "question": "What happened next?" style question\n'
            '   - "options": array of 4 options\n'
            '   - "correct": 0-based index of correct option\n\n'
            "Include at least 1 order challenge and 1 predict challenge.\n"
            "Events MUST be in correct chronological order.\n"
            "Return ONLY the JSON object — no markdown, no extra text."
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            data = self._parse_json_object(response)
            return data
        except Exception:
            return {"title": f"Timeline of {topic}", "events": [], "challenges": []}

    # ── Build-a-Diagram ──────────────────────────────────────────────────────

    async def playground_diagram(
        self, topic: str, role: str = "student", grade: int | None = None,
        board: str | None = None, student_mode: bool = True,
    ) -> dict:
        """Generate a concept diagram with nodes, connections, and zones."""
        audience = self._audience_context(role, grade, board, student_mode)
        prompt = (
            f"{audience}\n\n"
            f"Create a concept diagram about '{topic}' that a student can build.\n\n"
            "Return a JSON object with:\n"
            '1. "title": diagram title (3-6 words)\n'
            '2. "zones": array of 2-4 logical groupings/categories. Each zone:\n'
            '   - "id": short identifier (e.g. "inputs", "process", "outputs")\n'
            '   - "label": display label (e.g. "Inputs", "Process", "Outputs")\n\n'
            '3. "nodes": array of 6-10 concept nodes. Each node:\n'
            '   - "id": unique short identifier (e.g. "n1", "n2")\n'
            '   - "label": the concept name (or "???" for mystery nodes)\n'
            '   - "type": "normal" or "mystery"\n'
            '   - "zone": which zone id this node correctly belongs to\n'
            '   - "hint": (mystery nodes only) a hint about what this node should be labeled\n'
            '   - "answer": (mystery nodes only) the correct label\n\n'
            '4. "connections": array of 4-8 connections between nodes. Each:\n'
            '   - "from": source node id\n'
            '   - "to": target node id\n'
            '   - "label": relationship description (2-4 words)\n\n'
            "Include 1-2 mystery nodes that students must identify.\n"
            "Nodes should form a logical flow or process diagram.\n"
            "Return ONLY the JSON object — no markdown, no extra text."
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            data = self._parse_json_object(response)
            return data
        except Exception:
            return {"title": f"{topic} Diagram", "zones": [], "nodes": [], "connections": []}


# ---------------------------------------------------------------------------
# Module-level singleton — one AIService shared across all requests.
# Reusing the same instance keeps HTTP connections to Gemini/OpenAI/Anthropic
# alive between requests, eliminating the ~7-8s per-request TCP handshake
# overhead that occurs when creating a new client on every call.
# ---------------------------------------------------------------------------
_default_service: "AIService | None" = None


def get_ai_service() -> AIService:
    global _default_service
    if _default_service is None:
        _default_service = AIService()
        # Pre-initialize all provider clients so their HTTP sessions are warm
        # before the first user request arrives.
        _default_service._get_gemini_async()
        _default_service._get_openai()
        _default_service._get_anthropic()
    return _default_service
