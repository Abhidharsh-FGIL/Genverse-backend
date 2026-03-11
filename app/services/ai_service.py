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


class AIService:
    """Unified AI service wrapping Gemini, Anthropic Claude, and OpenAI."""

    def __init__(self):
        self._gemini_client = None
        self._openai_client = None
        self._anthropic_client = None

    def _get_gemini(self):
        if not self._gemini_client and settings.GOOGLE_GEMINI_API_KEY:
            import google.generativeai as genai
            genai.configure(api_key=settings.GOOGLE_GEMINI_API_KEY)
            self._gemini_client = genai.GenerativeModel(settings.AI_PRIMARY_MODEL)
        return self._gemini_client

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
                "TONE: Warm, playful, like their favourite teacher.\n"
                "- Start with a fun hook: a tiny story, 'imagine this…', or a surprising fact.\n"
                "- Use everyday objects, animals, food as analogies.\n"
                "- Use 'First… Then… Finally…' patterns for steps.\n"
                "- Celebrate curiosity: 'Great question!' or 'You're thinking like a little scientist!'\n"
                "- Suggest a fun try-at-home activity when the topic allows."
            ),
            "upper-primary": (
                "TONE: Friendly and encouraging, like a cool older sibling who knows stuff.\n"
                "- Open with 'Did you know?' fun facts or relatable scenarios.\n"
                "- Use daily life, sports, nature examples they'd connect with.\n"
                "- Include 'Think about it' moments and quick challenges.\n"
                "- Break complex topics into numbered steps.\n"
                "- End with 'What would happen if…?' scenarios."
            ),
            "middle-school": (
                "TONE: Clear and structured, balancing depth with accessibility.\n"
                "- Start with thought-provoking real-world connections.\n"
                "- Encourage critical thinking: 'What would happen if we changed X?'\n"
                "- Walk through reasoning step-by-step.\n"
                "- Connect concepts across subjects.\n"
                "- Present common misconceptions and ask them to spot the error."
            ),
            "high-school": (
                "TONE: Precise and academic but still supportive.\n"
                "- Open with a conceptual question or real-world problem.\n"
                "- Use Socratic prompts: 'Think about this before reading the answer…'\n"
                "- Include 'Exam corner' tips and common pitfalls.\n"
                "- Connect to competitive exams (JEE, NEET, CUET) where relevant.\n"
                "- End with a practice problem or discussion question."
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
                "- Use a warm, playful, encouraging tone — like their favourite teacher.\n"
                "- Avoid abstract concepts — make everything concrete and visual.\n"
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
        if language and language.lower() != "english":
            parts.append(f"Respond in {language}. Use {language} script and phrasing naturally.")

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

        # Content length
        length_map = {
            "small": "LENGTH: Be extremely brief — 1-3 sentences maximum. Just the core answer, nothing extra.",
            "brief": "LENGTH: Keep it concise — 1-2 short paragraphs. Get to the point fast.",
            "summary": "LENGTH: Give a focused summary — 2-3 paragraphs covering key points without excessive detail.",
            "medium": "LENGTH: Provide a moderately detailed response covering key points with some examples.",
            "detailed": "LENGTH: Provide a comprehensive, in-depth response with thorough explanations, examples, and structure.",
            "deep-dive": (
                "LENGTH: Provide an exhaustive deep-dive. Cover every important aspect, "
                "edge case, and example. Use headings and sub-sections to organise."
            ),
        }
        content_length = chat_settings.get("content_length", "medium")
        parts.append(length_map.get(content_length, length_map["medium"]))

        if chat_settings.get("explain_3ways"):
            parts.append(
                "IMPORTANT — Explain in 3 Ways: Structure your response with these three clearly labelled sections "
                "directly inside your answer (do NOT add a separate card or section after — include all three here):\n"
                "**Analogy:** A simple, relatable analogy or metaphor that makes the concept easy to grasp.\n"
                "**Technical:** A precise, formal definition or technical explanation.\n"
                "**Real-World Example:** A concrete, real-world application or example of the concept in action."
            )

        if chat_settings.get("examples"):
            parts.append("Always include concrete, real-world examples to illustrate every concept you explain.")

        output_mode = chat_settings.get("output_mode", "text")
        if output_mode == "structured":
            parts.append("Structure your response with clear headings (##) and logical sections.")
        elif output_mode == "bullets":
            parts.append("Present information primarily using bullet points and numbered lists.")

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
            "Never use \\(...\\) or \\[...\\] notation.\n"
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
            "briefly acknowledge it and gently steer back: 'I'm best at helping with learning! What subject are you working on?'"
        )
        if context_str:
            system_prompt += f"\n{context_str}"
        if settings_str:
            system_prompt += f"\n\n{settings_str}"
        if grade_context_instruction:
            system_prompt += f"\n\n{grade_context_instruction}"
        full_prompt = system_prompt + "\n\n" + "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )

        response_text: str | None = None

        if has_files:
            # File-based queries → Anthropic Claude primary, OpenAI fallback, Gemini fallback
            anthropic = self._get_anthropic()
            if anthropic and response_text is None:
                try:
                    claude_messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in messages if m["role"] != "system"
                    ]
                    if not claude_messages or claude_messages[0]["role"] != "user":
                        claude_messages.insert(0, {"role": "user", "content": "Hello"})
                    response = await anthropic.messages.create(
                        model=settings.AI_DOCUMENT_MODEL,
                        max_tokens=4096,
                        system=system_prompt,
                        messages=claude_messages,
                    )
                    response_text = response.content[0].text
                except Exception as e:
                    print(f"[AIService] Anthropic chat failed: {e}", flush=True)
            openai = self._get_openai()
            if openai and response_text is None:
                try:
                    response = await openai.chat.completions.create(
                        model=settings.AI_FALLBACK_MODEL,
                        messages=[{"role": "system", "content": system_prompt}] + messages,
                    )
                    response_text = response.choices[0].message.content
                except Exception:
                    pass
            gemini = self._get_gemini()
            if gemini and response_text is None:
                try:
                    response = gemini.generate_content(full_prompt)
                    response_text = response.text
                except Exception:
                    pass
        else:
            # Direct questions → Gemini primary, OpenAI fallback
            gemini = self._get_gemini()
            if gemini and response_text is None:
                try:
                    response = gemini.generate_content(full_prompt)
                    response_text = response.text
                except Exception:
                    pass
            openai = self._get_openai()
            if openai and response_text is None:
                try:
                    response = await openai.chat.completions.create(
                        model=settings.AI_FALLBACK_MODEL,
                        messages=[{"role": "system", "content": system_prompt}] + messages,
                    )
                    response_text = response.choices[0].message.content
                except Exception:
                    pass

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
        """Stream from Gemini without blocking the event loop.

        Gemini's SDK is synchronous, so we call next() on the iterator
        inside asyncio.to_thread to keep the event loop free. This lets
        FastAPI flush each SSE chunk to the client immediately.
        """
        gemini = self._get_gemini()
        if not gemini:
            return

        # Start the streaming call in a thread (the initial request blocks)
        def _start_stream():
            return gemini.generate_content(full_prompt, stream=True)

        response = await asyncio.to_thread(_start_stream)
        it = iter(response)

        while True:
            # Get next chunk without blocking the event loop
            chunk = await asyncio.to_thread(next, it, _SENTINEL)
            if chunk is _SENTINEL:
                break
            try:
                text = chunk.text
                if text:
                    yield text
            except (ValueError, AttributeError):
                # Thinking/reasoning chunks may not have .text — skip
                continue

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

    async def stream_chat(
        self, messages: List[dict], context: dict | None = None,
        chat_settings: dict | None = None, has_files: bool = False,
    ) -> AsyncIterator[str]:
        """SSE streaming chat with AI.

        Routing:
          - has_files=True  → OpenAI (better at document Q&A), Gemini fallback
          - has_files=False → Gemini Pro (direct questions), OpenAI fallback
        """
        # --- Content Guard: pre-generation check ---
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

        grade_context_instruction = None
        if user_query:
            guard_result = await guard.run_input_pipeline(user_query)
            guard.log_guard_event(user_query, guard_result)
            if guard_result.action != GuardAction.ALLOW:
                yield guard_result.message
                return
            # Carry grade-relevance context for the system prompt
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
            "Never use \\(...\\) or \\[...\\] notation.\n"
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
            "briefly acknowledge it and gently steer back: 'I'm best at helping with learning! What subject are you working on?'"
        )
        if context_str:
            system_prompt += f"\n{context_str}"
        if settings_str:
            system_prompt += f"\n\n{settings_str}"
        if grade_context_instruction:
            system_prompt += f"\n\n{grade_context_instruction}"
        full_prompt = system_prompt + "\n\n" + "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )

        # --- Helper: stream from providers with output safety check ---
        accumulated = ""
        next_check_at = 500  # check output every ~500 chars

        async def _provider_stream():
            """Yield chunks from the appropriate provider chain."""
            if has_files:
                if self._get_anthropic():
                    try:
                        async for chunk in self._stream_anthropic(system_prompt, messages):
                            yield chunk
                        return
                    except Exception as e:
                        print(f"[AIService] Anthropic stream failed: {e}", flush=True)
                if self._get_openai():
                    try:
                        async for chunk in self._stream_openai(system_prompt, messages):
                            yield chunk
                        return
                    except Exception as e:
                        print(f"[AIService] OpenAI stream fallback failed: {e}", flush=True)
                if self._get_gemini():
                    try:
                        async for chunk in self._stream_gemini(full_prompt):
                            yield chunk
                        return
                    except Exception as e:
                        print(f"[AIService] Gemini stream fallback failed: {e}", flush=True)
            else:
                if self._get_gemini():
                    try:
                        async for chunk in self._stream_gemini(full_prompt):
                            yield chunk
                        return
                    except Exception as e:
                        print(f"[AIService] Gemini stream failed: {e}", flush=True)
                if self._get_openai():
                    try:
                        async for chunk in self._stream_openai(system_prompt, messages):
                            yield chunk
                        return
                    except Exception as e:
                        print(f"[AIService] OpenAI stream fallback failed: {e}", flush=True)
            yield "AI service not configured."

        async for chunk in _provider_stream():
            accumulated += chunk
            yield chunk
            # Periodic output safety check
            if len(accumulated) >= next_check_at:
                next_check_at += 500
                output_check = guard.check_output(accumulated)
                if output_check.action != GuardAction.ALLOW:
                    guard.log_guard_event(user_query, output_check)
                    yield "\n\n" + output_check.message
                    return

        # Final output check on complete text
        output_check = guard.check_output(accumulated)
        if output_check.action != GuardAction.ALLOW:
            guard.log_guard_event(user_query, output_check)
            yield "\n\n" + output_check.message

    async def ask_document(self, query: str, context: str, ai_context: dict | None = None) -> str:
        """RAG query against extracted document text."""
        prompt = f"""You are a document assistant. Answer based ONLY on the provided document context.
If the answer is not found in the context, say so.

Document context:
{context[:8000]}

Question: {query}

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
    ) -> List[dict]:
        """Generate practice assessment questions as JSON — respects all config options."""
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
        if mcq_count > 0 and len(subtypes) > 1:
            mcq_subtype_counts = self._distribute_questions(
                mcq_count, {s: 1 for s in subtypes}
            )
        elif mcq_count > 0:
            mcq_subtype_counts = {subtypes[0]: mcq_count}

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
                    dist_lines.append(f"      • {subtype_labels.get(s, s)}: {sc}")

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
        if source_text and source_text.strip():
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Assessment prompt: using source_text (%d chars, truncating to 12000)",
                len(source_text),
            )
            source_section = (
                "SOURCE TEXT (generate questions ONLY from this content, do not use outside knowledge):\n"
                f"---\n{source_text[:12000]}\n---"
            )
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

        prompt = f"""You are an expert question paper setter. Generate exactly {question_count} questions for a {mode} assessment.

SUBJECT: {subject or topics_str}
TOPICS: {topics_str}
GRADE: {f'Grade {grade}' if grade else 'General'}{f' ({board})' if board else ''}
DIFFICULTY: {difficulty}
MODE: {mode}
{blooms_section}
{neg_section}

ALLOWED QUESTION TYPES — STRICTLY: {allowed_types_str}
You MUST NOT generate any question with a "type" outside this list. Every single question must use only these types.

EXACT QUESTION DISTRIBUTION (generate exactly this many of each type — no more, no less, no substitutions):
{chr(10).join(dist_lines)}
{topic_section}

{source_section}

QUESTION FORMAT RULES — follow exactly:
1. MCQ (standard): 4 distinct options as a list. Exactly one correct.
   "options": ["option1", "option2", "option3", "option4"]
   "correct_answer": the exact correct option string.

2. MCQ (case): Include a brief scenario/passage (2-4 sentences) in "text" above the question.
   Then ask a question about it. 4 options as above.

3. MCQ (assertion_reason): Two statements.
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
- "subtype": for MCQ — one of "standard" | "case" | "assertion_reason" | "higher_order"; for all others — null
- "text": the full question text (string)
- "options": array of strings for mcq/true_false/match; null for fill/short/long
- "pairs": array of {{"left":..., "right":...}} objects for match; null for all others
- "correct_answer": string (required for all types)
- "explanation": 1-2 sentence explanation of why the answer is correct
- "marks": 1 for mcq/fill/true_false; 2 for short/match; 4 for long
- "blooms_level": one of "remember" | "understand" | "apply" | "analyze" | "evaluate" | "create"

⚠️ FINAL CHECK BEFORE OUTPUT: Verify that every object's "type" field is one of {allowed_types_str}. If any object has a different type, correct it before returning.

Return ONLY the raw JSON array. No markdown fences, no explanation text outside the array."""

        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
        except Exception:
            pass
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
    ) -> List[dict]:
        """Generate chapter titles and descriptions only — no full content."""
        min_ch, max_ch = chapter_range

        tone_context = {
            "academic": "formal and scholarly",
            "simple": "beginner-friendly and easy to follow",
            "story_based": "narrative-driven with engaging storytelling",
            "exam_oriented": "focused on exam-relevant topics and key facts",
        }.get(tone, "educational")

        prompt = f"""You are an expert educational author. Create a chapter outline for an eBook.

Title: {title}
Topic: {topic}
Subject: {subject or "General"}
Language: {language}
Number of chapters: between {min_ch} and {max_ch}
Writing style: {tone_context}

Generate a logical, well-structured chapter outline where:
- Chapter titles are concise and clear (4-8 words)
- Descriptions are 1-2 sentences explaining what the chapter covers
- Chapters flow naturally from foundational concepts to advanced ones
- The tone/style "{tone}" is reflected in how chapters are framed

Return ONLY valid JSON in this exact structure:
{{
  "chapters": [
    {{
      "title": "Chapter title here",
      "description": "Brief description of what this chapter covers."
    }}
  ]
}}"""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            data = json.loads(cleaned)
            return data.get("chapters", [])
        except Exception:
            return []

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
                    model="gemini-3-pro-image-preview",
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

        loop = asyncio.get_event_loop()

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
    ) -> dict:
        """Generate structured eBook content as JSON, then generate images."""
        if chapters:
            outline_str = "\n".join(
                f"- {ch.get('title', '')}" + (f": {ch.get('description', '')}" if ch.get('description') else "")
                for ch in chapters
            )
        else:
            outline_str = "\n".join(f"- {item}" for item in (outline or []))
        min_ch, max_ch = chapter_range

        tone_instructions = {
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

        size_content_guides = {
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

        tone_guide = tone_instructions.get(tone, tone_instructions["academic"])
        size_guide = size_content_guides.get(book_size, size_content_guides["short"])

        language_names = {
            "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
            "fr": "French", "de": "German", "es": "Spanish", "zh": "Chinese",
            "ar": "Arabic", "pt": "Portuguese",
        }
        language_name = language_names.get(language, language.upper())

        chapters_provided = bool(chapters and len(chapters) > 0)
        chapter_count_instruction = (
            f"Use EXACTLY the {len(chapters)} chapters listed in the outline below — do not add, remove, or reorder them."
            if chapters_provided
            else f"Generate between {min_ch} and {max_ch} chapters — choose the exact count that best covers the topic."
        )

        assessment_section = ""
        final_assessment_json = '"final_assessment": null'
        assessment_enabled = bool(assessment_config and assessment_config.get("enabled"))
        if assessment_enabled:
            difficulty = assessment_config.get("difficulty", "medium")
            q_types = assessment_config.get("questionTypes", ["MCQ"])
            blooms = assessment_config.get("bloomsLevel", "understand")
            q_types_str = ", ".join(q_types)

            type_instructions = []
            json_fields = []
            if "MCQ" in q_types:
                type_instructions.append('- For MCQ: include in "mcq_questions" with "chapter_number", "question", "options" (4 choices), "answer" (correct option text).')
                json_fields.append('    "mcq_questions": [\n      { "chapter_number": 1, "question": "...", "options": ["...", "...", "...", "..."], "answer": "..." }\n    ]')
            if "Fill in Blank" in q_types:
                type_instructions.append('- For Fill in Blank: include in "fill_in_blank_questions" with "chapter_number", "question" (sentence with ___ for the blank), "answer" (word/phrase that fills the blank).')
                json_fields.append('    "fill_in_blank_questions": [\n      { "chapter_number": 1, "question": "The ___ process converts sunlight into energy.", "answer": "photosynthesis" }\n    ]')
            if "Short Answer" in q_types:
                type_instructions.append('- For Short Answer: include in "short_answer_questions" with "chapter_number", "question", "answer" (2-3 sentence model answer).')
                json_fields.append('    "short_answer_questions": [\n      { "chapter_number": 1, "question": "...", "answer": "..." }\n    ]')
            if "Long Answer" in q_types:
                type_instructions.append('- For Long Answer: include in "long_answer_questions" with "chapter_number", "question", "answer" (detailed model answer).')
                json_fields.append('    "long_answer_questions": [\n      { "chapter_number": 1, "question": "...", "answer": "..." }\n    ]')

            instructions_str = "\n".join(type_instructions)
            assessment_section = f"""
ASSESSMENT REQUIREMENTS:
- Place ALL assessment questions in the root-level "final_assessment" section — NOT inside individual chapters.
- Generate 3-5 questions per chapter, distributed across all chapters of the book.
- Question types to include: {q_types_str}
- Difficulty: {difficulty}
- Bloom's Taxonomy level: {blooms}
{instructions_str}
- Only include JSON keys for the selected question types above — omit others entirely.
- Group questions by type in order — this is the final section of the book.
"""
            fields_str = ",\n".join(json_fields)
            final_assessment_json = f'"final_assessment": {{\n{fields_str}\n  }}'

        assessment_layout_line = (
            '  • End Pages — Assessment Section: all MCQs grouped together, then Short Answers, then Long Answers'
            if assessment_enabled else ""
        )
        no_assessment_note = (
            "" if assessment_enabled
            else '- Do NOT include any assessment questions. Set "final_assessment" to null.'
        )

        prompt = f"""Create a complete structured educational eBook with the following specifications.

LANGUAGE REQUIREMENT: Write ALL content — titles, descriptions, chapter bodies, key points, summaries, questions — in {language_name}. Do NOT use any other language.

Title: {title}
Author: {author or "Anonymous"}
Subject: {subject or "General"}
Grade: {grade or "General"}
Book Size: {book_size.capitalize()} — TARGET: {size_guide["total_pages"]} pages total
Writing Tone: {tone.replace("_", " ").title()}

BOOK PAGE LAYOUT (strictly follow this structure):
  • Page 1   — Cover Page: full-page book cover (image generated separately)
  • Page 2   — Title Page: book title centered large, "by {{author}}" centered below it
  • Page 3   — Book Summary: 4-10 sentences giving a comprehensive overview of the entire book
  • Page 4   — Table of Contents: numbered chapter list
  • {size_guide["content_pages"]} — Chapters numbered "1. Title", "2. Title", etc. (one chapter per page range)
{assessment_layout_line}
  • Final Page — Thank you / hope message for the reader

TONE INSTRUCTIONS (apply to every chapter):
{tone_guide}

CONTENT DEPTH PER CHAPTER (calibrated to fill {size_guide["total_pages"]} pages total):
- Length: {size_guide["paragraphs"]} — {size_guide["words_hint"]}
- Structure: {size_guide["depth"]}
- Key points: {size_guide["key_points"]}
{assessment_section}
{f'Chapter Outline:{chr(10)}{outline_str}' if outline_str else ''}

REQUIREMENTS:
1. {chapter_count_instruction}
2. Every chapter MUST meet the word count target ({size_guide["words_hint"]}). Short chapters that do not fill their page budget are NOT acceptable.
3. Apply both the tone style and depth level consistently across ALL chapters.
4. The "content" field must be the FULL chapter body — not a placeholder, stub, or summary.
5. The "key_points" array must list the most important facts/concepts from the chapter.
6. The "summary" must be 1-2 sentences recapping the chapter.
7. Do NOT reuse identical phrasing across chapters — each chapter must feel distinct.
8. If a chapter description is given in the outline, use it to guide the content scope.
9. The "title_page.description" must be a compelling 2-3 sentence overview of the entire book.
10. All text must be written in {language_name}.
11. The "book_summary" field must be 4-10 sentences giving a comprehensive overview of the ENTIRE book — its scope, key themes, and what the reader will learn.
12. The "thank_you_message" must be 2-3 warm, encouraging sentences wishing the reader well after completing the book.
13. FORMATTING: Do NOT use markdown headings (# ## ###) inside the "content" field. Use plain text paragraphs only. If you need sub-sections, use **bold text** for sub-headings on their own line — never use # or ## or ### symbols.
{no_assessment_note}

Return ONLY valid JSON in this exact structure (no markdown fences, no extra keys):
{{
  "title": "{title}",
  "author": "{author or 'Anonymous'}",
  "language": "{language}",
  "book_size": "{book_size}",
  "tone": "{tone}",
  "title_page": {{
    "title": "...",
    "author": "...",
    "subtitle": "...",
    "description": "..."
  }},
  "book_summary": "...",
  "table_of_contents": [
    {{ "chapter_number": 1, "title": "..." }}
  ],
  "chapters": [
    {{
      "chapter_number": 1,
      "title": "...",
      "content": "...",
      "key_points": ["...", "..."],
      "summary": "..."
    }}
  ],
  {final_assessment_json},
  "thank_you_message": "..."
}}"""
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            ebook_data = json.loads(cleaned)
        except Exception:
            ebook_data = {
                "title": title,
                "author": author or "Anonymous",
                "language": language,
                "book_size": book_size,
                "tone": tone,
                "title_page": {"title": title, "author": author or "Anonymous", "subtitle": "", "description": ""},
                "book_summary": "",
                "table_of_contents": [{"chapter_number": 1, "title": "Chapter 1"}],
                "chapters": [{"chapter_number": 1, "title": "Chapter 1", "content": response, "key_points": [], "summary": ""}],
                "final_assessment": None,
                "thank_you_message": f"Thank you for reading {title}. We hope this book has been a valuable and enriching experience for you.",
            }

        # Generate infographic images using Gemini
        if image_density != "minimal":
            try:
                generated_chapters = ebook_data.get("chapters", [])
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
                pass  # Image generation is non-blocking — proceed without images

        return ebook_data

    async def generate_mindmap(
        self,
        topic: str,
        subject: str | None,
        grade: int | None,
        board: str | None,
        depth: int,
    ) -> dict:
        """Generate a mind map structure as JSON."""
        prompt = f"""Create a mind map for:
Topic: {topic}
Subject: {subject or "General"}
Grade: {grade or "General"}
Board: {board or "General"}
Depth: {depth} levels

ACCURACY & SPELLING (CRITICAL):
- Every word in every node label MUST be spelled correctly.
- Scientific terms, proper nouns, formulas, and dates must be 100% accurate.
- Double-check all technical vocabulary and terminology before including it.
- Use proper capitalisation for proper nouns and sentence-case for other labels.
- Do NOT abbreviate in a way that changes meaning or creates ambiguity.

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

        prompt = f"""You are an expert teacher creating a lesson plan. Use every detail below to calibrate the plan.

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
        response = await self.chat([{"role": "user", "content": prompt}])
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
            return {"title": f"Lesson Plan: {topic}", "objectives": [topic], "timeEstimate": 45, "steps": []}

    async def generate_rubric(
        self, board: str, grade: int, subject: str, topic: str, criteria_count: int,
        difficulty_level: str = 'medium'
    ) -> List[dict]:
        """Generate grading rubric criteria."""
        import uuid as _uuid
        difficulty_guidance = {
            'simple': 'Use straightforward, basic descriptors suitable for foundational understanding.',
            'medium': 'Use moderately detailed descriptors that require applied understanding.',
            'complex': 'Use rigorous, nuanced descriptors requiring higher-order thinking and mastery.',
        }.get(difficulty_level, 'Use moderately detailed descriptors.')
        prompt = f"""Create a detailed grading rubric for:
Subject: {subject}
Board: {board}
Grade: {grade}
Topic: {topic}
Difficulty Level: {difficulty_level} — {difficulty_guidance}
Number of criteria: {criteria_count}

Return JSON array of criteria:
[
  {{
    "id": "criterion_1",
    "title": "...",
    "weight": 25,
    "linkedOutcome": "...",
    "levels": [
      {{"level": "Excellent", "score": 4, "description": "..."}},
      {{"level": "Good", "score": 3, "description": "..."}},
      {{"level": "Satisfactory", "score": 2, "description": "..."}},
      {{"level": "Needs Improvement", "score": 1, "description": "..."}}
    ]
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
            prompt = f"""You are an expert teacher. Grade this student submission using the rubric criteria.

Student: {student_label}
{"Questions & Answers:" + chr(10) + qa_context if qa_context else ""}
{"Submission text:" + chr(10) + submission_text if submission_text else ""}

Rubric criteria:
{rubric_context}

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

Important: criterionTitle in criterionScores must exactly match the title field of each rubric criterion. Points must be a valid integer within that criterion's level range."""

        try:
            response = await self.chat([{"role": "user", "content": prompt}])
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned)
        except Exception:
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
        true_false_count: int = 0,
        match_count: int = 0,
        difficulty: str = "medium",
        lesson_plan_context: dict | None = None,
        rubric_criteria: list | None = None,
        source_text: str | None = None,
    ) -> List[dict]:
        """Generate structured assignment questions for an AssignmentEditor.

        When a lesson plan is provided, questions are derived from its objectives and steps.
        When rubric criteria are provided, questions are aligned to each criterion so the
        assessment can be evaluated against the rubric.
        """
        parts = []
        if mcq_count:
            parts.append(f"{mcq_count} MCQ (multiple choice)")
        if fib_count:
            parts.append(f"{fib_count} Fill-in-the-blank")
        if short_answer_count:
            parts.append(f"{short_answer_count} Short answer")
        if true_false_count:
            parts.append(f"{true_false_count} True/False")
        if match_count:
            parts.append(f"{match_count} Match-the-following")
        types_str = ", ".join(parts) or "5 Short answer"

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

        prompt = f"""Generate assignment questions for a Grade {grade} {subject} class.
Topic: {topic}
Difficulty: {difficulty}
Question breakdown: {types_str}
{source_section}{lesson_plan_section}{rubric_section}
Return a JSON object with a "questions" array. Each question must follow this schema exactly:
- type: one of "mcq", "fill-blank", "short-answer", "true-false", "match"
- text: the question text
- points: integer (2 for fill-blank/true-false, 5 for mcq/short-answer, 10 for match)
- For MCQ: include "options" (array of 4 strings), "correctAnswer" (index 0-3 as number), and "explanation" (1-2 sentences explaining why the correct answer is right)
- For fill-blank: include "correctAnswer" as a string and "explanation" (1-2 sentences explaining the answer)
- For true-false: include "correctAnswer" as "true" or "false"
- For match: include "matchPairs" as array of {{"left": "...", "right": "..."}} (4-5 pairs)
- For short-answer: no extra fields needed

The "explanation" field is REQUIRED for MCQ and fill-blank questions. It helps students understand why the answer is correct.

Return ONLY valid JSON, no markdown.
Example: {{"questions": [{{"type": "mcq", "text": "...", "options": ["A","B","C","D"], "correctAnswer": 0, "points": 5, "explanation": "Tuple is immutable because..."}}]}}
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
        if grade:
            system += f"\nAdapt for Grade {grade} students."

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

    async def generate_career_profile(self, user_id: str, db) -> dict:
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

        # ── Build prompt context ──────────────────────────────────────────────
        has_data = bool(attempt_rows or mastery_data or user_messages)

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

SUBJECT PERFORMANCE (from assessments):
{subject_text}

TOPIC MASTERY:
{mastery_text}

RECENT AI CHAT TOPICS (user's questions — use to infer interests):
{chat_context}

PAST CAREER GUIDANCE SESSIONS:
{past_sessions_text}

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

    async def generate_assessment_recommendations(self, user_id: str, db) -> List[dict]:
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

RECENT ASSESSMENT ATTEMPTS (most recent first):
{attempts_summary}

TOPIC MASTERY (weakest first):
{mastery_summary}

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

    async def generate_assessment_summary(self, user_id: str, db) -> dict:
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

        total_attempts = len(rows)
        overall_avg = round(sum(r[0].percentage or 0 for r in rows) / total_attempts, 1) if rows else 0
        best_overall = round(max((r[0].percentage or 0 for r in rows), default=0), 1)

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

        prompt = f"""You are an expert AI learning coach. Analyse this student's complete Assessment Hub usage data and produce a personalised coaching summary.

TOTAL ASSESSMENTS TAKEN: {total_attempts}
OVERALL AVERAGE SCORE: {overall_avg}%
PERSONAL BEST SCORE: {best_overall}%

PER-SUBJECT STATS:
{subject_text}

RECENT ATTEMPTS (most recent first):
{attempts_text}

TOPIC MASTERY (strongest first):
{mastery_text}

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

            section = f"\n### {subj_name} (Chapters: {chapters_str})"
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

        prompt = f"""You are an expert question paper setter for school students. Generate exactly {question_count} questions for an institutional evaluation paper.

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
   "text": "Assertion (A): [statement]\\nReason (R): [statement]\\nChoose the correct option:"
   "options": ["Both A and R are true, and R is the correct explanation of A", "Both A and R are true, but R is not the correct explanation of A", "A is true but R is false", "A is false but R is true"]

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
- "subject": the subject name
- "chapter": the chapter name
- "blooms_level": one of "remember"|"understand"|"apply"|"analyze"|"evaluate"|"create"

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

            # Fix invalid JSON backslash escapes from AI-generated LaTeX.
            # AI puts LaTeX like \text, \frac, \times, \alpha inside JSON strings.
            # JSON parser interprets \t as tab, \f as form feed, \n as newline, etc.
            # Fix: any \ followed by 2+ letters is a LaTeX command, not a JSON escape.
            # Valid JSON escapes (\n, \t, \r, \b, \f) are always a single char after \.
            def _fix_latex_escapes(s: str) -> str:
                return re.sub(r'\\(?=[a-zA-Z]{2})', r'\\\\', s)

            try:
                questions = json.loads(cleaned)
            except json.JSONDecodeError:
                cleaned = _fix_latex_escapes(cleaned)
                questions = json.loads(cleaned)
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

        prompt = (
            "Extract a concise, specific YouTube search query (5-10 words max) that would find "
            "the most relevant educational video for this topic.\n"
            f"{grade_instruction}\n"
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
        prompt = f"""You are a mind-map extractor. Your ONLY job is to read the AI explanation below and
reorganise the SAME information into a hierarchical mind map. Do NOT add external knowledge.
Every single node label MUST come directly from a heading, key phrase, term, fact, or example
that appears in the explanation text.

Topic: {user_message[:500]}
Grade Level: {grade_str}
Language: {language}

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

    async def generate_chat_infographic(
        self,
        user_message: str,
        ai_response: str,
        grade: int | None = None,
        language: str = "English",
    ) -> dict:
        """Generate an infographic using Gemini image generation with JSON fallback."""
        import base64
        import asyncio

        grade_str = f"grade {grade}" if grade else "general"
        topic = user_message[:300]
        explanation = ai_response[:3000]

        prompt = (
            f"Create a beautiful, colorful educational infographic poster about: {topic}\n\n"
            f"Grade Level: {grade_str}\n"
            f"Language: {language}\n\n"
            f"Key information to include in the infographic:\n{explanation}\n\n"
            "Design requirements:\n"
            "- Bold title at the top\n"
            "- Use vibrant colors, icons, and visual hierarchy\n"
            "- Organize information into clear visual sections with headings\n"
            "- Include key facts, statistics, and dates as visual callouts\n"
            "- Use arrows, connectors, or flow lines between related concepts\n"
            "- Make text readable and well-spaced\n"
            "- Professional infographic style suitable for students\n"
            "- Dark or colored background with contrasting text\n"
            "- Do NOT leave large empty spaces — fill with relevant visual elements\n\n"
            "ACCURACY & SPELLING (CRITICAL):\n"
            "- Every word, name, term, date, and formula MUST be spelled correctly.\n"
            "- Copy scientific terms, proper nouns, and formulas EXACTLY from the key information above.\n"
            "- Double-check ALL text before rendering — any spelling mistake will make the infographic unusable.\n"
            "- Use proper capitalisation for headings and proper nouns.\n"
            "- If including numbers or statistics, verify they match the source information exactly."
        )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=settings.GEMINI_API_KEY)

            def _generate():
                return client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                    ),
                )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _generate)

            for part in response.parts:
                if part.inline_data is not None:
                    img_bytes = part.inline_data.data
                    mime = part.inline_data.mime_type or "image/png"
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    data_uri = f"data:{mime};base64,{b64}"
                    print(f"[Infographic] Generated image ({len(img_bytes)} bytes)")
                    return {
                        "image_base64": data_uri,
                        "title": topic,
                        "mode": "image",
                    }
            print("[Infographic] No image part in response, falling back to JSON mode")
        except Exception as e:
            print(f"[Infographic] Image generation failed: {e}, falling back to JSON mode")

        # Fallback: generate JSON-based infographic using primary text model
        fallback_prompt = f"""You are an infographic designer. Convert this into a structured infographic layout.

Topic: {topic}
Grade Level: {grade_str}
Language: {language}

── AI EXPLANATION ──
{explanation}
── END ──

ACCURACY & SPELLING (CRITICAL):
- Every word, name, term, date, and formula in the output MUST be spelled correctly.
- Copy scientific terms, proper nouns, and formulas EXACTLY from the explanation above.
- Double-check ALL text strings before including them — spelling mistakes make the infographic unusable.
- Use proper capitalisation for headings and proper nouns.

Return ONLY raw JSON with: title, subtitle, sections (array of heading/icon/color/facts/highlight), keyTakeaway.
Icons: BookOpen, Globe, Clock, Lightbulb, Users, Star, Target, Zap, Award, Shield, Brain, Heart, TrendingUp, BarChart, Layers"""

        response_text = await self.chat([{"role": "user", "content": fallback_prompt}])
        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(cleaned[start:end])
                result["mode"] = "json"
                return result
        except Exception:
            pass
        return {"title": user_message[:50], "subtitle": "", "sections": [], "keyTakeaway": "", "mode": "json"}

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

    async def extract_text_from_file(self, file_path: str, language: str = "en") -> str:
        """Extract text from a file (PDF, DOCX, image, etc.)."""
        path = Path(file_path)
        if not path.exists():
            return ""

        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                # Try PyMuPDF first (handles complex/large PDFs better)
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    text = "\n".join(page.get_text() or "" for page in doc)
                    doc.close()
                    if text.strip():
                        return text
                except Exception as e:
                    print(f"[OCR] PyMuPDF failed, falling back to PyPDF2: {e}")
                # Fallback to PyPDF2
                import PyPDF2
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                return text

            elif ext in (".docx",):
                from docx import Document
                doc = Document(file_path)
                return "\n".join(para.text for para in doc.paragraphs)

            elif ext in (".txt", ".md"):
                return path.read_text(encoding="utf-8", errors="ignore")

            elif ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
                # Map extension to correct MIME type for Gemini vision
                mime_map = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".webp": "image/webp",
                    ".gif": "image/gif",
                }
                mime_type = mime_map.get(ext, "image/jpeg")

                with open(file_path, "rb") as f:
                    image_data = f.read()

                print(f"[OCR] Image read: {len(image_data)} bytes, mime={mime_type}")

                # Use the newer google-genai SDK for reliable multimodal support
                try:
                    from google import genai
                    from google.genai import types

                    client = genai.Client(api_key=settings.GEMINI_API_KEY)
                    response = client.models.generate_content(
                        model=settings.AI_PRIMARY_MODEL,
                        contents=[
                            f"Extract all text from this image. The text may be in {language} language. "
                            "Return only the extracted text, preserving the original formatting. "
                            "If there is no text in the image, return an empty string.",
                            types.Part.from_bytes(data=image_data, mime_type=mime_type),
                        ],
                    )
                    text = response.text or ""
                    print(f"[OCR] Extracted {len(text)} chars from image")
                    return text
                except Exception as img_err:
                    print(f"[OCR] Image extraction failed: {img_err}")
                    import traceback
                    traceback.print_exc()
                    return ""

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
    def _audience_context(role: str = "student", grade: int | None = None) -> str:
        """Build an audience description string for AI prompts."""
        if role in ("teacher", "org_admin", "guardian") or (grade is None and role == "normal_user"):
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
            )
        if grade and grade <= 8:
            return (
                f"The audience is middle-school students (grade {grade}). "
                "Use age-appropriate vocabulary, relatable examples, and moderate challenge. "
                "Roleplay: be an adventurous explorer character who discovers facts alongside the student."
            )
        return (
            f"The audience is high-school students (grade {grade or 'unknown'}). "
            "Use clear academic language, exam-style rigour, and a mix of easy to hard. "
            "Roleplay: be a cool mentor/coach character who pushes the student to excel."
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

    async def playground_match_pairs(self, topic: str, role: str = "student", grade: int | None = None) -> list:
        """Match Mania / Concept Connect: generate 6 term-definition pairs."""
        audience = self._audience_context(role, grade)
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

    async def playground_swipe_facts(self, topic: str, role: str = "student", grade: int | None = None) -> list:
        """Swipe & Sort: generate 10 true/false statements."""
        audience = self._audience_context(role, grade)
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

    async def playground_speed_quiz(self, topic: str, role: str = "student", grade: int | None = None) -> dict:
        """Speed Blitz: generate 10 MCQ questions + a challenger persona."""
        audience = self._audience_context(role, grade)
        is_adult = role in ("teacher", "org_admin", "guardian") or (grade is None and role == "normal_user")

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

    async def playground_roleplay(self, topic: str, role: str = "student", grade: int | None = None) -> dict:
        """Generate an immersive roleplay scenario with character, scenes, and choices."""
        audience = self._audience_context(role, grade)
        is_adult = role in ("teacher", "org_admin", "guardian") or (grade is None and role == "normal_user")

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

    async def playground_imagine(self, topic: str, role: str = "student", grade: int | None = None) -> dict:
        """Generate a creative 'What if?' scenario with open-ended prompts."""
        audience = self._audience_context(role, grade)
        is_adult = role in ("teacher", "org_admin", "guardian") or (grade is None and role == "normal_user")

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
        max_points: int = 20, role: str = "student", grade: int | None = None
    ) -> dict:
        """Evaluate a creative open-ended answer."""
        audience = self._audience_context(role, grade)
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
