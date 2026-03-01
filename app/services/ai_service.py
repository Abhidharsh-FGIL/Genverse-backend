"""
AI Service - Wraps Google Gemini (primary) and OpenAI (fallback) for all AI operations.

Configure the AI provider via .env:
  GOOGLE_GEMINI_API_KEY=...
  OPENAI_API_KEY=...
  AI_PRIMARY_MODEL=gemini-2.5-flash
"""
import json
from typing import AsyncIterator, List, Optional, Any, Dict
from pathlib import Path

from app.config import settings


class AIService:
    """Unified AI service wrapping Gemini and OpenAI."""

    def __init__(self):
        self._gemini_client = None
        self._openai_client = None

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

    def _build_context_prompt(self, context: dict | None) -> str:
        if not context:
            return ""
        parts = []
        if context.get("grade"):
            parts.append(f"Grade: {context['grade']}")
        if context.get("board"):
            parts.append(f"Board: {context['board']}")
        if context.get("subject"):
            parts.append(f"Subject: {context['subject']}")
        if context.get("language"):
            parts.append(f"Response language: {context['language']}")
        if context.get("difficulty"):
            parts.append(f"Difficulty level: {context['difficulty']}")
        if context.get("tone"):
            parts.append(f"Tone: {context['tone']}")
        return "\n".join(parts) if parts else ""

    def _build_settings_prompt(self, chat_settings: dict | None) -> str:
        """Build a system prompt section from per-chat AI settings."""
        if not chat_settings:
            return ""
        parts = []

        personality_map = {
            "mentor": "Act as a supportive mentor who guides with encouragement and wisdom. Inspire curiosity and celebrate progress.",
            "coach": "Act as a focused, results-driven coach. Be direct, action-oriented, and push the student to think critically and improve.",
            "tutor": "Act as a patient tutor who explains concepts step by step, checking for understanding.",
            "friend": "Act as a knowledgeable friend — explain things in a casual, relatable, conversational way without being overly formal.",
            "professor": "Act as an authoritative professor delivering comprehensive, academically rigorous responses with precise terminology.",
            "technical-expert": "Act as a senior technical expert. Use precise, industry-standard terminology, dive into implementation details, and reference best practices.",
            "helpful": "Be helpful and informative in your responses.",
        }
        personality = chat_settings.get("personality", "helpful")
        parts.append(personality_map.get(personality, personality_map["helpful"]))

        difficulty_map = {
            "easy": "Use very simple language suitable for complete beginners. Avoid technical jargon entirely; prefer everyday analogies and short sentences.",
            "medium": "Use clear explanations with moderate technical depth, suitable for intermediate learners who know the basics.",
            "hard": "Use advanced concepts and technical terminology appropriate for advanced learners. Do not over-explain foundational concepts.",
            "expert": "Assume expert-level knowledge. Skip introductory definitions, focus on nuance, edge cases, trade-offs, and cutting-edge aspects of the topic.",
        }
        difficulty = chat_settings.get("difficulty", "medium")
        parts.append(difficulty_map.get(difficulty, difficulty_map["medium"]))

        length_map = {
            "small": "Be extremely brief — answer in 1-3 sentences maximum. No preamble, no lists, just the core answer.",
            "brief": "Keep responses concise (1-2 paragraphs max). Get to the point quickly.",
            "summary": "Give a focused summary response (2-3 paragraphs). Cover the key points without going into excessive detail.",
            "medium": "Provide moderately detailed responses covering key points without being exhaustive.",
            "detailed": "Provide comprehensive, in-depth explanations covering all relevant aspects thoroughly with examples and structure.",
            "deep-dive": "Provide an exhaustive, thorough deep-dive. Cover every important aspect, subtlety, edge case, and example. Use headings and structure to organise the response.",
        }
        content_length = chat_settings.get("content_length", "medium")
        parts.append(length_map.get(content_length, length_map["medium"]))

        if chat_settings.get("explain_3ways"):
            parts.append(
                "IMPORTANT — Explain in 3 Ways: Structure your response with these three clearly labelled sections "
                "directly inside your answer (do NOT add a separate card or section after — include all three here):\n"
                "**🎯 Analogy:** A simple, relatable analogy or metaphor that makes the concept easy to grasp.\n"
                "**⚙️ Technical:** A precise, formal definition or technical explanation.\n"
                "**🌍 Real-World Example:** A concrete, real-world application or example of the concept in action."
            )

        if chat_settings.get("examples"):
            parts.append("Always include concrete, real-world examples to illustrate concepts.")

        if chat_settings.get("mind_map"):
            parts.append(
                "At the end of your response, include a brief mind-map outline using bullet points "
                "showing how the key concepts relate to each other."
            )

        output_mode = chat_settings.get("output_mode", "text")
        if output_mode == "structured":
            parts.append("Structure your response with clear headings (##) and logical sections.")
        elif output_mode == "bullets":
            parts.append("Present information primarily using bullet points and numbered lists.")

        if chat_settings.get("student_mode"):
            parts.append(
                "You are in Student Mode. Use encouraging language, celebrate correct answers, "
                "and break down complex topics into digestible steps."
            )

        # Note: followup, next_steps, and practice are handled as separate UI cards
        # after the response — do NOT include them inside the response body.

        return "\n".join(parts) if parts else ""

    async def chat(self, messages: List[dict], context: dict | None = None, chat_settings: dict | None = None) -> str:
        """Non-streaming chat with AI."""
        context_str = self._build_context_prompt(context)
        settings_str = self._build_settings_prompt(chat_settings)
        system_prompt = (
            "You are Genverse.ai, an AI-powered educational assistant. "
            "FORMATTING RULES — follow these strictly:\n"
            "1. Math: use $...$ for inline math and $$...$$ for display/block math. "
            "Never use \\(...\\) or \\[...\\] notation.\n"
            "2. Chemical equations: use $\\ce{...}$ notation (e.g. $\\ce{H2O}$, $\\ce{2H2 + O2 -> 2H2O}$).\n"
            "3. Tables: use standard Markdown pipe table syntax (| col | col | with a header separator row).\n"
            "4. Lists, headings, bold, italic, code blocks: use standard Markdown syntax."
        )
        if context_str:
            system_prompt += f"\n{context_str}"
        if settings_str:
            system_prompt += f"\n\n{settings_str}"
        full_prompt = system_prompt + "\n\n" + "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )

        gemini = self._get_gemini()
        if gemini:
            try:
                response = gemini.generate_content(full_prompt)
                return response.text
            except Exception:
                pass

        openai = self._get_openai()
        if openai:
            try:
                response = await openai.chat.completions.create(
                    model=settings.AI_FALLBACK_MODEL,
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                )
                return response.choices[0].message.content
            except Exception:
                pass

        return "AI service is not configured or all providers failed. Please check your API keys."

    async def stream_chat(
        self, messages: List[dict], context: dict | None = None, chat_settings: dict | None = None
    ) -> AsyncIterator[str]:
        """SSE streaming chat with AI."""
        context_str = self._build_context_prompt(context)
        settings_str = self._build_settings_prompt(chat_settings)
        system_prompt = (
            "You are Genverse.ai, an AI-powered educational assistant. "
            "FORMATTING RULES — follow these strictly:\n"
            "1. Math: use $...$ for inline math and $$...$$ for display/block math. "
            "Never use \\(...\\) or \\[...\\] notation.\n"
            "2. Chemical equations: use $\\ce{...}$ notation (e.g. $\\ce{H2O}$, $\\ce{2H2 + O2 -> 2H2O}$).\n"
            "3. Tables: use standard Markdown pipe table syntax (| col | col | with a header separator row).\n"
            "4. Lists, headings, bold, italic, code blocks: use standard Markdown syntax."
        )
        if context_str:
            system_prompt += f"\n{context_str}"
        if settings_str:
            system_prompt += f"\n\n{settings_str}"
        full_prompt = system_prompt + "\n\n" + "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )

        gemini = self._get_gemini()
        if gemini:
            try:
                response = gemini.generate_content(full_prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                return
            except Exception:
                pass

        openai = self._get_openai()
        if openai:
            stream = await openai.chat.completions.create(
                model=settings.AI_FALLBACK_MODEL,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return

        yield "AI service not configured."

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
            source_section = (
                "SOURCE TEXT (generate questions ONLY from this content, do not use outside knowledge):\n"
                f"---\n{source_text[:5000]}\n---"
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
        """Retrieve images using Google Custom Search API for ebook cover and chapters."""
        import asyncio
        import base64
        import aiohttp

        _API_KEY = "AIzaSyA_Zfen4abuUycPzB-p12i4zGrbOe7o0ng"
        _CX = "1536a454c256149b5"
        _SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
        _HEADERS = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.google.com",
        }

        grade_str = f"Grade {grade}" if grade else ""
        subj_str = subject or ""

        images_per_chapter = {"minimal": 0, "standard": 1, "visual_heavy": 2}.get(image_density, 1)
        result: dict = {"cover_image": None, "chapter_images": {}}

        async def _search_and_fetch(query: str) -> str | None:
            """Search Google Images for *query* and return the first result as a base64 data URL."""
            try:
                params = {
                    "key": _API_KEY,
                    "cx": _CX,
                    "searchType": "image",
                    "q": query,
                    "num": 1,
                    "safe": "active",
                    "imgType": "photo",
                }
                timeout = aiohttp.ClientTimeout(total=20)
                async with aiohttp.ClientSession(headers=_HEADERS, timeout=timeout) as session:
                    async with session.get(_SEARCH_URL, params=params) as resp:
                        if resp.status != 200:
                            return None
                        data = await resp.json()
                        items = data.get("items", [])
                        if not items:
                            return None
                        img_url: str = items[0]["link"]

                    async with session.get(img_url) as img_resp:
                        if img_resp.status != 200:
                            return None
                        img_bytes = await img_resp.read()
                        content_type = img_resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
                        if not content_type.startswith("image/"):
                            content_type = "image/jpeg"
                        b64 = base64.b64encode(img_bytes).decode()
                        return f"data:{content_type};base64,{b64}"
            except Exception:
                return None

        def _chapter_query(ch: dict, img_idx: int) -> str:
            ch_title = ch.get("title", "")
            key_pts = ch.get("key_points", []) or []
            concepts = " ".join(str(k) for k in key_pts[:2]) if key_pts else ""
            parts = [p for p in [subj_str, grade_str, ch_title, concepts] if p]
            query = " ".join(parts).strip()
            if img_idx > 0:
                query += " diagram illustration"
            return query

        tasks: list[tuple[str, int | None, object]] = []

        cover_query_parts = [p for p in [subj_str, grade_str, title, "educational"] if p]
        cover_query = " ".join(cover_query_parts)
        tasks.append(("cover", None, _search_and_fetch(cover_query)))

        if images_per_chapter > 0:
            max_chapters = 10
            for i, ch in enumerate(chapters[:max_chapters]):
                for img_idx in range(images_per_chapter):
                    tasks.append((f"ch_{i}_{img_idx}", i, _search_and_fetch(_chapter_query(ch, img_idx))))

        data_urls = await asyncio.gather(*[t[2] for t in tasks])
        for (key, ch_idx, _), data_url in zip(tasks, data_urls):
            if key == "cover":
                result["cover_image"] = data_url
            elif data_url and ch_idx is not None:
                ch_key = str(ch_idx)
                result["chapter_images"].setdefault(ch_key, [])
                result["chapter_images"][ch_key].append(data_url)

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

        # Generate images via Google Custom Search if requested
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
            except Exception:
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
- For MCQ: include "options" (array of 4 strings) and "correctAnswer" (index 0-3 as number)
- For fill-blank: include "correctAnswer" as a string
- For true-false: include "correctAnswer" as "true" or "false"
- For match: include "matchPairs" as array of {{"left": "...", "right": "..."}} (4-5 pairs)
- For short-answer: no extra fields needed

Return ONLY valid JSON, no markdown.
Example: {{"questions": [{{"type": "mcq", "text": "...", "options": ["A","B","C","D"], "correctAnswer": 0, "points": 5}}]}}
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
        prompt = f"""Analyze career compatibility for a student:
Interests: {', '.join(interests)}
Strengths: {', '.join(strengths)}
Target careers: {', '.join(target_careers or ['Not specified'])}
Grade: {grade or 'Not specified'}

Return JSON:
{{
  "top_careers": ["..."],
  "compatibility_scores": {{"career_name": 0-100}},
  "strengths_analysis": "...",
  "recommended_paths": ["..."],
  "skills_to_develop": ["..."],
  "roadmap": "..."
}}

Return ONLY valid JSON.
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
                    {"title": "Take your first assessment", "type": "explore", "priority": 90, "subject": None, "action_href": "/u/assessments"},
                    {"title": "Create a topic-based quiz", "type": "explore", "priority": 75, "subject": None, "action_href": "/u/assessments"},
                    {"title": "Upload study material to vault", "type": "explore", "priority": 50, "subject": None, "action_href": "/u/library"},
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
      "subject": "subject name or null",
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
                "goals": [{"title": "Keep taking assessments", "type": "practice", "priority": 70, "subject": None, "action_href": "/u/assessments"}],
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
        self, paper_id: str, subjects: List[dict], question_types: List[str] | None
    ) -> List[dict]:
        """Generate institutional evaluation questions."""
        types_str = ", ".join(question_types or ["MCQ", "Short Answer"])
        subjects_str = json.dumps(subjects, indent=2)
        prompt = f"""Generate exam questions for an institutional assessment:
Subjects config:
{subjects_str[:2000]}
Question types: {types_str}

For each chapter in each subject, generate questions based on weightage.
Return JSON array:
[
  {{
    "type": "MCQ|fill|true_false|short|long",
    "text": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "A",
    "marks": 1,
    "subject": "...",
    "chapter": "...",
    "difficulty": "easy|medium|hard",
    "explanation": "..."
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

    async def extract_video_search_query(self, user_message: str, ai_response: str) -> str:
        """Extract the best YouTube search query from a Q&A exchange."""
        prompt = (
            "Extract a concise, specific YouTube search query (5-8 words max) that would find "
            "the most relevant educational video for this topic.\n\n"
            f"User asked: {user_message[:300]}\n"
            f"Topic summary: {ai_response[:400]}\n\n"
            "Return ONLY the search query string, nothing else."
        )
        response = await self.chat([{"role": "user", "content": prompt}])
        return response.strip().strip('"').strip("'")[:100]

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
                import PyPDF2
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                return text

            elif ext in (".docx",):
                from docx import Document
                doc = Document(file_path)
                return "\n".join(para.text for para in doc.paragraphs)

            elif ext in (".txt",):
                return path.read_text(encoding="utf-8", errors="ignore")

            elif ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
                # For images, use Gemini's vision capability or pytesseract
                gemini = self._get_gemini()
                if gemini:
                    import google.generativeai as genai
                    vision_model = genai.GenerativeModel("gemini-2.5-flash")
                    with open(file_path, "rb") as f:
                        image_data = f.read()
                    response = vision_model.generate_content([
                        "Extract all text from this image. Return only the extracted text.",
                        {"mime_type": "image/jpeg", "data": image_data},
                    ])
                    return response.text
                return ""

        except Exception as e:
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
