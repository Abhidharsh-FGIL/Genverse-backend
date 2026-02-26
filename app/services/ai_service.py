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
            "mentor": "Act as a supportive mentor who guides with encouragement and wisdom.",
            "tutor": "Act as a patient tutor who explains concepts step by step, checking for understanding.",
            "friend": "Act as a knowledgeable friend — explain things in a casual, relatable way without being overly formal.",
            "professor": "Act as an authoritative professor delivering comprehensive, academically rigorous responses.",
            "helpful": "Be helpful and informative in your responses.",
        }
        personality = chat_settings.get("personality", "helpful")
        if personality in personality_map:
            parts.append(personality_map[personality])

        difficulty_map = {
            "easy": "Use simple language suitable for beginners. Avoid technical jargon and prefer analogies.",
            "medium": "Use clear explanations with moderate technical depth, suitable for intermediate learners.",
            "hard": "Use advanced concepts and technical terminology appropriate for advanced or expert learners.",
        }
        difficulty = chat_settings.get("difficulty", "medium")
        if difficulty in difficulty_map:
            parts.append(difficulty_map[difficulty])

        length_map = {
            "brief": "Keep responses concise (1-2 paragraphs max). Get to the point quickly.",
            "summary": "Keep responses concise (1-2 paragraphs max). Get to the point quickly.",
            "medium": "Provide moderately detailed responses covering key points without being exhaustive.",
            "detailed": "Provide comprehensive, in-depth explanations covering all relevant aspects thoroughly.",
        }
        content_length = chat_settings.get("content_length", "medium")
        if content_length in length_map:
            parts.append(length_map[content_length])

        if chat_settings.get("explain_3ways"):
            parts.append(
                "When explaining concepts, provide THREE distinct explanations: "
                "(1) A simple analogy or metaphor, (2) A technical/formal definition, "
                "(3) A real-world application or example."
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

    async def generate_ebook(
        self,
        title: str,
        subject: str | None,
        grade: int | None,
        language: str,
        source_type: str,
        outline: List[str] | None,
        page_count: int,
    ) -> dict:
        """Generate structured eBook content as JSON."""
        outline_str = "\n".join(f"- {item}" for item in (outline or []))
        prompt = f"""Create a structured educational eBook:
Title: {title}
Subject: {subject or "General"}
Grade: {grade or "General"}
Language: {language}
Pages: ~{page_count}
{f'Outline:{chr(10)}{outline_str}' if outline_str else ''}

Return JSON with structure:
{{
  "title": "...",
  "chapters": [
    {{
      "title": "...",
      "content": "...",
      "key_points": ["..."],
      "summary": "..."
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
            return {"title": title, "chapters": [{"title": "Chapter 1", "content": response}]}

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
        self, class_id: str, topic: str, board: str, grade: int, subject: str, additional_context: str | None = None
    ) -> dict:
        """Generate a structured lesson plan."""
        prompt = f"""Create a detailed lesson plan:
Topic: {topic}
Subject: {subject}
Board: {board}
Grade: {grade}
{f'Additional context: {additional_context}' if additional_context else ''}

Return JSON:
{{
  "title": "...",
  "objectives": ["..."],
  "timeEstimate": 45,
  "steps": [
    {{"step": 1, "title": "...", "description": "...", "duration": 10}}
  ],
  "practiceTasks": ["..."],
  "formativeCheck": "...",
  "homework": "...",
  "differentiation": {{
    "easy": "...",
    "standard": "...",
    "advanced": "..."
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
            return {"title": f"Lesson Plan: {topic}", "objectives": [topic], "timeEstimate": 45, "steps": []}

    async def generate_rubric(
        self, board: str, grade: int, subject: str, topic: str, criteria_count: int
    ) -> List[dict]:
        """Generate grading rubric criteria."""
        import uuid as _uuid
        prompt = f"""Create a detailed grading rubric for:
Subject: {subject}
Board: {board}
Grade: {grade}
Topic: {topic}
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
    ) -> List[dict]:
        """Generate structured assignment questions for an AssignmentEditor."""
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

        prompt = f"""Generate assignment questions for a Grade {grade} {subject} class.
Topic: {topic}
Difficulty: {difficulty}
Question breakdown: {types_str}

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
        """Predict the most likely follow-up questions a student would ask next."""
        prompt = (
            f"Based on this educational Q&A exchange, predict the {count} most likely "
            "follow-up questions the student would want to ask next. "
            "Make questions specific, curious, and naturally flowing from the topic discussed.\n\n"
            f"Student asked: {user_message[:600]}\n\n"
            f"AI responded (summary): {ai_response[:1200]}\n\n"
            f"Return ONLY a JSON array of {count} question strings. "
            'Example: ["What is X?", "How does Y work?"]'
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
        """Generate actionable next-study-steps a student should take after this response."""
        prompt = (
            f"Based on this educational Q&A, suggest {count} specific, actionable next steps "
            "a student should take to deepen their understanding of this topic. "
            "Each step should be a short, natural sentence the student can send as their next question. "
            "Make them concrete and topic-specific.\n\n"
            f"Student asked: {user_message[:600]}\n\n"
            f"AI responded (summary): {ai_response[:1200]}\n\n"
            f"Return ONLY a JSON array of {count} short strings. "
            'Example: ["Explain photosynthesis with a diagram", "Give me practice problems on this"]'
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
        """Split text into overlapping chunks for RAG."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

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

    async def generate_audiobook(
        self,
        ebook_json: dict | None,
        language: str,
        voice_profile: str | None,
    ) -> dict:
        """Generate audiobook metadata (actual TTS integration placeholder)."""
        # In a real implementation, this would call a TTS API (ElevenLabs, Google TTS, etc.)
        # For now, return a placeholder
        return {
            "audio_path": None,
            "duration_seconds": None,
            "message": "Audiobook generation requires TTS service configuration",
        }
