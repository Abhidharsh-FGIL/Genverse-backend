"""The single AI Assistant prompt builder (plan §2; audit §7.4 duplication).

`build_system_prompt` is the only place the chat system prompt is assembled;
AIService.chat() and AIService.stream_chat() both call it. Sections, in order:

    <role> <hard_rules>                      always
    <language>                               user-facing calls (a context or settings was passed)
    <learner_profile>                        only when student mode is ON and there is a grade/board/subject (G9)
    <response_settings>                      when chat settings were passed — one line per active dimension (P6)
    <precedence>                             user-facing calls

Internal service calls (JSON generators: context=None, chat_settings=None) get
<role> + <hard_rules> only, so their own language/format instructions are
never contradicted.

`provider` is accepted so call sites don't change later; it only affects
message *structure* (Phase 3), never the prompt text.
"""
import re
from typing import Literal, Optional

from app.prompts.base import render_hard_rules, render_role
from app.prompts.language import language_reminder, language_section
from app.prompts.learner import board_profile_key, learner_profile
from app.prompts.matrices import grade_band
from app.prompts.precedence import PRECEDENCE
from app.prompts.settings import (
    THREE_WAYS_MARKER,
    THREE_WAYS_OFF_AFTER_USE,
    difficulty_line,
    examples_line,
    length_line,
    personality_line,
    three_ways_line,
)
from app.schemas.ai_payloads import AIContextPayload, ChatSettingsPayload

Provider = Literal["gemini", "claude", "openai"]


def effective_student_mode(ctx: Optional[AIContextPayload], cs: Optional[ChatSettingsPayload]) -> bool:
    """Single student-mode authority (audit I7).

    Chat settings win. Without chat settings, the context's flag counts only if
    the caller actually sent it (plan Phase 2 addition 4): a caller that sends no
    student_mode gets no learner profile.
    """
    if cs is not None:
        return cs.student_mode
    return bool(ctx is not None and "student_mode" in ctx.model_fields_set and ctx.student_mode)


def history_uses_three_ways(messages: list[dict]) -> bool:
    """True if an earlier assistant answer in the window used the 3-ways format (plan D5)."""
    return any(m.get("role") == "assistant" and THREE_WAYS_MARKER in (m.get("content") or "")
               for m in messages)


TASK_PREAMBLE = ("Instructions from the feature you are serving — follow them for this answer; they take "
                 "priority over the board's default answer framing:")


def _task_body(task: str) -> str:
    return f"{TASK_PREAMBLE}\n{task}"


def caller_task(messages: list[dict]) -> Optional[str]:
    """System-role turns sent by the calling feature, joined (R1). None if there are none.

    Before the refactor these reached the model inside the conversation (Gemini's
    flattened prompt / OpenAI's message list); `to_provider_messages` drops
    system turns, so they are moved into the system prompt instead.
    """
    parts = [(m.get("content") or "").strip() for m in messages if m.get("role") == "system"]
    parts = [p for p in parts if p]
    return "\n\n".join(parts) or None


_THREE_WAYS_BLOCK = re.compile(r"(?:^[ \t]*---[ \t]*\n\s*)?\*\*Analogy:\*\*", re.M)


def strip_three_ways(messages: list[dict]) -> list[dict]:
    """Copy of `messages` with the Explain-in-3-Ways block removed from assistant turns (O2).

    Used when 3-ways is OFF: earlier answers in the window would otherwise show
    the model the Analogy / Technical / Real-World Example pattern to copy (a
    prompt line alone did not stop gpt-4o-mini). Only blocks that carry the
    "**Analogy:**" label followed by another 3-ways label are cut, from the
    "---" divider (if any). Returns new dicts; stored history is untouched.
    """
    out = []
    for m in messages:
        content = m.get("content") or ""
        if m.get("role") == "assistant":
            match = _THREE_WAYS_BLOCK.search(content)
            if match and re.search(r"\*\*(?:Technical|Real-World Example):\*\*", content[match.end():]):
                m = {**m, "content": content[:match.start()].rstrip()}
        out.append(m)
    return out


def _section(tag: str, body: str) -> str:
    return f"<{tag}>\n{body}\n</{tag}>"


def build_system_prompt(
    ctx: Optional[AIContextPayload],
    settings: Optional[ChatSettingsPayload],
    provider: Provider = "gemini",
    *,
    history_has_three_ways: bool = False,
    extra_instruction: Optional[str] = None,
    question: Optional[str] = None,
    task: Optional[str] = None,
) -> str:
    """Assemble the full system prompt.

    `extra_instruction` = guard grade_context (non-stream path).
    `question` = the student's raw message (exam-intent detection for exam notes).
    `task` = system-role instructions from the calling feature (e.g. the
    Playground mode prompt), rendered as a final <task> section (regression R1).
    The AI Assistant never sends system turns, so its prompt is unaffected.
    """
    is_user_chat = ctx is not None or settings is not None
    # The AI Assistant is the only caller that sends chat settings (R3).
    is_assistant = settings is not None
    student = effective_student_mode(ctx, settings)
    sections = [_section("role", render_role(student)),
                _section("hard_rules", render_hard_rules(assistant=is_assistant))]
    if not is_user_chat:
        if task:
            sections.append(_section("task", _task_body(task)))
        return "\n\n".join(sections)

    grade = ctx.grade if ctx else None
    board = ctx.board if ctx else None
    board_key = ctx.board_key if ctx else None
    subject = ctx.subject if ctx else None
    language = (ctx.language if ctx else None) or "en"
    content_length = settings.content_length if settings else "summary"

    # Learner-specific inputs exist only in student mode (G9).
    band = grade_band(grade) if student and grade else None
    sections.append(_section("language", language_section(
        language,
        band=band,
        grade=grade if student else None,
        subject=subject if student else None,
    )))

    if student:
        profile = learner_profile(grade, board, board_key, subject, content_length, question)
        if profile:
            sections.append(_section("learner_profile", profile))

    if settings is not None:
        lines = [
            length_line(band, content_length, language),
            difficulty_line(settings.difficulty, grade if student else None, band, board_profile_key(board_key)),
            personality_line(settings.personality, band, content_length, student),
        ]
        if settings.explain_3ways:
            lines.append(three_ways_line(content_length))
        elif history_has_three_ways:
            lines.append(THREE_WAYS_OFF_AFTER_USE)
        if settings.examples:
            lines.append(examples_line(content_length))
        sections.append(_section("response_settings", "\n".join(lines)))

    sections.append(_section("precedence", PRECEDENCE))
    if task:
        sections.append(_section("task", _task_body(task)))
    prompt = "\n\n".join(sections)
    if extra_instruction:
        prompt += f"\n\n{extra_instruction}"
    return prompt


# Placeholder for a history window that starts with an assistant turn. Neutral
# on purpose: the old "Hello" could trigger the greeting rule (precedence 3).
EARLIER_CONVERSATION = "(Continuing our earlier conversation.)"


def to_provider_messages(messages: list[dict], language: Optional[str]) -> list[dict]:
    """Provider-ready turns, identical for Gemini, Claude and OpenAI (audit P5, G14).

    - drops system-role turns (callers' system turns go into the system prompt
      as <task> via `caller_task`, regression R1);
    - merges consecutive same-role turns (G14: the Claude path claimed to, but didn't);
    - guarantees the first turn is a user turn;
    - appends the language reminder to the LAST user turn (P5).
    Returns NEW dicts — the caller's list (chat history) is never modified, so the
    reminder is never persisted and never reaches the guard (amendment 9).
    """
    turns: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content") or ""
        if turns and turns[-1]["role"] == role:
            turns[-1] = {"role": role, "content": f"{turns[-1]['content']}\n\n{content}"}
        else:
            turns.append({"role": role, "content": content})
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": EARLIER_CONVERSATION})
    if language is not None:
        for i in range(len(turns) - 1, -1, -1):
            if turns[i]["role"] == "user":
                turns[i] = {"role": "user", "content": f"{turns[i]['content']}\n\n{language_reminder(language)}"}
                break
    return turns


_SECTION_RE = re.compile(r"<(\w+)>\n(.*?)\n</\1>", re.S)


def split_sections(prompt: str) -> dict[str, str]:
    """Parse a rendered prompt back into {tag: body} (tests, preview script)."""
    return {m.group(1): m.group(2) for m in _SECTION_RE.finditer(prompt)}
