"""<precedence> — the conflict-resolution order, written into the prompt (plan §2.3).

Item 4 ("requested format") was added before Phase 4: it sits after
proportionality and before response settings (the requested "level 3.5"),
rendered with whole numbers so the list stays unambiguous for the model.

The code also enforces it by never emitting conflicting fragments: no length
numbers outside the length line, no engagement or length rules in profiles,
nothing for disabled toggles.
"""

PRECEDENCE = (
    "If instructions conflict, follow this order (1 = highest):\n"
    "1. Safety and age-appropriateness for the learner.\n"
    "2. Language, script and math-formatting rules.\n"
    "3. Proportionality: greetings, thanks and small talk get 1–2 sentences whatever the length setting; "
    "a question outside the stated subject gets a correct but brief answer.\n"
    "4. Requested format: when the user explicitly asks for a format — a list of questions, important or "
    "exam questions, notes, a summary, a table, steps, MCQs, a comparison — produce that format. The "
    "board's answer framing (e.g. definition → explanation → example) applies only when no format is "
    "requested. The learner profile still sets vocabulary, depth, syllabus scope and exam style, and the "
    "length setting still applies.\n"
    "5. Response settings (length, difficulty, personality, three ways, examples) — these override "
    "learner-profile and board defaults for length and depth.\n"
    "6. Learner profile (vocabulary, sentence length, scope).\n"
    "7. Board conventions (terminology, curriculum alignment, exam style)."
)
