# AI Assistant tailoring verification — quick

> **NOT production routing — structural checks only (generation fell back to gpt-4o-mini; judge gpt-4o-mini)**

Run 2026-10-05 13:50 · backend http://127.0.0.1:8001 (git 56ffa5f) · user tailoring-eval@genverse.dev · judge openai:gpt-4o-mini · cases 102

## Provider / fallback usage

| Served by | Answers |
|---|---|
| openai:gpt-4o-mini | 101 |
| canned reply (no LLM) | 1 |

Fallbacks: 3 · breaker trips: 3 · guard fail-open: 31 · interrupted (G6): 0 · errors: 0

## Automatic checks per sweep (passed / applicable)

| Sweep | length_ok | sentence_ok | three_ways_ok | exam_notes_ok | ending_ok | parenthetical_ok | purity_ok | no_grade_board_mention | katex_ok | delims_ok | currency_ok | no_emoji | correct | no_formula_young | rag_both_sources | three_ways_stops_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 10/12 | 12/12 | 12/12 | 9/9 | 12/12 | – | 12/12 | 12/12 | 3/3 | 9/12 | 12/12 | 12/12 | 4/4 | 2/2 | – | – |
| S2 | 20/21 | 21/21 | 21/21 | – | 21/21 | – | 21/21 | 21/21 | 7/7 | 14/21 | 21/21 | 21/21 | 7/7 | – | – | – |
| S3 | 9/12 | 12/12 | 12/12 | – | 9/9 | – | 12/12 | 12/12 | 6/6 | 8/12 | 12/12 | 12/12 | 4/4 | – | – | – |
| S4 | 10/15 | 12/12 | 15/15 | 15/15 | 12/12 | – | 15/15 | 15/15 | 5/5 | 11/15 | 15/15 | 15/15 | 5/5 | – | – | – |
| S5 | 10/10 | 10/10 | 10/10 | – | 9/10 | – | 10/10 | 10/10 | 5/5 | 5/10 | 10/10 | 10/10 | 5/5 | – | – | – |
| S6 | 6/12 | – | – | – | 12/12 | 11/12 | 6/12 | 12/12 | 6/6 | 10/12 | 12/12 | 12/12 | 4/4 | – | – | – |
| S7 | 2/2 | 1/2 | 2/2 | 1/1 | 1/2 | – | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | – | – | – | 1/1 |
| S8 | 2/3 | – | 3/3 | 3/3 | – | – | 3/3 | 3/3 | 1/1 | 2/3 | 3/3 | 3/3 | 1/1 | – | – | – |
| S9 | 3/3 | 3/3 | 2/2 | 2/2 | 2/2 | – | 3/3 | 3/3 | 2/2 | 3/3 | 3/3 | 3/3 | – | – | – | – |
| S10 | 5/5 | 3/3 | – | – | – | 2/2 | 5/5 | 5/5 | 2/2 | 5/5 | 5/5 | 5/5 | – | – | – | – |
| S11 | 4/4 | 1/1 | – | – | – | 3/3 | 4/4 | 4/4 | – | 4/4 | 4/4 | 4/4 | – | – | – | – |
| S12 | 2/2 | 2/2 | – | – | – | – | 2/2 | 2/2 | 1/1 | 2/2 | 2/2 | 2/2 | – | – | 0/2 | – |

Truncation suspects (heuristic — finish_reason is not visible over HTTP): none

## Judge rubric (1–5, settings visible)

| Sweep | grade_fit | board_alignment | difficulty_fit | length_fit | personality_fit | language_quality | format_compliance | factual_correctness |
|---|---|---|---|---|---|---|---|---|
| S1 | 4.50 | 4.20 | 4.58 | 4.83 | 4.83 | 4.92 | 4.83 | 5.00 |
| S2 | 4.90 | 4.95 | 4.95 | 4.90 | 4.86 | 5.00 | 4.86 | 5.00 |
| S3 | 4.83 | 4.67 | 4.67 | 5.00 | 5.00 | 5.00 | 3.90 | 5.00 |
| S4 | 4.40 | 4.73 | 4.60 | 4.73 | 4.67 | 4.93 | 4.58 | 5.00 |
| S5 | 4.80 | 5.00 | 4.90 | 4.90 | 4.50 | 5.00 | 4.60 | 5.00 |
| S6 | 4.83 | 4.64 | 4.75 | 4.92 | 4.67 | 5.00 | 4.58 | 4.92 |
| S7 | 4.33 | 4.67 | 4.67 | 4.67 | 4.00 | 5.00 | 4.00 | 5.00 |
| S8 | 4.67 | 5.00 | 4.67 | 5.00 | 4.67 | 5.00 | 5.00 | 5.00 |
| S9 | 3.67 | 3.67 | 3.33 | 4.67 | 4.33 | 5.00 | 5.00 | 5.00 |
| S10 | 5.00 | 4.60 | 4.80 | 4.80 | 4.40 | 5.00 | 4.60 | 5.00 |
| S11 | 1.00 | 1.00 | 1.00 | 2.00 | 2.00 | 4.25 | 5.00 | – |
| S12 | 5.00 | 4.00 | 4.50 | 5.00 | 4.50 | 5.00 | 5.00 | 5.00 |

## Grade trends (S1, concept questions)

| Grade | Avg sentence words | Words | Vocab (≥7-letter share) |
|---|---|---|---|
| 2 | 12.3 | 74 | 0.203 |
| 5 | 15.2 | 106 | 0.263 |
| 8 | 13.7 | 132 | 0.305 |
| 11 | 18.4 | 154 | 0.327 |

Spearman vs grade — sentence length 0.8, words 1.0, vocabulary 1.0 (target ≥ 0.8; few points in --quick).

## Length vs word count (S3)

| Grade | Question | small | summary | detailed | deep-dive | strictly rising |
|---|---|---|---|---|---|---|
| 9 | q0 | 66 | 149 | 239 | 359 | PASS |
| 9 | q1 | 73 | 138 | 200 | 233 | PASS |
| 9 | q2 | 48 | 153 | 251 | 495 | PASS |

## Distinctness (cosine similarity of word counts; < 0.90 = clearly different)

| Sweep | Q | A | B | similarity |
|---|---|---|---|---|
| S5 | q1 | coach | friend | 0.991 ⚠ |
| S5 | q1 | coach | technical-expert | 0.981 ⚠ |
| S5 | q1 | mentor | technical-expert | 0.98 ⚠ |
| S5 | q1 | friend | technical-expert | 0.979 ⚠ |
| S2 | q1 | IB | State Board | 0.975 ⚠ |
| S2 | q1 | Cambridge | ICSE | 0.972 ⚠ |
| S2 | q1 | Cambridge | State Board | 0.969 ⚠ |
| S2 | q1 | CBSE | State Board | 0.968 ⚠ |
| S5 | q1 | coach | mentor | 0.967 ⚠ |
| S2 | q1 | Cambridge | None | 0.962 ⚠ |
| S2 | q1 | None | State Board | 0.962 ⚠ |
| S5 | q1 | friend | mentor | 0.962 ⚠ |
| S2 | q1 | CBSE | Tamil Nadu State Board | 0.961 ⚠ |
| S2 | q1 | ICSE | State Board | 0.961 ⚠ |
| S2 | q1 | State Board | Tamil Nadu State Board | 0.96 ⚠ |
| S5 | q1 | friend | professor | 0.959 ⚠ |
| S5 | q1 | coach | professor | 0.958 ⚠ |
| S5 | q1 | professor | technical-expert | 0.958 ⚠ |
| S2 | q1 | CBSE | IB | 0.957 ⚠ |
| S2 | q1 | IB | None | 0.956 ⚠ |
| S2 | q1 | CBSE | Cambridge | 0.955 ⚠ |
| S2 | q1 | Cambridge | IB | 0.953 ⚠ |
| S2 | q1 | CBSE | None | 0.948 ⚠ |
| S2 | q1 | ICSE | None | 0.945 ⚠ |
| S2 | q1 | IB | ICSE | 0.944 ⚠ |

Max similarity 0.991 · pairs ≥ 0.90: 40 of 92

## Blind identification (judge, settings hidden, order shuffled)

**grade** — accuracy 6/12 = 50% (chance 25%)

| truth \ guess | Grade 11 | Grade 2 | Grade 5 | Grade 8 | other |
|---|---|---|---|---|---|
| Grade 11 | 0 | 0 | 1 | 2 | 0 |
| Grade 2 | 0 | 3 | 0 | 0 | 0 |
| Grade 5 | 1 | 0 | 2 | 0 | 0 |
| Grade 8 | 2 | 0 | 0 | 1 | 0 |

**board** — accuracy 2/15 = 13% (chance 20%)

| truth \ guess | CBSE | Cambridge | IB | ICSE | State Board | other |
|---|---|---|---|---|---|---|
| CBSE | 0 | 0 | 0 | 1 | 2 | 0 |
| Cambridge | 1 | 2 | 0 | 0 | 0 | 0 |
| IB | 0 | 1 | 0 | 1 | 1 | 0 |
| ICSE | 0 | 0 | 3 | 0 | 0 | 0 |
| State Board | 2 | 0 | 0 | 1 | 0 | 0 |

**personality** — accuracy 2/10 = 20% (chance 20%)

| truth \ guess | coach | friend | mentor | professor | technical-expert | other |
|---|---|---|---|---|---|---|
| coach | 0 | 2 | 0 | 0 | 0 | 0 |
| friend | 1 | 0 | 1 | 0 | 0 | 0 |
| mentor | 0 | 0 | 0 | 2 | 0 | 0 |
| professor | 1 | 0 | 1 | 0 | 0 | 0 |
| technical-expert | 0 | 0 | 0 | 0 | 2 | 0 |

**length** — accuracy 6/12 = 50% (chance 25%)

| truth \ guess | deep-dive | detailed | small | summary | other |
|---|---|---|---|---|---|
| deep-dive | 2 | 0 | 0 | 1 | 0 |
| detailed | 0 | 2 | 0 | 1 | 0 |
| small | 0 | 0 | 2 | 1 | 0 |
| summary | 1 | 1 | 1 | 0 | 0 |

**Difficulty ordering** — Spearman of judged order vs set order per group: [-0.4, -0.2, 0.8] (1.0 = perfectly rising easy<medium<hard<expert)

## Correctness (problem-solving questions)

30/30 answers state the stored numeric answer.


## Enhancements (subset; same bodies as the frontend)

| Case | Endpoint | HTTP | Items | Script purity | Sample |
|---|---|---|---|---|---|
| S1_g2_q0_s0 | follow-ups | 200 | 4 | 1.0 | ["What are the main stages of the photosynthesis process?", "How do different light wavele |
| S1_g2_q0_s0 | next-steps | 200 | 4 | 1.0 | ["Summarize photosynthesis in 5 bullet points", "Create a 5-question quiz on photosynthesi |
| S1_g2_q0_s0 | practice-exercises | 200 | 3 | 1.0 | [{"question": "What do plants use to make their food?", "answer": "Plants use sunlight, ca |
| S1_g2_q0_s0 | mindmap | 200 | None | 1.0 | {"label": "What is photosynthesis?", "children": [{"label": "Photosynthesis process", "chi |
| S1_g11_q0_s0 | follow-ups | 200 | 4 | 1.0 | ["What are the different stages of photosynthesis and how do they interconnect?", "How do  |
| S1_g11_q0_s0 | next-steps | 200 | 4 | 1.0 | ["Summarize photosynthesis in 5 key points", "Create a quiz with 5 questions about photosy |
| S1_g11_q0_s0 | practice-exercises | 200 | 3 | 1.0 | [{"question": "Was ist die Fotosynthese?", "answer": "Die Fotosynthese ist der Prozess, be |
| S1_g11_q0_s0 | mindmap | 200 | None | 1.0 | {"label": "What is photosynthesis?", "children": [{"label": "Definition", "children": [{"l |
| S6_g8_ta_q0_s0 | follow-ups | 200 | 4 | 0.892 | ["போட்டோசின்தெசிஸ் செயல்முறையில் குளோரோபிளாஸ்ட்களின் பாதுகாப்பு மற்றும் செயல்பாடு என்ன?",  |
| S6_g8_ta_q0_s0 | next-steps | 200 | 4 | 1.0 | ["போட்டோசின்தெசிஸ் செயல்முறையை 5 புள்ளிகளில் சுருக்கமாக கூறு", "இந்த துறையுடன் தொடர்புடைய  |
| S6_g8_ta_q0_s0 | practice-exercises | 200 | 3 | 0.594 | [{"question": "போட்டோசின்தெசிஸ் என்ன?", "answer": "போட்டோசின்தெசிஸ் என்பது தாவரங்கள் sunli |
| S6_g8_ta_q0_s0 | mindmap | 200 | None | 0.378 | {"label": "போட்டோசின்தெசிஸ்", "children": [{"label": "செயல்முறை", "children": [{"label": " |
| S10_g10_ta_important_questions_q0_s0 | follow-ups | 200 | 4 | 1.0 | ["ஒளியின் பிரதிபலிப்பில் பரவலான மற்றும் ஒரே கோணத்தில் பிரதிபலிப்பில் உள்ள மாற்றங்களை விளக் |
| S10_g10_ta_important_questions_q0_s0 | next-steps | 200 | 4 | 0.971 | ["ஒளியின் பிரதிபலிப்பை 5 முக்கிய புள்ளிகளில் சுருக்கவும்", "ஒரு 5 கேள்வികളുടെ க்விஸ் உருவா |
| S10_g10_ta_important_questions_q0_s0 | practice-exercises | 200 | 3 | 0.249 | [{"question": "ஒளியின் பிரதிபலிப்பு என்ன?", "answer": "ஒளியின் பிரதிபலிப்பு என்பது ஒளி ஒரு |
| S10_g10_ta_important_questions_q0_s0 | mindmap | 200 | None | 0.645 | {"label": "ஒளியின் பிரதிபலிப்பு", "children": [{"label": "ஒளியின் பிரதிபலிப்பு என்ன?", "ch |

## 10 weakest cases

### S6_g8_ta_q1_s0 — failed: ['length_ok', 'parenthetical_ok', 'purity_ok', 'delims_ok']
Settings: grade 8, CBSE, ta, sm=True, summary, medium, mentor, 3ways=False, examples=False, fmt=None
Likely fragment: response_settings Length line (settings.length_line / matrices.LENGTH_WORDS); language.language_section (D9 parenthetical rule); language.language_section + language_reminder; hard rule 4 (math)
Judge: {}

> ஒரு செவ்வகத்தின் பரப்பு (Area) கணக்கிடுவதற்கான சூத்திரம்: 
> 
> \[
> \text{Area} = \text{length} \times \text{width}
> \]
> 
> இந்த செவ்வகத்தின் நீளம் 12 செ.மீ மற்றும் அகலம் 5 செ.மீ ஆகும். 
> 
> இதனால், பரப்பு:
> 
> \[
> \text{Area} = 12\,\mathrm{cm} \times 5\,\mathrm{cm} = 60\,\mathrm{cm^2}
> \]
> 
> எனவே, செவ்வகத்தின் பரப்பு 60 செ.மீ². 
> 
> உங்களுக்குத் தெரியுமா, செவ்வகத்தின் பரப்பை கணக்கிடுவதற்கு மற்ற எதாவது சூத்திரங்கள் உள்ளனவா?

### S6_g8_ar_q1_s0 — failed: ['length_ok', 'purity_ok', 'delims_ok']
Settings: grade 8, CBSE, ar, sm=True, summary, medium, mentor, 3ways=False, examples=False, fmt=None
Likely fragment: response_settings Length line (settings.length_line / matrices.LENGTH_WORDS); language.language_section + language_reminder; hard rule 4 (math)
Judge: {}

> لحساب مساحة المستطيل، نستخدم الصيغة:
> 
> $$
> \text{المساحة} = \text{الطول} \times \text{العرض}
> $$
> 
> حيث الطول هو \(12\,\mathrm{cm}\) والعرض هو \(5\,\mathrm{cm}\).
> 
> إذًا:
> 
> $$
> \text{المساحة} = 12\,\mathrm{cm} \times 5\,\mathrm{cm} = 60\,\mathrm{cm^2}
> $$
> 
> لذا، مساحة المستطيل هي \(60\,\mathrm{cm^2}\). 
> 
> هل يمكنك أن تخبرني ما هي الأشكال الأخرى التي تعرفها؟

### S7_g5_3ways_on_q0_s0 — failed: ['sentence_ok', 'ending_ok']
Settings: grade 5, CBSE, en, sm=True, summary, medium, mentor, 3ways=True, examples=False, fmt=None
Likely fragment: learner_profile band fragment (learner.band_fragment); settings.personality_line (_ENDING)
Judge: {"grade_fit": "Concept is appropriate but examples exceed Grade 5 complexity.", "format_compliance": "Included examples instead of focusing solely on explanations."}

> Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy. During photosynthesis, these organisms use sunlight, carbon dioxide from the air, and water from the soil to produce glucose (a type of sugar) and oxygen. This process mainly occurs in the leaves, where chlorophyll, the green pigment, captures sunlight.
> 
> For example, when a plant gets sunlight, it takes in carbon dioxide and water, and through photosynthesis, it makes its food

### S3_g9_deep-dive_q1_s0 — failed: ['length_ok', 'delims_ok']
Settings: grade 9, CBSE, en, sm=True, deep-dive, medium, mentor, 3ways=False, examples=False, fmt=None
Likely fragment: response_settings Length line (settings.length_line / matrices.LENGTH_WORDS); hard rule 4 (math)
Judge: {"format_compliance": "Includes a format that was not requested."}

> To solve the quadratic equation \(x^2 - 5x + 6 = 0\), we can use the factorization method. Here are the steps:
> 
> ### Step 1: Identify the equation
> The equation is given as:
> \[
> x^2 - 5x + 6 = 0
> \]
> 
> ### Step 2: Factor the quadratic
> We need to factor the quadratic expression \(x^2 - 5x + 6\). We look for two numbers that multiply to \(6\) (the constant term) and add up to \(-5\) (the coefficient of \(x\)).
> 
> The numbers \(-2\) and \(-3\) satisfy these conditions:
> - \(-2 \times -3 = 6\)
> - \(-2 + -3 = 

### S4_g8_medium_q1_s0 — failed: ['length_ok', 'delims_ok']
Settings: grade 8, CBSE, en, sm=True, summary, medium, mentor, 3ways=False, examples=False, fmt=None
Likely fragment: response_settings Length line (settings.length_line / matrices.LENGTH_WORDS); hard rule 4 (math)
Judge: {}

> To find the area of a rectangle, you can use the formula:
> 
> \[
> \text{Area} = \text{Length} \times \text{Width}
> \]
> 
> In this case, the length of the rectangle is \(12\,\mathrm{cm}\) and the width is \(5\,\mathrm{cm}\). 
> 
> Now, substitute the values into the formula:
> 
> \[
> \text{Area} = 12\,\mathrm{cm} \times 5\,\mathrm{cm} = 60\,\mathrm{cm^2}
> \]
> 
> Thus, the area of the rectangle is \(60\,\mathrm{cm^2}\). 
> 
> Can you tell me what the perimeter of the rectangle would be?

### S1_g5_q1_s0 — failed: ['length_ok', 'delims_ok']
Settings: grade 5, CBSE, en, sm=True, summary, medium, mentor, 3ways=False, examples=False, fmt=None
Likely fragment: response_settings Length line (settings.length_line / matrices.LENGTH_WORDS); hard rule 4 (math)
Judge: {}

> To find the area of a rectangle, we use the formula:
> 
> \[
> \text{Area} = \text{Length} \times \text{Width}
> \]
> 
> Here, the length of the rectangle is 12 cm and the width is 5 cm. 
> 
> Now, we can calculate the area:
> 
> \[
> \text{Area} = 12\,\mathrm{cm} \times 5\,\mathrm{cm} = 60\,\mathrm{cm}^2
> \]
> 
> So, the area of the rectangle is \(60\,\mathrm{cm}^2\). 
> 
> Can you think of another shape and how to find its area?

### S1_g8_q1_s0 — failed: ['length_ok', 'delims_ok']
Settings: grade 8, CBSE, en, sm=True, summary, medium, mentor, 3ways=False, examples=False, fmt=None
Likely fragment: response_settings Length line (settings.length_line / matrices.LENGTH_WORDS); hard rule 4 (math)
Judge: {}

> To find the area of a rectangle, you can use the formula:
> 
> \[
> \text{Area} = \text{Length} \times \text{Width}
> \]
> 
> In this case, the length of the rectangle is \(12\,\mathrm{cm}\) and the width is \(5\,\mathrm{cm}\).
> 
> Now, substituting the values into the formula:
> 
> \[
> \text{Area} = 12\,\mathrm{cm} \times 5\,\mathrm{cm} = 60\,\mathrm{cm^2}
> \]
> 
> Therefore, the area of the rectangle is \(60\,\mathrm{cm^2}\). 
> 
> Can you tell me what the perimeter of the rectangle would be?

### S2_g10_StateBoard_q1_s0 — failed: ['length_ok', 'delims_ok']
Settings: grade 10, State Board, en, sm=True, summary, medium, mentor, 3ways=False, examples=False, fmt=None
Likely fragment: response_settings Length line (settings.length_line / matrices.LENGTH_WORDS); hard rule 4 (math)
Judge: {}

> To solve the quadratic equation \(x^2 - 5x + 6 = 0\), we can factor it. We look for two numbers that multiply to \(6\) (the constant term) and add up to \(-5\) (the coefficient of \(x\)). 
> 
> The numbers \(-2\) and \(-3\) satisfy these conditions because:
> 
> \[
> -2 \times -3 = 6 \quad \text{and} \quad -2 + -3 = -5
> \]
> 
> Thus, we can factor the equation as:
> 
> \[
> (x - 2)(x - 3) = 0
> \]
> 
> Next, we set each factor equal to zero:
> 
> 1. \(x - 2 = 0 \Rightarrow x = 2\)
> 2. \(x - 3 = 0 \Rightarrow x = 3\)
> 
> Therefore

### S3_g9_detailed_q1_s0 — failed: ['length_ok', 'delims_ok']
Settings: grade 9, CBSE, en, sm=True, detailed, medium, mentor, 3ways=False, examples=False, fmt=None
Likely fragment: response_settings Length line (settings.length_line / matrices.LENGTH_WORDS); hard rule 4 (math)
Judge: {}

> To solve the quadratic equation \( x^2 - 5x + 6 = 0 \), we can use the factorization method.
> 
> ### Step 1: Write the equation
> The equation is already given as:
> $$
> x^2 - 5x + 6 = 0
> $$
> 
> ### Step 2: Factor the quadratic
> We need to find two numbers that multiply to \( 6 \) (the constant term) and add up to \( -5 \) (the coefficient of \( x \)). The numbers \( -2 \) and \( -3 \) satisfy these conditions because:
> - \( -2 \times -3 = 6 \)
> - \( -2 + (-3) = -5 \)
> 
> Thus, we can factor the equation as:
> $$
> (

### S4_g8_hard_q1_s0 — failed: ['length_ok', 'delims_ok']
Settings: grade 8, CBSE, en, sm=True, summary, hard, mentor, 3ways=False, examples=False, fmt=None
Likely fragment: response_settings Length line (settings.length_line / matrices.LENGTH_WORDS); hard rule 4 (math)
Judge: {}

> To find the area of a rectangle, you can use the formula:
> 
> \[
> \text{Area} = \text{Length} \times \text{Width}
> \]
> 
> In this case, the length of the rectangle is \(12\,\mathrm{cm}\) and the width is \(5\,\mathrm{cm}\). 
> 
> Now, substitute the values into the formula:
> 
> \[
> \text{Area} = 12\,\mathrm{cm} \times 5\,\mathrm{cm}
> \]
> 
> Calculating this gives:
> 
> \[
> \text{Area} = 60\,\mathrm{cm^2}
> \]
> 
> Thus, the area of the rectangle is \(60\,\mathrm{cm^2}\). 
> 
> What is the area of a rectangle with a length of \(10

## Tokens & cost

- HTTP generations: 104 (token usage is not exposed over HTTP; estimated output ≈ 26,204 tokens from answer length).
- Judge (openai:gpt-4o-mini): 116 calls, 97,529 in / 25,881 out → $0.03.
- Generation cost estimate: see provider mix; $0.06 if all served by gpt-4o-mini.

