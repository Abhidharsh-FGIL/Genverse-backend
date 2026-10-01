# Tailoring eval — phase2

Run at 2026-10-01T12:45:14.028915+00:00 · generator routing: primary `gemini-3.5-flash`, documents `claude-sonnet-4-20250514`, fallback `gpt-4o-mini` · forced generator: openai · judge `claude-sonnet-5-5` · guard LLM classifier: False
Cases: 30 (26 LLM generations, 4 router canned replies). Combos 1, 4, 7, 12, 13, 18; probes a1_greeting_en, a2_greeting_ta, b_off_subject, c_g2_derivative, d1_rag_g5, d2_rag_g11.

**Provider mix (who actually answered):** openai:gpt-4o-mini × 26

## Per-dimension averages (1–5; LLM generations only)

| Dimension | Avg | Target |
|---|---|---|
| grade_appropriateness | 3.15 | ≥ 4.0 |
| board_alignment | 2.82 | ≥ 4.0 |
| difficulty_fit | 2.73 | ≥ 4.0 |
| language_purity | 4.23 | ≥ 4.0 |
| latex_validity | 3.5 | ≥ 4.0 |
| personality_fit | 2.58 | ≥ 4.0 |
| length_compliance (auto) | 3.5 | ≥ 4.0 |

Automatic script purity (share of letters in the target script): mean 0.96, min 0.723.

## Cost & tokens

| Role | Model | Calls | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|---|
| generation | gpt-4o-mini | 26 | 23,393 | 7,351 | 0.0079 |
| judge | claude-sonnet-5-5 | 26 | 39,379 | 5,291 | price unknown |

**Total for priced models: $0.0079** — plus claude-sonnet-5-5 (price unknown; add it to `MODEL_PRICES_USD_PER_MTOK`).
Excludes content-guard classifier calls (separate client). Prices come from `MODEL_PRICES_USD_PER_MTOK`.

## Truncation / finish_reason

No truncation: every generation finished normally.

| Case | Provider | Model | Output tokens | finish_reason |
|---|---|---|---|---|
| c1_photosynthesis | openai | gpt-4o-mini | 33 | stop |
| c1_quadratic | openai | gpt-4o-mini | 151 | stop |
| c1_newton2 | openai | gpt-4o-mini | 33 | stop |
| c4_photosynthesis | openai | gpt-4o-mini | 316 | stop |
| c4_quadratic | openai | gpt-4o-mini | 383 | stop |
| c4_newton2 | openai | gpt-4o-mini | 290 | stop |
| c4_greeting | openai | gpt-4o-mini | 17 | stop |
| c7_photosynthesis | openai | gpt-4o-mini | 443 | stop |
| c7_quadratic | openai | gpt-4o-mini | 301 | stop |
| c7_newton2 | openai | gpt-4o-mini | 506 | stop |
| c7_greeting | openai | gpt-4o-mini | 27 | stop |
| c12_photosynthesis | openai | gpt-4o-mini | 444 | stop |
| c12_quadratic | openai | gpt-4o-mini | 443 | stop |
| c12_newton2 | openai | gpt-4o-mini | 448 | stop |
| c13_photosynthesis | openai | gpt-4o-mini | 489 | stop |
| c13_quadratic | openai | gpt-4o-mini | 408 | stop |
| c13_newton2 | openai | gpt-4o-mini | 668 | stop |
| c18_photosynthesis | openai | gpt-4o-mini | 301 | stop |
| c18_quadratic | openai | gpt-4o-mini | 232 | stop |
| c18_newton2 | openai | gpt-4o-mini | 243 | stop |
| p_a1_greeting_en | openai | gpt-4o-mini | 12 | stop |
| p_a2_greeting_ta | openai | gpt-4o-mini | 18 | stop |
| p_b_off_subject | openai | gpt-4o-mini | 85 | stop |
| p_c_g2_derivative | openai | gpt-4o-mini | 112 | stop |
| p_d1_rag_g5 | openai | gpt-4o-mini | 262 | stop |
| p_d2_rag_g11 | openai | gpt-4o-mini | 686 | stop |

## Per-case scores

| Case | Lang | Len target | Words | Len | Grade | Board | Diff | Lang | LaTeX | Pers | KaTeX err | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c1_photosynthesis | en | [5, 30] | 25 | 5 | 4 | 4 | 5 | 5 | None | 3 | – | Simple, accurate for Grade 2 and short. 'Air' is loose (carbon dioxide), acceptable at this level. Mentor persona is weak: no guiding questi |
| c1_quadratic | en | [5, 30] | 60 | 2 | 2 | 2 | 2 | 5 | 5 | 2 | 0 | Math is correct, but quadratic factorisation is Class 9-10 content, far above Grade 2. The answer gives no scaffolding or simplification for |
| c1_newton2 | en | [5, 30] | 27 | 5 | 3 | 3 | 3 | 5 | None | 2 | – | Language is simple and suits Grade 2, but 'more force means more movement' is imprecise and omits mass (acceleration depends on force and ma |
| c1_greeting | en | <= 2 sentences | 9 | 3 | – | – | – | – | – | – | – | router canned reply (no LLM) |
| c4_photosynthesis | hi | [99, 154] | 175 | 4 | 3 | 3 | 3 | 3 | 3 | 3 | – | Chemical formulas in plain Unicode rather than LaTeX; glucose formula too advanced for Grade 5. Uses transliterated 'फोटोसिंथेसिस' instead o |
| c4_quadratic | hi | [99, 154] | 162 | 4 | 2 | 2 | 3 | 3 | 5 | 3 | 0 | Quadratic equations are Class 10 content, too advanced for Grade 5, and the answer gives no scaffolding. The math is correct. The Real-World |
| c4_newton2 | hi | [99, 154] | 183 | 4 | 2 | 2 | 2 | 3 | 4 | 2 | 0 | Law is stated wrongly: force direction is said to be opposite, and 'उसकी परिमाण' is unclear. 'गाड़ी को मथते हैं' is odd wording, 'वेलोसिटी'  |
| c4_greeting | hi | <= 2 sentences | 11 | 3 | 5 | None | 5 | 5 | None | 4 | – | Short, correct Hindi greeting suitable for a Grade 5 learner. It correctly ignores the 3-ways structure for small talk. Mentor tone is mildl |
| c7_photosynthesis | ta | [210, 315] | 123 | 2 | 2 | 2 | 2 | 1 | 3 | 2 | 0 | The Tamil is garbled and the term 'மக்‌ஷணிகை' is wrong (it should be ஒளிச்சேர்க்கை). Chloroplasts are mistranslated, fungi are wrongly liste |
| c7_quadratic | ta | [210, 315] | 78 | 1 | 3 | 3 | 2 | 2 | 5 | 3 | 0 | The math is correct, but the Tamil is garbled: Russian word 'коэффициент', nonsensical phrases like 'பருத்திருப்பதற்கு' and 'இரண்டு நிலவுகள் |
| c7_newton2 | ta | [210, 315] | 147 | 3 | 2 | 3 | 1 | 2 | 5 | 2 | 0 | Tamil is garbled and wrong: 'force' is rendered as 'மாறுபாடு', 'acceleration' as 'அடிப்படை மாற்றம்', and the title is nonsense. The statemen |
| c7_greeting | ta | <= 2 sentences | 11 | 3 | 4 | None | 4 | 4 | None | 3 | – | Short two-sentence greeting in Tamil, correct length. Grammar is slightly off: 'என்ன விஷயத்தில் உதவி தேவை?' reads awkwardly, and 'விஷயம்' is |
| c12_photosynthesis | en | [380, 550] | 281 | 3 | 3 | 3 | 3 | 5 | 3 | 3 | 0 | Content is accurate but pitched at Class 11-12 (thylakoid, NADPH, Calvin cycle) rather than CBSE Class 10, which stresses stomata, chlorophy |
| c12_quadratic | en | [380, 550] | 155 | 2 | 4 | 4 | 2 | 5 | 3 | 3 | 0 | Math is correct and factorisation suits the syllabus, but the problem is very easy for a 'hard' setting, with no deeper methods or discrimin |
| c12_newton2 | en | [380, 550] | 254 | 3 | 3 | 3 | 2 | 5 | 2 | 2 | – | Too basic for 'hard' and 'detailed'. It omits momentum form (NCERT: F proportional to rate of change of momentum), derivation, vector nature |
| c12_greeting | en | <= 2 sentences | 13 | 3 | – | – | – | – | – | – | – | router canned reply (no LLM) |
| c13_photosynthesis | en | [700, 1100] | 296 | 2 | 3 | 3 | 3 | 5 | 3 | 2 | 0 | Question is biology, not Grade 11 Mathematics; the answer doesn't address the mismatch. Content is accurate but only moderately deep for 'ex |
| c13_quadratic | en | [700, 1100] | 182 | 1 | 3 | 3 | 1 | 5 | 2 | 2 | – | Math is correct, but the answer is basic with no expert depth, no verification, no discriminant or Vieta connection, and no deep-dive. Tone  |
| c13_newton2 | en | [700, 1100] | 388 | 2 | 3 | 3 | 2 | 5 | 5 | 3 | 0 | Too basic for expert deep-dive: no calculus form F=dp/dt, no variable mass, no derivation, no Class 11 momentum treatment. Mentions momentum |
| c13_greeting | en | <= 2 sentences | 13 | 3 | – | – | – | – | – | – | – | router canned reply (no LLM) |
| c18_photosynthesis | en | [120, 200] | 192 | 5 | 4 | None | 3 | 5 | 3 | 3 | 0 | Accurate and well structured, but hard difficulty is only partly met: no photosystems, electron transport, chemiosmosis, RuBisCO, G3P or C3/ |
| c18_quadratic | en | [120, 200] | 116 | 4 | 4 | None | 2 | 5 | 3 | 2 | – | Solution is correct, but it is a basic factorisation walkthrough with no depth, which is too easy for 'hard' difficulty. Tone is plain and t |
| c18_newton2 | en | [120, 200] | 177 | 5 | 3 | None | 2 | 5 | 3 | 2 | 0 | Content is correct but introductory, not hard-level: it omits the vector form, the momentum form F = dp/dt, inertial frames, and variable ma |
| c18_greeting | en | <= 2 sentences | 9 | 5 | – | – | – | – | – | – | – | router canned reply (no LLM) |
| p_a1_greeting_en | en | <= 2 sentences | 10 | 5 | 5 | None | 5 | 5 | None | 3 | – | Correct short greeting (one sentence plus an offer to help), as required for small talk. English is clean, no grade or board mentioned. Tone |
| p_a2_greeting_ta | ta | <= 2 sentences | 6 | 5 | 4 | None | 4 | 3 | None | 2 | – | Length is correct for a greeting (one short line). But the Tamil is ungrammatical: 'என்னுடைய உதவியை ஏதாவது தேவையா?' is awkward and wrong, sh |
| p_b_off_subject | en | [0, 180] | 64 | 5 | 4 | None | 4 | 5 | None | 3 | – | Correct (Valmiki) and brief, with no forced link to maths. Content is slightly Hindu-centric but acceptable. Mentor persona is weak: generic |
| p_c_g2_derivative | en | [50, 90] | 96 | 4 | 3 | None | 3 | 5 | None | 3 | – | Answered without refusing and the idea of change is kid-friendly. However, it uses 'function' and 'specific point', which are too advanced.  |
| p_d1_rag_g5 | en | [90, 140] | 142 | 4 | 2 | 2 | 1 | 4 | 3 | 3 | 0 | Pitched far above Grade 5: thylakoid, stroma, Rubisco, 3-PGA, G3P, ATP/NADPH stoichiometry. Experiment A (discs floating under LED after 7 m |
| p_d2_rag_g11 | en | [380, 550] | 432 | 5 | 2 | 3 | 2 | 5 | 3 | 2 | 0 | Omits Experiment A (discs floated after 7 minutes) and invents '30 minutes' for B. Pitched at the documents' level with heavy jargon. It doe |

## Lowest-scoring cases

### c7_photosynthesis — mean 2.00
Settings: `{"student_mode": true, "content_length": "detailed", "difficulty": "hard", "personality": "coach", "examples": true, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 8, "board": "ICSE", "subject": "Science", "language": "ta"}`
Judge: The Tamil is garbled and the term 'மக்‌ஷணிகை' is wrong (it should be ஒளிச்சேர்க்கை). Chloroplasts are mistranslated, fungi are wrongly listed as photosynthesisers, and the examples are meaningless. The content is shallow for 'hard' and 'detailed'. The coach tone is weak. \ce is used inside $...$, which may not render. The equation itself is correct, though it is not balanced with light and chlorophyll conditions.

> ### மக்‌ஷணிகை (Photosynthesis)
> 
> மக்‌ஷணிகை என்பது தாவரங்கள், பச்சை algae மற்றும் சில பூஞ்சைகள் ஆகியவற்றால் சூரிய ஒளி, கார்பன் டயொிக்சைட் மற்றும் நீரைப் பயன்படுத்தி சதைப்பொருட்களை (சர்க்கரை) தயாரிக்கும் செயல்முறை ஆகும். இதன் மூலம், உயிருக்கு அவசியமான ஆலோசனை மீண்டும் விடுகிறது.
> 
> #### முறை:
> 1. **சூரிய ஒளி**: தாவரங்கள் தங்கள் ஆக்ஸிக்கும் (chloroplasts) கச்சு மீது சூரியஒளியைக் கண்டு, அதில் உள்ள சக்தியை பயன்படுத்துகின்றன.
> 2. **கார்பன் டயொிக்சைட்**: காற்றில் உள்ள கார்பன் டயொிக்சைட்டை தாவரங்கள் உறிஞ்சுகின்றன.
> 3. **நீர்**: மண்ணில் இருந்து தாவரங்கள் நீரை உறிஞ்சுகின்றன.
> 
> இந்த மூன்று பொருட்களின் எதிர் ஆற்ற…

### c13_quadratic — mean 2.43
Settings: `{"student_mode": true, "content_length": "deep-dive", "difficulty": "expert", "personality": "professor", "examples": false, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 11, "board": "CBSE", "subject": "Mathematics", "language": "en"}`
Judge: Math is correct, but the answer is basic with no expert depth, no verification, no discriminant or Vieta connection, and no deep-dive. Tone is not professorial. It uses \( \) and \[ \] instead of the required $...$ and $$...$$ delimiters.

> To solve the quadratic equation \(x^2 - 5x + 6 = 0\), we will use the factorization method.
> 
> ### Step 1: Identify coefficients
> The given quadratic equation can be written in the standard form \(ax^2 + bx + c = 0\), where:
> - \(a = 1\),
> - \(b = -5\),
> - \(c = 6\).
> 
> ### Step 2: Factor the quadratic
> We need to find two numbers that multiply to \(c = 6\) and add up to \(b = -5\). The pairs of factors of 6 are:
> - \(1 \times 6\)
> - \(2 \times 3\)
> 
> The pair that sums to \(-5\) is \(-2\) and \(-3\):
> \[
> -2 + (-3) = -5 \quad \text{and} \quad -2 \times -3 = 6
> \]
> 
> Using these factors, we can factor the quadr…

### c7_newton2 — mean 2.57
Settings: `{"student_mode": true, "content_length": "detailed", "difficulty": "hard", "personality": "coach", "examples": true, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 8, "board": "ICSE", "subject": "Science", "language": "ta"}`
Judge: Tamil is garbled and wrong: 'force' is rendered as 'மாறுபாடு', 'acceleration' as 'அடிப்படை மாற்றம்', and the title is nonsense. The statement of the law is incorrect, since it never mentions momentum or net force. Force is called 'the unit of energy'. There is no hard-level depth, no momentum form and no units. The coach tone is weak. The LaTeX is fine.

> ### நியூட்டனைப் பெற்று இரண்டாவது சட்டம்
> 
> நியூட்டன் விதிகள் என்பது இயற்பியல் எளிய மற்றும் அடிப்படையான விதிகளின் தொகுப்பாகும். நியூட்டன் தனது இரண்டாவது சட்டத்தில் பட்சமாக குறிக்கிறார்:
> 
> **"ஒரு மாறுபாடு (force) க்குபொருந்திய மாசு (mass) உடைய பொருளுக்கு எதிர்வினையாக உருவான மாறுபாடு அதன் மாசின் அளவையும் அதன் அடிப்படை மாற்றத்தையும் (acceleration) நேரான முறையில் சொல்வதாகும்."**
> 
> இதில், மாறுபாடு (force) என்னும் அளவைக் கருதுவோம், இது சக்தியின் அலகாகும். இதனால், பொருளின் மாசு (mass) $m$ மற்றும் அதற்கான அடிப்படை மாற்றம் (acceleration) $a$ என்பவற்றுடன் கூடிய உறவைக் காணலாம். 
> 
> நியூட்டன் இரண்டாவது சட்டத்தின…

### c4_newton2 — mean 2.71
Settings: `{"student_mode": true, "content_length": "summary", "difficulty": "medium", "personality": "mentor", "examples": false, "explain_3ways": true}` · context: `{"student_mode": true, "grade": 5, "board": "CBSE", "subject": null, "language": "hi"}`
Judge: Law is stated wrongly: force direction is said to be opposite, and 'उसकी परिमाण' is unclear. 'गाड़ी को मथते हैं' is odd wording, 'वेलोसिटी' is an English term, and the 3 ways are shallow. It ends by asking the student to explain, which is a weak mentor move. F=ma is beyond Grade 5 scope, with no simplification.

> न्यूटन का दूसरा नियम गति के बारे में सबसे महत्वपूर्ण नियमों में से एक है। इसे इस प्रकार परिभाषित किया जा सकता है: किसी वस्तु की गति में परिवर्तन की दर उसकी परिमाण के सीधे आनुपातिक होती है, और यह उस पर लग रहे बल के विपरीत दिशा में होती है। इसे गणितीय रूप में इस प्रकार व्यक्त किया जाता है:
> 
> $$ F = m \cdot a $$ 
> 
> जहाँ $F$ बल है, $m$ वस्तु का द्रव्यमान है, और $a$ त्वरण है। 
> 
> यानी, जब आप किसी वस्तु पर अधिक बल लगाते हैं, तो उसका त्वरण (वेलोसिटी में परिवर्तन) भी अधिक होगा। उदाहरण के लिए, अगर आप एक गाड़ी को मथते हैं, तो उसके गति में वृद्धि होगी। 
> 
> ---
> 
> **Analogy:** यह एक खेल जैसा है, जब आप गेंद पर ज्य…

### c7_quadratic — mean 2.71
Settings: `{"student_mode": true, "content_length": "detailed", "difficulty": "hard", "personality": "coach", "examples": true, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 8, "board": "ICSE", "subject": "Science", "language": "ta"}`
Judge: The math is correct, but the Tamil is garbled: Russian word 'коэффициент', nonsensical phrases like 'பருத்திருப்பதற்கு' and 'இரண்டு நிலவுகள்'. Step 2 is confusing, since it says sum instead of product and sum. It is too easy for 'hard' and 'detailed', has no worked example beyond the one problem, and the coach tone is weak. The challenge question is a nice touch. The scope is Class 10, above grade 8, though that is acceptable for hard difficulty.

> இந்த சமன்பாட்டை $x^2 - 5x + 6 = 0$ என்னும் உருவத்தில் கொடுக்கப்பட்டுள்ளது. இதனை பருத்திருப்பதற்கு, நாம் பின்னணி முறையை (factoring) பயன்படுத்தலாம்.
> 
> 1. சமன்பாட்டில் $x^2$ க்கான коэффициентை (coefficient) தேவைப்படுகிறது.
> 2. $6$ யின் கூட்டல் அல்லது கூட்டல் உள்ளது, இது $-5$ யை பெற்றுக்கொள்ள $-2$ மற்றும் $-3$ யின் கூட்டல் மூலம் கிடைக்கும்.
> 
> இதன் மூலம், சமன்பாட்டை அடுத்தவாறு எழுதலாம்:
> 
> $$(x - 2)(x - 3) = 0$$ 
> 
> இப்போது, இதனை பூஜ்யமாகச் செய்யும்போது, இரண்டு நிலவுகள் $x - 2 = 0$ அல்லது $x - 3 = 0$ கிடைக்கும்.
> 
> இதனால், $x$ கான மதிப்புகள்:
> 
> 1. $x - 2 = 0 \Rightarrow x = 2$
> 2. $x - 3 = 0 \Rightarrow x = 3…

### p_d1_rag_g5 — mean 2.71
Settings: `{"student_mode": true, "content_length": "summary", "difficulty": "medium", "personality": "mentor", "examples": false, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 5, "board": "CBSE", "subject": null, "language": "en"}`
Judge: Pitched far above Grade 5: thylakoid, stroma, Rubisco, 3-PGA, G3P, ATP/NADPH stoichiometry. Experiment A (discs floating under LED after 7 minutes) is omitted. \ce{} is misused for '3-PGA' and for CO2 outside a chemistry context. Mentor tone is weak, and the closing question is vague.

> Photosynthesis has two main stages: the light-dependent reactions and the light-independent reactions (Calvin-Benson cycle). 
> 
> 1. **Light-Dependent Reactions**: These occur in the thylakoid membranes of chloroplasts. They use sunlight to produce ATP and NADPH, which are energy carriers. 
> 
> 2. **Light-Independent Reactions (Calvin-Benson Cycle)**: This stage takes place in the stroma. It fixes carbon dioxide ($\ce{CO2}$) using an enzyme called Rubisco, producing 3-phosphoglycerate ($\ce{3-PGA}$). Then, ATP and NADPH from the first stage convert $\ce{3-PGA}$ into glyceraldehyde-3-phosphate ($\ce{…

### c1_quadratic — mean 2.86
Settings: `{"student_mode": true, "content_length": "small", "difficulty": "easy", "personality": "mentor", "examples": false, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 2, "board": "State Board", "subject": null, "language": "en"}`
Judge: Math is correct, but quadratic factorisation is Class 9-10 content, far above Grade 2. The answer gives no scaffolding or simplification for a young learner and uses terms like 'quadratic' and 'factors'. Tone is flat, not a warm mentor voice. Length is acceptable and LaTeX is valid.

> This is a math problem about solving a quadratic equation. The equation is $x^2 - 5x + 6 = 0$.
> 
> You can find the solution by factoring the equation. The factors of $6$ that add up to $-5$ are $-2$ and $-3$. So, we can write it as:
> 
> $$(x - 2)(x - 3) = 0$$
> 
> Now, set each factor to zero:
> 
> 1. $x - 2 = 0 \Rightarrow x = 2$
> 2. $x - 3 = 0 \Rightarrow x = 3$
> 
> So, the solutions are $x = 2$ and $x = 3$.

### c12_newton2 — mean 2.86
Settings: `{"student_mode": true, "content_length": "detailed", "difficulty": "hard", "personality": "mentor", "examples": false, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 10, "board": "CBSE", "subject": null, "language": "en"}`
Judge: Too basic for 'hard' and 'detailed'. It omits momentum form (NCERT: F proportional to rate of change of momentum), derivation, vector nature, and no challenging problems. It uses \( \) and \[ \] instead of $ delimiters. It includes an example despite examples being off. Mentor tone is weak.

> Newton's second law of motion states that the acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. This can be expressed with the formula:
> 
> \[
> F = m \cdot a
> \]
> 
> where:
> - \( F \) is the net force acting on the object (measured in Newtons, \(\mathrm{N}\)),
> - \( m \) is the mass of the object (measured in kilograms, \(\mathrm{kg}\)),
> - \( a \) is the acceleration of the object (measured in meters per second squared, \(\mathrm{m/s^2}\)).
> 
> ### Explanation
> 
> This law indicates that the greater the force applied to an object, the grea…

### c13_photosynthesis — mean 3.00
Settings: `{"student_mode": true, "content_length": "deep-dive", "difficulty": "expert", "personality": "professor", "examples": false, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 11, "board": "CBSE", "subject": "Mathematics", "language": "en"}`
Judge: Question is biology, not Grade 11 Mathematics; the answer doesn't address the mismatch. Content is accurate but only moderately deep for 'expert deep-dive'. It includes an unrequested daily-life example (examples off). Tone is generic, not professorial. \ce used inside $...$ may not render.

> **Defining Photosynthesis**
> 
> Photosynthesis is the biochemical process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. It primarily occurs in the chloroplasts of plant cells, involving the absorption of sunlight, carbon dioxide, and water.
> 
> **Explanation of the Process**
> 
> Photosynthesis involves two main stages: the light-dependent reactions and the light-independent reactions (Calvin cycle).
> 
> 1. **Light-dependent Reactions**: 
>    - These reactions occur in the thylakoid membranes of the chloroplasts. 
>    - When light is absorbed by …

### c4_photosynthesis — mean 3.14
Settings: `{"student_mode": true, "content_length": "summary", "difficulty": "medium", "personality": "mentor", "examples": false, "explain_3ways": true}` · context: `{"student_mode": true, "grade": 5, "board": "CBSE", "subject": null, "language": "hi"}`
Judge: Chemical formulas in plain Unicode rather than LaTeX; glucose formula too advanced for Grade 5. Uses transliterated 'फोटोसिंथेसिस' instead of 'प्रकाश संश्लेषण'. The 'उदाहरण' inside the main text conflicts with examples being off. Mentor tone is mild, ending question is fine. Three sections are present but thin.

> फोटोसिंथेसिस वह प्रक्रिया है जिसमें पौधे सूर्य के प्रकाश, जल और कार्बन डाइऑक्साइड का उपयोग करके भोजन तैयार करते हैं। इसमें पौधे अपने पत्तों में मौजूद क्लोरोफिल की मदद से सूरज की रोशनी को अवशोषित करते हैं।
> 
> इस प्रक्रिया के दौरान, पौधे जल (H₂O) और कार्बन डाइऑक्साइड (CO₂) को मिलाकर ग्लूकोज (C₆H₁₂O₆) और ऑक्सीजन (O₂) का निर्माण करते हैं। यह ऑक्सीजन हमारे लिए बहुत महत्वपूर्ण है क्योंकि हम इसे सांस लेने के लिए उपयोग करते हैं। उदाहरण के लिए, जब आप एक पेड़ के नीचे बैठते हैं, तो वह सूर्य की रोशनी का उपयोग करते हुए आपके लिए ऑक्सीजन का उत्पादन कर रहा होता है।
> 
> ---
> 
> **Analogy:** जैसे एक रसोइया सामग्री लेकर…

## Probes

| Probe | Expectation | Verdict | Detail |
|---|---|---|---|
| p_a1_greeting_en | 1–2 sentences despite deep-dive + expert | **PASS** | 2 sentence(s), 10 words |
| p_a2_greeting_ta | 1–2 sentences despite deep-dive + expert | **PASS** | 2 sentence(s), 6 words |
| p_b_off_subject | correct but brief | **PASS** | 64 words; judge kept_brief=True |
| p_c_g2_derivative | answers simply, no refusal, no grade mention | **PASS** | refused=False, age_appropriate=True, grade score 3 |
| p_d1_rag_g5 | uses BOTH sources (G2) at the learner's level | **FAIL** | G2 FAIL (planted facts — vault [], attached []); learner level (judge): False |
| p_d2_rag_g11 | uses BOTH sources (G2) at the learner's level | **FAIL** | G2 FAIL (planted facts — vault [], attached ['30 minutes', 'dark beaker']); learner level (judge): False |

## Before / after — vs `baseline`

Baseline generator mix: openai:gpt-4o-mini × 26; judge `claude-sonnet-5-5`.

| Dimension | Before | After | Δ |
|---|---|---|---|
| grade_appropriateness | 3.04 | 3.15 | +0.11 |
| board_alignment | 2.59 | 2.82 | +0.23 |
| difficulty_fit | 2.44 | 2.73 | +0.29 |
| language_purity | 4.27 | 4.23 | -0.04 |
| latex_validity | 3 | 3.5 | +0.50 |
| personality_fit | 2.65 | 2.58 | -0.07 |
| length_compliance (auto) | 2.65 | 3.5 | +0.85 |

| Probe | Before | After | After detail |
|---|---|---|---|
| p_a1_greeting_en | FAIL | **PASS** | 2 sentence(s), 10 words |
| p_a2_greeting_ta | FAIL | **PASS** | 2 sentence(s), 6 words |
| p_b_off_subject | PASS | **PASS** | 64 words; judge kept_brief=True |
| p_c_g2_derivative | FAIL | **PASS** | refused=False, age_appropriate=True, grade score 3 |
| p_d1_rag_g5 | FAIL | **FAIL** | G2 FAIL (planted facts — vault [], attached []); learner level (judge): False |
| p_d2_rag_g11 | FAIL | **FAIL** | G2 FAIL (planted facts — vault [], attached ['30 minutes', 'dark beaker']); learner level (judge): False |

| Case | Before mean | After mean | Before words | After words | Length target |
|---|---|---|---|---|---|
| c1_photosynthesis | 3.33 | 4.33 | 63 | 25 | [5, 30] |
| c1_quadratic | 2.43 | 2.86 | 77 | 60 | [5, 30] |
| c1_newton2 | 3 | 3.5 | 74 | 27 | [5, 30] |
| c4_photosynthesis | 2.67 | 3.14 | 261 | 175 | [99, 154] |
| c4_quadratic | 2.29 | 3.14 | 250 | 162 | [99, 154] |
| c4_newton2 | 2.57 | 2.71 | 278 | 183 | [99, 154] |
| c4_greeting | 4.4 | 4.4 | 12 | 11 | <= 2 sentences |
| c7_photosynthesis | 1.33 | 2 | 113 | 123 | [210, 315] |
| c7_quadratic | 2.14 | 2.71 | 88 | 78 | [210, 315] |
| c7_newton2 | 2.14 | 2.57 | 120 | 147 | [210, 315] |
| c7_greeting | 4.4 | 3.6 | 11 | 11 | <= 2 sentences |
| c12_photosynthesis | 3.71 | 3.29 | 327 | 281 | [380, 550] |
| c12_quadratic | 3.14 | 3.29 | 141 | 155 | [380, 550] |
| c12_newton2 | 3.43 | 2.86 | 320 | 254 | [380, 550] |
| c13_photosynthesis | 2.86 | 3 | 324 | 296 | [700, 1100] |
| c13_quadratic | 2.43 | 2.43 | 162 | 182 | [700, 1100] |
| c13_newton2 | 2.71 | 3.29 | 351 | 388 | [700, 1100] |
| c18_photosynthesis | 3.33 | 3.83 | 189 | 192 | [120, 200] |
| c18_quadratic | 3.33 | 3.33 | 155 | 116 | [120, 200] |
| c18_newton2 | 3.17 | 3.33 | 209 | 177 | [120, 200] |
| p_a1_greeting_en | 4 | 4.6 | 13 | 10 | <= 2 sentences |
| p_a2_greeting_ta | 3.2 | 3.6 | 11 | 6 | <= 2 sentences |
| p_b_off_subject | 4.2 | 4.2 | 114 | 64 | [0, 180] |
| p_c_g2_derivative | 2.83 | 3.6 | 151 | 96 | [50, 90] |
| p_d1_rag_g5 | 2.71 | 2.71 | 280 | 142 | [90, 140] |
| p_d2_rag_g11 | 3 | 3.14 | 280 | 432 | [380, 550] |

## Language calibration (amendment 4)

Actual words ÷ midpoint of the (multiplier-adjusted) target, per language. Values near 1.0 mean the multiplier is right; a consistent offset across languages means the prompt, not the multiplier, is off.

| Language | n | mean ratio |
|---|---|---|
| en | 12 | 0.99 |
| hi | 3 | 1.37 |
| ta | 3 | 0.44 |

## Measured output tokens per word (amendment 5)

| Language | n | mean | max |
|---|---|---|---|
| en | 16 | 1.73 | 2.86 |
| hi | 3 | 1.92 | 2.36 |
| ta | 3 | 3.63 | 3.86 |
