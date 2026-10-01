# Tailoring eval — baseline

Run at 2026-10-01T12:22:59.664230+00:00 · generator routing: primary `gemini-3.5-flash`, documents `claude-sonnet-4-20250514`, fallback `gpt-4o-mini` · forced generator: none (default routing) · judge `claude-sonnet-5-5` · guard LLM classifier: True
Cases: 30 (26 LLM generations, 4 router canned replies). Combos 1, 4, 7, 12, 13, 18; probes a1_greeting_en, a2_greeting_ta, b_off_subject, c_g2_derivative, d1_rag_g5, d2_rag_g11.

**Provider mix (who actually answered):** openai:gpt-4o-mini × 26

## Per-dimension averages (1–5; LLM generations only)

| Dimension | Avg | Target |
|---|---|---|
| grade_appropriateness | 3.04 | ≥ 4.0 |
| board_alignment | 2.59 | ≥ 4.0 |
| difficulty_fit | 2.44 | ≥ 4.0 |
| language_purity | 4.27 | ≥ 4.0 |
| latex_validity | 3 | ≥ 4.0 |
| personality_fit | 2.65 | ≥ 4.0 |
| length_compliance (auto) | 2.65 | ≥ 4.0 |

Automatic script purity (share of letters in the target script): mean 0.97, min 0.806.

## Cost & tokens

| Role | Model | Calls | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|---|
| generation | gpt-4o-mini | 26 | 58,534 | 8,110 | 0.0136 |
| judge | claude-sonnet-5-5 | 26 | 40,701 | 5,343 | price unknown |

**Total for priced models: $0.0136** — plus claude-sonnet-5-5 (price unknown; add it to `MODEL_PRICES_USD_PER_MTOK`).
Excludes content-guard classifier calls (separate client). Prices come from `MODEL_PRICES_USD_PER_MTOK`.

## Truncation / finish_reason

No truncation: every generation finished normally.

| Case | Provider | Model | Output tokens | finish_reason |
|---|---|---|---|---|
| c1_photosynthesis | openai | gpt-4o-mini | 82 | stop |
| c1_quadratic | openai | gpt-4o-mini | 182 | stop |
| c1_newton2 | openai | gpt-4o-mini | 103 | stop |
| c4_photosynthesis | openai | gpt-4o-mini | 510 | stop |
| c4_quadratic | openai | gpt-4o-mini | 497 | stop |
| c4_newton2 | openai | gpt-4o-mini | 551 | stop |
| c4_greeting | openai | gpt-4o-mini | 23 | stop |
| c7_photosynthesis | openai | gpt-4o-mini | 348 | stop |
| c7_quadratic | openai | gpt-4o-mini | 349 | stop |
| c7_newton2 | openai | gpt-4o-mini | 308 | stop |
| c7_greeting | openai | gpt-4o-mini | 29 | stop |
| c12_photosynthesis | openai | gpt-4o-mini | 511 | stop |
| c12_quadratic | openai | gpt-4o-mini | 303 | stop |
| c12_newton2 | openai | gpt-4o-mini | 485 | stop |
| c13_photosynthesis | openai | gpt-4o-mini | 541 | stop |
| c13_quadratic | openai | gpt-4o-mini | 391 | stop |
| c13_newton2 | openai | gpt-4o-mini | 563 | stop |
| c18_photosynthesis | openai | gpt-4o-mini | 312 | stop |
| c18_quadratic | openai | gpt-4o-mini | 340 | stop |
| c18_newton2 | openai | gpt-4o-mini | 382 | stop |
| p_a1_greeting_en | openai | gpt-4o-mini | 18 | stop |
| p_a2_greeting_ta | openai | gpt-4o-mini | 21 | stop |
| p_b_off_subject | openai | gpt-4o-mini | 155 | stop |
| p_c_g2_derivative | openai | gpt-4o-mini | 231 | stop |
| p_d1_rag_g5 | openai | gpt-4o-mini | 461 | stop |
| p_d2_rag_g11 | openai | gpt-4o-mini | 414 | stop |

## Per-case scores

| Case | Lang | Len target | Words | Len | Grade | Board | Diff | Lang | LaTeX | Pers | KaTeX err | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c1_photosynthesis | en | [5, 30] | 63 | 2 | 3 | 3 | 4 | 5 | None | 3 | – | The language is simple and the length is short. However, the 'green hat' analogy is misleading, because photosynthesis happens in the leaves |
| c1_quadratic | en | [5, 30] | 77 | 1 | 2 | 2 | 2 | 5 | 2 | 3 | – | Quadratic factorisation is Class 10 content, far above Grade 2, and the answer makes no attempt to simplify or scaffold. It uses \( \) delim |
| c1_newton2 | en | [5, 30] | 74 | 2 | 3 | 3 | 3 | 5 | 2 | 3 | – | Toy car analogy suits Grade 2, but the formula F = m × a is not in LaTeX, which breaks the math rule. Also a mild inaccuracy: force gives ac |
| c1_greeting | en | <= 2 sentences | 9 | 3 | – | – | – | – | – | – | – | router canned reply (no LLM) |
| c4_photosynthesis | hi | [99, 154] | 261 | 2 | 3 | 3 | 3 | 3 | None | 2 | – | Too long for a summary; adds a separate 'उदाहरण' section although examples are off. Technical section wrongly calls photosynthesis 'एरोबिक'  |
| c4_quadratic | hi | [99, 154] | 250 | 2 | 2 | 2 | 3 | 3 | 2 | 2 | – | Quadratic factorisation is Class 10 content, too advanced for Grade 5, and no scaffolding is given. The answer uses \[ \] and \( \) instead  |
| c4_newton2 | hi | [99, 154] | 278 | 2 | 2 | 2 | 2 | 3 | 5 | 2 | 0 | F=ma is a Class 9 topic, and the answer gives no scaffolding for Grade 5. Examples are off, yet a worked example is included, and the answer |
| c4_greeting | hi | <= 2 sentences | 12 | 3 | 5 | None | 5 | 5 | None | 4 | – | Short, correct 2-sentence Hindi greeting, suitable for the situation. It correctly skips the 3-ways sections. Mentor tone is mild but presen |
| c7_photosynthesis | ta | [210, 315] | 113 | 2 | 1 | 1 | 1 | 1 | None | 2 | – | Largely incoherent Tamil with factual errors: 'கேரோபிளாஸ்ட்ஸ்' for chloroplasts, nonsense 'three main aspects', 'உலோக தொகுப்புகள்', 'நிலநடுக |
| c7_quadratic | ta | [210, 315] | 88 | 2 | 2 | 2 | 2 | 2 | 3 | 2 | 0 | Final answer x=2,3 is correct, but the explanation says the numbers 'add to 6, subtract to -5', which is wrong. It also has the garbled head |
| c7_newton2 | ta | [210, 315] | 120 | 2 | 2 | 2 | 1 | 1 | 5 | 2 | 0 | Conceptual errors: 'force' translated as ஆற்றல் (energy), 'mass' as 'மாசு' (pollution), acceleration as 'முறைமை வேகம்'. Stray Russian text,  |
| c7_greeting | ta | <= 2 sentences | 11 | 3 | 5 | None | 5 | 5 | None | 4 | – | Short two-sentence Tamil greeting, appropriate for small talk. It doesn't mention grade or board and has no emojis. Coach persona is mild; i |
| c12_photosynthesis | en | [380, 550] | 327 | 4 | 4 | 3 | 3 | 5 | 4 | 3 | 0 | Accurate and detailed, but "hard" difficulty is underserved; it stays at a textbook-summary level. It includes an Example section though exa |
| c12_quadratic | en | [380, 550] | 141 | 1 | 4 | 4 | 2 | 5 | 3 | 3 | – | Math is correct and fits Class 10 factorisation. But difficulty 'hard' and 'detailed' are not met: the solution is basic, with no splitting- |
| c12_newton2 | en | [380, 550] | 320 | 4 | 3 | 3 | 2 | 5 | 3 | 4 | 0 | Content is basic and misses CBSE's momentum form (F = Δp/Δt) and its derivation, so it is too easy for 'hard' and 'detailed'. It gives an ex |
| c12_greeting | en | <= 2 sentences | 9 | 3 | – | – | – | – | – | – | – | router canned reply (no LLM) |
| c13_photosynthesis | en | [700, 1100] | 324 | 2 | 3 | 3 | 2 | 5 | 3 | 2 | 0 | Question is biology, not Grade 11 Mathematics; answer doesn't acknowledge the mismatch. Content is accurate but intro-level, not expert or d |
| c13_quadratic | en | [700, 1100] | 162 | 1 | 3 | 3 | 1 | 5 | 2 | 2 | – | Math is correct, but the answer is basic and not an expert deep-dive: no discriminant, quadratic formula, completing the square, Vieta's rel |
| c13_newton2 | en | [700, 1100] | 351 | 2 | 3 | 3 | 1 | 5 | 3 | 2 | 0 | Content is basic, Class 9 level, not expert deep-dive for Grade 11. It omits F = dp/dt, vector form, momentum, inertial frames, and calculus |
| c13_greeting | en | <= 2 sentences | 9 | 5 | – | – | – | – | – | – | – | router canned reply (no LLM) |
| c18_photosynthesis | en | [120, 200] | 189 | 5 | 3 | None | 2 | 5 | 3 | 2 | 0 | Content is accurate but only textbook-level; 'hard' absolute difficulty and technical-expert tone call for depth (photosystems II/I, Z-schem |
| c18_quadratic | en | [120, 200] | 155 | 5 | 4 | None | 2 | 5 | 2 | 2 | – | Math is correct, but uses \( \) and \[ \] instead of the required $...$/$$...$$. Difficulty 'hard' is ignored: only basic factoring, no disc |
| c18_newton2 | en | [120, 200] | 209 | 4 | 3 | None | 2 | 5 | 3 | 2 | 0 | Content is correct but elementary for 'hard' absolute difficulty: no momentum form F=dp/dt, vector nature, inertial frames, or variable mass |
| c18_greeting | en | <= 2 sentences | 9 | 5 | – | – | – | – | – | – | – | router canned reply (no LLM) |
| p_a1_greeting_en | en | <= 2 sentences | 13 | 3 | 5 | None | None | 5 | None | 3 | – | Length is correct for a greeting (two short sentences), and it doesn't mention grade or board. However, 'Hey! ... study buddy' is casual and |
| p_a2_greeting_ta | ta | <= 2 sentences | 11 | 3 | 4 | None | 4 | 3 | None | 2 | – | Length is correct (two short sentences). Tamil has errors: 'பயில நண்பர்' is awkward and ungrammatical, and 'எந்த பாடத்தில் உதவி தேவை' is clu |
| p_b_off_subject | en | [0, 180] | 114 | 5 | 4 | None | 4 | 5 | None | 3 | – | Correct answer (Valmiki, Adi Kavi, seven Kandas) in appropriate language, with no forced link to maths. It is a bit longer than a brief answ |
| p_c_g2_derivative | en | [50, 90] | 151 | 2 | 2 | None | 2 | 5 | 2 | 4 | – | Bicycle analogy is good, but it uses abstract f(x), f'(x), x^2 and 'rate of change', which is too advanced for a Grade 2 learner. LaTeX uses |
| p_d1_rag_g5 | en | [90, 140] | 280 | 2 | 2 | 2 | 1 | 5 | 4 | 3 | 0 | Pitched far above Grade 5: RuBP, 3-PGA, G3P, Rubisco, thylakoid, ATP/NADPH stoichiometry. Experiment A's 7-minute result was not stated and  |
| p_d2_rag_g11 | en | [380, 550] | 280 | 3 | 2 | 3 | 2 | 5 | None | 3 | – | Omits experiment A (discs floating after 7 minutes under LED), so the vault experiment is missing. Source-level jargon (3-PGA, Rubisco, 3 AT |

## Lowest-scoring cases

### c7_photosynthesis — mean 1.33
Settings: `{"student_mode": true, "content_length": "detailed", "difficulty": "hard", "personality": "coach", "examples": true, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 8, "board": "ICSE", "subject": "Science", "language": "ta"}`
Judge: Largely incoherent Tamil with factual errors: 'கேரோபிளாஸ்ட்ஸ்' for chloroplasts, nonsense 'three main aspects', 'உலோக தொகுப்புகள்', 'நிலநடுக்கம்'. No word equation, no chlorophyll or glucose, no examples. Not detailed or hard; coach tone weak. No grade or board mention.

> போட்டோசின்தசின் என்பது செடியின் முன்னணி உணர்ச்சி செயல்முறை. இது எளிதில் விளக்கமாகக் கூறும்போது, செடிகள் சூரிய ஒளியைக் கொண்டு, காற்றில் உள்ள கார்பன் டி ஆக்சைடு மற்றும் நீரை ஒன்றிணைத்து, சாகுபடி உணவுகளை உருவாக்கும் செயல் ஆகும். 
> 
> இந்த செயல்முறை மூன்று முக்கிய அம்சங்களை கொண்டுள்ளது: ஒளி செல் சிதைவுப் மீண்டும், கேரோபிளாஸ்ட்ஸ் (தரம் அமைப்பிலுள்ள இடங்கள்) மற்றும் சுருக்கமான நிறங்களை உருவாக்குவதற்கான செயல்முறைகளை உள்ளடக்கியது. 
> 
> செடியின் வாழ்வில் இந்த செயல்முறை முக்கியமானது; உலோக தொகுப்புகளை உருவாக்குவதற்கு உதவும், மேலும், இது விலங்கு மற்றும் மனிதனான உயிரினங்களுக்கு ஆக்ஸிஜனை வழங்குகிறது. இது வனம் மற்…

### c7_quadratic — mean 2.14
Settings: `{"student_mode": true, "content_length": "detailed", "difficulty": "hard", "personality": "coach", "examples": true, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 8, "board": "ICSE", "subject": "Science", "language": "ta"}`
Judge: Final answer x=2,3 is correct, but the explanation says the numbers 'add to 6, subtract to -5', which is wrong. It also has the garbled heading 'செயல்நிறுத்தம் சமூக இழுவைகள்' and an irrelevant 'Singapore chemistry method' remark. It is too easy for 'hard', has no examples, uses unicode x² inside LaTeX, and the coach tone is weak.

> இந்த சமன்பாட்டை ($x² - 5x + 6 = 0$) நாங்கள் சீர்குலைப்பு முறை (factoring method) கொண்டு தீர்ப்போம்.
> 
> செயல்நிறுத்தம் சமூக இழுவைகள்:
> 
> 1. $x² - 5x + 6$ என்பதில், இரண்டு எண்கள் தேவைப்படுகின்றன, இவை கூட்டுவது 6 ஆகவும், கழித்தால் -5 ஆகவும் இருக்க வேண்டும். 
> 2. இதற்கான எண்கள் 2 மற்றும் 3 ஆகும், ஏனெனில் $2 \cdot 3 = 6$ மற்றும் $2 + 3 = 5$.
> 
> இதனால், $x² - 5x + 6$ என்பதைக் கிட்டத்தட்ட இவ்வாறு எழுதலாம்:
> 
> $$(x - 2)(x - 3) = 0.$$
> 
> இப்போது, இதனை 0 ஆக சமமாகக் கொள்வதால், நமக்கு இரண்டு தீர்வுகள் கிடைக்கும்:
> 
> 1. $x - 2 = 0 \Rightarrow x = 2$,
> 2. $x - 3 = 0 \Rightarrow x = 3$.
> 
> முடிவு: $x = 2$ மற்றும் $x = 3$ ஆக…

### c7_newton2 — mean 2.14
Settings: `{"student_mode": true, "content_length": "detailed", "difficulty": "hard", "personality": "coach", "examples": true, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 8, "board": "ICSE", "subject": "Science", "language": "ta"}`
Judge: Conceptual errors: 'force' translated as ஆற்றல் (energy), 'mass' as 'மாசு' (pollution), acceleration as 'முறைமை வேகம்'. Stray Russian text, garbled sentences, no worked numeric example, momentum definition missing. Too shallow for hard/detailed. Coach tone weak; closing question is nonsensical.

> நியூட்டனின் இரண்டாவது சட்டம் என்பது, பொருளின் ஆற்றல் (force) மற்றும் அதன் மாசு (mass) இன் தொடர்புகளை விளக்குகிறது. இது மிகவும் முக்கியமான ஒழுங்குகளை வழங்குகிறது, குறிப்பாக $F = ma$ என்ற சமன்பாட்டை பயன்படுத்துகிறது, இதில் $F$ என்பது ஆற்றல், $m$ என்பது மாசு, மற்றும் $a$ என்பது முறைமை வேகம் (acceleration).
> 
> இந்த சட்டத்தின் படி, ஒரு பொருளுக்கு வெளிப்புற ஆற்றல் (external force) ஏற்படும்போது அது அதற்கேற்ப வேகம் மாறும். மேலும், பொருளின் மாசு மற்றும் அதில் உள்ள ஆற்றல் அளவை அதிகரிப்பது, அதன் மாறுதலை குறைக்கும். எடுத்துக்காட்டாக, நீங்கள் ஒரு சீரான மாசு (из весом) கொண்ட கார் முன் push செய்யும் போது, நீங்…

### c4_quadratic — mean 2.29
Settings: `{"student_mode": true, "content_length": "summary", "difficulty": "medium", "personality": "mentor", "examples": false, "explain_3ways": true}` · context: `{"student_mode": true, "grade": 5, "board": "CBSE", "subject": null, "language": "hi"}`
Judge: Quadratic factorisation is Class 10 content, too advanced for Grade 5, and no scaffolding is given. The answer uses \[ \] and \( \) instead of $ delimiters. The real-world example is mathematically wrong (6 split into 2 and 3 is not the roots). The analogy is vague. Tone is flat, not mentor-like. Length is more than a summary. 'अंश' is used wrongly for 'संख्या'.

> इस समीकरण को हल करने के लिए हम इसे फैक्टर कर सकते हैं। समीकरण है:
> 
> \[ x^2 - 5x + 6 = 0 \]
> 
> 1. **समीकरण को फैक्टर करें:** हमें ऐसे दो अंशों को खोजना है जिनका गुणनफल 6 और योगफल -5 हो। ये अंश हैं -2 और -3।
> 
>    इस प्रकार, समीकरण को फैक्टर किया जा सकता है:
>    \[ (x - 2)(x - 3) = 0 \]
> 
> 2. **फैक्टर को शून्य पर सेट करें:** अब, हम प्रत्येक फैक्टर को शून्य पर सेट करते हैं:
>    \[ x - 2 = 0 \quad \text{या} \quad x - 3 = 0 \]
> 
> 3. **इसे हल करें:**
>    - पहले फैक्टर से: \( x - 2 = 0 \Rightarrow x = 2 \)
>    - दूसरे फैक्टर से: \( x - 3 = 0 \Rightarrow x = 3 \)
> 
> इस प्रकार, समीकरण के हल हैं:
> \[ x = 2 \quad \text{…

### c1_quadratic — mean 2.43
Settings: `{"student_mode": true, "content_length": "small", "difficulty": "easy", "personality": "mentor", "examples": false, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 2, "board": "State Board", "subject": null, "language": "en"}`
Judge: Quadratic factorisation is Class 10 content, far above Grade 2, and the answer makes no attempt to simplify or scaffold. It uses \( \) delimiters instead of the required $...$. The math is correct. The tone is neutral rather than mentor-like, though the closing check question helps a little. The length is acceptable.

> To solve the equation \(x^2 - 5x + 6 = 0\), we can factor it. 
> 
> 1. **Write the equation**: \(x^2 - 5x + 6\).
> 2. **Find two numbers** that multiply to \(6\) and add up to \(-5\). The numbers are \(-2\) and \(-3\).
> 3. **Factor the equation**: \((x - 2)(x - 3) = 0\).
> 4. **Set each factor to zero**:
>    - \(x - 2 = 0 \Rightarrow x = 2\)
>    - \(x - 3 = 0 \Rightarrow x = 3\)
> 
> So, the solutions are \(x = 2\) and \(x = 3\).
> 
> Does that make sense?

### c13_quadratic — mean 2.43
Settings: `{"student_mode": true, "content_length": "deep-dive", "difficulty": "expert", "personality": "professor", "examples": false, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 11, "board": "CBSE", "subject": "Mathematics", "language": "en"}`
Judge: Math is correct, but the answer is basic and not an expert deep-dive: no discriminant, quadratic formula, completing the square, Vieta's relations or complex roots. It uses \( \) and \[ \] delimiters instead of $...$. The tone is a friendly tutor, not a professor.

> To solve the quadratic equation \( x^2 - 5x + 6 = 0 \), we can factor it. We are looking for two numbers that multiply to \( 6 \) (the constant term) and add to \( -5 \) (the coefficient of \( x \)).
> 
> 1. **Identify the factors**:
>    The numbers \( -2 \) and \( -3 \) satisfy both conditions:
>    - \( -2 \times -3 = 6 \)
>    - \( -2 + (-3) = -5 \)
> 
> 2. **Factor the equation**:
>    We can factor the quadratic as:
>    \[
>    (x - 2)(x - 3) = 0
>    \]
> 
> 3. **Set each factor to zero**:
>    To find the values of \( x \), we set each factor equal to zero:
>    - \( x - 2 = 0 \)  →  \( x = 2 \)
>    - \( x - 3 = 0 …

### c4_newton2 — mean 2.57
Settings: `{"student_mode": true, "content_length": "summary", "difficulty": "medium", "personality": "mentor", "examples": false, "explain_3ways": true}` · context: `{"student_mode": true, "grade": 5, "board": "CBSE", "subject": null, "language": "hi"}`
Judge: F=ma is a Class 9 topic, and the answer gives no scaffolding for Grade 5. Examples are off, yet a worked example is included, and the answer is not a summary. The analogy is weak and confuses mass with weight. The Hindi mixes in English words and has typos (म/s², न्युटन). The tone is not mentor-like.

> न्यूटन का द्वितीय नियम हमें यह बताता है कि किसी वस्तु पर लगने वाला बल (force) और उसकी स्थिति में बदलाव (acceleration) के बीच संबंध क्या होता है। इसे निम्नलिखित बिंदुओं में समझा जा सकता है:
> 
> 1. **बल (Force)**: बल एक ऐसा प्रभाव है जो किसी वस्तु की गति को बदलता है। इसे न्यूटन (N) में मापा जाता है।
>    
> 2. **त्वरण (Acceleration)**: यह वह दर है जिस पर किसी वस्तु की गति (velocity) बदलती है। इसका माप मीटर प्रति सेकंड स्क्वायर (m/s²) में किया जाता है।
> 
> 3. **न्युटन का द्वितीय नियम**: यह नियम कहता है कि किसी वस्तु के ऊपर लगने वाला कुल बल उसकी द्रव्यमान (mass) और उसके त्वरण का गुणनफल होता है। इसे निम्नलिख…

### c4_photosynthesis — mean 2.67
Settings: `{"student_mode": true, "content_length": "summary", "difficulty": "medium", "personality": "mentor", "examples": false, "explain_3ways": true}` · context: `{"student_mode": true, "grade": 5, "board": "CBSE", "subject": null, "language": "hi"}`
Judge: Too long for a summary; adds a separate 'उदाहरण' section although examples are off. Technical section wrongly calls photosynthesis 'एरोबिक' and uses heavy terms (अकार्बनिक/कार्बनिक). Uses transliterated 'फोटोसिंथेसिस' rather than 'प्रकाश संश्लेषण'. Tone is flat, with little mentor warmth.

> **फोटोसिंथेसिस** एक जैविक प्रक्रिया है जिसके द्वारा पौधे, शैवाल और कुछ बैक्टीरिया सूर्य के प्रकाश का उपयोग करके भोजन बनाते हैं। यह प्रक्रिया मुख्य रूप से पौधों के पत्तों में होती है, जहाँ क्लोरोफिल नामक हरित पदार्थ मौजूद होता है। फोटोसिंथेसिस की प्रक्रिया को निम्नलिखित बिंदुओं में समझा जा सकता है:
> 
> 1. **प्रकाश का अवशोषण:** पौधे अपनी पत्तियों में उपस्थित क्लोरोफिल के माध्यम से सूरज की रोशनी को अवशोषित करते हैं।
> 2. **कार्बन डाइऑक्साइड का प्रयोग:** पौधे वायुमंडल से कार्बन डाइऑक्साइड लेते हैं, जो उनके पत्तों के छोटे छिद्रों से प्रवेश करती है।
> 3. **पानी का महत्व:** पौधे जड़ों के माध्यम से मिट्टी से…

### c13_newton2 — mean 2.71
Settings: `{"student_mode": true, "content_length": "deep-dive", "difficulty": "expert", "personality": "professor", "examples": false, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 11, "board": "CBSE", "subject": "Mathematics", "language": "en"}`
Judge: Content is basic, Class 9 level, not expert deep-dive for Grade 11. It omits F = dp/dt, vector form, momentum, inertial frames, and calculus. It includes a worked example despite examples being off. It uses \( \) delimiters instead of $...$. The tone is generic rather than professorial. It mentions 'board exams' but no specific board or grade.

> Newton's Second Law of Motion is a fundamental principle in physics that describes the relationship between the **force** acting on an object, its **mass**, and its **acceleration**. Here’s a structured explanation:
> 
> ### Definition
> - Newton's Second Law states that the **acceleration** of an object is directly proportional to the **net force** acting on it and inversely proportional to its **mass**. This relationship can be mathematically expressed as:
> 
> $$
> F = m \cdot a
> $$
> 
> Where:
> - \(F\) is the net force applied to the object (in newtons, N),
> - \(m\) is the mass of the object (in kilograms, k…

### p_d1_rag_g5 — mean 2.71
Settings: `{"student_mode": true, "content_length": "summary", "difficulty": "medium", "personality": "mentor", "examples": false, "explain_3ways": false}` · context: `{"student_mode": true, "grade": 5, "board": "CBSE", "subject": null, "language": "en"}`
Judge: Pitched far above Grade 5: RuBP, 3-PGA, G3P, Rubisco, thylakoid, ATP/NADPH stoichiometry. Experiment A's 7-minute result was not stated and was only 'assumed'. Experiment B's '30 minutes' is unsupported by the source. Too long for a summary, and the tone is only lightly mentor-like.

> Did you know that photosynthesis is like a two-part dance where light and dark reactions work together to produce energy for plants? Here's how it all connects based on your notes:
> 
> ### Two Stages of Photosynthesis
> 
> 1. **Light Reactions**:
>    - Occur in the **thylakoid membranes** of the chloroplasts.
>    - Capture light energy, converting it into **ATP** and **NADPH**.
>    - Water molecules $\ce{H2O}$ are split, releasing oxygen $\ce{O2}$ as a by-product.
> 
> 2. **Calvin Cycle (Light-Independent Reactions)**:
>    - Takes place in the **stroma** of the chloroplasts.
>    - Utilizes ATP and NADPH produ…

## Probes

| Probe | Expectation | Verdict | Detail |
|---|---|---|---|
| p_a1_greeting_en | 1–2 sentences despite deep-dive + expert | **FAIL** | 3 sentence(s), 13 words |
| p_a2_greeting_ta | 1–2 sentences despite deep-dive + expert | **FAIL** | 3 sentence(s), 11 words |
| p_b_off_subject | correct but brief | **PASS** | 114 words; judge kept_brief=True |
| p_c_g2_derivative | answers simply, no refusal, no grade mention | **FAIL** | refused=False, age_appropriate=False, grade score 2 |
| p_d1_rag_g5 | uses BOTH sources (G2) at the learner's level | **FAIL** | G2 FAIL (planted facts — vault [], attached ['30 minutes', 'dark beaker', 'never floated']); learner level (judge): False |
| p_d2_rag_g11 | uses BOTH sources (G2) at the learner's level | **FAIL** | G2 FAIL (planted facts — vault [], attached ['30 minutes', 'dark beaker']); learner level (judge): False |

## Language calibration (amendment 4)

Actual words ÷ midpoint of the (multiplier-adjusted) target, per language. Values near 1.0 mean the multiplier is right; a consistent offset across languages means the prompt, not the multiplier, is off.

| Language | n | mean ratio |
|---|---|---|
| en | 12 | 1.53 |
| hi | 3 | 2.08 |
| ta | 3 | 0.41 |

## Measured output tokens per word (amendment 5)

| Language | n | mean | max |
|---|---|---|---|
| en | 16 | 1.73 | 2.41 |
| hi | 3 | 1.97 | 1.99 |
| ta | 3 | 3.2 | 3.97 |
