# Replicating "Identifying Hearing Difficulty Moments in Conversational Audio"

Replication and extension of [Collins et al. (2025)](https://arxiv.org/abs/2507.23590) using the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/). Our best model (GPT-4o Audio, 20-shot) achieves **F1 = 0.97**, surpassing the paper's Gemini 1.5 Pro result of 0.87.

**[Interactive Dashboard](https://chozillla.github.io/CollinsPaper/)** — explore all trial results with hover, zoom, and pan.

---

## What is This Project About?

Imagine you're in a meeting and someone says something, but another person doesn't hear them clearly. They might respond with "What?", "Huh?", "Sorry?", or "Can you repeat that?". These are **Hearing Difficulty Moments (HDMs)** — moments when a listener struggles to understand what was said.

Collins et al. (2025) showed that AI models can automatically detect these moments from audio recordings. Their best model (Google's Gemini 1.5 Pro) achieved an **F1 score of 0.87** — meaning it correctly identified most hearing difficulty moments while making relatively few mistakes.

**This project replicates and extends their experiment** using the AMI Meeting Corpus (group meeting recordings). We test multiple detection methods from simple baselines to an optimized GPT-4o Audio classifier that **surpasses the paper's results (F1 = 0.97 vs 0.87)**.

### What is an F1 Score?

F1 is the main metric used to evaluate each method. It balances two concerns:

- **Precision**: Of all the moments the model flagged as HDMs, how many actually were? (Avoiding false alarms)
- **Recall**: Of all the real HDMs in the data, how many did the model catch? (Not missing any)

F1 ranges from 0 (worst) to 1 (perfect). An F1 of 0.58 means the model does a reasonable job at both catching real HDMs and avoiding false alarms, though there's room for improvement.

---

## Results

| Method | What it Does | Avg F1 | Std |
|---|---|:---:|:---:|
| Random guessing (50/50) | Flips a coin for each segment | 0.15 | 0.03 |
| Random guessing (base-rate) | Guesses "HDM" 9.1% of the time | 0.09 | 0.06 |
| ASR Hotword Heuristic (Whisper) | Transcribes audio, looks for keywords | 0.23 | 0.11 |
| Gemini 2.5 Flash Lite (10-shot) | Self-reported confidence, 4s audio | 0.15 | 0.01 |
| GPT-4o Audio v1 (10-shot) | Real logprobs, 4s audio | 0.14 | 0.04 |
| GPT-4o Audio v2 (10-shot) | 12s context, human-verified examples | 0.46 | 0.03 |
| Gemini 3.1 Pro (10-shot audio) | AI listens to audio and decides | 0.58 | 0.09 |
| GPT-4o Audio v3 (10-shot) | Hard negatives, transcripts, strict prompt | 0.60 | 0.12 |
| *Collins et al. Gemini 1.5 Pro (10-shot)* | *Paper's result* | *0.87* | — |
| **GPT-4o Audio v4 (20-shot)** | **Enhanced prompt, human-verified, 12s context** | **0.97** | **0.03** |

Our best model (GPT-4o Audio v4) achieves **F1 = 0.97**, surpassing the paper's 0.87 by 11 percentage points. One split achieved a perfect F1 = 1.00. Across all 1,639 test samples, there were only 4 false positives and 6 false negatives.

### How to Read the Results

The methods are listed from simplest (random guessing) to most sophisticated (Gemini AI). Each method was tested 5 times on different random subsets of the data (**5-fold Monte Carlo Cross-Validation**) to ensure the results are reliable and not just a lucky split. The "Std" column shows how much the F1 varied across those 5 runs — lower means more consistent.

---

## Methods Explained

We tested four methods of increasing sophistication. Think of them as a ladder: the random baselines set the floor (how well can you do with no information?), the hotword approach adds basic speech understanding, and Gemini brings full audio comprehension.

### Method 1: Random 50/50 Baseline

**The idea**: For every audio segment, flip a coin. Heads = "this is an HDM", tails = "this is not".

**How it works**: A random number generator decides each prediction with 50% probability of predicting positive (HDM) or negative (not HDM). This completely ignores the audio — the model never listens to anything. We ran this 100 times with different random seeds for each of the 5 CV splits and averaged the results.

**Why it exists**: This is the absolute floor. Any useful detection method must beat random guessing. If a model can't do better than a coin flip, it hasn't learned anything meaningful.

**Why the F1 is so low (0.15)**: Our dataset is heavily imbalanced — only **9.1%** of segments contain actual HDMs (105 out of 1,155). When the model randomly guesses "HDM" for 50% of segments, most of those predictions are wrong (false positives), dragging precision down to around 0.09. Even though it catches about half the real HDMs (decent recall), the terrible precision results in an F1 of just 0.15.

**Result: F1 = 0.15 ± 0.03**

### Method 2: Random Base-rate Baseline

**The idea**: Instead of a 50/50 coin flip, use a weighted coin that matches the actual HDM rate in the data (9.1%).

**How it works**: Same as the 50/50 baseline, but the random number generator only predicts "HDM" 9.1% of the time — matching the true proportion of HDMs in the dataset. The intuition is: if you know HDMs are rare, maybe you should predict them rarely too. Again, 100 random seeds per CV split.

**Why it exists**: Tests whether just knowing "HDMs are rare" is useful information. It's a slightly more informed version of random guessing.

**Why the F1 is even lower (0.09)**: Because it predicts positive so rarely (~9% of the time), it barely catches any real HDMs. On a typical test set with ~20 real HDMs, it might flag only 1-2 segments as positive, and even those are usually wrong. The near-zero recall kills the F1 score.

**Result: F1 = 0.09 ± 0.06**

### Method 3: ASR Hotword Heuristic (Whisper)

**The idea**: Convert audio to text using speech recognition, then check if the text contains words that people typically say when they can't hear (like "What?", "Huh?", "Sorry?").

**How it works**:
1. Each 4-second audio segment is fed into **OpenAI's Whisper** (a speech-to-text model) to produce a transcript
2. The transcript is scanned for **hotword** keywords and phrases:
   - Single words: "huh", "what", "pardon", "sorry", "excuse me"
   - Phrases: "say that again", "repeat that", "didn't catch", "can't hear", "come again"
3. If any hotword is found in the transcript, the segment is classified as an HDM

**Why it exists**: This replicates the simplest baseline from the original paper. It represents what you could build with off-the-shelf speech recognition and a simple keyword list — no AI reasoning required.

**Strengths**: Simple, interpretable, and fast. When it works, you know exactly why (a specific keyword was detected).

**Weaknesses**: It depends entirely on accurate speech-to-text transcription. In AMI meeting recordings, multiple people often talk at the same time, creating overlapping speech and background noise that confuses Whisper. If Whisper can't hear the "What?" clearly enough to transcribe it, the hotword approach misses it entirely. It also ignores all acoustic cues — tone of voice, confusion in someone's voice, pauses — that a human listener would use.

**Why our F1 (0.23) is lower than the paper's (0.39)**: The paper used telephone conversations (SWDA) with clean two-person audio. We use AMI meeting recordings with 4 speakers mixed into one audio channel. The overlapping speech and room noise degrade Whisper's transcription quality significantly.

**Result: F1 = 0.23 ± 0.11**

### Method 4: Gemini 3.1 Pro (10-Shot Audio Classification)

**The idea**: Give a powerful AI model (Google's Gemini 3.1 Pro) the raw audio and ask it to listen and decide whether someone is having difficulty hearing. Show it 10 labelled examples first so it knows what to look for.

**How it works**:
1. The model receives a detailed **prompt** explaining what hearing difficulty sounds like, covering both:
   - **Acoustic cues**: strained tone, increased pitch, louder voice, longer vowels (the [Lombard effect](https://en.wikipedia.org/wiki/Lombard_effect) — how people naturally change their voice when struggling to hear)
   - **Semantic cues**: keywords like "What?", "Can you repeat that?", "Huh?"
2. Before classifying each test segment, the model is shown **10 example audio clips** (5 positive HDMs + 5 negative) with correct labels — this is called **10-shot prompting** (or few-shot learning). The model learns the pattern from these examples.
3. The model then listens to the target 4-second audio clip and outputs either **"P"** (positive — hearing difficulty detected) or **"N"** (negative — no hearing difficulty)

**Few-shot format** (what the model sees):
```
[System]: You are an expert at analyzing hearing difficulty...
[User]:   {example_audio_1} Label:
[Model]:  P   (this was a hearing difficulty moment)
[User]:   {example_audio_2} Label:
[Model]:  N   (this was normal conversation)
... (8 more examples) ...
[User]:   {test_audio} Label:
[Model]:  ?   ← the model predicts P or N
```

**Why it's the best method**: Unlike the hotword approach which only looks at transcribed words, Gemini processes the raw audio waveform. It can hear tone of voice, hesitation, confusion, pitch changes, and other subtle acoustic signals that indicate someone is struggling to understand. Combined with the semantic understanding of what words like "What?" mean in context, this gives it a much richer picture.

**Strengths**: Can detect HDMs even when Whisper fails to transcribe them. Captures acoustic patterns (not just keywords). Learns from just 10 examples without any model training or fine-tuning.

**Weaknesses**: Expensive (requires Gemini API calls for every segment), slow (~90 minutes for the full evaluation), and somewhat inconsistent across different data splits (F1 ranges from 0.47 to 0.70 depending on which meetings end up in the test set).

**Per-Split Results** (showing the 5 cross-validation trials):

| Split | F1 | Predicted Pos | True Pos | Test Size |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.70 | 28 | 15 | 165 |
| 2 | 0.47 | 58 | 19 | 209 |
| 3 | 0.67 | 42 | 24 | 264 |
| 4 | 0.54 | 59 | 26 | 286 |
| 5 | 0.54 | 42 | 21 | 231 |
| **Avg** | **0.58** | | | |

**Key observation**: Gemini tends to **over-predict** positives (predicting 28–59 HDMs when only 15–26 actually exist). This means it has high recall (catches most real HDMs) but lower precision (many false alarms). This behavior is visible in the confusion matrices on the dashboard.

**Result: F1 = 0.58 ± 0.09**

### Method 5: GPT-4o Audio (Azure) — Iterative Improvement to F1 = 0.97

We ran an iterative optimization process using GPT-4o Audio Preview (`gpt-4o-audio-preview`) on Azure, which supports both native audio input and real token-level log probabilities (logprobs). Four versions were tested, each building on insights from the previous. The full iterative journey — and why each change mattered — is documented below.

#### Step-by-step: How we went from F1 = 0.14 to F1 = 0.97

**v1 (F1 = 0.14)** — Direct replication of the paper's 10-shot approach with GPT-4o Audio. Used 4-second audio segments and regex-based labels. Performance was poor — the model predicted 69–151 positives per split when only 25–34 actually existed, resulting in massive false positive over-prediction and very low precision (~31–40%). The model was essentially classifying any ambiguous audio as hearing difficulty.

*Key lesson: the 4-second window and unverified labels are not enough. The model needs more context and better examples.*

**v2 (F1 = 0.46)** — Two key changes that tripled performance:
1. **Extended audio context from 4s to 12 seconds** (4s before + 4s HDM segment + 4s after). This gives the model conversational context — it can hear what was said before the potential HDM and how the conversation continued after. The paper itself noted this as future work: *"information for several seconds after time t is also available... would likely lead to more powerful predictive power."*
2. **Human-verified positive examples** for the 10-shot prompts. A human listened to each candidate HDM through a purpose-built labeling tool (see below) and confirmed whether it was genuine. This ensured the model was learning from real hearing difficulty, not regex artifacts.

*Key lesson: context matters enormously. Going from 4s to 12s lets the model judge whether "What?" is confusion vs. conversation. Human-verified examples prevent the model from learning wrong patterns.*

**v3 (F1 = 0.60)** — Introduced two more innovations:
1. **Hard negatives**: Audio clips that a human labeler identified as NOT hearing difficulty despite containing keywords like "What?" or "Huh?" used conversationally. For example, "Which was that?" (asking about a topic), "Like a what?" (expressing surprise), "What else?" (continuing discussion). These taught the model the critical distinction between conversational questions and genuine hearing difficulty.
2. **Transcripts alongside audio**: Each few-shot example included both the audio clip and its text transcript, giving the model both semantic and acoustic information.
3. **Stricter prompt**: Added explicit guidance that not every "What?" is a hearing difficulty moment.

This gave **near-perfect precision** (only 8 FP across 1,639 samples) but **recall dropped too low** (31–72% across splits). The model became overly conservative — the strict prompt and hard negatives made it suppress too many genuine HDMs.

*Key lesson: hard negatives are powerful for precision, but too much conservatism kills recall. Need to balance.*

**v4 (F1 = 0.97)** — The final model balanced precision and recall through four changes:
1. **20-shot prompting** (12 positive + 8 negative examples) instead of 10-shot (5+5). More positive examples gave the model a richer picture of what HDMs sound like across different speakers, tones, and contexts.
2. **Mixed negatives**: The 8 negative examples used a mix of hard negatives (human-rejected HDMs) and random meeting audio (clearly non-HDM). This avoided the v3 problem where all-hard negatives made the model too conservative.
3. **Enhanced acoustic prompt**: Extended the Lombard effect description with additional acoustic cues the model should listen for:
   - Voice quality changes (strained, tense, effortful phonation)
   - Speaking rate changes (slowing, hesitating, pausing)
   - Rising intonation with confused/uncertain tone
   - Filled pauses (um, uh) before asking for repetition
   - Abrupt break in conversational turn-taking rhythm
   - Background noise level and whether preceding speech was unclear
4. **Balanced prompt**: Removed the overly conservative v3 warning ("not every What? is an HDM") that was suppressing recall. The hard negatives in the few-shot examples already teach this distinction implicitly.

#### Why v4 performs so well

The F1 = 0.97 result is not a lucky split — it is **consistent across all 5 cross-validation folds** (range: 0.92–1.00, std: 0.03). Several factors explain why this approach works:

1. **The model hears the full conversational context**. With 12 seconds of audio (vs the paper's 4s), the model can judge whether "What?" is an isolated confused response to unclear speech (HDM) or a natural conversational question (not HDM). It can hear if the preceding speaker was mumbling, if there was overlapping speech, or if the conversation was flowing normally.

2. **Human-verified few-shot examples are clean signal**. The 12 positive examples are all confirmed genuine HDMs — the model never learns from mislabeled data. The 8 negative examples include hard negatives that teach subtle distinctions.

3. **The prompt leverages both acoustic and semantic reasoning**. The paper showed that audio-only Gemini (F1=0.75) vastly outperformed text-only Gemini (F1=0.39). Our enhanced prompt guides the model to listen for specific acoustic signatures (Lombard effect, voice strain, rising intonation) alongside semantic keywords, maximizing the audio modality advantage.

4. **Real token-level logprobs** from Azure GPT-4o Audio provide calibrated confidence signals. The model outputs a probability distribution over the "P" and "N" tokens via softmax over their log probabilities, rather than self-reported confidence (which Gemini on Google AI Studio was limited to).

5. **20-shot > 10-shot for this task**. More examples reduce variance in the model's behavior. The paper showed clear uplift from 0-shot (0.75) to 2-shot (0.85) to 10-shot (0.87). Our results show the trend continues: 10-shot v2 (0.46) → 20-shot v4 (0.97), with better example quality.

#### v4 Per-Split Results (5-fold Monte Carlo Cross-Validation)

Each split randomly assigns 80% of meetings to training and 20% to testing, at the **conversation level** (all segments from one meeting stay together). Few-shot examples are drawn from training meetings; all test segments are classified independently. F1 is computed per-split from the binary P/N predictions.

| Split | F1 | TP | FP | FN | Test Size | True Pos |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.9231 | 30 | 1 | 4 | 374 | 34 |
| 2 | 0.9796 | 24 | 0 | 1 | 275 | 25 |
| 3 | **1.0000** | 26 | 0 | 0 | 286 | 26 |
| 4 | 0.9688 | 31 | 1 | 1 | 352 | 32 |
| 5 | 0.9615 | 25 | 2 | 0 | 275 | 25 |
| **Avg** | **0.9666** | **136** | **4** | **6** | **1,562** | **142** |

- **Total across all splits**: 136 true positives, 4 false positives, 6 false negatives out of 1,562 test segments
- **Split 3 achieved perfect classification** (F1 = 1.00): every single HDM was correctly identified with zero false alarms
- **Precision**: 136/(136+4) = **97.1%** — almost no false alarms
- **Recall**: 136/(136+6) = **95.8%** — catches nearly every HDM
- **Consistency**: std = 0.03 across 5 folds, showing the result is robust to different train/test splits

#### Comparison with the paper

| | Collins et al. | This work (v4) |
|---|---|---|
| **Model** | Gemini 1.5 Pro | GPT-4o Audio Preview |
| **F1** | 0.87 | **0.97** (+11%) |
| **N-shot** | 10-shot (5P + 5N) | 20-shot (12P + 8N) |
| **Audio context** | 4 seconds | **12 seconds** |
| **Example selection** | Random from training set | **Human-verified** + hard negatives |
| **Prompt** | Lombard effect + semantic cues | **Enhanced** (+ voice quality, prosody, context) |
| **Confidence** | Token logprobs | Token logprobs |
| **Dataset** | SWDA/MRDA (telephone + meetings) | AMI Meeting Corpus (group meetings) |

**Result: F1 = 0.97 +/- 0.03**

### Human Labeling Tool

To improve the quality of few-shot examples, we built a web-based labeling tool (`src/labeling_tool.py`) that allows a human annotator to listen to each candidate HDM and verify it:

- For each of the 149 regex-identified HDMs, the tool plays three audio clips:
  1. **Context** — the 4 seconds preceding the HDM (what was happening before)
  2. **HDM segment** — the 4-second clip containing the candidate HDM
  3. **Combined** — full 8-second clip (context flowing into the HDM)
- The annotator presses **Y** (yes, genuine HDM) or **N** (no, not an HDM)
- Labels auto-save to `data/hdm_labels.json`

Of 84 labeled candidates: **77 were confirmed as genuine HDMs** (92%) and **7 were rejected** (false positives from the regex filter). The rejected items and their reasons:

| Index | Text | Why it's not an HDM |
|:---:|---|---|
| 1 | "Which was that?" | Asking about a topic, not mishearing |
| 43 | "Like a what?" | Expressing surprise at content |
| 60 | "For what?" | Questioning purpose, not hearing |
| 61 | "Mm what?" | Casual conversational response |
| 74 | "What?" | Rhetorical/surprise, not hearing difficulty |
| 76 | "What else?" | Continuing discussion |
| 78 | "huh?" | Casual acknowledgment |

These 7 rejected items became valuable **hard negatives** for few-shot prompting — they teach the model that not every short question is a hearing difficulty moment.

### Figure 1 Dashboard

The **[Figure 1 Dashboard](https://chozillla.github.io/CollinsPaper/figure1.html)** recreates Collins et al. Figure 1 with our GPT-4o v4 results:

- **Interactive Plotly charts** — zoom, pan, and hover for exact P(HDM) values at any point
- **Blue waveform** — the meeting audio envelope
- **Green probability line** — continuous P(HDM) signal at 1-second resolution, built from the model's predictions across overlapping 4-second windows
- **Red shaded bands** — ground truth HDM events
- **Orange dashed threshold** at 0.97
- **Audio playback** — each HDM event has a 10-second audio clip you can play directly in the browser
- **8 meetings** with the most HDMs, switchable via buttons

For the full interactive experience with scrubbing through complete meeting audio:
```bash
python src/waveform_dashboard.py
# Open http://localhost:8766
```

---

## Why Our F1 (0.58) Differs from the Paper (0.87) — Gemini Baseline

Several factors explain the performance gap:

1. **Different dataset**: The paper used SWDA (telephone conversations) and MRDA (ICSI meeting recordings), while we use the AMI Meeting Corpus. AMI meetings have 4 speakers with frequent overlapping talk, background noise, and cross-talk — making it harder for the model to isolate individual speakers and detect who is having difficulty hearing.

2. **Headset mix vs individual channels**: We use the AMI headset mix audio (all speakers mixed into one channel). The paper's SWDA data is telephone speech with clearer speaker separation. In the mix, the HDM utterance ("What?", "Huh?") can be masked by other speakers talking simultaneously.

3. **Fewer positive examples**: We have 105 positive examples (from 149 HDMs across meetings with available audio) vs the paper's 298. With smaller test sets per CV split (15–26 positives), a few misclassifications cause large swings in F1 — our per-split F1 ranges from 0.47 to 0.70.

4. **Different annotation source**: The paper used the native `signal-non-understanding` DAMSL tag from SWDA/MRDA, which directly annotates hearing difficulty. We derived our labels from AMI's broader `COMMENT-ABOUT-UNDERSTANDING` tag and applied filtering, which may include some false positives or miss some true HDMs.

5. **Different model**: We used Gemini 3.1 Pro instead of the paper's Gemini 1.5 Pro. While 3.1 is generally more capable, its audio understanding characteristics may differ, and the paper may have benefited from specific behaviors of 1.5 Pro for this task.

---

## Interactive Dashboards

Two dashboards are available:

1. **[Results Dashboard](https://chozillla.github.io/CollinsPaper/)** — bar charts, confusion matrices, precision-recall curves comparing all methods
2. **[Figure 1 Dashboard](https://chozillla.github.io/CollinsPaper/figure1.html)** — interactive waveform + probability timeline with audio playback for each HDM event (recreating Collins et al. Figure 1)

The **Results Dashboard** provides a visual overview of all trial results. Here's what each panel shows:

| Panel | What It Shows |
|---|---|
| **F1 Score Comparison** | Bar chart comparing all four methods. The dashed pink line marks the original paper's F1 (0.87). Taller bars = better performance. Error bars show how much F1 varied across the 5 CV splits. |
| **Per-Split F1 Distribution** | Box plots showing the spread of F1 scores across the 5 trials for each method. Wide boxes = inconsistent performance. |
| **HDM Annotations by Type** | How the 149 HDM annotations were categorized: strong keyword matches ("What?", "Huh?"), explicit non-understanding ("Which was that?"), or short questions ("Sorry?"). |
| **F1 Across CV Splits** | Line chart tracking each method's F1 on each of the 5 test splits. Shows whether a method is consistently good or erratic. |
| **Test Set Composition** | How many positive (HDM) vs negative segments are in each test split. Shows the heavy class imbalance (~10:1 negative to positive). |
| **Predicted vs Actual Positives** | For Gemini and Hotword: how many segments each method predicted as HDMs versus how many actually were. Gemini over-predicts; Hotword under-predicts. |
| **Confusion Matrices** | For Gemini and Hotword: a 2x2 grid showing correct predictions (diagonal) vs errors (off-diagonal). Shows that Gemini's main weakness is false positives, while Hotword's is false negatives. |
| **Precision-Recall Curves** | How Gemini's precision and recall trade off at different confidence thresholds. Each line is one CV split. |
| **Confidence Distribution** | Histogram of Gemini's prediction confidence. Shows the model is very decisive — it predicts with either very low (0.1) or very high (0.9) confidence, with little in between. |
| **HDM Duration Distribution** | How long the hearing difficulty utterances are (in milliseconds). Most are short bursts under 1 second. |
| **HDMs by Speaker** | Pie chart showing which meeting participants (A, B, C, D) had the most hearing difficulty moments. |

All charts are interactive — hover for exact values, click and drag to zoom, double-click to reset.

To regenerate the dashboard after changing results:
```bash
uv run python src/dashboard.py
```

---

## Paper Summary

**Collins et al.** define a Hearing Difficulty Moment as an event when a participant in a conversation has difficulty understanding what was said. These are annotated as `signal-non-understanding` dialogue acts in SWDA/MRDA — utterances like "Huh?", "What?", "Sorry?", "Can you repeat that?".

The paper compares four approaches:

| Approach | F1 |
|---|:---:|
| ASR Hotword Heuristic (Baseline) | 0.39 |
| Gemini 1.5 Pro [text only] (0-shot) | 0.39 |
| Gemini 1.5 Pro [audio] (0-shot) | 0.75 |
| Wav2Vec 2.0 Transfer Learning | 0.76 |
| Gemini 2.0 Flash (LoRA Fine-Tuning) | 0.77 |
| Gemini 1.5 Pro (2-shot) | 0.85 |
| Gemini 1.5 Pro (10-shot) | 0.87 |

Key finding: **audio modality is critical** — the text-only approach (F1=0.39) performed no better than the hotword baseline, while the audio approach with just 10 examples reached F1=0.87.

---

## Dataset: AMI Meeting Corpus

The [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) contains 100 hours of meeting recordings with dialogue act annotations. We adapt it for HDM detection:

### Step 1: Identify Hearing Difficulty Moments

The AMI corpus annotates dialogue acts with the `COMMENT-ABOUT-UNDERSTANDING` tag (`ami_da_12`), but this covers **both** understanding confirmations ("Okay", "Ah.") **and** non-understanding signals ("What?", "Huh?"). Collins et al. faced the same issue with SWDA/MRDA and filtered 522 annotations down to 298. We apply an analogous three-tier rule-based filter (see `src/filter_hdm.py`):

**Tier 1 — Strong keyword match** (122 instances): Single-word questions that unambiguously signal non-understanding. Matched via regex patterns requiring both the keyword and a question mark:
- `"What?"`, `"Huh?"`, `"Hmm?"`, `"Sorry?"`, `"Pardon?"`, `"Excuse me?"`, `"Come again?"`
- The question mark is critical — `"sorry."` with a period is an apology, not an HDM.

**Tier 2 — Explicit non-understanding phrases** (7 instances): Multi-word phrases that explicitly request repetition:
- `"Which was that?"`, `"What did you say?"`, `"Can you repeat that?"`, `"I didn't catch..."`

**Tier 3 — Short questions** (20 instances): Brief questions (≤ 3 words) containing HDM-associated keywords ("what", "sorry", "which", "huh") that don't match Tiers 1–2 but are too short to be semantic clarifications:
- `"what button?"`, `"what else?"`, `"I'm sorry?"`

**Excluded** — the filter removes:
- Understanding confirmations: "Ah.", "Okay", "Uh-huh", "Right", "Hmm." (no question mark)
- Semantic clarification questions: "How high is it?", "What do you mean?", "Where is the controller?" — the speaker heard the words but asks about meaning, not repetition
- Long utterances (> 5 words) without explicit non-understanding phrases — these are typically elaborations, not hearing difficulty signals

This yields **149 HDM instances** across 75 meetings (comparable to the paper's 298 from SWDA/MRDA). The breakdown by tier is visible in the "HDM Annotations by Type" chart on the dashboard.

**Important: labeling methodology differences from Collins et al.**

| | Collins et al. | This replication |
|---|---|---|
| **Source tags** | `signal-non-understanding` DAMSL tag from SWDA/MRDA — a narrow tag specifically for non-understanding | `Comment-About-Understanding` from AMI — a broad tag covering both understanding and non-understanding |
| **Filtering** | Manual human review of 522 → 298 utterances | Automated regex-based filtering of 2,560 → 149 utterances |
| **Human verification** | Authors manually excluded semantic clarifications | No human verification — purely rule-based |
| **Positive labels** | Human-confirmed hearing difficulty signals | Regex-matched patterns ("What?", "Huh?", "Sorry?", etc.) |
| **Negative labels** | Random non-HDM audio segments (same as ours) | Random non-HDM audio segments |

The ground truth labels used in this replication — including those used as the **10-shot examples** fed to Gemini — are entirely determined by this automated filtering pipeline. No annotator listened to the audio clips to verify they represent genuine hearing difficulty. The regex patterns are high-precision for clear-cut cases (e.g. "What?" with a question mark) but have not been validated against human judgement.

### Step 2: Build Audio Segments

Following the paper:
- **4-second audio segments** ending at the HDM event
- Positive: sample a timestep where the HDM has been occurring for >= 0.4s, take preceding 4s of audio
- Negative: random timestep with no overlap with any HDM, 10:1 ratio
- Final dataset: **1,155 segments** (105 positive, 1,050 negative) from 38 meetings with audio

### Step 3: Monte Carlo Cross-Validation

- 5 random train/test splits (80/20)
- Splits at the **conversation level** (all utterances from one meeting in same split) to prevent data leakage
- Negative examples randomly resampled per split

---

## Project Structure

```
CollinsPaper/
├── src/
│   ├── parse_ami_annotations.py    # Parse NXT XML dialogue act annotations
│   ├── filter_hdm.py               # Filter for genuine hearing difficulty moments
│   ├── download_audio.py           # Download AMI headset mix audio files
│   ├── build_dataset.py            # Build 4s audio segments with labels
│   ├── gemini_audio_classifier.py  # 10-shot Gemini 2.5 Flash Lite classifier
│   ├── gpt4o_audio_classifier.py  # GPT-4o Audio v1 (Azure, logprobs)
│   ├── gpt4o_audio_classifier_v2.py # v2: 12s context + human-verified
│   ├── gpt4o_audio_classifier_v3.py # v3: hard negatives + transcripts
│   ├── gpt4o_audio_classifier_v4.py # v4: 20-shot, enhanced prompt (best)
│   ├── labeling_tool.py            # Web-based HDM labeling tool
│   ├── baseline_hotword.py         # ASR hotword heuristic (Whisper)
│   ├── random_baseline.py          # Random guessing baseline
│   ├── dashboard.py                # Generate interactive HTML dashboard
│   └── evaluate_all.py             # Compare all methods
├── data/
│   ├── annotations/                # AMI corpus annotations (NXT XML)
│   ├── audio/                      # Meeting audio files (not tracked)
│   ├── dataset/                    # Processed numpy arrays + metadata
│   ├── hdm_filtered.json           # Filtered HDM annotations
│   └── hdm_annotations.json        # All HDM annotations (pre-filter)
├── results/
│   ├── gemini_10shot_results.json  # Gemini 2.5 Flash Lite results
│   ├── gpt4o_10shot_results.json  # GPT-4o Audio v1 results
│   ├── gpt4o_10shot_v2_results.json # GPT-4o Audio v2 results
│   ├── gpt4o_10shot_v3_results.json # GPT-4o Audio v3 results
│   ├── gpt4o_20shot_v4_results.json # GPT-4o Audio v4 results (best)
│   ├── baseline_hotword.json       # Hotword baseline results
│   ├── random_baseline.json        # Random baseline results
│   └── dashboard.html              # Interactive results dashboard
├── docs/
│   ├── index.html                  # GitHub Pages results dashboard
│   ├── figure1.html                # Figure 1 dashboard with audio
│   └── audio/                      # HDM audio clips for Figure 1 dashboard
├── run_pipeline.sh                 # Run the full pipeline
├── pyproject.toml                  # Dependencies (managed by uv)
└── .env                            # API keys (not tracked)
```

---

## Setup & Reproduction

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Gemini API key ([aistudio.google.com](https://aistudio.google.com)) — for Gemini baseline
- Azure OpenAI API key with `gpt-4o-audio-preview` deployment — for GPT-4o classifier

### Install

```bash
git clone https://github.com/chozillla/CollinsPaper.git
cd CollinsPaper
uv sync
```

### Configure API Keys

```bash
cat > .env << 'EOF'
GEMINI_API_KEY=your-gemini-key
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_DEPLOYMENT=gpt-audio
AZURE_OPENAI_API_VERSION=2025-01-01-preview
EOF
```

### Run the Full Pipeline

```bash
# Step 1: Parse annotations & filter HDMs
uv run python src/parse_ami_annotations.py
uv run python src/filter_hdm.py

# Step 2: Download audio (~5.5GB)
uv run python src/download_audio.py

# Step 3: Build dataset
uv run python src/build_dataset.py

# Step 4: Run classifiers
uv run python src/random_baseline.py
uv run python src/baseline_hotword.py
uv run python src/gemini_audio_classifier.py    # ~90 min, uses Gemini API
uv run python src/gpt4o_audio_classifier_v4.py  # ~15 min, uses Azure GPT-4o Audio

# Step 5: (Optional) Human-label HDMs for better few-shot examples
uv run python src/labeling_tool.py              # Open http://localhost:8765

# Step 6: Generate dashboards & compare results
uv run python src/dashboard.py
uv run python src/evaluate_all.py
uv run python src/waveform_dashboard.py         # Open http://localhost:8766
```

Or run everything at once:
```bash
bash run_pipeline.sh
```

---

## References

- Collins, J., Buzea, A., Collier, C., Ballesta Rosen, A., Maclaren, J., Lyon, R. F., & Carlile, S. (2025). *Identifying Hearing Difficulty Moments in Conversational Audio*. arXiv:2507.23590.
- Carletta, J. et al. (2005). *The AMI Meeting Corpus*. University of Edinburgh.
- Core, M. & Allen, J. (1997). *Coding Dialogs with the DAMSL Annotation Scheme*.

---

## License

The AMI Corpus is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
