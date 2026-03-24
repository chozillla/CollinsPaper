# Replicating "Identifying Hearing Difficulty Moments in Conversational Audio"

Replication of [Collins et al. (2025)](https://arxiv.org/abs/2507.23590) using the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) and Gemini 3.1 Pro.

The original paper detects **Hearing Difficulty Moments (HDMs)** — instances where a listener struggles to comprehend dialogue (e.g., "What?", "Huh?", "Sorry?") — using multimodal audio language models. Their best approach (Gemini 1.5 Pro, 10-shot prompting) achieved **F1 = 0.87** on the SWDA/MRDA datasets.

We replicate the 10-shot audio LLM prompting method on a different dataset (AMI Meeting Corpus) with a newer model (Gemini 3.1 Pro).

---

## Results

| Method | Avg F1 (5-fold MCCV) |
|---|:---:|
| Random guessing (50/50) | 0.15 |
| Random guessing (base-rate) | 0.09 |
| ASR Hotword Heuristic (Whisper) | 0.23 |
| **Gemini 3.1 Pro (10-shot audio)** | **0.58** |
| *Collins et al. Gemini 1.5 Pro (10-shot)* | *0.87* |

The 10-shot Gemini classifier is **3.8x better than random guessing** and **2.5x better than the hotword baseline**, confirming that audio language models can detect hearing difficulty moments from conversational audio.

The gap vs. the paper (0.58 vs 0.87) is expected due to differences in dataset (AMI multi-speaker meetings vs SWDA/MRDA telephone conversations), audio format (headset mix vs individual channels), and fewer positive examples (105 vs 298).

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

The AMI corpus uses the `COMMENT-ABOUT-UNDERSTANDING` dialogue act tag (`ami_da_12`), which covers both understanding and non-understanding. We filter to keep only genuine **signal-non-understanding** utterances:

- **Keep**: "What?", "Huh?", "Sorry?", "Pardon?", "Excuse me?", "What was that?", "I didn't catch that"
- **Remove**: Understanding confirmations ("Ah.", "Okay", "Uh-huh")
- **Remove**: Semantic clarification questions ("How high is it?", "What do you mean?")
- **Remove**: Acknowledgments without question marks ("Hmm.", "sorry.")

This yields **149 HDM instances** across 75 meetings (comparable to the paper's 298 from SWDA/MRDA).

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

## 10-Shot Audio LLM Approach

We replicate the paper's best method (Section 2.3) using **Gemini 3.1 Pro** instead of Gemini 1.5 Pro.

### Prompt

The exact prompt from the paper instructs the model to consider:

1. **Non-semantic cues**: tone, pitch, Lombard effect indicators (increased fundamental frequency, spectral tilting, increased vowel duration, etc.)
2. **Semantic cues**: keywords like "What?", "Can you repeat that?", "Huh?"
3. **Subjectivity**: focus on the speaker's experience

The model outputs "P" (positive/hearing difficulty) or "N" (negative).

### Few-Shot Format

For 10-shot classification:
1. Present 5 positive + 5 negative audio examples with labels from the training set
2. Append the target audio segment
3. Model completes with P or N

```
[System]: {classification prompt}
[User]:   {audio_example_1} Label:
[Model]:  P
[User]:   {audio_example_2} Label:
[Model]:  N
...
[User]:   {target_audio} Label:
[Model]:  ?  ← model predicts P or N
```

### Per-Split Results

| Split | F1 | Predicted Pos | True Pos | Test Size |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.70 | 28 | 15 | 165 |
| 2 | 0.47 | 58 | 19 | 209 |
| 3 | 0.67 | 42 | 24 | 264 |
| 4 | 0.54 | 59 | 26 | 286 |
| 5 | 0.54 | 42 | 21 | 231 |
| **Avg** | **0.58** | | | |

---

## Project Structure

```
CollinsPaper/
├── src/
│   ├── parse_ami_annotations.py    # Parse NXT XML dialogue act annotations
│   ├── filter_hdm.py               # Filter for genuine hearing difficulty moments
│   ├── download_audio.py           # Download AMI headset mix audio files
│   ├── build_dataset.py            # Build 4s audio segments with labels
│   ├── gemini_audio_classifier.py  # 10-shot Gemini 3.1 Pro classifier
│   ├── baseline_hotword.py         # ASR hotword heuristic (Whisper)
│   ├── random_baseline.py          # Random guessing baseline
│   └── evaluate_all.py             # Compare all methods
├── data/
│   ├── annotations/                # AMI corpus annotations (NXT XML)
│   ├── audio/                      # Meeting audio files (not tracked)
│   ├── dataset/                    # Processed numpy arrays + metadata
│   ├── hdm_filtered.json           # Filtered HDM annotations
│   └── hdm_annotations.json        # All HDM annotations (pre-filter)
├── results/
│   ├── gemini_10shot_results.json  # Gemini 3.1 Pro results
│   ├── baseline_hotword.json       # Hotword baseline results
│   └── random_baseline.json        # Random baseline results
├── run_pipeline.sh                 # Run the full pipeline
├── pyproject.toml                  # Dependencies (managed by uv)
└── .env                            # API keys (not tracked)
```

---

## Setup & Reproduction

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Gemini API key ([aistudio.google.com](https://aistudio.google.com))

### Install

```bash
git clone <repo-url>
cd CollinsPaper
uv sync
```

### Configure API Key

```bash
echo "GEMINI_API_KEY=your-key-here" > .env
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

# Step 5: Compare results
uv run python src/evaluate_all.py
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
