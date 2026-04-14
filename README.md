# Detecting Hearing Difficulty Moments in Meeting Audio

A replication and extension of [Collins et al. (2025)](https://arxiv.org/abs/2507.23590) — using Gemini 2.5 Flash on Vertex AI to automatically detect moments when listeners struggle to understand what was said in the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/).

---

## Background

**Hearing Difficulty Moments (HDMs)** are brief moments in conversation when a listener fails to understand what was said — signaled by responses like *"What?"*, *"Huh?"*, *"Sorry?"*, or *"Which was that?"*. Collins et al. (2025) demonstrated that large language models can detect these moments from audio alone, achieving F1 = 0.87 with Gemini 1.5 Pro using 10-shot prompting.

We distinguish two types of HDM:

- **Type A — Acoustic:** The listener physically misheard the audio (unclear speech, background noise, overlapping speakers)
- **Type B — Comprehension:** The listener heard the words but lacked the language or comprehension to understand (unfamiliar jargon, accented speech, complex phrasing)

This project replicates the Collins et al. approach on the full AMI Meeting Corpus and extends it with:

- A **sliding window classifier** that produces a continuous P(HDM) probability signal across entire meetings (not just per-segment classification)
- A **human labeling pipeline** where annotators listen to full meeting audio and place typed HDM markers (A or B), replacing the prior AI-only labels
- An **interactive waveform dashboard** with audio playback, recreating the paper's Figure 1 visualization
- **Signal quality metrics** for evaluating detection accuracy per meeting

---

## Pipeline Overview

| Color | Stage |
|-------|-------|
| 🔵 Blue | Data ingestion — AMI corpus audio and annotations |
| 🩷 Pink | Human labeling — annotators listen and mark Type A/B HDMs |
| 🟢 Green | AI classification — Gemini 10-shot using human labels as training data |
| 🟡 Yellow | Cross-validation — verify 10-shot examples generalize to unseen meetings |

```mermaid
flowchart TB
    AMI(("AMI Corpus\n75 meetings · WAV audio")) --> PARSE["Parse XML annotations\nfind dialogue acts tagged as 'comment about understanding'"]
    PARSE --> CANDIDATES["149 HDM candidate moments"]
    AMI --> HUMAN["Humans listen to each audio clip"]
    CANDIDATES --> HUMAN
    HUMAN --> TYPEA["Type A: Acoustic\nlistener misheard what was said"]
    HUMAN --> TYPEB["Type B: Comprehension\nlistener heard it but couldn't make sense of it"]
    TYPEA --> LABELS["Human-verified HDM labels\nused as training data for the model"]
    TYPEB --> LABELS
    LABELS --> SHOTS["Pick 10-shot examples for the model\n5 real HDMs + 5 non-HDM clips"]
    AMI --> GEMINI["Run Gemini 2.5 Flash on full meetings\nslides a 12s window across the audio"]
    SHOTS --> GEMINI
    GEMINI --> SIGNAL["Continuous HDM probability\n0 = no difficulty · 1 = difficulty detected"]
    SIGNAL --> CV["Cross-validation\ntest on meetings the model hasn't seen"]
    LABELS --> CV
    CV --> RESULT["Does it generalize?\nhow often it catches real HDMs vs. false alarms"]

    style AMI fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style PARSE fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style CANDIDATES fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style HUMAN fill:#fce7f3,stroke:#ec4899,color:#831843
    style TYPEA fill:#fce7f3,stroke:#ec4899,color:#831843
    style TYPEB fill:#fce7f3,stroke:#ec4899,color:#831843
    style LABELS fill:#fce7f3,stroke:#ec4899,color:#831843
    style SHOTS fill:#d1fae5,stroke:#10b981,color:#064e3b
    style GEMINI fill:#d1fae5,stroke:#10b981,color:#064e3b
    style SIGNAL fill:#d1fae5,stroke:#10b981,color:#064e3b
    style CV fill:#fef3c7,stroke:#f59e0b,color:#78350f
    style RESULT fill:#fef3c7,stroke:#f59e0b,color:#78350f
```

---

## The Data

The [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) is a well-known dataset of 100 hours of recorded meetings. In the scenario portion, teams of 4 people role-play designing a TV remote control across multiple sessions. Participants include a Project Manager, Marketing Expert, User Interface Designer, and Industrial Designer.

Each meeting ID encodes its metadata — `ES2002b` means **E**dinburgh site, **S**cenario, group **2002**, session **b** (functional design). Recordings come from three sites (Edinburgh, Idiap in Switzerland, TNO in the Netherlands) with different room acoustics. Most participants are non-native English speakers, which is relevant since accented speech may affect both HDM frequency and model detection.

**HDM extraction:** The AMI corpus includes dialogue act annotations with a `COMMENT-ABOUT-UNDERSTANDING` tag. We parsed the XML annotations and applied a three-tier keyword filter to identify **149 confirmed HDMs** across **75 meetings**. These range from simple *"Sorry?"* to more complex *"Which was that?"*. Each HDM is paired with 10x negative samples (non-HDM segments) for a total of **1,639 segments**.

| | Count |
|---|---|
| Meetings with HDMs | 75 |
| Positive segments (HDMs) | 149 |
| Negative segments | 1,490 |
| Total segments | 1,639 |

Sessions follow four design phases:

| Session | Phase | Avg Duration |
|---------|-------|-------------|
| a | Kick-off — team intros, task overview | ~20 min |
| b | Functional design — requirements, specs | ~37 min |
| c | Conceptual design — components, UI concepts | ~38 min |
| d | Detailed design — final evaluation | ~35 min |

---

## What Has Been Done

### 1. Sliding Window Classifier

The core contribution. Gemini 2.5 Flash processes each meeting as a series of overlapping 12-second audio windows (4s context + 4s target + 4s after) at 4-second intervals. For each window, the model outputs a single "P" (positive) or "N" (negative) token. We extract the logprob for each token and compute:

```
P(HDM) = softmax(logprob_P, logprob_N)
```

This produces a continuous 0-1 probability signal across the entire meeting — not just a binary label per segment. Vertex AI is required because the standard Gemini API does not support token-level logprobs.

**Results completed:**
- 10-shot (5P + 5N examples): all 75 meetings
- Zero-shot: all 75 meetings
- Gemini 3.1 Pro backup: all 75 meetings

### 2. Baseline Methods

- **ASR Hotword Heuristic** — keyword matching on common HDM phrases
- **Random Baselines** — 50/50 coin flip and base-rate classifiers

### 3. Signal Quality Evaluation

Per-meeting scoring (0-100) combining detection accuracy and specificity:

| Metric | Mean |
|--------|------|
| Alignment Score | 74.3 |
| Peak probability near HDMs | 0.82 |
| Noise floor (non-HDM regions) | 0.17 |
| False alarm rate | 10.2% |
| Signal-to-noise ratio | 7.5 |

88% of meetings have their HDMs detected (peak P(HDM) > 0.5).

### 4. Interactive Dashboard

A waveform dashboard (`waveform_dashboard.py`, port 8766) recreating Collins et al. Figure 1 with full audio playback:

- Blue waveform with green P(HDM) probability overlay
- Red bands marking ground-truth HDM events
- Click-to-seek audio playback with keyboard shortcuts
- Per-HDM clip playback for individual events
- Sort and filter by alignment score, recording site, session phase, or participant group

### 5. Human Labeling Pipeline

The 10-shot examples that drive the classifier must come from human judgment, not AI. We built a labeling pipeline (`human_labeling_pipeline.py`, port 8770) where annotators:

1. Listen to full meeting audio with the waveform visible
2. Click the waveform at any timestamp where they hear an HDM
3. Classify each as **Type A** (acoustic — misheard) or **Type B** (comprehension — heard but didn't understand)
4. Optionally add notes (e.g. "overlapping speakers", "strong accent")

Multiple labelers are supported via `--labeler` flag for inter-annotator agreement. Labels are saved per-labeler per-meeting to `data/human_hdm_labels.json` and serve as training data for the classifier.

### 6. Cross-Validation

To verify that the human-labeled 10-shot examples generalize to unseen audio, we run K-fold cross-validation at the meeting level (`cross_validate_human_labels.py`):

1. Split labeled meetings into K folds (no meeting appears in both train and test)
2. For each fold, select 5P + 5N few-shot examples from training meetings' human labels
3. Run Gemini sliding window on held-out test meetings using those examples
4. Score: does P(HDM) peak near human-labeled timestamps in the test set?

Reports per-fold and aggregate metrics: detection rate, mean peak probability, false alarm rate, and noise floor.

### 7. Validation

- **Data leakage audit** — 8-point check confirming no train/test contamination (see `VALIDATION.md`)
- **Prediction browser** — manual inspection of every classification with audio

---

## Quick Start

### Prerequisites

- Python 3.12+
- Google Cloud account with Vertex AI API enabled
- AMI Meeting Corpus audio (16kHz mono WAV)

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install google-genai numpy soundfile tqdm python-dotenv scikit-learn plotly torch transformers
```

### Authenticate with Vertex AI

Vertex AI is required for logprobs — the standard Gemini API doesn't support them.

```bash
gcloud auth application-default login --project <YOUR_PROJECT_ID>
```

### Run

```bash
# Run the sliding window classifier (all meetings, 10-shot)
python src/gemini_sliding_window.py

# Single meeting
python src/gemini_sliding_window.py --meeting ES2003b

# Zero-shot mode
python src/gemini_sliding_window.py --shots 0

# Launch the interactive dashboard
python src/waveform_dashboard.py
# Open http://localhost:8766

# Evaluate signal quality
python src/evaluate_signal.py --plot

# Launch the human labeling pipeline
python src/human_labeling_pipeline.py --labeler alice
# Open http://localhost:8770

# Cross-validate that human labels generalize
python src/cross_validate_human_labels.py --labeler alice --folds 5
python src/cross_validate_human_labels.py --dry-run   # preview splits only
```

---

## Project Structure

```
src/
  gemini_sliding_window.py     # Main classifier — Gemini 2.5 Flash sliding window
  waveform_dashboard.py        # Interactive dashboard with audio (port 8766)
  evaluate_signal.py           # Signal quality scoring and plots
  build_dataset.py             # Dataset construction from AMI annotations
  filter_hdm.py                # HDM annotation filtering (3-tier keyword filter)
  parse_ami_annotations.py     # AMI XML annotation parser
  download_audio.py            # AMI audio downloader
  baseline_hotword.py          # ASR hotword baseline
  random_baseline.py           # Random baselines
  gemini_audio_classifier.py   # Gemini per-segment baseline classifier
  figure1_dashboard.py         # Static Figure 1 (Plotly HTML)
  validation_dashboard.py      # Prediction browser with audio
  validation_audit.py          # Data leakage audit
  labeling_tool.py             # Legacy AI-label verification tool (Yes/No per HDM)
  human_labeling_pipeline.py   # Human labeling pipeline — Type A/B markers (port 8770)
  cross_validate_human_labels.py  # K-fold CV: do human 10-shot examples generalize?

results/
  sliding_window_10shot/       # 10-shot P(HDM) results (75 meetings)
  sliding_window/              # Zero-shot P(HDM) results (75 meetings)
  sliding_window_gemini31pro_backup/  # Gemini 3.1 Pro backup (75 meetings)
  signal_evaluation.html       # Signal quality plots
  baseline_hotword.json        # Hotword baseline results
  random_baseline.json         # Random baseline results

data/
  hdm_annotations.json         # All parsed HDM annotations (2,560 entries)
  hdm_filtered.json            # Filtered HDM set (149 positives)
  hdm_labels.json              # Legacy AI-label verification (yes/no)
  human_hdm_labels.json        # Human labels (Type A/B, per labeler) — training data
  audio/                       # AMI WAV files (not tracked)
  dataset/                     # Built dataset + segments (not tracked)

results/
  cross_validation/            # K-fold CV results (per-fold + summary)
```

---

## Reference

Collins, J., Banos, A., Culley, C., Ballesta Rosen, A., Machum, J., Lyon, R. F., & Carlile, S. (2025). *Identifying Hearing Difficulty Moments in Conversational Audio*. arXiv:2507.23590.
