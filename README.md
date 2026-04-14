# Replicating "Identifying Hearing Difficulty Moments in Conversational Audio"

Replication and extension of [Collins et al. (2025)](https://arxiv.org/abs/2507.23590) using the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/).

**Gemini 2.5 Flash** on Vertex AI with token-level logprobs generates a continuous P(HDM) probability signal across entire meetings, matching the paper's Figure 1 visualization with interactive audio playback.

---

## What is This Project About?

Collins et al. (2025) showed that AI models can automatically detect **Hearing Difficulty Moments (HDMs)** — moments when a listener struggles to understand what was said (e.g. "What?", "Huh?", "Sorry?"). Their best model (Gemini 1.5 Pro, 10-shot) achieved F1 = 0.87.

This project replicates and extends their work on the AMI Meeting Corpus using **Gemini 2.5 Flash** with a sliding window approach to produce a continuous probability signal, and an interactive dashboard to explore the results with audio playback.

---

## What Has Been Done

### Completed Work

1. **Dataset Construction** — Parsed AMI Meeting Corpus XML annotations, identified 149 HDMs using a three-tier keyword filter on `COMMENT-ABOUT-UNDERSTANDING` tags, with 10x negative sampling (1,490 non-HDM segments) across 75 meetings.

2. **Baseline Methods**
   - **ASR Hotword Heuristic** (`baseline_hotword.py`) — keyword-matching baseline using common HDM phrases
   - **Random Baselines** (`random_baseline.py`) — random 50/50 and base-rate classifiers for lower bounds

3. **Wav2Vec 2.0 Transfer Learning** (`wav2vec_classifier.py`) — Fine-tuned wav2vec2-base-960h with a 2-layer DNN classification head, following the paper's Method 2 hyperparameters.

4. **Audio LM Prompting** (`audio_lm_prompting.py`) — Whisper transcription + LLM classification, replicating the paper's Method 3 approach.

5. **Gemini Audio Classifier** (`gemini_audio_classifier.py`) — Direct Gemini classification baseline on individual segments.

6. **Gemini 2.5 Flash Sliding Window** (`gemini_sliding_window.py`) — The main contribution. Runs Gemini 2.5 Flash on Vertex AI with logprobs across entire meetings in 12s windows (4s context + 4s target + 4s after) at 4s step intervals. Produces a continuous P(HDM) probability signal from softmax over "P"/"N" token logprobs. Supports both **zero-shot** and **10-shot** (5P + 5N examples) prompting.
   - **Zero-shot results**: 75 meetings completed (`results/sliding_window/`)
   - **10-shot results**: 75 meetings completed (`results/sliding_window_10shot/`)
   - **Gemini 3.1 Pro backup**: 75 meetings (`results/sliding_window_gemini31pro_backup/`)

7. **Interactive Waveform Dashboard** (`waveform_dashboard.py`) — The main visualization tool, recreating Collins et al. Figure 1 with full audio playback. Features blue waveform, green probability line, red HDM bands, orange threshold, click-to-seek, audio scrubber, speed control, few-shot example panels, and per-HDM clip playback.

8. **Static Figure 1 Dashboard** (`figure1_dashboard.py`) — Generates a static Plotly HTML (`results/figure1_dashboard.html`) with the same visual style but no audio playback.

9. **Results Dashboard** (`dashboard.py`) — Trial results visualization with Plotly charts, outputs to `results/dashboard.html` and `docs/index.html`.

10. **Validation & Auditing**
    - **Validation Dashboard** (`validation_dashboard.py`) — Prediction browser with audio for inspecting individual classifications
    - **Validation Audit** (`validation_audit.py`) — Data leakage checks and cross-validation integrity verification

11. **Human Labeling Tool** (`labeling_tool.py`) — Web-based tool for manual HDM labeling with audio playback.

---

## Setup

### Prerequisites

- Python 3.12
- Google Cloud account with Vertex AI API enabled (for Gemini sliding window)
- AMI Meeting Corpus audio files (16kHz mono WAV)

### Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install google-genai numpy soundfile tqdm python-dotenv scikit-learn plotly torch transformers
```

### Vertex AI Authentication

Vertex AI is required for token-level logprobs — the standard Gemini API (Google AI Studio) does not support them.

1. Install the Google Cloud CLI:
```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh --quiet
```

2. Authenticate:
```bash
./google-cloud-sdk/bin/gcloud auth application-default login --project <YOUR_PROJECT_ID>
```

3. Ensure the Vertex AI API is enabled on your GCP project.

### Environment Variables

```bash
# .env file
GEMINI_API_KEY=...  # For Gemini baseline (Google AI Studio)
# Vertex AI auth handled via gcloud credentials
```

---

## Usage

### Running the Sliding Window Classifier

```bash
# All meetings, 10-shot (default)
python src/gemini_sliding_window.py

# Single meeting
python src/gemini_sliding_window.py --meeting ES2003b

# Zero-shot mode
python src/gemini_sliding_window.py --shots 0

# Custom step size and parallelism
python src/gemini_sliding_window.py --step 2 --workers 15
```

The classifier:
- Extracts 12s audio windows (4s context + 4s target + 4s after) at regular intervals
- Sends each window to Gemini 2.5 Flash with the paper's HDM detection prompt
- Extracts P(HDM) from the logprob distribution over "P" and "N" tokens
- Saves results incrementally per meeting — automatically resumes if interrupted

### How Logprobs Work

The model outputs a single token ("P" or "N") with log probabilities:

```python
config = types.GenerateContentConfig(
    max_output_tokens=1,
    temperature=0,
    response_logprobs=True,
    logprobs=5,
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)
```

From the response:
```
log_p = logprob for "P" token
log_n = logprob for "N" token
prob_p = softmax(log_p, log_n)  →  continuous 0-1 probability of HDM
```

### Interactive Dashboard

```bash
python src/waveform_dashboard.py
# Open http://localhost:8766
```

Features:
- **Blue waveform** — audio amplitude envelope
- **Green probability line** — continuous P(HDM) from Gemini sliding window
- **Red shaded bands** — ground truth HDM events
- **Orange dashed threshold** — decision boundary
- **Audio playback** with scrubber, click-to-seek on chart, and speed control
- **Per-HDM clip playback** — listen to individual events
- **Few-shot examples panel** — audio for each P/N example used in prompting

---

## Dataset: AMI Meeting Corpus

The [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) contains 100 hours of meeting recordings with dialogue act annotations.

- **75 meetings** with audio, **1,639 total segments** (149 positive, 1,490 negative)
- **5-fold Monte Carlo cross-validation** at the meeting level (80/20 split)
- Audio: 16kHz mono WAV (headset mix channel)

---

## Project Structure

```
src/
  gemini_sliding_window.py     # Gemini 2.5 Flash sliding window (Vertex AI logprobs)
  gemini_audio_classifier.py   # Gemini baseline classifier
  waveform_dashboard.py        # Interactive Figure 1 dashboard with audio (port 8766)
  figure1_dashboard.py         # Static Plotly Figure 1 (HTML output)
  dashboard.py                 # Trial results dashboard (HTML output)
  validation_dashboard.py      # Prediction browser with audio
  validation_audit.py          # Data leakage checks
  evaluate_all.py              # Run all methods and generate summary table
  build_dataset.py             # Dataset construction
  filter_hdm.py                # HDM annotation filtering
  labeling_tool.py             # Human labeling web tool
  audio_lm_prompting.py        # Audio LM prompting (Whisper + LLM)
  wav2vec_classifier.py        # Wav2Vec 2.0 transfer learning classifier
  baseline_hotword.py          # ASR hotword baseline
  random_baseline.py           # Random baselines
  download_audio.py            # AMI audio downloader
  parse_ami_annotations.py     # AMI XML annotation parser

results/
  sliding_window/              # Zero-shot Gemini P(HDM) per meeting (75 meetings)
  sliding_window_10shot/       # 10-shot Gemini P(HDM) per meeting (75 meetings)
  sliding_window_gemini31pro_backup/  # Gemini 3.1 Pro results backup (75 meetings)
  gemini_10shot_results.json   # Gemini baseline results
  baseline_hotword.json        # Hotword baseline results
  random_baseline.json         # Random baseline results

data/
  audio/                       # AMI meeting WAV files (16kHz mono, not tracked)
  dataset/                     # dataset_meta.json + segments (not tracked)
  hdm_annotations.json         # Parsed HDM annotations
  hdm_filtered.json            # Filtered HDM set
  labeling_clips/              # Audio clips for human labeling

docs/
  index.html                   # GitHub Pages dashboard
```

---

## Paper Reference

Collins, J., Banos, A., Culley, C., Ballesta Rosen, A., Machum, J., Lyon, R. F., & Carlile, S. (2025). *Identifying Hearing Difficulty Moments in Conversational Audio*. arXiv:2507.23590.
