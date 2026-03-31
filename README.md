# Replicating "Identifying Hearing Difficulty Moments in Conversational Audio"

Replication and extension of [Collins et al. (2025)](https://arxiv.org/abs/2507.23590) using the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/).

**Gemini 2.5 Flash** on Vertex AI with token-level logprobs generates a continuous P(HDM) probability signal across entire meetings, matching the paper's Figure 1 visualization with interactive audio playback.

---

## What is This Project About?

Collins et al. (2025) showed that AI models can automatically detect **Hearing Difficulty Moments (HDMs)** — moments when a listener struggles to understand what was said (e.g. "What?", "Huh?", "Sorry?"). Their best model (Gemini 1.5 Pro, 10-shot) achieved F1 = 0.87.

This project replicates and extends their work on the AMI Meeting Corpus using **Gemini 2.5 Flash** with a sliding window approach to produce a continuous probability signal, and an interactive dashboard to explore the results with audio playback.

---

## Setup: Gemini 2.5 Flash on Vertex AI

We use Gemini 2.5 Flash on Google Cloud Vertex AI to generate a **continuous P(HDM) probability signal** across entire meetings. Vertex AI is required because it is the only way to get token-level log probabilities from Gemini models — the standard Gemini API (Google AI Studio) does not support logprobs.

### Why Vertex AI Logprobs?

- **Discriminative signal**: Gemini's logprobs produce a clean probability curve — mostly near 0 with sharp spikes at HDM events (~94% of windows have prob_p < 0.1)
- **Matches the paper**: The continuous probability line is exactly what Collins et al. plot in Figure 1
- **Speed**: Gemini 2.5 Flash processes ~7 windows/sec, fast enough to cover all 75 meetings

### How to Set Up Vertex AI Auth

1. Install the Google Cloud CLI:
```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh --quiet
```

2. Authenticate with your Google Cloud project:
```bash
./google-cloud-sdk/bin/gcloud auth application-default login --project <YOUR_PROJECT_ID>
```
This opens a browser for OAuth approval. Once approved, credentials are saved locally and the Vertex AI SDK picks them up automatically.

3. Ensure the Vertex AI API is enabled on your GCP project.

### Running the Sliding Window Classifier

```bash
# All meetings (4s step, ~37,000 windows total)
python src/gemini_sliding_window.py

# Single meeting
python src/gemini_sliding_window.py --meeting ES2003b

# Custom step size and parallelism
python src/gemini_sliding_window.py --step 2 --workers 15
```

The classifier:
- Extracts 12s audio windows (4s context + 4s target + 4s after) at regular intervals
- Sends each window to Gemini 2.5 Flash with the paper's HDM detection prompt
- Extracts P(HDM) from the logprob distribution over "P" and "N" tokens
- Saves results incrementally per meeting to `results/sliding_window/`
- Automatically resumes if interrupted — skips already-processed windows

### How Logprobs Work

The model outputs a single token ("P" or "N") with log probabilities for each option:

```python
config = types.GenerateContentConfig(
    max_output_tokens=1,
    temperature=0,
    response_logprobs=True,   # Enable logprobs
    logprobs=5,               # Return top 5 alternatives
    thinking_config=types.ThinkingConfig(thinking_budget=0),  # Disable thinking
)
```

From the response, we extract:
```
log_p = logprob for "P" token
log_n = logprob for "N" token
prob_p = softmax(log_p, log_n)  # Normalized probability of HDM
```

This gives a continuous 0-1 probability signal, not just a binary P/N decision.

---

## Interactive Dashboard

The main visualization tool recreates Collins et al. Figure 1 with full audio playback:

```bash
python src/waveform_dashboard.py
# Open http://localhost:8766
```

Features:
- **Blue waveform** — audio amplitude envelope
- **Green probability line** — continuous P(HDM) from the Gemini sliding window
- **Red shaded bands** — ground truth HDM events
- **Orange dashed threshold** — decision boundary
- **Audio playback** with scrubber, click-to-seek on chart, and speed control
- **Per-HDM clip playback** — listen to individual events

The dashboard automatically loads sliding window data from `results/sliding_window/` when available.

---

## Dataset: AMI Meeting Corpus

The [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) contains 100 hours of meeting recordings with dialogue act annotations. We identified 149 HDMs using a three-tier keyword filter on the `COMMENT-ABOUT-UNDERSTANDING` tag, with 10x negative sampling (1,490 non-HDM segments).

- **75 meetings** with audio, **1,639 total segments** (149 positive, 1,490 negative)
- **5-fold Monte Carlo cross-validation** at the meeting level (80/20 split)
- Audio: 16kHz mono WAV (headset mix channel)

---

## Project Structure

```
src/
  gemini_sliding_window.py     # Gemini 2.5 Flash sliding window (Vertex AI logprobs)
  gemini_audio_classifier.py   # Gemini baseline classifier
  waveform_dashboard.py        # Interactive Figure 1 dashboard (main visualization)
  validation_dashboard.py      # Prediction browser with audio
  validation_audit.py          # Data leakage checks
  build_dataset.py             # Dataset construction
  filter_hdm.py                # HDM annotation filtering
  labeling_tool.py             # Human labeling web tool
  baseline_hotword.py          # ASR hotword baseline
  random_baseline.py           # Random baselines
  download_audio.py            # AMI audio downloader
  parse_ami_annotations.py     # AMI XML annotation parser

results/
  sliding_window/              # Gemini continuous P(HDM) per meeting
  gemini_10shot_results.json   # Gemini baseline results

data/
  audio/                       # AMI meeting WAV files (16kHz mono)
  dataset/                     # dataset_meta.json + segments
```

---

## Environment

```bash
# Python 3.12 with venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Required env vars (.env file)
GEMINI_API_KEY=...            # For Gemini baseline (Google AI Studio)
# Vertex AI auth via gcloud (for sliding window logprobs)
```

---

## Paper Reference

Collins, J., Banos, A., Culley, C., Ballesta Rosen, A., Machum, J., Lyon, R. F., & Carlile, S. (2025). *Identifying Hearing Difficulty Moments in Conversational Audio*. arXiv:2507.23590.
