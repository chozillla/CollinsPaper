# Sliding Window P(HDM) Signal — Run Log & Replication Guide

## Overview

This documents the process of generating the continuous P(HDM) probability
signal for all 75 meetings, matching the methodology in Collins et al. Figure 1.

The signal is produced by running Gemini at regular intervals across each
meeting's audio, extracting token-level logprobs for P/N, and computing a
relative P(HDM) confidence at each timestep.

## Paper Reference

Collins et al. — *Identifying Hearing Difficulty Moments in Conversational Audio*
- Figure 1 uses **Gemini 1.5 Pro** with **10-shot prompting**
- 4-second audio windows sampled every **1000ms** (1-second step)
- Probability = softmax of P vs N token logprobs

## Model Selection History

| Attempt | Model | Issue |
|---------|-------|-------|
| 1 | `gemini-3.1-pro-preview` | Only 5 discrete prob values (0.01, 0.05, 0.50, 0.95, 0.99) — binary spikes instead of smooth curve |
| 2 | `gemini-1.5-pro` | Not available on Vertex AI (deprecated) |
| 3 | `gemini-2.5-pro` | Thinking model — consumes all output tokens on internal reasoning, logprobs land on thinking tokens not P/N. `thinking_budget=0` not supported. |
| 4 | **`gemini-2.5-flash`** | Works. Granular logprobs (500+ unique values per meeting). `thinking_budget=0` supported. ~3 it/s with 4 workers. |

## Current Configuration

```
Model:           gemini-2.5-flash (Vertex AI)
Project:         dmt-discov-poc-prj-6258
Location:        us-central1
Step size:       1.0 seconds (matching paper's 1000ms)
Window:          4s target + 4s context before + 4s context after = 12s clip
Max workers:     8 (parallel API calls)
Temperature:     0
Logprobs:        Top 5, thinking_budget=0
Max output:      1 token
Prompt:          Zero-shot with detailed system prompt (same as paper)
```

## How to Run

### Full run (all 75 meetings)
```bash
python src/gemini_sliding_window.py --step 1
```

### Single meeting
```bash
python src/gemini_sliding_window.py --meeting ES2002b --step 1
```

### Resume interrupted run
The script automatically resumes — it checks existing windows in each
meeting's JSON file and only processes missing timestamps. Safe to kill
and restart.

### Background run (recommended for full dataset)
```bash
nohup python src/gemini_sliding_window.py --step 1 > results/sliding_window/run_25flash.log 2>&1 &
```

### Monitor progress
```bash
# Current meeting and progress
tail -c 200 results/sliding_window/run_25flash.log

# Count completed meetings
ls results/sliding_window/*.json | grep -v run | wc -l
```

## Probability Extraction Method

1. Send 12-second audio clip (4s context + 4s target + 4s context) to Gemini
2. Model outputs a single token (P or N)
3. Extract top-5 logprobs from the response
4. Find log_p (logprob of "P" token) and log_n (logprob of "N" token)
5. Compute `prob_p = softmax(log_p, log_n)` using the log-sum-exp trick:
   ```
   max_log = max(log_p, log_n)
   prob_p = exp(log_p - max_log) / (exp(log_p - max_log) + exp(log_n - max_log))
   ```
6. If logprobs unavailable (API error), fallback to pred=-1, prob_p=0.5

## Dashboard Smoothing

The raw prob_p signal is noisy (varies frame-to-frame). The dashboard
(`src/waveform_dashboard.py`) applies post-processing before display:

1. **Spike removal**: Windows where pred != P but prob_p > 0.8 are replaced
   with the median of their neighbors (contradictory logprob artifacts)
2. **Gaussian smoothing**: sigma=1.0 (light smoothing) applied to produce
   a smooth curve while preserving peaks above the 0.97 threshold, matching the paper's Figure 1

Raw data in JSON files is unsmoothed. Smoothing is display-only.

## Output Format

Each meeting produces `results/sliding_window/{meeting_id}.json`:
```json
{
  "meeting_id": "ES2002b",
  "duration": 2279.75,
  "step_s": 1.0,
  "n_windows": 2276,
  "model": "gemini-2.5-flash",
  "windows": [
    {"time": 4.0, "pred": 0, "prob_p": 0.0141},
    {"time": 5.0, "pred": 0, "prob_p": 0.0224},
    ...
  ]
}
```

Fields per window:
- `time`: center time in seconds
- `pred`: 1 (P/positive), 0 (N/negative), -1 (API error)
- `prob_p`: probability of positive class [0.0, 1.0]

## Run Progress (as of 2026-04-05)

**Status: IN PROGRESS** — nohup process PID 474578

### Optimization timeline
1. Initial run: 4 workers, ~2.5-3 it/s → completed 25/75 meetings
2. Restarted with 8 workers, ~4-5 it/s → processing remaining 50 meetings
3. Incremental save interval increased from 50 to 200 windows (less I/O)

### Completed (25/75):
ES2002b, ES2002c, ES2002d, ES2003b, ES2004a, ES2004c, ES2005c, ES2005d,
ES2007a, ES2007b, ES2007d, ES2008b, ES2008d, ES2009d, ES2010c, ES2010d,
ES2011b, ES2011c, ES2011d, ES2012a, ES2013a, ES2014b, ES2014d, ES2015b,
ES2015c

### Remaining (50/75):
ES2015d, IS1000a-IS1009d (20 meetings), TS3004a-TS3012d (30 meetings)

Estimated completion: ~5-6 hours at improved rate (~4-5 it/s with 8 workers)

## Previous Results

Old Gemini 3.1 Pro results (4s step, binary logprobs) are backed up in:
`results/sliding_window_gemini31pro_backup/`

## Prerequisites

```bash
pip install google-genai soundfile numpy python-dotenv tqdm
```

Vertex AI auth must be configured:
```bash
gcloud auth application-default login
```

## Cost Estimate

- ~150,000 API calls total (75 meetings x ~2000 windows each)
- Each call: 12s audio + system prompt + 1 token output
- Gemini 2.5 Flash pricing applies
