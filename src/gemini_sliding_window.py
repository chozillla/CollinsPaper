"""
Gemini Sliding Window — Continuous P(HDM) signal via Vertex AI logprobs.

Matches Collins et al. Figure 1: runs Gemini at regular intervals across
each meeting to produce a continuous probability signal.

Uses Vertex AI for logprobs support (not available on standard Gemini API).
Supports 10-shot prompting (5P + 5N) matching the paper's methodology.

Usage:
    python src/gemini_sliding_window.py                    # all meetings, 10-shot
    python src/gemini_sliding_window.py --meeting ES2003b  # one meeting
    python src/gemini_sliding_window.py --shots 0          # zero-shot (no examples)
    python src/gemini_sliding_window.py --step 4           # 4s step size
"""

import gc
import io
import sys
import json
import math
import time
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

load_dotenv()

from google import genai
from google.genai import types

ROOT = Path(__file__).parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
DATASET_DIR = ROOT / "data" / "dataset"
RESULTS_DIR = ROOT / "results"
SAMPLE_RATE = 16000
SEGMENT_DURATION = 4.0
CONTEXT_BEFORE = 4.0
CONTEXT_AFTER = 4.0
MAX_WORKERS = 8
PROJECT_ID = "dmt-discov-poc-prj-6258"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"

N_POS_SHOTS = 5
N_NEG_SHOTS = 5

SYSTEM_PROMPT = """You are an expert at analyzing if a speaker in a given conversation is having difficulties understanding or hearing at a given moment. Please consider the following factors:

* **Non-semantic information:** Assess the general tone and pitch expressed in the speakers voice. Is it predominantly strained, or at ease? The lombard effect describes the things to look out for when someone is struggling in conversation:
  - increase in phonetic fundamental frequencies
  - shift in energy from low frequency bands to middle or high bands
  - increase in sound intensity
  - increase in vowel duration
  - spectral tilting
  - shift in formant center frequencies for F1 (mainly) and F2
  - the duration of content words are prolonged to a greater degree in noise than function words
  - greater lung volumes are used
* **Semantic information:** Pay attention to what they are saying and any keywords which might indicate that they are struggling to understand something. Are they asking for clarifications? Common examples to look out for (not exhaustive):
  - What?
  - Can you repeat that?
  - I didn't catch that?
  - Huh?
  - Sorry?
* **Subjectivity:** Recognize that some experiences are inherently subjective. Focus on the speaker's experience rather than your personal opinions. Do you think they are having a moment of hearing difficulty?

Use all of the context available but make your judgement only on if the current moment (ie. the end of the audio) is a hearing difficulty event or not.

Answer only with "P" for POSITIVE meaning a hearing difficulty event or "N" for NEGATIVE meaning it isn't a hearing difficulty event. Do not include any other rationale or text in your response."""

USER_QUESTION = "Is the speaker having a hearing difficulty moment? Answer P or N only."


def get_client():
    return genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def audio_to_wav_bytes(audio_array, sr=SAMPLE_RATE):
    buf = io.BytesIO()
    sf.write(buf, audio_array.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def extract_window(audio_data, center_time):
    """Extract a 12s window: 4s before + 4s target + 4s after."""
    sr = SAMPLE_RATE
    hdm_end = center_time
    hdm_start = center_time - SEGMENT_DURATION
    clip_start = hdm_start - CONTEXT_BEFORE
    clip_end = hdm_end + CONTEXT_AFTER
    start_sample = max(0, int(clip_start * sr))
    end_sample = min(len(audio_data), int(clip_end * sr))
    return audio_data[start_sample:end_sample]


# --- Few-shot support ---

def load_few_shot_pool():
    """Load all candidate examples for few-shot prompting."""
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    human_labels = {}
    labels_path = ROOT / "data" / "hdm_labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            human_labels = json.load(f)

    segments = np.load(DATASET_DIR / "audio_segments.npy")
    n_pos = len(meta["positive"])

    positives = []
    for i, ex in enumerate(meta["positive"]):
        hl = human_labels.get(str(i))
        positives.append({
            "idx": i,
            "meeting_id": ex["meeting_id"],
            "audio": segments[i],
            "label": 1,
            "verified": hl == "yes",
            "hard_negative": hl == "no",
        })

    negatives = []
    for i, ex in enumerate(meta["negative"]):
        negatives.append({
            "idx": n_pos + i,
            "meeting_id": ex["meeting_id"],
            "audio": segments[n_pos + i],
            "label": 0,
        })

    n_verified = sum(1 for p in positives if p["verified"])
    n_hard_neg = sum(1 for p in positives if p["hard_negative"])
    print(f"  Few-shot pool: {n_verified} verified positives, "
          f"{n_hard_neg} hard negatives, {len(negatives)} negatives")

    return {"positives": positives, "negatives": negatives}


def select_few_shot_examples(pool, exclude_meeting_id):
    """Select 5P + 5N examples, excluding the current meeting to avoid leakage."""
    rng = np.random.RandomState(hash(exclude_meeting_id) % (2**31))

    # Positive: prefer human-verified from other meetings
    avail_pos = [p for p in pool["positives"]
                 if p["meeting_id"] != exclude_meeting_id]
    verified = [p for p in avail_pos if p["verified"]]
    pos_pool = verified if len(verified) >= N_POS_SHOTS else avail_pos
    pos_indices = rng.choice(len(pos_pool), size=min(N_POS_SHOTS, len(pos_pool)), replace=False)
    selected_pos = [pos_pool[i] for i in pos_indices]

    # Negative: random from other meetings
    avail_neg = [n for n in pool["negatives"]
                 if n["meeting_id"] != exclude_meeting_id]
    neg_indices = rng.choice(len(avail_neg), size=min(N_NEG_SHOTS, len(avail_neg)), replace=False)
    selected_neg = [avail_neg[i] for i in neg_indices]

    # Interleave: P, N, P, N, ...
    examples = []
    pi, ni = 0, 0
    while pi < len(selected_pos) or ni < len(selected_neg):
        if pi < len(selected_pos):
            examples.append(selected_pos[pi])
            pi += 1
        if ni < len(selected_neg):
            examples.append(selected_neg[ni])
            ni += 1

    return examples


def build_few_shot_prefix(examples):
    """Build the multi-turn Content prefix for few-shot examples.

    Each example becomes a user turn (audio + question) and a model turn (P or N).
    Built once per meeting, reused for every window.
    """
    contents = []
    for ex in examples:
        wav_bytes = audio_to_wav_bytes(ex["audio"])
        label_str = "P" if ex["label"] == 1 else "N"

        contents.append(
            types.Content(role="user", parts=[
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                types.Part(text=USER_QUESTION),
            ])
        )
        contents.append(
            types.Content(role="model", parts=[
                types.Part(text=label_str),
            ])
        )

    return contents


# --- API request ---

def _make_request(client, contents):
    """Make a single API request with logprobs."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1,
            temperature=0,
            response_logprobs=True,
            logprobs=5,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    answer = (response.text or "").strip().upper()
    prediction = 1 if answer == "P" else 0

    prob_p = 0.5
    lr = response.candidates[0].logprobs_result
    if lr and lr.top_candidates:
        log_p, log_n = None, None
        for alt in lr.top_candidates[0].candidates:
            tok = alt.token.strip().upper()
            if tok == "P":
                log_p = alt.log_probability
            elif tok == "N":
                log_n = alt.log_probability
        if log_p is not None and log_n is not None:
            max_log = max(log_p, log_n)
            prob_p = math.exp(log_p - max_log) / (
                math.exp(log_p - max_log) + math.exp(log_n - max_log)
            )

    return prediction, round(prob_p, 4)


def classify_window(client, target_audio, shot_prefix=None):
    """Classify a single audio window using Gemini with logprobs."""
    wav_bytes = audio_to_wav_bytes(target_audio)

    target_content = types.Content(role="user", parts=[
        types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
        types.Part(text=USER_QUESTION),
    ])

    if shot_prefix:
        contents = list(shot_prefix) + [target_content]
    else:
        contents = [target_content]

    try:
        return _make_request(client, contents)
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            for attempt in range(5):
                wait = (2 ** attempt) * 2
                time.sleep(wait)
                try:
                    return _make_request(client, contents)
                except Exception:
                    continue
        print(f"    API error: {e}")
        return -1, 0.5


def load_meeting_audio(meeting_id):
    audio_path = AUDIO_DIR / f"{meeting_id}.Mix-Headset.wav"
    if not audio_path.exists():
        return None
    data, sr = sf.read(str(audio_path), dtype="float32")
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return data


def process_meeting(client, meeting_id, audio_data, step_s, output_path,
                    max_workers, shot_prefix=None):
    """Run sliding window across one meeting."""
    duration = len(audio_data) / SAMPLE_RATE

    # Load existing results if resuming
    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        for entry in data.get("windows", []):
            existing[entry["time"]] = entry

    # Generate window times
    times = []
    t = SEGMENT_DURATION
    while t <= duration:
        t_rounded = round(t, 1)
        if t_rounded not in existing:
            times.append(t_rounded)
        t += step_s

    if not times:
        print(f"  {meeting_id}: all {len(existing)} windows already done, skipping")
        return len(existing)

    n_shots = len(shot_prefix) // 2 if shot_prefix else 0
    print(f"  {meeting_id}: {len(times)} new windows ({len(existing)} existing), "
          f"duration={duration:.0f}s, step={step_s}s, shots={n_shots}")

    results_map = {}
    errors = 0
    pbar = tqdm(total=len(times), desc=f"  {meeting_id}", leave=False)

    def classify_at_time(t):
        clip = extract_window(audio_data, t)
        if len(clip) < SAMPLE_RATE:
            return t, 0, 0.0
        return (t,) + classify_window(client, clip, shot_prefix)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_at_time, t): t for t in times}
        for future in as_completed(futures):
            t, pred, prob = future.result()
            if pred == -1:
                errors += 1
            results_map[t] = {"time": t, "pred": pred, "prob_p": prob}
            pbar.update(1)

            # Save incrementally every 200 windows
            if len(results_map) % 200 == 0:
                _save_meeting(output_path, meeting_id, duration, step_s,
                              existing, results_map, n_shots)

    pbar.close()
    _save_meeting(output_path, meeting_id, duration, step_s,
                  existing, results_map, n_shots)

    total = len(existing) + len(results_map)
    print(f"  {meeting_id}: done — {total} total windows, {errors} errors")
    return total


def _save_meeting(output_path, meeting_id, duration, step_s,
                  existing, new_results, n_shots=0):
    all_windows = list(existing.values()) + list(new_results.values())
    all_windows.sort(key=lambda w: w["time"])
    data = {
        "meeting_id": meeting_id,
        "duration": round(duration, 2),
        "step_s": step_s,
        "n_windows": len(all_windows),
        "model": MODEL_NAME,
        "n_shots": n_shots,
        "windows": all_windows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(description="Gemini sliding window HDM classifier")
    parser.add_argument("--meeting", type=str, help="Process only this meeting ID")
    parser.add_argument("--step", type=float, default=1.0, help="Step size in seconds")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max parallel API calls")
    parser.add_argument("--shots", type=int, default=10,
                        help="Number of few-shot examples (0 for zero-shot)")
    args = parser.parse_args()

    n_shots = args.shots
    output_dir = RESULTS_DIR / ("sliding_window_10shot" if n_shots > 0
                                else "sliding_window")

    print(f"Gemini Sliding Window HDM Classifier ({MODEL_NAME} on Vertex AI)")
    print(f"Mode: {n_shots}-shot" if n_shots > 0 else "Mode: zero-shot")
    print("=" * 60)

    client = get_client()

    # Get meeting list
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)
    all_mids = sorted(set(ex["meeting_id"] for ex in meta["positive"] + meta["negative"]))

    if args.meeting:
        if args.meeting not in all_mids:
            print(f"Meeting {args.meeting} not found!")
            return
        meeting_ids = [args.meeting]
    else:
        meeting_ids = all_mids

    meeting_ids = [mid for mid in meeting_ids
                   if (AUDIO_DIR / f"{mid}.Mix-Headset.wav").exists()]

    # Load few-shot pool if using shots
    few_shot_pool = None
    if n_shots > 0:
        print("Loading few-shot example pool...")
        few_shot_pool = load_few_shot_pool()

    output_dir.mkdir(parents=True, exist_ok=True)

    total_windows = 0
    total_meetings = len(meeting_ids)
    t0 = time.time()

    for i, mid in enumerate(meeting_ids):
        print(f"\n[{i+1}/{total_meetings}] Processing {mid}...")
        audio = load_meeting_audio(mid)
        if audio is None:
            continue

        # Build few-shot prefix for this meeting (excluding its own examples)
        shot_prefix = None
        if few_shot_pool and n_shots > 0:
            examples = select_few_shot_examples(few_shot_pool, mid)
            shot_prefix = build_few_shot_prefix(examples)
            print(f"  Using {len(examples)} few-shot examples "
                  f"({sum(1 for e in examples if e['label']==1)}P + "
                  f"{sum(1 for e in examples if e['label']==0)}N)")

        output_path = output_dir / f"{mid}.json"
        n = process_meeting(client, mid, audio, args.step, output_path,
                            args.workers, shot_prefix)
        total_windows += n

        del audio
        gc.collect()

        elapsed = time.time() - t0
        rate = total_windows / elapsed if elapsed > 0 else 0
        remaining = (total_meetings - i - 1) * (elapsed / (i + 1))
        print(f"  Progress: {total_windows} windows, {rate:.1f} win/s, "
              f"~{remaining/60:.0f}m remaining")

    elapsed = time.time() - t0
    print(f"\nDone! {total_windows} windows across {total_meetings} meetings "
          f"in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
