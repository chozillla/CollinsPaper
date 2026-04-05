"""
Gemini Sliding Window — Continuous P(HDM) signal via Vertex AI logprobs.

Matches Collins et al. Figure 1: runs Gemini at regular intervals across
each meeting to produce a continuous probability signal.

Uses Vertex AI for logprobs support (not available on standard Gemini API).

Usage:
    python src/gemini_sliding_window.py                    # all meetings
    python src/gemini_sliding_window.py --meeting ES2003b  # one meeting
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
OUTPUT_DIR = RESULTS_DIR / "sliding_window"
SAMPLE_RATE = 16000
SEGMENT_DURATION = 4.0
CONTEXT_BEFORE = 4.0
CONTEXT_AFTER = 4.0
MAX_WORKERS = 8
PROJECT_ID = "dmt-discov-poc-prj-6258"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"

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


def _make_request(client, wav_bytes):
    """Make a single API request with logprobs."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(role="user", parts=[
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                types.Part(text="Is the speaker having a hearing difficulty moment? Answer P or N only."),
            ]),
        ],
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


def classify_window(client, target_audio):
    """Classify a single audio window using Gemini with logprobs."""
    wav_bytes = audio_to_wav_bytes(target_audio)

    try:
        return _make_request(client, wav_bytes)
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            for attempt in range(5):
                wait = (2 ** attempt) * 2
                time.sleep(wait)
                try:
                    return _make_request(client, wav_bytes)
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


def process_meeting(client, meeting_id, audio_data, step_s, output_path, max_workers):
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

    print(f"  {meeting_id}: {len(times)} new windows ({len(existing)} existing), "
          f"duration={duration:.0f}s, step={step_s}s")

    results_map = {}
    errors = 0
    pbar = tqdm(total=len(times), desc=f"  {meeting_id}", leave=False)

    def classify_at_time(t):
        clip = extract_window(audio_data, t)
        if len(clip) < SAMPLE_RATE:
            return t, 0, 0.0
        return (t,) + classify_window(client, clip)

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
                              existing, results_map)

    pbar.close()
    _save_meeting(output_path, meeting_id, duration, step_s, existing, results_map)

    total = len(existing) + len(results_map)
    print(f"  {meeting_id}: done — {total} total windows, {errors} errors")
    return total


def _save_meeting(output_path, meeting_id, duration, step_s, existing, new_results):
    all_windows = list(existing.values()) + list(new_results.values())
    all_windows.sort(key=lambda w: w["time"])
    data = {
        "meeting_id": meeting_id,
        "duration": round(duration, 2),
        "step_s": step_s,
        "n_windows": len(all_windows),
        "model": MODEL_NAME,
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
    args = parser.parse_args()

    print(f"Gemini Sliding Window HDM Classifier ({MODEL_NAME} on Vertex AI)")
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_windows = 0
    total_meetings = len(meeting_ids)
    t0 = time.time()

    for i, mid in enumerate(meeting_ids):
        print(f"\n[{i+1}/{total_meetings}] Processing {mid}...")
        audio = load_meeting_audio(mid)
        if audio is None:
            continue

        output_path = OUTPUT_DIR / f"{mid}.json"
        n = process_meeting(client, mid, audio, args.step, output_path, args.workers)
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
