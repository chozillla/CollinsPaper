"""
Cross-Validation Pipeline — verify that human-labeled 10-shot examples generalize.

Takes human HDM labels (Type A/B) from the labeling pipeline and tests whether
using them as few-shot examples for Gemini produces reliable P(HDM) signals
on held-out meetings the model has never seen.

Process:
  1. Load human labels from data/human_hdm_labels.json
  2. Split meetings into K folds (meeting-level, no leakage)
  3. For each fold:
     - Select 10-shot examples (5P + 5N) from training meetings' human labels
     - Run Gemini sliding window on each test meeting
     - Score: does the P(HDM) signal peak near human-labeled HDM timestamps?
  4. Report per-fold and aggregate metrics

Usage:
    python src/cross_validate_human_labels.py                  # 5-fold CV
    python src/cross_validate_human_labels.py --folds 3        # 3-fold
    python src/cross_validate_human_labels.py --labeler alice   # use specific labeler
    python src/cross_validate_human_labels.py --dry-run         # show splits, no API calls
"""

import gc
import io
import json
import math
import time
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

ROOT = Path(__file__).parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
HUMAN_LABELS_FILE = ROOT / "data" / "human_hdm_labels.json"
RESULTS_DIR = ROOT / "results" / "cross_validation"
SAMPLE_RATE = 16000
SEGMENT_DURATION = 4.0
CONTEXT_BEFORE = 4.0
CONTEXT_AFTER = 4.0
MAX_WORKERS = 8
PROJECT_ID = "dmt-discov-poc-prj-6258"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"
RANDOM_SEED = 42

N_POS_SHOTS = 5
N_NEG_SHOTS = 5

SYSTEM_PROMPT = """You are an expert listener analysing a recording from the AMI Meeting Corpus.
Your job is to determine if the audio contains a Hearing Difficulty Moment (HDM) — a moment
where a listener struggles to understand what was said.

Listen for:
* Clarification requests ("What?", "Sorry?", "Huh?", "Pardon?")
* Explicit non-understanding ("I didn't catch that", "Which was that?")
* Requests for repetition ("Could you say that again?", "Can you repeat?")
* Confused responses suggesting mishearing or lack of comprehension

Respond with a single letter:
P — if this segment contains a hearing difficulty moment
N — if this segment does NOT contain a hearing difficulty moment"""

USER_QUESTION = "Does this audio contain a Hearing Difficulty Moment? Respond P or N."


def get_client():
    return genai.Client(
        vertexai=True, project=PROJECT_ID, location=LOCATION
    )


def audio_to_wav_bytes(audio_array):
    buf = io.BytesIO()
    sf.write(buf, audio_array, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def extract_window(audio, center_time):
    """Extract a 12s window (4s context + 4s target + 4s after) centered on target."""
    sr = SAMPLE_RATE
    target_start = center_time - SEGMENT_DURATION
    window_start = target_start - CONTEXT_BEFORE
    window_end = center_time + CONTEXT_AFTER

    s = max(0, int(window_start * sr))
    e = min(len(audio), int(window_end * sr))
    return audio[s:e]


def load_human_labels(labeler="default"):
    """Load human labels and group by meeting."""
    if not HUMAN_LABELS_FILE.exists():
        print(f"ERROR: No human labels found at {HUMAN_LABELS_FILE}")
        print("Run the human labeling pipeline first:")
        print("  python src/human_labeling_pipeline.py --labeler <name>")
        return None

    with open(HUMAN_LABELS_FILE) as f:
        all_labels = json.load(f)

    if labeler not in all_labels:
        available = list(all_labels.keys())
        print(f"ERROR: Labeler '{labeler}' not found. Available: {available}")
        return None

    labeler_data = all_labels[labeler]

    # Only include meetings that have labels and audio
    meetings = {}
    for mid, hdms in labeler_data.items():
        if not hdms:
            continue
        audio_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        if audio_path.exists():
            meetings[mid] = hdms

    return meetings


def create_folds(meeting_ids, n_folds, seed=RANDOM_SEED):
    """Split meetings into K folds for cross-validation."""
    rng = np.random.RandomState(seed)
    shuffled = list(meeting_ids)
    rng.shuffle(shuffled)

    folds = [[] for _ in range(n_folds)]
    for i, mid in enumerate(shuffled):
        folds[i % n_folds].append(mid)

    return folds


def extract_hdm_clip(audio, hdm_time):
    """Extract the 12s audio clip around an HDM timestamp."""
    return extract_window(audio, hdm_time)


def extract_negative_clip(audio, duration, hdm_times, rng):
    """Extract a random 12s clip that doesn't overlap with any HDM."""
    margin = SEGMENT_DURATION + CONTEXT_BEFORE
    for _ in range(50):
        t = rng.uniform(SEGMENT_DURATION, duration)
        if all(abs(t - h) > margin for h in hdm_times):
            return extract_window(audio, t), t
    return None, None


def select_shots_from_human_labels(labeled_meetings, exclude_mids, rng):
    """Select 5P + 5N few-shot examples from training meetings' human labels."""

    # Collect all positive examples from training meetings
    pos_candidates = []
    for mid, hdms in labeled_meetings.items():
        if mid in exclude_mids:
            continue
        audio_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        if not audio_path.exists():
            continue
        for hdm in hdms:
            pos_candidates.append({
                "meeting_id": mid,
                "time": hdm["time"],
                "type": hdm["type"],
                "label": 1,
            })

    if len(pos_candidates) < N_POS_SHOTS:
        print(f"  WARNING: only {len(pos_candidates)} positive examples available")

    # Select positive shots
    n_pos = min(N_POS_SHOTS, len(pos_candidates))
    pos_indices = rng.choice(len(pos_candidates), size=n_pos, replace=False)
    selected_pos = [pos_candidates[i] for i in pos_indices]

    # Select negative shots — random non-HDM clips from training meetings
    neg_candidates = []
    train_mids = [mid for mid in labeled_meetings if mid not in exclude_mids]
    for mid in train_mids:
        hdm_times = [h["time"] for h in labeled_meetings[mid]]
        audio_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        info = sf.info(str(audio_path))
        neg_candidates.append({
            "meeting_id": mid,
            "duration": info.duration,
            "hdm_times": hdm_times,
        })

    selected_neg = []
    neg_rng = np.random.RandomState(rng.randint(2**31))
    attempts = 0
    while len(selected_neg) < N_NEG_SHOTS and attempts < 100:
        cand = neg_candidates[neg_rng.randint(len(neg_candidates))]
        mid = cand["meeting_id"]
        audio, sr = sf.read(str(AUDIO_DIR / f"{mid}.Mix-Headset.wav"), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        clip, t = extract_negative_clip(audio, cand["duration"], cand["hdm_times"], neg_rng)
        del audio
        if clip is not None:
            selected_neg.append({
                "meeting_id": mid,
                "time": t,
                "type": None,
                "label": 0,
                "_audio": clip,
            })
        attempts += 1

    # Load audio for positive shots
    audio_cache = {}
    for shot in selected_pos:
        mid = shot["meeting_id"]
        if mid not in audio_cache:
            audio, sr = sf.read(str(AUDIO_DIR / f"{mid}.Mix-Headset.wav"), dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio_cache[mid] = audio
        shot["_audio"] = extract_hdm_clip(audio_cache[mid], shot["time"])

    for a in audio_cache.values():
        del a
    gc.collect()

    # Interleave P and N
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
    """Build multi-turn Content prefix from human-labeled examples."""
    contents = []
    for ex in examples:
        wav_bytes = audio_to_wav_bytes(ex["_audio"])
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


def classify_window(client, target_audio, shot_prefix):
    """Classify a single window via Gemini with logprobs."""
    wav_bytes = audio_to_wav_bytes(target_audio)
    target_content = types.Content(role="user", parts=[
        types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
        types.Part(text=USER_QUESTION),
    ])
    contents = list(shot_prefix) + [target_content]

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

    return round(prob_p, 4)


def score_meeting(windows, human_hdms, tolerance=8.0):
    """Score how well the P(HDM) signal aligns with human labels.

    For each human-labeled HDM, check if there's a probability peak nearby.
    Returns detection rate, mean peak probability, and false alarm rate.
    """
    times = np.array([w["time"] for w in windows])
    probs = np.array([w["prob_p"] for w in windows])
    hdm_times = [h["time"] for h in human_hdms]

    # Detection: does the signal peak near each human HDM?
    detected = 0
    peak_probs = []
    for ht in hdm_times:
        mask = np.abs(times - ht) <= tolerance
        if mask.any():
            peak = probs[mask].max()
            peak_probs.append(peak)
            if peak > 0.5:
                detected += 1

    detection_rate = detected / len(hdm_times) if hdm_times else 0
    mean_peak = np.mean(peak_probs) if peak_probs else 0

    # False alarm: high probability far from any HDM
    non_hdm_mask = np.ones(len(times), dtype=bool)
    for ht in hdm_times:
        non_hdm_mask &= np.abs(times - ht) > tolerance
    non_hdm_probs = probs[non_hdm_mask]
    false_alarm_rate = (non_hdm_probs > 0.5).mean() if len(non_hdm_probs) > 0 else 0

    noise_floor = non_hdm_probs.mean() if len(non_hdm_probs) > 0 else 0

    return {
        "n_hdms": len(hdm_times),
        "detected": detected,
        "detection_rate": round(detection_rate, 3),
        "mean_peak_prob": round(float(mean_peak), 3),
        "false_alarm_rate": round(float(false_alarm_rate), 3),
        "noise_floor": round(float(noise_floor), 3),
    }


def run_fold(client, fold_idx, train_mids, test_mids, labeled_meetings,
             step_s, max_workers):
    """Run one fold of cross-validation."""
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}: {len(train_mids)} train, {len(test_mids)} test")
    print(f"  Train: {', '.join(sorted(train_mids))}")
    print(f"  Test:  {', '.join(sorted(test_mids))}")

    # Select few-shot examples from training meetings
    rng = np.random.RandomState(RANDOM_SEED + fold_idx)
    examples = select_shots_from_human_labels(labeled_meetings, set(test_mids), rng)
    n_pos = sum(1 for e in examples if e["label"] == 1)
    n_neg = sum(1 for e in examples if e["label"] == 0)
    print(f"  Shots: {n_pos}P + {n_neg}N from training meetings")

    shot_prefix = build_few_shot_prefix(examples)

    # Clean up audio from examples
    for ex in examples:
        del ex["_audio"]
    gc.collect()

    # Run on each test meeting
    fold_results = []
    for mid in sorted(test_mids):
        print(f"\n  Testing {mid}...")
        audio, sr = sf.read(str(AUDIO_DIR / f"{mid}.Mix-Headset.wav"), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        duration = len(audio) / sr

        # Generate windows
        windows = []
        t = SEGMENT_DURATION
        while t <= duration:
            windows.append(round(t, 1))
            t += step_s

        print(f"    {len(windows)} windows, {duration:.0f}s duration")

        # Classify each window
        results = []
        errors = 0
        pbar = tqdm(total=len(windows), desc=f"    {mid}", leave=False)

        def classify_at(t):
            clip = extract_window(audio, t)
            if len(clip) < SAMPLE_RATE:
                return t, 0.5
            try:
                prob = classify_window(client, clip, shot_prefix)
                return t, prob
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    for attempt in range(3):
                        time.sleep(2 ** attempt * 2)
                        try:
                            prob = classify_window(client, clip, shot_prefix)
                            return t, prob
                        except Exception:
                            continue
                return t, 0.5

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(classify_at, t): t for t in windows}
            for future in as_completed(futures):
                t, prob = future.result()
                results.append({"time": t, "prob_p": prob})
                pbar.update(1)
        pbar.close()

        results.sort(key=lambda w: w["time"])

        # Score against human labels
        human_hdms = labeled_meetings[mid]
        scores = score_meeting(results, human_hdms)
        scores["meeting_id"] = mid
        scores["fold"] = fold_idx

        print(f"    Detection: {scores['detected']}/{scores['n_hdms']} "
              f"({scores['detection_rate']:.0%}), "
              f"peak={scores['mean_peak_prob']:.2f}, "
              f"FAR={scores['false_alarm_rate']:.1%}")

        fold_results.append({
            "meeting_id": mid,
            "scores": scores,
            "windows": results,
            "human_hdms": human_hdms,
            "examples_used": [
                {"meeting_id": e["meeting_id"], "time": e["time"],
                 "type": e["type"], "label": e["label"]}
                for e in examples
            ],
        })

        del audio
        gc.collect()

    return fold_results


def print_summary(all_results):
    """Print cross-validation summary table."""
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Meeting':<12} {'Fold':>4} {'HDMs':>5} {'Det':>4} {'Rate':>6} "
          f"{'Peak':>6} {'FAR':>6} {'Noise':>6}")
    print("-" * 70)

    all_scores = []
    for fold_results in all_results:
        for r in fold_results:
            s = r["scores"]
            all_scores.append(s)
            print(f"{s['meeting_id']:<12} {s['fold']+1:>4} {s['n_hdms']:>5} "
                  f"{s['detected']:>4} {s['detection_rate']:>6.0%} "
                  f"{s['mean_peak_prob']:>6.2f} {s['false_alarm_rate']:>6.1%} "
                  f"{s['noise_floor']:>6.2f}")

    print("-" * 70)
    n = len(all_scores)
    if n > 0:
        avg_det = np.mean([s["detection_rate"] for s in all_scores])
        avg_peak = np.mean([s["mean_peak_prob"] for s in all_scores])
        avg_far = np.mean([s["false_alarm_rate"] for s in all_scores])
        avg_noise = np.mean([s["noise_floor"] for s in all_scores])
        total_hdms = sum(s["n_hdms"] for s in all_scores)
        total_det = sum(s["detected"] for s in all_scores)
        print(f"{'MEAN':<12} {'':>4} {total_hdms:>5} {total_det:>4} {avg_det:>6.0%} "
              f"{avg_peak:>6.2f} {avg_far:>6.1%} {avg_noise:>6.2f}")

    return all_scores


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate human-labeled 10-shot examples")
    parser.add_argument("--labeler", default="default", help="Labeler name")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--step", type=float, default=4.0, help="Window step size (seconds)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel API calls")
    parser.add_argument("--dry-run", action="store_true", help="Show splits only, no API calls")
    args = parser.parse_args()

    print("Cross-Validation: Human-Labeled 10-Shot Examples")
    print(f"Labeler: {args.labeler} | Folds: {args.folds} | Step: {args.step}s")
    print("=" * 60)

    # Load human labels
    labeled_meetings = load_human_labels(args.labeler)
    if labeled_meetings is None:
        return

    meeting_ids = sorted(labeled_meetings.keys())
    total_hdms = sum(len(hdms) for hdms in labeled_meetings.values())
    print(f"Loaded {total_hdms} human HDM labels across {len(meeting_ids)} meetings")

    # Create folds
    folds = create_folds(meeting_ids, args.folds)

    if args.dry_run:
        print(f"\nDRY RUN — fold splits:")
        for i, fold in enumerate(folds):
            test_mids = fold
            train_mids = [m for m in meeting_ids if m not in test_mids]
            train_hdms = sum(len(labeled_meetings[m]) for m in train_mids)
            test_hdms = sum(len(labeled_meetings[m]) for m in test_mids)
            print(f"\n  Fold {i+1}: {len(train_mids)} train ({train_hdms} HDMs) → "
                  f"{len(test_mids)} test ({test_hdms} HDMs)")
            print(f"    Test: {', '.join(sorted(test_mids))}")
        return

    # Run cross-validation
    client = get_client()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    t0 = time.time()

    for i, fold in enumerate(folds):
        test_mids = fold
        train_mids = [m for m in meeting_ids if m not in test_mids]
        fold_results = run_fold(client, i, train_mids, test_mids,
                                labeled_meetings, args.step, args.workers)
        all_results.append(fold_results)

        # Save fold results incrementally
        fold_path = RESULTS_DIR / f"fold_{i+1}.json"
        with open(fold_path, "w") as f:
            json.dump(fold_results, f, indent=2)

    elapsed = time.time() - t0
    all_scores = print_summary(all_results)

    # Save aggregate results
    summary = {
        "labeler": args.labeler,
        "n_folds": args.folds,
        "n_meetings": len(meeting_ids),
        "total_hdms": total_hdms,
        "step_s": args.step,
        "model": MODEL_NAME,
        "elapsed_s": round(elapsed, 1),
        "per_meeting": all_scores,
        "aggregate": {
            "mean_detection_rate": round(np.mean([s["detection_rate"] for s in all_scores]), 3),
            "mean_peak_prob": round(np.mean([s["mean_peak_prob"] for s in all_scores]), 3),
            "mean_false_alarm_rate": round(np.mean([s["false_alarm_rate"] for s in all_scores]), 3),
            "mean_noise_floor": round(np.mean([s["noise_floor"] for s in all_scores]), 3),
        },
    }
    with open(RESULTS_DIR / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
