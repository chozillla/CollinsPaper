"""
Import human labels from Excel/CSV and run Gemini 10-shot on meetings.

Reads a spreadsheet with human HDM annotations (timestamp + Type A/B),
uses them as 10-shot training examples, and runs the Gemini sliding window
classifier on target meetings.

Expected spreadsheet columns:
    meeting_id  — e.g. "ES2002b"
    timestamp   — seconds into the meeting, e.g. 19.66
    type        — "A" or "B"
    note        — (optional) annotator note

Supported formats: .xlsx, .xls, .csv

Usage:
    # Import labels and run on all other meetings
    python src/import_labels_and_run.py data/labels.xlsx

    # Import and run on specific meetings
    python src/import_labels_and_run.py data/labels.xlsx --target ES2003b ES2004a

    # Import only (save to human_hdm_labels.json, don't run Gemini)
    python src/import_labels_and_run.py data/labels.xlsx --import-only

    # Specify labeler name and sheet
    python src/import_labels_and_run.py data/labels.xlsx --labeler alice --sheet "Sheet1"

    # Preview what would happen
    python src/import_labels_and_run.py data/labels.xlsx --dry-run
"""

import gc
import io
import json
import math
import time
import argparse
import numpy as np
import pandas as pd
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
    sr = SAMPLE_RATE
    target_start = center_time - SEGMENT_DURATION
    window_start = target_start - CONTEXT_BEFORE
    window_end = center_time + CONTEXT_AFTER
    s = max(0, int(window_start * sr))
    e = min(len(audio), int(window_end * sr))
    return audio[s:e]


# ── Read spreadsheet ──────────────────────────────────────────────────

def read_spreadsheet(path, sheet=None):
    """Read Excel or CSV into a list of {meeting_id, timestamp, type, note}."""
    path = Path(path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return None

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        kwargs = {"sheet_name": sheet} if sheet else {}
        df = pd.read_excel(path, **kwargs)
    else:
        print(f"ERROR: Unsupported format '{ext}'. Use .xlsx, .xls, or .csv")
        return None

    # Normalize column names — flexible matching
    col_map = {}
    for col in df.columns:
        lower = str(col).strip().lower().replace(" ", "_")
        if "meeting" in lower or lower == "id":
            col_map[col] = "meeting_id"
        elif "time" in lower or lower == "timestamp" or lower == "ts":
            col_map[col] = "timestamp"
        elif lower in ("type", "hdm_type", "label", "category"):
            col_map[col] = "type"
        elif "note" in lower or "comment" in lower:
            col_map[col] = "note"

    df = df.rename(columns=col_map)

    # Validate required columns
    missing = [c for c in ["meeting_id", "timestamp", "type"] if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"  Found columns: {list(df.columns)}")
        print(f"\n  Expected columns (flexible naming):")
        print(f"    meeting_id  — meeting ID (e.g. 'ES2002b')")
        print(f"    timestamp   — seconds into meeting (e.g. 19.66)")
        print(f"    type        — 'A' or 'B'")
        print(f"    note        — (optional) annotator comment")
        return None

    # Clean and validate
    df["meeting_id"] = df["meeting_id"].astype(str).str.strip()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["type"] = df["type"].astype(str).str.strip().str.upper()

    # Drop rows with invalid data
    before = len(df)
    df = df.dropna(subset=["meeting_id", "timestamp"])
    df = df[df["type"].isin(["A", "B"])]
    after = len(df)
    if after < before:
        print(f"  Dropped {before - after} invalid rows (missing data or type not A/B)")

    if "note" not in df.columns:
        df["note"] = ""
    df["note"] = df["note"].fillna("").astype(str)

    labels = []
    for _, row in df.iterrows():
        labels.append({
            "meeting_id": row["meeting_id"],
            "time": round(float(row["timestamp"]), 2),
            "type": row["type"],
            "note": row["note"],
        })

    return labels


# ── Save to human_hdm_labels.json ────────────────────────────────────

def save_to_labels_file(labels, labeler="default"):
    """Merge imported labels into human_hdm_labels.json."""
    existing = {}
    if HUMAN_LABELS_FILE.exists():
        with open(HUMAN_LABELS_FILE) as f:
            existing = json.load(f)

    if labeler not in existing:
        existing[labeler] = {}

    # Group by meeting
    by_meeting = defaultdict(list)
    for lbl in labels:
        by_meeting[lbl["meeting_id"]].append({
            "time": lbl["time"],
            "type": lbl["type"],
            "note": lbl["note"],
        })

    # Merge — replace per meeting (new import overwrites old labels for that meeting)
    for mid, hdms in by_meeting.items():
        existing[labeler][mid] = hdms

    with open(HUMAN_LABELS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    return by_meeting


# ── Build 10-shot examples from imported labels ──────────────────────

def build_shots_from_labels(labels_by_meeting, exclude_mids=None):
    """Select 5P + 5N examples from the imported human labels."""
    exclude_mids = exclude_mids or set()
    rng = np.random.RandomState(42)

    # Collect positive candidates
    pos_candidates = []
    for mid, hdms in labels_by_meeting.items():
        if mid in exclude_mids:
            continue
        audio_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        if not audio_path.exists():
            continue
        for hdm in hdms:
            pos_candidates.append({"meeting_id": mid, **hdm, "label": 1})

    if not pos_candidates:
        print("ERROR: No positive examples available for shot selection")
        return None

    # Select positives
    n_pos = min(N_POS_SHOTS, len(pos_candidates))
    idx = rng.choice(len(pos_candidates), size=n_pos, replace=False)
    selected_pos = [pos_candidates[i] for i in idx]

    # Load audio for positives
    audio_cache = {}
    for shot in selected_pos:
        mid = shot["meeting_id"]
        if mid not in audio_cache:
            audio, sr = sf.read(str(AUDIO_DIR / f"{mid}.Mix-Headset.wav"), dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio_cache[mid] = audio
        shot["_audio"] = extract_window(audio_cache[mid], shot["time"])

    # Select negatives — random non-HDM clips
    selected_neg = []
    train_mids = [m for m in labels_by_meeting if m not in exclude_mids
                  and (AUDIO_DIR / f"{m}.Mix-Headset.wav").exists()]

    for mid in train_mids:
        if mid not in audio_cache:
            audio, sr = sf.read(str(AUDIO_DIR / f"{mid}.Mix-Headset.wav"), dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio_cache[mid] = audio

    neg_rng = np.random.RandomState(43)
    attempts = 0
    while len(selected_neg) < N_NEG_SHOTS and attempts < 100:
        mid = train_mids[neg_rng.randint(len(train_mids))]
        audio = audio_cache[mid]
        duration = len(audio) / SAMPLE_RATE
        hdm_times = [h["time"] for h in labels_by_meeting[mid]]
        margin = SEGMENT_DURATION + CONTEXT_BEFORE

        t = neg_rng.uniform(SEGMENT_DURATION, duration)
        if all(abs(t - h) > margin for h in hdm_times):
            clip = extract_window(audio, t)
            if len(clip) >= SAMPLE_RATE:
                selected_neg.append({
                    "meeting_id": mid, "time": round(t, 2),
                    "type": None, "label": 0, "_audio": clip,
                })
        attempts += 1

    # Clean up audio cache
    for k in list(audio_cache.keys()):
        del audio_cache[k]
    gc.collect()

    # Interleave
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
    """Build multi-turn Content prefix for Gemini."""
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


# ── Run sliding window on target meetings ─────────────────────────────

def classify_window(client, target_audio, shot_prefix):
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


def run_meeting(client, mid, shot_prefix, step_s, max_workers, output_dir):
    """Run sliding window on a single meeting."""
    audio_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr

    output_path = output_dir / f"{mid}.json"

    # Check for existing results
    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        for w in data.get("windows", []):
            existing[w["time"]] = w

    # Generate window times
    times = []
    t = SEGMENT_DURATION
    while t <= duration:
        t_rounded = round(t, 1)
        if t_rounded not in existing:
            times.append(t_rounded)
        t += step_s

    if not times:
        print(f"  {mid}: all {len(existing)} windows done, skipping")
        del audio
        return len(existing)

    print(f"  {mid}: {len(times)} new windows ({len(existing)} existing), "
          f"{duration:.0f}s")

    results_map = {}
    pbar = tqdm(total=len(times), desc=f"  {mid}", leave=False)

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
                        return t, classify_window(client, clip, shot_prefix)
                    except Exception:
                        continue
            return t, 0.5

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_at, t): t for t in times}
        for future in as_completed(futures):
            t, prob = future.result()
            results_map[t] = {"time": t, "prob_p": prob}
            pbar.update(1)

            if len(results_map) % 200 == 0:
                _save(output_path, mid, duration, step_s, existing, results_map)

    pbar.close()
    _save(output_path, mid, duration, step_s, existing, results_map)

    total = len(existing) + len(results_map)
    print(f"  {mid}: done — {total} windows")

    del audio
    gc.collect()
    return total


def _save(output_path, mid, duration, step_s, existing, new_results):
    all_windows = list(existing.values()) + list(new_results.values())
    all_windows.sort(key=lambda w: w["time"])
    data = {
        "meeting_id": mid,
        "duration": round(duration, 2),
        "step_s": step_s,
        "n_windows": len(all_windows),
        "model": MODEL_NAME,
        "source": "human_labels_10shot",
        "windows": all_windows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Import human labels from Excel/CSV and run Gemini 10-shot")
    parser.add_argument("file", help="Path to Excel (.xlsx) or CSV (.csv) file")
    parser.add_argument("--labeler", default="default", help="Labeler name for storage")
    parser.add_argument("--sheet", default=None, help="Excel sheet name (default: first)")
    parser.add_argument("--target", nargs="*", help="Specific meeting IDs to run on")
    parser.add_argument("--step", type=float, default=4.0, help="Window step size (seconds)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel API calls")
    parser.add_argument("--import-only", action="store_true", help="Import labels only, skip Gemini")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    args = parser.parse_args()

    print("Import Human Labels & Run Gemini 10-Shot")
    print("=" * 55)

    # Step 1: Read spreadsheet
    print(f"\nReading {args.file}...")
    labels = read_spreadsheet(args.file, sheet=args.sheet)
    if labels is None:
        return

    # Summarize
    by_meeting = defaultdict(list)
    type_counts = {"A": 0, "B": 0}
    for lbl in labels:
        by_meeting[lbl["meeting_id"]].append(lbl)
        type_counts[lbl["type"]] += 1

    print(f"  {len(labels)} labels across {len(by_meeting)} meetings")
    print(f"  Type A (acoustic): {type_counts['A']}")
    print(f"  Type B (comprehension): {type_counts['B']}")
    print(f"  Meetings: {', '.join(sorted(by_meeting.keys()))}")

    if args.dry_run:
        print(f"\nDRY RUN — would save to {HUMAN_LABELS_FILE} under labeler '{args.labeler}'")
        if not args.import_only:
            labeled_mids = set(by_meeting.keys())
            if args.target:
                target_mids = args.target
            else:
                all_audio = {p.name.replace(".Mix-Headset.wav", "")
                             for p in AUDIO_DIR.glob("*.Mix-Headset.wav")}
                target_mids = sorted(all_audio - labeled_mids)
            print(f"  Would run Gemini on {len(target_mids)} meetings")
            print(f"  Using {min(N_POS_SHOTS, len(labels))}P + {N_NEG_SHOTS}N shots "
                  f"from {len(by_meeting)} labeled meetings")
        return

    # Step 2: Save to human_hdm_labels.json
    print(f"\nSaving labels to {HUMAN_LABELS_FILE} (labeler: {args.labeler})...")
    labels_by_meeting = save_to_labels_file(labels, args.labeler)
    print(f"  Saved {len(labels)} labels for {len(labels_by_meeting)} meetings")

    if args.import_only:
        print("\nDone (import only).")
        return

    # Step 3: Determine target meetings
    labeled_mids = set(labels_by_meeting.keys())
    if args.target:
        target_mids = [m for m in args.target
                       if (AUDIO_DIR / f"{m}.Mix-Headset.wav").exists()]
    else:
        # Run on all meetings with audio that aren't in the labeled set
        all_audio = sorted(
            p.name.replace(".Mix-Headset.wav", "")
            for p in AUDIO_DIR.glob("*.Mix-Headset.wav")
        )
        target_mids = [m for m in all_audio if m not in labeled_mids]

    print(f"\nTarget: {len(target_mids)} meetings to classify")

    # Step 4: Build 10-shot examples from human labels
    print("\nBuilding 10-shot examples from human labels...")
    examples = build_shots_from_labels(labels_by_meeting, exclude_mids=set(target_mids))
    if examples is None:
        return

    n_pos = sum(1 for e in examples if e["label"] == 1)
    n_neg = sum(1 for e in examples if e["label"] == 0)
    print(f"  Selected {n_pos}P + {n_neg}N examples:")
    for ex in examples:
        tag = f"Type {ex['type']}" if ex["type"] else "Negative"
        print(f"    {ex['meeting_id']} @ {ex['time']:.1f}s — {tag}")

    shot_prefix = build_few_shot_prefix(examples)

    # Clean up example audio
    for ex in examples:
        if "_audio" in ex:
            del ex["_audio"]
    gc.collect()

    # Step 5: Run Gemini on target meetings
    output_dir = RESULTS_DIR / "human_labels_10shot"
    output_dir.mkdir(parents=True, exist_ok=True)
    client = get_client()

    print(f"\nRunning Gemini sliding window ({args.step}s step)...")
    print(f"Output: {output_dir}/")
    t0 = time.time()
    total_windows = 0

    for i, mid in enumerate(target_mids):
        print(f"\n[{i+1}/{len(target_mids)}] {mid}")
        n = run_meeting(client, mid, shot_prefix, args.step, args.workers, output_dir)
        total_windows += n

    elapsed = time.time() - t0
    print(f"\nDone! {total_windows} windows across {len(target_mids)} meetings "
          f"in {elapsed/60:.1f} minutes")
    print(f"Results: {output_dir}/")


if __name__ == "__main__":
    main()
