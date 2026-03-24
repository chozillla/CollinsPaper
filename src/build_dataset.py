"""
Build the audio segment dataset for HDM classification.

Following Collins et al.:
- 4-second audio segments
- Positive: sample timestep where HDM has been occurring >= 0.4s, take preceding 4s
- Negative: random timestep with no overlap with any positive event, take preceding 4s
- 10:1 negative to positive ratio
- Conversations grouped for train/test (no leakage)
- Monte Carlo cross-validation: 5 splits, 80/20
"""

import json
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict

AUDIO_DIR = Path("data/audio")
DATASET_DIR = Path("data/dataset")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_DURATION = 4.0  # seconds
MIN_EVENT_OVERLAP = 0.4  # seconds - event must have been occurring for at least this
SAMPLE_RATE = 16000
NEG_TO_POS_RATIO = 10
NUM_SPLITS = 5
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def load_audio(meeting_id):
    """Load the headset mix audio for a meeting."""
    audio_path = AUDIO_DIR / f"{meeting_id}.Mix-Headset.wav"
    if not audio_path.exists():
        return None, None
    data, sr = sf.read(str(audio_path))
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    return data, sr


def extract_segment(audio_data, end_time_sec, duration=SEGMENT_DURATION, sr=SAMPLE_RATE):
    """Extract a segment of `duration` seconds ending at `end_time_sec`."""
    end_sample = int(end_time_sec * sr)
    start_sample = int((end_time_sec - duration) * sr)

    if start_sample < 0:
        return None

    if end_sample > len(audio_data):
        return None

    segment = audio_data[start_sample:end_sample]

    # Pad if needed
    expected_length = int(duration * sr)
    if len(segment) < expected_length:
        segment = np.pad(segment, (expected_length - len(segment), 0))

    return segment


def sample_positive_examples(hdms, audio_data, sr):
    """Sample positive (HDM) examples.

    For each HDM, sample a timestep where the event has been occurring
    for at least MIN_EVENT_OVERLAP seconds, then take the preceding 4s.
    """
    examples = []
    for hdm in hdms:
        start = hdm["start_time"]
        end = hdm["end_time"]
        duration = end - start

        # The event must be at least MIN_EVENT_OVERLAP long
        # Sample a point where event has been occurring >= 0.4s
        earliest_sample_point = start + MIN_EVENT_OVERLAP
        if earliest_sample_point > end:
            # Event too short, use the end
            sample_point = end
        else:
            sample_point = random.uniform(earliest_sample_point, end)

        # Extract 4s of audio ending at sample_point
        segment = extract_segment(audio_data, sample_point, SEGMENT_DURATION, sr)
        if segment is not None:
            examples.append({
                "meeting_id": hdm["meeting_id"],
                "speaker": hdm["speaker"],
                "sample_time": round(sample_point, 3),
                "hdm_start": start,
                "hdm_end": end,
                "text": hdm["text"],
                "label": 1,
                "audio": segment,
            })
    return examples


def sample_negative_examples(hdms, audio_data, sr, n_negatives, meeting_id):
    """Sample negative examples that don't overlap with any HDM."""
    audio_duration = len(audio_data) / sr

    # Build list of HDM intervals to avoid
    hdm_intervals = [(h["start_time"], h["end_time"]) for h in hdms]
    # Expand intervals by segment duration to avoid overlap
    forbidden = []
    for s, e in hdm_intervals:
        forbidden.append((s - SEGMENT_DURATION, e))

    def is_valid_point(t):
        """Check if sampling a 4s segment ending at t overlaps any HDM."""
        seg_start = t - SEGMENT_DURATION
        seg_end = t
        for fs, fe in forbidden:
            if seg_start < fe and seg_end > fs:
                return False
        return True

    # Sample valid negative points
    examples = []
    max_attempts = n_negatives * 50
    attempts = 0

    while len(examples) < n_negatives and attempts < max_attempts:
        t = random.uniform(SEGMENT_DURATION + 1.0, audio_duration - 1.0)
        if is_valid_point(t):
            segment = extract_segment(audio_data, t, SEGMENT_DURATION, sr)
            if segment is not None:
                examples.append({
                    "meeting_id": meeting_id,
                    "speaker": "mix",
                    "sample_time": round(t, 3),
                    "hdm_start": None,
                    "hdm_end": None,
                    "text": "",
                    "label": 0,
                    "audio": segment,
                })
        attempts += 1

    return examples


def create_splits(meetings_with_hdms, num_splits=NUM_SPLITS, train_ratio=TRAIN_RATIO):
    """Create Monte Carlo cross-validation splits at the conversation level."""
    meeting_ids = list(meetings_with_hdms.keys())
    splits = []

    rng = random.Random(RANDOM_SEED)

    for i in range(num_splits):
        rng.shuffle(meeting_ids)
        n_train = int(len(meeting_ids) * train_ratio)
        train_meetings = set(meeting_ids[:n_train])
        test_meetings = set(meeting_ids[n_train:])
        splits.append({
            "train": train_meetings,
            "test": test_meetings,
        })
    return splits


def main():
    random.seed(RANDOM_SEED)

    # Load HDMs
    with open("data/hdm_filtered.json") as f:
        all_hdms = json.load(f)

    # Group by meeting
    meetings_hdms = defaultdict(list)
    for h in all_hdms:
        meetings_hdms[h["meeting_id"]].append(h)

    print(f"Total HDMs: {len(all_hdms)}")
    print(f"Meetings: {len(meetings_hdms)}")

    # Check which meetings have audio
    available_meetings = {}
    for mid in meetings_hdms:
        audio_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        if audio_path.exists():
            available_meetings[mid] = meetings_hdms[mid]
    print(f"Meetings with audio: {len(available_meetings)}")

    if not available_meetings:
        print("ERROR: No audio files found. Run download_audio.py first.")
        return

    # Process each meeting
    all_positive = []
    all_negative = []

    for meeting_id, hdms in sorted(available_meetings.items()):
        print(f"\nProcessing {meeting_id} ({len(hdms)} HDMs)...")
        audio_data, sr = load_audio(meeting_id)
        if audio_data is None:
            print(f"  SKIP: Could not load audio")
            continue

        audio_dur = len(audio_data) / sr
        print(f"  Audio: {audio_dur:.1f}s ({sr}Hz)")

        # Sample positive examples
        positives = sample_positive_examples(hdms, audio_data, sr)
        print(f"  Positives: {len(positives)}")

        # Sample negative examples (10:1 ratio)
        n_neg = len(positives) * NEG_TO_POS_RATIO
        negatives = sample_negative_examples(hdms, audio_data, sr, n_neg, meeting_id)
        print(f"  Negatives: {len(negatives)}")

        all_positive.extend(positives)
        all_negative.extend(negatives)

    print(f"\n=== Dataset Summary ===")
    print(f"Total positive: {len(all_positive)}")
    print(f"Total negative: {len(all_negative)}")
    print(f"Ratio: 1:{len(all_negative)/max(len(all_positive),1):.1f}")

    # Create Monte Carlo CV splits
    splits = create_splits(available_meetings)

    # Save dataset metadata (without audio, which is saved separately)
    dataset_meta = {
        "positive": [{k: v for k, v in ex.items() if k != "audio"} for ex in all_positive],
        "negative": [{k: v for k, v in ex.items() if k != "audio"} for ex in all_negative],
        "splits": [
            {"train": list(s["train"]), "test": list(s["test"])}
            for s in splits
        ],
        "config": {
            "segment_duration": SEGMENT_DURATION,
            "sample_rate": SAMPLE_RATE,
            "neg_to_pos_ratio": NEG_TO_POS_RATIO,
            "num_splits": NUM_SPLITS,
            "train_ratio": TRAIN_RATIO,
        }
    }
    with open(DATASET_DIR / "dataset_meta.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    # Save audio segments as numpy arrays
    all_examples = all_positive + all_negative
    audio_segments = np.array([ex["audio"] for ex in all_examples], dtype=np.float32)
    labels = np.array([ex["label"] for ex in all_examples], dtype=np.int64)
    np.save(DATASET_DIR / "audio_segments.npy", audio_segments)
    np.save(DATASET_DIR / "labels.npy", labels)

    print(f"\nSaved to {DATASET_DIR}/")
    print(f"  audio_segments.npy: {audio_segments.shape}")
    print(f"  labels.npy: {labels.shape}")
    print(f"  dataset_meta.json")

    # Per-split statistics
    for i, split in enumerate(splits):
        pos_train = sum(1 for ex in all_positive if ex["meeting_id"] in split["train"])
        pos_test = sum(1 for ex in all_positive if ex["meeting_id"] in split["test"])
        print(f"\n  Split {i}: train_pos={pos_train}, test_pos={pos_test}, "
              f"train_meetings={len(split['train'])}, test_meetings={len(split['test'])}")


if __name__ == "__main__":
    main()
