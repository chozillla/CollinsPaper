"""
ASR Hotword Heuristic Baseline (Method 1 from Collins et al.)

1. Transcribe audio segments using Whisper (open-source equivalent of Chirp 2)
2. Check for hearing difficulty hotwords in transcription
3. Evaluate with Monte Carlo cross-validation
"""

import json
import numpy as np
import whisper
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_recall_curve, classification_report
from tqdm import tqdm

DATASET_DIR = Path("data/dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000

# Hearing difficulty hotwords (from Collins et al.)
HOTWORDS = {
    "what", "huh", "pardon", "sorry", "excuse", "repeat",
    "say again", "didn't catch", "didn't hear", "didn't get",
    "can't hear", "couldn't hear", "come again",
}


def transcribe_segments(audio_segments, model):
    """Transcribe audio segments using Whisper."""
    transcriptions = []
    for i in tqdm(range(len(audio_segments)), desc="Transcribing"):
        audio = audio_segments[i].astype(np.float32)
        # Whisper expects float32 audio
        result = model.transcribe(audio, language="en", fp16=False)
        transcriptions.append(result["text"].strip().lower())
    return transcriptions


def hotword_predict(transcription):
    """Check if transcription contains hearing difficulty hotwords."""
    text = transcription.lower()
    for hotword in HOTWORDS:
        if hotword in text:
            return 1
    return 0


def evaluate_split(audio_segments, labels, meta, split_idx, whisper_model):
    """Evaluate on one Monte Carlo CV split."""
    split = meta["splits"][split_idx]
    train_meetings = set(split["train"])
    test_meetings = set(split["test"])

    all_examples = meta["positive"] + meta["negative"]

    # Get test indices
    test_indices = [
        i for i, ex in enumerate(all_examples)
        if ex["meeting_id"] in test_meetings
    ]

    test_audio = audio_segments[test_indices]
    test_labels = labels[test_indices]

    # Transcribe test audio
    transcriptions = transcribe_segments(test_audio, whisper_model)

    # Predict using hotword heuristic
    predictions = [hotword_predict(t) for t in transcriptions]

    f1 = f1_score(test_labels, predictions, zero_division=0)

    return {
        "split": split_idx,
        "f1": f1,
        "n_test": len(test_indices),
        "n_pos": int(test_labels.sum()),
        "n_pred_pos": sum(predictions),
        "report": classification_report(test_labels, predictions, output_dict=True, zero_division=0),
    }


def main():
    print("Loading dataset...")
    audio_segments = np.load(DATASET_DIR / "audio_segments.npy")
    labels = np.load(DATASET_DIR / "labels.npy")
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    print(f"Dataset: {len(audio_segments)} segments, {labels.sum()} positive")

    print("\nLoading Whisper model (base)...")
    whisper_model = whisper.load_model("base")

    results = []
    for split_idx in range(len(meta["splits"])):
        print(f"\n--- Split {split_idx + 1}/{len(meta['splits'])} ---")
        result = evaluate_split(audio_segments, labels, meta, split_idx, whisper_model)
        results.append(result)
        print(f"F1: {result['f1']:.4f} (test={result['n_test']}, pos={result['n_pos']}, pred_pos={result['n_pred_pos']})")

    avg_f1 = np.mean([r["f1"] for r in results])
    std_f1 = np.std([r["f1"] for r in results])
    print(f"\n=== ASR Hotword Baseline ===")
    print(f"Average F1: {avg_f1:.4f} (+/- {std_f1:.4f})")

    # Save results
    output = {
        "method": "ASR Hotword Heuristic (Whisper base)",
        "avg_f1": avg_f1,
        "std_f1": std_f1,
        "splits": [{k: v for k, v in r.items()} for r in results],
    }
    with open(RESULTS_DIR / "baseline_hotword.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results saved to {RESULTS_DIR}/baseline_hotword.json")


if __name__ == "__main__":
    main()
