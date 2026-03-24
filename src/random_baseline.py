"""
Random guessing baseline for comparison with the 10-shot Gemini classifier.

Simulates a classifier that randomly assigns P/N labels,
using the same Monte Carlo CV splits and evaluation protocol.
We run multiple random seeds per split to get stable estimates.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, classification_report

DATASET_DIR = Path("data/dataset")
RESULTS_DIR = Path("results")
NUM_RANDOM_SEEDS = 100  # average over many random runs per split


def evaluate_random_split(labels, meta, split_idx, base_rate=0.5):
    """Evaluate random guessing on one CV split."""
    split = meta["splits"][split_idx]
    test_meetings = set(split["test"])
    all_examples = meta["positive"] + meta["negative"]

    test_indices = [i for i, ex in enumerate(all_examples) if ex["meeting_id"] in test_meetings]
    test_labels = labels[test_indices]

    f1s = []
    for seed in range(NUM_RANDOM_SEEDS):
        rng = np.random.RandomState(seed + split_idx * 1000)
        predictions = rng.choice([0, 1], size=len(test_labels), p=[1 - base_rate, base_rate])
        f1 = f1_score(test_labels, predictions, zero_division=0)
        f1s.append(f1)

    return {
        "split": split_idx,
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "n_test": len(test_indices),
        "n_pos": int(test_labels.sum()),
        "positive_rate": float(test_labels.mean()),
    }


def main():
    print("Random Guessing Baseline")
    print("=" * 55)

    labels = np.load(DATASET_DIR / "labels.npy")
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    print(f"Dataset: {len(labels)} segments, {labels.sum()} positive")
    print(f"Overall positive rate: {labels.mean():.3f}")
    print(f"Averaging over {NUM_RANDOM_SEEDS} random seeds per split\n")

    # Test with 50/50 random guessing
    print("--- 50/50 Random Guessing ---")
    results_50 = []
    for split_idx in range(len(meta["splits"])):
        result = evaluate_random_split(labels, meta, split_idx, base_rate=0.5)
        results_50.append(result)
        print(f"  Split {split_idx+1}: F1 = {result['f1_mean']:.4f} (+/- {result['f1_std']:.4f}) "
              f"[test={result['n_test']}, pos={result['n_pos']}]")

    avg_f1_50 = np.mean([r["f1_mean"] for r in results_50])
    print(f"  Average F1: {avg_f1_50:.4f}")

    # Test with base-rate-matched random guessing (same positive rate as data)
    overall_pos_rate = float(labels.mean())
    print(f"\n--- Base-rate ({overall_pos_rate:.2f}) Random Guessing ---")
    results_br = []
    for split_idx in range(len(meta["splits"])):
        result = evaluate_random_split(labels, meta, split_idx, base_rate=overall_pos_rate)
        results_br.append(result)
        print(f"  Split {split_idx+1}: F1 = {result['f1_mean']:.4f} (+/- {result['f1_std']:.4f})")

    avg_f1_br = np.mean([r["f1_mean"] for r in results_br])
    print(f"  Average F1: {avg_f1_br:.4f}")

    # Compare with Gemini results
    print(f"\n{'=' * 55}")
    print(f"COMPARISON")
    print(f"{'=' * 55}")
    print(f"{'Method':<40} {'Avg F1':>8}")
    print(f"{'-'*55}")
    print(f"{'Random (50/50)':<40} {avg_f1_50:>8.4f}")
    print(f"{'Random (base-rate matched)':<40} {avg_f1_br:>8.4f}")

    hotword_path = RESULTS_DIR / "baseline_hotword.json"
    if hotword_path.exists():
        with open(hotword_path) as f:
            hotword = json.load(f)
        print(f"{'ASR Hotword Heuristic (Whisper)':<40} {hotword['avg_f1']:>8.4f}")

    gemini_path = RESULTS_DIR / "gemini_10shot_results.json"
    if gemini_path.exists():
        with open(gemini_path) as f:
            gemini = json.load(f)
        print(f"{'Gemini 3.1 Pro (10-shot audio)':<40} {gemini['avg_f1']:>8.4f}")

    print(f"{'Collins et al. Gemini 1.5 Pro (10-shot)':<40} {'0.8700':>8}")
    print(f"{'-'*55}")

    # Save
    output = {
        "method": "Random Guessing Baseline",
        "random_50_50": {
            "avg_f1": avg_f1_50,
            "splits": results_50,
        },
        "random_base_rate": {
            "base_rate": overall_pos_rate,
            "avg_f1": avg_f1_br,
            "splits": results_br,
        },
    }
    with open(RESULTS_DIR / "random_baseline.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/random_baseline.json")


if __name__ == "__main__":
    main()
