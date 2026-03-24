"""
Run all evaluation methods and generate summary results table + plots.
Compares to Collins et al. Table 1 results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")


def load_results():
    """Load all available results."""
    results = {}

    files = {
        "ASR Hotword Heuristic": "baseline_hotword.json",
        "Wav2Vec 2.0 Transfer Learning": "wav2vec_results.json",
        "Audio LM (text-only approx.)": "audio_lm_results.json",
    }

    for name, filename in files.items():
        path = RESULTS_DIR / filename
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)

    return results


def print_results_table(results):
    """Print comparison table similar to paper's Table 1."""
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)

    # Paper's results (Table 1)
    paper_results = {
        "ASR Hotword Heuristic (Baseline)": 0.39,
        "Gemini 1.5 Pro [text only] (0-shot)": 0.39,
        "Gemini 1.5 Pro [audio] (0-shot)": 0.75,
        "Wav2Vec 2.0 Transfer Learning": 0.76,
        "Gemini 2.0 Flash (LoRA Fine-Tuning)": 0.77,
        "Gemini 1.5 Pro (2-shot)": 0.85,
        "Gemini 1.5 Pro (10-shot)": 0.87,
    }

    print(f"\n{'Approach':<45} {'Paper F1':>10} {'Our F1':>10}")
    print("-"*70)

    for name, paper_f1 in paper_results.items():
        our_f1 = ""
        for our_name, our_result in results.items():
            if "Hotword" in name and "Hotword" in our_name:
                our_f1 = f"{our_result['avg_f1']:.2f}"
            elif "Wav2Vec" in name and "Wav2Vec" in our_name:
                our_f1 = f"{our_result['avg_f1']:.2f}"
            elif "text only" in name and "text-only" in our_name:
                our_f1 = f"{our_result['avg_f1']:.2f}"

        print(f"{name:<45} {paper_f1:>10.2f} {our_f1:>10}")

    # Our additional results
    print("-"*70)
    for name, result in results.items():
        if not any(k in name for k in ["Hotword", "Wav2Vec", "text-only"]):
            print(f"{'(ours) ' + name:<45} {'':>10} {result['avg_f1']:>10.2f}")


def plot_precision_recall(results):
    """Generate precision-recall curves (like paper's Figure 5)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = {"Wav2Vec": "blue", "Audio LM": "red", "Hotword": "green"}

    for name, result in results.items():
        if "splits" in result:
            # Use first split's precision-recall curve if available
            for split_result in result["splits"]:
                if "precision_curve" in split_result and "recall_curve" in split_result:
                    precision = split_result["precision_curve"]
                    recall = split_result["recall_curve"]
                    color = "blue"
                    for key, c in colors.items():
                        if key in name:
                            color = c
                            break
                    ax.plot(recall, precision, label=f"{name} (F1={result['avg_f1']:.2f})",
                            color=color, linewidth=2)
                    break

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "precision_recall_curves.png", dpi=150)
    print(f"Saved precision-recall plot to {RESULTS_DIR}/precision_recall_curves.png")


def plot_f1_comparison(results):
    """Bar chart comparing F1 scores."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Paper results
    paper_methods = [
        "ASR Hotword\n(Paper)", "Wav2Vec 2.0\n(Paper)",
        "Gemini text-only\n(Paper)", "Gemini audio 0-shot\n(Paper)",
        "Gemini 2-shot\n(Paper)", "Gemini 10-shot\n(Paper)",
    ]
    paper_f1s = [0.39, 0.76, 0.39, 0.75, 0.85, 0.87]

    # Our results
    our_methods = []
    our_f1s = []
    our_stds = []
    for name, result in results.items():
        short_name = name.replace("Transfer Learning", "").strip()
        our_methods.append(f"{short_name}\n(Ours/AMI)")
        our_f1s.append(result["avg_f1"])
        our_stds.append(result.get("std_f1", 0))

    # Plot
    x_paper = np.arange(len(paper_methods))
    x_ours = np.arange(len(our_methods)) + len(paper_methods) + 1

    ax.bar(x_paper, paper_f1s, color="steelblue", alpha=0.7, label="Collins et al. (SWDA/MRDA)")
    ax.bar(x_ours, our_f1s, yerr=our_stds, color="coral", alpha=0.7, label="Ours (AMI Corpus)")

    all_x = list(x_paper) + list(x_ours)
    all_labels = paper_methods + our_methods
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=8, ha="center")
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Hearing Difficulty Moment Detection: Paper vs Our Replication (AMI)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(paper_f1s):
        ax.text(x_paper[i], v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(our_f1s):
        ax.text(x_ours[i], v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "f1_comparison.png", dpi=150)
    print(f"Saved F1 comparison plot to {RESULTS_DIR}/f1_comparison.png")


def main():
    results = load_results()

    if not results:
        print("No results found. Run the individual model scripts first:")
        print("  uv run python src/baseline_hotword.py")
        print("  uv run python src/wav2vec_classifier.py")
        print("  uv run python src/audio_lm_prompting.py")
        return

    print(f"Loaded results for: {list(results.keys())}")

    print_results_table(results)
    plot_precision_recall(results)
    plot_f1_comparison(results)

    # Save combined summary
    summary = {
        "paper": "Collins et al. - Identifying Hearing Difficulty Moments in Conversational Audio",
        "replication_dataset": "AMI Meeting Corpus",
        "original_dataset": "SWDA + MRDA",
        "results": {
            name: {
                "avg_f1": r["avg_f1"],
                "std_f1": r.get("std_f1", None),
                "method": r.get("method", name),
            }
            for name, r in results.items()
        }
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()
