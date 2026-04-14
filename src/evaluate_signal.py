"""
Signal Quality Evaluation — Measures alignment between model P(HDM) and ground truth.

Computes per-meeting and aggregate metrics for the sliding window predictions:
- NABC: Normalized Area Between Curves (lower = better match to ground truth)
- Peak@HDM: Max model probability within ±6s of each true HDM (higher = better detection)
- False Alarm Rate: Fraction of non-HDM windows with prob > 0.5
- Noise Floor: Mean probability in non-HDM regions
- SNR: Signal-to-Noise Ratio (peak at HDM / noise floor)
- Alignment Score: Composite 0-100 score combining detection and specificity

Usage:
    python src/evaluate_signal.py                          # print summary table
    python src/evaluate_signal.py --plot                   # generate plots
    python src/evaluate_signal.py --dir sliding_window     # evaluate zero-shot results
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.integrate import trapezoid

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
DATASET_PATH = ROOT / "data" / "dataset" / "dataset_meta.json"

# How far around an HDM to consider as "HDM region" for the ground truth signal
HDM_REGION_RADIUS = 4.0  # seconds each side
# How far around an HDM to search for peak model response
PEAK_SEARCH_RADIUS = 6.0  # seconds each side
# Threshold for counting false alarms
FA_THRESHOLD = 0.5


def load_ground_truth():
    """Load HDM annotations grouped by meeting."""
    meta = json.load(open(DATASET_PATH))
    hdms_by_meeting = defaultdict(list)
    for s in meta["positive"]:
        hdms_by_meeting[s["meeting_id"]].append({
            "start": s["hdm_start"],
            "end": s["hdm_end"],
            "text": s.get("text", ""),
        })
    return hdms_by_meeting


def evaluate_meeting(meeting_id, data, hdms):
    """Compute signal quality metrics for a single meeting."""
    windows = data["windows"]
    times = np.array([w["time"] for w in windows])
    probs = np.array([w["prob_p"] for w in windows])
    duration = data["duration"]

    # Build binary ground truth signal at the same timepoints
    gt = np.zeros_like(probs)
    for h in hdms:
        mask = (times >= h["start"] - HDM_REGION_RADIUS) & (times <= h["end"] + HDM_REGION_RADIUS)
        gt[mask] = 1.0

    # 1. Normalized Area Between Curves
    nabc = trapezoid(np.abs(probs - gt), times) / duration

    # 2. Peak probability near each HDM
    peaks_at_hdm = []
    for h in hdms:
        nearby = probs[
            (times >= h["start"] - PEAK_SEARCH_RADIUS) &
            (times <= h["end"] + PEAK_SEARCH_RADIUS)
        ]
        if len(nearby) > 0:
            peaks_at_hdm.append(float(nearby.max()))
    peak_at_hdm = np.mean(peaks_at_hdm) if peaks_at_hdm else 0.0

    # 3. False alarm rate and noise floor (non-HDM regions only)
    non_hdm_mask = gt == 0
    non_hdm_probs = probs[non_hdm_mask]
    if len(non_hdm_probs) > 0:
        false_alarm_rate = float((non_hdm_probs > FA_THRESHOLD).sum()) / len(non_hdm_probs)
        noise_floor = float(non_hdm_probs.mean())
    else:
        false_alarm_rate = 0.0
        noise_floor = 0.0

    # 4. Signal-to-Noise Ratio
    snr = peak_at_hdm / max(noise_floor, 0.001)

    # 5. Composite Alignment Score (0-100)
    #    - Detection component (50%): how well does the model spike at HDMs?
    #    - Specificity component (50%): how quiet is the signal elsewhere?
    detection_score = peak_at_hdm  # 0-1, higher is better
    specificity_score = 1.0 - min(noise_floor / 0.5, 1.0)  # 0-1, penalize noise floors above 0.5
    alignment_score = 100.0 * (0.5 * detection_score + 0.5 * specificity_score)

    return {
        "meeting": meeting_id,
        "n_hdms": len(hdms),
        "duration": duration,
        "n_windows": len(windows),
        "nabc": nabc,
        "peak_at_hdm": peak_at_hdm,
        "false_alarm_rate": false_alarm_rate,
        "noise_floor": noise_floor,
        "snr": snr,
        "alignment_score": alignment_score,
    }


def evaluate_all(results_subdir="sliding_window_10shot"):
    """Evaluate all meetings in a results directory."""
    sw_dir = RESULTS_DIR / results_subdir
    hdms_by_meeting = load_ground_truth()

    results = []
    for jf in sorted(sw_dir.glob("*.json")):
        mid = jf.stem
        data = json.load(open(jf))
        hdms = hdms_by_meeting.get(mid, [])
        if not hdms:
            continue
        results.append(evaluate_meeting(mid, data, hdms))

    results.sort(key=lambda r: r["alignment_score"], reverse=True)
    return results


def print_summary(results):
    """Print results table and aggregate statistics."""
    print(f"\n{'Meeting':<12} {'HDMs':>4} {'Score':>6} {'NABC':>7} {'Peak':>6} "
          f"{'FA%':>6} {'Noise':>6} {'SNR':>6}")
    print("-" * 68)

    for r in results:
        print(f"{r['meeting']:<12} {r['n_hdms']:>4} {r['alignment_score']:>6.1f} "
              f"{r['nabc']:>7.4f} {r['peak_at_hdm']:>6.3f} "
              f"{r['false_alarm_rate']*100:>5.1f}% {r['noise_floor']:>6.3f} "
              f"{r['snr']:>6.1f}")

    # Aggregate stats
    scores = [r["alignment_score"] for r in results]
    peaks = [r["peak_at_hdm"] for r in results]
    nabcs = [r["nabc"] for r in results]
    fas = [r["false_alarm_rate"] for r in results]
    noises = [r["noise_floor"] for r in results]
    snrs = [r["snr"] for r in results]

    print("\n" + "=" * 68)
    print("AGGREGATE STATISTICS")
    print("=" * 68)
    print(f"  Alignment Score:  mean={np.mean(scores):.1f}  median={np.median(scores):.1f}  "
          f"std={np.std(scores):.1f}  range=[{min(scores):.1f}, {max(scores):.1f}]")
    print(f"  Peak@HDM:         mean={np.mean(peaks):.3f}  median={np.median(peaks):.3f}")
    print(f"  NABC:             mean={np.mean(nabcs):.4f}  median={np.median(nabcs):.4f}")
    print(f"  False Alarm Rate: mean={np.mean(fas)*100:.1f}%  median={np.median(fas)*100:.1f}%")
    print(f"  Noise Floor:      mean={np.mean(noises):.3f}  median={np.median(noises):.3f}")
    print(f"  SNR:              mean={np.mean(snrs):.1f}  median={np.median(snrs):.1f}")

    # Grade distribution
    excellent = sum(1 for s in scores if s >= 80)
    good = sum(1 for s in scores if 60 <= s < 80)
    fair = sum(1 for s in scores if 40 <= s < 60)
    poor = sum(1 for s in scores if s < 40)
    print(f"\n  Grade Distribution ({len(results)} meetings):")
    print(f"    Excellent (>=80): {excellent:>3} ({excellent/len(results)*100:.0f}%)")
    print(f"    Good   (60-79):   {good:>3} ({good/len(results)*100:.0f}%)")
    print(f"    Fair   (40-59):   {fair:>3} ({fair/len(results)*100:.0f}%)")
    print(f"    Poor     (<40):   {poor:>3} ({poor/len(results)*100:.0f}%)")

    # Detection rate
    detected = sum(1 for p in peaks if p > 0.5)
    print(f"\n  HDMs detected (Peak > 0.5): {detected}/{len(results)} meetings ({detected/len(results)*100:.0f}%)")


def generate_plots(results, output_dir=None):
    """Generate evaluation visualizations."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir = Path(output_dir)

    scores = [r["alignment_score"] for r in results]
    meetings = [r["meeting"] for r in results]
    peaks = [r["peak_at_hdm"] for r in results]
    noises = [r["noise_floor"] for r in results]
    fas = [r["false_alarm_rate"] for r in results]
    nabcs = [r["nabc"] for r in results]

    # Color by grade
    colors = []
    for s in scores:
        if s >= 80:
            colors.append("#2ecc71")  # green
        elif s >= 60:
            colors.append("#3498db")  # blue
        elif s >= 40:
            colors.append("#f39c12")  # orange
        else:
            colors.append("#e74c3c")  # red

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Alignment Score by Meeting (sorted)",
            "Score Distribution",
            "Peak@HDM vs Noise Floor",
            "NABC vs False Alarm Rate",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # 1. Bar chart of scores sorted
    fig.add_trace(go.Bar(
        x=meetings, y=scores,
        marker_color=colors,
        text=[f"{s:.0f}" for s in scores],
        textposition="outside",
        textfont_size=8,
        showlegend=False,
    ), row=1, col=1)
    fig.update_xaxes(tickangle=90, tickfont_size=7, row=1, col=1)
    fig.update_yaxes(title_text="Score", range=[0, 105], row=1, col=1)

    # 2. Histogram of scores
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=20,
        marker_color="#3498db",
        showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title_text="Alignment Score", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    # Add mean/median lines
    fig.add_vline(x=np.mean(scores), line_dash="dash", line_color="red",
                  annotation_text=f"mean={np.mean(scores):.0f}", row=1, col=2)
    fig.add_vline(x=np.median(scores), line_dash="dot", line_color="green",
                  annotation_text=f"median={np.median(scores):.0f}", row=1, col=2)

    # 3. Peak vs Noise scatter
    fig.add_trace(go.Scatter(
        x=noises, y=peaks,
        mode="markers+text",
        text=meetings,
        textposition="top center",
        textfont_size=6,
        marker=dict(size=8, color=scores, colorscale="RdYlGn", cmin=0, cmax=100,
                    colorbar=dict(title="Score", x=0.45, len=0.4, y=0.15)),
        showlegend=False,
    ), row=2, col=1)
    fig.update_xaxes(title_text="Noise Floor", row=2, col=1)
    fig.update_yaxes(title_text="Peak@HDM", row=2, col=1)
    # Ideal region
    fig.add_shape(type="rect", x0=0, x1=0.1, y0=0.5, y1=1.0,
                  fillcolor="green", opacity=0.08, line_width=0,
                  row=2, col=1)

    # 4. NABC vs FA rate
    fig.add_trace(go.Scatter(
        x=fas, y=nabcs,
        mode="markers+text",
        text=meetings,
        textposition="top center",
        textfont_size=6,
        marker=dict(size=8, color=scores, colorscale="RdYlGn", cmin=0, cmax=100),
        showlegend=False,
    ), row=2, col=2)
    fig.update_xaxes(title_text="False Alarm Rate", row=2, col=2)
    fig.update_yaxes(title_text="NABC (lower=better)", row=2, col=2)

    fig.update_layout(
        title="Signal Quality Evaluation — Gemini 2.5 Flash 10-Shot Sliding Window",
        height=900,
        width=1400,
        template="plotly_white",
        font=dict(size=11),
    )

    out_path = output_dir / "signal_evaluation.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\nPlot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate sliding window signal quality")
    parser.add_argument("--dir", default="sliding_window_10shot",
                        help="Results subdirectory (default: sliding_window_10shot)")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    print(f"Evaluating: results/{args.dir}/")
    print("=" * 68)

    results = evaluate_all(args.dir)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)

    if args.plot:
        generate_plots(results)


if __name__ == "__main__":
    main()
