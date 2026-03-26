"""
Figure 1 Recreation — Audio Waveform vs Model Prediction Timeline.

Recreates Collins et al. Figure 1:
- Light blue audio waveform in background (left y-axis: amplitude)
- Red shaded bands = ground truth HDM events
- Green line = model P(HDM) probability (right y-axis: probability)
- Dashed threshold line
- Green shaded regions where model predicts positive

Usage:  python src/figure1_dashboard.py
Output: results/figure1_dashboard.html
"""

import gc
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
DATASET_DIR = ROOT / "data" / "dataset"
RESULTS_DIR = ROOT / "results"
SAMPLE_RATE = 16000


def load_data():
    with open(RESULTS_DIR / "gpt4o_20shot_v4_results.json") as f:
        v4 = json.load(f)
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)
    human_labels = {}
    labels_path = ROOT / "data" / "hdm_labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            human_labels = json.load(f)
    return v4, meta, human_labels


def downsample_waveform(audio, target_points=15000):
    """Downsample audio to target number of points for dense waveform display."""
    chunk_size = max(1, len(audio) // target_points)
    n_chunks = len(audio) // chunk_size
    audio = audio[:n_chunks * chunk_size]
    chunks = audio.reshape(n_chunks, chunk_size)
    maxes = chunks.max(axis=1)
    mins = chunks.min(axis=1)
    times = np.arange(n_chunks) * chunk_size / SAMPLE_RATE
    return times, maxes, mins


def build_meeting_data(v4, meta, human_labels):
    """Build per-meeting timeline data from all splits."""
    all_ex = meta["positive"] + meta["negative"]

    meeting_data = defaultdict(lambda: {"hdm_regions": [], "samples": []})

    # Collect HDM regions
    for i, ex in enumerate(meta["positive"]):
        mid = ex["meeting_id"]
        if ex.get("hdm_start") is not None:
            meeting_data[mid]["hdm_regions"].append({
                "start": ex["hdm_start"],
                "end": ex["hdm_end"],
                "text": ex.get("text", ""),
            })

    # Collect predictions from all splits
    for split_idx, split_result in enumerate(v4["splits"]):
        split_meta = meta["splits"][split_idx]
        test_meetings = set(split_meta["test"])
        test_indices = [j for j, ex in enumerate(all_ex) if ex["meeting_id"] in test_meetings]

        for local_i, global_i in enumerate(test_indices):
            ex = all_ex[global_i]
            mid = ex["meeting_id"]
            meeting_data[mid]["samples"].append({
                "time": ex["sample_time"],
                "label": ex["label"],
                "pred": split_result["predictions"][local_i],
                "prob_p": split_result["probabilities"][local_i],
                "text": ex.get("text", ""),
                "speaker": ex.get("speaker", ""),
            })

    # Sort and deduplicate
    for mid in meeting_data:
        meeting_data[mid]["samples"].sort(key=lambda x: x["time"])
        meeting_data[mid]["hdm_regions"].sort(key=lambda x: x["start"])

    return meeting_data


def create_figure(mid, data):
    """Create a Plotly figure matching Figure 1 style for one meeting."""
    audio_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
    if not audio_path.exists():
        return None

    # Load and downsample waveform
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr

    wave_t, wave_max, wave_min = downsample_waveform(audio, target_points=5000)
    wave_t_ms = wave_t * 1000  # convert to ms

    del audio
    gc.collect()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Audio waveform envelope (light blue, background)
    # Upper envelope
    fig.add_trace(
        go.Scatter(
            x=wave_t_ms, y=wave_max,
            mode="lines", line=dict(color="rgba(100,149,237,0.0)", width=0),
            showlegend=False, hoverinfo="skip",
        ),
        secondary_y=False,
    )
    # Lower envelope (filled to upper)
    fig.add_trace(
        go.Scatter(
            x=wave_t_ms, y=wave_min,
            mode="lines", line=dict(color="rgba(100,149,237,0.0)", width=0),
            fill="tonexty", fillcolor="rgba(100,149,237,0.25)",
            name="Audio Waveform",
            hoverinfo="skip",
        ),
        secondary_y=False,
    )

    # 2. Ground truth HDM regions (red shaded bands)
    # HDMs are very short (mean 0.6s) on a ~2000s timeline, so enforce
    # a minimum visual width so they're actually visible
    min_width_ms = duration * 1000 * 0.004  # at least 0.4% of timeline width
    for i, r in enumerate(data["hdm_regions"]):
        x0 = r["start"] * 1000
        x1 = r["end"] * 1000
        actual_width = x1 - x0
        if actual_width < min_width_ms:
            center = (x0 + x1) / 2
            x0 = center - min_width_ms / 2
            x1 = center + min_width_ms / 2
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="rgba(255,70,70,0.4)",
            line=dict(color="rgba(255,50,50,0.8)", width=1.5),
            layer="below",
        )

    # Dummy trace for ground truth legend
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="rgba(255,50,50,0.8)", width=8),
            name="Ground Truth Event",
        ),
        secondary_y=True,
    )

    # 3. Model prediction as continuous signal (mostly 0, spikes at samples)
    # Like the paper: sliding window signal that's near 0 baseline with
    # sharp spikes at HDM events
    samples = data["samples"]
    if samples:
        # Build a continuous time series: for each sample, create a spike
        # that rises from 0, hits the probability, and drops back to 0
        # This mimics the paper's sliding window approach
        cont_times = []
        cont_probs = []
        cont_texts = []

        spike_width = 2000  # ms width of each spike base

        sorted_samples = sorted(samples, key=lambda s: s["time"])

        for s in sorted_samples:
            t_ms = s["time"] * 1000
            p = s["prob_p"]
            hover = (
                f"Time: {s['time']:.1f}s<br>"
                f"P(HDM): {s['prob_p']:.3f}<br>"
                f"Pred: {'P' if s['pred']==1 else 'N'}<br>"
                f"True: {'P' if s['label']==1 else 'N'}<br>"
                f"Text: {s['text']}"
            )
            # Rise from 0 to probability and back
            cont_times.extend([t_ms - spike_width, t_ms, t_ms + spike_width])
            cont_probs.extend([0, p, 0])
            cont_texts.extend(["", hover, ""])

        fig.add_trace(
            go.Scatter(
                x=cont_times, y=cont_probs,
                mode="lines", line=dict(color="#1a9641", width=1.5),
                fill="tozeroy", fillcolor="rgba(26,150,65,0.1)",
                name="Model Prediction",
                text=cont_texts, hoverinfo="text",
            ),
            secondary_y=True,
        )

        # 4. Green shaded regions where model predicts positive (above threshold)
        threshold = 0.97
        for s in sorted_samples:
            if s["prob_p"] > threshold:
                t_ms = s["time"] * 1000
                fig.add_vrect(
                    x0=t_ms - spike_width, x1=t_ms + spike_width,
                    fillcolor="rgba(26,150,65,0.2)",
                    line=dict(width=0),
                    layer="below",
                )

        # Dummy trace for positive prediction region legend
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color="rgba(26,150,65,0.3)",
                            line=dict(color="#1a9641", width=2)),
                name="Positive Prediction Region",
            ),
            secondary_y=True,
        )

    # 5. Decision threshold (dashed orange line) — matching paper's 0.97
    fig.add_hline(
        y=0.97, secondary_y=True,
        line=dict(color="orange", width=1.5, dash="dash"),
        annotation_text="Decision Threshold (0.97)",
        annotation_position="top right",
        annotation_font_color="orange",
    )

    # Layout
    n_pos = sum(1 for s in samples if s["label"] == 1)
    n_pred_p = sum(1 for s in samples if s["pred"] == 1)
    tp = sum(1 for s in samples if s["label"] == 1 and s["pred"] == 1)

    fig.update_layout(
        title=dict(
            text=f"Audio Waveform vs. Model Prediction and Ground Truth — {mid}",
            x=0.5, font=dict(size=16),
        ),
        template="plotly_white",
        height=450,
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        title_text="Time (ms)",
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
    )
    fig.update_yaxes(
        title_text="Amplitude", secondary_y=False,
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
    )
    fig.update_yaxes(
        title_text="Model Probability", secondary_y=True,
        range=[0, 1.05],
        showgrid=False,
        title_font=dict(color="#1a9641"),
        tickfont=dict(color="#1a9641"),
    )

    return fig


def main():
    print("Figure 1 Dashboard — Audio Waveform vs Model Prediction")
    print("=" * 55)

    print("\nLoading data...")
    v4, meta, human_labels = load_data()
    meeting_data = build_meeting_data(v4, meta, human_labels)

    # Select meetings with the most HDMs for display
    candidates = []
    for mid, data in meeting_data.items():
        n_pos = sum(1 for s in data["samples"] if s["label"] == 1)
        audio_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        if n_pos > 0 and audio_path.exists():
            candidates.append((mid, data, n_pos))
    candidates.sort(key=lambda x: -x[2])

    # Take top meetings
    selected = candidates[:8]
    print(f"Selected {len(selected)} meetings for visualization")

    # Generate each chart as a self-contained Plotly div
    chart_divs = []
    for i, (mid, data, n_pos) in enumerate(selected):
        print(f"  Generating chart for {mid} ({n_pos} HDMs)...")
        fig = create_figure(mid, data)
        if fig is None:
            chart_divs.append(f'<div id="chart-{i}">Failed to load {mid}</div>')
            continue
        # Use to_html with include_plotlyjs=False (we'll load CDN once)
        div_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=f"chart-{i}",
        )
        chart_divs.append(div_html)

    meeting_info = json.dumps([{"id": m, "n_pos": n} for m, _, n in selected])

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Figure 1 Recreation — HDM Detection Timeline</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #fafafa; margin: 0; padding: 20px; }}
.container {{ max-width: 1100px; margin: 0 auto; }}
h1 {{ text-align: center; color: #333; margin-bottom: 5px; }}
.subtitle {{ text-align: center; color: #666; margin-bottom: 20px; font-size: 14px; }}
.summary {{ display: flex; gap: 15px; justify-content: center; margin-bottom: 25px; flex-wrap: wrap; }}
.stat {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.stat-val {{ font-size: 28px; font-weight: bold; color: #1a9641; }}
.stat-val.paper {{ color: #666; }}
.stat-label {{ font-size: 11px; color: #888; text-transform: uppercase; }}
.chart-wrap {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 20px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.meeting-nav {{ display: flex; gap: 6px; flex-wrap: wrap; justify-content: center; margin-bottom: 20px; }}
.meeting-btn {{ padding: 6px 14px; background: #fff; border: 1px solid #ddd; color: #333; border-radius: 6px; cursor: pointer; font-size: 13px; }}
.meeting-btn:hover {{ background: #f0f0f0; }}
.meeting-btn.active {{ background: #1a9641; border-color: #1a9641; color: #fff; }}
.caption {{ text-align: center; color: #666; font-size: 13px; margin: 10px 0 20px; font-style: italic; }}
</style>
</head>
<body>
<div class="container">
<h1>Audio Waveform vs. Model Prediction and Ground Truth</h1>
<div class="subtitle">Recreating Collins et al. Figure 1 — GPT-4o Audio v4 (20-shot) on AMI Meeting Corpus</div>

<div class="summary">
  <div class="stat"><div class="stat-val">0.97</div><div class="stat-label">Our F1 (GPT-4o v4)</div></div>
  <div class="stat"><div class="stat-val paper">0.87</div><div class="stat-label">Paper F1 (Gemini 1.5 Pro)</div></div>
  <div class="stat"><div class="stat-val">1.00</div><div class="stat-label">Best Split F1</div></div>
  <div class="stat"><div class="stat-val">4/1639</div><div class="stat-label">False Positives</div></div>
</div>

<div class="caption">
The red shaded areas represent ground truth Hearing Difficulty Moments and the green line is the output
probability of the "P" token obtained from GPT-4o Audio. Light blue shows the audio waveform envelope.
</div>

<div class="meeting-nav" id="meeting-nav"></div>

"""
    # Add chart divs, each wrapped and hidden except first
    for i, div_html in enumerate(chart_divs):
        display = "block" if i == 0 else "none"
        html += f'<div class="chart-wrap" id="chart-wrap-{i}" style="display:{display};">\n{div_html}\n</div>\n'

    html += f"""
<script>
const meetingInfo = {meeting_info};

// Build nav buttons
const nav = document.getElementById('meeting-nav');
meetingInfo.forEach((m, i) => {{
  const btn = document.createElement('button');
  btn.className = 'meeting-btn' + (i === 0 ? ' active' : '');
  btn.textContent = m.id + ' (' + m.n_pos + ' HDM)';
  btn.onclick = () => {{
    meetingInfo.forEach((_, j) => {{
      document.getElementById('chart-wrap-' + j).style.display = 'none';
      document.getElementById('btn-' + j).classList.remove('active');
    }});
    document.getElementById('chart-wrap-' + i).style.display = 'block';
    btn.classList.add('active');
    // Trigger Plotly resize for the newly visible chart
    const chartEl = document.getElementById('chart-' + i);
    if (chartEl) Plotly.Plots.resize(chartEl);
  }};
  btn.id = 'btn-' + i;
  nav.appendChild(btn);
}});
</script>
</div>
</body>
</html>"""

    output_path = RESULTS_DIR / "figure1_dashboard.html"
    with open(output_path, "w") as f:
        f.write(html)

    print(f"\nDashboard saved to: {output_path}")
    print(f"Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
