"""
Figure 1 Recreation — Audio Waveform vs Model Prediction Timeline.

Recreates Collins et al. Figure 1 using Gemini 2.5 Flash sliding window data:
- Light blue audio waveform in background (left y-axis: amplitude)
- Red shaded bands = ground truth HDM events
- Green line = continuous P(HDM) probability from Gemini logprobs (right y-axis)
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
SW_DIR = RESULTS_DIR / "sliding_window"
SAMPLE_RATE = 16000


def load_data():
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)
    return meta


def get_hdm_regions(meta):
    """Extract ground truth HDM regions per meeting."""
    regions = defaultdict(list)
    for ex in meta["positive"]:
        mid = ex["meeting_id"]
        if ex.get("hdm_start") is not None:
            regions[mid].append({
                "start": ex["hdm_start"],
                "end": ex["hdm_end"],
                "text": ex.get("text", ""),
            })
    for mid in regions:
        regions[mid].sort(key=lambda r: r["start"])
    return regions


def load_sliding_window(mid):
    """Load sliding window results for a meeting."""
    sw_path = SW_DIR / f"{mid}.json"
    if not sw_path.exists():
        return None
    with open(sw_path) as f:
        return json.load(f)


def downsample_waveform(audio, target_points=5000):
    chunk_size = max(1, len(audio) // target_points)
    n_chunks = len(audio) // chunk_size
    audio = audio[:n_chunks * chunk_size]
    chunks = audio.reshape(n_chunks, chunk_size)
    maxes = chunks.max(axis=1)
    mins = chunks.min(axis=1)
    times = np.arange(n_chunks) * chunk_size / SAMPLE_RATE
    return times, maxes, mins


def create_figure(mid, hdm_regions, sw_data):
    """Create a Plotly figure matching Figure 1 style for one meeting."""
    audio_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
    if not audio_path.exists():
        return None

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr

    wave_t, wave_max, wave_min = downsample_waveform(audio, target_points=5000)

    del audio
    gc.collect()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Audio waveform envelope (light blue)
    fig.add_trace(
        go.Scatter(
            x=wave_t, y=wave_max,
            mode="lines", line=dict(color="rgba(100,149,237,0.0)", width=0),
            showlegend=False, hoverinfo="skip",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=wave_t, y=wave_min,
            mode="lines", line=dict(color="rgba(100,149,237,0.0)", width=0),
            fill="tonexty", fillcolor="rgba(100,149,237,0.25)",
            name="Audio Waveform",
            hoverinfo="skip",
        ),
        secondary_y=False,
    )

    # 2. Ground truth HDM regions (red shaded bands)
    min_width = duration * 0.004
    for r in hdm_regions:
        x0 = r["start"]
        x1 = r["end"]
        actual_width = x1 - x0
        if actual_width < min_width:
            center = (x0 + x1) / 2
            x0 = center - min_width / 2
            x1 = center + min_width / 2
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

    # 3. Continuous probability signal from Gemini sliding window
    windows = sw_data["windows"]
    sw_times = [w["time"] for w in windows]
    sw_probs = [w["prob_p"] for w in windows]
    sw_texts = [
        f"Time: {w['time']:.1f}s<br>P(HDM): {w['prob_p']:.3f}<br>"
        f"Pred: {'P' if w['pred']==1 else 'N'}"
        for w in windows
    ]

    fig.add_trace(
        go.Scatter(
            x=sw_times, y=sw_probs,
            mode="lines", line=dict(color="#1a9641", width=1.5),
            fill="tozeroy", fillcolor="rgba(26,150,65,0.08)",
            name="Model Prediction",
            text=sw_texts, hoverinfo="text",
        ),
        secondary_y=True,
    )

    # 4. Green shaded regions where model predicts positive
    step = sw_data.get("step_s", 4)
    region_start = None
    for i, w in enumerate(windows):
        if w["pred"] == 1 and region_start is None:
            region_start = w["time"] - step / 2
        elif w["pred"] != 1 and region_start is not None:
            fig.add_vrect(
                x0=region_start, x1=windows[i-1]["time"] + step / 2,
                fillcolor="rgba(26,150,65,0.2)",
                line=dict(width=0), layer="below",
            )
            region_start = None
    if region_start is not None:
        fig.add_vrect(
            x0=region_start, x1=windows[-1]["time"] + step / 2,
            fillcolor="rgba(26,150,65,0.2)",
            line=dict(width=0), layer="below",
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

    # 5. Decision threshold (dashed orange line)
    fig.add_hline(
        y=0.5, secondary_y=True,
        line=dict(color="orange", width=1.5, dash="dash"),
        annotation_text="Threshold (0.5)",
        annotation_position="top right",
        annotation_font_color="orange",
    )

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
        title_text="Time (s)", title_font=dict(size=9),
        tickfont=dict(size=9),
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
    )
    fig.update_yaxes(
        title_text="Amplitude", secondary_y=False,
        title_font=dict(size=9), tickfont=dict(size=9),
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
    )
    fig.update_yaxes(
        title_text="P(HDM)", secondary_y=True,
        range=[0, 1.05],
        showgrid=False,
        title_font=dict(size=9, color="#1a9641"),
        tickfont=dict(size=9, color="#1a9641"),
    )

    return fig


def main():
    print("Figure 1 Dashboard — Gemini 2.5 Flash Sliding Window")
    print("=" * 55)

    print("\nLoading data...")
    meta = load_data()
    hdm_regions = get_hdm_regions(meta)

    # Find meetings with sliding window data and HDMs
    candidates = []
    for sw_file in sorted(SW_DIR.glob("*.json")):
        if sw_file.stem == "run":
            continue
        mid = sw_file.stem
        n_hdm = len(hdm_regions.get(mid, []))
        audio_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        if n_hdm > 0 and audio_path.exists():
            candidates.append((mid, n_hdm))

    candidates.sort(key=lambda x: -x[1])
    selected = candidates[:8]
    print(f"Selected {len(selected)} meetings for visualization")

    chart_divs = []
    for i, (mid, n_hdm) in enumerate(selected):
        print(f"  Generating chart for {mid} ({n_hdm} HDMs)...")
        sw_data = load_sliding_window(mid)
        fig = create_figure(mid, hdm_regions.get(mid, []), sw_data)
        if fig is None:
            chart_divs.append(f'<div id="chart-{i}">Failed to load {mid}</div>')
            continue
        div_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=f"chart-{i}",
        )
        chart_divs.append(div_html)

    meeting_info = json.dumps([{"id": m, "n_pos": n} for m, n in selected])

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
<div class="subtitle">Recreating Collins et al. Figure 1 — Gemini 2.5 Flash (Vertex AI logprobs) on AMI Meeting Corpus</div>

<div class="caption">
The red shaded areas represent ground truth Hearing Difficulty Moments and the green line is the
continuous P(HDM) probability from Gemini 2.5 Flash logprobs (4s sliding window). Light blue shows the audio waveform envelope.
</div>

<div class="meeting-nav" id="meeting-nav"></div>

"""
    for i, div_html in enumerate(chart_divs):
        display = "block" if i == 0 else "none"
        html += f'<div class="chart-wrap" id="chart-wrap-{i}" style="display:{display};">\n{div_html}\n</div>\n'

    html += f"""
<script>
const meetingInfo = {meeting_info};

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
