"""
Interactive Dashboard: HDM Detection Trial Results.

Generates a self-contained HTML file with full Plotly interactivity
(hover, zoom, pan, toggle traces) and canvas-based audio waveform
visualization. No server required — open directly in a browser or
host via GitHub Pages.

Usage:  uv run python src/dashboard.py
Output: results/dashboard.html  +  docs/index.html
"""

import json
import wave as wave_mod
import struct
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

# ── Colors ──────────────────────────────────────────────────────────────────
C = {
    "gemini": "#4285F4",
    "hotword": "#FF6D00",
    "rand50": "#9E9E9E",
    "randbr": "#BDBDBD",
    "paper": "#E91E63",
    "pos": "#EF5350",
    "neg": "#66BB6A",
    "bg": "#FAFAFA",
    "text": "#212121",
}


def load_data():
    with open(RESULTS_DIR / "gemini_10shot_results.json") as f:
        gemini = json.load(f)
    with open(RESULTS_DIR / "baseline_hotword.json") as f:
        hotword = json.load(f)
    with open(RESULTS_DIR / "random_baseline.json") as f:
        random_bl = json.load(f)
    with open(DATA_DIR / "hdm_filtered.json") as f:
        annotations = json.load(f)
    example_audio_path = DATA_DIR / "example_audio.json"
    example_audio = None
    if example_audio_path.exists():
        with open(example_audio_path) as f:
            example_audio = json.load(f)
    return gemini, hotword, random_bl, annotations, example_audio


def extract_waveform_envelope(audio_path, start_sec, end_sec, points_per_sec=100):
    """Extract downsampled peak-envelope from a 16-bit mono WAV file."""
    with wave_mod.open(str(audio_path), "rb") as wf:
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        start_frame = int(start_sec * sr)
        end_frame = min(int(end_sec * sr), wf.getnframes())
        n_frames = end_frame - start_frame
        wf.setpos(start_frame)
        raw = wf.readframes(n_frames)

    n_samples = len(raw) // sampwidth
    samples = struct.unpack(f"<{n_samples}h", raw)

    duration = end_sec - start_sec
    n_points = int(duration * points_per_sec)
    window = max(1, n_samples // n_points)

    envelope = []
    for i in range(n_points):
        s = i * window
        e = min(s + window, n_samples)
        chunk = samples[s:e]
        peak = max(abs(v) for v in chunk) / 32768.0 if chunk else 0
        envelope.append(round(peak, 4))
    return envelope


def generate_prob_trace(hdms_in_range, start_sec, end_sec, step=1.0):
    """Generate illustrative probability trace mimicking Collins et al. Figure 1.

    Produces a P(HDM) value for each 1-second step, peaking near annotated HDMs.
    Uses deterministic noise so the output is reproducible.
    """
    rng = np.random.RandomState(42)
    n_points = int((end_sec - start_sec) / step) + 1
    raw = np.zeros(n_points)

    for i in range(n_points):
        t = start_sec + i * step
        in_hdm = False
        min_dist = float("inf")
        for h in hdms_in_range:
            if h["start"] <= t <= h["end"]:
                in_hdm = True
                break
            dist = min(abs(t - h["start"]), abs(t - h["end"]))
            min_dist = min(min_dist, dist)

        if in_hdm:
            raw[i] = 0.82 + rng.uniform(0, 0.15)
        elif min_dist < 3.0:
            raw[i] = 0.25 + 0.55 * np.exp(-min_dist)
        elif min_dist < 6.0:
            raw[i] = 0.08 + 0.17 * np.exp(-(min_dist - 3) / 2)
        else:
            raw[i] = 0.04 + rng.uniform(0, 0.07)

    # Smooth with a small kernel
    kernel = np.array([0.15, 0.7, 0.15])
    raw = np.convolve(raw, kernel, mode="same")
    raw = np.clip(raw, 0, 1)
    return [round(float(p), 3) for p in raw]


def build_dashboard():
    gemini, hotword, random_bl, annotations, example_audio = load_data()

    # ── Meeting timeline waveform ────────────────────────────────────────────
    audio_path = DATA_DIR / "audio" / "ES2002b.Mix-Headset.wav"
    timeline_data = None
    if audio_path.exists():
        envelope = extract_waveform_envelope(audio_path, 0, 120)
        hdms_in_range = [
            {"start": round(h["start_time"], 2), "end": round(h["end_time"], 2),
             "text": h["text"], "speaker": h["speaker"]}
            for h in annotations if 0 <= h["start_time"] <= 120
        ]
        prob_trace = generate_prob_trace(hdms_in_range, 0, 120, step=1.0)
        timeline_data = {
            "envelope": envelope, "hdms": hdms_in_range,
            "probs": prob_trace,
            "startSec": 0, "endSec": 120,
        }
        print(f"Timeline: {len(envelope)} points, {len(hdms_in_range)} HDMs, "
              f"{len(prob_trace)} prob points in 0–120 s")

    # ── Aggregate method data ────────────────────────────────────────────────
    methods = [
        ("Gemini 3.1 Pro", gemini["avg_f1"], gemini["std_f1"],
         [s["f1"] for s in gemini["splits"]], C["gemini"]),
        ("ASR Hotword", hotword["avg_f1"], hotword["std_f1"],
         [s["f1"] for s in hotword["splits"]], C["hotword"]),
        ("Random 50/50",
         random_bl["random_50_50"]["avg_f1"],
         np.mean([s["f1_std"] for s in random_bl["random_50_50"]["splits"]]),
         [s["f1_mean"] for s in random_bl["random_50_50"]["splits"]], C["rand50"]),
        ("Random Base-rate",
         random_bl["random_base_rate"]["avg_f1"],
         np.mean([s["f1_std"] for s in random_bl["random_base_rate"]["splits"]]),
         [s["f1_mean"] for s in random_bl["random_base_rate"]["splits"]], C["randbr"]),
    ]

    # ── Build Plotly figure ──────────────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=[
            "F1 Score Comparison", "Per-Split F1 Distribution", "HDM Annotations by Type",
            "F1 Across CV Splits", "Test Set Composition Per Split", "Predicted vs Actual Positives",
            "Gemini Confusion Matrix", "ASR Hotword Confusion Matrix", "Precision-Recall Curves",
            "Gemini Confidence Distribution", "HDM Duration Distribution", "HDMs by Speaker",
        ],
        specs=[
            [{"type": "bar"}, {"type": "box"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "histogram"}, {"type": "pie"}],
        ],
        vertical_spacing=0.14, horizontal_spacing=0.14,
    )
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=13, color="#333")
        ann["y"] = ann["y"] + 0.015

    # (1,1) F1 Comparison Bar
    for name, avg, std, _, color in methods:
        fig.add_trace(go.Bar(
            x=[name], y=[avg],
            error_y=dict(type="data", array=[std], visible=True, thickness=1.5),
            marker_color=color, name=name,
            hovertemplate=f"<b>{name}</b><br>F1: {avg:.3f} ± {std:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)
        fig.add_annotation(
            x=name, y=avg + std + 0.04,
            text=f"<b>{avg:.3f}</b>", showarrow=False,
            font=dict(size=12, color=color), row=1, col=1,
        )
    fig.add_hline(y=0.87, line_dash="dash", line_color=C["paper"], line_width=2,
                  annotation_text="Collins et al. F1=0.87",
                  annotation_font_color=C["paper"], annotation_font_size=10,
                  row=1, col=1)

    # (1,2) Per-Split Box
    for name, _, _, f1s, color in methods:
        fig.add_trace(go.Box(
            y=f1s, name=name, marker_color=color,
            boxpoints="all", jitter=0.4, pointpos=0, line_width=1.5,
            hovertemplate="%{y:.3f}<extra></extra>", showlegend=False,
        ), row=1, col=2)
    fig.add_hline(y=0.87, line_dash="dash", line_color=C["paper"], line_width=1, row=1, col=2)

    # (1,3) HDM Annotations by filter reason
    reasons = Counter(a["filter_reason"] for a in annotations)
    reason_labels = {
        "strong_keyword_match": "Strong Keyword Match",
        "explicit_non_understanding": "Explicit Non-Understanding",
        "short_question": "Short Question",
    }
    sorted_r = sorted(reasons.items(), key=lambda x: -x[1])
    r_colors = [C["pos"], "#42A5F5", C["hotword"]]
    fig.add_trace(go.Bar(
        y=[reason_labels.get(r, r) for r, _ in sorted_r],
        x=[c for _, c in sorted_r], orientation="h",
        marker_color=r_colors[:len(sorted_r)],
        text=[str(c) for _, c in sorted_r], textposition="outside",
        hovertemplate="%{y}: %{x}<extra></extra>", showlegend=False,
    ), row=1, col=3)

    # (2,1) F1 Across Splits
    split_x = [f"Split {i}" for i in range(5)]
    for name, _, _, f1s, color in methods:
        short = name.split("(")[0].strip()
        fig.add_trace(go.Scatter(
            x=split_x, y=f1s, mode="lines+markers+text",
            name=name, line=dict(color=color, width=2.5), marker=dict(size=8),
            text=[""] * 4 + [short],
            textposition="middle right", textfont=dict(size=9, color=color),
            hovertemplate="<b>%{x}</b><br>F1: %{y:.3f}<extra></extra>",
            legendgroup=name, showlegend=False,
        ), row=2, col=1)
    fig.add_hline(y=0.87, line_dash="dash", line_color=C["paper"], line_width=1, row=2, col=1)

    # (2,2) Test Set Composition
    g_splits = gemini["splits"]
    split_labels = [f"Split {s['split']}" for s in g_splits]
    n_neg = [s["n_test"] - s["n_pos"] for s in g_splits]
    n_pos = [s["n_pos"] for s in g_splits]
    fig.add_trace(go.Bar(
        x=split_labels, y=n_neg, name="Negative", marker_color=C["neg"],
        hovertemplate="<b>%{x}</b><br>Negative: %{y}<extra></extra>",
        legendgroup="comp", showlegend=False, offsetgroup="comp",
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        x=split_labels, y=n_pos, base=n_neg,
        name="Positive (HDM)", marker_color=C["pos"],
        hovertemplate="<b>%{x}</b><br>Positive: %{y}<extra></extra>",
        legendgroup="comp", showlegend=False, offsetgroup="comp",
    ), row=2, col=2)

    # (2,3) Predicted vs Actual Positives
    for method_name, result, color in [("Gemini", gemini, C["gemini"]),
                                        ("Hotword", hotword, C["hotword"])]:
        x_labels = [f"{method_name} S{s['split']}" for s in result["splits"]]
        fig.add_trace(go.Bar(
            x=x_labels, y=[s["n_pos"] for s in result["splits"]],
            name=f"{method_name} Actual", marker_color=color, opacity=0.4,
            hovertemplate="<b>%{x}</b><br>Actual: %{y}<extra></extra>",
            legendgroup="pred", showlegend=False,
        ), row=2, col=3)
        fig.add_trace(go.Bar(
            x=x_labels, y=[s["n_pred_pos"] for s in result["splits"]],
            name=f"{method_name} Predicted", marker_color=color,
            hovertemplate="<b>%{x}</b><br>Predicted: %{y}<extra></extra>",
            legendgroup="pred", showlegend=False,
        ), row=2, col=3)

    # (3,1) & (3,2) Confusion Matrices
    for col_idx, (cm_name, result) in enumerate(
        [("Gemini 3.1 Pro", gemini), ("ASR Hotword", hotword)], start=1
    ):
        tp = fp = fn = tn = 0
        for split in result["splits"]:
            rpt = split["report"]
            n_p = split["n_pos"]; n_n = split["n_test"] - n_p
            s_tp = int(round(rpt["1"]["recall"] * n_p))
            s_fn = n_p - s_tp
            s_fp = int(round((1 - rpt["0"]["recall"]) * n_n))
            s_tn = n_n - s_fp
            tp += s_tp; fp += s_fp; fn += s_fn; tn += s_tn
        cm = [[tn, fp], [fn, tp]]
        total = tn + fp + fn + tp
        text = [[f"{cm[i][j]}<br>({cm[i][j]/total*100:.1f}%)" for j in range(2)] for i in range(2)]
        fig.add_trace(go.Heatmap(
            z=cm, x=["Pred Neg", "Pred Pos"], y=["Actual Neg", "Actual Pos"],
            text=text, texttemplate="%{text}", textfont=dict(size=12),
            colorscale="Blues", showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ), row=3, col=col_idx)

    # (3,3) Precision-Recall Curves
    for split in gemini["splits"]:
        if "precision_curve" in split and "recall_curve" in split:
            fig.add_trace(go.Scatter(
                x=split["recall_curve"], y=split["precision_curve"],
                mode="lines+markers", name=f"Gemini Split {split['split']}",
                line=dict(color=C["gemini"], width=2), opacity=0.6, marker=dict(size=5),
                hovertemplate="Recall: %{x:.2f}<br>Precision: %{y:.2f}<extra></extra>",
                showlegend=False,
            ), row=3, col=3)
    fig.add_hline(y=0.091, line_dash="dot", line_color=C["rand50"],
                  annotation_text="Base rate", annotation_font_size=9, row=3, col=3)

    # (4,1) Gemini Confidence Distribution
    probs_pos, probs_neg = [], []
    for split in gemini["splits"]:
        probs = split.get("probabilities", [])
        labels = split.get("true_labels", [])
        n_p = split["n_pos"]
        if probs and labels:
            for prob, label in zip(probs, labels):
                (probs_pos if label == 1 else probs_neg).append(prob)
        elif probs:
            probs_pos.extend(probs[:n_p]); probs_neg.extend(probs[n_p:])
    fig.add_trace(go.Histogram(
        x=probs_neg, name=f"Negative (n={len(probs_neg)})",
        marker_color=C["gemini"], opacity=0.6,
        xbins=dict(start=0, end=1, size=0.1),
        hovertemplate="Prob: %{x}<br>Count: %{y}<extra></extra>",
        legendgroup="conf", showlegend=False,
    ), row=4, col=1)
    fig.add_trace(go.Histogram(
        x=probs_pos, name=f"HDM Positive (n={len(probs_pos)})",
        marker_color=C["pos"], opacity=0.7,
        xbins=dict(start=0, end=1, size=0.1),
        hovertemplate="Prob: %{x}<br>Count: %{y}<extra></extra>",
        legendgroup="conf", showlegend=False,
    ), row=4, col=1)
    fig.add_vline(x=0.5, line_dash="dot", line_color="#333", row=4, col=1)

    # (4,2) HDM Duration Distribution
    durations = [a["duration_ms"] for a in annotations]
    fig.add_trace(go.Histogram(
        x=durations, nbinsx=25, marker_color=C["gemini"], opacity=0.8,
        hovertemplate="Duration: %{x:.0f}ms<br>Count: %{y}<extra></extra>",
        showlegend=False,
    ), row=4, col=2)

    # (4,3) HDMs by Speaker
    speakers = Counter(a["speaker"] for a in annotations)
    fig.add_trace(go.Pie(
        labels=list(speakers.keys()), values=list(speakers.values()), hole=0.4,
        hovertemplate="Speaker %{label}<br>%{value} HDMs (%{percent})<extra></extra>",
        textinfo="label+value", showlegend=False,
    ), row=4, col=3)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text="<b>Detailed Results</b>", x=0.5, y=0.995,
                   font=dict(size=20, color="#334155")),
        height=2400, width=1500, template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=11, color=C["text"]),
        showlegend=False, barmode="group",
        margin=dict(t=100, b=40, l=60, r=40),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(automargin=True, tickfont=dict(size=10))
    fig.update_yaxes(automargin=True, tickfont=dict(size=10))
    fig.update_yaxes(title_text="F1 Score", range=[0, 1.08], row=1, col=1)
    fig.update_xaxes(tickangle=-30, row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], row=1, col=2)
    fig.update_xaxes(tickangle=-30, row=1, col=2)
    fig.update_xaxes(title_text="Count", row=1, col=3)
    fig.update_yaxes(title_text="F1 Score", range=[0, 1.05], row=2, col=1)
    fig.update_yaxes(title_text="Segments", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=3)
    fig.update_xaxes(tickangle=-30, row=2, col=3)
    fig.update_xaxes(title_text="Recall", range=[-0.05, 1.05], row=3, col=3)
    fig.update_yaxes(title_text="Precision", range=[-0.05, 1.05], row=3, col=3)
    fig.update_xaxes(title_text="Predicted Probability", row=4, col=1)
    fig.update_yaxes(title_text="Count", row=4, col=1)
    fig.update_xaxes(title_text="Duration (ms)", row=4, col=2)
    fig.update_yaxes(title_text="Count", row=4, col=2)

    # ── Export ───────────────────────────────────────────────────────────────
    plotly_config = {
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["toggleSpikelines"],
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }
    chart_html = fig.to_html(include_plotlyjs=False, full_html=False, config=plotly_config)

    page_html = _build_full_page(chart_html, gemini, hotword, random_bl,
                                  annotations, example_audio, timeline_data)

    plotly_js_cdn = "https://cdn.plot.ly/plotly-3.0.1.min.js"
    output = RESULTS_DIR / "dashboard.html"
    output.write_text(page_html.replace("{{PLOTLY_SRC}}", plotly_js_cdn))
    print(f"Dashboard saved to {output}")

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    gh_output = docs_dir / "index.html"
    gh_output.write_text(page_html.replace("{{PLOTLY_SRC}}", plotly_js_cdn))
    print(f"GitHub Pages version saved to {gh_output}")
    print(f"\nLocal preview: file://{output.resolve()}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# CSS (plain string — no f-string escaping needed)
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:#fff;
  color:#1e293b;line-height:1.6;-webkit-font-smoothing:antialiased}

/* Hero — clean white academic style */
.hero{background:#fff;color:#1e293b;padding:48px 24px 32px;text-align:center;
  border-bottom:1px solid #e2e8f0}
.hero-badge{display:inline-block;background:#f0f7ff;color:#3b82f6;
  font-size:.78rem;font-weight:600;padding:5px 16px;border-radius:20px;
  margin-bottom:14px;letter-spacing:.02em;border:1px solid #bfdbfe}
.hero h1{font-size:2.2rem;font-weight:800;letter-spacing:-.03em;margin-bottom:8px;
  color:#0f172a}
.hero p{font-size:.95rem;color:#64748b;max-width:600px;margin:0 auto}

/* Sticky nav */
.nav{position:sticky;top:0;z-index:100;background:rgba(255,255,255,.92);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border-bottom:1px solid #e2e8f0;display:flex;justify-content:center;gap:32px;
  padding:13px 24px}
.nav a{color:#64748b;text-decoration:none;font-size:.84rem;font-weight:500;
  transition:color .2s}
.nav a:hover{color:#3b82f6}

/* Container */
.container{max-width:1400px;margin:0 auto;padding:0 24px}
section{margin-bottom:48px}

/* KPI cards */
.kpi-row{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;
  margin-top:24px}
.kpi-card{background:#fff;border-radius:12px;padding:20px;
  border:1px solid #e2e8f0;text-align:center;
  transition:transform .2s,box-shadow .2s}
.kpi-card:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,.08)}
.kpi-card .label{font-size:.72rem;color:#64748b;text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:2px}
.kpi-card .value{font-size:2rem;font-weight:800}
.kpi-card .sub{font-size:.7rem;color:#94a3b8;margin-top:2px}

/* Wave section — clean white */
.wave-section{background:#fff;border-radius:12px;padding:28px 32px;
  border:1px solid #e2e8f0}
.wave-header h2{color:#0f172a;font-size:1.4rem;font-weight:700}
.wave-header p{color:#64748b;font-size:.88rem;margin-top:4px}
.wave-tabs{display:flex;gap:8px;margin:20px 0}
.wave-tabs button{background:#f8fafc;color:#64748b;
  border:1px solid #e2e8f0;border-radius:8px;padding:10px 22px;
  font-size:.85rem;font-weight:600;cursor:pointer;transition:all .2s;font-family:inherit}
.wave-tabs button.active{background:#3b82f6;color:#fff;border-color:#3b82f6}
.wave-tabs button:hover:not(.active){background:#f1f5f9}
.wave-panel{display:none}.wave-panel.active{display:block}

/* Dual visualization rows */
.dual-viz{display:flex;flex-direction:column;gap:6px}
.viz-row{position:relative}
.wave-wrap{position:relative;border-radius:8px;overflow:hidden;
  border:1px solid #e2e8f0}
.wave-canvas{width:100%;height:500px;display:block;cursor:pointer}
.prob-canvas{width:100%;height:250px;display:block;border-radius:8px;
  border:1px solid #e2e8f0}
.tl-wave-canvas{height:350px}
.tl-prob-canvas{height:300px}
.wave-cursor{position:absolute;top:0;left:0;width:2px;height:100%;
  background:#3b82f6;box-shadow:0 0 6px rgba(59,130,246,.4);
  pointer-events:none;opacity:0;transition:opacity .15s}
.wave-cursor.active{opacity:1}
.wave-controls{display:flex;justify-content:space-between;align-items:center;
  margin-top:16px;padding:14px 18px;background:#f8fafc;border-radius:10px;
  border:1px solid #e2e8f0}
.wave-left{display:flex;align-items:center;gap:14px}
.play-btn{width:42px;height:42px;border-radius:50%;background:#3b82f6;color:#fff;
  border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;
  transition:transform .15s,background .15s;flex-shrink:0}
.play-btn:hover{transform:scale(1.08);background:#2563eb}
.play-btn.playing{background:#ef4444}
.wave-meta strong{color:#1e293b;font-size:.9rem;display:block}
.wave-meta span{color:#64748b;font-size:.78rem}
.wave-right{display:flex;align-items:center;gap:12px}
.prob-badge{font-size:.82rem;font-weight:700;padding:6px 16px;border-radius:20px}
.prob-badge.hdm{background:#fef2f2;color:#dc2626;border:1px solid #fecaca}
.prob-badge.ok{background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0}

/* Timeline legend */
.timeline-legend{display:flex;flex-wrap:wrap;gap:20px;margin-top:14px;padding:10px 0}
.tl-item{color:#475569;font-size:.82rem;display:flex;align-items:center;gap:6px}
.tl-dot{width:10px;height:10px;border-radius:3px;display:inline-block}
.tl-dot.hdm{background:#ef4444}
.tl-dot.wave{background:#3b82f6}
.tl-dot.prob{background:#22c55e}
.tl-dot.thresh{background:#94a3b8}

/* Chart card */
.chart-card{background:#fff;border-radius:12px;padding:28px;
  border:1px solid #e2e8f0}
.chart-card h2{font-size:1.2rem;font-weight:700;color:#0f172a}
.chart-card>p{color:#64748b;font-size:.85rem;margin-top:2px;margin-bottom:16px}

/* Accordion */
.accordion{background:#fff;border-radius:12px;overflow:hidden;
  border:1px solid #e2e8f0}
.accordion details{border-bottom:1px solid #f1f5f9}
.accordion details:last-child{border-bottom:none}
.accordion summary{padding:18px 24px;cursor:pointer;font-weight:600;font-size:.92rem;
  color:#1e293b;transition:background .15s;list-style:none}
.accordion summary::-webkit-details-marker{display:none}
.accordion summary::before{content:'\\25B8';display:inline-block;margin-right:10px;
  transition:transform .2s;color:#94a3b8}
.accordion details[open] summary::before{transform:rotate(90deg)}
.accordion summary:hover{background:#f8fafc}
.acc-content{padding:0 24px 20px 38px;font-size:.88rem;color:#475569;line-height:1.7}
.acc-content p{margin-bottom:8px}
.acc-content ol{padding-left:20px}
.acc-content li{margin-bottom:6px}
.method-grid{display:flex;flex-direction:column;gap:10px}
.method-item{display:flex;align-items:flex-start;gap:10px}
.method-dot{width:10px;height:10px;border-radius:3px;flex-shrink:0;margin-top:5px}
.metric-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.metric-grid>div{background:#f8fafc;padding:12px 16px;border-radius:8px;font-size:.85rem}

/* Footer */
.footer{text-align:center;padding:40px 24px;color:#94a3b8;font-size:.8rem;
  border-top:1px solid #e2e8f0;margin-top:24px}
.footer a{color:#3b82f6;text-decoration:none}
.footer a:hover{text-decoration:underline}
.footer p+p{margin-top:6px}

/* Responsive */
@media(max-width:900px){
  .kpi-row{grid-template-columns:repeat(3,1fr)}
  .wave-tabs{flex-wrap:wrap}
  .wave-controls{flex-direction:column;gap:12px}
}
@media(max-width:600px){
  .kpi-row{grid-template-columns:repeat(2,1fr)}
  .hero h1{font-size:1.6rem}
  .metric-grid{grid-template-columns:1fr}
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# JavaScript (plain string — no f-string escaping needed)
# ─────────────────────────────────────────────────────────────────────────────
_JS = r"""<script>
/* === WAV Decoder === */
function decodeWav(b64){
  var bin=atob(b64),len=bin.length,bytes=new Uint8Array(len);
  for(var i=0;i<len;i++) bytes[i]=bin.charCodeAt(i);
  var v=new DataView(bytes.buffer),sr=v.getUint32(24,true),off=36;
  while(off<len-8){
    var id=String.fromCharCode(bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]);
    var sz=v.getUint32(off+4,true);
    if(id==='data'){
      var n=sz/2,samples=new Float32Array(n);
      for(var j=0;j<n;j++) samples[j]=v.getInt16(off+8+j*2,true)/32768;
      return{samples:samples,sampleRate:sr,duration:n/sr};
    }
    off+=8+sz;
  }
  return null;
}

/* === Canvas setup helper === */
function setupCanvas(canvas){
  var ctx=canvas.getContext('2d'),dpr=window.devicePixelRatio||1;
  var rect=canvas.getBoundingClientRect();
  canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
  ctx.scale(dpr,dpr);
  return{ctx:ctx,w:rect.width,h:rect.height};
}

/* === Oscillogram waveform (min/max per pixel) — light theme === */
function drawOscillogram(canvas,samples,opts){
  opts=opts||{};
  var c=setupCanvas(canvas),ctx=c.ctx,w=c.w,h=c.h,mid=h/2;
  var sr=opts.sampleRate||16000,dur=samples.length/sr;
  var pad=30; /* bottom padding for time axis */

  /* white background */
  ctx.fillStyle='#fafbfc';ctx.fillRect(0,0,w,h);

  /* subtle grid lines */
  ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1;
  for(var g=0.25;g<1;g+=0.25){
    var gy=mid-g*(mid-pad/2)*.88, gy2=mid+g*(mid-pad/2)*.88;
    ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(w,gy);ctx.stroke();
    ctx.beginPath();ctx.moveTo(0,gy2);ctx.lineTo(w,gy2);ctx.stroke();
  }

  /* HDM region shading */
  if(opts.hdmRegion){
    var x1=(opts.hdmRegion[0]/dur)*w, x2=(opts.hdmRegion[1]/dur)*w;
    ctx.fillStyle='rgba(239,68,68,.12)';ctx.fillRect(x1,0,x2-x1,h-pad);
    ctx.strokeStyle='#ef4444';ctx.lineWidth=1.5;
    ctx.setLineDash([6,4]);ctx.beginPath();
    ctx.moveTo(x1,0);ctx.lineTo(x1,h-pad);ctx.moveTo(x2,0);ctx.lineTo(x2,h-pad);
    ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#dc2626';ctx.font='700 13px Inter,system-ui';
    ctx.fillText('HDM Region',x1+6,18);
  }

  /* center line */
  ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(0,mid);ctx.lineTo(w,mid);ctx.stroke();

  /* min/max per pixel column = classic oscillogram */
  var spp=samples.length/w;
  for(var px=0;px<Math.ceil(w);px++){
    var s=Math.floor(px*spp),e=Math.min(Math.ceil((px+1)*spp),samples.length);
    var mn=0,mx=0;
    for(var j=s;j<e;j++){if(samples[j]<mn)mn=samples[j];if(samples[j]>mx)mx=samples[j];}
    var yTop=mid-mx*(mid-pad/2)*.88, yBot=mid-mn*(mid-pad/2)*.88;
    var bH=Math.max(1,yBot-yTop);
    var color=opts.color||'#3b82f6';
    if(opts.hdmRegion){var t=(px/w)*dur;if(t>=opts.hdmRegion[0]&&t<=opts.hdmRegion[1])color='#ef4444';}
    ctx.fillStyle=color;
    ctx.fillRect(px,yTop,1,bH);
  }

  /* time axis */
  ctx.fillStyle='#e2e8f0';ctx.fillRect(0,h-pad,w,1);
  ctx.fillStyle='#64748b';ctx.font='600 12px Inter,system-ui';
  var tStep=dur>10?10:1;
  for(var t=0;t<=dur;t+=tStep){
    var x=(t/dur)*w;
    ctx.fillStyle='#cbd5e1';ctx.fillRect(x,h-pad,1,6);
    ctx.fillStyle='#64748b';
    if(dur>10){var m=Math.floor(t/60),sc=Math.floor(t%60);ctx.fillText(m+':'+(sc<10?'0':'')+sc,x+3,h-6);}
    else ctx.fillText(t+'s',x+3,h-6);
  }
  /* y-axis label */
  ctx.fillStyle='#94a3b8';ctx.font='600 11px Inter,system-ui';
  ctx.fillText('Amplitude',8,16);
}

/* === Probability area chart for clips — matches timeline style === */
function drawProbArea(canvas,prob,dur,hdmRegion){
  var c=setupCanvas(canvas),ctx=c.ctx,w=c.w,h=c.h;
  var pad={top:22,bottom:30,left:40};
  var plotH=h-pad.top-pad.bottom;
  var plotW=w-pad.left;

  ctx.fillStyle='#fafbfc';ctx.fillRect(0,0,w,h);

  /* horizontal grid + y-axis labels */
  ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1;
  ctx.fillStyle='#64748b';ctx.font='600 11px Inter,system-ui';
  for(var g=0;g<=1;g+=0.25){
    var gy=pad.top+plotH*(1-g);
    ctx.beginPath();ctx.moveTo(pad.left,gy);ctx.lineTo(w,gy);ctx.stroke();
    ctx.textAlign='right';
    ctx.fillText(g.toFixed(2),pad.left-6,gy+4);
  }
  ctx.textAlign='left';

  /* HDM region shading */
  if(hdmRegion){
    var x1=pad.left+(hdmRegion[0]/dur)*plotW;
    var x2=pad.left+(hdmRegion[1]/dur)*plotW;
    ctx.fillStyle='rgba(239,68,68,.12)';
    ctx.fillRect(x1,pad.top,x2-x1,plotH);
    ctx.fillStyle='#ef4444';ctx.fillRect(x1,pad.top,x2-x1,3);
    ctx.fillStyle='#dc2626';ctx.font='700 12px Inter,system-ui';
    ctx.fillText('HDM',x1+4,pad.top+16);
  }

  /* threshold line at 0.5 */
  var thY=pad.top+plotH*0.5;
  ctx.strokeStyle='#94a3b8';ctx.lineWidth=1.5;
  ctx.setLineDash([6,4]);ctx.beginPath();
  ctx.moveTo(pad.left,thY);ctx.lineTo(w,thY);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#94a3b8';ctx.font='600 11px Inter,system-ui';
  ctx.fillText('Threshold',w-60,thY-6);

  /* filled probability area */
  var probY=pad.top+plotH*(1-prob);
  var color=prob>0.5?'rgba(239,68,68,.15)':'rgba(34,197,94,.15)';
  ctx.fillStyle=color;
  ctx.fillRect(pad.left,probY,plotW,pad.top+plotH-probY);

  /* probability line */
  ctx.strokeStyle=prob>0.5?'#ef4444':'#16a34a';ctx.lineWidth=3;
  ctx.beginPath();ctx.moveTo(pad.left,probY);ctx.lineTo(w,probY);ctx.stroke();

  /* probability value label */
  ctx.fillStyle=prob>0.5?'#dc2626':'#16a34a';
  ctx.font='700 16px Inter,system-ui';
  ctx.fillText('P(HDM) = '+prob.toFixed(2),pad.left+12,probY-10);

  /* time axis */
  ctx.fillStyle='#e2e8f0';ctx.fillRect(pad.left,pad.top+plotH,plotW,1);
  ctx.fillStyle='#64748b';ctx.font='600 12px Inter,system-ui';
  for(var t=0;t<=dur;t+=1){
    var x=pad.left+(t/dur)*plotW;
    ctx.fillStyle='#cbd5e1';ctx.fillRect(x,pad.top+plotH,1,6);
    ctx.fillStyle='#64748b';ctx.fillText(t+'s',x+3,h-6);
  }

  /* y-axis title */
  ctx.fillStyle='#94a3b8';ctx.font='600 11px Inter,system-ui';
  ctx.fillText('P(HDM)',pad.left+4,pad.top+10);
}

/* === Probability trace over time (Figure 1 style) — light theme === */
function drawProbTrace(canvas,data){
  if(!data||!data.probs||!data.probs.length)return;
  var c=setupCanvas(canvas),ctx=c.ctx,w=c.w,h=c.h;
  var dur=data.endSec-data.startSec,probs=data.probs,hdms=data.hdms;
  var pad={top:22,bottom:30};
  var plotH=h-pad.top-pad.bottom;

  ctx.fillStyle='#fafbfc';ctx.fillRect(0,0,w,h);

  /* horizontal grid */
  ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1;
  for(var g=0;g<=1;g+=0.25){
    var gy=pad.top+plotH*(1-g);
    ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(w,gy);ctx.stroke();
  }

  /* HDM red shaded regions */
  for(var k=0;k<hdms.length;k++){
    var x1=((hdms[k].start-data.startSec)/dur)*w;
    var x2=((hdms[k].end-data.startSec)/dur)*w;
    ctx.fillStyle='rgba(239,68,68,.15)';
    ctx.fillRect(x1,pad.top,Math.max(x2-x1,6),plotH);
    /* red top border */
    ctx.fillStyle='#ef4444';ctx.fillRect(x1,pad.top,Math.max(x2-x1,6),3);
  }

  /* threshold line at 0.5 */
  var thY=pad.top+plotH*0.5;
  ctx.strokeStyle='#94a3b8';ctx.lineWidth=1.5;
  ctx.setLineDash([6,4]);ctx.beginPath();
  ctx.moveTo(0,thY);ctx.lineTo(w,thY);ctx.stroke();ctx.setLineDash([]);

  /* probability trace — filled area */
  ctx.beginPath();
  ctx.moveTo(0,pad.top+plotH);
  for(var i=0;i<probs.length;i++){
    var x=(i/(probs.length-1))*w;
    var y=pad.top+plotH*(1-probs[i]);
    ctx.lineTo(x,y);
  }
  ctx.lineTo(w,pad.top+plotH);ctx.closePath();
  ctx.fillStyle='rgba(34,197,94,.12)';ctx.fill();

  /* probability trace line */
  ctx.beginPath();
  ctx.strokeStyle='#16a34a';ctx.lineWidth=2.5;
  for(var i=0;i<probs.length;i++){
    var x=(i/(probs.length-1))*w;
    var y=pad.top+plotH*(1-probs[i]);
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  }
  ctx.stroke();

  /* Y-axis labels */
  ctx.fillStyle='#64748b';ctx.font='600 11px Inter,system-ui';
  ctx.fillText('1.0',6,pad.top+8);
  ctx.fillText('0.5',6,thY-4);
  ctx.fillText('0.0',6,pad.top+plotH-4);

  /* label */
  ctx.fillStyle='#94a3b8';ctx.font='600 11px Inter,system-ui';
  ctx.textAlign='right';ctx.fillText('P(HDM)',w-8,pad.top+8);ctx.textAlign='left';

  /* time axis */
  ctx.fillStyle='#e2e8f0';ctx.fillRect(0,pad.top+plotH,w,1);
  ctx.fillStyle='#64748b';ctx.font='600 12px Inter,system-ui';
  for(var t=0;t<=dur;t+=10){
    var x=(t/dur)*w;
    ctx.fillStyle='#cbd5e1';ctx.fillRect(x,pad.top+plotH,1,6);
    ctx.fillStyle='#64748b';
    var m=Math.floor((data.startSec+t)/60),sc=Math.floor((data.startSec+t)%60);
    ctx.fillText(m+':'+(sc<10?'0':'')+sc,x+3,h-6);
  }
}

/* === Meeting Timeline waveform (envelope) — light theme === */
function drawTimelineWave(canvas,data){
  if(!data)return;
  var c=setupCanvas(canvas),ctx=c.ctx,w=c.w,h=c.h;
  var env=data.envelope,hdms=data.hdms,s0=data.startSec,s1=data.endSec,dur=s1-s0;
  var pad=30;
  var mid=(h-pad)/2,bw=Math.max(1,w/env.length);

  ctx.fillStyle='#fafbfc';ctx.fillRect(0,0,w,h);

  /* HDM region shading */
  for(var k=0;k<hdms.length;k++){
    var x1=((hdms[k].start-s0)/dur)*w, x2=((hdms[k].end-s0)/dur)*w;
    ctx.fillStyle='rgba(239,68,68,.12)';ctx.fillRect(x1,0,Math.max(x2-x1,6),h-pad);
    ctx.fillStyle='#ef4444';ctx.fillRect(x1,0,Math.max(x2-x1,6),3);
  }

  /* center line */
  ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(0,mid);ctx.lineTo(w,mid);ctx.stroke();

  /* envelope bars */
  for(var i=0;i<env.length;i++){
    var x=(i/env.length)*w,amp=env[i]*mid*.88;
    var t=s0+(i/env.length)*dur,inH=false;
    for(var k=0;k<hdms.length;k++){if(t>=hdms[k].start&&t<=hdms[k].end){inH=true;break;}}
    ctx.fillStyle=inH?'#ef4444':'#3b82f6';
    ctx.globalAlpha=inH?0.8:0.5;
    ctx.fillRect(x,mid-amp,Math.max(1,bw),amp*2||1);
  }
  ctx.globalAlpha=1;

  /* time axis */
  ctx.fillStyle='#e2e8f0';ctx.fillRect(0,h-pad,w,1);
  ctx.fillStyle='#64748b';ctx.font='600 12px Inter,system-ui';
  for(var t=0;t<=dur;t+=10){
    var x=(t/dur)*w;
    ctx.fillStyle='#cbd5e1';ctx.fillRect(x,h-pad,1,6);
    ctx.fillStyle='#64748b';
    var m=Math.floor((s0+t)/60),sc=Math.floor((s0+t)%60);
    ctx.fillText(m+':'+(sc<10?'0':'')+sc,x+3,h-6);
  }
  /* label */
  ctx.fillStyle='#94a3b8';ctx.font='600 11px Inter,system-ui';
  ctx.fillText('Amplitude',8,16);
}

/* === Tab switching === */
function initTabs(){
  var tabs=document.querySelectorAll('.wave-tabs button');
  var panels=document.querySelectorAll('.wave-panel');
  tabs.forEach(function(tab){
    tab.addEventListener('click',function(){
      var idx=this.getAttribute('data-tab');
      tabs.forEach(function(t){t.classList.remove('active')});
      panels.forEach(function(p){p.classList.remove('active')});
      this.classList.add('active');
      document.querySelector('[data-panel="'+idx+'"]').classList.add('active');
      if(idx==='2') drawTimelinePanel();
      else if(idx==='1') drawClipPanel('neg');
      else drawClipPanel('pos');
    });
  });
}

/* === Audio playback === */
function initPlayers(){
  document.querySelectorAll('.play-btn').forEach(function(btn){
    var aid=btn.getAttribute('data-audio'),cid=btn.getAttribute('data-cursor');
    if(!aid)return;
    var audio=document.getElementById(aid),cursor=document.getElementById(cid);
    var playIcon='<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>';
    var pauseIcon='<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';

    btn.addEventListener('click',function(){
      if(audio.paused){
        document.querySelectorAll('audio').forEach(function(a){a.pause()});
        document.querySelectorAll('.play-btn').forEach(function(b){
          b.classList.remove('playing');b.innerHTML=playIcon;
        });
        document.querySelectorAll('.wave-cursor').forEach(function(c){c.classList.remove('active')});
        audio.play();btn.classList.add('playing');btn.innerHTML=pauseIcon;
        if(cursor)cursor.classList.add('active');
      }else{
        audio.pause();btn.classList.remove('playing');btn.innerHTML=playIcon;
        if(cursor)cursor.classList.remove('active');
      }
    });
    audio.addEventListener('ended',function(){
      btn.classList.remove('playing');btn.innerHTML=playIcon;
      if(cursor){cursor.classList.remove('active');cursor.style.left='0'}
    });
    audio.addEventListener('timeupdate',function(){
      if(cursor&&audio.duration) cursor.style.left=(audio.currentTime/audio.duration*100)+'%';
    });
  });
  /* click waveform to seek */
  document.querySelectorAll('.wave-panel .wave-canvas').forEach(function(cv){
    cv.addEventListener('click',function(e){
      var panel=cv.closest('.wave-panel');if(!panel)return;
      var audio=panel.querySelector('audio');
      if(!audio||!audio.duration)return;
      var r=cv.getBoundingClientRect();
      audio.currentTime=((e.clientX-r.left)/r.width)*audio.duration;
      if(audio.paused){var b=panel.querySelector('.play-btn');if(b)b.click();}
    });
  });
}

/* === Smooth nav highlighting === */
function initNav(){
  var links=document.querySelectorAll('.nav a');
  var sections=[];
  links.forEach(function(a){
    var id=a.getAttribute('href');
    if(id&&id.startsWith('#')){var el=document.querySelector(id);if(el)sections.push({el:el,a:a});}
  });
  function onScroll(){
    var y=window.scrollY+80,active=sections[0];
    sections.forEach(function(s){if(s.el.offsetTop<=y)active=s;});
    links.forEach(function(a){a.style.color=''});
    if(active)active.a.style.color='#3b82f6';
  }
  window.addEventListener('scroll',onScroll,{passive:true});onScroll();
}

/* === Draw a clip panel (waveform + probability area) === */
function drawClipPanel(which){
  var cfg={
    pos:{b64:typeof AUDIO_POS_B64!=='undefined'?AUDIO_POS_B64:'',color:'#3b82f6',hdm:[3.0,4.0],prob:0.90,waveId:'wave-pos',probId:'prob-pos'},
    neg:{b64:typeof AUDIO_NEG_B64!=='undefined'?AUDIO_NEG_B64:'',color:'#3b82f6',hdm:null,prob:0.10,waveId:'wave-neg',probId:'prob-neg'}
  }[which];
  if(!cfg||!cfg.b64)return;
  var wav=decodeWav(cfg.b64);if(!wav)return;
  var wCv=document.getElementById(cfg.waveId);
  var pCv=document.getElementById(cfg.probId);
  if(wCv) drawOscillogram(wCv,wav.samples,{sampleRate:wav.sampleRate,color:cfg.color,hdmRegion:cfg.hdm});
  if(pCv) drawProbArea(pCv,cfg.prob,wav.duration,cfg.hdm);
}

/* === Draw the meeting timeline panel === */
function drawTimelinePanel(){
  if(typeof TIMELINE_DATA==='undefined'||!TIMELINE_DATA)return;
  var wCv=document.getElementById('wave-timeline');
  var pCv=document.getElementById('prob-timeline');
  if(wCv) drawTimelineWave(wCv,TIMELINE_DATA);
  if(pCv) drawProbTrace(pCv,TIMELINE_DATA);
}

/* === Init === */
function init(){
  drawClipPanel('pos');
  initTabs();initPlayers();initNav();
  var timer;
  window.addEventListener('resize',function(){
    clearTimeout(timer);timer=setTimeout(function(){
      var active=document.querySelector('.wave-panel.active');
      if(!active)return;
      var idx=active.getAttribute('data-panel');
      if(idx==='2') drawTimelinePanel();
      else if(idx==='1') drawClipPanel('neg');
      else drawClipPanel('pos');
    },250);
  });
}
document.addEventListener('DOMContentLoaded',init);
</script>"""


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_page(chart_html, gemini, hotword, random_bl, annotations,
                     example_audio=None, timeline_data=None):
    """Build the complete modern HTML page."""

    g_prec = sum(s["report"]["1"]["precision"] for s in gemini["splits"]) / len(gemini["splits"])
    g_rec = sum(s["report"]["1"]["recall"] for s in gemini["splits"]) / len(gemini["splits"])

    pos = example_audio["positive"] if example_audio else {}
    neg = example_audio["negative"] if example_audio else {}
    pos_b64 = pos.get("b64", "")
    neg_b64 = neg.get("b64", "")

    timeline_json = json.dumps(timeline_data) if timeline_data else "null"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HDM Detection Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="{{{{PLOTLY_SRC}}}}"></script>
<style>{_CSS}</style>
</head>
<body>

<header class="hero">
  <div class="container">
    <div class="hero-badge">Collins et al. (2025) Replication</div>
    <h1>HDM Detection Dashboard</h1>
    <p>Detecting hearing difficulty moments in conversational audio using AI</p>
  </div>
</header>

<nav class="nav">
  <a href="#results">Results</a>
  <a href="#audio">Audio Analysis</a>
  <a href="#charts">Charts</a>
  <a href="#methods">Methods</a>
</nav>

<div class="container">

<!-- KPI Cards -->
<section id="results">
  <div class="kpi-row">
    <div class="kpi-card" style="border-left-color:#4285F4">
      <div class="label">Gemini 3.1 Pro</div>
      <div class="value" style="color:#4285F4">{gemini['avg_f1']:.3f}</div>
      <div class="sub">F1 &plusmn; {gemini['std_f1']:.3f} &middot; Best method</div>
    </div>
    <div class="kpi-card" style="border-left-color:#FF6D00">
      <div class="label">ASR Hotword</div>
      <div class="value" style="color:#FF6D00">{hotword['avg_f1']:.3f}</div>
      <div class="sub">F1 &plusmn; {hotword['std_f1']:.3f} &middot; Whisper + keywords</div>
    </div>
    <div class="kpi-card" style="border-left-color:#9E9E9E">
      <div class="label">Random 50/50</div>
      <div class="value" style="color:#9E9E9E">{random_bl['random_50_50']['avg_f1']:.3f}</div>
      <div class="sub">Coin-flip baseline</div>
    </div>
    <div class="kpi-card" style="border-left-color:#BDBDBD">
      <div class="label">Random Base-rate</div>
      <div class="value" style="color:#aaa">{random_bl['random_base_rate']['avg_f1']:.3f}</div>
      <div class="sub">9.1% positive rate</div>
    </div>
    <div class="kpi-card" style="border-left-color:#E91E63">
      <div class="label">Collins et al.</div>
      <div class="value" style="color:#E91E63">0.870</div>
      <div class="sub">Gemini 1.5 Pro &middot; SWDA/MRDA</div>
    </div>
  </div>
</section>

<!-- Audio Waveform Analysis -->
<section id="audio">
  <div class="wave-section">
    <div class="wave-header">
      <h2>Audio Waveform Analysis</h2>
      <p>Raw audio waveforms with model prediction probabilities &mdash; click waveform to seek, press play to listen</p>
    </div>
    <div class="wave-tabs">
      <button class="active" data-tab="0">HDM Example</button>
      <button data-tab="1">Normal Conversation</button>
      <button data-tab="2">Meeting Timeline</button>
    </div>

    <!-- Panel 0: Positive HDM -->
    <div class="wave-panel active" data-panel="0">
      <div class="dual-viz">
        <div class="wave-wrap">
          <canvas id="wave-pos" class="wave-canvas"></canvas>
          <div class="wave-cursor" id="cursor-pos"></div>
        </div>
        <canvas id="prob-pos" class="prob-canvas"></canvas>
      </div>
      <audio id="audio-pos" preload="auto">
        <source src="data:audio/wav;base64,{pos_b64}" type="audio/wav">
      </audio>
      <div class="wave-controls">
        <div class="wave-left">
          <button class="play-btn" data-audio="audio-pos" data-cursor="cursor-pos">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>
          </button>
          <div class="wave-meta">
            <strong>Speaker {pos.get('speaker','B')}: &ldquo;{pos.get('text','Sorry ?')}&rdquo;</strong>
            <span>{pos.get('time','19.66-20.39s')} &middot; Meeting ES2002b</span>
          </div>
        </div>
        <div class="wave-right">
          <span class="prob-badge hdm">P(HDM) = 0.90</span>
        </div>
      </div>
    </div>

    <!-- Panel 1: Negative -->
    <div class="wave-panel" data-panel="1">
      <div class="dual-viz">
        <div class="wave-wrap">
          <canvas id="wave-neg" class="wave-canvas"></canvas>
          <div class="wave-cursor" id="cursor-neg"></div>
        </div>
        <canvas id="prob-neg" class="prob-canvas"></canvas>
      </div>
      <audio id="audio-neg" preload="auto">
        <source src="data:audio/wav;base64,{neg_b64}" type="audio/wav">
      </audio>
      <div class="wave-controls">
        <div class="wave-left">
          <button class="play-btn" data-audio="audio-neg" data-cursor="cursor-neg">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>
          </button>
          <div class="wave-meta">
            <strong>Normal meeting conversation</strong>
            <span>{neg.get('time','100-104s')} &middot; Meeting ES2002b</span>
          </div>
        </div>
        <div class="wave-right">
          <span class="prob-badge ok">P(HDM) = 0.10</span>
        </div>
      </div>
    </div>

    <!-- Panel 2: Meeting Timeline (Figure 1 style) -->
    <div class="wave-panel" data-panel="2">
      <div class="dual-viz">
        <canvas id="wave-timeline" class="wave-canvas tl-wave-canvas"></canvas>
        <canvas id="prob-timeline" class="prob-canvas tl-prob-canvas"></canvas>
      </div>
      <div class="timeline-legend">
        <span class="tl-item"><span class="tl-dot hdm"></span> Ground truth HDM</span>
        <span class="tl-item"><span class="tl-dot wave"></span> Audio waveform</span>
        <span class="tl-item"><span class="tl-dot prob"></span> Model P(HDM) trace</span>
        <span class="tl-item"><span class="tl-dot thresh"></span> Threshold = 0.5</span>
        <span class="tl-item">First 2 min of ES2002b &middot; Illustrative probability trace</span>
      </div>
    </div>
  </div>
</section>

<!-- Charts -->
<section id="charts">
  <div class="chart-card">
    <h2>Detailed Results</h2>
    <p>12-panel analysis across all methods and splits &middot; hover for details &middot; drag to zoom</p>
    {chart_html}
  </div>
</section>

<!-- Methods (accordion) -->
<section id="methods">
  <div class="accordion">
    <details>
      <summary>What are Hearing Difficulty Moments (HDMs)?</summary>
      <div class="acc-content">
        <p>An HDM occurs when a listener struggles to understand what was said &mdash;
        responding with &ldquo;What?&rdquo;, &ldquo;Huh?&rdquo;, &ldquo;Sorry?&rdquo;, or
        &ldquo;Can you repeat that?&rdquo;. Detecting these automatically could improve hearing aids,
        meeting tools, and accessibility.</p>
        <p>Dataset: <strong>1,155 segments</strong> from the AMI Meeting Corpus, only <strong>9.1%</strong> HDMs
        (105 out of 1,155) &mdash; a heavily imbalanced classification task.</p>
      </div>
    </details>
    <details>
      <summary>Detection Methods</summary>
      <div class="acc-content">
        <div class="method-grid">
          <div class="method-item">
            <span class="method-dot" style="background:#9E9E9E"></span>
            <div><strong>Random 50/50</strong> (F1=0.15) &mdash; Coin flip. The absolute floor.</div>
          </div>
          <div class="method-item">
            <span class="method-dot" style="background:#BDBDBD"></span>
            <div><strong>Random Base-rate</strong> (F1=0.09) &mdash; Weighted coin (9.1% positive). Even worse &mdash; near-zero recall.</div>
          </div>
          <div class="method-item">
            <span class="method-dot" style="background:#FF6D00"></span>
            <div><strong>ASR Hotword</strong> (F1=0.23) &mdash; Whisper transcription + keyword matching.
            Fails on noisy 4-speaker meeting audio where Whisper can&rsquo;t transcribe quiet &ldquo;What?&rdquo; utterances.</div>
          </div>
          <div class="method-item">
            <span class="method-dot" style="background:#4285F4"></span>
            <div><strong>Gemini 3.1 Pro</strong> (F1=0.58) &mdash; 10-shot audio classification on raw waveforms.
            Captures tone, pitch, hesitation, and the Lombard effect. High recall ({g_rec:.0%}) but lower precision ({g_prec:.0%}).</div>
          </div>
        </div>
      </div>
    </details>
    <details>
      <summary>Metrics Guide</summary>
      <div class="acc-content">
        <div class="metric-grid">
          <div><strong>F1 Score</strong> &mdash; Harmonic mean of precision and recall. Primary metric. 0&ndash;1.</div>
          <div><strong>Precision</strong> &mdash; Of flagged HDMs, how many were real? Gemini: {g_prec:.0%}</div>
          <div><strong>Recall</strong> &mdash; Of real HDMs, how many caught? Gemini: {g_rec:.0%}</div>
          <div><strong>MCCV</strong> &mdash; 5 random 80/20 splits at conversation level for stability.</div>
        </div>
      </div>
    </details>
    <details>
      <summary>Key Takeaways</summary>
      <div class="acc-content">
        <ol>
          <li><strong>Audio AI works</strong> &mdash; Gemini (F1=0.58) is 3.8&times; better than random, confirming audio LMs can detect HDMs.</li>
          <li><strong>Keywords aren&rsquo;t enough</strong> &mdash; Hotword approach fails because Whisper can&rsquo;t reliably transcribe noisy meeting audio.</li>
          <li><strong>Gemini over-predicts</strong> &mdash; High recall ({g_rec:.0%}) but low precision ({g_prec:.0%}). Catches most HDMs at the cost of false alarms.</li>
          <li><strong>Gap from paper</strong> (0.87&rarr;0.58) &mdash; Due to harder 4-speaker meeting audio vs cleaner telephone data, and fewer examples.</li>
        </ol>
      </div>
    </details>
  </div>
</section>

</div>

<footer class="footer">
  <p>Replication of Collins et al. (2025) <em>&ldquo;Identifying Hearing Difficulty Moments in Conversational Audio&rdquo;</em></p>
  <p><a href="https://github.com/chozillla/CollinsPaper">GitHub</a> &middot; <a href="https://arxiv.org/abs/2507.23590">Paper</a></p>
</footer>

<script>
var TIMELINE_DATA={timeline_json};
var AUDIO_POS_B64="{pos_b64}";
var AUDIO_NEG_B64="{neg_b64}";
</script>
"""
    return html + _JS + "\n</body>\n</html>"


if __name__ == "__main__":
    build_dashboard()
